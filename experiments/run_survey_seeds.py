# run_survey_seeds.py -- 5-seed evaluation for the multi-omics survey comparison.
# For each (protocol, method, seed): SAME fixed splits, init + batch order vary
# with seed.  Methods: BANDRP, AttnOmics, MMCL, CLCLSA, MultiMLP, MultiEviDRP,
# EviDRPv2 (channel-attention evidential).  Writes per-seed rows + mean/std
# summary CSVs (per protocol, so protocols can run as parallel processes).
#
# Usage: python run_survey_seeds.py --P P0 --seeds 0,1,2,3,4 \
#        --methods bandrp,attn,mmcl,clclsa,mlp,evi,evi2 --threads 6 --tag seeds
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from models_survey import (BANDRP, AttnOmics, MMCL, CLCLSA, MultiMLP, MultiEviDRP,
                           DeepDTF, DELFOS, FourierDrug)
from evidrp_v2 import EviDRPv2, evi2_loss, nig_nll, evi_reg
from train_utils import rmse, pearson, r2, spearman, conformal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPL = os.path.join(ROOT, "splits")
FEAT = os.path.join(ROOT, "feat")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

EPOCHS, BATCH = 40, 2048
# EviDRP-v2: two-stage training. Stage 1 = pure MSE point fit (guaranteed point
# accuracy); stage 2 = NIG fit of the evidential head on detached point estimates.
EVI2 = {"stage2": True, "single_stage": False, "n_stage2": 30, "w_reg": 0.05,
        "epochs": 40, "cosine": False}
EVI2_ARCH = {"interact": True, "use_bn": True, "drug_layers": 1,
             "dsoft": 0, "csoft": 0, "soft_tau": 0.25, "csoft_tau": 0.25,
             "interact_mode": "hadamard", "ch_layers": 1, "dt": False, "attn": True}


def drug_soft_cluster(drugs, tr_drugs, K, tau=0.25, seed=0):
    """K-means over training drugs -> soft membership [N,K] of every drug wrt the
    training centroids (softmax over -||x-c||^2/tau). OOD-friendly for drug-blind."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=K, random_state=seed, n_init=5).fit(drugs[tr_drugs])
    d2 = ((drugs[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
    d2 = d2 / d2.mean()
    w = np.exp(-d2 / tau)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def cell_soft_cluster(cells, tr_cells, K, tau=0.25, seed=0, sub=50000):
    """K-means on a subsample of training cells -> soft membership [N,K] of every
    cell wrt the training-cell centroids. OOD-friendly for cold cells (P1)."""
    from sklearn.cluster import KMeans
    rng = np.random.RandomState(seed)
    sub = min(sub, len(tr_cells))
    subi = cells[tr_cells[rng.permutation(len(tr_cells))[:sub]]]
    km = KMeans(n_clusters=K, random_state=seed, n_init=3).fit(subi)
    d2 = ((cells[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
    d2 = d2 / d2.mean()
    w = np.exp(-d2 / tau)
    return (w / w.sum(1, keepdims=True)).astype(np.float32)


def load_protocol_multi(P):
    tr = np.load(os.path.join(SPL, f"{P}_train.npy"))
    va = np.load(os.path.join(SPL, f"{P}_val.npy"))
    te = np.load(os.path.join(SPL, f"{P}_test.npy"))
    cell_idx = np.load(os.path.join(SPL, "cell_idx.npy"))
    drug_idx = np.load(os.path.join(SPL, "drug_idx.npy"))
    y = np.load(os.path.join(SPL, "y.npy"))
    cells = np.load(os.path.join(FEAT, f"{P}_cells_multi.npy"))   # n_cells x 384
    drugs = np.load(os.path.join(FEAT, f"{P}_drugs.npy"))          # n_drugs x 128
    return tr, va, te, cell_idx, drug_idx, y, cells, drugs


def make_model(name):
    if name == "bandrp":
        return BANDRP()
    if name == "attn":
        return AttnOmics()
    if name == "mmcl":
        return MMCL()
    if name == "clclsa":
        return CLCLSA()
    if name == "mlp":
        return MultiMLP()
    if name == "evi":
        return MultiEviDRP()
    if name == "evi2":
        m = EviDRPv2(interact=EVI2_ARCH["interact"],
                     use_bn=EVI2_ARCH.get("use_bn", True),
                     drug_dim=128 + EVI2_ARCH.get("dsoft", 0),
                     drug_layers=EVI2_ARCH.get("drug_layers", 1),
                     csoft_dim=EVI2_ARCH.get("csoft", 0),
                     ch_layers=EVI2_ARCH.get("ch_layers", 1),
                     interact_mode=EVI2_ARCH.get("interact_mode", "hadamard"),
                     dt_dim=128 if EVI2_ARCH.get("dt", False) else 0,
                     attn=EVI2_ARCH.get("attn", True))
        if m.interact_mode in ("gate", "gatevec"):
            m.gate_softplus = False    # raw gate (learned scalar can be negative)
        return m
    if name == "deepdtf":
        return DeepDTF()
    if name == "delfos":
        return DELFOS()
    if name == "fourierdrug":
        return FourierDrug()
    raise ValueError(name)


def train_seed(model, name, tr_idx, va_idx, cell_idx, drug_idx, y, cells, drugs, seed,
               cs=None, dt=None, fp=None, gdata=None):
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    fit_cell = torch.from_numpy(cell_idx[tr_idx].astype(np.int64)).to(dev)
    fit_drug = torch.from_numpy(drug_idx[tr_idx].astype(np.int64)).to(dev)
    fit_y = torch.from_numpy(y[tr_idx].astype(np.float32)).to(dev)
    val_cell = torch.from_numpy(cell_idx[va_idx].astype(np.int64)).to(dev)
    val_drug = torch.from_numpy(drug_idx[va_idx].astype(np.int64)).to(dev)
    val_y = torch.from_numpy(y[va_idx].astype(np.float32)).to(dev)
    tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
    td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
    tcs = torch.from_numpy(cs.astype(np.float32)).to(dev) if cs is not None else None
    tdt = dt.to(dev) if dt is not None else None
    tfp = torch.from_numpy(fp.astype(np.float32)).to(dev) if fp is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = EVI2.get("epochs", EPOCHS) if name == "evi2" else EPOCHS
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if name == "evi2" and EVI2.get("cosine", False) else None)
    best_rmse, best_state, bad = float("inf"), None, 0

    def batches(n, batch, seed):
        r = np.random.RandomState(seed)
        idx = r.permutation(n)
        return [idx[i:i + batch] for i in range(0, n, batch)]

    for ep in range(epochs):
        model.train()
        for b in batches(len(fit_cell), BATCH, seed + ep):
            c = tc[fit_cell[b]]
            d = td[fit_drug[b]]
            t = fit_y[b]
            if name == "mmcl":
                loss = model.compute_loss(c, d, t)
            elif name == "fourierdrug":
                loss = model.compute_loss(c, d, t)
            elif name == "clclsa":
                pred = model(c, d).squeeze(-1)
                om = [torch.relu(e(c[:, i * 128:(i + 1) * 128])) for i, e in enumerate(model.ae)]
                z = torch.stack(om, dim=1)
                za = F.normalize(z.reshape(z.shape[0], -1), dim=-1)
                logits = za @ za.t() / 0.1
                mask = torch.eye(logits.shape[0], device=logits.device).bool()
                nce = -(logits[mask].mean() - torch.logsumexp(logits.masked_fill(mask, -1e9), dim=1).mean())
                loss = F.mse_loss(pred, t) + 0.3 * nce
            elif name == "evi":
                mu, v, a, b = model(c, d)
                Omega = 2.0 * b * (1.0 + v)
                nll = (0.5 * torch.log(torch.tensor(np.pi, dtype=mu.dtype)) - 0.5 * torch.log(v)
                       - a * torch.log(Omega)
                       + (a + 0.5) * torch.log(v * (t - mu) ** 2 + Omega)
                       + torch.lgamma(a) - torch.lgamma(a + 0.5)).mean()
                reg = torch.mean(torch.abs(t - mu) * (2.0 * v + a)) * 0.1
                loss = nll + reg
            elif name in ("evi2",):
                sc = tcs[fit_cell[b]] if tcs is not None else None
                dd = tdt[fit_drug[b]] if tdt is not None else None
                if EVI2.get("single_stage", False):
                    # ablation: joint evidential training (annealed NIG), no stage-2
                    mu, v, a, b = model(c, d, None, sc, None, dd)
                    loss = evi2_loss(t, mu, v, a, b, ep, epochs,
                                     mse_frac=EVI2.get("single_mse_frac", 0.5),
                                     w_reg=EVI2["w_reg"])
                else:
                    mu, *_ = model(c, d, None, sc, None, dd)
                    loss = F.mse_loss(mu, t)     # stage-1 point fit (pure MSE)
            else:
                pred = model(c, d)
                loss = F.mse_loss(pred, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        model.eval()
        with torch.no_grad():
            if name in ("evi2",):
                sc = tcs[val_cell] if tcs is not None else None
                dd = tdt[val_drug] if tdt is not None else None
                mu, *_ = model(tc[val_cell], td[val_drug], None, sc, None, dd)
            elif name == "evi":
                mu, *_ = model(tc[val_cell], td[val_drug])
            else:
                mu = model(tc[val_cell], td[val_drug])
            vrmse = float(torch.sqrt(F.mse_loss(mu, val_y)).item())
        if vrmse < best_rmse - 1e-5:
            best_rmse, bad = vrmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if name in ("evi2",) and EVI2.get("stage2", False) and not EVI2.get("single_stage", False):
        model = calibrate_evidential(model, tr_idx, va_idx, cell_idx, drug_idx,
                                     y, cells, drugs, seed,
                                     n_stage2=EVI2["n_stage2"],
                                     w_reg=EVI2["w_reg"], cs=cs, dt=tdt)
    return model


def calibrate_evidential(model, tr_idx, va_idx, cell_idx, drug_idx, y, cells, drugs,
                         seed, n_stage2=15, lr=1e-3, patience=4, w_reg=0.1, cs=None,
                         dt=None):
    """Stage-2: fit the evidential head (v,a,b) with the NIG objective on DETACHED
    point predictions, freezing the point head + trunk. Point accuracy is untouched;
    the evidential head learns calibrated aleatoric/epistemic variance."""
    torch.manual_seed(seed + 1000)
    dev = next(model.parameters()).device
    fit_cell = torch.from_numpy(cell_idx[tr_idx].astype(np.int64)).to(dev)
    fit_drug = torch.from_numpy(drug_idx[tr_idx].astype(np.int64)).to(dev)
    fit_y = torch.from_numpy(y[tr_idx].astype(np.float32)).to(dev)
    val_cell = torch.from_numpy(cell_idx[va_idx].astype(np.int64)).to(dev)
    val_drug = torch.from_numpy(drug_idx[va_idx].astype(np.int64)).to(dev)
    val_y = torch.from_numpy(y[va_idx].astype(np.float32)).to(dev)
    tc = torch.from_numpy(cells.astype(np.float32)).to(dev)
    td = torch.from_numpy(drugs.astype(np.float32)).to(dev)
    tcs = torch.from_numpy(cs.astype(np.float32)).to(dev) if cs is not None else None
    tdt = dt.to(dev) if dt is not None else None
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.out_ev.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.out_ev.parameters(), lr=lr, weight_decay=1e-5)
    best_nll, best_state, bad = float("inf"), None, 0

    def freeze_bn(m):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.eval()

    def batches(n, batch, seed):
        r = np.random.RandomState(seed)
        idx = r.permutation(n)
        return [idx[i:i + batch] for i in range(0, n, batch)]

    for ep in range(n_stage2):
        model.train()
        freeze_bn(model)     # keep BN running stats at stage-1 values
        for b in batches(len(fit_cell), BATCH, seed + 2000 + ep):
            c = tc[fit_cell[b]]
            d = td[fit_drug[b]]
            t = fit_y[b]
            sc = tcs[fit_cell[b]] if tcs is not None else None
            dd = tdt[fit_drug[b]] if tdt is not None else None
            mu, v, a, bb = model(c, d, None, sc, None, dd)
            loss = (nig_nll(t, mu.detach(), v, a, bb)
                    + evi_reg(t, mu.detach(), v, a, w_reg))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            scv = tcs[val_cell] if tcs is not None else None
            ddv = tdt[val_drug] if tdt is not None else None
            mu, v, a, bb = model(tc[val_cell], td[val_drug], None, scv, None, ddv)
            vnll = float(nig_nll(val_y, mu, v, a, bb).item())
        if vnll < best_nll - 1e-5:
            best_nll, bad = vnll, 0
            best_state = {k: v.clone() for k, v in model.out_ev.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.out_ev.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def infer(model, name, c_ix, d_ix, tc, td, tcs=None, tdt=None, fp=None, gdata=None):
    dev = next(model.parameters()).device
    c_ix_t = torch.from_numpy(c_ix.astype(np.int64)).to(dev)
    d_ix_t = torch.from_numpy(d_ix.astype(np.int64)).to(dev)
    c = tc[c_ix_t]
    d = td[d_ix_t]
    with torch.no_grad():
        if name in ("evi2",):
            sc = tcs[c_ix_t] if tcs is not None else None
            dd = tdt[d_ix_t] if tdt is not None else None
            mu, v, a, b = model(c, d, None, sc, None, dd)
            mu = mu.cpu().numpy()
            ale = (b / (a - 1.0)).cpu().numpy()
            epi = (b / (v * (a - 1.0))).cpu().numpy()
            total = np.sqrt(ale + epi)
            return mu, total, {"ale": np.sqrt(ale), "epi": np.sqrt(epi)}
        if name == "evi":
            mu, v, a, b = model(c, d)
            mu = mu.cpu().numpy()
            ale = (b / (a - 1.0)).cpu().numpy()
            epi = (b / (v * (a - 1.0))).cpu().numpy()
            total = np.sqrt(ale + epi)
            return mu, total, {"ale": np.sqrt(ale), "epi": np.sqrt(epi)}
        mu = model(c, d).cpu().numpy()
        return mu, np.ones_like(mu), {}


def run(P, methods, seeds, threads, tag):
    torch.set_num_threads(threads)
    tr, va, te, cell_idx, drug_idx, y, cells, drugs = load_protocol_multi(P)
    # EviDRP data-organization features: soft cluster memberships (fit on TRAINING
    # splits only). dsoft augments the drug vectors; csoft feeds the cell encoder.
    cs = None
    if any(m in ("evi2",) for m in methods) and EVI2_ARCH.get("csoft", 0) > 0:
        tr_cells = np.unique(cell_idx[tr])
        cs = cell_soft_cluster(cells, tr_cells, EVI2_ARCH["csoft"],
                               tau=EVI2_ARCH.get("csoft_tau", 0.25))
        print(f"  [evi2] cell-soft-cluster K={EVI2_ARCH['csoft']} "
              f"on {len(tr_cells)} training cells", flush=True)
    if any(m in ("evi2",) for m in methods) and EVI2_ARCH.get("dsoft", 0) > 0:
        tr_drugs = np.unique(drug_idx[tr])
        soft = drug_soft_cluster(drugs, tr_drugs, EVI2_ARCH["dsoft"],
                                 tau=EVI2_ARCH.get("soft_tau", 0.25))
        drugs = np.concatenate([drugs, soft], axis=1)
        print(f"  [evi2] drug-soft-cluster K={EVI2_ARCH['dsoft']} "
              f"on {len(tr_drugs)} training drugs -> drug feats {drugs.shape[1]}",
              flush=True)
    tdt = None
    if any(m in ("evi2",) for m in methods) and EVI2_ARCH.get("dt", False):
        dt_feat = np.load(os.path.join(FEAT, f"{P}_drugtarget.npy"))
        tdt = torch.from_numpy(dt_feat.astype(np.float32))
        print(f"  [evi2] drug-target pathway profile {dt_feat.shape}", flush=True)
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))
    tcs = torch.from_numpy(cs.astype(np.float32)) if cs is not None else None
    print(f"[{P}] train {len(tr)} val {len(va)} test {len(te)} | "
          f"seeds {seeds} | methods {methods}", flush=True)
    rows = []
    for seed in seeds:
        for name in methods:
            m = make_model(name)
            t0 = time.time()
            m = train_seed(m, name, tr, va, cell_idx, drug_idx, y, cells, drugs, seed,
                           cs=cs if name in ("evi2",) else None,
                           dt=tdt if (name in ("evi2",) and tdt is not None) else None)
            mu, sig, _ = infer(m, name, cell_idx[te], drug_idx[te], tc, td, tcs, tdt)
            mu_va, sig_va, _ = infer(m, name, cell_idx[va], drug_idx[va], tc, td, tcs, tdt)
            picp, mpiw = conformal(y[va], mu_va, sig_va, y[te], mu, sig, alpha=0.1,
                                   studentized=True)
            resid = np.abs(y[te] - mu)
            sp = spearman(sig, resid) if name in ("evi", "evi2") else np.nan
            rows.append({"protocol": P, "method": name, "seed": seed,
                         "rmse": round(rmse(y[te], mu), 4),
                         "pearson": round(pearson(y[te], mu), 4),
                         "r2": round(r2(y[te], mu), 4),
                         "spearman_unc_err": round(sp, 4) if sp == sp else np.nan,
                         "picp": round(picp, 4), "mpiw": round(mpiw, 4),
                         "train_s": int(time.time() - t0)})
            print(f"  s{seed} {name}: rmse={rows[-1]['rmse']} r={rows[-1]['pearson']} "
                  f"picp={picp:.3f} mpiw={mpiw:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"survey_seeds_{P}_{tag}.csv"), index=False)
    # summary: mean +- std over seeds (with protocol column restored)
    g = df.groupby("method").agg(
        protocol=("protocol", "first"),
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        pearson=("pearson", "mean"), pearson_sd=("pearson", "std"),
        r2=("r2", "mean"), r2_sd=("r2", "std"),
        spearman_unc_err=("spearman_unc_err", "mean"),
        picp=("picp", "mean"), mpiw=("mpiw", "mean")).reset_index()
    g.to_csv(os.path.join(RES, f"survey_seeds_{P}_{tag}_summary.csv"), index=False)
    print(f"\n=== {P} summary ===", flush=True)
    print(g.to_string(index=False), flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", default="P0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--methods", default="bandrp,attn,mmcl,clclsa,mlp,evi,evi2")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tag", default="seeds")
    ap.add_argument("--evi2-no-interact", action="store_true")
    ap.add_argument("--evi2-no-bn", action="store_true")
    ap.add_argument("--evi2-no-attn", action="store_true",
                    help="ablation: equal-weight channel pooling instead of attention")
    ap.add_argument("--evi2-single-stage", action="store_true",
                    help="ablation: joint annealed evidential training, no stage-2")
    ap.add_argument("--evi2-w-reg", type=float, default=0.05,
                    help="ablation: evidential regularizer weight in stage-2")
    ap.add_argument("--evi2-drug-layers", type=int, default=1)
    ap.add_argument("--evi2-ch-layers", type=int, default=1)
    ap.add_argument("--evi2-dsoft", type=int, default=0)
    ap.add_argument("--evi2-csoft", type=int, default=0)
    ap.add_argument("--evi2-interact-mode", default="hadamard",
                    choices=["hadamard", "norm", "gate"])
    ap.add_argument("--evi2-dt", action="store_true",
                    help="append drug-target pathway profile to the drug vector")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = [s for s in args.methods.split(",") if s]
    if args.evi2_no_interact:
        EVI2_ARCH["interact"] = False
    if args.evi2_no_bn:
        EVI2_ARCH["use_bn"] = False
    if args.evi2_no_attn:
        EVI2_ARCH["attn"] = False
    if args.evi2_single_stage:
        EVI2["single_stage"] = True
    EVI2["w_reg"] = args.evi2_w_reg
    EVI2_ARCH["drug_layers"] = args.evi2_drug_layers
    EVI2_ARCH["ch_layers"] = args.evi2_ch_layers
    EVI2_ARCH["dsoft"] = args.evi2_dsoft
    EVI2_ARCH["csoft"] = args.evi2_csoft
    EVI2_ARCH["interact_mode"] = args.evi2_interact_mode
    EVI2_ARCH["dt"] = args.evi2_dt
    run(args.P, methods, seeds, args.threads, args.tag)
    print("DONE", args.P)
