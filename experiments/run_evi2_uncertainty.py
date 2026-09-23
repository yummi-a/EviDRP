# run_evi2_uncertainty.py -- evi2 uncertainty evaluation on P0 (single seed).
# Produces the evi2-specific data the revised paper needs (extended baselines and
# the old flat model are dropped):
#   * results/conformal_evi2.csv   -- vanilla + studentized PICP/MPIW on P0
#   * results/ood_evi2.csv         -- OOD detection (P0 model vs P1/P2 test rows)
#   * results/preds/P0_evi2.npz    -- P0 test preds (mu, sig, ale, epi) -> fig6
#   * results/preds/P0_evi2_riskcoverage.npz -- selective-prediction curve -> fig4
# Uses the SAME EviDRP-v2 config as the survey v3 runs (flags mirror
# run_survey_seeds.py), so the numbers are consistent with tab:survey / fig8.
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_survey_seeds import (load_protocol_multi, make_model, train_seed, infer,
                              EVI2, EVI2_ARCH)
from train_utils import conformal, risk_coverage, ood_auc, spearman, rmse, pearson
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PRED = os.path.join(RES, "preds")
SPL = os.path.join(ROOT, "splits")
os.makedirs(PRED, exist_ok=True)


def ood_rows(P):
    """Test row indices for OOD setting P (global row space)."""
    return np.load(os.path.join(SPL, f"{P}_test.npy"))


def bootstrap_ci(x, B=1000, seed=0, q=(2.5, 97.5)):
    """Percentile bootstrap CI for the mean of x."""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    means = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        means[b] = x[idx].mean()
    return float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def main(seed, threads):
    torch.set_num_threads(threads)
    tr, va, te, cell_idx, drug_idx, y, cells, drugs = load_protocol_multi("P0")
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))
    print(f"[P0] train {len(tr)} val {len(va)} test {len(te)} | "
          f"evi2 config: interact={EVI2_ARCH['interact']} "
          f"use_bn={EVI2_ARCH.get('use_bn', True)} "
          f"interact_mode={EVI2_ARCH.get('interact_mode')} "
          f"ch_layers={EVI2_ARCH.get('ch_layers')}", flush=True)

    t0 = time.time()
    m = make_model("evi2")
    m = train_seed(m, "evi2", tr, va, cell_idx, drug_idx, y, cells, drugs, seed)
    print(f"  evi2 trained in {time.time()-t0:.0f}s", flush=True)

    # ---- P0 test predictions (fig6 decomposition + risk-coverage) ----
    mu, sig, extra = infer(m, "evi2", cell_idx[te], drug_idx[te], tc, td)
    mu_va, sig_va, _ = infer(m, "evi2", cell_idx[va], drug_idx[va], tc, td)
    np.savez(os.path.join(PRED, "P0_evi2.npz"), mu=mu, sig=sig, y=y[te],
             ale=extra["ale"], epi=extra["epi"])
    print(f"  P0 test: rmse={rmse(y[te], mu):.4f} r={pearson(y[te], mu):.4f} "
          f"spearman={spearman(sig, np.abs(y[te]-mu)):.4f}", flush=True)

    # ---- conformal: vanilla + studentized (fig3 / text) ----
    rows = []
    for stud in [False, True]:
        picp, mpiw = conformal(y[va], mu_va, sig_va, y[te], mu, sig,
                               alpha=0.1, studentized=stud)
        rows.append({"protocol": "P0", "method": "evi2", "studentized": stud,
                     "picp": round(picp, 4), "mpiw": round(mpiw, 4)})
        print(f"  conformal studentized={stud}: picp={picp:.4f} mpiw={mpiw:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "conformal_evi2.csv"), index=False)

    # ---- risk-coverage on P0 test (fig4) ----
    rc = {"evi2": risk_coverage(y[te], mu, sig)}
    np.savez(os.path.join(PRED, "P0_evi2_riskcoverage.npz"), evi2=rc["evi2"])
    print("  risk-coverage saved", flush=True)

    # ---- OOD detection: P0 model vs P1/P2 test rows (fig5) ----
    ood_sets = {"P1_coldcells": "P1", "P2_unseendrugs": "P2"}
    id_ix, id_dx = cell_idx[te], drug_idx[te]
    mu_id, sig_id, _ = infer(m, "evi2", id_ix, id_dx, tc, td)
    out = []
    for name, P in ood_sets.items():
        tex = ood_rows(P)
        mu_od, sig_od, _ = infer(m, "evi2", cell_idx[tex], drug_idx[tex], tc, td)
        n = min(len(sig_id), len(sig_od))
        sid = sig_id[:n]
        sod = sig_od[:n]
        a = np.concatenate([sid, sod])
        lbl = np.concatenate([np.zeros(n), np.ones(n)])
        auc = ood_auc(lbl, a)
        # bootstrap CIs: mean sigma (ID / OOD) and AUROC
        ci_id = bootstrap_ci(sid, B=1000, seed=seed)
        ci_od = bootstrap_ci(sod, B=1000, seed=seed)
        rng = np.random.RandomState(seed)
        aucs = np.empty(1000)
        for b in range(1000):
            idx = rng.randint(0, 2 * n, 2 * n)
            if len(np.unique(lbl[idx])) < 2:
                aucs[b] = np.nan
            else:
                aucs[b] = roc_auc_score(lbl[idx], a[idx])
        ci_auc = (float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5)))
        out.append({"setting": f"IDvs_{name}", "method": "evi2", "auc": round(auc, 4),
                    "auc_lo": round(ci_auc[0], 4), "auc_hi": round(ci_auc[1], 4),
                    "mean_sigma_id": round(float(np.mean(sid)), 5),
                    "sigma_id_lo": round(ci_id[0], 5), "sigma_id_hi": round(ci_id[1], 5),
                    "mean_sigma_ood": round(float(np.mean(sod)), 5),
                    "sigma_ood_lo": round(ci_od[0], 5), "sigma_ood_hi": round(ci_od[1], 5)})
        print(f"  OOD {name}: auc={auc:.3f} [{ci_auc[0]:.3f},{ci_auc[1]:.3f}] "
              f"(id={np.mean(sid):.4f} [{ci_id[0]:.4f},{ci_id[1]:.4f}] "
              f"ood={np.mean(sod):.4f} [{ci_od[0]:.4f},{ci_od[1]:.4f}])", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(RES, "ood_evi2.csv"), index=False)
    print("DONE run_evi2_uncertainty.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--evi2-no-interact", action="store_true")
    ap.add_argument("--evi2-no-bn", action="store_true")
    ap.add_argument("--evi2-ch-layers", type=int, default=1)
    ap.add_argument("--evi2-interact-mode", default="hadamard",
                    choices=["hadamard", "norm", "gate"])
    args = ap.parse_args()
    if args.evi2_no_interact:
        EVI2_ARCH["interact"] = False
    if args.evi2_no_bn:
        EVI2_ARCH["use_bn"] = False
    EVI2_ARCH["ch_layers"] = args.evi2_ch_layers
    EVI2_ARCH["interact_mode"] = args.evi2_interact_mode
    main(args.seed, args.threads)