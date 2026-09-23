# run_unc_baselines.py -- uncertainty baselines for the EviDRP paper:
#   * MC-dropout   (N=50 stochastic forward passes of a point MultiMLP)
#   * Deep Ensemble (K=5 independently trained MultiMLPs)
# Same data, splits, training discipline and metrics as run_survey_seeds.py, so
# the rows merge into the survey summary (methods "mc", "ens").
# Writes survey_seeds_{P}_unc.csv + _summary.csv in the v3 schema.
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_survey_seeds import (load_protocol_multi, train_seed, infer,
                              EVI2, EVI2_ARCH, RES)
from models_survey import MultiMLP
from train_utils import rmse, pearson, r2, spearman, conformal

MC_PASSES = 50
ENS_SIZE = 5


def mc_dropout_pred(model, c, d, n=MC_PASSES):
    """Stochastic forward passes with dropout active -> mean, std."""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(model(c, d).numpy())
    mu = np.mean(preds, 0)
    sig = np.std(preds, 0)
    return mu, sig


def ensemble_pred(members, c, d):
    preds = []
    for m in members:
        m.eval()
        with torch.no_grad():
            preds.append(m(c, d).numpy())
    mu = np.mean(preds, 0)
    sig = np.std(preds, 0)
    return mu, sig


def main(P, seeds, threads, tag):
    torch.set_num_threads(threads)
    tr, va, te, cell_idx, drug_idx, y, cells, drugs = load_protocol_multi(P)
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))
    print(f"[{P}] train {len(tr)} val {len(va)} test {len(te)} | "
          f"seeds {seeds} | MC={MC_PASSES} passes, ensemble K={ENS_SIZE}", flush=True)
    rows = []
    for seed in seeds:
        # K independently trained point models (member seeds derived from the
        # outer seed, so a given outer seed reproduces the whole ensemble)
        members = []
        for k in range(ENS_SIZE):
            m = MultiMLP()
            t0 = time.time()
            m = train_seed(m, "mlp", tr, va, cell_idx, drug_idx, y, cells, drugs,
                           seed * ENS_SIZE + k)
            members.append(m)
            print(f"  s{seed} member{k}: {time.time()-t0:.0f}s", flush=True)
        c_te = tc[torch.from_numpy(cell_idx[te].astype(np.int64))]
        d_te = td[torch.from_numpy(drug_idx[te].astype(np.int64))]
        c_va = tc[torch.from_numpy(cell_idx[va].astype(np.int64))]
        d_va = td[torch.from_numpy(drug_idx[va].astype(np.int64))]
        for name, (mu, sig), (mu_va, sig_va) in [
                ("mc", mc_dropout_pred(members[0], c_te, d_te),
                 mc_dropout_pred(members[0], c_va, d_va)),
                ("ens", ensemble_pred(members, c_te, d_te),
                 ensemble_pred(members, c_va, d_va))]:
            picp, mpiw = conformal(y[va], mu_va, sig_va, y[te], mu, sig,
                                   alpha=0.1, studentized=True)
            resid = np.abs(y[te] - mu)
            sp = spearman(sig, resid)
            rows.append({"protocol": P, "method": name, "seed": seed,
                         "rmse": round(rmse(y[te], mu), 4),
                         "pearson": round(pearson(y[te], mu), 4),
                         "r2": round(r2(y[te], mu), 4),
                         "spearman_unc_err": round(sp, 4),
                         "picp": round(picp, 4), "mpiw": round(mpiw, 4)})
            print(f"  s{seed} {name}: rmse={rows[-1]['rmse']} r={rows[-1]['pearson']} "
                  f"picp={picp:.3f} mpiw={mpiw:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"survey_seeds_{P}_{tag}.csv"), index=False)
    g = df.groupby("method").agg(
        protocol=("protocol", "first"),
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        pearson=("pearson", "mean"), pearson_sd=("pearson", "std"),
        r2=("r2", "mean"), r2_sd=("r2", "std"),
        spearman_unc_err=("spearman_unc_err", "mean"),
        picp=("picp", "mean"), mpiw=("mpiw", "mean")).reset_index()
    g.to_csv(os.path.join(RES, f"survey_seeds_{P}_{tag}_summary.csv"), index=False)
    print(f"\n=== {P} uncertainty baselines summary ===", flush=True)
    print(g.to_string(index=False), flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", default="P0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--tag", default="unc")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    main(args.P, seeds, args.threads, args.tag)
    print("DONE", args.P)