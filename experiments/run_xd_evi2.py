# run_xd_evi2.py -- cross-dataset validation (reviewer R3-M1).
# Train v3 EviDRP (evi2) and the MultiMLP point baseline on one screen's rows,
# evaluate on ALL rows of the other screen (train on CTRPv2 -> test GDSCv2, and
# the reverse). Same P0 feature space (P0 multi-channel cell PCA + drug PCA),
# same training discipline and same v3 config as the survey, so rows merge into
# the survey summary (protocol names XD_GDSC2CTRPv2 / XD_CTRPv2toGDSC).
#
# Usage: python run_xd_evi2.py --dir GDSC2CTRPv2 --seeds 0,1,2,3,4 --threads 6
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from run_survey_seeds import (make_model, train_seed, infer, EVI2, EVI2_ARCH,
                              RES, FEAT)
from train_utils import load_crossdataset, rmse, pearson, r2, spearman, conformal
from models_survey import MultiMLP

DIRS = {"GDSC2CTRPv2": ("GDSCv2", "CTRPv2"), "CTRPv2toGDSC": ("CTRPv2", "GDSCv2")}


def main(direction, seeds, threads):
    torch.set_num_threads(threads)
    src, tgt = DIRS[direction]
    proto = f"XD_{direction}"
    tr, va, te, cell_idx, drug_idx, y, _, _ = load_crossdataset(src, tgt)
    # evi2 uses the multi-channel cell features (P0 feature space); mlp too, so
    # both methods see identical inputs (384-dim cells x 128-dim drugs).
    cells = np.load(os.path.join(FEAT, "P0_cells_multi.npy"))
    drugs = np.load(os.path.join(FEAT, "P0_drugs.npy"))
    tc = torch.from_numpy(cells.astype(np.float32))
    td = torch.from_numpy(drugs.astype(np.float32))
    print(f"[{proto}] train={len(tr)} val={len(va)} test(tgt)={len(te)} "
          f"({src} -> {tgt})", flush=True)

    rows, preds = [], {}
    for method in ["evi2", "mlp"]:
        for seed in seeds:
            t0 = time.time()
            m = make_model("evi2") if method == "evi2" else MultiMLP()
            m = train_seed(m, method, tr, va, cell_idx, drug_idx, y, cells, drugs,
                           seed)
            mu, sig, extra = infer(m, method, cell_idx[te], drug_idx[te], tc, td)
            mu_va, sig_va, _ = infer(m, method, cell_idx[va], drug_idx[va], tc, td)
            picp, mpiw = conformal(y[va], mu_va, sig_va, y[te], mu, sig,
                                   alpha=0.1, studentized=True)
            sp = spearman(sig, np.abs(y[te] - mu))
            rows.append({"method": method, "protocol": proto, "seed": seed,
                         "rmse": rmse(y[te], mu), "pearson": pearson(y[te], mu),
                         "r2": r2(y[te], mu), "spearman_unc_err": sp,
                         "picp": picp, "mpiw": mpiw})
            if seed == seeds[0]:
                preds[f"{method}_mu"] = mu
                preds[f"{method}_sig"] = sig
                preds["y"] = y[te]
                if extra:
                    preds[f"{method}_ale"] = extra["ale"]
                    preds[f"{method}_epi"] = extra["epi"]
            print(f"  {method} seed {seed}: rmse={rows[-1]['rmse']:.4f} "
                  f"r={rows[-1]['pearson']:.4f} spearman={sp:.3f} "
                  f"picp={picp:.3f} mpiw={mpiw:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"survey_seeds_{proto}_xd.csv"), index=False)
    summ = (df.groupby(["method", "protocol"])
              .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                   pearson=("pearson", "mean"), pearson_sd=("pearson", "std"),
                   r2=("r2", "mean"), r2_sd=("r2", "std"),
                   spearman_unc_err=("spearman_unc_err", "mean"),
                   picp=("picp", "mean"), mpiw=("mpiw", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(RES, f"survey_seeds_{proto}_summary.csv"), index=False)
    np.savez(os.path.join(RES, "preds", f"XD_{direction}_evi2v3.npz"), **preds)
    print("DONE", proto, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, choices=list(DIRS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    main(args.dir, [int(s) for s in args.seeds.split(",")], args.threads)