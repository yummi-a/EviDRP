# make_features_multimodal.py
# Per-protocol PCA reduction of cell features PER OMICS CHANNEL (fit on TRAIN cells only).
#   raw cell: 930 x 15000 = top-5000 genes x {expr, cnv, mut} (columns ordered [expr|cnv|mut])
#   output  : 930 x (3*OMICS_DIM) per protocol, z-scored with train stats.
# Used to reproduce multi-omics survey models (BANDRP/AttnOmics/MMCL/CLCLSA) fairly:
#   same train-only PCA discipline as make_features.py, but channel-structured so
#   attention/contrastive fusion over omics is meaningful.
import json
import os
import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO = os.path.join(ROOT, "processed")
SPL = os.path.join(ROOT, "splits")
FEAT = os.path.join(ROOT, "feat")
os.makedirs(FEAT, exist_ok=True)

TOPGENES = 5000
OMICS_DIM = 128          # per-channel PCA dim -> 3*128 = 384 cell vector
SEED = 42

cells = np.load(os.path.join(PRO, "cells_feat.npy"))    # 930 x 15000
drugs = np.load(os.path.join(PRO, "drug_feat.npy"))     # 634 x 2048 (raw fp)
cell_idx = np.load(os.path.join(SPL, "cell_idx.npy"))
drug_idx = np.load(os.path.join(SPL, "drug_idx.npy"))
n_cells = cells.shape[0]
assert cells.shape[1] == 3 * TOPGENES, cells.shape

# split into per-omics channels (same gene order as build_dataset.py)
ch = {"expr": cells[:, :TOPGENES],
      "cnv":  cells[:, TOPGENES:2 * TOPGENES],
      "mut":  cells[:, 2 * TOPGENES:]}
print("channels:", {k: v.shape for k, v in ch.items()})

info = {"cell_raw_dim": int(cells.shape[1]), "n_top_genes": TOPGENES,
        "omics": list(ch.keys()), "omics_dim": OMICS_DIM,
        "cell_dim": 3 * OMICS_DIM, "n_cells": n_cells}

for P in ["P0", "P1", "P2"]:
    tr = np.load(os.path.join(SPL, f"{P}_train.npy"))
    tr_cells = np.unique(cell_idx[tr])
    out = []
    for name, X in ch.items():
        pca = PCA(n_components=OMICS_DIM, random_state=SEED)
        pca.fit(X[tr_cells])
        pc = pca.transform(X)                              # 930 x 128
        mu = pc[tr_cells].mean(0)
        sd = pc[tr_cells].std(0) + 1e-8
        out.append((pc - mu) / sd)
    cell_m = np.hstack(out).astype(np.float32)            # 930 x 384
    np.save(os.path.join(FEAT, f"{P}_cells_multi.npy"), cell_m)
    info[f"{P}_explained_var_total"] = round(sum(
        PCA(n_components=OMICS_DIM, random_state=SEED).fit(ch[k][np.unique(cell_idx[tr])])
        .explained_variance_ratio_.sum() for k in ch) / 3, 4)
    print(f"{P}: cells_multi {cell_m.shape} (3x{OMICS_DIM})")

with open(os.path.join(FEAT, "feature_info_multi.json"), "w") as fh:
    json.dump(info, fh, indent=1)
print(json.dumps(info, indent=1))
print("DONE make_features_multimodal.py")
