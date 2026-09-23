# make_features.py
# Per-protocol PCA reduction of cell features (fit on TRAIN cells only, no leakage)
# + shared drug fingerprint features.
#   cell raw: 930 x 15000  (top-5000 genes x {expr, cnv, mut})
#   cell pca: 930 x CELL_DIM per protocol (z-scored with train stats)
#   drug fp:  634 x 2048  Morgan binary (kept raw for first linear layer)
import json
import os
import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO = os.path.join(ROOT, "processed")
SPL = os.path.join(ROOT, "splits")
FEAT = os.path.join(ROOT, "feat")
os.makedirs(FEAT, exist_ok=True)
CELL_DIM = 256
DRUG_DIM = 128
SEED = 42
rng = np.random.RandomState(SEED)

cells = np.load(os.path.join(PRO, "cells_feat.npy"))    # 930 x 15000
drugs = np.load(os.path.join(PRO, "drug_feat.npy"))     # 634 x 2048
cell_idx = np.load(os.path.join(SPL, "cell_idx.npy"))   # per row
drug_idx = np.load(os.path.join(SPL, "drug_idx.npy"))   # per row
print("cells", cells.shape, "drugs", drugs.shape)

info = {"cell_raw_dim": int(cells.shape[1]), "cell_dim": CELL_DIM,
        "drug_raw_dim": int(drugs.shape[1]), "drug_dim": DRUG_DIM,
        "n_cells": cells.shape[0], "n_drugs": drugs.shape[0]}

for P in ["P0", "P1", "P2"]:
    tr = np.load(os.path.join(SPL, f"{P}_train.npy"))
    tr_cells = np.unique(cell_idx[tr])
    tr_drugs = np.unique(drug_idx[tr])
    # PCA on train cells only
    pca = PCA(n_components=CELL_DIM, random_state=SEED)
    pca.fit(cells[tr_cells])
    ev = float(pca.explained_variance_ratio_.sum())
    pcs = pca.transform(cells)                        # 930 x CELL_DIM
    mu = pcs[tr_cells].mean(0)
    sd = pcs[tr_cells].std(0) + 1e-8
    pcs_z = (pcs - mu) / sd
    np.save(os.path.join(FEAT, f"{P}_cells.npy"), pcs_z.astype(np.float32))
    np.save(os.path.join(FEAT, f"{P}_cell_mu.npy"), mu.astype(np.float32))
    np.save(os.path.join(FEAT, f"{P}_cell_sd.npy"), sd.astype(np.float32))
    info[f"{P}_explained_var"] = round(ev, 4)
    # PCA on train drugs only
    pdca = PCA(n_components=DRUG_DIM, random_state=SEED)
    pdca.fit(drugs[tr_drugs])
    dev = float(pdca.explained_variance_ratio_.sum())
    pds = pdca.transform(drugs)                       # 634 x DRUG_DIM
    dmu = pds[tr_drugs].mean(0)
    dsd = pds[tr_drugs].std(0) + 1e-8
    pds_z = (pds - dmu) / dsd
    np.save(os.path.join(FEAT, f"{P}_drugs.npy"), pds_z.astype(np.float32))
    np.save(os.path.join(FEAT, f"{P}_drug_mu.npy"), dmu.astype(np.float32))
    np.save(os.path.join(FEAT, f"{P}_drug_sd.npy"), dsd.astype(np.float32))
    info[f"{P}_drug_explained_var"] = round(dev, 4)
    print(f"{P}: cell PCA {len(tr_cells)} -> {CELL_DIM} (var {ev:.3f}), "
          f"drug PCA {len(tr_drugs)} -> {DRUG_DIM} (var {dev:.3f}), z-scored on train stats")

with open(os.path.join(FEAT, "feature_info.json"), "w") as fh:
    json.dump(info, fh, indent=1)
print(json.dumps(info, indent=1))
print("DONE make_features.py")
