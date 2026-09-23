# make_splits.py
# Deterministic 8:1:1 (train:val:test) protocol splits over the combined DRP dataset.
#   P0: random 80/10/10 over (cell,drug) pairs         (in-domain)
#   P1: cold cells -- held-out cell lines (tissue-stratified)  (new cell generalization)
#   P2: drug-blind -- held-out drugs (dataset-stratified)      (new drug generalization)
# Saves row-index splits + per-row cell/drug index + target.
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRO = os.path.join(ROOT, "processed")
SPL = os.path.join(ROOT, "splits")
os.makedirs(SPL, exist_ok=True)
SEED = 42
TR, VA, TE = 0.8, 0.1, 0.1

r = pd.read_csv(os.path.join(PRO, "response_long.csv"))
cm = pd.read_csv(os.path.join(PRO, "cells_meta.csv"))
dm = pd.read_csv(os.path.join(PRO, "drugs_meta.csv"))
n_cells = len(cm)
n_drugs = len(dm)

cell_idx_map = {c: i for i, c in enumerate(cm["cell"].tolist())}
drug_idx_map = {d: i for i, d in enumerate(dm["drug"].tolist())}

r["cell_idx"] = r["ach"].map(cell_idx_map)
r["drug_idx"] = r["drug"].map(drug_idx_map)
assert r["cell_idx"].notna().all() and r["drug_idx"].notna().all()
y = r["aac"].to_numpy(np.float32)
cell_idx = r["cell_idx"].to_numpy(np.int32)
drug_idx = r["drug_idx"].to_numpy(np.int32)
idx_all = np.arange(len(r))


def split3(rows, stratify=None):
    """Return (80%, 10%, 10%) row split of `rows` (optionally stratified)."""
    a, rest = train_test_split(rows, test_size=TE + VA, random_state=SEED, stratify=stratify)
    nval = int(len(rows) * VA)
    # split rest into val/test keeping stratification of rest
    if stratify is not None:
        st = stratify[rest]
    else:
        st = None
    _, test = train_test_split(rest, test_size=TE / (TE + VA), random_state=SEED, stratify=st)
    val = np.array(sorted(set(rest.tolist()) - set(test.tolist())))
    return a, val, test


# ---- P0: random pair split ----
tr, va, te = split3(idx_all)
np.save(os.path.join(SPL, "P0_train.npy"), tr)
np.save(os.path.join(SPL, "P0_val.npy"), va)
np.save(os.path.join(SPL, "P0_test.npy"), te)
print(f"P0: train {len(tr)} ({len(tr)/len(idx_all):.2f}), val {len(va)}, test {len(te)}")

# ---- P1: cold cells, stratified by tissue ----
tissue = cm.set_index("cell")["tissue"].to_dict()
tissue_counts = pd.Series(tissue.values()).value_counts()
rare = set(tissue_counts[tissue_counts < 5].index)
tissue_b = [("Other" if t in rare else t) for t in tissue.values()]
tissue_map = {t: i for i, t in enumerate(sorted(set(tissue_b)))}
cell_tissue = np.array([tissue_map[t] for t in tissue_b])  # 930
cells = np.arange(n_cells)
tr_cells, rest_cells = train_test_split(cells, test_size=VA + TE, random_state=SEED,
                                        stratify=cell_tissue)
# split the held-out cells into val/test randomly (small groups can't stratify)
rest_perm = np.random.RandomState(SEED).permutation(rest_cells)
nva = int(len(rest_cells) * VA / (VA + TE))
va_cells = np.sort(rest_perm[:nva])
te_cells = np.sort(rest_perm[nva:])
for name, cs in [("train", tr_cells), ("val", va_cells), ("test", te_cells)]:
    mask = np.isin(cell_idx, list(set(cs)))
    np.save(os.path.join(SPL, f"P1_{name}.npy"), idx_all[mask])
    print(f"P1 {name}: {len(cs)} cells -> {int(mask.sum())} rows")

# ---- P2: drug-blind, stratified by dataset ----
drugs = np.arange(n_drugs)
drug2ds = r.groupby("drug_idx")["dataset"].first().to_dict()
drug_ds = np.array([0 if drug2ds[d] == "CTRPv2" else 1 for d in drugs])
tr_drugs, rest_drugs = train_test_split(drugs, test_size=VA + TE, random_state=SEED,
                                        stratify=drug_ds)
rest_perm = np.random.RandomState(SEED).permutation(rest_drugs)
nva = int(len(rest_drugs) * VA / (VA + TE))
va_drugs = np.sort(rest_perm[:nva])
te_drugs = np.sort(rest_perm[nva:])
for name, ds_ in [("train", tr_drugs), ("val", va_drugs), ("test", te_drugs)]:
    mask = np.isin(drug_idx, list(set(ds_)))
    np.save(os.path.join(SPL, f"P2_{name}.npy"), idx_all[mask])
    print(f"P2 {name}: {len(ds_)} drugs -> {int(mask.sum())} rows")

np.save(os.path.join(SPL, "cell_idx.npy"), cell_idx)
np.save(os.path.join(SPL, "drug_idx.npy"), drug_idx)
np.save(os.path.join(SPL, "y.npy"), y)
np.save(os.path.join(SPL, "ach_list.npy"), np.array(cm["cell"].tolist()))
np.save(os.path.join(SPL, "drug_list.npy"), np.array(dm["drug"].tolist()))

info = {
    "seed": SEED, "split": "8:1:1",
    "n_cells": n_cells, "n_drugs": n_drugs, "n_rows": len(r),
    "P1_test_cells": len(te_cells), "P1_val_cells": len(va_cells),
    "P2_test_drugs": len(te_drugs), "P2_val_drugs": len(va_drugs),
}
with open(os.path.join(SPL, "splits_info.json"), "w") as fh:
    json.dump(info, fh, indent=1)
print(json.dumps(info, indent=1))
print("DONE make_splits.py")
