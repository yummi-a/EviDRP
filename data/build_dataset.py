# build_dataset.py
# Build unified drug-response-prediction dataset from FRESH sources (D:/BI/new_data only).
# Does NOT touch D:/BI/data/raw (goal constraint).
#
# Pipeline:
#   1. Cell lines: map CTRPv2/GDSCv2 cell names -> DepMap ModelID (ACH-...) via
#      strict normalized-name match + GDSC COSMIC id match. Unmapped lines are dropped.
#   2. Response: long-format AAC / IC50 tables filtered to mapped cells.
#   3. Drugs: Morgan (ECFP) 2048-bit fingerprints from SMILES (RDKit).
#   4. Omics: expression + CNV + mutation matrices restricted to mapped cells and
#      the top-K most variable genes (K=5000, by expression variance over mapped cells).
#   5. Assemble unified tensors + metadata; generate P0/P1/P2 protocol splits.
#
# Outputs (processed/): response_long.csv, cells_feat.npy, drug_feat.npy,
#   gene_names.txt, id maps, dataset_meta.json, splits_*.json
#
# Reproducible: fixed RNG seed.

import json
import os
import sys
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.error")  # silence SMILES parse spam

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw")
PRO = os.path.join(ROOT, "processed")
SEED = 42
TOPGENES = 5000
FPRINT_BITS = 2048
FPRINT_RADIUS = 2
rng = np.random.RandomState(SEED)

os.makedirs(PRO, exist_ok=True)
os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)


def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def parse_gene(col):
    """'TSPAN6 (7105)' -> ('TSPAN6', 7105)"""
    m = re.match(r"(.+?)\s*\((\d+)\)", col)
    return (m.group(1), int(m.group(2))) if m else (col, None)


# ---------------------------------------------------------------- step 1: cells
print("== 1/5 cells ==")
models = pd.read_csv(os.path.join(RAW, "Model.csv"))
tissue = dict(zip(models["ModelID"], models["OncotreeLineage"]))

map_ct = pd.read_csv(os.path.join(PRO, "cell_map_ctrpv2.csv"))
map_gd = pd.read_csv(os.path.join(PRO, "cell_map_gdsc.csv"))
# cell_map_ctrpv2: unique.cellid -> ach ; cell_map_gdsc: unique.cellid -> ach
ct_ach = dict(zip(map_ct["unique.cellid"], map_ct["ach"]))
gd_ach = dict(zip(map_gd["unique.cellid"], map_gd["ach"]))

# ---------------------------------------------------------------- step 2: response
print("== 2/5 response ==")
ct = pd.read_csv(os.path.join(PRO, "ctrpv2_response.csv"))
gd = pd.read_csv(os.path.join(PRO, "gdscv2_response.csv"))

def prep(df, cellmap, ds):
    out = df[["cellid", "drugid", "aac_recomputed", "ic50_recomputed"]].copy()
    out.columns = ["cell_orig", "drug", "aac", "ic50"]
    out["ach"] = out["cell_orig"].map(cellmap)
    out["dataset"] = ds
    n0 = len(out)
    out = out[out["ach"].notna()].reset_index(drop=True)
    print(f"  {ds}: {n0} measurements -> {len(out)} after cell map "
          f"({out['ach'].nunique()} cells, {out['drug'].nunique()} drugs)")
    return out

ct = prep(ct, ct_ach, "CTRPv2")
gd = prep(gd, gd_ach, "GDSCv2")

# aac sanity (both recomputed by PharmacoGx, in [0,1])
for d, df in (("CTRPv2", ct), ("GDSCv2", gd)):
    print(f"  {d} aac: min={df['aac'].min():.4f} max={df['aac'].max():.4f} "
          f"mean={df['aac'].mean():.4f} | ic50 non-null={df['ic50'].notna().sum()}")

# ---------------------------------------------------------------- step 3: drugs
print("== 3/5 drugs ==")
ctd = pd.read_csv(os.path.join(PRO, "ctrpv2_drugs.csv"))
gdd = pd.read_csv(os.path.join(PRO, "gdscv2_drugs.csv"))

# GDSCv2: figure out which column response drugid matches
gdd_keys = {
    "unique.drugid": set(gdd["unique.drugid"].astype(str)),
    "DRUG_NAME": set(gdd["DRUG_NAME"].astype(str)),
}
gd_resp_drugs = set(gd["drug"].astype(str))
for col, keys in gdd_keys.items():
    cov = len(gd_resp_drugs & keys)
    print(f"  GDSCv2 response drugs match drugs[{col}]: {cov}/{len(gd_resp_drugs)}")
# CTRPv2
ct_drugs_set = set(ct["drug"].astype(str))
ct_cov = len(ct_drugs_set & set(ctd["drugid"].astype(str)))
print(f"  CTRPv2 response drugs match drugs[drugid]: {ct_cov}/{len(ct_drugs_set)}")

# fingerprint helper (canonical SMILES for cross-dataset matching; InChI not compiled)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=FPRINT_RADIUS, fpSize=FPRINT_BITS)


def fp(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    canon = Chem.MolToSmiles(mol)
    bits = np.zeros((FPRINT_BITS,), dtype=np.float32)
    bv = morgan_gen.GetFingerprint(mol)
    for i in bv.GetOnBits():
        bits[i] = 1.0
    return canon, bits

# build drug tables
def build_drugs(drugdf, idcol, smilecol, ds):
    rows = []
    for _, r in drugdf.iterrows():
        did = str(r[idcol])
        canon, b = fp(r[smilecol])
        rows.append({"drug": did, "dataset": ds, "smiles_valid": b is not None,
                     "canonical_smiles": canon, "fp": b})
    d = pd.DataFrame(rows)
    print(f"  {ds} drugs: {len(d)} total, {d['smiles_valid'].sum()} with valid SMILES/fp")
    return d

dct = build_drugs(ctd, "drugid", "cpd_smiles", "CTRPv2")
dgd = build_drugs(gdd, "unique.drugid", "smiles", "GDSCv2")

# cross-dataset drug overlap by canonical SMILES
sm_ct = dct[dct["smiles_valid"]].groupby("canonical_smiles")["drug"].apply(list).to_dict()
sm_gd = dgd[dgd["smiles_valid"]].groupby("canonical_smiles")["drug"].apply(list).to_dict()
shared_sm = sorted(set(sm_ct) & set(sm_gd))
print(f"  shared drugs by canonical SMILES between datasets: {len(shared_sm)}")
for k in shared_sm[:10]:
    print(f"    {sm_ct[k]} <=> {sm_gd[k]}")

# ---------------------------------------------------------------- step 4: omics
print("== 4/5 omics ==")
all_ach = sorted(pd.concat([ct["ach"], gd["ach"]]).dropna().unique().tolist())
print(f"  mapped cells across both datasets: {len(all_ach)}")

# expression: variance-based gene selection over mapped cells
expr = pd.read_csv(os.path.join(RAW, "OmicsExpressionProteinCodingGenesTPMLogp1.csv"))
expr = expr.rename(columns={expr.columns[0]: "ModelID"})
expr = expr[expr["ModelID"].isin(all_ach)].set_index("ModelID").astype(np.float32)
expr_genes = [parse_gene(c) for c in expr.columns]
expr_sym = [g[0] for g in expr_genes]
# keep unique symbols (drop duplicates like gene family names)
seen = {}
for i, s in enumerate(expr_sym):
    seen.setdefault(s, i)
uniq_idx = sorted(seen.values())
expr_sym = [expr_sym[i] for i in uniq_idx]
uniq_expr = expr.iloc[:, uniq_idx].copy()
uniq_expr.columns = range(uniq_expr.shape[1])  # positional labels 0..n-1
var = uniq_expr.var(axis=0).sort_values(ascending=False)
top_pos = var.index[:TOPGENES].tolist()  # integer positions
top_sym = [expr_sym[i] for i in top_pos]
expr_top = uniq_expr.iloc[:, top_pos].copy()
expr_top.columns = top_sym
print(f"  expression: {expr.shape[0]} cells x {len(expr_sym)} unique genes; selected top {TOPGENES}")

cnv = pd.read_csv(os.path.join(RAW, "OmicsCNGene.csv"))
cnv = cnv.rename(columns={cnv.columns[0]: "ModelID"})
cnv = cnv[cnv["ModelID"].isin(all_ach)].set_index("ModelID").astype(np.float32)
cnv_sym = [parse_gene(c)[0] for c in cnv.columns]
cnv.columns = cnv_sym

mut = pd.read_csv(os.path.join(RAW, "OmicsSomaticMutationsMatrixDamaging.csv"))
mut = mut.rename(columns={mut.columns[0]: "ModelID"})
mut = mut[mut["ModelID"].isin(all_ach)].set_index("ModelID").astype(np.float32)
mut_sym = [parse_gene(c)[0] for c in mut.columns]
mut.columns = mut_sym

# align all three to the same top-gene order
expr_top = expr_top.copy()
expr_top.columns = top_sym
cnv_top = cnv.reindex(columns=top_sym).fillna(0.0)
mut_top = mut.reindex(columns=top_sym).fillna(0.0)
print(f"  before row intersect: expr {expr_top.shape}, cnv {cnv_top.shape}, mut {mut_top.shape}")

# keep only cells present in ALL THREE omics matrices (no imputation)
common = sorted(set(expr_top.index) & set(cnv_top.index) & set(mut_top.index))
order = pd.Index(common)
expr_top = expr_top.reindex(order).astype(np.float32)
cnv_top = cnv_top.reindex(order).astype(np.float32)
mut_top = mut_top.reindex(order).astype(np.float32)
assert expr_top.shape == cnv_top.shape == mut_top.shape
print(f"  final omics cells (in all 3 matrices): {len(common)}; "
      f"shapes {expr_top.shape}")
cells_feat = np.hstack([expr_top.values, cnv_top.values, mut_top.values]).astype(np.float32)
print(f"  cells_feat: {cells_feat.shape}")

# ---------------------------------------------------------------- step 5: assemble
print("== 5/5 assemble ==")
resp = pd.concat([ct, gd], ignore_index=True)
# keep only cells present in all three omics matrices
resp = resp[resp["ach"].isin(common)].reset_index(drop=True)
# keep only drugs with valid fingerprints (per-dataset mask)
valid_drugs_ct = set(dct.loc[dct["smiles_valid"], "drug"])
valid_drugs_gd = set(dgd.loc[dgd["smiles_valid"], "drug"])
mask = np.zeros(len(resp), dtype=bool)
mask |= (resp["dataset"] == "CTRPv2") & resp["drug"].isin(valid_drugs_ct)
mask |= (resp["dataset"] == "GDSCv2") & resp["drug"].isin(valid_drugs_gd)
resp = resp[mask].reset_index(drop=True)
resp["aac"] = resp["aac"].astype(np.float32)
resp["ic50"] = resp["ic50"].astype(np.float32)
print(f"  final response: {len(resp)} measurements, "
      f"{resp['ach'].nunique()} cells, {resp['drug'].nunique()} drugs")

# drug fingerprint matrix aligned to sorted unique drugs
drug_list = sorted(resp["drug"].unique().tolist())
drows = []
for d in drug_list:
    # find fp (search both drug tables)
    b = None
    row = dct[dct["drug"] == d]
    if len(row):
        b = row.iloc[0]["fp"]
    else:
        row = dgd[dgd["drug"] == d]
        if len(row):
            b = row.iloc[0]["fp"]
    if b is None:
        raise RuntimeError(f"drug {d} has no fp")
    drows.append(b)
drug_feat = np.vstack(drows).astype(np.float32)
print(f"  drug_feat: {drug_feat.shape}")

# save
np.save(os.path.join(PRO, "cells_feat.npy"), cells_feat)
np.save(os.path.join(PRO, "drug_feat.npy"), drug_feat)
with open(os.path.join(PRO, "gene_names.txt"), "w") as fh:
    fh.write("\n".join(top_sym))
resp.to_csv(os.path.join(PRO, "response_long.csv"), index=False)
pd.DataFrame({"cell": common, "tissue": [tissue.get(a, "NA") for a in common]}).to_csv(
    os.path.join(PRO, "cells_meta.csv"), index=False)
pd.DataFrame({"drug": drug_list}).to_csv(os.path.join(PRO, "drugs_meta.csv"), index=False)
dct.drop(columns=["fp"]).to_csv(os.path.join(PRO, "drugs_ctrpv2_feat.csv"), index=False)
dgd.drop(columns=["fp"]).to_csv(os.path.join(PRO, "drugs_gdscv2_feat.csv"), index=False)

meta = {
    "seed": SEED,
    "n_measurements": int(len(resp)),
    "n_cells": int(resp["ach"].nunique()),
    "n_drugs": int(resp["drug"].nunique()),
    "n_top_genes": TOPGENES,
    "n_cells_features": int(cells_feat.shape[1]),
    "n_drug_features": int(drug_feat.shape[1]),
    "fingerprint": f"Morgan r={FPRINT_RADIUS} {FPRINT_BITS}bit",
    "cell_map_coverage": {
        "CTRPv2": f"{int(map_ct['ach'].notna().sum())}/{len(map_ct)}",
        "GDSCv2": f"{int(map_gd['ach'].notna().sum())}/{len(map_gd)}",
    },
    "measurements_by_dataset": resp.groupby("dataset").size().to_dict(),
    "cells_by_dataset": resp.groupby("dataset")["ach"].nunique().to_dict(),
    "drugs_by_dataset": resp.groupby("dataset")["drug"].nunique().to_dict(),
    "shared_drugs_canonical_smiles": len(shared_sm),
    "aac_mean": float(resp["aac"].mean()),
    "aac_std": float(resp["aac"].std()),
}
with open(os.path.join(PRO, "dataset_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
print(json.dumps(meta, indent=1))
print("DONE build_dataset.py")
