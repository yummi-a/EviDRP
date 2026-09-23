# Packaging notes

Provenance for this folder. Not needed to use or reproduce the code — it records what was copied
from where, what was changed, and what was left out, for anyone comparing this package against the
original working tree (`DRP/new_data/`, `DRP/new_paper/`).

## What was copied

A **copy**, never a move — the original tree is untouched.

| Destination | Source |
|---|---|
| `data/` | `new_data/{build_dataset,make_splits,make_features,make_features_multimodal}.py` |
| `src/` | `new_data/exp/{models,train_utils,models_survey,evidrp_v2}.py` |
| `experiments/` | 5 scripts from `new_data/exp/` |
| `figures/` | `new_data/exp/make_figures_nature.py` |
| `paper/figs/` | the 2 `new_paper/figs/*.pdf` the paper includes |
| `results/` | 26 CSVs + 3 NPZ from `new_data/results/` |

`assets/fig1_architecture.png` (the README's Fig. 1) is a raster of `paper/figs/fig1_architecture.pdf`
made with PyMuPDF, written by hand like the files below.

The build is scripted and idempotent: `DRP/_mk_evidrp.py` (kept at the repo root, outside this
folder). It wipes and rebuilds `data/ src/ experiments/ figures/ paper/ results/` from the source
tree, applying the rewrites and trims below. Re-running it reproduces this folder exactly.

`.gitignore`, `README.md`, `requirements.txt`, `assets/`, this file and the paper PDF at the root are
written by hand: `_mk_evidrp.py` neither writes nor deletes them, so re-running it cannot undo them.

## Scope

The package contains what the paper PDF at the root cites. The rule, applied one artifact at a time:

- a **script** stays iff its output is a number, table or figure in that PDF;
- a **result file** stays iff a kept script writes it *and* the paper reports it;
- a **figure PDF** stays iff the paper includes it.

The paper has 2 figures (framework overview; risk-coverage), 3 tables (survey accuracy; uncertainty
quality and cost; decoupling ablation) and a cross-dataset paragraph. `_mk_evidrp.py` mirrors that
one-for-one in `EXP_FILES`, `FIG_FILES`, `RESULT_KEEP`, `PRED_KEEP` and `PAPER_FIGS`; anything not
named there is out.

Both paper figures are shipped as PDFs, but only one has a generator here: Fig. 2 comes from
`figures/make_figures_nature.py`. The upstream `fig1_drawio.py` fails that rule — see *Dropped*.

## Path rewrites

Only paths changed; no algorithm, hyperparameter or metric was touched.

- Scripts moved one level down from a root that already held `results/`, `feat/`, `splits/`, so
  `ROOT = dirname(dirname(__file__))` now resolves to `eviDRP/` instead of `new_data/` and the
  data-path logic needed no edits.
- `experiments/*` add `../src` to `sys.path`, since the library modules moved out of the flat `exp/`
  folder. Scripts that already had a `sys.path.insert` line got the extra line.
- Figure scripts write to `paper/figs` instead of `../new_paper/figs`.

## Behaviour fixes

One correction, on the paper's critical path:

- `run_survey_seeds.py` no longer imports or instantiates the KAN baseline: `EviDRPKAN` is in no
  table, figure or number of the paper, and the unconditional `from kan import EviDRPKAN` forced
  `src/kan.py` into the package to keep the hub importable.

## Dropped

Beyond the allow-lists, everything else in the two source trees is out of scope, including:

- `new_data/exp/fig1_drawio.py`: it draws a superseded single-row pipeline layout and writes it to
  `paper/figs/fig1_architecture.pdf` — the paper's **Fig. 1**, which is a four-panel framework
  diagram, a different figure. Verified by exporting the script's `.drawio` source with the draw.io
  CLI: the result carries none of the four-panel figure's text ("1. DRUG AND CELL-LINE INPUTS",
  "Per-omics Encoder"). Shipping it would let a re-run overwrite the paper's Fig. 1 with the wrong
  diagram, so Fig. 1 is committed without a generator (see *Not reproducible*). The upstream
  hand-authored `new_paper/figs/fig1_architecture.svg` carries the four-panel figure's labels and is
  almost certainly its artwork, but it is not a reproducible source either — no script renders it —
  so it was left upstream rather than shipped as one;
- the **other manuscript**: `new_paper/{main,supplementary}.tex`, their compiled PDFs and the 7 figure
  PDFs only they include (`fig3_calibration`, `fig5_ood`, `fig6_uncertainty`, `fig8_survey`,
  `fig10_uncbaselines`, `figS_single_ablation`, `figS2_cross_platform`) — a different paper with
  different tables, kept upstream in `new_paper/`;
- **experiments the paper does not report**: the 2026 graph baselines (`run_2026.py`,
  `summarize_2026.py`, `results_2026/`), single-omics and component ablations
  (`run_singlemodal_ablation.py`, `make_figS_single_ablation.py`, `make_ablation_tables.py`), bucket /
  subgroup analysis (`bucket_analysis.py`, `gen_bucket_s3.py`), cross-platform and shared-drug
  analyses (`analyze_cross_platform.py`, `fig_s2_cross_platform.py`, `saltstrip_shared_drugs.py`,
  `analyze_xd_shared.py`, `run_xd_shared_seeds.py`), significance testing (`significance.py`,
  `make_sig_table.py`), split-variability controls (`run_split_var.py`, `run_p2_split_var.py`),
  independent calibration (`run_indep_cal.py`), scoring rules (`run_scoring.py`), variance
  diagnostics (`var_cal_diag.py`), duplication control (`analyze_dup.py`), the older benchmark runner
  (`run_experiments.py`) and its survey fallback (`run_survey.py`);
- **superseded drafts and their outputs**: `dev_evi_v2.py`, `merge_evi2.py`, `scan_singlestage.py`,
  `focus_singlestage.py`, `diag.py`, `analyze_seeds.py`, `analyze_diagnostics.py`,
  `run_calib_sens.py`, `make_results_tables.py`, `survey_table_tex.py`, `fig1_pipeline.py`,
  `make_figures.py`, the KAN sweep, and the `_smoke` / `_s0check` / `_v3smoke` / `v2*` survey drafts;
- **feature builders nothing kept reads**: `make_features_extra.py`, `make_features_drugtarget.py`
  (the drug-target profile is switched off in the shipped config) and `prep_2026.py` (graphs for the
  2026 baselines);
- `results/models/*.pt` (no script loads a checkpoint), `data/raw/` (2.9 GB) and `processed/`
  (141 MB), both rebuildable from the public sources.

## Not reproducible from this package

- **Fig. 1**: `paper/figs/fig1_architecture.pdf` is the four-panel framework diagram as it appears in
  the paper (a Chromium/Skia render), but no script here draws it. It is committed as the paper's
  artwork.

## Verification performed

- Every number in the paper's Tables 1–3 and §3.6 recomputes exactly from the 26 shipped CSVs
  (`DRP/_verify_paper_numbers.py`): 15 Table-1 cells, 3 Table-2 rows, 3 cost rows, 2 Table-3 rows,
  2 cross-dataset rows — all match at the precision printed in the PDF.
- `fig4_riskcoverage.pdf` regenerates byte-identically from `figures/make_figures_nature.py` once
  embedded PDF timestamps are ignored (`_pdfcmp.py`), and `fig1_architecture.pdf` matches its
  upstream copy byte-for-byte — it is committed, not generated.
- All 14 modules import cleanly (`_import_smoke.py`): 9 with no missing dependency, 5 that load
  `processed/` or `splits/` at import time failing on `FileNotFoundError` only. No ImportError.
- Removing the KAN baseline leaves `make_model` intact for the remaining methods, and the evi2 model
  still reports 133,509 parameters — the count in Table 2.
- `.gitignore` patterns are anchored to the repository root, so `paper/figs/` — where the paper's
  figures live — is never caught by a `figs/`-style rule.

**Not verified**: the training scripts were not re-run (they need `processed/`, `splits/` and `feat/`,
which are not shipped, plus CPU/GPU hours), so the verified chain is results → figures → paper, not
raw data → results.
