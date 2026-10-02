# NTTS 2027 abstract — two-stage training of the COICOP TTC classifier

Quarto source of the abstract submitted to [NTTS 2027](https://cros.ec.europa.eu/NTTS2027)
(call open 1 June – 15 Oct 2026, blind review, Word template, max 4 pages).

| File | Role |
|---|---|
| `abstract.qmd` | The abstract. Renders to `abstract.docx` using `reference.docx` (derived from the NTTS template) as `reference-doc`. |
| `NTTS2027_Abstract_template (1).docx` | Official template (styles, A4 page setup). Do not edit. |
| `NTTS2027_Abstract_COICOP_TTC.docx` | Previous draft (architecture comparison, March 2026 numbers). Kept for reference. |
| `scripts/fetch_s3_results.py` | **Source of Tables 2–3 and Figure 2.** Recomputes the accuracies of the 30 Sep 2026 models (stage 1, annotations only, stage 1 + 2) from the prediction files of the Argo workflows on S3 (`aws` CLI, read-only): overall → `results/ttc_2026-09.json`, by collection channel (paper diaries / paper receipts / app receipts, from `filename_previous`) → `results/by_source.csv`, by seen/new text → `results/by_text_overlap.csv`. |
| `scripts/fetch_mlflow_results.py` | Exports the MLflow runs cited in the text to `results/mlflow_runs.csv` and derives a few numbers from the evaluation artifacts (`results/derived.json`). Read-only. |
| `scripts/code_distribution.py` | Counts examples per ECOICOP level-4 code in the raw scanner extraction, the stage-1 corpus (DDC + synthetic) and the annotated set → `results/code_distribution.csv` + `results/code_distribution_summary.json`. Needs `DDC_ENCRYPTION_KEY` (parquet key of the DDC files) in the environment; the key is never stored. Read-only. |
| `scripts/make_figures.py` | Builds `figures/fig_code_distribution.png` (Figure 1) and `figures/fig_levels.png` (Figure 2) from the CSVs. |
| `scripts/make_reference_doc.py` | Copies the NTTS template to `reference.docx` (git-ignored) and adds the paragraph and table styles pandoc uses but the template lacks (`Table` with borders and bold header row, `Table Caption`, `Image Caption`, `Compact`, `First Paragraph`). Without it tables have no borders and very tall rows. |
| `scripts/fix_docx_tables.py` | Post-processes `abstract.docx`: absolute table width and explicit cell widths on every pandoc table (idempotent). |
| `render.sh` | The whole chain above in one command. |

## Render

```bash
cd ntts2027/
uv sync
export MLFLOW_TRACKING_USERNAME=...   # read-only account
export MLFLOW_TRACKING_PASSWORD=...
export DDC_ENCRYPTION_KEY=...           # optional: only to recompute the code distribution
uv run python scripts/fetch_mlflow_results.py   # refresh results/*.csv|json (optional, files are committed)
uv run python scripts/make_figures.py           # refresh figures/fig1_levels.png
./render.sh                                     # reference.docx + figure + quarto render + table fix
```

`render.sh` runs, in order: `scripts/make_reference_doc.py` (template + the `Table`, `Table Caption`, `Image Caption`, `Compact` and `First Paragraph` styles pandoc needs), `scripts/code_distribution.py` (only if `DDC_ENCRYPTION_KEY` is set, otherwise the committed results are kept), `scripts/make_figures.py`, `quarto render abstract.qmd`, then `scripts/fix_docx_tables.py` which gives every pandoc table an absolute width and explicit cell widths so that Word, LibreOffice and online previewers all lay the columns out the same way.

```bash
# equivalent, step by step
uv run python scripts/make_reference_doc.py
quarto render abstract.qmd
uv run python scripts/fix_docx_tables.py abstract.docx
```

`MLFLOW_TRACKING_URI` defaults to the project's MLflow server; override it if needed.
Credentials are never stored in this folder.

## Where the numbers come from

**Current (30 Sep 2026, branch `ttc-new-train` of `InseeFrLab/codif-coicop-bdf`)**, evaluation set `raw_test_without_regex` of the 2026 wave 1 (14,695 lines):

| Row | Argo workflow | MLflow run |
|---|---|---|
| Stage 1 only | `train-ttc-rfsz6`, output `base/` | `feeaae8bd0354a479e92eb5ba182b765` (emb 64, lr 1e-3, batch 64) |
| Stage 1 + 2 | `train-ttc-rfsz6`, output `fine-tuned/` | `8355c81055fb448dbb66e35a1333e142` (fine-tuning lr 1e-4) |
| Annotations only | `train-ttc-annotations-4l2qv` | `9065f72379d44cd1b48f8a3878d1c105` |

**Warning:** these two workflows were submitted with the run ids of the first runs (`train-ttc-57c27`, `train-ttc-annotations-ggf9v`), so their S3 outputs sit under those prefixes (`data/workflow_outputs/train-ttc/train-ttc-57c27/{base,fine-tuned}/`, `.../train-ttc-annotations/train-ttc-annotations-ggf9v/annotations/`) and **overwrote** the first runs' outputs (emb 128, lr 0.1: MLflow `1be89906…`, `17ab0561…`, `643c9aaf…`). Pilot-2024-only experiments: `train-ttc-mpcmw` (stage 1 `b666c409…`, fine-tuned `82ddefeb…`, output `fine-tuned-2024/`) and `train-ttc-annotations-bdj8g` (`9aec5cd7…`, output `annotations-2024/`). Model in production at the time: `argo/params.yaml` `classify-ttc-model-uri` = April run `cacf2603…` (64-dimensional embeddings, not 128 as written for the earlier stage-1 run below). Further reports: `report_model_comparison.md`, `report_by_source.md`; companion abstract of a colleague: `colleague.txt`.

Data behind Table 1 / Figure 1: raw scanner extraction `data/ddc_raw/ddc_raw_20260904.parquet`, stage-1 corpus `data/ddc_raw/ddc_train_20260930-full.parquet` (1,807,302 rows = 1,741,059 scanner + 66,243 synthetic, 316 codes), annotated set `annotations_full.parquet` (112,681 lines, 349 codes) — recomputed by `scripts/code_distribution.py`. Training details from MLflow: stage 1 Adam lr 0.1, batch 256, stopped at epoch 15 (val acc 0.790); fine-tuning lr 0.01, epoch 35, 89,422/22,356 train/val lines, 903 dropped; annotations only lr 0.1, epoch 7, 90,115/22,529, 37 dropped.

**Previous (April 2026), old evaluation set, no longer cited in the text:**

Headline (Table 2, Figure 2), same evaluation set `raw_test_without_regex` (7,161 lines):

| Row | MLflow experiment / run | Note |
|---|---|---|
| Stage 1 only | `ttc-basic` (9) / `46e66d62…` | flat, 1.6 M scanner + synthetic, emb 128, 12 epochs |
| Stage 1 + 2 | `ttc-finetune-basic` (10) / `cacf2603…` | fine-tuned on 107 k annotated lines, 92 epochs. **Model deployed in `argo/codif-pipeline.yaml` (`ttc-model-uri`).** |

For the record (not in the text): `b3601a19…`, `24bcffe1…` (same pair evaluated on the full `raw_test`, 7,972 lines).
Ablations: `7e1e4998…` (DDC only), `5ea0476e…` (+GTIN), `afadee02…` / `047422ca…` / `58b213d9…` (fine-tuning length, March pilot).
Everything is listed with run ids in `scripts/fetch_mlflow_results.py` and exported to `results/mlflow_runs.csv`.

Data described in the text: `s3://projet-budget-famille/data/training/ddc-training-dataset-9899.parquet`
(pre-training corpus) and `s3://projet-budget-famille/data/output-annotation-consolidated/annotations-consolidated-2026-04-15/`
(`raw_train.parquet`, `raw_test.parquet`). Code distribution (Table 1, Figure 1): raw scanner extraction `data/ddc_raw/annee=2025/ddc_bdf_1112.parquet` + `annee=2026/ddc_bdf_010203.parquet`, the corpus above, and `raw_train.parquet`; summary in `results/code_distribution_summary.json`.

## Before submitting

- [ ] Delete the *Side note for the authors* callout at the end of `abstract.qmd`.
- [ ] Open `abstract.docx` in Word: check it fits in 4 pages, captions are "Figure 1." below / "Table 1." above, tables have borders, no leftover highlighting.
- [ ] Figure and table numbers are written by hand in `abstract.qmd` (Quarto cross-references were removed because they wrap docx tables in an over-wide layout table); keep them in sync if you add one.
- [ ] Blind review: no author names or affiliation anywhere. Reference 2 (torchTextClassifiers GitHub organisation) is the only hint of provenance; keep or replace with a generic "open-source library" citation as the team prefers.
- [ ] Optional: run the from-scratch baseline described in the side note and add it as a third row of Table 1.
- [ ] Submit through CMT: <https://cmt3.research.microsoft.com/NTTS2027/Submission/Index> (deadline 15 October 2026).
