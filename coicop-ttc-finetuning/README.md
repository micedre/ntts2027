# NTTS 2027 abstract — two-stage training of the COICOP TTC classifier

Quarto source of the abstract submitted to [NTTS 2027](https://cros.ec.europa.eu/NTTS2027)
(call open 1 June – 15 Oct 2026, blind review, Word template, max 4 pages).

| File | Role |
|---|---|
| `abstract.qmd` | The abstract. Renders to `abstract.docx` using `reference.docx` (derived from the NTTS template) as `reference-doc`. |
| `NTTS2027_Abstract_template (1).docx` | Official template (styles, A4 page setup). Do not edit. |
| `NTTS2027_Abstract_COICOP_TTC.docx` | Previous draft (architecture comparison, March 2026 numbers). Kept for reference. |
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
