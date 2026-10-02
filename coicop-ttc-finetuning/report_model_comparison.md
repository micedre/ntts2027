# April model vs 30 Sep 2026 model — which one is better?

*Standalone report (not part of the abstract). Code: `scripts/compare_models.py`; tables: `results/model_comparison*.csv|json`.*

## Verdict

**The April model is (slightly) better.** On the same 14,695 test lines, the April production model
(`mlflow-artifacts:/10/cacf2603514b4887bbfb77e2654c9bc1/artifacts/model`) beats the model fine-tuned on
2026-09-30 (`train-ttc-57c27`, MLflow run `17ab05618a91478a8fb9246842037386`) at every level:

| | April | September | Δ (Sept − April) | 95 % CI |
|---|---:|---:|---:|---|
| L1 division, top-1 | **86.5** | 84.6 | −1.9 | [−2.4, −1.5] |
| L2 group, top-1 | **84.9** | 82.7 | −2.2 | [−2.7, −1.7] |
| L3 class, top-1 | **79.1** | 77.1 | −2.0 | [−2.5, −1.4] |
| **L4 full code, top-1** | **71.8** | 70.1 | **−1.6** | [−2.1, −1.1] |
| L4 full code, top-5 | **84.5** | 83.9 | −0.6 | [−1.0, −0.2] |

The gap is small (about 1.6 points, 240 lines out of 14,695) but statistically clear (paired bootstrap, 2,000
resamples; exact McNemar p = 2·10⁻¹⁰ at L4 top-1). At L1 top-5 the two are indistinguishable (−0.3, CI [−0.6, 0.0]).
The September model does **not** justify replacing the April one on accuracy.

## What was compared, and how

- **Same lines for both models.** The April run only logs metrics on the old evaluation set (7,161 lines, 75.8 % at L4);
  the September evaluation uses the 14,695 lines of the first 2026 wave that the dictionary step does not resolve.
  These numbers are **not comparable**, so the April model was downloaded from MLflow and run here on the September
  test set (`raw_test_without_regex`, `l_pr_product` column).
- **Same scoring rule** as the workflow's `evaluate` step: truth and prediction truncated to level 4, level-4 codes ending in
  `.0` folded into their parent (`mapping_lvl4`), then truncated to level N; constant N at all levels; top-5 = right
  code among the five best.
- **Checks that the inference is right** (both pass at 100 %): the April model reloaded here reproduces its logged
  predictions on the old evaluation set (7,161 lines, top-1 and top-2..5 identical), and the September model reproduces
  the workflow's `predictions.parquet` (14,695 lines, 70.12 % at L4, as in `evaluation_report.txt`).
- **Text fed to the models.** `l_pr_product` is normalised once in the data. The April evaluation used it as is; the Argo
  `predict-evaluate` step normalises it *again* (`preprocess_text`) before predicting, which changes 6,592 of the 14,695
  texts, and the September model was fine-tuned on twice-normalised text. Each model is therefore headlined with the
  variant it was built for (April: as is, September: twice). The conclusion does not depend on this choice:

| Text variant | April L4 top-1 / top-5 | September L4 top-1 / top-5 |
|---|---:|---:|
| as is (once normalised) | 71.75 / 84.50 | 68.40 / 82.90 |
| normalised twice (as in the Argo pipeline) | 72.36 / 84.89 | 70.12 / 83.91 |

  April is ahead under both variants (by 3.4 and 2.2 points at L4 top-1).
- **No leakage in favour of either model.** No test line id appears in either training set; the share of test lines whose
  normalised text also occurs in the training set is 46 % (April's 107,886 lines) and 48 % (September's 112,681 lines).

## Where the gap is

L4 accuracy (top-1 / top-5), from `results/model_comparison_by_group.csv`:

| Subset (lines) | April | September | Δ top-1 [95 % CI] |
|---|---:|---:|---|
| All (14,695) | 71.8 / 84.5 | 70.1 / 83.9 | −1.6 [−2.1, −1.1] |
| True code known to both models (13,750) | 76.7 / 90.3 | 74.7 / 89.2 | −2.0 [−2.5, −1.5] |
| Paper diaries (10,731) | 72.7 / 84.4 | 70.9 / 83.5 | −1.8 [−2.4, −1.2] |
| Paper receipts (1,980) | 69.7 / 85.7 | 68.8 / 86.7 | −0.8 [−2.3, +0.7] |
| Receipts scanned in the app (1,984) | 68.8 / 83.9 | 67.3 / 83.5 | −1.5 [−3.0, +0.1] |
| Text seen in training (7,001) | 81.8 / 91.0 | 80.8 / 91.0 | −1.0 [−1.5, −0.4] |
| New text (7,694) | 62.6 / 78.6 | 60.4 / 77.4 | −2.2 [−3.1, −1.4] |

- The gap is significant on diaries (73 % of the lines) and on new texts; on the two receipt channels (about 2,000 lines
  each) it is within the noise, and on paper receipts the September top-5 is nominally higher (+1.0, not significant).
- The label spaces are not identical (both have 316 labels but not the same ones). Restricting to lines whose true code
  is known to both models does not reverse the ranking (−2.0 points).

Per level-4 code (`results/model_comparison_by_code.csv`, 283 codes with test lines): April is better on 106 codes, September on 49,
net −239 lines (= the −1.63 points). Both models are right on 9,720 lines and wrong on 3,568; April alone is right on 823,
September alone on 584.

- April's biggest wins: **11.1.1.1** (245 lines, 61 % → 33 %, −70 lines), **11.1.1.2** (574 lines, 35 % → 25 %, −56), then
  01.1.9.1, 01.1.2.2, 01.1.3.1 (−16 to −19 lines each).
- September's biggest wins: **06.2.3.1** (113 lines, 21 % → 58 %, +42), the technical code **98.1.1** (234 lines, 0 % → 16 %, +37),
  **07.3.2.1** (71 lines, 55 % → 89 %, +24), 01.2.1 and 01.2.2 (+21 to +22).

## Do the technical codes 98/99 explain the gap?

No. Both models predict 98/99 codes (April gets 54–69 % on 99.1, 99.2, 99.4; only 98.1.1 is missed entirely, 0 % against 16 % for
September), and these 734 lines (5 %) are where September is *ahead*. Excluding them makes April's lead larger
(`results/model_comparison_by_group.csv`):

| L4 (lines) | April | September | Δ top-1 [95 % CI] | Δ top-5 [95 % CI] |
|---|---:|---:|---|---|
| All (14,695) | 71.8 / 84.5 | 70.1 / 83.9 | −1.6 [−2.1, −1.1] | −0.6 [−1.0, −0.2] |
| Without 98/99 (13,961) | 74.6 / 87.5 | 72.7 / 86.4 | **−1.9 [−2.4, −1.4]** | −1.0 [−1.4, −0.7] |
| Only 98/99 (734) | 17.7 / 28.5 | 21.0 / 36.1 | +3.3 [+0.7, +5.9] | +7.6 [+4.2, +11.2] |

On 98/99 lines September is better at every level (top-5 +7.6 points at L4), but these lines are too few to change the overall ranking.

## Why might the newer model not be better?  (hypotheses, not tested)

The two models differ in many ways at once, so no single cause can be read off these results:

- **Fine-tuning learning rate.** MLflow: September fine-tuning used lr 0.01 and stopped at epoch 35; the April production
  run used lr 10⁻⁴ for 92 epochs (`results/mlflow_runs.csv`). A 100× larger step may move the pre-trained weights further from
  the scanner-data solution. Stage 1 also used lr 0.1 (batch 256) vs 10⁻³ (batch 512) in April.
- **Embedding size.** The April artefact's metadata records 64 dimensions, the September one 128.
- **Stage-1 corpus.** September: 1.81 M rows (1.74 M scanner + 66 k synthetic, from a new extraction and a new synthetic set); April: 1.6 M
  (1.52 M scanner + 74 k synthetic). The pre-training benefit measured earlier for September (+3.7 points over training on the
  annotations alone) could have been larger with the April hyper-parameters; this was not tested.
- **Fine-tuning data.** September adds receipts and diary lines (19 k vs 14 k pilot lines) but has no `copain` lines (1,546 in April's set).
- **Text handling.** September was fine-tuned on twice-normalised text; the effect of this on training is unknown.

A direct test would be to rerun the September workflow with the April hyper-parameters (at least fine-tuning lr 10⁻⁴; the April
model's saved metadata gives embedding 64, which differs from what the April MLflow parameters suggest, so check it first) and evaluate
on the same 14,695 lines.

## Limits

- One test set (2026 wave 1), 14,695 lines: half of the texts also occur in the training sets, which flatters absolute figures (both models score 60–63 % on new texts).
- Lines resolved by the dictionary step are excluded, so the end-to-end accuracy of the pipeline is higher for both models.
- The models were not compared on a second, independent test set, and one training run per model is available (no seed-to-seed variance),
  so a 1–2 point difference between two runs of the *same* recipe cannot be ruled out; the confidence intervals only reflect the sampling of test lines.
- Credentials used to read MLflow and S3 were provided through environment variables and are not stored in the repository.
