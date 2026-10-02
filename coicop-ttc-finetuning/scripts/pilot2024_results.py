"""Pilot-2024-only models: accuracy by collection channel and annotated-set statistics (read-only, S3 only).

Models (predictions read from the S3 outputs of the workflows, no model is reloaded):
  * stage 1 only       : ``train-ttc-mpcmw/base``            (MLflow b666c409, emb 128, lr 1e-3, batch 256)
  * pilot only         : ``train-ttc-annotations-bdj8g``     (MLflow 9aec5cd7, annotations 2024 only)
  * stage 1 + 2        : ``train-ttc-mpcmw/fine-tuned-2024`` (MLflow 82ddefeb, fine-tuned on the 2024 pilot only)
Same test set and scoring rule as ``fetch_s3_results.accuracies``.

Outputs in ``results/``: pilot2024.json, pilot2024_by_source.csv, pilot2024_by_text_overlap.csv, pilot2024_tests.csv.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_s3_results import SOURCES, accuracies, s3_read, source_of, trunc

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results"
BUCKET = "s3://projet-budget-famille/data"
RUN = f"{BUCKET}/workflow_runs/2026-09-30/train-ttc-mpcmw"
MODELS = {
    "stage1_only": f"{BUCKET}/workflow_outputs/train-ttc/train-ttc-mpcmw/base",
    "pilot_only": f"{BUCKET}/workflow_outputs/train-ttc-annotations/train-ttc-annotations-bdj8g/annotations-2024",
    "stage1_stage2": f"{BUCKET}/workflow_outputs/train-ttc/train-ttc-mpcmw/fine-tuned-2024",
}
EXPECTED = {"stage1_only": 48.3, "pilot_only": 67.8, "stage1_stage2": 69.7}  # L4 top-1 of the evaluation reports
N_BOOT = 2000


def flags(pred: pd.DataFrame, mapping: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    fold = lambda s: s.astype(str).map(lambda c: mapping.get(c, c))
    t = trunc(fold(trunc(pred["code"], 4)), n)
    hit = [trunc(fold(trunc(pred["predicted_code" if k == 1 else f"predicted_code_top{k}"], 4)), n) == t for k in range(1, 6)]
    top5 = hit[0].copy()
    for h in hit[1:]:
        top5 |= h
    return hit[0].to_numpy(), top5.to_numpy()


def paired(a: np.ndarray, b: np.ndarray, rng) -> dict:
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = 100 * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return {"diff_points": round(100 * float(b.mean() - a.mean()), 2),
            "ci95_low": round(float(np.percentile(d, 2.5)), 2), "ci95_high": round(float(np.percentile(d, 97.5)), 2)}


def distribution(codes: pd.Series) -> dict:
    c = codes.value_counts()
    return {"rows": int(c.sum()), "n_codes": int(len(c)), "median_per_code": float(c.median()),
            "top10_share": round(float(c.head(10).sum() / c.sum()), 4),
            "food_share": round(float(codes.str.startswith("01").mean()), 4), "codes_lt_100": int((c < 100).sum())}


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        mapping_df = s3_read(f"{RUN}/prune-codes/mapping_lvl4.parquet", tmp)
        mapping = dict(zip(mapping_df["code"], mapping_df["code_parent_equivalent"]))
        full = s3_read(f"{RUN}/build-datasets/annotations_full.parquet", tmp)
        train = s3_read(f"{RUN}/prune-codes/annotations_train_pruned.parquet", tmp)
        preds = {k: s3_read(f"{p}/predictions.parquet", tmp) for k, p in MODELS.items()}

    # ---- fine-tuning set of the pilot-only runs: the 2024 lines
    pilot = full[full["annee"] == 2024]
    pilot_train = train[train["annee"] == 2024]
    l4 = lambda s: s.astype(str).map(lambda c: mapping.get(c, c)).pipe(lambda x: trunc(x, 4))
    info = {"pilot_rows": int(len(pilot)), "pilot_rows_in_label_space": int(len(pilot_train)),
            "pilot_source_counts": {k: int(v) for k, v in pilot["source"].value_counts().items()},
            "pilot_distribution": distribution(trunc(pilot["code"], 4).map(lambda c: mapping.get(c, c)))}
    seen_texts = set(pilot["l_pr_product"].astype(str))

    overall, rows, overlap_rows, cmp_rows = {}, [], [], []
    for k, p in preds.items():
        overall[k] = accuracies(p, mapping)
        assert abs(overall[k]["L4_top1"] - EXPECTED[k]) < 0.06, (k, overall[k])
        p["channel"] = source_of(p["filename_previous"])
        p["seen"] = p["l_pr_product"].astype(str).isin(seen_texts)
        for key, name, _ in SOURCES:
            rows.append({"model": k, "source": key, "source_name": name, **accuracies(p[p["channel"] == key], mapping)})
        rows.append({"model": k, "source": "receipts", "source_name": "Receipts (app + paper)",
                     **accuracies(p[p["channel"].isin(["tickets_appli", "tickets_papier"])], mapping)})
        rows.append({"model": k, "source": "all", "source_name": "All", **overall[k]})
        for seen, name in [(True, "text seen in the pilot"), (False, "new text")]:
            for key in ["all"] + [s for s, _, _ in SOURCES]:
                sub = p[p["seen"] == seen] if key == "all" else p[(p["seen"] == seen) & (p["channel"] == key)]
                overlap_rows.append({"model": k, "source": key, "text": name, "N": len(sub), **accuracies(sub, mapping)})
        print(k, overall[k]["L4_top1"], overall[k]["L4_top5"])

    # ---- paired bootstrap: effect of stage 1 (stage1_stage2 vs pilot_only) and of fine-tuning (vs stage1_only)
    base = preds["stage1_stage2"]
    chan = base["channel"].to_numpy()
    groups = {"all": np.ones(len(base), bool), "receipts": np.isin(chan, ["tickets_appli", "tickets_papier"])}
    groups.update({s: chan == s for s, _, _ in SOURCES})
    for g, mask in groups.items():
        for ref, new in [("pilot_only", "stage1_stage2"), ("stage1_only", "stage1_stage2")]:
            for metric, j in (("top1", 0), ("top5", 1)):
                a = flags(preds[ref], mapping, 4)[j][mask]
                b = flags(preds[new], mapping, 4)[j][mask]
                cmp_rows.append({"group": g, "N": int(mask.sum()), "ref": ref, "new": new, "metric": metric, **paired(a, b, rng)})
    info["test_text_seen_in_pilot_share"] = round(float(base["seen"].mean()), 4)

    OUT.mkdir(exist_ok=True)
    (OUT / "pilot2024.json").write_text(json.dumps({"overall": overall, **info}, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "pilot2024_by_source.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT / "pilot2024_by_text_overlap.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(OUT / "pilot2024_tests.csv", index=False)
    print(json.dumps(info, indent=1))
    print(pd.DataFrame(rows)[["model", "source", "N", "L4_top1", "L4_top5"]].to_string())
    print(pd.DataFrame(cmp_rows).to_string())


if __name__ == "__main__":
    main()
