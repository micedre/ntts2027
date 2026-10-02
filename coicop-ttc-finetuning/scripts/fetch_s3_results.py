"""Accuracy of the 30 Sep 2026 retrained models, overall and by collection channel (read-only).

Reads the prediction files written by the Argo workflows ``train-ttc-57c27``
(stage 1 = ``base``, stage 1 + 2 = ``fine-tuned``) and ``train-ttc-annotations``
(annotations only) from S3 with the ``aws`` CLI (credentials from the environment),
and recomputes the accuracies with the rule of the workflow's ``evaluate`` step:
truth and predictions are truncated to level 4, mapped with ``mapping_lvl4`` (level-4
codes ending in ``.0`` fold into their parent), then truncated to level N; equality;
N is constant across levels. The overall numbers must match ``evaluation_report.txt``.

Writes ``results/ttc_2026-09.json`` (overall), ``results/by_source.csv`` and
``results/by_text_overlap.csv`` (lines whose normalised text also occurs in the
fine-tuning set vs lines whose text is new).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT_JSON = HERE / "results" / "ttc_2026-09.json"
OUT_SOURCE = HERE / "results" / "by_source.csv"
OUT_OVERLAP = HERE / "results" / "by_text_overlap.csv"

BUCKET = "s3://projet-budget-famille/data"
MODELS = {
    "stage1_only": f"{BUCKET}/workflow_outputs/train-ttc/train-ttc-57c27/base",
    "annotations_only": f"{BUCKET}/workflow_outputs/train-ttc-annotations/train-ttc-annotations-ggf9v/annotations",
    "stage1_stage2": f"{BUCKET}/workflow_outputs/train-ttc/train-ttc-57c27/fine-tuned",
}
MAPPING = f"{BUCKET}/workflow_runs/2026-09-30/train-ttc-57c27/prune-codes/mapping_lvl4.parquet"
TRAIN = f"{BUCKET}/workflow_runs/2026-09-30/train-ttc-57c27/build-datasets/annotations_full.parquet"

# Collection channel of a test line, read from the file it came from.
SOURCES = [
    ("tickets_appli", "Receipts scanned in the app", "tickets_appli"),
    ("tickets_papier", "Paper receipts", "tickets_papier"),
    ("carnets_papier", "Paper diaries", "carnets_papier"),
]


def s3_read(uri: str, tmp: Path) -> pd.DataFrame:
    dest = tmp / uri.rsplit("/", 1)[-1]
    subprocess.run(["aws", "s3", "cp", "--only-show-errors", uri, str(dest)], check=True)
    return pd.read_parquet(dest)


def trunc(s: pd.Series, n: int) -> pd.Series:
    return s.astype(str).map(lambda c: ".".join(c.split(".")[:n]))


def accuracies(pred: pd.DataFrame, mapping: dict[str, str]) -> dict:
    fold = lambda s: s.astype(str).map(lambda c: mapping.get(c, c))
    truth4 = fold(trunc(pred["code"], 4))
    top = [fold(trunc(pred["predicted_code"], 4))]
    top += [fold(trunc(pred[f"predicted_code_top{k}"], 4)) for k in range(2, 6)]
    out = {"N": len(pred)}
    for n in range(1, 5):
        t = trunc(truth4, n)
        hit = [(trunc(p, n) == t) for p in top]
        top1, top5 = hit[0], hit[0].copy()
        for h in hit[1:]:
            top5 |= h
        out[f"L{n}_top1"] = round(100 * float(top1.mean()), 2)
        out[f"L{n}_top5"] = round(100 * float(top5.mean()), 2)
    return out


def source_of(filename: pd.Series) -> pd.Series:
    out = pd.Series("other", index=filename.index)
    for key, _, needle in SOURCES:
        out[filename.astype(str).str.contains(needle)] = key
    return out


def main() -> None:
    overall, rows, overlap_rows = {}, [], []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        m = s3_read(MAPPING, tmp)
        mapping = dict(zip(m["code"], m["code_parent_equivalent"]))
        train = s3_read(TRAIN, tmp)
        seen_texts = set(train["l_pr_product"].astype(str))
        for label, prefix in MODELS.items():
            pred = s3_read(f"{prefix}/predictions.parquet", tmp)
            overall[label] = accuracies(pred, mapping)
            pred["channel"] = source_of(pred["filename_previous"])
            for key, name, _ in SOURCES + [("other", "Other", "")]:
                sub = pred[pred["channel"] == key]
                if len(sub):
                    rows.append({"model": label, "source": key, "source_name": name, **accuracies(sub, mapping)})
            pred["seen"] = pred["l_pr_product"].astype(str).isin(seen_texts)
            for seen, name in [(True, "text seen in fine-tuning set"), (False, "new text")]:
                for key in [None] + [k for k, _, _ in SOURCES]:
                    sub = pred[pred["seen"] == seen]
                    if key:
                        sub = sub[sub["channel"] == key]
                    overlap_rows.append({"model": label, "source": key or "all", "text": name, **accuracies(sub, mapping)})
            print(label, overall[label])

        # Composition of the fine-tuning set, to interpret the differences by channel.
        # MLflow runs behind the S3 prefixes above (the workflows train-ttc-rfsz6 and train-ttc-annotations-4l2qv were
        # submitted with the run_id of the first runs and overwrote their outputs).
        overall["mlflow_runs"] = {
            "stage1_only": "feeaae8bd0354a479e92eb5ba182b765",
            "stage1_stage2": "8355c81055fb448dbb66e35a1333e142",
            "annotations_only": "9065f72379d44cd1b48f8a3878d1c105",
        }
        overall["train_source_counts"] = {k: int(v) for k, v in train["source"].value_counts().items()}
        overall["train_rows"] = int(len(train))
        overall["train_distinct_codes"] = int(trunc(train["code"], 4).nunique())

    by_source = pd.DataFrame(rows)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(overall, indent=2))
    by_source.to_csv(OUT_SOURCE, index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT_OVERLAP, index=False)
    print(by_source.to_string())
    print(f"wrote {OUT_JSON} and {OUT_SOURCE}")


if __name__ == "__main__":
    main()
