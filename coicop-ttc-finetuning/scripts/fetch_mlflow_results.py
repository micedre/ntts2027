"""Export the MLflow runs cited in the NTTS 2027 abstract (read-only).

Reads params/metrics of a fixed list of runs through the MLflow REST API and
writes ``results/mlflow_runs.csv``. Also downloads the evaluation predictions
artifact of the two headline runs to derive a few numbers the abstract quotes
(size of the evaluation set, share of unreachable codes).

Credentials come from the environment, never from the code:

    export MLFLOW_TRACKING_URI=https://projet-budget-famille-mlflow.user.lab.sspcloud.fr
    export MLFLOW_TRACKING_USERNAME=...
    export MLFLOW_TRACKING_PASSWORD=...
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent.parent
OUT_CSV = HERE / "results" / "mlflow_runs.csv"
OUT_DERIVED = HERE / "results" / "derived.json"

# (label, run_id, note) — order matters for the tables in the abstract.
RUNS = [
    # Headline pair: same eval set (raw_test_without_regex), flat classifier.
    ("flat_pretrained", "46e66d62712a4ca384e3a8940835b69d", "exp 9 ttc-basic, 1.6M scanner+synthetic"),
    ("flat_finetuned_prod", "cacf2603514b4887bbfb77e2654c9bc1", "exp 10 ttc-finetune-basic, production model"),
    # Flat classifier on raw_test (full consolidated test split), for the record.
    ("flat_pretrained_rawtest", "b3601a1900494bcb8addb0a66b6af63f", "exp 9, 1.53M, eval raw_test"),
    ("flat_finetuned_rawtest", "24bcffe175bd49f096c621d91ceacbd6", "exp 10, eval raw_test"),
    # Pre-training data ablation (flat, eval raw_test).
    ("flat_ddc_only", "7e1e4998a7b44a70b182b41ab3629edb", "exp 9, DDC only 221k"),
    ("flat_ddc_synth_gtin", "5ea0476e40364e1a993536918d20c086", "exp 9, DDC+synthetic+GTIN 2.49M"),
    # Fine-tuning length ablation (March pilot-only runs, eval annotations_finetune).
    ("pilot_ft_1_epoch", "afadee02ec6f426bbcb1e3308e80316f", "exp 10, 1 epoch"),
    ("pilot_ft_100_epochs", "217dc97962ad4cc08476d8acbc910b93", "exp 10, 100 epochs, eval annotations_clean"),
    ("pilot_ft_early_stopped", "047422ca72174554a2e5f6c88635b6d2", "exp 10, early stopped (219 epochs)"),
    ("pilot_pretrained", "58b213d98d8b415e9a51cbb8d9b39647", "exp 9, 1.53M, eval annotations_finetune"),
]

PARAM_KEYS = [
    "classifier_type", "task", "data_path", "annotations_path", "model_path",
    "num_samples", "unique_codes", "embedding_dim", "batch_size", "lr",
    "num_epochs", "eval_data_path", "max_level", "n_attention_layers",
]
METRIC_KEYS = (
    [f"eval_level{l}_top-{k}" for l in range(1, 5) for k in (1, 5)]
    + ["epoch", "val_accuracy", "train_samples", "val_samples",
       "ft_train_samples", "ft_val_samples", "ft_dropped_samples"]
)


def _session() -> tuple[requests.Session, str]:
    uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        "https://projet-budget-famille-mlflow.user.lab.sspcloud.fr",
    ).rstrip("/")
    user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    pwd = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if not (user and pwd):
        sys.exit("Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD")
    s = requests.Session()
    s.auth = (user, pwd)
    return s, uri


def get_run(s: requests.Session, uri: str, run_id: str) -> dict:
    r = s.get(f"{uri}/api/2.0/mlflow/runs/get", params={"run_id": run_id}, timeout=60)
    r.raise_for_status()
    return r.json()["run"]


def get_artifact(s: requests.Session, uri: str, run_id: str, path: str) -> bytes:
    r = s.get(f"{uri}/get-artifact", params={"run_id": run_id, "path": path}, timeout=120)
    r.raise_for_status()
    return r.content


def level(code: str, n: int) -> str:
    return ".".join(str(code).split(".")[:n])


def derive_from_predictions(pred: pd.DataFrame, label: str) -> dict:
    """Numbers the abstract quotes that are not logged as MLflow metrics.

    Replicates ``codif-ttc/src/evaluation/topk_accuracy.py``: for level N the
    denominator is the set of rows whose true code is annotated at least to
    depth N (``levelN`` not null) and whose prediction has that depth.
    """
    truths = pred["code"].astype(str)
    top1 = pred["predicted_code"].astype(str)
    technical = truths.str.match(r"^(98|99)")
    depth = truths.str.count(r"\.") + 1
    per_level = {}
    for n in range(1, 5):
        valid = pred[f"level{n}"].notna() & pred[f"predicted_level{n}"].notna()
        sub = pred[valid]
        hit1 = sub[f"predicted_level{n}"].astype(str) == sub[f"level{n}"].astype(str)
        hit5 = hit1.copy()
        for k in range(2, 6):
            hit5 |= sub[f"predicted_level{n}_top{k}"].astype(str) == sub[f"level{n}"].astype(str)
        per_level[n] = {"N": int(valid.sum()), "top1": round(float(hit1.mean()), 4), "top5": round(float(hit5.mean()), 4)}
    # Label space actually reachable: codes emitted anywhere in the top-5 lists
    emitted: set[str] = set()
    for c in ["predicted_code"] + [f"predicted_code_top{k}" for k in range(2, 6)]:
        emitted |= set(pred[c].dropna().astype(str))
    l4_truth = pred["level4"].dropna().astype(str)
    out = {
        "n_eval_rows": int(len(pred)),
        "n_distinct_true_codes": int(truths.nunique()),
        "truth_depth_distribution": {int(k): int(v) for k, v in depth.value_counts().sort_index().items()},
        "share_technical_codes_98_99": round(float(technical.mean()), 4),
        "n_codes_emitted_in_top5": len(emitted),
        "n_distinct_true_level4_codes": int(l4_truth.nunique()),
        "share_level4_rows_with_reachable_code": round(float(l4_truth.isin(emitted).mean()), 4),
        "per_level": per_level,
    }
    return {label: out}


def main() -> None:
    s, uri = _session()
    rows = []
    for label, run_id, note in RUNS:
        run = get_run(s, uri, run_id)
        info, data = run["info"], run.get("data", {})
        params = {p["key"]: p["value"] for p in data.get("params", [])}
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        row = {
            "label": label,
            "run_id": run_id,
            "experiment_id": info["experiment_id"],
            "run_name": info.get("run_name"),
            "start_time": pd.to_datetime(info["start_time"], unit="ms").strftime("%Y-%m-%d"),
            "note": note,
        }
        row.update({k: params.get(k) for k in PARAM_KEYS})
        row.update({k: metrics.get(k) for k in METRIC_KEYS})
        rows.append(row)
        print(f"{label:28s} L1={metrics.get('eval_level1_top-1')!s:8.8} L4={metrics.get('eval_level4_top-1')!s:8.8} top5={metrics.get('eval_level4_top-5')!s:8.8}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    derived: dict = {}
    for label, run_id in [("flat_pretrained", RUNS[0][1]), ("flat_finetuned_prod", RUNS[1][1])]:
        raw = get_artifact(s, uri, run_id, "evaluation/eval_predictions.parquet")
        pred = pd.read_parquet(io.BytesIO(raw))
        derived.update(derive_from_predictions(pred, label))
    OUT_DERIVED.write_text(json.dumps(derived, indent=2))
    print(json.dumps(derived, indent=2))


if __name__ == "__main__":
    main()
