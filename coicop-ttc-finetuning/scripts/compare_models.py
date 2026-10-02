"""Compare the April production model with the 30 Sep 2026 fine-tuned model (read-only).

Both models are torchTextClassifiers "basic" models stored as MLflow artifacts. They are
run here on the *same* lines: the 14,695 test lines of the first 2026 wave that the
dictionary step does not resolve (``raw_test_without_regex``), and scored with the rule of
the workflow's evaluate step (see ``fetch_s3_results.accuracies``).

Needs torch + torchtextclassifiers (==1.0.x) + mlflow, so it is not run with the project's
lightweight environment. Credentials come from the environment:
    MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, AWS_*

Text fed to the models. ``l_pr_product`` is already normalised once. The April evaluation used it
as is; the Argo ``predict-evaluate`` step (``predict-basic``) normalises it a second time
(``preprocess_text`` of codif-coicop-bdf) before predicting, and the September model was fine-tuned on
text normalised twice. Both variants are scored for both models ("as_is", "twice"); the headline
compares each model with the variant it was built for (April: as_is, September: twice).
Set TTC_REPO to a checkout of codif-coicop-bdf (branch ttc-new-train) to get ``preprocess_text``.

Sanity checks done before scoring:
  * the April model reloaded here reproduces its logged predictions on the old eval set
    (``evaluation/eval_predictions.parquet`` of the April run);
  * the September model reloaded here reproduces the workflow's predictions on S3.

Outputs (``results/``): model_comparison.json, model_comparison_by_level.csv,
model_comparison_by_group.csv, model_comparison_by_code.csv.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from torchTextClassifiers import torchTextClassifiers

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts"))
from fetch_s3_results import MAPPING, TRAIN, MODELS, accuracies, s3_read, source_of, trunc  # noqa: E402

RUNS = {
    "april": "cacf2603514b4887bbfb77e2654c9bc1",
    "september": "17ab05618a91478a8fb9246842037386",
}
TEST = "s3://projet-budget-famille/data/workflow_runs/2026-09-30/train-ttc-57c27/classify-regex/raw_test_without_regex.parquet"
TEXT_COL = "l_pr_product"
TOP_K = 5
RESULTS = HERE / "results"
REPO = Path(os.environ.get("TTC_REPO", "codif-coicop-bdf"))
CACHE = Path(os.environ.get("MODEL_CACHE", tempfile.gettempdir())) / "ttc_models"
N_BOOT = 2000
SEED = 0


def normalise_again(texts: pd.Series) -> list[str]:
    """Second pass of ``preprocess_text``, as done by the Argo predict-basic step."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("data_preparation", REPO / "classify-ttc/src/preprocessing/data_preparation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stop = json.loads((REPO / "classify-ttc/data/text/stopwords.json").read_text(encoding="utf-8"))
    df = pd.DataFrame({"t": texts.astype(str).to_numpy()})
    return mod.preprocess_text(df, "t", stop)["t"].tolist()


def load_model(run_id: str, cache: Path):
    path = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model", dst_path=str(cache / run_id)))
    meta = pickle.load(open(path / "basic_metadata.pkl", "rb"))
    idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}
    clf = torchTextClassifiers.load(path / "model")
    return clf, idx_to_label


def predict(clf, idx_to_label: dict[int, str], texts: list[str], batch: int = 2048) -> np.ndarray:
    """(n, TOP_K) array of predicted codes, best first."""
    out = []
    for i in range(0, len(texts), batch):
        res = clf.predict(np.array(texts[i : i + batch]), top_k=TOP_K)
        idx = res["prediction"].numpy()
        out.append(np.vectorize(idx_to_label.get)(idx))
    return np.vstack(out)


def as_pred_frame(test: pd.DataFrame, top: np.ndarray) -> pd.DataFrame:
    df = test.copy()
    df["predicted_code"] = top[:, 0]
    for k in range(2, TOP_K + 1):
        df[f"predicted_code_top{k}"] = top[:, k - 1]
    return df


def correct_flags(pred: pd.DataFrame, mapping: dict[str, str], level: int) -> tuple[np.ndarray, np.ndarray]:
    fold = lambda s: s.astype(str).map(lambda c: mapping.get(c, c))
    truth = trunc(fold(trunc(pred["code"], 4)), level)
    hits = [trunc(fold(trunc(pred["predicted_code"], 4)), level) == truth]
    hits += [trunc(fold(trunc(pred[f"predicted_code_top{k}"], 4)), level) == truth for k in range(2, TOP_K + 1)]
    top5 = hits[0].copy()
    for h in hits[1:]:
        top5 |= h
    return hits[0].to_numpy(), top5.to_numpy()


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """Difference in accuracy (b - a, in points), 95 % CI over lines, and exact McNemar counts."""
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = 100 * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    from math import comb

    m = only_a + only_b
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return {
        "diff_points": round(100 * float(b.mean() - a.mean()), 2),
        "ci95_low": round(float(np.percentile(d, 2.5)), 2),
        "ci95_high": round(float(np.percentile(d, 97.5)), 2),
        "only_april_right": only_a,
        "only_september_right": only_b,
        "mcnemar_p": float(p),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        cache = CACHE
        m = s3_read(MAPPING, tmp)
        mapping = dict(zip(m["code"], m["code_parent_equivalent"]))
        test = s3_read(TEST, tmp)
        train = s3_read(TRAIN, tmp)
        sept_s3 = s3_read(MODELS["stage1_stage2"] + "/predictions.parquet", tmp)

        models = {k: load_model(r, cache) for k, r in RUNS.items()}

        # ---- sanity check 1: April model reloaded here vs its logged predictions on the old eval set
        old_path = mlflow.artifacts.download_artifacts(
            run_id=RUNS["april"], artifact_path="evaluation/eval_predictions.parquet", dst_path=str(tmp)
        )
        old = pd.read_parquet(old_path)
        chk = predict(*models["april"], old[TEXT_COL].astype(str).tolist())
        agree_old = float((chk[:, 0] == old["predicted_code"].astype(str).to_numpy()).mean())
        top5_old = float(np.mean([old[f"predicted_code_top{k}"].astype(str).to_numpy() == chk[:, k - 1] for k in range(2, 6)]))
        print(f"April model vs logged predictions on old eval set: top-1 agreement {agree_old:.4f}, top-2..5 {top5_old:.4f}")

        # ---- predictions on the September test set
        variants = {"as_is": test[TEXT_COL].astype(str).tolist(), "twice": normalise_again(test[TEXT_COL])}
        n_changed = sum(a != b for a, b in zip(variants["as_is"], variants["twice"]))
        print(f"{n_changed} of {len(test)} test texts change under a second normalisation")
        allp = {(k, v): as_pred_frame(test, predict(*mod, variants[v])) for k, mod in models.items() for v in variants}
        preds = {"april": allp[("april", "as_is")], "september": allp[("september", "twice")]}

        # ---- sanity check 2: September model reloaded here vs workflow predictions
        sept_local = allp[("september", "twice")]["predicted_code"].to_numpy()
        sept_ref = sept_s3["predicted_code"].astype(str).to_numpy()
        agree_sept = float((sept_local == sept_ref).mean())
        ref_scores = accuracies(sept_s3, mapping)
        loc_scores = accuracies(allp[("september", "twice")], mapping)
        print(f"September model vs workflow predictions: agreement {agree_sept:.4f}; L4 top-1 {loc_scores['L4_top1']} vs {ref_scores['L4_top1']}")

    for p in list(preds.values()) + list(allp.values()):
        p["channel"] = source_of(p["filename_previous"])
    seen_texts = set(train[TEXT_COL].astype(str))
    for p in list(preds.values()) + list(allp.values()):
        p["seen"] = p[TEXT_COL].astype(str).isin(seen_texts)

    # label space of each model = labels it can emit
    labels = {k: set(pickle.load(open(next((CACHE / RUNS[k]).rglob("basic_metadata.pkl")), "rb"))["label_names"]) for k in RUNS}
    fold = lambda s: s.astype(str).map(lambda c: mapping.get(c, c))
    truth4 = fold(trunc(test["code"], 4))
    lab4 = {k: {mapping.get(c, c) for c in (".".join(x.split(".")[:4]) for x in v)} for k, v in labels.items()}
    common = truth4.isin(lab4["april"] & lab4["september"]).to_numpy()

    groups = {"all": np.ones(len(test), bool), "true code in both label spaces": common}
    tech = truth4.str.match(r"^(98|99)").to_numpy()
    groups["excluding technical codes 98/99"] = ~tech
    groups["technical codes 98/99 only"] = tech
    for key in ["tickets_appli", "tickets_papier", "carnets_papier"]:
        groups[f"channel: {key}"] = (preds["april"]["channel"] == key).to_numpy()
    groups["text seen in training set"] = preds["april"]["seen"].to_numpy()
    groups["new text"] = ~preds["april"]["seen"].to_numpy()

    variant_table = [
        {"model": k, "text": v, **accuracies(df_, mapping)} for (k, v), df_ in allp.items()
    ]
    pd.DataFrame(variant_table).to_csv(RESULTS / "model_comparison_text_variants.csv", index=False)
    print(pd.DataFrame(variant_table)[["model", "text", "L1_top1", "L4_top1", "L4_top5"]].to_string())
    by_level, by_group, out = [], [], {
        "checks": {"april_vs_logged_top1_agreement": agree_old, "april_vs_logged_top2_5_agreement": top5_old,
                   "september_vs_workflow_top1_agreement": agree_sept},
        "n_test": int(len(test)),
        "n_texts_changed_by_second_normalisation": int(n_changed),
        "label_space": {"april_labels": len(labels["april"]), "september_labels": len(labels["september"]),
                        "identical": labels["april"] == labels["september"]},
    }
    flags = {(k, n): correct_flags(preds[k], mapping, n) for k in RUNS for n in range(1, 5)}
    for name, mask in groups.items():
        for n in range(1, 5):
            for metric, j in (("top1", 0), ("top5", 1)):
                a, b = flags[("april", n)][j][mask], flags[("september", n)][j][mask]
                row = {"group": name, "N": int(mask.sum()), "level": n, "metric": metric,
                       "april": round(100 * float(a.mean()), 2), "september": round(100 * float(b.mean()), 2)}
                row.update(paired_bootstrap(a, b, rng))
                by_group.append(row)
    for n in range(1, 5):
        for metric, j in (("top1", 0), ("top5", 1)):
            by_level.append(next(r for r in by_group if r["group"] == "all" and r["level"] == n and r["metric"] == metric))

    # per-code (level 4) winners
    a1, b1 = flags[("april", 4)][0], flags[("september", 4)][0]
    code = pd.DataFrame({"code_l4": truth4.to_numpy(), "april_ok": a1, "sept_ok": b1})
    per = code.groupby("code_l4").agg(n=("april_ok", "size"), april=("april_ok", "mean"), september=("sept_ok", "mean")).reset_index()
    per["gain_sept_minus_april_lines"] = ((per["september"] - per["april"]) * per["n"]).round(1)
    per = per.sort_values("gain_sept_minus_april_lines")

    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(by_level).to_csv(RESULTS / "model_comparison_by_level.csv", index=False)
    pd.DataFrame(by_group).to_csv(RESULTS / "model_comparison_by_group.csv", index=False)
    per.to_csv(RESULTS / "model_comparison_by_code.csv", index=False)
    out["disagreement_L4_top1"] = {
        "both_right": int((a1 & b1).sum()), "only_april": int((a1 & ~b1).sum()),
        "only_september": int((~a1 & b1).sum()), "both_wrong": int((~a1 & ~b1).sum()),
    }
    (RESULTS / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(by_level).to_string())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
