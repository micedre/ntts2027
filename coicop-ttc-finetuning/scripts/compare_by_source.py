"""Accuracy by collection channel of the last retrained models vs the model currently in production (read-only).

Models (all loaded from MLflow, never from the S3 prefixes, which are overwritten by re-submitted workflows):
  * current        : ``classify-ttc-model-uri`` of argo/params.yaml (April model, run cacf2603…)
  * rfsz6 fine-tuned / rfsz6 stage 1 : workflow ``train-ttc-rfsz6`` (emb 64, lr 1e-3, batch 64, fine-tuning lr 1e-4)
  * 4l2qv annotations only           : workflow ``train-ttc-annotations-4l2qv`` (emb 64, lr 1e-3, batch 64)

Test set: the 14,695 lines of the first 2026 wave not resolved by the dictionary step; channel = file of origin
(``filename_previous``). Same scoring rule as the workflows' evaluate step (see ``fetch_s3_results.accuracies``).
Run with the environment of ``compare_models.py`` (torch, torchtextclassifiers 1.0.x, mlflow, TTC_REPO, MLFLOW_*, AWS_*).

Outputs in ``results/``: by_source_accuracy.csv, by_source_vs_current.csv, by_source_codes.csv, by_source.json.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_models as cm  # noqa: E402
from fetch_s3_results import MAPPING, TRAIN, accuracies, s3_read, source_of, trunc  # noqa: E402

TEST = cm.TEST
MODELS = {
    "current (April)": ("cacf2603514b4887bbfb77e2654c9bc1", "as_is"),
    "rfsz6 fine-tuned": ("8355c81055fb448dbb66e35a1333e142", "twice"),
    "rfsz6 stage 1 only": ("feeaae8bd0354a479e92eb5ba182b765", "twice"),
    "4l2qv annotations only": ("9065f72379d44cd1b48f8a3878d1c105", "twice"),
}
# Overall numbers printed in the workflows' evaluation reports (L4 top-1, L4 top-5), used as a check.
EXPECTED = {
    "rfsz6 fine-tuned": (72.87, 85.68),
    "rfsz6 stage 1 only": (45.07, 65.09),
    "4l2qv annotations only": (72.26, 87.45),
    "current (April)": (71.75, 84.50),  # computed earlier with this same code, see report_model_comparison.md
}
CHANNELS = ["carnets_papier", "tickets_papier", "tickets_appli"]
REF = "current (April)"
OUT = cm.RESULTS


def main() -> None:
    rng = np.random.default_rng(cm.SEED)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        m = s3_read(MAPPING, tmp)
        mapping = dict(zip(m["code"], m["code_parent_equivalent"]))
        test = s3_read(TEST, tmp)
        train = s3_read(TRAIN, tmp)
        loaded = {k: cm.load_model(r, cm.CACHE) for k, (r, _) in MODELS.items()}
    variants = {"as_is": test[cm.TEXT_COL].astype(str).tolist(), "twice": cm.normalise_again(test[cm.TEXT_COL])}
    preds = {k: cm.as_pred_frame(test, cm.predict(*loaded[k], variants[v])) for k, (_, v) in MODELS.items()}

    # ---- checks against the reports
    checks = {}
    for k, p in preds.items():
        s = accuracies(p, mapping)
        exp = EXPECTED[k]
        ok = abs(s["L4_top1"] - exp[0]) < 0.011 and abs(s["L4_top5"] - exp[1]) < 0.011
        checks[k] = {"L4_top1": s["L4_top1"], "L4_top5": s["L4_top5"], "expected": exp, "matches": bool(ok)}
        print(k, checks[k])
    if not all(c["matches"] for c in checks.values()):
        sys.exit("A reloaded model does not reproduce its evaluation report: stopping before interpreting anything.")

    # ---- groups
    truth4 = trunc(test["code"], 4).map(lambda c: mapping.get(c, c))
    tech = truth4.str.match(r"^(98|99)").to_numpy()
    chan = source_of(test["filename_previous"]).to_numpy()
    seen = test[cm.TEXT_COL].astype(str).isin(set(train[cm.TEXT_COL].astype(str))).to_numpy()
    groups = {"all": np.ones(len(test), bool)}
    for c in CHANNELS:
        groups[c] = chan == c
    for c in CHANNELS:
        groups[f"{c} | text seen in training"] = (chan == c) & seen
        groups[f"{c} | new text"] = (chan == c) & ~seen
    groups["excluding 98/99"] = ~tech
    groups["98/99 only"] = tech
    for c in CHANNELS:
        groups[f"{c} | excluding 98/99"] = (chan == c) & ~tech

    flags = {(k, n): cm.correct_flags(preds[k], mapping, n) for k in MODELS for n in range(1, 5)}
    acc_rows, cmp_rows = [], []
    for g, mask in groups.items():
        for k in MODELS:
            for n in range(1, 5):
                for metric, j in (("top1", 0), ("top5", 1)):
                    acc_rows.append({"group": g, "N": int(mask.sum()), "model": k, "level": n, "metric": metric,
                                     "accuracy": round(100 * float(flags[(k, n)][j][mask].mean()), 2)})
        for k in MODELS:
            if k == REF:
                continue
            for n in range(1, 5):
                for metric, j in (("top1", 0), ("top5", 1)):
                    a, b = flags[(REF, n)][j][mask], flags[(k, n)][j][mask]
                    cmp_rows.append({"group": g, "N": int(mask.sum()), "model": k, "level": n, "metric": metric,
                                     "current": round(100 * float(a.mean()), 2), "model_acc": round(100 * float(b.mean()), 2),
                                     **{f"vs_current_{x}": y for x, y in cm.paired_bootstrap(a, b, rng).items()}})
        # effect of pre-training at equal settings
        for n in range(1, 5):
            for metric, j in (("top1", 0), ("top5", 1)):
                a, b = flags[("4l2qv annotations only", n)][j][mask], flags[("rfsz6 fine-tuned", n)][j][mask]
                cmp_rows.append({"group": g, "N": int(mask.sum()), "model": "rfsz6 fine-tuned vs 4l2qv (effect of pre-training)",
                                 "level": n, "metric": metric, "current": round(100 * float(a.mean()), 2),
                                 "model_acc": round(100 * float(b.mean()), 2),
                                 **{f"vs_current_{x}": y for x, y in cm.paired_bootstrap(a, b, rng).items()}})

    # ---- per code, per channel: rfsz6 fine-tuned vs current (L4 top-1)
    a1, b1 = flags[(REF, 4)][0], flags[("rfsz6 fine-tuned", 4)][0]
    code_rows = []
    for c in CHANNELS:
        d = pd.DataFrame({"code_l4": truth4.to_numpy()[chan == c], "cur": a1[chan == c], "new": b1[chan == c]})
        per = d.groupby("code_l4").agg(n=("cur", "size"), current=("cur", "mean"), new=("new", "mean")).reset_index()
        per["gain_lines"] = ((per["new"] - per["current"]) * per["n"]).round(1)
        per.insert(0, "channel", c)
        code_rows.append(per)
    codes = pd.concat(code_rows)

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(acc_rows).to_csv(OUT / "by_source_accuracy.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(OUT / "by_source_vs_current.csv", index=False)
    codes.to_csv(OUT / "by_source_codes.csv", index=False)
    dis = {c: {"both_right": int((a1 & b1 & (chan == c)).sum()), "only_current": int((a1 & ~b1 & (chan == c)).sum()),
               "only_new": int((~a1 & b1 & (chan == c)).sum()), "both_wrong": int((~a1 & ~b1 & (chan == c)).sum())}
           for c in CHANNELS}
    (OUT / "by_source.json").write_text(json.dumps({"checks": checks, "disagreement_L4_top1_rfsz6_ft_vs_current": dis,
                                                   "n_by_channel": {c: int((chan == c).sum()) for c in CHANNELS}}, indent=2))
    print(json.dumps(dis, indent=2))


if __name__ == "__main__":
    main()
