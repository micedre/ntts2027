"""Build Figure 1 of the abstract from results/mlflow_runs.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
CSV = HERE / "results" / "mlflow_runs.csv"
OUT = HERE / "figures" / "fig_levels.png"
DIST_CSV = HERE / "results" / "code_distribution.csv"
OUT2 = HERE / "figures" / "fig_code_distribution.png"

LEVELS = ["Division (L1)", "Group (L2)", "Class (L3)", "Full code (L4)"]


def fig1() -> None:
    df = pd.read_csv(CSV).set_index("label")
    pre = df.loc["flat_pretrained"]
    ft = df.loc["flat_finetuned_prod"]

    top1_pre = [100 * pre[f"eval_level{l}_top-1"] for l in range(1, 5)]
    top1_ft = [100 * ft[f"eval_level{l}_top-1"] for l in range(1, 5)]
    top5_pre = [100 * pre[f"eval_level{l}_top-5"] for l in range(1, 5)]
    top5_ft = [100 * ft[f"eval_level{l}_top-5"] for l in range(1, 5)]

    x = range(len(LEVELS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    b1 = ax.bar([i - w / 2 for i in x], top1_pre, w, color="#9e9e9e", label="Stage 1 only: scanner + synthetic (top-1)")
    b2 = ax.bar([i + w / 2 for i in x], top1_ft, w, color="#1f4e79", label="Stage 2: fine-tuned on annotated survey data (top-1)")
    ax.scatter([i - w / 2 for i in x], top5_pre, marker="_", s=420, color="#424242", linewidths=2, label="top-5", zorder=3)
    ax.scatter([i + w / 2 for i in x], top5_ft, marker="_", s=420, color="#0b2a44", linewidths=2, zorder=3)

    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(list(x), LEVELS)
    ax.set_ylabel("Accuracy on evaluation set (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


def fig2() -> None:
    """Rank-frequency curves of examples per level-4 code."""
    if not DIST_CSV.exists():
        print(f"{DIST_CSV} missing, skipping Figure 2 (run scripts/code_distribution.py)")
        return
    d = pd.read_csv(DIST_CSV)
    series = [
        ("raw_ddc", "Raw scanner data (3.5 M rows)", "#9e9e9e", "-"),
        ("corpus_all", "Stage-1 corpus: scanner + synthetic (1.6 M)", "#1f4e79", "-"),
        ("annotated", "Annotated survey set (108 k)", "#c0504d", "--"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    for key, label, color, ls in series:
        n = d.loc[d.dataset == key, "n"].sort_values(ascending=False).to_numpy()
        ax.plot(range(1, len(n) + 1), n, color=color, ls=ls, lw=1.8, label=f"{label}, {len(n)} codes")
        ax.scatter([len(n)], [n[-1]], color=color, s=18, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("ECOICOP level-4 codes, ranked by number of examples")
    ax.set_ylabel("Examples per code (log scale)")
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT2)
    print(f"wrote {OUT2}")


def main() -> None:
    fig1()
    fig2()


if __name__ == "__main__":
    main()
