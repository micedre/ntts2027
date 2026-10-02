"""Build the figures of the abstract from results/ttc_2026-09.json and results/code_distribution.csv."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
JSON = HERE / "results" / "pilot2024.json"
OUT = HERE / "figures" / "fig_levels.png"
DIST_CSV = HERE / "results" / "code_distribution.csv"
OUT2 = HERE / "figures" / "fig_code_distribution.png"
BY_SOURCE = HERE / "results" / "pilot2024_by_source.csv"
OUT3 = HERE / "figures" / "fig_by_channel.png"

LEVELS = ["Division (L1)", "Group (L2)", "Class (L3)", "Full code (L4)"]


def fig1() -> None:
    res = json.loads(JSON.read_text())["overall"]
    series = [
        ("stage1_only", "Stage 1 only (scanner + synthetic)", "#9e9e9e", "#424242"),
        ("pilot_only", "Pilot set only, from scratch", "#7fb3d5", "#2e6f95"),
        ("stage1_stage2", "Stage 1 + 2 (pre-trained, then fine-tuned)", "#1f4e79", "#0b2a44"),
    ]
    x = list(range(len(LEVELS)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    for j, (key, label, color, dark) in enumerate(series):
        off = (j - 1) * w
        top1 = [res[key][f"L{l}_top1"] for l in range(1, 5)]
        top5 = [res[key][f"L{l}_top5"] for l in range(1, 5)]
        bars = ax.bar([i + off for i in x], top1, w, color=color, label=f"{label} (top-1)")
        ax.scatter([i + off for i in x], top5, marker="_", s=300, color=dark, linewidths=2, zorder=3,
                   label="top-5" if j == 0 else None)
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x, LEVELS)
    ax.set_ylabel("Accuracy on evaluation set (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False)
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
        ("raw_ddc", "Raw scanner data (10.3 M rows)", "#9e9e9e", "-"),
        ("corpus_all", "Stage-1 corpus: scanner + synthetic (1.8 M)", "#1f4e79", "-"),
        ("annotated", "Annotated survey set (113 k)", "#c0504d", "--"),
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


def fig3() -> None:
    """Full-code top-1 accuracy by collection channel, with the gain brought by pre-training."""
    d = pd.read_csv(BY_SOURCE)
    chans = [("tickets_appli", "Receipts scanned\nin the app"), ("tickets_papier", "Paper receipts"), ("carnets_papier", "Paper diaries")]
    series = [
        ("stage1_only", "Stage 1 only (scanner + synthetic)", "#9e9e9e"),
        ("pilot_only", "Pilot set only, from scratch", "#7fb3d5"),
        ("stage1_stage2", "Stage 1 + 2 (pre-trained, then fine-tuned)", "#1f4e79"),
    ]
    x = list(range(len(chans)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    val = {}
    for j, (key, label, color) in enumerate(series):
        ys = [float(d[(d.model == key) & (d.source == c)]["L4_top1"].iloc[0]) for c, _ in chans]
        val[key] = ys
        bars = ax.bar([i + (j - 1) * w for i in x], ys, w, color=color, label=label)
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 2), textcoords="offset points")
    for i in x:
        gain = val["stage1_stage2"][i] - val["pilot_only"][i]
        top = max(val["stage1_stage2"][i], val["pilot_only"][i]) + 5
        ax.annotate(f"pre-training: {gain:+.1f} pts", (i + 0.5 * w, top), ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#0b6b3a" if gain > 0 else "#9b2c2c")
    ax.set_xticks(x, [n for _, n in chans])
    ax.set_ylabel("Full-code (L4) top-1 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT3)
    print(f"wrote {OUT3}")


def main() -> None:
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
