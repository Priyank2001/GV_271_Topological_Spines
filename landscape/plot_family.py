"""
Figure for step 4 (comparison/results/loss_landscape_family.png), from loss_landscape_family.json.

    python3 -m landscape.plot_family

A  single-model landscapes: every one is a single basin, so only the mean loss over the box varies; it is
   plotted against test accuracy (colour = factor changed from the baseline).
B1 subspace through four solutions per configuration: the models' loss and the top of the lowest path between
   them; the barrier is the gap between the two.
B2 how much lower the lowest path is than the straight segment.
B3 graph minima vs real minima (gradient descent): the extra graph minima are artefacts.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .accuracy import RESULTS
from .family import spearman

COLORS = {"training stage": "#2a6fdb", "width": "#e8710a", "learning rate": "#7b3fa0", "baseline": "#8a8a8a"}


def factor(r: dict) -> str:
    if r["epoch"] != 30:
        return "training stage"
    if r["width"] != 32:
        return "width"
    if r["lr"] != 0.05:
        return "learning rate"
    return "baseline"


def scatter(ax, x, y, groups, **kw):
    for name, color in COLORS.items():
        sel = np.array([g == name for g in groups])
        if sel.any():
            ax.scatter(np.asarray(x)[sel], np.asarray(y)[sel], c=color, edgecolors="white", linewidths=0.7, zorder=3, **kw)


def style(ax, title):
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("test accuracy")


def main() -> None:
    res = json.loads((RESULTS / "loss_landscape_family.json").read_text())
    rows, cfg = res["rows"], res["configs"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), layout="constrained")

    acc = [r["test_acc"] for r in rows]
    scatter(ax[0], acc, [r["loss_mean"] for r in rows], [factor(r) for r in rows], s=40)
    ax[0].set_yscale("log")
    style(ax[0], f"A. One model, box a ∈ [−1, 1]⁴: always a single basin.\nOnly the mean loss varies "
                 f"(ρ = {spearman([r['loss_mean'] for r in rows], acc):+.2f}), in opposite\ndirections for training "
                 "stage and width")
    ax[0].set_ylabel("mean loss over the box")

    cacc = [c["test_acc"] for c in cfg]
    cg = [factor(c) for c in cfg]
    scatter(ax[1], cacc, [c["minimum_loss_mean"] for c in cfg], cg, s=46, marker="o")
    scatter(ax[1], cacc, [c["path_top_mean"] for c in cfg], cg, s=60, marker="^")
    for c in cfg:
        ax[1].plot([c["test_acc"]] * 2, [c["minimum_loss_mean"], c["path_top_mean"]], color="#bbbbbb", lw=1, zorder=1)
    style(ax[1], "B1. Four solutions per configuration: the barrier\n(grey) grows only because the minima (●) sink;\n"
                 f"the top of the lowest path (▲) stays near 2.0–2.4 (ρ = {spearman([c['path_top_mean'] for c in cfg], cacc):+.2f})")
    ax[1].set_ylabel("training objective")

    scatter(ax[2], cacc, [c["low_vs_straight"] for c in cfg], cg, s=46)
    style(ax[2], "B2. Lowest-path barrier / straight-line barrier:\nearly checkpoints are joined by much lower curved\n"
                 f"paths (ρ = {spearman([c['low_vs_straight'] for c in cfg], cacc):+.2f})")
    ax[2].set_ylabel("lowest / straight")

    scatter(ax[3], cacc, [c["minima@0.01"] for c in cfg], cg, s=46, marker="s")
    scatter(ax[3], cacc, [c["verified_minima"] for c in cfg], cg, s=46, marker="o")
    style(ax[3], "B3. Minima in the subspace: graph (■) vs real,\nby gradient descent (●). Real minima: 4 in every\n"
                 "configuration; the extras are kNN artefacts")
    ax[3].set_ylabel("minima (persistence ≥ 0.01)")
    ax[3].set_yticks(range(3, 8))

    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=c, mec="white", label=n) for n, c in COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9,
               title="changed from the baseline (width 32, lr 0.05, 30 epochs)", title_fontsize=9)
    fig.suptitle("Is a better model's loss landscape simpler? Not in its topology: one basin per model, four basins per "
                 "four solutions, whatever the quality", fontsize=12, x=0.01, ha="left")
    out = RESULTS / "loss_landscape_family.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
