"""
Figure for step 2: accuracy of the scattered kNN extremum graph against the dense PL grid.

    python3 -m landscape.plot_accuracy --runs cnn_m4_e30 cnn_m4_e30_spawn2 --tau 0.005

Reads comparison/results/loss_landscape_accuracy_<run>.json; one row per landscape, two panels:
minima found (against the reference's count) and median barrier error (loss units, relative), against N. One colour per
k (fixed order), Sobol solid, uniform dashed; mean over seeds with a +-1 sd band.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from .accuracy import RESULTS

K_COLORS = {8: "#2a6fdb", 15: "#e8710a", 30: "#7b3fa0"}
STYLE = {"sobol": "-", "uniform": "--"}
PANELS = [("found_minima", "Minima found (grey line: the reference's count; every reference minimum is found in all runs)", None),
          ("barrier_rel_err_median", "Barrier height error vs reference, median over pairs of minima (relative, in loss)", None)]
TITLES = {"cnn_m4_e30": "Independent runs: 4 deep minima on a high plateau",
          "cnn_m4_e30_spawn2": "Children of one parent: 4 shallow minima, low saddles"}


def series(rows, tau, sampler, k, field):
    ns = sorted({r["n"] for r in rows})
    mean, sd = [], []
    for n in ns:
        v = np.array([r[field] for r in rows if r["tau"] == tau and r["sampler"] == sampler and r["k"] == k
                      and r["n"] == n and r[field] is not None and np.isfinite(r[field])], float)
        mean.append(v.mean() if len(v) else np.nan)
        sd.append(v.std() if len(v) else np.nan)
    return np.array(ns), np.array(mean), np.array(sd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["cnn_m4_e30", "cnn_m4_e30_spawn2"])
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = {r: json.loads((RESULTS / f"loss_landscape_accuracy_{r}.json").read_text()) for r in args.runs}
    fig, axes = plt.subplots(len(args.runs), 2, figsize=(13, 4.0 * len(args.runs)), squeeze=False)
    for row, run in enumerate(args.runs):
        rows = data[run]["rows"]
        ref = data[run]["reference"][str(args.tau)]["minima"]
        for col, (field, label, ylim) in enumerate(PANELS):
            ax = axes[row, col]
            for k, color in K_COLORS.items():
                for sampler, ls in STYLE.items():
                    n, m, s = series(rows, args.tau, sampler, k, field)
                    if np.isnan(m).all():
                        continue
                    ax.plot(n, m, ls, color=color, lw=2, marker="o", ms=4,
                            label=f"k = {k}, {sampler}" if (row, col) == (0, 0) else None)
                    ax.fill_between(n, m - s, m + s, color=color, alpha=0.12, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted({r["n"] for r in rows}))
            ax.set_xticklabels([f"{x // 1024}k" for x in sorted({r["n"] for r in rows})])
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(label, fontsize=8.5, loc="left", color="#444444")
            if field == "found_minima":
                ax.axhline(ref, color="#888888", lw=1.2, zorder=0)
                ax.set_ylim(0, None)
            else:
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if row == len(args.runs) - 1:
                ax.set_xlabel("scattered samples N")
        axes[row, 0].annotate(f"{TITLES.get(run, run)} (reference: {ref} minima)", xy=(0, 1.13),
                              xycoords="axes fraction", fontsize=11, fontweight="bold")
    fig.legend(loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Scattered kNN extremum graph vs a 49³ PL grid: every reference minimum is found in every run; false minima\n"
                 f"come from k = 8 (and uniform samples on the high plateau); barrier errors shrink with N (τ = {args.tau})",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), h_pad=3.5)
    out = args.out or RESULTS / f"loss_landscape_accuracy_tau{args.tau}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
