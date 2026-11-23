"""
Figure for step 3: the 2-D slices themselves, with what the full-subspace graph says about their minima.

    python3 -m landscape.plot_slices --runs cnn_m4_e30 cnn_m4_e30_spawn2 --max-planes 4

One row per run: the planes through three models (up to ``--max-planes``) and Li et al.'s random plane around
model 0. Heatmap: log10 training loss (one colour scale per panel). Markers: trained models (star), slice
minima that are true minima of the full subspace (white circle), displaced minima (red cross), slice saddles
(small black diamond). Classification read from comparison/results/loss_landscape_slices_<run>.json.
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import freudenthal_grid_graph

from .accuracy import RESULTS
from .run import LandscapeRun
from .slices import triple_planes


def slice_graph(loss, n, lo, hi, tau):
    f = (-np.log(loss) - lo) / (hi - lo)
    return extremum_graph(f, freudenthal_grid_graph((n, n))).simplify(tau)


def full_range(run: LandscapeRun, res: dict) -> tuple:
    st = res["settings"]
    s = run.grid(st["n_full"])[0] if res["full"]["kind"].startswith("grid") else run.scattered("sobol", st["n_full"], 0)
    g = -np.log(s["loss"])
    return float(g.min()), float(g.max())


def panel(ax, u, loss, n, eg, title, models_uv=None, placement=None):
    U = u[:, 0].reshape(n, n)
    V = u[:, 1].reshape(n, n)
    Z = np.log10(loss).reshape(n, n)
    im = ax.pcolormesh(U, V, Z, cmap="viridis_r", shading="auto", rasterized=True)
    ax.contour(U, V, Z, levels=12, colors="white", linewidths=0.4, alpha=0.5)
    pk, sd = u[eg.peaks], u[eg.saddles]
    ax.scatter(sd[:, 0], sd[:, 1], marker="D", s=22, c="black", edgecolors="white", linewidths=0.6, zorder=4)
    for i, p in enumerate(pk):
        true = placement is None or placement[i]["true"]
        if true:
            ax.scatter(*p, marker="o", s=70, facecolors="white", edgecolors="black", linewidths=1.2, zorder=5)
        else:
            ax.scatter(*p, marker="X", s=90, c="#d62728", edgecolors="white", linewidths=0.8, zorder=5)
    if models_uv is not None:
        ax.scatter(models_uv[:, 0], models_uv[:, 1], marker="*", s=150, c="#ffd200", edgecolors="black",
                   linewidths=0.8, zorder=6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["cnn_m4_e30", "cnn_m4_e30_spawn2"])
    ap.add_argument("--max-planes", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ncol = args.max_planes + 1
    fig, axes = plt.subplots(len(args.runs), ncol, figsize=(3.3 * ncol, 3.6 * len(args.runs)), squeeze=False,
                             layout="constrained")
    for row, name in enumerate(args.runs):
        res = json.loads((RESULTS / f"loss_landscape_slices_{name}.json").read_text())
        run = LandscapeRun.load(name)
        st, lo_hi = res["settings"], full_range(run, res)
        n, tau = st["n_slice"], st["tau"]
        rows = {r["slice"]: r for r in res["slices"]}
        fake = sum(1 for e in res.get("extra_minima", []) if not e["stays_in_own_basin"])
        planes = triple_planes(run.subspace.anchors)[:args.max_planes]
        for col, plane in enumerate(planes):
            s = run.plane(plane, n)
            r = rows[plane.name]
            eg = slice_graph(s["loss"], n, *lo_hi, tau)
            uv = (run.subspace.anchors[list(plane.models)] - plane.origin) @ plane.basis.T
            panel(axes[row, col], s["u"], s["loss"], n, eg,
                  f"models {', '.join(map(str, plane.models))}: {r['true']} of {res['full']['minima'] - fake} minima"
                  + (f", {r['displaced']} displaced" if r["displaced"] else ""), uv, r["placement"])
        s = run.li_plane(0, n)
        eg = slice_graph(s["loss"], n, *lo_hi, tau)
        panel(axes[row, ncol - 1], s["u"], s["loss"], n, eg, "Li et al. random plane, model 0",
              np.zeros((1, 2)))
        for col in range(len(planes), ncol - 1):
            axes[row, col].axis("off")
        mins = f"{res['full']['minima'] - fake} minima" + (f" (+{fake} sampling artefact, removed by descent)" if fake else "")
        title = axes[row, 0].annotate(f"{name}: full {res['dim']}-D subspace has {mins}, "
                              f"{res['full']['saddles']} saddles ({res['full']['loops']} loops) in the graph",
                              xy=(0, 1.14), xycoords="axes fraction", fontsize=11, fontweight="bold")
        title.set_in_layout(False)
    handles = [plt.Line2D([], [], marker="*", ls="", ms=12, mfc="#ffd200", mec="black", label="trained model"),
               plt.Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec="black", label="slice minimum = true minimum"),
               plt.Line2D([], [], marker="X", ls="", ms=9, mfc="#d62728", mec="white", label="slice minimum, displaced"),
               plt.Line2D([], [], marker="D", ls="", ms=6, mfc="black", mec="white", label="slice saddle")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("2-D slices of the loss landscape (colour: log10 training loss, own scale per panel)",
                 fontsize=12, x=0.01, ha="left")
    fig.get_layout_engine().set(h_pad=0.3, hspace=0.12)
    out = args.out or RESULTS / ("loss_landscape_slices_" + "_".join(args.runs) + ".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
