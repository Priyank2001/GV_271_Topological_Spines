#!/usr/bin/env python3
"""
mapper_compare.py  (v2)

Mapper (KeplerMapper, a standard implementation) against the extremum graph on the same
data and the same function: the bc_dsga dataset (add_bc_dsga.py), lens / f = deviation
from normal tissue.

    cd ~/Downloads/GV_271_Topological_Spines
    python3 -m pip install kmapper scikit-learn networkx
    python3 ~/Downloads/mapper_compare.py

Writes comparison/results/mapper_vs_spine.md (tables), comparison/results/mapper_bc_dsga.png and
comparison/results/mapper_bc_dsga.html (one Mapper graph, nodes coloured by the spine peak
most of their tumours belong to; the HTML needs internet for its d3 library).

What is compared
  * Spine: our extremum graph on the kNN graph (k = 15), peaks kept above 4 % of the range of f.
  * Mapper: lens = the deviation (raw, and smoothed as in the spine), cover of n intervals
    with overlap o, DBSCAN clustering in the same 1000 dimensions (eps = a quantile q of the
    distance to the 5th neighbour, min_samples = 3); 2 lenses x 3 n x 2 o x 2 q = 24 settings.
  * A Mapper "branch tip" is a node whose mean lens value is higher than all its neighbours'
    (the end of a flare). Each tip is matched to the spine peak region holding most of its
    tumours; "purity" is that share.
  * Spine stability: the same peaks for k in {10, 15, 20, 30} and 2 or 3 rounds of
    smoothing; for each reference group, the best overlap (Jaccard) with any group there.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
DS = ROOT / "generator" / "datasets" / "bc_dsga"
if not (DS / "points.npy").exists():
    sys.exit("run add_bc_dsga.py first (and run this from the GV_271_Topological_Spines folder)")
try:
    import kmapper as km
    import networkx as nx
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
except ImportError as err:
    sys.exit(f"missing package ({err.name}):  python3 -m pip install kmapper scikit-learn networkx")
sys.path.insert(0, str(ROOT))
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402

P = np.load(DS / "points.npy").astype(np.float64)
raw = np.load(DS / "raw_deviation.npy").astype(np.float64)
import json  # noqa: E402
names = json.loads((DS / "truth.json").read_text())["dimension_names"]
CUT = 0.04


def smoothed(k, rounds):
    g = knn_graph(P.astype(np.float32), k)
    rows = np.repeat(np.arange(g.num_nodes), g.degree)
    deg = np.maximum(g.degree, 1)
    v = raw.copy()
    for _ in range(rounds):
        v = 0.5 * v + 0.5 * np.bincount(rows, weights=v[g.indices], minlength=g.num_nodes) / deg
    return g, (v - v.min()) / (v.max() - v.min())


def spine(k, rounds):
    g, v = smoothed(k, rounds)
    return extremum_graph(v, g).simplify(CUT), v


ref, f_ref = spine(15, 2)
reg = ref.region
lines = ["# Mapper vs extremum graph on bc_dsga (lens / f = deviation from normal)", ""]
lines += [f"Spine (k = 15, smoothed twice, peaks above {CUT:.0%} of the range): {ref.num_peaks} peaks.", "",
          "| Spine peak | Tumours | Persistence (share of range) | Genes most raised | Genes most lowered |",
          "|---|---|---|---|---|"]
for p in range(ref.num_peaks):
    m = reg == p
    d = (P[m].mean(0) - P[~m].mean(0)) / (P.std(0) + 1e-9)
    up = ", ".join(names[i] for i in np.argsort(-d)[:5])
    dn = ", ".join(names[i] for i in np.argsort(d)[:3])
    pers = ref.peak_persistence[p]
    lines.append(f"| {p} | {int(m.sum())} | {'global maximum' if not np.isfinite(pers) else f'{pers:.3f}'} | {up} | {dn} |")

# ---- spine stability ------------------------------------------------------------------
best = {p: [] for p in range(ref.num_peaks)}
for k, rounds in itertools.product([10, 15, 20, 30], [2, 3]):
    s, _ = spine(k, rounds)
    for p in best:
        A = reg == p
        best[p].append(max((A & (s.region == q)).sum() / (A | (s.region == q)).sum() for q in range(s.num_peaks)))
lines += ["", "## Spine stability (k in 10, 15, 20, 30; smoothing 2 or 3 rounds: 8 settings)", "",
          "| Spine peak | Median best overlap (Jaccard) | Lowest | Settings with overlap >= 0.5 |", "|---|---|---|---|"]
for p, b in best.items():
    b = np.array(b)
    lines.append(f"| {p} | {np.median(b):.2f} | {b.min():.2f} | {(b >= 0.5).sum()}/8 |")

# ---- Mapper grid ------------------------------------------------------------------------
kd = np.sort(NearestNeighbors(n_neighbors=6).fit(P).kneighbors(P)[0][:, 5])
mapper = km.KeplerMapper(verbose=0)
hits = {p: 0 for p in range(ref.num_peaks)}
rows_md = []
keep_graph = None
for lens_name, lens in [("raw deviation", raw), ("smoothed deviation", f_ref)]:
    for n, ov, q in itertools.product([10, 15, 20], [0.3, 0.5], [0.5, 0.7]):
        G = mapper.map(lens.reshape(-1, 1), P, cover=km.Cover(n_cubes=n, perc_overlap=ov),
                       clusterer=DBSCAN(eps=float(np.quantile(kd, q)), min_samples=3))
        g = nx.Graph()
        g.add_nodes_from(G["nodes"])
        for a, bs in G["links"].items():
            g.add_edges_from((a, b) for b in bs)
        mean = {k: lens[v].mean() for k, v in G["nodes"].items()}
        tips = [k for k in g.nodes if all(mean[k] > mean[j] for j in g.neighbors(k))]
        matched, purity = set(), []
        for t in tips:
            r_, c_ = np.unique(reg[np.array(G["nodes"][t])], return_counts=True)
            matched.add(int(r_[np.argmax(c_)]))
            purity.append(c_.max() / c_.sum())
        for p in matched:
            hits[p] += 1
        rows_md.append(f"| {lens_name} | {n} | {ov} | q{q} | {g.number_of_nodes()} | "
                       f"{nx.number_connected_components(g)} | {len(tips)} | {sorted(matched)} | "
                       f"{np.mean(purity) if purity else 0:.2f} |")
        if lens_name == "smoothed deviation" and (n, ov, q) == (15, 0.5, 0.5):
            keep_graph = G
lines += ["", "## Mapper (KeplerMapper) over 24 settings", "",
          "| Lens | Intervals | Overlap | eps | Nodes | Pieces | Branch tips | Spine peaks the tips fall in | Tip purity |",
          "|---|---|---|---|---|---|---|---|---|"] + rows_md
lines += ["", "## Each spine peak: how many of the 24 Mapper settings have a branch tip in it", "",
          "| Spine peak | Tumours | Settings |", "|---|---|---|"]
lines += [f"| {p} | {int((reg == p).sum())} | {hits[p]}/24 |" for p in range(ref.num_peaks)]

out = ROOT / "comparison" / "results"
out.mkdir(parents=True, exist_ok=True)
(out / "mapper_vs_spine.md").write_text("\n".join(lines) + "\n")
if keep_graph is not None:
    # KeplerMapper colours a node by the MEAN of the colour values of its points, so peak ids
    # cannot be used directly (the mean of 1 and 3 is not peak 2). One column per spine peak
    # (1 if the tumour is in it) gives, per node, the share of its tumours in that peak; the
    # page has a menu to choose the peak.
    onehot = np.eye(ref.num_peaks)[reg]
    mapper.visualize(keep_graph, path_html=str(out / "mapper_bc_dsga.html"),
                     title="Mapper on bc_dsga (lens: smoothed deviation; 15 intervals, overlap 0.5); "
                           "colour: share of the node's tumours in the chosen spine peak",
                     color_values=onehot,
                     color_function_name=[f"share in spine peak {p_}" for p_ in range(ref.num_peaks)])
try:                                       # a static picture as well (the HTML needs internet for d3)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = nx.Graph()
    g.add_nodes_from(keep_graph["nodes"])
    for a, bs in keep_graph["links"].items():
        g.add_edges_from((a, b) for b in bs)
    major = {k: np.bincount(reg[np.array(v)], minlength=ref.num_peaks).argmax() for k, v in keep_graph["nodes"].items()}
    size = {k: len(v) for k, v in keep_graph["nodes"].items()}
    pos = nx.spring_layout(g, seed=1, k=0.35)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#999", width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[cmap(major[k] % 10) for k in g.nodes],
                           node_size=[20 + 6 * size[k] for k in g.nodes], edgecolors="white", linewidths=0.5)
    for p_ in range(ref.num_peaks):
        ax.scatter([], [], color=cmap(p_ % 10), label=f"spine peak {p_} ({int((reg == p_).sum())})")
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.set_title("Mapper on bc_dsga (lens: smoothed deviation, 15 intervals, overlap 0.5)\n"
                 "node colour: spine peak most of its tumours belong to; size: tumours", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "mapper_bc_dsga.png")
except Exception as err:                  # the tables are the result; the picture is extra
    print(f"(no PNG: {err})")
print("\n".join(lines))
print(f"\nwrote {out / 'mapper_vs_spine.md'}, {out / 'mapper_bc_dsga.html'} and {out / 'mapper_bc_dsga.png'}")
