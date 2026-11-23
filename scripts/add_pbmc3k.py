#!/usr/bin/env python3
"""
add_pbmc3k.py  (v2)

Adds "pbmc3k_types" to the GV_271 Topological Spines viewer: 2,638 blood cells (10x Genomics PBMC
3k, the standard single-cell example), one point per cell, with reference cell types as known
labels, and writes a short result file comparison/results/pbmc3k_types.md. (The "pbmc3k" entry
of generator/make_pbmc3k_dataset.py is a separate, unlabelled version: 50 PCs, no cell types.)
Needs add_bc_tcga.py applied (for known labels in the viewer).

    cd ~/Downloads/GV_271_Topological_Spines
    python3 ~/Downloads/add_pbmc3k.py [path to pbmc3k_counts.npz]
    python3 web/build_data.py pbmc3k_types

pbmc3k_counts.npz (default: data/pbmc3k_types/pbmc3k_counts.npz in the repo, else
~/Downloads/pbmc3k_counts.npz): the raw counts of 10x Genomics'
PBMC 3k data set (2,700 cells x 32,738 genes, as in scanpy's pbmc3k), plus each cell's Leiden
cluster from a published processed copy (github.com/BrainStOrmics/BIA-CellHashtag). The
clusters are named by the standard marker genes (as in the scanpy / Seurat tutorials) and are
used only as known labels, never to build f or the graph.

How the point cloud is made (the standard single-cell recipe)
  * cells with 200 to 2,499 detected genes and under 5 % mitochondrial reads; genes seen in at
    least 3 cells; counts scaled to 10,000 per cell, log(1 + x);
  * the 2,000 most variable genes (normalised dispersion), standardised, clipped at 10;
  * points: the first 20 principal components (named by their strongest genes), where the
    kNN graph (k = 15) is built, as in every single-cell neighbourhood graph;
  * f: density, -log of the mean distance to the 15 nearest cells in the first 10 principal
    components, scaled to 0..1, not smoothed. Peaks: the common cell states; saddles: how two
    cell types grade into each other; persistence: how distinct a cell type is.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "web" / "build_data.py").exists():
    sys.exit("run this from the GV_271_Topological_Spines folder (the one that has web/build_data.py)")
if "def classes_name" not in (ROOT / "web" / "build_data.py").read_text():
    sys.exit("run add_bc_tcga.py first")
sys.path.insert(0, str(ROOT))
from pipeline.neighbor_graph import knn_graph  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402

_IN_REPO = ROOT / "data" / "pbmc3k_types" / "pbmc3k_counts.npz"
SRC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else \
    (_IN_REPO if _IN_REPO.exists() else Path.home() / "Downloads" / "pbmc3k_counts.npz")
if not SRC.exists():
    sys.exit(f"not found: {SRC}\npass the path of pbmc3k_counts.npz")
N_HVG, N_PC, DENSITY_PC, K = 2000, 20, 10, 15
NAMES = {"0": "CD4 T", "1": "CD14 monocyte", "2": "B", "3": "CD8 T", "4": "FCGR3A monocyte",
         "5": "NK", "6": "Dendritic", "7": "Megakaryocyte"}
ORDER = ["CD4 T", "CD8 T", "NK", "B", "CD14 monocyte", "FCGR3A monocyte", "Dendritic", "Megakaryocyte"]

# ---- 1. counts, quality control, normalisation ------------------------------------------
z = np.load(SRC, allow_pickle=True)
n_cells, n_genes = (int(x) for x in z["shape"])
counts = np.zeros((n_cells, n_genes), dtype=np.float32)
for i in range(n_cells):
    a, b = z["indptr"][i], z["indptr"][i + 1]
    counts[i, z["indices"][a:b]] = z["data"][a:b]
genes, cells, leiden = np.array(z["genes"]), np.array(z["cells"]), np.array(z["leiden"])
detected = (counts > 0).sum(1)
mito = counts[:, np.char.startswith(genes.astype(str), "MT-")].sum(1) / counts.sum(1)
keep_c = (detected >= 200) & (detected < 2500) & (mito < 0.05)
keep_g = (counts[keep_c] > 0).sum(0) >= 3
X = counts[keep_c][:, keep_g].astype(np.float64)
genes, cells, leiden = genes[keep_g], cells[keep_c], leiden[keep_c]
X = np.log1p(X * (1e4 / X.sum(1, keepdims=True)))
print(f"{X.shape[0]} cells, {X.shape[1]} genes after quality control")

# ---- 2. variable genes, principal components -----------------------------------------------
mu, var = X.mean(0), X.var(0)
log_disp = np.log(np.maximum(np.where(mu > 0, var / np.maximum(mu, 1e-12), 0), 1e-12))
lm = np.log1p(mu)
bins = np.digitize(lm, np.linspace(lm.min(), lm.max(), 20))
norm_disp = np.zeros_like(log_disp)
for b in np.unique(bins):
    m = bins == b
    sd = log_disp[m].std()
    norm_disp[m] = (log_disp[m] - log_disp[m].mean()) / (sd if sd > 0 else 1)
hvg = np.argsort(-np.where((mu > 0.0125) & (mu < 3), norm_disp, -np.inf))[:N_HVG]
Zs = np.clip((X[:, hvg] - mu[hvg]) / np.sqrt(var[hvg]), -10, 10)
U, S, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
PC = U * S
P = PC[:, :N_PC].astype(np.float32)
pc_names = []
for i in range(N_PC):
    w = Vt[i]
    hi, lo = genes[hvg][np.argmax(w)], genes[hvg][np.argmin(w)]
    pc_names.append(f"PC{i + 1} ({hi}+ / {lo}-)")

# ---- 3. f: density ----------------------------------------------------------------------------
Q = PC[:, :DENSITY_PC]
sq = (Q ** 2).sum(1)
d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * Q @ Q.T, 0))
np.fill_diagonal(d, np.inf)
f = -np.log(np.sort(d, axis=1)[:, :K].mean(1))
f = (f - f.min()) / (f.max() - f.min())

cell_type = np.array([NAMES.get(c, "") for c in leiden])
names = [t for t in ORDER if t in set(cell_type)]
out = ROOT / "generator" / "datasets" / "pbmc3k_types"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", P)
np.save(out / "values.npy", f.astype(np.float64))
truth = {
    "name": "pbmc3k_types", "dimensions": N_PC, "num_points": int(P.shape[0]),
    "description": ("10x PBMC 3k blood cells after the standard single-cell recipe; points: the first 20 principal "
                    "components of the 2,000 most variable genes; f: density (-log mean distance to the 15 nearest "
                    "cells in the first 10 components), scaled to 0..1."),
    "source": str(SRC), "dimension_names": pc_names, "sample_ids": [str(c) for c in cells],
    "extrema": {"maxima": []},
    "point_classes": [{"name": "Cell type (reference)", "names": names,
                       "values": [names.index(t) if t in names else -1 for t in cell_type]}],
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {P.shape[0]} cells, {N_PC} dimensions")

# ---- 4. register the viewer entry -------------------------------------------------------------
bd = ROOT / "web" / "build_data.py"
src = bd.read_text()
if '"pbmc3k_types"' in src:
    print("web/build_data.py already has pbmc3k_types")
else:
    entry = (
        '    "pbmc3k_types": dict(\n'
        '        label="pbmc3k_types (2,638 blood cells with cell types, 20 PCs)", dataset="pbmc3k_types", k=15,\n'
        '        initial_threshold=0.04, embed=EMBEDS,\n'
        '        text="Blood cells (10x PBMC 3k). f = density of cells. Peaks: cell types; saddles: how two '
        'types grade into each other; persistence: how distinct a type is. Clustering -> Known labels: '
        'reference cell types."),\n'
    )
    new, count = re.subn(r"(ENTRIES\s*=\s*\{\n(?:#[^\n]*\n)*)", lambda m: m.group(1) + entry, src, count=1)
    if count != 1:
        sys.exit("could not find 'ENTRIES = {' in web/build_data.py; add the entry by hand")
    bd.write_text(new)
    print("added pbmc3k_types to web/build_data.py")

# ---- 5. what the spine says, with a stability check ------------------------------------------
TAU = 0.04
g = knn_graph(P, K)
spine = extremum_graph(f, g).simplify(TAU)


def name_of(mask):
    c = collections.Counter(cell_type[mask])
    (t, k), *rest = c.most_common()
    if rest and rest[0][1] >= 0.1 * mask.sum():           # a second type with at least 10 %
        t = f"{t} + {rest[0][0]}"
    return t, k / mask.sum()


def regions(pts, k, tau):
    eg = extremum_graph(f, knn_graph(pts, k)).simplify(tau)
    return [set(np.flatnonzero(eg.region == p)) for p in range(eg.num_peaks)]


grid = [(k, tau) for k in (10, 15, 20, 30) for tau in (0.02, 0.04, 0.08)]
others = [regions(P, k, tau) for k, tau in grid]
rows = []
for p in np.argsort(-spine.region_size):
    m = spine.region == p
    t, purity = name_of(m)
    mem = set(np.flatnonzero(m))
    found = np.mean([max(len(mem & r) / len(mem | r) for r in regs) >= 0.5 for regs in others])
    diff = X[m].mean(0) - X[~m].mean(0)
    up = ", ".join(genes[np.argsort(-diff)[:5]])
    pers = spine.peak_persistence[p]
    rows.append((t, int(m.sum()), purity, "highest" if not np.isfinite(pers) else f"{pers:.3f}", up, found))
L = ["# pbmc3k_types: what the spine shows", "",
     f"{P.shape[0]} blood cells; kNN graph (k = {K}) on 20 principal components; f = density; persistence "
     f"threshold {TAU}. Cell types are the reference labels (not used to build anything).", "",
     "| Peak = mostly | Cells | Largest type | Persistence | Top genes (higher than the rest) | Found in 12 settings |",
     "|---|---|---|---|---|---|"]
for t, n, pur, pers, up, found in rows:
    L.append(f"| {t} | {n} | {pur:.0%} | {pers} | {up} | {found:.0%} |")
L += ["", "Saddles (highest first: the two types that grade into each other most easily):", ""]
order = np.argsort(-spine.saddle_values)
for s in order:
    a, b = spine.saddle_peaks[s]
    kind = "merge" if spine.saddle_is_merge[s] else "loop"
    L.append(f"* {name_of(spine.region == a)[0]} -- {name_of(spine.region == b)[0]}: saddle at "
             f"{spine.saddle_values[s]:.2f} ({kind})")
L += ["", "Stability: settings k in (10, 15, 20, 30) x threshold in (0.02, 0.04, 0.08); a peak is found when "
      "some peak there shares at least half of its cells (Jaccard >= 0.5)."]
res = ROOT / "comparison" / "results"
res.mkdir(parents=True, exist_ok=True)
(res / "pbmc3k_types.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
print(f"\nwrote {res / 'pbmc3k_types.md'}\nnext:  python3 web/build_data.py pbmc3k_types")
