#!/usr/bin/env python3
"""
add_bc_dsga.py  (v1)

Adds "bc_dsga" to the GV_271 Topological Spines viewer: the 529 BC-TCGA tumours, with
f = how far each tumour deviates from normal breast tissue (the disease component of
Nicolau et al.'s DSGA, as used as the Mapper lens in Nicolau, Levine and Carlsson 2011).
Needs add_bc_tcga.py applied first (for the BC-TCGA files and the export changes).

    cd ~/Downloads/GV_271_Topological_Spines
    python3 ~/Downloads/add_bc_dsga.py [folder with BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt]
    python3 web/build_data.py bc_dsga

How it is built
  1. Healthy-state model from the 61 normal samples: each normal is "flattened" (replaced
     by its least-squares fit from the other normals, which removes sample-specific noise);
     the model is the span of the flattened normals' principal components covering 90 % of
     their variance.
  2. Disease component of a tumour: the tumour minus its projection on that model.
  3. points: the 529 tumours; dimensions: the 1000 genes whose disease component varies most
     (log ratios, not rescaled, so the distances keep their units).
  4. f: the size of the disease component (Euclidean norm over all genes), averaged with its
     15 nearest tumours twice (half own value, half the neighbours' mean). The raw deviation
     is rough on the neighbour graph (60 peaks, no persistence gap); two rounds of averaging
     give a clear gap (7 peaks above 4 % of the range).
Simplifications against the original DSGA: one fixed variance cut (90 %) for the model,
and the plain Euclidean norm as the size of the deviation.
"""
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

SRC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else \
    Path.home() / "Downloads" / "v3cc2p38hb-1" / "BC-TCGA" / "BC-TCGA"
N_GENES, MODEL_VARIANCE, K, SMOOTH = 1000, 0.90, 15, 2


def read_table(path):
    with open(path) as fh:
        samples = fh.readline().rstrip("\n").split("\t")[1:]
        genes, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            genes.append(parts[0])
            rows.append([float(x) if x not in ("", "NA", "NaN", "null") else np.nan for x in parts[1:]])
    return genes, samples, np.array(rows, dtype=np.float64)


g_t, s_t, T = read_table(SRC / "BC-TCGA-Tumor.txt")
g_n, s_n, N = read_table(SRC / "BC-TCGA-Normal.txt")
if g_t != g_n:
    sys.exit("the tumour and normal files list different genes")
keep = ~(np.isnan(T).any(axis=1) | np.isnan(N).any(axis=1))
genes = np.array(g_t)[keep]
T, N = T[keep], N[keep]                                   # genes x samples
print(f"{T.shape[1]} tumours, {N.shape[1]} normals, {len(genes)} genes")

# ---- 1. healthy-state model --------------------------------------------------------
Nf = np.empty_like(N)
for i in range(N.shape[1]):
    others = np.delete(N, i, axis=1)
    coef, *_ = np.linalg.lstsq(others, N[:, i], rcond=None)
    Nf[:, i] = others @ coef
U, S, _ = np.linalg.svd(Nf, full_matrices=False)
r = int(np.searchsorted(np.cumsum(S ** 2) / (S ** 2).sum(), MODEL_VARIANCE) + 1)
B = U[:, :r]

# ---- 2. disease component -----------------------------------------------------------
Dc = (T - B @ (B.T @ T)).T                                # tumours x genes
dev = np.linalg.norm(Dc, axis=1)
held = []
for i in range(N.shape[1]):                               # the same measure on held-out normals
    Ui, _, _ = np.linalg.svd(np.delete(Nf, i, axis=1), full_matrices=False)
    held.append(np.linalg.norm(N[:, i] - Ui[:, :r] @ (Ui[:, :r].T @ N[:, i])))
print(f"healthy model: {r} components; deviation median {np.median(dev):.1f} for tumours, "
      f"{np.median(held):.1f} for held-out normals")

# ---- 3. points and 4. f ---------------------------------------------------------------
top = np.argsort(-Dc.var(axis=0), kind="stable")[:N_GENES]
P = Dc[:, top].astype(np.float32)
g = knn_graph(P, K)
rows = np.repeat(np.arange(g.num_nodes), g.degree)
deg = np.maximum(g.degree, 1)
f = dev.copy()
for _ in range(SMOOTH):
    f = 0.5 * f + 0.5 * np.bincount(rows, weights=f[g.indices], minlength=g.num_nodes) / deg
f = (f - f.min()) / (f.max() - f.min())                 # 0..1, like the other datasets (thresholds are in f's units)

out = ROOT / "generator" / "datasets" / "bc_dsga"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", P)
np.save(out / "values.npy", f.astype(np.float32))
np.save(out / "raw_deviation.npy", dev.astype(np.float32))
truth = {
    "name": "bc_dsga", "dimensions": int(P.shape[1]), "num_points": int(P.shape[0]),
    "description": (f"BC-TCGA tumours, deviation from normal tissue (DSGA disease component; healthy model: "
                    f"{r} components of the flattened normals). Dimensions: the {N_GENES} genes whose disease "
                    f"component varies most. f: size of the disease component, averaged with the {K} nearest "
                    f"tumours {SMOOTH} times, scaled to 0..1."),
    "source": str(SRC), "dimension_names": genes[top].tolist(), "sample_ids": list(s_t),
    "extrema": {"maxima": []},
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {P.shape[0]} points, {P.shape[1]} dimensions")

# ---- register the viewer entry ----------------------------------------------------------
bd = ROOT / "web" / "build_data.py"
src = bd.read_text()
if '"bc_dsga"' in src:
    print("web/build_data.py already has bc_dsga")
else:
    entry = (
        '    "bc_dsga": dict(\n'
        f'        label="bc_dsga (tumours, deviation from normal, {P.shape[1]}D)", dataset="bc_dsga", k={K},\n'
        '        initial_threshold=0.04, embed=EMBEDS,\n'
        '        text="BC-TCGA tumours. f = how far each tumour deviates from normal breast tissue (DSGA disease '
        'component, smoothed over neighbours, scaled to 0..1). Peaks: the most deviant tumour types."),\n'
    )
    new, count = re.subn(r"(ENTRIES\s*=\s*\{\n(?:#[^\n]*\n)*)", lambda m: m.group(1) + entry, src, count=1)
    if count != 1:
        sys.exit("could not find 'ENTRIES = {' in web/build_data.py; add the entry by hand")
    bd.write_text(new)
    print("added bc_dsga to web/build_data.py")
print("\nnext:  python3 web/build_data.py bc_dsga")
