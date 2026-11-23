#!/usr/bin/env python3
"""
add_bc_tcga.py  (v1)

Adds the TCGA breast cancer gene expression data (BC-TCGA: 529 tumour and 61 normal
tissue samples, about 17,800 genes) to the GV_271 Topological Spines viewer as the point
cloud "bc_tcga". Run from the repo root:

    cd ~/Downloads/GV_271_Topological_Spines
    python3 ~/Downloads/add_bc_tcga.py [folder with BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt]
    python3 web/build_data.py bc_tcga

The folder defaults to ~/Downloads/v3cc2p38hb-1/BC-TCGA/BC-TCGA.

How the point cloud is made
  * points: one per sample (tissue); genes with missing values are dropped.
  * dimensions: the 1000 most variable genes, each standardised (mean 0, standard
    deviation 1), plus the well-known markers ERBB2, GRB7 and MKI67 if they are not
    already among them (so they can be looked up in the viewer). Neighbours are found
    in these dimensions.
  * f: density. -log of the mean distance from a sample to its 15 nearest samples,
    measured in the first 10 principal components of the 1000 genes, scaled to 0..1.
    Peaks of f are groups of samples with similar expression. Density measured directly
    on 1000 genes is nearly flat (in that many noisy dimensions all distances look
    alike), which is why it is measured in the principal components.
  * known labels: tumour / normal, shown by the viewer's Clustering section
    ("Known labels") but never used to build f or the graph.

Also teaches web/build_data.py and pipeline/export_json.py to pass gene names and known
labels to the viewer (once; re-running is safe; backups *.before_bc_tcga).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "web" / "build_data.py").exists():
    sys.exit("run this from the GV_271_Topological_Spines folder (the one that has web/build_data.py)")

SRC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else \
    Path.home() / "Downloads" / "v3cc2p38hb-1" / "BC-TCGA" / "BC-TCGA"
TUMOUR, NORMAL = SRC / "BC-TCGA-Tumor.txt", SRC / "BC-TCGA-Normal.txt"
for p in (TUMOUR, NORMAL):
    if not p.exists():
        sys.exit(f"not found: {p}\npass the folder that has BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt")

N_GENES, MARKERS, PCS, K_DENSITY = 1000, ["ERBB2", "GRB7", "MKI67"], 10, 15


# ---- 1. read the two tables (rows: genes, columns: samples) ----------------------
def read_table(path):
    with open(path) as fh:
        samples = fh.readline().rstrip("\n").split("\t")[1:]
        genes, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            genes.append(parts[0])
            rows.append([float(x) if x not in ("", "NA", "NaN", "null") else np.nan for x in parts[1:]])
    return genes, samples, np.array(rows, dtype=np.float64)


g_t, s_t, T = read_table(TUMOUR)
g_n, s_n, N = read_table(NORMAL)
if g_t != g_n:
    sys.exit("the tumour and normal files list different genes")
genes = np.array(g_t)
X = np.concatenate([T, N], axis=1).T                     # samples x genes
labels = np.array([0] * len(s_t) + [1] * len(s_n))       # 0 tumour, 1 normal
keep = ~np.isnan(X).any(axis=0)
X, genes = X[:, keep], genes[keep]
print(f"{X.shape[0]} samples ({len(s_t)} tumour, {len(s_n)} normal), {X.shape[1]} genes without missing values")

# ---- 2. dimensions: the most variable genes, standardised --------------------------
var = X.var(axis=0)
top = np.argsort(-var, kind="stable")[:N_GENES]
extra = [int(np.where(genes == m)[0][0]) for m in MARKERS if m in genes and np.where(genes == m)[0][0] not in top]
cols = np.concatenate([top, np.array(extra, dtype=int)])
Z = X[:, cols]
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
names = genes[cols].tolist()

# ---- 3. f: density in the first principal components of the top genes ------------
U, S, _ = np.linalg.svd(Z[:, :N_GENES], full_matrices=False)
P = U[:, :PCS] * S[:PCS]
D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(D, np.inf)
mean_nn = np.sort(D, axis=1)[:, :K_DENSITY].mean(axis=1)
f = -np.log(mean_nn)
f = (f - f.min()) / (f.max() - f.min())
explained = float((S[:PCS] ** 2).sum() / (S ** 2).sum())

# ---- 4. dataset folder -------------------------------------------------------------
out = ROOT / "generator" / "datasets" / "bc_tcga"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", Z.astype(np.float32))
np.save(out / "values.npy", f.astype(np.float32))
truth = {
    "name": "bc_tcga",
    "dimensions": int(Z.shape[1]),
    "num_points": int(Z.shape[0]),
    "description": (f"TCGA breast cancer gene expression: {len(s_t)} tumour and {len(s_n)} normal samples. "
                    f"Dimensions: the {N_GENES} most variable genes (standardised) plus {len(extra)} markers. "
                    f"f: density (-log mean distance to {K_DENSITY} nearest samples) in the first {PCS} principal "
                    f"components ({100 * explained:.0f} % of the variance), scaled to 0..1."),
    "source": str(SRC),
    "dimension_names": names,
    "sample_ids": list(s_t) + list(s_n),
    "point_classes": {"name": "sample type", "names": ["tumour", "normal"], "values": labels.tolist()},
    "extrema": {"maxima": []},
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {Z.shape[0]} points, {Z.shape[1]} dimensions "
      f"(first {PCS} components explain {100 * explained:.0f} % of the variance)")


# ---- 5. pass gene names and known labels to the viewer -----------------------------
def patch(rel, pairs, done_marker):
    p = ROOT / rel
    s = p.read_text()
    if done_marker in s:
        print(f"{rel}: already done")
        return
    orig = s
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{rel}: could not place an edit (file differs from what I expect). Send me the file.")
        s = s.replace(old, new, 1)
    p.with_name(p.name + ".before_bc_tcga").write_text(orig)
    p.write_text(s)
    print(f"{rel}: updated")


patch("pipeline/export_json.py", json.loads(r'''[["\ndef coords_document(points: np.ndarray, names: Optional[list] = None) -> dict:\n    \"\"\"Original coordinates of a point cloud for the viewer's parallel coordinates\n", "\ndef coords_document(points: np.ndarray, names: Optional[list] = None,\n                    classes: Optional[dict] = None) -> dict:\n    \"\"\"Original coordinates of a point cloud for the viewer's parallel coordinates\n"], ["    below a pixel on any axis.\n    \"\"\"\n", "    below a pixel on any axis.\n\n    ``classes``: optional known labels of the points, ``{\"name\": str, \"names\": [str, ...],\n    \"values\": [int per point]}`` (e.g. tumour / normal), offered by the viewer's Clustering\n    section as \"known labels\" to set beside the peak regions.\n    \"\"\"\n"], ["                     \"data\": base64.b64encode(q.tobytes()).decode(\"ascii\")})\n    return {\"version\": 1, \"num_points\": int(n), \"dimensions\": int(d),\n            \"encoding\": \"uint16-base64\", \"axes\": axes}\n\n", "                     \"data\": base64.b64encode(q.tobytes()).decode(\"ascii\")})\n    doc = {\"version\": 1, \"num_points\": int(n), \"dimensions\": int(d),\n           \"encoding\": \"uint16-base64\", \"axes\": axes}\n    if classes:\n        values = [int(v) for v in classes[\"values\"]]\n        if len(values) != n:\n            raise ValueError(f\"{len(values)} class labels for {n} points\")\n        doc[\"classes\"] = {\"name\": str(classes.get(\"name\", \"class\")),\n                          \"names\": [str(x) for x in classes[\"names\"]], \"values\": values}\n    return doc\n\n"]]'''), 'doc["classes"]')
patch("web/build_data.py", json.loads(r'''[["            raise SystemExit(f\"{name}: the dataset no longer matches {main_file.name}; rebuild it first\")\n    path = write_json(coords_document(ds.points), OUT / f\"{name}.coords.json\")\n    return path.name\n", "            raise SystemExit(f\"{name}: the dataset no longer matches {main_file.name}; rebuild it first\")\n    # optional, from truth.json: gene / feature names, and known labels of the points\n    path = write_json(coords_document(ds.points, names=ds.truth.get(\"dimension_names\"),\n                                      classes=ds.truth.get(\"point_classes\")),\n                      OUT / f\"{name}.coords.json\")\n    return path.name\n"], ["\n\ndef main() -> None:\n", "\n\ndef classes_name(spec: dict) -> str | None:\n    \"\"\"Name of the known labels of a point-cloud entry (truth.json ``point_classes``), or None.\"\"\"\n    if \"dataset\" not in spec:\n        return None\n    from pipeline.data_loader import load_dataset\n    c = load_dataset(spec[\"dataset\"]).truth.get(\"point_classes\")\n    return str(c.get(\"name\", \"class\")) if c else None\n\n\ndef main() -> None:\n"], ["                index[name][\"coords_file\"] = write_coords(name, spec)\n                print(f\"{name:11s} coords: {index[name]['coords_file'] or '(grid, none needed)'}\")\n", "                index[name][\"coords_file\"] = write_coords(name, spec)\n                index[name][\"classes\"] = classes_name(spec)\n                print(f\"{name:11s} coords: {index[name]['coords_file'] or '(grid, none needed)'}\")\n"], ["                       \"variants\": variants, \"comparisons\": comparisons,\n                       \"coords_file\": write_coords(name, spec)}\n        print(f\"{name:11s} {len(doc['peaks']):4d} peaks {len(doc['saddles']):5d} saddles \"\n", "                       \"variants\": variants, \"comparisons\": comparisons,\n                       \"coords_file\": write_coords(name, spec),\n                       \"classes\": classes_name(spec)}\n        print(f\"{name:11s} {len(doc['peaks']):4d} peaks {len(doc['saddles']):5d} saddles \"\n"]]'''), "def classes_name")

# ---- 6. register the viewer entry ----------------------------------------------------
bd = ROOT / "web" / "build_data.py"
src = bd.read_text()
if '"bc_tcga"' in src:
    print("web/build_data.py already has bc_tcga")
else:
    entry = (
        '    "bc_tcga": dict(\n'
        '        label="bc_tcga (breast cancer genes, 1003D)", dataset="bc_tcga", k=15,\n'
        '        simplify=0.01, initial_threshold=0.05, embed=EMBEDS,\n'
        '        text="TCGA breast cancer: 529 tumour and 61 normal samples, 1000 most variable genes. '
        'f = density of samples; Clustering -> Known labels shows tumour / normal."),\n'
    )
    new, count = re.subn(r"(ENTRIES\s*=\s*\{\n)", lambda m: m.group(1) + entry, src, count=1)
    if count != 1:
        sys.exit("could not find 'ENTRIES = {' in web/build_data.py; add the entry by hand")
    bd.write_text(new)
    print("added bc_tcga to web/build_data.py")

print("\nnext:  python3 web/build_data.py bc_tcga")
