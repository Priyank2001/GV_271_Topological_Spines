#!/usr/bin/env python3
"""
add_nki.py  (v3)

Adds the NKI breast cancer cohort (van't Veer et al. 2002, van de Vijver et al. 2002; the
cohort used in Meenaly Yadav's thesis and in Nicolau et al. 2011) to the GV_271
Topological Spines viewer as the point cloud "nki". The data comes from the Bioconductor
package breastCancerNKI (data.world, cited in the thesis, has shut down). No R is needed:
the R data file is read with the Python package ``rdata`` (python3 -m pip install rdata).

Run from the repo root, after add_bc_tcga.py:

    cd ~/Downloads/GV_271_Topological_Spines
    python3 ~/Downloads/add_nki.py [path to breastCancerNKI_*.tar.gz]
    python3 web/build_data.py nki

The archive defaults to the newest ~/Downloads/breastCancerNKI_*.tar.gz.

The point cloud, built exactly like bc_tcga
  * points: the 337 tumour samples; probes missing in over 10 % of samples are dropped,
    the remaining gaps (about 0.7 % of values) are filled with the probe's mean.
  * dimensions: the 1000 most variable probes, standardised, named by gene symbol, plus the
    markers ESR1, ERBB2, MKI67, FOXA1 and GRB7 if not already among them.
  * f: density, -log mean distance to the 15 nearest samples in the first 10 principal
    components of the 1000 probes, scaled to 0..1.
  * known labels (Clustering -> Known labels; never used to build f or the graph):
    ER status, histological grade, distant metastasis (event), death (event); "unknown"
    where the cohort has no value.

Also lets a dataset carry several label sets (pipeline/export_json.py, web/build_data.py;
backups *.before_nki). Re-running is safe. Reading the R file takes a few minutes.
"""
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "web" / "build_data.py").exists():
    sys.exit("run this from the GV_271_Topological_Spines folder (the one that has web/build_data.py)")
if "def classes_name" not in (ROOT / "web" / "build_data.py").read_text():
    sys.exit("run add_bc_tcga.py first")
# Only rdata's parser is needed (numpy only). Load it without the rest of the package, whose
# other parts pull in xarray / pandas and can fail with some versions of those.
import importlib.util
import types
_spec = importlib.util.find_spec("rdata")
if _spec is None:
    sys.exit("this needs the Python package rdata:  python3 -m pip install rdata")
_pkg = types.ModuleType("rdata")
_pkg.__path__ = list(_spec.submodule_search_locations)
sys.modules["rdata"] = _pkg
for _mod in ("lzma", "bz2"):           # some Python builds lack these; nki.rda is gzip, so a
    try:                               # placeholder is enough for the parser to load
        __import__(_mod)
    except ImportError:
        sys.modules[_mod] = types.ModuleType(_mod)
try:
    from rdata.parser import RObjectType as T, parse_data
except Exception as err:          # show the real reason
    sys.exit(f"could not load rdata's parser: {type(err).__name__}: {err}")

if len(sys.argv) > 1:
    ARCHIVE = Path(sys.argv[1]).expanduser()
else:
    found = sorted((Path.home() / "Downloads").glob("breastCancerNKI_*.tar.gz"))
    if not found:
        sys.exit("no ~/Downloads/breastCancerNKI_*.tar.gz; pass the archive's path")
    ARCHIVE = found[-1]
print(f"reading {ARCHIVE}")

N_GENES, MARKERS, PCS, K_DENSITY = 1000, ["ESR1", "ERBB2", "MKI67", "FOXA1", "GRB7"], 10, 15
NA_INT = -2147483648                       # R's NA for integers and logicals


# ---- 1. read nki.rda (an ExpressionSet) without R ---------------------------------
with tarfile.open(ARCHIVE) as tar:
    member = next(m for m in tar.getmembers() if m.name.endswith("data/nki.rda"))
    raw = tar.extractfile(member).read()
parsed = parse_data(raw)


def deref(o):
    while o is not None and o.info.type == T.REF:
        o = o.referenced_object
    return o


def tagname(t):
    t = deref(t)
    if t is None:
        return None
    v = t.value if isinstance(t.value, (bytes, str)) else deref(t.value)
    if hasattr(v, "value"):
        v = v.value
    return v.decode() if isinstance(v, bytes) else v


def pairlist(o):
    out = {}
    o = deref(o)
    while o is not None and o.info.type == T.LIST:
        out[tagname(o.tag)] = deref(o.value[0])
        o = deref(o.value[1]) if len(o.value) > 1 else None
    return out


def attrs(o):
    return pairlist(o.attributes) if o.attributes is not None else {}


def strvec(o):
    return [deref(x).value.decode() if deref(x) is not None and deref(x).value is not None else None
            for x in o.value]


def dataframe(o):
    cols = attrs(o)
    out = {}
    for name, c in zip(strvec(cols["names"]), o.value):
        c = deref(c)
        if c.info.type == T.STR:
            out[name] = strvec(c)
        else:
            v = np.array(c.value, dtype=float)
            v[v == NA_INT] = np.nan
            lev = attrs(c).get("levels")
            out[name] = ([strvec(lev)[int(i) - 1] if i == i and i > 0 else None for i in v]
                         if lev is not None else v)
    rn = cols.get("row.names")
    return out, (strvec(rn) if rn is not None and rn.info.type == T.STR else None)


eset = parsed.object.value[0]
A = attrs(eset)
pheno, samples = dataframe(attrs(A["phenoData"])["data"])
feat, _ = dataframe(attrs(A["featureData"])["data"])
env = A["assayData"].value
buckets = [deref(env.frame)] + (list(deref(env.hash_table).value) if deref(env.hash_table) is not None
                                and deref(env.hash_table).value is not None else [])
assay = {}
for b in buckets:
    b = deref(b)
    while b is not None and b.info.type == T.LIST:
        assay[tagname(b.tag)] = deref(b.value[0])
        b = deref(b.value[1]) if len(b.value) > 1 else None
ex = assay["exprs"]
nrow, ncol = (int(x) for x in attrs(ex)["dim"].value)
E = np.array(ex.value, dtype=float).reshape(ncol, nrow)          # R is column-major: row = sample
dimnames = attrs(ex)["dimnames"]
probes = strvec(deref(dimnames.value[0]))
symbol_of = dict(zip(feat["probe"], feat["HUGO.gene.symbol"]))
print(f"{ncol} samples, {nrow} probes")

# ---- 2. dimensions: the most variable probes, standardised ---------------------------
keep = np.isnan(E).mean(axis=0) <= 0.10          # drop probes missing in over 10 % of samples
E, probes = E[:, keep], [p for p, k in zip(probes, keep) if k]
E = np.where(np.isnan(E), np.nanmean(E, axis=0), E)   # fill the few gaps (~0.7 %) with the probe's mean
symbols = [symbol_of.get(p) or p for p in probes]
var = E.var(axis=0)
top = list(np.argsort(-var, kind="stable")[:N_GENES])
for m in MARKERS:                          # the most variable probe of each marker gene
    cand = [i for i, s in enumerate(symbols) if s == m]
    if cand and not any(symbols[i] == m for i in top):
        top.append(max(cand, key=lambda i: var[i]))
Z = E[:, top]
Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
names, seen = [], {}
for i in top:                              # unique names: gene symbol, then the probe if repeated
    s = symbols[i]
    names.append(s if s not in seen else f"{s} ({probes[i]})")
    seen[s] = True

# ---- 3. f: density in the first principal components of the top probes -------------
U, S, _ = np.linalg.svd(Z[:, :N_GENES], full_matrices=False)
P = U[:, :PCS] * S[:PCS]
D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2))
np.fill_diagonal(D, np.inf)
f = -np.log(np.sort(D, axis=1)[:, :K_DENSITY].mean(axis=1))
f = (f - f.min()) / (f.max() - f.min())
explained = float((S[:PCS] ** 2).sum() / (S ** 2).sum())


# ---- 4. known labels -------------------------------------------------------------------
def coded(col, mapping):
    v = np.asarray(pheno[col], dtype=float)
    return [mapping.get(int(x), -1) if x == x else -1 for x in v]


classes = [
    {"name": "ER status", "names": ["ER-", "ER+"], "values": coded("er", {0: 0, 1: 1})},
    {"name": "grade", "names": ["grade 1", "grade 2", "grade 3"], "values": coded("grade", {1: 0, 2: 1, 3: 2})},
    {"name": "distant metastasis", "names": ["no metastasis", "metastasis"], "values": coded("e.dmfs", {0: 0, 1: 1})},
    {"name": "death", "names": ["alive", "died"], "values": coded("e.os", {0: 0, 1: 1})},
]
clinical = {k: [None if (isinstance(x, float) and x != x) else x for x in
                (v.tolist() if isinstance(v, np.ndarray) else v)]
            for k, v in pheno.items() if k in ("age", "er", "grade", "e.dmfs", "t.dmfs", "e.os", "t.os", "node", "treatment")}

# ---- 5. dataset folder ---------------------------------------------------------------------
out = ROOT / "generator" / "datasets" / "nki"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", Z.astype(np.float32))
np.save(out / "values.npy", f.astype(np.float32))
truth = {
    "name": "nki",
    "dimensions": int(Z.shape[1]),
    "num_points": int(Z.shape[0]),
    "description": (f"NKI breast cancer cohort (Bioconductor breastCancerNKI): {Z.shape[0]} tumours. "
                    f"Dimensions: the {N_GENES} most variable probes (standardised) plus "
                    f"{Z.shape[1] - N_GENES} markers. f: density (-log mean distance to {K_DENSITY} nearest "
                    f"samples) in the first {PCS} principal components ({100 * explained:.0f} % of the "
                    "variance), scaled to 0..1."),
    "source": str(ARCHIVE),
    "dimension_names": names,
    "sample_ids": samples,
    "clinical": clinical,
    "point_classes": classes,
    "extrema": {"maxima": []},
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {Z.shape[0]} points, {Z.shape[1]} dimensions "
      f"(first {PCS} components explain {100 * explained:.0f} % of the variance)")


# ---- 6. several label sets per dataset (export_json.py, build_data.py) ---------------
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
    bk = p.with_name(p.name + ".before_nki")
    if not bk.exists():
        bk.write_text(orig)
    p.write_text(s)
    print(f"{rel}: updated")


patch("pipeline/export_json.py", json.loads(r'''[["    ``classes``: optional known labels of the points, ``{\"name\": str, \"names\": [str, ...],\n    \"values\": [int per point]}`` (e.g. tumour / normal), offered by the viewer's Clustering\n    section as \"known labels\" to set beside the peak regions.\n    \"\"\"\n", "    ``classes``: optional known labels of the points, ``{\"name\": str, \"names\": [str, ...],\n    \"values\": [int per point, -1 unknown]}`` (e.g. tumour / normal), or a list of such sets\n    (e.g. ER status, grade, metastasis), offered by the viewer's Clustering section as\n    \"known labels\" to set beside the peak regions.\n    \"\"\"\n"], ["    if classes:\n        values = [int(v) for v in classes[\"values\"]]\n        if len(values) != n:\n            raise ValueError(f\"{len(values)} class labels for {n} points\")\n        doc[\"classes\"] = {\"name\": str(classes.get(\"name\", \"class\")),\n                          \"names\": [str(x) for x in classes[\"names\"]], \"values\": values}\n    return doc\n", "    if classes:\n        sets = []\n        for c in (classes if isinstance(classes, (list, tuple)) else [classes]):\n            values = [int(v) for v in c[\"values\"]]          # -1: unknown\n            if len(values) != n:\n                raise ValueError(f\"{len(values)} class labels for {n} points\")\n            sets.append({\"name\": str(c.get(\"name\", \"class\")),\n                         \"names\": [str(x) for x in c[\"names\"]], \"values\": values})\n        doc[\"classes\"] = sets[0] if len(sets) == 1 else sets\n    return doc\n"]]'''), "sets = []")
patch("web/build_data.py", json.loads(r'''[["    c = load_dataset(spec[\"dataset\"]).truth.get(\"point_classes\")\n    return str(c.get(\"name\", \"class\")) if c else None\n\n", "    c = load_dataset(spec[\"dataset\"]).truth.get(\"point_classes\")\n    if not c:\n        return None\n    sets = c if isinstance(c, list) else [c]\n    return \", \".join(str(s.get(\"name\", \"class\")) for s in sets)\n\n"]]'''), "sets = c if isinstance(c, list)")

# ---- 7. register the viewer entry ----------------------------------------------------
bd = ROOT / "web" / "build_data.py"
src = bd.read_text()
if '"nki"' in src:
    print("web/build_data.py already has nki")
else:
    entry = (
        '    "nki": dict(\n'
        f'        label="nki (breast cancer genes, NKI cohort, {Z.shape[1]}D)", dataset="nki", k=15,\n'
        '        initial_threshold=0.05, embed=EMBEDS,\n'
        '        text="NKI breast cancer cohort (as in the thesis): 337 tumours, 1000 most variable genes. '
        'f = density of samples; Clustering -> Known labels: ER status, grade, metastasis, death."),\n'
    )
    new, count = re.subn(r"(ENTRIES\s*=\s*\{\n(?:#[^\n]*\n)*)", lambda m: m.group(1) + entry, src, count=1)
    if count != 1:
        sys.exit("could not find 'ENTRIES = {' in web/build_data.py; add the entry by hand")
    bd.write_text(new)
    print("added nki to web/build_data.py")

print("\nnext:  python3 web/build_data.py nki")
