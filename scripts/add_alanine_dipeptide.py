#!/usr/bin/env python3
"""
add_alanine_dipeptide.py  (v1)

Adds "alanine_dipeptide" to the GV_271 Topological Spines viewer: shapes of a small peptide
from a molecular dynamics simulation, one point per saved shape, and writes a short result
file comparison/results/alanine_dipeptide.md. Needs add_bc_tcga.py and add_bc_scalars.py
applied (known labels and the Scalar menu).

    cd ~/Downloads/GV_271_Topological_Spines
    python3 ~/Downloads/add_alanine_dipeptide.py [path to alanine_dipeptide_md.npz]
    python3 web/build_data.py alanine_dipeptide

alanine_dipeptide_md.npz (default: data/alanine_dipeptide/ in the repo, else ~/Downloads) comes from
alanine_dipeptide_md.py (OpenMM 8: amber14 force field, OBC2 implicit water, 300 K, Langevin
thermostat, 4 fs steps with hydrogen mass 4 amu, one shape saved every 2 ps). It holds the
positions of the 10 heavy (non-hydrogen) atoms and the backbone angles phi and psi.

Why this data: here every part of the extremum graph has a physical meaning. The molecule
spends its time in a few stable shapes (metastable states); f is high where shapes are common.
  * peak       = a stable shape (a free-energy minimum);
  * saddle     = the way from one stable shape to another (a transition state);
  * persistence = how hard it is to leave a shape (the barrier), in the units of f.

How the point cloud is made
  * points: every 3rd saved shape (6 ps apart); dimensions: the 45 distances between the
    10 heavy atoms, in angstrom (they do not change when the molecule turns or moves);
  * f (main): density, -log of the mean distance to the 15 nearest shapes, scaled to 0..1;
  * second function (Scalar menu): -free energy from the phi / psi histogram,
    -F / kT = ln P(phi, psi) (10 degree bins), so persistence there is a barrier in kT;
  * known labels: the standard regions of the Ramachandran plot (beta, PPII, alpha-R,
    alpha-L, other) from phi and psi; never used to build f or the graph.
"""
import collections
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "web" / "build_data.py").exists():
    sys.exit("run this from the GV_271_Topological_Spines folder (the one that has web/build_data.py)")
if "def scalar_menu" not in (ROOT / "web" / "build_data.py").read_text():
    sys.exit("run add_bc_scalars.py first (it adds the Scalar menu this dataset uses)")
sys.path.insert(0, str(ROOT))
from pipeline.neighbor_graph import knn_graph  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from comparison.variants import scalar_values  # noqa: E402

_IN_REPO = ROOT / "data" / "alanine_dipeptide" / "alanine_dipeptide_md.npz"
SRC = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else \
    (_IN_REPO if _IN_REPO.exists() else Path.home() / "Downloads" / "alanine_dipeptide_md.npz")
if not SRC.exists():
    sys.exit(f"not found: {SRC}\npass the path of alanine_dipeptide_md.npz")
EVERY, K, BIN = 3, 15, 10.0
STATES = ["beta", "PPII", "alpha-R", "alpha-L", "other"]

z = np.load(SRC, allow_pickle=True)
pos, phi, psi, atoms = z["pos"][::EVERY], z["phi"][::EVERY], z["psi"][::EVERY], [str(a) for a in z["heavy"]]
n = pos.shape[0]
print(f"{n} shapes ({z['pos'].shape[0]} saved, every {EVERY}rd kept), {len(atoms)} heavy atoms")


def state(f, s):
    if f < 0:
        if s > 100 or s < -150:
            return "beta" if f < -110 else "PPII"
        if -120 < s < 50:
            return "alpha-R"
        return "other"
    return "alpha-L" if -50 < s < 100 else "other"


label = np.array([state(a, b) for a, b in zip(phi, psi)])

# ---- points and the two functions --------------------------------------------------------
pairs = list(combinations(range(len(atoms)), 2))
P = np.stack([np.linalg.norm(pos[:, i] - pos[:, j], axis=1) for i, j in pairs], axis=1) * 10.0   # nm -> A
P = P.astype(np.float32)
names = [f"{atoms[i]}-{atoms[j]}" for i, j in pairs]
sq = (P.astype(np.float64) ** 2).sum(1)
d = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * P.astype(np.float64) @ P.T.astype(np.float64), 0))
np.fill_diagonal(d, np.inf)
density = -np.log(np.sort(d, axis=1)[:, :K].mean(1))
del d
edges = np.arange(-180, 180 + BIN, BIN)
H, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
iphi = np.clip(np.digitize(phi, edges) - 1, 0, len(edges) - 2)
ipsi = np.clip(np.digitize(psi, edges) - 1, 0, len(edges) - 2)
neg_free = np.log(H[iphi, ipsi] / H.sum())                 # -F/kT, never -inf (every shape is in its own bin)

g = knn_graph(P, K)
f_main = scalar_values(density, g, 0)
out = ROOT / "generator" / "datasets" / "alanine_dipeptide"
(out / "scalars").mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", P)
np.save(out / "values.npy", f_main.astype(np.float64))
np.save(out / "scalars" / "density.npy", density)
np.save(out / "scalars" / "free_energy.npy", neg_free)
present = [s for s in STATES if s in set(label)]
truth = {
    "name": "alanine_dipeptide", "dimensions": int(P.shape[1]), "num_points": n,
    "description": ("Alanine dipeptide shapes from molecular dynamics (OpenMM, implicit water, 300 K). Dimensions: the 45 "
                    "heavy-atom distances in angstrom. f: density of shapes (-log mean distance to the 15 nearest), "
                    "scaled to 0..1."),
    "source": str(SRC), "dimension_names": names, "sample_ids": [f"frame {i * EVERY}" for i in range(n)],
    "extrema": {"maxima": []},
    "point_classes": [{"name": "Ramachandran region", "names": present,
                       "values": [present.index(s) for s in label]}],
    "scalars": {
        "functions": [
            {"id": "density", "label": "Density of shapes", "file": "scalars/density.npy",
             "summary": "Peaks: shapes the molecule visits often."},
            {"id": "free_energy", "label": "-Free energy from phi/psi (kT)", "file": "scalars/free_energy.npy",
             "summary": "ln P(phi, psi) in 10 degree bins: persistence is a barrier in kT (times the range)."},
        ],
        "smooths": [0], "ks": [10, 20, 30], "main": {"id": "density", "smooth": 0},
    },
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {n} points, {P.shape[1]} dimensions")

bd = ROOT / "web" / "build_data.py"
src = bd.read_text()
if '"alanine_dipeptide"' not in src:
    entry = (
        '    "alanine_dipeptide": dict(\n'
        '        label="alanine_dipeptide (molecule shapes from simulation, 45D)", dataset="alanine_dipeptide", k=15,\n'
        '        initial_threshold=0.08, embed=EMBEDS,\n'
        '        text="Shapes of a small peptide from molecular dynamics. Peaks: stable shapes; saddles: transitions '
        'between them; persistence: how hard a shape is to leave. Scalar (f): density, or -free energy from the '
        'backbone angles (kT). Known labels: Ramachandran regions."),\n'
    )
    new, count = re.subn(r"(ENTRIES\s*=\s*\{\n(?:#[^\n]*\n)*)", lambda m: m.group(1) + entry, src, count=1)
    if count != 1:
        sys.exit("could not find 'ENTRIES = {' in web/build_data.py; add the entry by hand")
    bd.write_text(new)
    print("added alanine_dipeptide to web/build_data.py")

# ---- what the spine says ---------------------------------------------------------------------
TAU_DENSITY, TAU_KT = 0.08, 1.0       # density: fraction of its range; free energy: 1 kT (thermal energy)
L = ["# alanine_dipeptide: what the spine shows", "",
     f"{n} shapes from {z['pos'].shape[0] * 2 / 1000:.0f} ns of simulation; kNN graph (k = {K}) on 45 heavy-atom "
     f"distances. Persistence threshold: {TAU_DENSITY} of the range for density (the result is the same from 0.06 to "
     f"0.10), {TAU_KT:.0f} kT for free energy (smaller barriers are below thermal noise). Regions are the Ramachandran labels (not used to "
     "build anything).", "",
     "Time spent in each region: " + ", ".join(f"{s} {np.mean(label == s):.0%}" for s in present) + ".", ""]
for fid, raw, unit in [("density", density, None), ("free_energy", neg_free, "kT")]:
    v = scalar_values(raw, g, 0)
    rng = float(raw.max() - raw.min())
    sp = extremum_graph(v, g).simplify(TAU_KT / rng if unit else TAU_DENSITY)

    def name_of(mask):
        c = collections.Counter(label[mask])
        (t, k), *rest = c.most_common()
        return (f"{t} + {rest[0][0]}" if rest and rest[0][1] >= 0.1 * mask.sum() else t), k / mask.sum()

    L += [f"## f = {'density of shapes' if fid == 'density' else '-free energy from phi/psi'}", "",
          "| Peak = mostly | Shapes | Largest region | Persistence" + (" (kT)" if unit else "") + " | Median phi, psi |",
          "|---|---|---|---|---|"]
    for p in np.argsort(-sp.region_size):
        m = sp.region == p
        t, pur = name_of(m)
        pers = sp.peak_persistence[p]
        ps = "highest" if not np.isfinite(pers) else (f"{pers * rng:.1f}" if unit else f"{pers:.3f}")
        L.append(f"| {t} | {int(m.sum())} | {pur:.0%} | {ps} | {np.median(phi[m]):.0f}, {np.median(psi[m]):.0f} |")
    L += ["", "Saddles (highest first: the easiest transitions):", ""]
    for s in np.argsort(-sp.saddle_values):
        a, b = sp.saddle_peaks[s]
        sv = f"{sp.saddle_values[s] * rng + raw.min():.1f} kT" if unit else f"{sp.saddle_values[s]:.2f}"
        L.append(f"* {name_of(sp.region == a)[0]} -- {name_of(sp.region == b)[0]}: at {sv} "
                 f"({'merge' if sp.saddle_is_merge[s] else 'loop'})")
    L.append("")
    if unit:
        L += [f"In the viewer (f scaled to 0..1), 1 kT is a persistence of {1 / rng:.3f}.", ""]
L += ["Not visited: alpha-L (phi > 0) is reached only on much longer time scales, so it has no peak here."]
res = ROOT / "comparison" / "results"
res.mkdir(parents=True, exist_ok=True)
(res / "alanine_dipeptide.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
print(f"wrote {res / 'alanine_dipeptide.md'}\nnext:  python3 web/build_data.py alanine_dipeptide")
