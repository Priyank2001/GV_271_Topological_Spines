"""
Compare our extremum graph with the paper's Ackley example (Correa et al. 2011, Fig. 13a).

The paper says Ackley's function in d dimensions has 3^d minima (a 3x3x... lattice)
and that the centre links to 2d of them. We sample -Ackley on [-1.5, 1.5]^d grids
(minima become maxima) and compare peaks and links with the lattice.

Not a pytest (the 5D case takes about a minute). Run from the repo root:

    python3 tests/check_ackley.py            # d = 2, 3, 4  (add 5 as an argument for the slow case)
    python3 tests/check_ackley.py 2 3

Results are discussed in observations.md, 2026-11-10.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import grid_graph  # noqa: E402

GRID = {2: 301, 3: 81, 4: 41, 5: 25}


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return (-20 * np.exp(-0.2 * np.sqrt((x ** 2).sum(1) / d))
            - np.exp(np.cos(2 * np.pi * x).mean(1)) + 20 + np.e)


def check(d: int) -> None:
    n = GRID[d]
    axis = np.linspace(-1.5, 1.5, n)
    pts = np.stack(np.meshgrid(*[axis] * d, indexing="ij"), -1).reshape(-1, d)
    v = -ackley(pts)
    v = (v - v.min()) / (v.max() - v.min())
    t = time.time()
    eg = extremum_graph(v, grid_graph((n,) * d))
    pos = np.rint(pts[eg.peaks]).astype(int)
    lattice = set(itertools.product([-1, 0, 1], repeat=d))
    found = set(map(tuple, pos))
    # a true link joins lattice points that differ by 1 in exactly one coordinate
    true_links = {frozenset((p, q)) for p in found for q in found
                  if sum(abs(a - b) for a, b in zip(p, q)) == 1}
    ours = {frozenset((tuple(pos[a]), tuple(pos[b]))) for a, b in eg.saddle_peaks}
    centre = sum(1 for e in ours if tuple([0] * d) in e)
    print(f"d={d}: {eg.num_peaks} peaks (expect {3 ** d}), at lattice points: {found == lattice}; "
          f"links {len(ours)}: {len(ours & true_links)} of {len(true_links)} true found, "
          f"{len(ours - true_links)} extra; centre degree {centre} (expect {2 * d})  [{time.time() - t:.1f}s]")
    step = np.abs(pos[eg.saddle_peaks[:, 0]] - pos[eg.saddle_peaks[:, 1]]).sum(1)
    for s in sorted(set(step)):
        p = eg.saddle_persistence[step == s]
        print(f"     lattice distance {s}: {len(p):4d} links, saddle persistence {p.min():.3f} .. {p.max():.3f}")


if __name__ == "__main__":
    for d in [int(a) for a in sys.argv[1:]] or [2, 3, 4]:
        check(d)
