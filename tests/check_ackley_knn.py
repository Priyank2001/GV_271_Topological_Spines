"""
The Ackley check of check_ackley.py, on kNN graphs of random points instead of grids.

-Ackley on [-1.5, 1.5]^d has 3^d maxima on the lattice {-1, 0, 1}^d, and the
paper's graph (Correa et al. 2011, Fig. 13a) links lattice neighbours only
(points 1 step apart). Here the samples are N uniform random points and the
graph is ``knn_graph(points, k)``, so this tests the point-cloud path of the
pipeline for several k.

Sampling makes spurious peaks, so the graph is first simplified. The
threshold p_lo is the midpoint of the persistence gap between the 3^d-th and
the (3^d + 1)-th peak (this uses the known answer; the "largest gap" column
says whether the largest jump in the persistence curve is at the same place,
i.e. whether p_lo could be found without knowing it).

Columns:
  raw         peaks before simplification
  gap         persistence of the 3^d-th peak / the next one (bigger is easier)
  lg          the largest ratio in the persistence curve is at 3^d (y/n)
  lattice     the simplified peaks are exactly the lattice points (y/n) and the
              largest distance of a peak from its lattice point
  true        true links found / all true links
  diag        links between lattice points 2+ steps apart (the paper has none)
  p_hi err    fewest wrong links any single p_hi gives (true removed + diagonal
              kept), and that p_hi; 0 means p_hi separates them cleanly
  supp err    the same for a minimum ``saddle_support`` instead of p_hi

Not a pytest (several minutes). Run from the repo root:

    python3 tests/check_ackley_knn.py              # d = 2, 3, 4 with the default N and k
    python3 tests/check_ackley_knn.py 3            # one dimension
    python3 tests/check_ackley_knn.py 3 --n 25000 100000 400000 --k 10 --seeds 0 1 2

Results are discussed in observations.md, 2026-11-10.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_ackley import ackley  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402

N_DEFAULT = {2: 20_000, 3: 100_000, 4: 400_000, 5: 1_000_000}
K_DEFAULT = [4, 6, 10, 15, 25]


def best_split(true_vals: np.ndarray, diag_vals: np.ndarray, keep_below: bool):
    """Fewest errors of one cut on a value, and the cut.

    ``keep_below``: links with value <= cut are kept (p_hi); otherwise links with
    value >= cut are kept (minimum support). Errors = true links removed +
    diagonal links kept.
    """
    if not len(diag_vals):
        return 0, None
    vals = np.concatenate([true_vals, diag_vals])
    sign = 1 if keep_below else -1
    best = (len(true_vals) + len(diag_vals), None)
    for c in np.unique(vals):
        kept_true = (sign * true_vals <= sign * c).sum()
        kept_diag = (sign * diag_vals <= sign * c).sum()
        err = (len(true_vals) - kept_true) + kept_diag
        if err < best[0]:
            best = (int(err), float(c))
    return best


def check(d: int, n: int, k: int, seed: int, min_loop_relief: float) -> dict:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.5, 1.5, (n, d))
    v = -ackley(pts)
    v = (v - v.min()) / (v.max() - v.min())
    t0 = time.time()
    eg = extremum_graph(v, knn_graph(pts, k), min_loop_relief=min_loop_relief)
    m = 3 ** d

    pers = np.sort(eg.peak_persistence)[::-1]          # inf (global max) first
    if eg.num_peaks < m:
        return dict(d=d, n=n, k=k, seed=seed, raw=eg.num_peaks, short=True)
    nxt = pers[m] if eg.num_peaks > m else 0.0
    gap = pers[m - 1] / nxt if nxt > 0 else np.inf
    roots = int(np.isinf(pers).sum())                 # > 1 if the kNN graph falls apart
    finite = pers[roots:]
    ratios = finite[:-1] / np.maximum(finite[1:], 1e-12)
    largest_gap_at_m = eg.num_peaks == m or (len(ratios) > 0 and int(np.argmax(ratios)) + 1 + roots == m)
    threshold = (pers[m - 1] + nxt) / 2 if eg.num_peaks > m else 0.0

    g = eg.simplify(threshold)
    pos = pts[g.peaks]
    lat = np.rint(pos).astype(int)
    on_lattice = set(map(tuple, lat)) == set(itertools.product([-1, 0, 1], repeat=d))
    err = float(np.abs(pos - lat).max())
    step = np.abs(lat[g.saddle_peaks[:, 0]] - lat[g.saddle_peaks[:, 1]]).sum(1)
    true, diag = step == 1, step > 1
    num_true = 2 * d * 3 ** (d - 1)                  # lattice edges of {-1,0,1}^d
    centre = int(np.flatnonzero((lat == 0).all(1))[0]) if on_lattice else -1
    centre_deg = int(((g.saddle_peaks == centre).any(1) & true).sum())

    phi = best_split(g.saddle_persistence[true], g.saddle_persistence[diag], keep_below=True)
    sup = best_split(g.saddle_support[true].astype(float), g.saddle_support[diag].astype(float),
                     keep_below=False)
    return dict(d=d, n=n, k=k, seed=seed, raw=eg.num_peaks, short=False, gap=gap,
                lg=largest_gap_at_m, lattice=on_lattice, err=err,
                true=int(true.sum()), num_true=num_true, diag=int(diag.sum()),
                centre=centre_deg, phi=phi, sup=sup, secs=time.time() - t0)


def show(r: dict) -> None:
    head = f"d={r['d']} N={r['n']:>9,} k={r['k']:>2} seed={r['seed']}"
    if r["short"]:
        print(f"{head}: only {r['raw']} peaks, fewer than {3 ** r['d']}")
        return
    cut = lambda c: "-" if c is None else f"{c:.3f}" if c < 10 else f"{c:.0f}"
    print(f"{head}: raw {r['raw']:>6}  gap {r['gap']:>7.1f}  lg {'y' if r['lg'] else 'n'}  "
          f"lattice {'y' if r['lattice'] else 'n'} ({r['err']:.2f})  "
          f"true {r['true']:>3}/{r['num_true']:<3}  diag {r['diag']:>4}  centre {r['centre']:>2}  "
          f"p_hi err {r['phi'][0]:>3} @ {cut(r['phi'][1]):>5}  "
          f"supp err {r['sup'][0]:>3} @ {cut(r['sup'][1]):>5}  [{r['secs']:.1f}s]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dims", nargs="*", type=int, default=[2, 3, 4])
    ap.add_argument("--n", nargs="+", type=int, help="number of points (default depends on d)")
    ap.add_argument("--k", nargs="+", type=int, default=K_DEFAULT)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--min-loop-relief", type=float, default=0.5)
    a = ap.parse_args()
    for d in a.dims:
        for n in a.n or [N_DEFAULT[d]]:
            for k in a.k:
                for s in a.seeds:
                    show(check(d, n, k, s, a.min_loop_relief))


if __name__ == "__main__":
    main()
