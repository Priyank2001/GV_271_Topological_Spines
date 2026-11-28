#!/usr/bin/env python3
"""
One saddle shared by 10 peaks: a sampled "monkey saddle" of order 10 in 2D.

The points lie on a star of 20 thin arms (strips of width W) that start at
distance RHO from the origin, plus one point at the origin itself (the hub).
Arms alternate between

* **ridges** (even spokes, angle 36 j degrees): f rises from the hub value S0
  to a peak of height H[j] at distance TP and then falls off gently;
* **valleys** (odd spokes, angle 18 + 36 j degrees): f falls from S0 towards
  FMIN without any interior extremum.

    t = distance along the nearest spoke
    ridge j:  f = S0 + (H[j] - S0) * (t / TP) * exp(1 - t / TP)
    valley:   f = S0 + (FMIN - S0) * (1 - exp(-t / TP))
    hub:      f = S0

f does not depend on the offset across an arm: the crest of a ridge is flat
near its peak, and even a small cross term (-2 u^2) made a second, spurious
maximum on one arm.

So the hub has 10 ways up and 10 ways down, alternating: it is the one saddle
of all 10 peaks (a degenerate, order-10 monkey saddle), and the expected
extremum graph is a star, max_j -- saddle_0 for j = 0..9. Values are
normalised to [0, 1] by the sample range.

Why a star of arms and not a smooth field on the whole plane: a degenerate
saddle is not stable under sampling. Sampled densely around it, a kNN graph
joins neighbouring ridges right next to the centre, and the saddle breaks up
into a cluster of ordinary saddles (6-16 of them, 0.2-1.5 units from the
centre, in a test with k = 8..20). Here the ridges touch nowhere but at the
hub, so one vertex joins all 10 peaks.

The hub's 20 nearest neighbours are the 20 arm tips (one per arm, all at
distance RHO), so the dataset needs **k >= 20**: with a smaller k the hub
misses some tips and those arms are not connected to the rest of the graph.
Our pipeline stores a saddle of 10 peaks as 9 rows (highest peak, other) on
the same vertex (see pipeline/anyeg_saddles.py).
"""

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "datasets"
NAME = "star_saddle_2d"

M = 10                                  # peaks (ridges); there are as many valleys
RHO, L, TP, W = 0.5, 3.0, 2.0, 0.01     # arm start and end, peak distance, arm width
S0, FMIN = 0.45, 0.0                    # hub (saddle) value, where the valleys head
PER_UNIT = 300                          # points per unit of arm length
# Peak heights around the star (ridge j at 36 j degrees), mixed so neighbours differ.
H = np.array([1.0, 0.72, 0.9, 0.78, 0.96, 0.7, 0.84, 0.75, 0.93, 0.81])
SPOKE = np.pi * np.arange(2 * M) / M    # even: ridges, odd: valleys
SEED = 5


def field(p: np.ndarray) -> np.ndarray:
    """Raw (not normalised) field at 2D points ``p``, measured along the nearest spoke."""
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    th = np.arctan2(p[:, 1], p[:, 0])
    j = np.rint(th / (np.pi / M)).astype(int) % (2 * M)
    d = np.stack([np.cos(SPOKE[j]), np.sin(SPOKE[j])], 1)
    t = (p * d).sum(1)
    ridge = j % 2 == 0
    g = np.where(ridge, (t / TP) * np.exp(1 - t / TP), 1 - np.exp(-t / TP))
    top = np.where(ridge, H[j // 2], FMIN)
    return S0 + (top - S0) * g


def sample() -> np.ndarray:
    """The hub, then PER_UNIT * (L - RHO) points per arm; each arm's first point is its tip."""
    rng = np.random.default_rng(SEED)
    n = int(PER_UNIT * (L - RHO))
    arms = []
    for a in SPOKE:
        t = rng.uniform(RHO, L, n)
        u = rng.uniform(-W / 2, W / 2, n)
        t[0], u[0] = RHO, 0.0           # the tip, the arm's point nearest the hub
        arms.append(np.stack([t * np.cos(a) - u * np.sin(a), t * np.sin(a) + u * np.cos(a)], 1))
    return np.concatenate([np.zeros((1, 2))] + arms)


def main():
    pts = sample()
    raw = field(pts)
    lo, hi = float(raw.min()), float(raw.max())
    norm = lambda v: float((v - lo) / (hi - lo))
    vals = ((raw - lo) / (hi - lo)).astype(np.float32)

    maxima = [{"type": "max", "id": f"max_{j}",
               "center": [float(TP * np.cos(SPOKE[2 * j])), float(TP * np.sin(SPOKE[2 * j]))],
               "value": norm(H[j]), "angle_deg": float(36 * j),
               "persistence": norm(H[j]) - norm(S0)} for j in range(M)]
    saddle = {"type": "saddle", "id": "saddle_0", "center": [0.0, 0.0], "value": norm(S0),
              "multiplicity": M, "between": [m["id"] for m in maxima]}
    truth = {
        "name": NAME, "dimensions": 2, "num_points": int(pts.shape[0]), "mode": "star_saddle",
        "has_extrema_labels": True,
        "generator_metadata": {"M": M, "rho": RHO, "L": L, "tp": TP, "w": W, "s0": S0,
                               "fmin": FMIN, "per_unit": PER_UNIT, "heights": H.tolist(),
                               "seed": SEED, "min_k": 2 * M},
        "formula": "see generator/make_star_saddle_dataset.py",
        "value_range_raw": [0.0, 1.0],
        "critical_points": maxima + [saddle],
        "extremum_graph_edges": [[m["id"], "saddle_0"] for m in maxima],
        "extremum_graph_is_cycle": False,
        "hub_vertex": 0,
    }
    out = OUT_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", pts.astype(np.float32))
    np.save(out / "values.npy", vals)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"{NAME}: N={pts.shape[0]}, {M} peaks {[round(m['value'], 3) for m in maxima]}, "
          f"one saddle at the origin, f={saddle['value']:.3f}; use k >= {2 * M}")


if __name__ == "__main__":
    main()
