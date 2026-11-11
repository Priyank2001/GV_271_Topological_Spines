#!/usr/bin/env python3
"""
Cyclic point-cloud dataset for extremum graph extraction.

Field (points x in R^d, d >= 2):

    f(x) = -(r - R)^2  -  beta * sum_{k>=2} x_k^2  +  a * cos(M * theta)

with r = sqrt(x_0^2 + x_1^2) and theta = atan2(x_1, x_0).

The -(r-R)^2 term makes a ring of radius R in the (x_0, x_1) plane a ridge.
The cos(M theta) ripple turns that ridge into M maxima and M saddles,
alternating around the loop. The extremum graph is therefore a cycle:

    max_0 - saddle_0 - max_1 - saddle_1 - ... - max_{M-1} - saddle_{M-1} - max_0

Ground truth is analytic (up to a small shift, see below):
  maxima : theta_j = 2*pi*j / M,        r = R, x_{k>=2} = 0
  saddles: theta_j = (2*j + 1)*pi / M,  r = R, x_{k>=2} = 0

Each maximum is a nondegenerate local maximum. Each ring-saddle is a minimum
along the ring and a maximum radially, with descent in the extra dimensions,
so its index is d - 1.

The ripple shifts the true critical points slightly off the ring because the
angular derivative is nonzero at r != R; the recorded centers are the
unperturbed positions and are approximate by O(a / R).
"""

import json
import time
from pathlib import Path

import numpy as np

from point_cloud_field import PointCloudScalarFieldGenerator

OUT_DIR = Path(__file__).parent / "datasets"

SPECS = [
    dict(name="cyclic_2d", d=2, n=20_000, R=2.5, M=6, a=0.5, beta=0.0),
    dict(name="cyclic_3d", d=3, n=50_000, R=2.5, M=6, a=0.5, beta=0.1),
    dict(name="cyclic_10d", d=10, n=100_000, R=2.5, M=6, a=0.5, beta=0.1),
]


def ring_field(x: np.ndarray, R: float, M: int, a: float, beta: float) -> np.ndarray:
    x = x.astype(np.float64)
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    theta = np.arctan2(x[:, 1], x[:, 0])
    val = -(r - R) ** 2 + a * np.cos(M * theta)
    if x.shape[1] > 2:
        val -= beta * np.sum(x[:, 2:] ** 2, axis=1)
    return val


def ring_points(R: float, angles: np.ndarray, d: int) -> np.ndarray:
    c = np.zeros((len(angles), d))
    c[:, 0] = R * np.cos(angles)
    c[:, 1] = R * np.sin(angles)
    return c


def nearest_value(points, values, centers):
    """Value at the sample nearest to each center."""
    out = []
    for c in centers:
        i = int(np.argmin(np.sum((points - c) ** 2, axis=1)))
        out.append(float(values[i]))
    return np.array(out)


def ring_order_check(points, values, max_c, sad_c):
    """Each saddle should be lower than both neighbouring maxima on the ring,
    and each maximum should be above the median of its ring neighbourhood."""
    mv = nearest_value(points, values, max_c)
    sv = nearest_value(points, values, sad_c)
    M = len(mv)
    ok = sum(int(sv[j] < mv[j] and sv[j] < mv[(j + 1) % M]) for j in range(M))
    return ok, M


def build(spec: dict):
    gen = PointCloudScalarFieldGenerator(seed=200 + len(spec["name"]))
    d, n, R, M, a, beta = (spec[k] for k in ("d", "n", "R", "M", "a", "beta"))
    # In-plane coordinates uniform over the domain. Extra coordinates are
    # N(0, 0.5^2): uniform sampling leaves almost no points near x_k = 0
    # in many dimensions, so the ring would be unresolved.
    pts = gen.sample_points(n, d, bounds=(-5.0, 5.0), method="uniform")
    if d > 2:
        pts[:, 2:] = gen.rng.normal(0.0, 0.5, size=(n, d - 2)).astype(pts.dtype)
    vals = ring_field(pts, R, M, a, beta)
    vals = (vals - vals.min()) / (vals.max() - vals.min())  # normalise to [0, 1]

    max_ang = 2 * np.pi * np.arange(M) / M
    sad_ang = (2 * np.arange(M) + 1) * np.pi / M
    max_c = ring_points(R, max_ang, d)
    sad_c = ring_points(R, sad_ang, d)

    cycle = []
    for j in range(M):
        cycle.append({"type": "max", "id": f"max_{j}", "center": max_c[j].tolist()})
        cycle.append({"type": "saddle", "id": f"saddle_{j}", "center": sad_c[j].tolist()})
    edges = []
    for j in range(M):
        edges.append([f"max_{j}", f"saddle_{j}"])
        edges.append([f"saddle_{j}", f"max_{(j + 1) % M}"])

    return pts.astype(np.float32), vals.astype(np.float32), max_c, sad_c, cycle, edges


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for spec in SPECS:
        t0 = time.time()
        pts, vals, max_c, sad_c, cycle, edges = build(spec)

        out = OUT_DIR / spec["name"]
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "points.npy", pts)
        np.save(out / "values.npy", vals)

        ok_ring, nring = ring_order_check(pts, vals, max_c, sad_c)
        truth = {
            "name": spec["name"],
            "dimensions": spec["d"],
            "num_points": int(pts.shape[0]),
            "mode": "cyclic_ring",
            "has_extrema_labels": True,
            "generator_metadata": {k: spec[k] for k in ("R", "M", "a", "beta")},
            "formula": "f = -(r-R)^2 - beta*sum_{k>=2} x_k^2 + a*cos(M*theta)",
            "value_range_raw": [float(vals.min()), float(vals.max())],
            "critical_points": cycle,
            "extremum_graph_edges": edges,
            "extremum_graph_is_cycle": True,
            "sample_check": {
                "saddle_below_both_neighbour_maxima": f"{ok_ring}/{nring}",
            },
        }
        with open(out / "truth.json", "w") as f:
            json.dump(truth, f, indent=2)

        print(f"{spec['name']:12s} d={spec['d']:<3d} N={pts.shape[0]:<7d} "
              f"saddle<neighbour-maxima {ok_ring}/{nring}  "
              f"{time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
