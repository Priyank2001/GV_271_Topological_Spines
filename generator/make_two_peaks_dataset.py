#!/usr/bin/env python3
"""
Two peaks and one saddle in 2D: the smallest field with a non-trivial extremum graph.

    f(x, y) = exp(-|p - c1|^2 / (2 s^2)) + h * exp(-|p - c2|^2 / (2 s^2))

Two Gaussian bumps of different height (the second is ``h`` times the first), so
the lower peak has a clear persistence. Along the line joining the centres the
field has exactly one saddle; there is no other critical point. The expected
extremum graph is  max_0 -- saddle_0 -- max_1  (a path, no cycle).

Values are normalised to [0, 1]. ``truth.json`` lists the two maxima and the saddle
(found on the line y = 0 by root finding on the analytic field).
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from point_cloud_field import PointCloudScalarFieldGenerator

OUT_DIR = Path(__file__).parent / "datasets"
NAME = "two_peaks_2d"
N = 4000
C1, C2, S, H = (-1.6, 0.0), (1.6, 0.0), 1.0, 0.7
BOUNDS = (-4.0, 4.0)


def field(p):
    p = np.atleast_2d(np.asarray(p, float))
    g = lambda c, a: a * np.exp(-((p[:, 0] - c[0]) ** 2 + (p[:, 1] - c[1]) ** 2) / (2 * S * S))
    return g(C1, 1.0) + g(C2, H)


def main():
    gen = PointCloudScalarFieldGenerator(seed=7)
    pts = gen.sample_points(N, 2, bounds=BOUNDS, method="uniform")
    raw = field(pts)
    # normalise by the range over the sample, as the volcano generator does
    lo, hi = float(raw.min()), float(raw.max())
    norm = lambda v: float((v - lo) / (hi - lo))
    vals = ((raw - lo) / (hi - lo)).astype(np.float32)

    dfx = lambda x: (field([[x + 1e-5, 0]]) - field([[x - 1e-5, 0]]))[0]
    xs = brentq(dfx, C1[0] + 0.5, C2[0] - 0.5)
    xm = [brentq(dfx, c[0] - 0.4, c[0] + 0.4) for c in (C1, C2)]
    maxima = [{"type": "max", "id": f"max_{i}", "center": [float(x), 0.0],
               "value": norm(field([[x, 0]])[0])} for i, x in enumerate(xm)]
    saddle = {"type": "saddle", "id": "saddle_0", "center": [float(xs), 0.0],
              "value": norm(field([[xs, 0]])[0])}
    truth = {
        "name": NAME, "dimensions": 2, "num_points": int(N), "mode": "two_peaks",
        "has_extrema_labels": True,
        "generator_metadata": {"c1": C1, "c2": C2, "sigma": S, "h": H, "bounds": BOUNDS},
        "formula": "f = exp(-|p-c1|^2/2s^2) + h*exp(-|p-c2|^2/2s^2)",
        "value_range_raw": [0.0, 1.0],
        "critical_points": maxima + [saddle],
        "extremum_graph_edges": [["max_0", "saddle_0"], ["saddle_0", "max_1"]],
        "extremum_graph_is_cycle": False,
    }
    out = OUT_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", pts.astype(np.float32))
    np.save(out / "values.npy", vals)
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"{NAME}: N={N}, maxima {[round(m['value'], 3) for m in maxima]}, "
          f"saddle {saddle['value']:.3f} at x={xs:.3f}")


if __name__ == "__main__":
    main()
