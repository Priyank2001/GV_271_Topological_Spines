#!/usr/bin/env python3
"""
Volcano ("cycle of hills") datasets after Correa et al. 2011, Fig. 5 and section 3.4.

A crater with a rim of M hills. One rim saddle (between hill 0 and hill 1, at
30 degrees for M = 6) is cut down by a notch of depth b, so the rim can be
breached:

    f(r, theta) = F * inside(r) + Q * r^2 * inside(r)                   crater floor (a bowl)
                + exp(-(r - R)^2 / (2 W^2))                               the rim (a ring-shaped ridge)
                  * (1 - A * (1 - cos(M theta)) / 2)                      M hills along it
                  * (1 - b * exp(-dtheta^2 / (2 SB^2)))                   the notch, dtheta = theta - 30 deg
                - 0.03 * r * (1 - inside(r))                              plain outside, falling gently

with inside(r) = 1 / (1 + exp(-(R - r) / SIG)). Values are normalised to [0, 1].

What the paper says should happen (section 3.4 and Fig. 5): a saddle is
*ridge-like* when it is closer in value to its two maxima than to the
minimum on its other side (here the crater floor), and *valley-like*
otherwise. A ridge-like breach saddle keeps the ring of hills closed in the
spine (Fig. 5b); a valley-like one, where the rim is really breached, is cut
and the ring opens into a chain (Fig. 5a, "if we cut the cycle ... we remove
both the saddle in yellow and its incident arcs").

    volcano_intact    b = 0     ring; every rim saddle ridge-like
    volcano_notched   b = 0.2   still a ring: the notch saddle is closer to the hills than to the floor
    volcano_breached  b = 0.8   chain: the breach saddle is nearly as low as the crater floor

``truth.json`` records, per rim saddle, its value, its distance to the
lower neighbouring hill and to the crater floor, and the paper's
classification, all measured on a fine polar grid of the analytic field.
See tests/check_volcano.py and observations.md for what our extremum graph
does with these.
"""

import json
import time
from pathlib import Path

import numpy as np

from point_cloud_field import PointCloudScalarFieldGenerator

OUT_DIR = Path(__file__).parent / "datasets"

R, W, M, A = 2.5, 0.5, 6, 0.3          # rim radius and width, number of hills, hill/saddle contrast
F, Q, SIG = 0.35, 0.02, 0.2            # crater floor height, bowl curvature, softness of the crater wall
SB = 0.2                               # angular width of the notch (radians)
THETA_B = np.pi / M                    # the notch sits on the saddle between hill 0 and hill 1

SPECS = [
    dict(name="volcano_intact", b=0.0, n=30_000),
    dict(name="volcano_notched", b=0.2, n=30_000),
    dict(name="volcano_breached", b=0.8, n=30_000),
]


def volcano_field(x: np.ndarray, b: float) -> np.ndarray:
    """Raw (not normalised) field at 2D points ``x``."""
    x = np.asarray(x, dtype=np.float64)
    r = np.hypot(x[:, 0], x[:, 1])
    th = np.arctan2(x[:, 1], x[:, 0])
    inside = 1.0 / (1.0 + np.exp(-(R - r) / SIG))
    dth = np.angle(np.exp(1j * (th - THETA_B)))
    notch = 1.0 - b * np.exp(-dth ** 2 / (2 * SB ** 2))
    rim = np.exp(-(r - R) ** 2 / (2 * W ** 2)) * (1 - A * (1 - np.cos(M * th)) / 2) * notch
    return F * inside + Q * r ** 2 * inside + rim - 0.03 * r * (1 - inside)


def rim_profile(b: float, n_theta: int = 7200, n_r: int = 801):
    """Crest of the rim: for each angle the highest value over r in [1, 4]."""
    th = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r = np.linspace(1.0, 4.0, n_r)
    T, Rr = np.meshgrid(th, r, indexing="ij")
    pts = np.stack([Rr * np.cos(T), Rr * np.sin(T)], -1).reshape(-1, 2)
    v = volcano_field(pts, b).reshape(n_theta, n_r)
    k = v.argmax(1)
    return th, r[k], v[np.arange(n_theta), k]


def critical_points(b: float, lo: float, hi: float):
    """Hills (maxima of the crest), rim saddles (minima of the crest) and the crater floor,
    with values normalised by the dataset's [lo, hi]."""
    th, rr, crest = rim_profile(b)
    norm = lambda v: (v - lo) / (hi - lo)
    prev, nxt = np.roll(crest, 1), np.roll(crest, -1)
    hills = np.flatnonzero((crest > prev) & (crest >= nxt))
    sads = np.flatnonzero((crest < prev) & (crest <= nxt))
    floor_r = np.linspace(0, 2.0, 2001)
    floor_v = volcano_field(np.stack([floor_r, np.zeros_like(floor_r)], 1), b)
    floor = norm(floor_v.min())
    pos = lambda i: [float(rr[i] * np.cos(th[i])), float(rr[i] * np.sin(th[i]))]
    hills = hills[np.argsort(np.degrees(th[hills]) % 360)]
    sads = sads[np.argsort(np.degrees(th[sads]) % 360)]
    hill_v = norm(crest[hills])
    out_h = [{"type": "max", "id": f"max_{j}", "center": pos(i), "value": float(hill_v[j]),
              "angle_deg": float(np.degrees(th[i]) % 360)} for j, i in enumerate(hills)]
    out_s = []
    for j, i in enumerate(sads):
        a = np.degrees(th[i]) % 360
        # neighbouring hills: the ones just below and above this angle
        hv = sorted(out_h, key=lambda h: (h["angle_deg"] - a) % 360)
        up = min(hv[0]["value"], hv[-1]["value"]) - norm(crest[i])
        down = norm(crest[i]) - floor
        out_s.append({"type": "saddle", "id": f"saddle_{j}", "center": pos(i), "value": float(norm(crest[i])),
                      "angle_deg": float(a), "to_peaks": float(up), "to_floor": float(down),
                      "paper_class": "ridge" if up < down else "valley",
                      "between": [hv[-1]["id"], hv[0]["id"]]})
    return out_h, out_s, floor


def build(spec: dict):
    gen = PointCloudScalarFieldGenerator(seed=300 + len(spec["name"]))
    pts = gen.sample_points(spec["n"], 2, bounds=(-5.0, 5.0), method="uniform")
    raw = volcano_field(pts, spec["b"])
    lo, hi = float(raw.min()), float(raw.max())
    vals = (raw - lo) / (hi - lo)
    hills, sads, floor = critical_points(spec["b"], lo, hi)
    # The paper's expected maximum graph: every ridge-like rim saddle links its two hills.
    edges = []
    for s in sads:
        if s["paper_class"] == "ridge":
            edges += [[s["between"][0], s["id"]], [s["id"], s["between"][1]]]
    return pts.astype(np.float32), vals.astype(np.float32), hills, sads, floor, edges


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for spec in SPECS:
        t0 = time.time()
        pts, vals, hills, sads, floor, edges = build(spec)
        out = OUT_DIR / spec["name"]
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "points.npy", pts)
        np.save(out / "values.npy", vals)
        ridge = sum(s["paper_class"] == "ridge" for s in sads)
        truth = {
            "name": spec["name"],
            "dimensions": 2,
            "num_points": int(pts.shape[0]),
            "mode": "volcano",
            "has_extrema_labels": True,
            "generator_metadata": {"R": R, "W": W, "M": M, "A": A, "F": F, "Q": Q, "SIG": SIG,
                                   "SB": SB, "breach_angle_deg": float(np.degrees(THETA_B)), "b": spec["b"]},
            "formula": "see generator/make_volcano_dataset.py",
            "value_range_raw": [0.0, 1.0],
            "crater_floor_value": float(floor),
            "critical_points": hills + sads,
            "extremum_graph_edges": edges,
            "extremum_graph_is_cycle": ridge == len(sads),
            "paper_expectation": ("ring: every rim saddle is ridge-like" if ridge == len(sads) else
                                  f"chain: {len(sads) - ridge} rim saddle(s) valley-like, cut by p_hi"),
        }
        with open(out / "truth.json", "w") as f:
            json.dump(truth, f, indent=2)
        breach = min(sads, key=lambda s: abs(s["angle_deg"] - np.degrees(THETA_B)))
        print(f"{spec['name']:17s} b={spec['b']:<4} N={pts.shape[0]}  {len(hills)} hills, {len(sads)} saddles; "
              f"breach saddle f={breach['value']:.3f}: to peaks {breach['to_peaks']:.3f}, "
              f"to floor {breach['to_floor']:.3f} -> {breach['paper_class']}; "
              f"expect {truth['paper_expectation']}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
