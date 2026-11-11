#!/usr/bin/env python3
"""
Generate labelled point-cloud scalar-field datasets for extremum graph extraction.

Each dataset is written to datasets/<name>/ as:
  points.npy     (N, d) float32 sample positions
  values.npy     (N,)   float32 scalar value per point
  truth.json     ground truth: dimensions, mode, and (where known) the
                 extrema centers, amplitudes and saddle signs

Ground truth is exact only for "controlled_extrema" datasets, where the
maxima, minima and saddles are placed explicitly. Analytic and random
datasets carry no extrema labels and are meant for stress testing.

No neighbour graph is stored: the pipeline builds it after loading, with
whatever k (or grid connectivity) it needs.
"""

import json
import time
from pathlib import Path

import numpy as np

from point_cloud_field import PointCloudScalarFieldGenerator

OUT_DIR = Path(__file__).parent / "datasets"

# name, dimensions, num_points, builder kwargs
SPECS = [
    # Controlled extrema: known ground truth
    dict(name="ext_2d_small", d=2, n=5_000, kind="extrema",
         kw=dict(num_maxima=3, num_minima=3, num_saddles=2, width=0.6)),
    dict(name="ext_3d", d=3, n=20_000, kind="extrema",
         kw=dict(num_maxima=4, num_minima=4, num_saddles=3, width=0.8)),
    dict(name="ext_10d", d=10, n=50_000, kind="extrema",
         kw=dict(num_maxima=4, num_minima=4, num_saddles=3, width=1.5)),
    dict(name="ext_50d", d=50, n=100_000, kind="extrema",
         kw=dict(num_maxima=4, num_minima=4, num_saddles=3, width=3.0)),
    dict(name="ext_100d", d=100, n=100_000, kind="extrema",
         kw=dict(num_maxima=4, num_minima=4, num_saddles=3, width=4.0)),
    # Analytic / random: no extrema labels, stress tests
    dict(name="peaks_2d", d=2, n=10_000, kind="analytic",
         kw=dict(function="gaussian_peaks", num_peaks=5)),
    dict(name="rastrigin_3d", d=3, n=20_000, kind="analytic",
         kw=dict(function="rastrigin")),
    dict(name="smooth_20d", d=20, n=50_000, kind="smooth",
         kw=dict(num_features=256, length_scale=3.0)),
]


def build(gen: PointCloudScalarFieldGenerator, spec: dict):
    d, n = spec["d"], spec["n"]
    pts = gen.sample_points(n, d, bounds=(-5.0, 5.0), method="uniform")
    if spec["kind"] == "extrema":
        return gen.controlled_extrema(pts, **spec["kw"])
    if spec["kind"] == "analytic":
        return gen.analytic(pts, **spec["kw"])
    if spec["kind"] == "smooth":
        return gen.smooth_random(pts, **spec["kw"])
    raise ValueError(spec["kind"])


def check_extrema(field, truth_extrema) -> dict:
    """Sanity check: value at nearest sample to each max should exceed the
    median, and to each min should be below the median."""
    med = float(np.median(field.values))
    pts = field.points.astype(np.float64)
    report = {"maxima_above_median": 0, "minima_below_median": 0}
    for m in truth_extrema["maxima"]:
        i = int(np.argmin(np.sum((pts - np.array(m["center"])) ** 2, axis=1)))
        report["maxima_above_median"] += int(field.values[i] > med)
    for m in truth_extrema["minima"]:
        i = int(np.argmin(np.sum((pts - np.array(m["center"])) ** 2, axis=1)))
        report["minima_below_median"] += int(field.values[i] < med)
    return report


def main():
    OUT_DIR.mkdir(exist_ok=True)
    index = []
    for idx, spec in enumerate(SPECS):
        t0 = time.time()
        gen = PointCloudScalarFieldGenerator(seed=100 + idx)
        field = build(gen, spec)

        out = OUT_DIR / spec["name"]
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "points.npy", field.points)
        np.save(out / "values.npy", field.values)

        truth = {
            "name": spec["name"],
            "dimensions": field.dimensions,
            "num_points": field.num_points,
            "mode": field.mode,
            "has_extrema_labels": spec["kind"] == "extrema",
            "generator_metadata": field.metadata,
            "value_range": [float(field.values.min()), float(field.values.max())],
            "seed": 100 + idx,
        }
        if spec["kind"] == "extrema":
            truth["extrema"] = field.metadata["extrema"]
            truth["sample_check"] = check_extrema(field, field.metadata["extrema"])
        with open(out / "truth.json", "w") as f:
            json.dump(truth, f, indent=2)

        dt = time.time() - t0
        print(f"{spec['name']:14s} d={field.dimensions:<4d} N={field.num_points:<7d} "
              f"mode={field.mode:<18s} {dt:6.1f}s  -> {out}")
        index.append({k: truth[k] for k in ("name", "dimensions", "num_points", "mode",
                                            "has_extrema_labels")})

    with open(OUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nWrote {len(SPECS)} datasets to {OUT_DIR}")


if __name__ == "__main__":
    main()
