"""
Write ``fixtures/simplify_cases.json``: exported graphs plus what Python's
``ExtremumGraph.simplify`` (and ``arc_mask``) return for several thresholds. ``simplify.test.mjs``
checks that ``src/simplify.js`` gives the same answers.

Run from the repo root:  python3 web/tests/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from pipeline.data_loader import load_dataset  # noqa: E402
from pipeline.export_json import extremum_graph_to_dict  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import grid_graph, knn_graph  # noqa: E402
from pipeline.volume_loader import download, load_volume  # noqa: E402

THRESHOLDS = [0.0, 0.03, 0.08, 0.1, 0.2, 0.5, 2.0]
# (noise threshold, variation threshold) pairs for the paper's p_lo / p_hi
VARIATIONS = [(0.0, 0.05), (0.03, 0.2), (0.1, 0.1), (0.1, 0.5), (0.2, 0.0), (0.0, 100.0)]


def case(name, raw, base_threshold):
    base = raw.simplify(base_threshold)
    doc = extremum_graph_to_dict(base, include_region=False, meta={"name": name})
    def summary(s, t, v=None):
        return {
            "peaks": s.peaks.tolist(),
            "saddles": s.saddles.tolist(),
            "saddleEnds": s.saddle_peaks.tolist(),
            "size": s.region_size.tolist(),
            # saturated persistence: saddles (by vertex) whose arcs are drawn
            "arcs": base.saddles[base.arc_mask(t, v)].tolist(),
        }

    expected = [summary(base.simplify(t), t) for t in THRESHOLDS]    # one entry per threshold, same order
    variation = [{"threshold": t, "variation": v, **summary(base.simplify(t, variation=v), t, v)}
                 for t, v in VARIATIONS]
    return {"name": name, "doc": doc, "expected": expected, "variation": variation}


def main() -> None:
    cases = []
    for vol_name in ("neghip", "silicium"):
        download(vol_name)
        vol = load_volume(vol_name)
        raw = extremum_graph(vol.values.reshape(-1), grid_graph(vol.shape))
        cases.append(case(vol_name, raw, 0.02))
    ds = load_dataset("cyclic_10d")
    cases.append(case("cyclic_10d", extremum_graph(ds.values, knn_graph(ds.points, 20)), 0.005))

    rng = np.random.default_rng(5)                 # noisy field: many peaks, many ties in pairs
    pts = rng.uniform(size=(4000, 2))
    f = np.sin(7 * pts[:, 0]) * np.cos(7 * pts[:, 1]) + 0.1 * rng.normal(size=4000)
    cases.append(case("noisy_2d", extremum_graph(f, knn_graph(pts, 8)), 0.0))

    out = HERE / "fixtures" / "simplify_cases.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"thresholds": THRESHOLDS, "cases": cases},
                              separators=(",", ":")))
    for c in cases:
        print(c["name"], len(c["doc"]["peaks"]), "peaks,", len(c["doc"]["saddles"]), "saddles")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
