"""Tests for pipeline/export_json.py. Run from the repo root: python3 -m pytest tests"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.export_json import (check_export, export_extremum_graph,
                                  extremum_graph_to_dict, points_data, volume_data)
from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import NeighborGraph, grid_graph, knn_graph


def five_point_path():
    g = NeighborGraph.from_edges(5, np.array([[i, i + 1] for i in range(4)]))
    return extremum_graph(np.array([9, 4, 7, 3, 8.0]), g)


def test_graph_fields_survive_a_json_round_trip(tmp_path):
    eg = five_point_path()
    coords = np.arange(5, dtype=float)[:, None] * [1.0, 10.0]
    path = export_extremum_graph(eg, tmp_path / "sub" / "g.json", coords=coords,
                                 meta={"name": "toy"})
    doc = json.loads(path.read_text())
    check_export(doc)

    assert doc["meta"]["name"] == "toy" and doc["meta"]["num_points"] == 5
    assert doc["meta"]["value_range"] == [3.0, 9.0]
    assert [p["vertex"] for p in doc["peaks"]] == [0, 4, 2]
    assert doc["peaks"][0]["persistence"] is None          # inf -> null
    assert doc["peaks"][2]["persistence"] == 3.0
    assert [p["parent"] for p in doc["peaks"]] == [-1, 0, 0]
    assert doc["peaks"][2]["position"] == [2.0, 20.0]
    assert [s["peaks"] for s in doc["saddles"]] == [[0, 2], [1, 2]]
    assert [s["vertex"] for s in doc["saddles"]] == [1, 3]
    assert [s["saturation"] for s in doc["saddles"]] == [5.0, 5.0]
    assert doc["region"] == [0, 0, 2, 1, 1]
    assert [p["size"] for p in doc["peaks"]] == [2, 2, 1]


def test_parent_matches_the_extremum_graph():
    eg = five_point_path()
    doc = extremum_graph_to_dict(eg)
    assert [p["parent"] for p in doc["peaks"]] == eg.peak_parent.tolist()
    assert "position" not in doc["peaks"][0]               # no coords given


def test_region_and_curve_are_optional_and_curve_uses_the_unsimplified_graph():
    rng = np.random.default_rng(1)
    pts = rng.uniform(size=(1500, 2))
    f = np.sin(6 * pts[:, 0]) * np.cos(6 * pts[:, 1]) + 0.05 * rng.normal(size=1500)
    raw = extremum_graph(f, knn_graph(pts, 8))
    small = raw.simplify(0.3)
    assert small.num_peaks < raw.num_peaks

    doc = extremum_graph_to_dict(small, include_region=False,
                                 full_peak_persistence=raw.peak_persistence)
    check_export(doc)
    assert "region" not in doc
    curve = doc["persistence_curve"]
    assert len(curve) == raw.num_peaks - 1                 # global maximum left out
    assert curve == sorted(curve, reverse=True)
    assert doc["meta"]["simplify_threshold"] is None       # caller sets it via meta


@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.3])
def test_simplified_graph_passes_the_schema_check(threshold):
    rng = np.random.default_rng(2)
    pts = rng.uniform(size=(2000, 3))
    f = np.sin(5 * pts[:, 0]) + np.cos(4 * pts[:, 1]) * pts[:, 2]
    eg = extremum_graph(f, knn_graph(pts, 10)).simplify(threshold)
    check_export(extremum_graph_to_dict(eg, coords=pts))


def test_check_export_catches_a_broken_region():
    doc = extremum_graph_to_dict(five_point_path())
    doc["region"][0] = 1
    with pytest.raises(ValueError):
        check_export(doc)


def test_coords_can_be_a_function_for_grids():
    shape = (6, 7, 8)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    f = np.exp(-((x - .5) ** 2 + y ** 2 + z ** 2) * 4).ravel()
    eg = extremum_graph(f, grid_graph(shape))
    seen = []

    def coords(v):
        seen.append(len(v))
        zz, yy, xx = np.unravel_index(v, shape)
        return np.stack([xx, yy, zz], axis=-1)

    doc = extremum_graph_to_dict(eg, coords=coords)
    assert doc["peaks"][0]["position"] == [float(c) for c in
                                           np.array(np.unravel_index(eg.peaks[0], shape))[::-1]]
    assert max(seen) <= eg.num_peaks + eg.num_saddles      # never asked for all N voxels


def test_points_data_and_volume_data():
    d = points_data(np.array([0.123456789, 1.0]), np.array([[0.5, 1.5], [2.0, 3.0]]))
    assert d["values"] == [0.1235, 1.0] and d["embedding"] == [[0.5, 1.5], [2.0, 3.0]]
    with pytest.raises(ValueError):
        points_data(np.zeros(3), np.zeros((2, 2)))

    class Vol:
        shape, spacing, values = (2, 2, 3), np.array([1, 1, 2.0]), np.linspace(0, 1, 12).reshape(2, 2, 3)

    v = volume_data(Vol())
    assert v["shape"] == [2, 2, 3] and v["spacing"] == [1.0, 1.0, 2.0] and len(v["values"]) == 12


def test_nan_is_refused_at_write_time(tmp_path):
    from pipeline.export_json import write_json

    with pytest.raises(ValueError):
        write_json({"x": float("nan")}, tmp_path / "bad.json")


def test_ring_dataset_exports_the_six_cycle(tmp_path):
    from pipeline.data_loader import load_dataset

    ds = load_dataset("cyclic_2d")
    raw = extremum_graph(ds.values, knn_graph(ds.points, 20))
    eg = raw.simplify(0.04)
    path = export_extremum_graph(eg, tmp_path / "c.json", coords=ds.points,
                                 meta={"name": ds.name, "simplify_threshold": 0.04},
                                 full_peak_persistence=raw.peak_persistence,
                                 data=points_data(ds.values, ds.points))
    doc = json.loads(path.read_text())
    check_export(doc)
    assert len(doc["peaks"]) == 6 and len(doc["saddles"]) == 6
    assert doc["meta"]["simplify_threshold"] == 0.04
    assert len(doc["data"]["values"]) == ds.num_points == len(doc["region"])


def test_spine_is_exported_and_checked(tmp_path):
    from pipeline.data_loader import load_dataset
    from pipeline.spine_layout import spine_layout

    ds = load_dataset("cyclic_2d")
    eg = extremum_graph(ds.values, knn_graph(ds.points, 20)).simplify(0.04)
    layout = spine_layout(eg, ds.points)
    doc = json.loads(export_extremum_graph(eg, tmp_path / "s.json", coords=ds.points,
                                           spine=layout).read_text())
    check_export(doc)
    assert doc["spine"]["crossings"] == 0 and len(doc["spine"]["peaks"]) == 6

    with pytest.raises(ValueError, match="different graph"):
        extremum_graph_to_dict(five_point_path(), spine=layout)        # 3 peaks, not 6

    doc["spine"]["saddles"].pop()
    with pytest.raises(ValueError):
        check_export(doc)


def test_points_data_embeddings_are_scaled_flat_and_named():
    values = np.zeros(3)
    a = np.array([[10.0, 0.0], [20.0, 5.0], [30.0, 10.0]])       # wide, flat shape
    d = points_data(values, embeddings={"pca": a, "other": a[:, ::-1]})
    assert d["embedding_default"] == "pca" and list(d["embeddings"]) == ["pca", "other"]
    flat = d["embeddings"]["pca"]
    assert flat == [-1.0, -0.5, 0.0, 0.0, 1.0, 0.5]            # [-1, 1], aspect kept
    with pytest.raises(ValueError):
        points_data(values, embeddings={"bad": np.zeros((3, 3))})
