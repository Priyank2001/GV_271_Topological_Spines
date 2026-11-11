"""Tests for pipeline/spine_layout.py. Run from the repo root: python3 -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.spine_layout as sl
from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import knn_graph
from pipeline.spine_layout import (_pairwise_distances, _smacof, count_crossings,
                                   pca_layout, spine_layout, stress)


def test_pca_of_a_flat_cloud_keeps_all_distances():
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(30, 2)) * [5.0, 2.0]
    basis = np.linalg.qr(rng.normal(size=(10, 2)))[0]          # a plane inside 10D
    X = plane @ basis.T + 3.0
    xy = pca_layout(X)
    assert stress(xy, _pairwise_distances(X)) < 1e-9


def test_pca_pads_when_there_is_only_one_axis_or_one_point():
    assert pca_layout(np.array([[1.0, 2.0, 3.0]])).tolist() == [[0.0, 0.0]]
    line = pca_layout(np.array([[0.0], [1.0], [3.0]]))
    assert line.shape == (3, 2) and np.allclose(line[:, 1], 0)


def test_smacof_recovers_a_planar_configuration_from_a_bad_start():
    rng = np.random.default_rng(1)
    truth = rng.uniform(-5, 5, size=(25, 2))
    D = _pairwise_distances(truth)
    start = truth + rng.normal(scale=1.0, size=truth.shape)
    xy, _ = _smacof(start, D, max_iter=500, tol=1e-10)
    assert stress(start, D) > 0.1
    assert stress(xy, D) < 1e-3


def test_refinement_improves_on_pca_for_a_curved_cloud():
    # A helix cannot be flattened without error; stress majorisation should
    # still beat the straight PCA projection.
    t = np.linspace(0, 4 * np.pi, 40)
    X = np.stack([np.cos(t), np.sin(t), 0.5 * t], axis=1)
    D = _pairwise_distances(X)
    start = pca_layout(X)
    xy, _ = _smacof(start, D, max_iter=300, tol=1e-9)
    assert stress(xy, D) < stress(start, D)


def test_tiny_and_degenerate_inputs():
    D1 = np.zeros((1, 1))
    assert _smacof(np.zeros((1, 2)), D1, 10, 1e-6)[0].shape == (1, 2)
    D2 = np.array([[0.0, 4.0], [4.0, 0.0]])
    xy, _ = _smacof(np.zeros((2, 2)), D2, 10, 1e-6)
    assert np.linalg.norm(xy[0] - xy[1]) == pytest.approx(4.0)
    # Three nodes at the same place: no NaNs, no crash.
    same = np.ones((3, 5))
    xy, _ = _smacof(pca_layout(same), _pairwise_distances(same), 50, 1e-6)
    assert np.isfinite(xy).all()


def test_crossings():
    xy = np.array([[0, 0], [1, 1], [0, 1], [1, 0.0], [2, 2], [3, 3]])
    assert count_crossings(xy, np.array([[0, 1], [2, 3]])) == 1        # an X
    assert count_crossings(xy, np.array([[0, 1], [1, 4]])) == 0        # share node 1
    assert count_crossings(xy, np.array([[0, 3], [2, 1]])) == 0        # parallel
    assert count_crossings(xy, np.array([[0, 1]])) == 0
    # one link crossed by two others, which do not cross each other
    xy2 = np.array([[0, 0], [10, 0], [2, -1], [2, 1], [7, -1], [7, 1.0]])
    assert count_crossings(xy2, np.array([[0, 1], [2, 3], [4, 5]])) == 2


def test_crossing_count_is_the_same_when_chunked(monkeypatch):
    rng = np.random.default_rng(3)
    xy = rng.uniform(size=(40, 2))
    links = rng.integers(0, 40, size=(60, 2))
    links = links[links[:, 0] != links[:, 1]]
    monkeypatch.setattr(sl, "_CHUNK_PAIRS", 200)           # about 3 rows per chunk
    full = count_crossings(xy, links)
    # Brute-force reference with plain loops.
    ref = 0
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            a, b = links[i]
            c, d = links[j]
            if len({a, b, c, d}) < 4:
                continue
            o = sl._orient
            if o(xy[a], xy[b], xy[c]) * o(xy[a], xy[b], xy[d]) < 0 and \
               o(xy[c], xy[d], xy[a]) * o(xy[c], xy[d], xy[b]) < 0:
                ref += 1
    assert full == ref


def ring_graph(name="cyclic_3d", k=20):
    from pipeline.data_loader import load_dataset

    ds = load_dataset(name)
    return ds, extremum_graph(ds.values, knn_graph(ds.points, k)).simplify(0.04)


@pytest.mark.parametrize("name", ["cyclic_2d", "cyclic_3d"])
def test_ring_spine_is_a_clean_hexagon(name):
    ds, eg = ring_graph(name)
    layout = spine_layout(eg, ds.points)
    assert len(layout.peak_xy) == 6 and len(layout.saddle_xy) == 6
    assert layout.crossings == 0
    assert layout.stress <= layout.pca_stress + 1e-12
    assert layout.stress < 0.1
    allxy = np.concatenate([layout.peak_xy, layout.saddle_xy])
    assert np.abs(allxy).max() == pytest.approx(1.0)
    # layout distance times scale is close to the data distance of a linked pair
    p, s = (int(v) for v in eg.edges[0])
    want = np.linalg.norm(ds.points[eg.peaks[p]] - ds.points[eg.saddles[s]])
    got = np.linalg.norm(layout.peak_xy[p] - layout.saddle_xy[s]) * layout.scale
    assert got == pytest.approx(want, rel=0.2)


def test_layout_dict_is_json_ready():
    import json

    ds, eg = ring_graph("cyclic_2d")
    d = spine_layout(eg, ds.points).to_dict()
    json.dumps(d, allow_nan=False)
    assert len(d["peaks"]) == 6 and len(d["saddles"]) == 6 and d["method"] == "pca+stress"


def test_single_peak_graph_and_spacing():
    rng = np.random.default_rng(4)
    pts = rng.uniform(size=(200, 2))
    eg = extremum_graph(-np.linalg.norm(pts - 0.5, axis=1), knn_graph(pts, 8)).simplify(10.0)
    layout = spine_layout(eg, pts)
    assert layout.peak_xy.shape == (1, 2) and layout.saddle_xy.shape == (0, 2)
    assert layout.crossings == 0

    ds, eg = ring_graph("cyclic_2d")
    flat = spine_layout(eg, ds.points, spacing=np.array([1.0, 1.0]))
    wide = spine_layout(eg, ds.points, spacing=np.array([4.0, 1.0]))
    assert wide.scale > flat.scale                                      # stretched x axis


def test_too_many_nodes_is_refused(monkeypatch):
    ds, eg = ring_graph("cyclic_2d")
    monkeypatch.setattr(sl, "MAX_NODES", 5)
    with pytest.raises(ValueError, match="simplify"):
        spine_layout(eg, ds.points)
