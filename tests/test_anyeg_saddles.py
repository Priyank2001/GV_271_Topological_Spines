"""Tests for pipeline/anyeg_saddles.py: our extremum graph with AnyEG's saddles.

Run from the repo root: python3 -m pytest tests/test_anyeg_saddles.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")

from anyeg import FreudenthalGridNeighborhood, ScalarField  # noqa: E402
from anyeg.kernels import TorchKernels  # noqa: E402
from anyeg.neighborhood import NeighborTable  # noqa: E402
from pipeline.backends import NumpyBackend  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import freudenthal_grid_graph, knn_graph, ttk_grid_graph  # noqa: E402


def random_cloud(seed, n=1500, d=3):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(n, d))
    f = np.sin(5 * pts[:, 0]) * np.cos(4 * pts[:, 1]) + 0.3 * rng.normal(size=n)
    return pts, f


def anyeg_saddle_vertices(values, graph):
    rank = NumpyBackend().total_order(values)
    table = NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32))
    return set(TorchKernels("cpu").run(rank.astype(np.float32), table).saddles.tolist())


@pytest.mark.parametrize("seed,k", [(0, 6), (1, 10), (2, 20)])
def test_every_saddle_is_one_of_anyegs_and_every_pair_is_kept_once(seed, k):
    pts, f = random_cloud(seed)
    g = knn_graph(pts, k)
    eg = extremum_graph(f, g, min_loop_relief=0.0)
    assert set(eg.saddles.tolist()) <= anyeg_saddle_vertices(f, g)
    pairs = [tuple(p) for p in eg.saddle_peaks.tolist()]
    assert len(pairs) == len(set(pairs))                          # one saddle per pair of peaks
    assert (eg.saddle_peaks[:, 0] < eg.saddle_peaks[:, 1]).all()  # [higher, lower]
    rank = NumpyBackend().total_order(f)
    assert (np.diff(rank[eg.saddles]) <= 0).all()                 # highest first


@pytest.mark.parametrize("seed,k", [(0, 6), (1, 10), (2, 20)])
def test_both_peaks_are_reached_by_higher_neighbours(seed, k):
    pts, f = random_cloud(seed)
    g = knn_graph(pts, k)
    eg = extremum_graph(f, g, min_loop_relief=0.0)
    for s, (a, b) in zip(eg.saddles, eg.saddle_peaks):
        nb = g.neighbors(s)
        reached = set(eg.region[nb[f[nb] > f[s]]].tolist())
        assert {int(a), int(b)} <= reached


@pytest.mark.parametrize("seed,k", [(0, 6), (1, 10), (3, 15)])
def test_merge_tree_is_the_same_as_with_highest_passes(seed, k):
    # Every merge saddle passes AnyEG's test, so peak persistence (the merge tree) is unchanged.
    pts, f = random_cloud(seed)
    g = knn_graph(pts, k)
    new = extremum_graph(f, g, min_loop_relief=0.0)
    old = extremum_graph(f, g, min_loop_relief=0.0, saddles="passes")
    assert np.array_equal(new.peaks, old.peaks)
    assert np.allclose(new.peak_persistence, old.peak_persistence)
    assert sorted(new.saddles[new.saddle_is_merge].tolist()) == sorted(old.saddles[old.saddle_is_merge].tolist())


def test_ttk_grid_graph_is_anyegs_14_neighbour_grid():
    for shape in ((4, 5, 6), (3, 3, 7)):
        nz, ny, nx = shape
        table = FreudenthalGridNeighborhood().build(ScalarField(np.zeros(nx * ny * nz, np.float32),
                                                                dims=(nx, ny, nz)))
        ref = {tuple(sorted(e)) for e in table.edges().tolist()}
        assert {tuple(e) for e in ttk_grid_graph(shape).edges.tolist()} == ref
        assert {tuple(e) for e in freudenthal_grid_graph(shape).edges.tolist()} != ref


def test_unknown_saddle_method_is_rejected():
    pts, f = random_cloud(0, n=50)
    with pytest.raises(ValueError):
        extremum_graph(f, knn_graph(pts, 5), saddles="nope")
