"""Tests for anyeg/persistence.py: persistence and p* of AnyEG's extremum graph.

Run from the repo root: python3 -m pytest tests/test_anyeg_persistence.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")

from anyeg import ArcAssembler, NeighborTable, TorchKernels, compute_persistence  # noqa: E402
from pipeline import extremum_graph as ours  # noqa: E402
from pipeline.backends import NumpyBackend  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402

CPU = TorchKernels("cpu")


def anyeg_graph(values, table):
    """AnyEG's graph on the (value, index) ranks, as everywhere in this repo."""
    rank = NumpyBackend().total_order(np.asarray(values, dtype=np.float64))
    distinct = rank.astype(np.float32)
    g = ArcAssembler().assemble(distinct, table, CPU.run(distinct, table), heights=values)
    return g, rank


def table_from_edges(n, edges):
    nbrs = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    indptr = np.r_[0, np.cumsum([len(x) for x in nbrs])]
    return NeighborTable.from_csr(indptr, np.concatenate(nbrs).astype(np.int32))


def cloud(seed, n=1500, d=3):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(n, d))
    f = np.sin(5 * pts[:, 0]) * np.cos(4 * pts[:, 1]) + 0.3 * rng.normal(size=n)
    return pts, f


# ---- small cases by hand ------------------------------------------------------
def test_ridge_through_a_merged_peak():
    # A B C D E = 9 4 7 3 8 on a path (the pipeline's test of the same name).
    values = np.array([9, 4, 7, 3, 8.0])
    g, rank = anyeg_graph(values, table_from_edges(5, [(i, i + 1) for i in range(4)]))
    p = compute_persistence(g, values, rank)
    assert p.peaks.tolist() == [0, 4, 2]
    assert p.peak_persistence.tolist() == [np.inf, 5.0, 3.0]
    assert p.saddles.tolist() == [1, 3]
    assert p.saddle_peaks.tolist() == [[0, 2], [1, 2]]
    assert p.saddle_persistence.tolist() == [3.0, 5.0]
    assert p.saddle_saturation.tolist() == [5.0, 5.0]       # B carries D's ridge to A
    assert p.peak_parent.tolist() == [-1, 0, 0]             # C is gone first, so E's flow runs on to A
    assert p.arc_mask(4.0).tolist() == [True, True]
    assert p.surviving_peaks(4.0).tolist() == [True, True, False]


def test_a_saddle_reaching_three_maxima_gives_two_rows():
    # A star: the centre (1.0) is below three arms (5, 4, 3) that do not touch.
    values = np.array([1.0, 5.0, 4.0, 3.0])
    g, rank = anyeg_graph(values, table_from_edges(4, [(0, 1), (0, 2), (0, 3)]))
    p = compute_persistence(g, values, rank)
    assert p.saddles.tolist() == [0, 0]
    assert p.saddle_peaks.tolist() == [[0, 1], [0, 2]]
    assert p.saddle_is_merge.tolist() == [True, True]
    assert p.peak_persistence.tolist() == [np.inf, 3.0, 2.0]
    assert p.peak_parent.tolist() == [-1, 0, 0]
    assert p.saddle_saturation.tolist() == [3.0, 2.0]


def test_default_order_is_the_height_rank():
    values = np.array([9, 4, 7, 3, 8.0])
    g, rank = anyeg_graph(values, table_from_edges(5, [(i, i + 1) for i in range(4)]))
    a, b = compute_persistence(g, values), compute_persistence(g, values, rank)
    assert np.array_equal(a.saddle_saturation, b.saddle_saturation)


def test_bad_arguments_are_rejected():
    values = np.array([9, 4, 7.0])
    g, _ = anyeg_graph(values, table_from_edges(3, [(0, 1), (1, 2)]))
    with pytest.raises(ValueError, match="links"):
        compute_persistence(g, values, links="downhill")
    with pytest.raises(ValueError, match="heights"):
        compute_persistence(g, values[:2])


# ---- against the pipeline ------------------------------------------------------
@pytest.mark.parametrize("seed,k", [(0, 6), (1, 10), (2, 15)])
def test_uphill_links_give_exactly_the_merge_tree(seed, k):
    pts, f = cloud(seed)
    graph = knn_graph(pts, k)
    g, rank = anyeg_graph(f, NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32)))
    p = compute_persistence(g, f, rank, links="uphill")
    eg = ours.extremum_graph(f, graph, min_loop_relief=0.0, saddles="passes")
    assert np.array_equal(p.peaks, eg.peaks)
    assert np.allclose(p.peak_persistence, eg.peak_persistence)


@pytest.mark.parametrize("seed,k", [(0, 6), (1, 10), (2, 15)])
def test_all_links_never_raise_a_peaks_persistence(seed, k):
    # AnyEG's downhill arcs only add links, so groups merge earlier or at the same saddle.
    pts, f = cloud(seed)
    graph = knn_graph(pts, k)
    g, rank = anyeg_graph(f, NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32)))
    every = compute_persistence(g, f, rank, links="all")
    up = compute_persistence(g, f, rank, links="uphill")
    assert np.array_equal(every.peaks, up.peaks)
    assert (every.peak_persistence <= up.peak_persistence + 1e-12).all()


@pytest.mark.parametrize("links", ["all", "uphill"])
@pytest.mark.parametrize("seed", [0, 3])
def test_same_numbers_as_the_pipelines_sweep_and_saturation(seed, links):
    # anyeg/ may not import pipeline/, so the code is repeated; it must agree exactly.
    pts, f = cloud(seed, n=2500)
    graph = knn_graph(pts, 10)
    g, rank = anyeg_graph(f, NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32)))
    p = compute_persistence(g, f, rank, links=links)
    sweep = ours._sweep(f, p.peaks, p.saddles, p.saddle_peaks)
    assert np.array_equal(p.peak_persistence, sweep["peak"])
    assert np.array_equal(p.saddle_is_merge, sweep["is_merge"])
    assert np.array_equal(p.saddle_persistence, sweep["saddle"])
    sat = ours._saturation(f[p.peaks], f[p.saddles], p.saddle_peaks, sweep["saddle"], sweep["is_merge"])
    assert np.array_equal(p.saddle_saturation, sat)
    par = ours._cancellations(p.num_peaks, p.saddle_peaks, sweep["saddle"], sweep["is_merge"])["parent"]
    assert np.array_equal(p.peak_parent, par)
    assert (p.saddle_saturation >= p.saddle_persistence).all()


def test_rows_cover_every_maximum_pair_anyeg_draws():
    pts, f = cloud(4)
    graph = knn_graph(pts, 10)
    g, rank = anyeg_graph(f, NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32)))
    p = compute_persistence(g, f, rank)
    vertex = p.peaks
    # each AnyEG saddle reaching k maxima gives k - 1 rows, all from its highest maximum
    reached = {}
    for s, m in g.links():
        reached.setdefault(s, set()).add(m)
    assert p.num_rows == sum(len(m) - 1 for m in reached.values())
    for s, (a, b) in zip(p.saddles.tolist(), p.saddle_peaks.tolist()):
        assert {int(vertex[a]), int(vertex[b])} <= reached[s]
        assert int(vertex[a]) == max(reached[s], key=lambda m: rank[m])
