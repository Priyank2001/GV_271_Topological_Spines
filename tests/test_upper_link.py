"""Tests for the upper-link saddles: ``extremum_graph(..., saddles="upper_link")``.

Run from the repo root: python3 -m pytest tests/test_upper_link.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.backends import NumpyBackend  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import NeighborGraph, knn_graph, triangulated_grid_graph  # noqa: E402


def upper_link_pieces(v, rank, g):
    """Brute force: connected pieces of v's higher neighbours (joined when adjacent)."""
    left = {int(u) for u in g.neighbors(v) if rank[u] > rank[v]}
    pieces = []
    while left:
        piece, stack = set(), [left.pop()]
        while stack:
            x = stack.pop()
            piece.add(x)
            for y in g.neighbors(x).tolist():
                if y in left:
                    left.remove(y)
                    stack.append(y)
        pieces.append(piece)
    return pieces


def random_cases():
    rng = np.random.default_rng(5)
    for t in range(24):
        if t % 3 == 0:
            g = knn_graph(rng.random((250, 3)), 7, backend="numpy")
        else:
            g = triangulated_grid_graph(tuple(int(s) for s in rng.integers(3, 8, 2 + t % 2)))
        v = rng.random(g.num_nodes) if t % 2 else rng.integers(0, 4, g.num_nodes).astype(float)
        yield g, v


def test_triangulated_grid_has_the_freudenthal_neighbours():
    for shape, interior in (((5, 5), 6), ((5, 5, 5), 14), ((4, 4, 4, 4), 30)):
        g = triangulated_grid_graph(shape)
        assert g.degree.max() == interior
        coords = np.stack(np.unravel_index(np.arange(g.num_nodes), shape), axis=1)
        step = coords[g.edges[:, 1]] - coords[g.edges[:, 0]]
        assert set(map(tuple, step.tolist())) == {o for o in itertools.product((0, 1), repeat=len(shape))
                                                   if any(o)}


def test_pieces_match_brute_force():
    for g, v in random_cases():
        rank = NumpyBackend().total_order(v)
        eg = extremum_graph(v, g, backend="numpy")              # only for the regions
        region = eg.region
        ul = NumpyBackend().upper_link(rank, region, g)
        got = {}
        for vert, top in zip(ul.vertex.tolist(), ul.top.tolist()):
            got.setdefault(vert, []).append(top)
        want = {}
        for vert in range(g.num_nodes):
            pieces = upper_link_pieces(vert, rank, g)
            higher_elsewhere = any(region[u] != region[vert] for p in pieces for u in p)
            if len(pieces) >= 2 and higher_elsewhere:
                want[vert] = sorted((max(p, key=lambda u: rank[u]) for p in pieces),
                                    key=lambda u: -rank[u])
        assert got == want


def test_same_merge_tree_as_the_passes():
    for g, v in random_cases():
        a = extremum_graph(v, g, backend="numpy", min_loop_relief=0.0)
        b = extremum_graph(v, g, backend="numpy", min_loop_relief=0.0, saddles="upper_link")
        assert np.array_equal(a.peaks, b.peaks) and np.array_equal(a.region, b.region)
        assert np.allclose(a.peak_persistence, b.peak_persistence)
        assert set(a.saddles[a.saddle_is_merge].tolist()) == set(b.saddles[b.saddle_is_merge].tolist())


def test_a_saddle_joining_three_maxima_keeps_all_three_arcs():
    # Vertex 0 has three arms up to peaks A(10), B(9), C(8); the arms do not touch,
    # so 0 is a saddle with three pieces. Its steepest arm leads to A, so 0 lies in
    # A's region. A and C also meet higher up, at w = 7 (5.0), so 0 is not the
    # highest A-C pass and the pass method drops its arc to C.
    #    A=1  a1=2  B=3  b1=4  C=5  c1=6  w=7
    values = np.array([1.0, 10, 2.2, 9, 2.1, 8, 2.0, 5])
    edges = [[0, 2], [0, 4], [0, 6], [2, 1], [4, 3], [6, 5], [7, 1], [7, 5]]
    g = NeighborGraph.from_edges(8, np.array(edges))
    A, B, C = 0, 1, 2                                    # peak ids, highest first
    passes = extremum_graph(values, g, min_loop_relief=0.0).saddle_reach()
    ul = extremum_graph(values, g, min_loop_relief=0.0, saddles="upper_link")
    assert passes[0] == [A, B]
    assert ul.saddle_reach() == {0: [A, B, C], 7: [A, C]}
    assert (ul.saddles == 0).sum() == 2                  # 3 maxima = 2 entries at one vertex


def ring(m, n=121, R=2.5, a=0.5):
    """-(r - R)^2 + a (r/R)^m cos(m theta) on a triangulated grid: m peaks and m saddles on the ring."""
    x, y = np.meshgrid(np.linspace(-5, 5, n), np.linspace(-5, 5, n), indexing="ij")
    r, th = np.hypot(x, y), np.arctan2(y, x)
    f = -(r - R) ** 2 + a * (r / R) ** m * np.cos(m * th)
    return f.ravel(), triangulated_grid_graph(x.shape), np.stack([x.ravel(), y.ravel()], 1)


def test_two_saddles_between_the_same_two_peaks():
    f, g, xy = ring(2)
    a = extremum_graph(f, g)
    b = extremum_graph(f, g, saddles="upper_link")
    assert a.num_peaks == b.num_peaks == 2
    assert a.num_saddles == 1                            # the pass method: one per pair
    assert b.num_saddles == 2 and (b.saddle_peaks == [0, 1]).all()
    ys = np.sort(xy[b.saddles, 1])                       # one at the top of the ring, one at the bottom
    assert ys[0] == pytest.approx(-2.5, abs=0.2) and ys[1] == pytest.approx(2.5, abs=0.2)


def test_parallel_saddles_off_keeps_the_highest_per_pair_and_the_persistence():
    f, g, _ = ring(2)
    both = extremum_graph(f, g, saddles="upper_link")
    one = extremum_graph(f, g, saddles="upper_link", parallel_saddles=False)
    assert one.num_saddles == 1 and one.saddles[0] == both.saddles[0]
    for g2, v in random_cases():
        a = extremum_graph(v, g2, saddles="upper_link", min_loop_relief=0.0)
        b = extremum_graph(v, g2, saddles="upper_link", min_loop_relief=0.0, parallel_saddles=False)
        assert np.allclose(a.peak_persistence, b.peak_persistence)
        assert len({tuple(p) for p in b.saddle_peaks.tolist()}) == b.num_saddles


def test_a_slope_is_not_a_saddle():
    x, y = np.meshgrid(np.arange(20.0), np.arange(15.0), indexing="ij")
    f = (0.3 * x + y).ravel()
    g = triangulated_grid_graph(x.shape)
    eg = extremum_graph(f, g, saddles="upper_link")
    assert eg.num_peaks == 1 and eg.num_saddles == 0


def test_ackley_3d_true_links_and_far_fewer_false_ones():
    # -Ackley on [-1.5, 1.5]^3: 27 maxima on the lattice {-1, 0, 1}^3, linked only to
    # lattice neighbours (Correa et al. 2011, Fig. 13a). Same graph for both methods.
    d, res = 3, 37
    ax = np.linspace(-1.5, 1.5, res)
    X = np.stack(np.meshgrid(*[ax] * d, indexing="ij"), -1).reshape(-1, d)
    f = 20 * np.exp(-0.2 * np.sqrt((X ** 2).mean(1))) + np.exp(np.cos(2 * np.pi * X).mean(1))
    g = triangulated_grid_graph((res,) * d)
    lattice = np.rint(X).astype(int)

    def links(eg):
        p = np.sort(eg.peak_persistence)[::-1]
        s = eg.simplify((p[26] + p[27]) / 2)
        L = [tuple(v) for v in lattice[s.peaks].tolist()]
        assert len(set(L)) == 27
        pairs = {tuple(sorted((L[i], L[j]))) for i, j in s.saddle_peaks.tolist()}
        true = {q for q in pairs if sum(abs(u - v) for u, v in zip(*q)) == 1}
        return len(true), len(pairs) - len(true)

    true_pass, extra_pass = links(extremum_graph(f, g))
    true_ul, extra_ul = links(extremum_graph(f, g, saddles="upper_link"))
    assert true_pass == true_ul == 54
    assert extra_ul * 4 < extra_pass


def test_unknown_saddle_method_is_refused():
    with pytest.raises(ValueError):
        extremum_graph(np.zeros(3), NeighborGraph.from_edges(3, np.array([[0, 1]])), saddles="morse")
