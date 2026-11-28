"""Brute-force checks of the theorems in docs/extremum_graph_correctness.md.

Each test restates one theorem and checks it on random graphs against a
direct, slow computation that shares no code with pipeline/extremum_graph.py.
Run from the repo root: python3 -m pytest tests/test_extremum_graph_proof.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import grid_graph, knn_graph


# ---------------------------------------------------------------------------
# Random inputs: kNN graphs (often disconnected for small k) and grids, with
# continuous values and with heavy ties (integers in a small range).
# ---------------------------------------------------------------------------
def _cases():
    for seed in range(12):
        for k in (3, 5, 8):
            for ties in (False, True):
                yield seed, k, ties


def _make(seed, k, ties):
    rng = np.random.default_rng(seed)
    if k == 8 and seed % 3 == 0:                       # a few grids as well
        g = grid_graph((9, 8))
    else:
        g = knn_graph(rng.normal(size=(70, 3)), k=k)
    n = g.num_nodes
    values = rng.integers(0, 12, n).astype(float) if ties else rng.normal(size=n)
    return values, g


def _rank(values):
    """Position in the order by (value, index), the order the proof uses."""
    order = np.lexsort((np.arange(values.shape[0]), values))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0])
    return rank


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x


def _components(g, keep):
    """Component label of each vertex of G[keep] (-1 outside ``keep``)."""
    uf = _UF(g.num_nodes)
    for u, v in g.edges:
        if keep[u] and keep[v]:
            uf.p[uf.find(u)] = uf.find(v)
    return np.array([uf.find(v) if keep[v] else -1 for v in range(g.num_nodes)])


def _brute_persistence(values, g):
    """0-dim persistence of the superlevel filtration, vertex by vertex, elder rule.

    Returns {maximum vertex: persistence} and a Counter {vertex: number of
    components that die there}.
    """
    rank = _rank(values)
    uf, top = _UF(g.num_nodes), {}
    added = np.zeros(g.num_nodes, bool)
    pers, merges = {}, Counter()
    for v in np.argsort(-rank):
        roots = {uf.find(u) for u in g.neighbors(v) if added[u]}
        added[v] = True
        if not roots:                                  # v is a maximum: a birth
            top[v] = v
            pers[v] = np.inf
            continue
        tops = sorted((top[r] for r in roots), key=lambda m: -rank[m])
        for m in tops[1:]:                             # all but the eldest die at v
            pers[m] = values[m] - values[v]
        if len(tops) > 1:
            merges[v] += len(tops) - 1
        for r in roots:
            uf.p[r] = v
        top[v] = tops[0]
    return pers, merges


# ---------------------------------------------------------------------------
# Theorem 1: every saddle joins exactly two distinct maxima, and no pair twice.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,k,ties", list(_cases()))
def test_theorem1_every_saddle_has_degree_two(seed, k, ties):
    values, g = _make(seed, k, ties)
    eg = extremum_graph(values, g, min_loop_relief=0.0, saddles="passes")
    graphs = [eg] + [eg.simplify(t) for t in np.quantile(np.abs(values), [0.1, 0.3, 0.6])]
    for h in graphs:
        sp = h.saddle_peaks
        assert sp.shape == (h.num_saddles, 2)
        assert (sp[:, 0] != sp[:, 1]).all()
        assert len({tuple(r) for r in np.sort(sp, axis=1)}) == h.num_saddles
        assert h.edges.shape == (2 * h.num_saddles, 2)


# ---------------------------------------------------------------------------
# Theorem 2: the saddle of (A, B) is the bottleneck of the best path from
# max A to max B inside A u B.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,k,ties", list(_cases()))
def test_theorem2_saddle_is_the_bottleneck_inside_the_two_regions(seed, k, ties):
    values, g = _make(seed, k, ties)
    rank = _rank(values)
    eg = extremum_graph(values, g, min_loop_relief=0.0, saddles="passes")
    for s, (a, b) in zip(eg.saddles, eg.saddle_peaks):
        inside = (eg.region == a) | (eg.region == b)
        ma, mb = eg.peaks[a], eg.peaks[b]
        # Add the vertices of A u B from the top down; the vertex whose arrival
        # first connects ma and mb is the bottleneck of the best path.
        uf, added = _UF(g.num_nodes), np.zeros(g.num_nodes, bool)
        bottleneck = None
        for v in np.argsort(-rank):
            if not inside[v]:
                continue
            added[v] = True
            for u in g.neighbors(v):
                if added[u]:
                    uf.p[uf.find(u)] = uf.find(v)
            if added[ma] and added[mb] and uf.find(ma) == uf.find(mb):
                bottleneck = v
                break
        assert bottleneck == s


# ---------------------------------------------------------------------------
# Theorem 3: components of the superlevel set G_t <-> components of the
# region graph restricted to saddles at or above t.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,k,ties", list(_cases()))
def test_theorem3_superlevel_components_match_the_region_graph(seed, k, ties):
    values, g = _make(seed, k, ties)
    rank = _rank(values)
    eg = extremum_graph(values, g, min_loop_relief=0.0, saddles="passes")
    peak_rank, sad_rank = rank[eg.peaks], rank[eg.saddles]
    for t in range(0, g.num_nodes, 3):
        comp = _components(g, rank >= t)
        alive = np.flatnonzero(peak_rank >= t)
        uf = _UF(eg.num_peaks)
        for s in np.flatnonzero(sad_rank >= t):
            a, b = eg.saddle_peaks[s]
            uf.p[uf.find(a)] = uf.find(b)
        # Every component of G_t holds a maximum ...
        assert set(comp[rank >= t]) == {comp[eg.peaks[p]] for p in alive}
        # ... and two maxima share a component of G_t iff they share one of H_t.
        for i in alive:
            for j in alive:
                assert (comp[eg.peaks[i]] == comp[eg.peaks[j]]) == (uf.find(i) == uf.find(j))


# ---------------------------------------------------------------------------
# Theorem 4: peak persistence = 0-dim persistence of the superlevel filtration,
# and the merge saddles are exactly the vertices where components merge.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,k,ties", list(_cases()))
def test_theorem4_sweep_is_superlevel_persistence(seed, k, ties):
    values, g = _make(seed, k, ties)
    want_pers, want_merges = _brute_persistence(values, g)
    for relief in (0.0, 0.5):                          # the loop filter never drops a merge
        eg = extremum_graph(values, g, min_loop_relief=relief, saddles="passes")
        got = dict(zip(eg.peaks.tolist(), eg.peak_persistence.tolist()))
        assert got.keys() == want_pers.keys()
        for m, p in want_pers.items():
            assert got[m] == pytest.approx(p) if np.isfinite(p) else np.isinf(got[m])
        assert Counter(eg.saddles[eg.saddle_is_merge].tolist()) == want_merges
        assert np.isinf(eg.peak_persistence).sum() == g.num_components()


# ---------------------------------------------------------------------------
# Corollary 5: a merge saddle's upper link is disconnected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed,k,ties", list(_cases()))
def test_corollary5_merge_saddles_have_a_disconnected_upper_link(seed, k, ties):
    values, g = _make(seed, k, ties)
    rank = _rank(values)
    eg = extremum_graph(values, g, saddles="passes")
    for s in set(eg.saddles[eg.saddle_is_merge].tolist()):
        upper = np.zeros(g.num_nodes, bool)
        upper[[u for u in g.neighbors(s) if rank[u] > rank[s]]] = True
        comp = _components(g, upper)
        assert len(set(comp[upper])) >= 2
