"""
Two method-independent yardsticks for an extremum graph of a field on a graph.

Comparing AnyEG with our pipeline only by how much they agree says nothing
about which one is right where they differ. These two checks need neither
method, and each is the natural test for one of them, so both are reported:

1. **Merge tree (0-dimensional persistence) of the superlevel sets.** Sweep the
   field from the top down; a maximum is born when it appears, and dies at the
   saddle where its component joins one with a higher maximum (elder rule).
   This pairing is unique once the order is strict, and every extremum graph
   must encode it: replaying the graph's own links from the highest saddle
   down (``graph_persistence``) has to give every maximum the same death.
   A wrong link (joining two maxima whose superlevel components are still
   apart at that saddle) makes a maximum die too *early*; a missing merge
   saddle makes it die too *late* (or never).
   Computed here with a maximum spanning forest (SciPy) and a union-find over
   its N-1 edges, independent of both pipelines.

2. **PL saddle validity** (triangulated grids only). On the 14-neighbour
   Freudenthal grid the field is piecewise linear, and a vertex can be a
   saddle between maxima only if its *upper link* (neighbours higher than it,
   joined when they are neighbours of each other) has two or more pieces.
   ``pl_saddle_valid`` checks each reported saddle. The 6-neighbour grid is
   not a triangulation, so the test has no meaning there.

Our pipeline reproduces (1) by construction (its saddles are the merge tree's
passes) and AnyEG passes (2) by construction (its module 4 is this test), so
each yardstick mainly tells something about the *other* method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

ROOT = -1    # "death" of a maximum that never dies (top of a component)


# ---------------------------------------------------------------------------
# 1. Merge tree
# ---------------------------------------------------------------------------
def merge_tree_pairs(order: np.ndarray, edges: np.ndarray) -> Dict[int, int]:
    """{maximum vertex: rank of its death saddle} for the superlevel sweep (ROOT if none).

    ``order[v]`` is the strict rank of vertex v, ``edges`` an (E, 2) array of
    undirected graph edges.
    """
    order = np.asarray(order, dtype=np.int64)
    n = order.shape[0]
    a, b = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    level = np.minimum(order[a], order[b])               # an edge exists from this level down
    # Maximum spanning forest by level = minimum spanning forest by n - level (> 0).
    w = (n - level).astype(np.float64)
    mst = minimum_spanning_tree(coo_matrix((w, (a, b)), shape=(n, n)).tocsr()).tocoo()
    ta, tb = mst.row.astype(np.int64), mst.col.astype(np.int64)
    tlevel = np.minimum(order[ta], order[tb])
    by_level = np.argsort(-tlevel, kind="stable")

    parent = np.arange(n)
    top = np.arange(n)                                   # highest vertex of each component
    death: Dict[int, int] = {}
    for e in by_level.tolist():
        ra, rb = _find(parent, int(ta[e])), _find(parent, int(tb[e]))
        if ra == rb:
            continue
        lvl = int(tlevel[e])
        old, young = (ra, rb) if order[top[ra]] > order[top[rb]] else (rb, ra)
        if order[top[young]] != lvl:                     # not a regular vertex joining: a true pair
            death[int(top[young])] = lvl
        parent[young] = old
    has_higher = np.zeros(n, dtype=bool)
    has_higher[a[order[b] > order[a]]] = True
    has_higher[b[order[a] > order[b]]] = True
    for m in np.flatnonzero(~has_higher).tolist():
        death.setdefault(m, ROOT)
    return death


def graph_persistence(maxima: np.ndarray, links: np.ndarray, order: np.ndarray) -> Dict[int, int]:
    """The same pairing, replayed from a method's own graph.

    ``links`` is (L, 3): (saddle vertex, maximum, maximum). Links are taken from
    the highest saddle down; where two groups of maxima first meet, the group
    with the lower top dies at that saddle's rank.
    """
    order = np.asarray(order, dtype=np.int64)
    ids = {int(m): i for i, m in enumerate(np.asarray(maxima).tolist())}
    parent = np.arange(len(ids))
    top = np.asarray(maxima, dtype=np.int64).copy()
    death = {int(m): ROOT for m in ids}
    if len(links):
        links = np.asarray(links, dtype=np.int64)
        for s, m1, m2 in links[np.argsort(-order[links[:, 0]], kind="stable")].tolist():
            r1, r2 = _find(parent, ids[m1]), _find(parent, ids[m2])
            if r1 == r2:
                continue
            old, young = (r1, r2) if order[top[r1]] > order[top[r2]] else (r2, r1)
            death[int(top[young])] = int(order[s])
            parent[young] = old
    return death


@dataclass
class MergeTreeFidelity:
    reference_pairs: int          # maxima that die (all but one per connected component)
    exact: int                    # ... with exactly the reference death saddle
    early: int                    # die too high: some link joins maxima still apart there
    late: int                     # die too low, or never (missing merge saddle)
    significant_pairs: int        # reference pairs with persistence >= threshold (in heights)
    significant_exact: int
    components_reference: int
    components_graph: int
    missing_maxima: int           # reference maxima the method does not report

    @property
    def exact_fraction(self) -> float:
        return self.exact / self.reference_pairs if self.reference_pairs else 1.0

    @property
    def significant_fraction(self) -> float:
        return self.significant_exact / self.significant_pairs if self.significant_pairs else 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(exact_fraction=self.exact_fraction, significant_fraction=self.significant_fraction)
        return d


def merge_tree_fidelity(reference: Dict[int, int], derived: Dict[int, int], order: np.ndarray,
                        heights: np.ndarray, significant: float) -> MergeTreeFidelity:
    """Compare a method's pairing (``graph_persistence``) with the reference.

    ``significant`` is a persistence threshold in the units of ``heights``
    (the real values), used to report the pairs a user would keep.
    """
    vertex_at_rank = np.empty_like(order)
    vertex_at_rank[order] = np.arange(order.shape[0])
    exact = early = late = sig = sig_exact = 0
    pairs = 0
    for m, d_ref in reference.items():
        if d_ref == ROOT:
            continue
        pairs += 1
        d = derived.get(m, ROOT)
        ok = d == d_ref
        exact += ok
        if not ok:
            if d != ROOT and d > d_ref:
                early += 1
            else:
                late += 1
        if heights[m] - heights[vertex_at_rank[d_ref]] >= significant:
            sig += 1
            sig_exact += ok
    return MergeTreeFidelity(
        reference_pairs=pairs, exact=exact, early=early, late=late,
        significant_pairs=sig, significant_exact=sig_exact,
        components_reference=sum(d == ROOT for d in reference.values()),
        components_graph=sum(d == ROOT for d in derived.values()),
        missing_maxima=sum(m not in derived for m in reference),
    )


# ---------------------------------------------------------------------------
# 2. PL saddle validity
# ---------------------------------------------------------------------------
def upper_link_pieces(saddles: np.ndarray, table: np.ndarray, order: np.ndarray,
                      grid_dims: Tuple[int, int, int],
                      offsets: Sequence[Tuple[int, int, int]]) -> np.ndarray:
    """Number of connected pieces of each vertex's upper link on a stencil grid.

    ``table`` is the (N, M) neighbour table (-1 = none), ``grid_dims`` (nx, ny, nz)
    with x fastest, ``offsets`` the stencil: two link vertices are joined when
    their grid offset is in the stencil (for the Freudenthal stencil this is
    exactly "they span a triangle with the vertex").
    """
    saddles = np.asarray(saddles, dtype=np.int64)
    if saddles.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    nx, ny, _ = grid_dims
    nb = table[saddles].astype(np.int64)                          # (S, M)
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    upper = valid & (order[safe] > order[saddles][:, None])
    xyz = np.stack([safe % nx, (safe // nx) % ny, safe // (nx * ny)], axis=-1)  # (S, M, 3)
    diff = xyz[:, :, None, :] - xyz[:, None, :, :]                 # (S, M, M, 3)
    in_cube = (np.abs(diff) <= 1).all(axis=-1)
    code = ((diff + 1) * np.array([9, 3, 1])).sum(axis=-1)
    stencil = np.zeros(27, dtype=bool)
    for dx, dy, dz in offsets:
        stencil[(dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)] = True
        stencil[(-dx + 1) * 9 + (-dy + 1) * 3 + (-dz + 1)] = True
    adjacent = in_cube & stencil[np.clip(code, 0, 26)]
    adjacent &= upper[:, :, None] & upper[:, None, :]
    return _pieces(adjacent, upper)


def _pieces(adjacent: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Connected pieces per row of a batch of small graphs (transitive closure by squaring)."""
    s, m, _ = adjacent.shape
    reach = adjacent | (np.eye(m, dtype=bool)[None] & members[:, :, None])
    for _ in range(int(np.ceil(np.log2(max(m, 2)))) + 1):
        reach = np.einsum("sij,sjk->sik", reach.astype(np.uint8), reach.astype(np.uint8)) > 0
    smaller = np.tril(np.ones((m, m), dtype=bool), k=-1)
    reached_by_smaller = (reach & smaller[None]).any(axis=2)
    return (members & ~reached_by_smaller).sum(axis=1)


def pl_saddle_valid(saddles: np.ndarray, table: np.ndarray, order: np.ndarray,
                    grid_dims: Tuple[int, int, int], offsets,
                    chunk: int = 20000) -> Optional[np.ndarray]:
    """Boolean per saddle: its upper link has two or more pieces."""
    saddles = np.asarray(saddles, dtype=np.int64)
    out = [upper_link_pieces(saddles[i:i + chunk], table, order, grid_dims, offsets) >= 2
           for i in range(0, saddles.shape[0], chunk)]
    return np.concatenate(out) if out else np.zeros(0, dtype=bool)


def _find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
