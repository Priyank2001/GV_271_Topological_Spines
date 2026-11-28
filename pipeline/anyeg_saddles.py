"""
Saddles of our extremum graph, computed by AnyEG's GPU modules.

Our pipeline used to take, for every two basins that touch, the highest pass
between them (``ComputeBackend.passes``). This module replaces that step with
AnyEG's own saddle test (``anyeg.kernels``, modules 1-4; see
``anyeg/kernels/base.py``), run on *our* neighbour graph and *our* total order:

* **Saddle vertices** are AnyEG's: vertices with a strictly higher neighbour
  in another basin (module 3) whose higher neighbours form two or more
  connected pieces (module 4, the upper-link test).
* **Which maxima a saddle joins**: the basins of its *higher* neighbours.
  AnyEG's own arcs also run through lower neighbours, which makes false
  links (the fair comparison, observations.md 2026-11-23); only the uphill
  side is what the upper-link test looks at, so only it is used here.

AnyEG is given the (value, index) ranks as float32 values, so equal values
cannot make two vertices incomparable, and its steepest ascent picks the same
neighbour as ours: its maxima and basins are exactly our peaks and regions
(``anyeg_saddles`` checks this).

A saddle that joins k maxima becomes k - 1 rows ``(s, highest, other)``,
because ``ExtremumGraph`` joins exactly two peaks per saddle; this is the same
convention as ``comparison.variants.anyeg_to_extremum_graph``. Of all the rows
that join the same two peaks only the highest is kept, so there is at most one
saddle per pair of peaks, as before (on the ring datasets it is the very
vertex the highest-pass method picks; the lower ones sat 0.6-1.4 units away
from any true saddle). Every merge
saddle of the merge tree passes AnyEG's test (its higher neighbours lie in
different superlevel components, hence in different pieces of the upper link
and in different basins), so the persistence sweep stays exact.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .backends import AutoBackend, ComputeBackend, Passes, TorchBackend, _no_passes
from .neighbor_graph import NeighborGraph


def anyeg_saddles(rank: np.ndarray, region: np.ndarray, max_vertices: np.ndarray,
                  graph: NeighborGraph, backend: ComputeBackend) -> Passes:
    """AnyEG's saddles joined to the peaks their higher neighbours ascend to.

    ``region`` is the peak id of every vertex and ``max_vertices`` the vertex
    of every peak id, as computed by ``extremum_graph``. Returns ``Passes``
    sorted from the highest saddle down (rows of one saddle by the other
    peak's id), with only the highest row per pair of peaks; ``support`` is the number of the saddle's higher neighbours
    in the two basins of the row.
    """
    from anyeg.kernels import TorchKernels
    from anyeg.neighborhood import NeighborTable

    n = graph.num_nodes
    if n > 2 ** 24:
        raise ValueError("ranks above 2^24 are not exact in float32 (AnyEG's value type)")
    table = NeighborTable.from_csr(graph.indptr, graph.indices.astype(np.int32))
    out = TorchKernels(_device_for(backend, n)).run(rank.astype(np.float32), table)
    if not np.array_equal(out.final_link, max_vertices[region]):
        raise RuntimeError("AnyEG's basins differ from ours; the two orders disagree")
    return _join_uphill(out.saddles, rank, region, graph)


def _join_uphill(saddles: np.ndarray, rank: np.ndarray, region: np.ndarray,
                 graph: NeighborGraph) -> Passes:
    """One row per (saddle, other peak) for the peaks its higher neighbours reach."""
    if saddles.shape[0] == 0:
        return _no_passes()
    deg = graph.degree[saddles]
    src = np.repeat(saddles, deg)
    offsets = np.arange(src.shape[0]) - np.repeat(np.cumsum(deg) - deg, deg)
    nbr = graph.indices[np.repeat(graph.indptr[saddles], deg) + offsets]
    up = rank[nbr] > rank[src]
    src, peak = src[up], region[nbr[up]]

    # (saddle, peak) pairs with the number of higher neighbours behind each.
    key = np.unique(np.stack([src, peak], axis=1), axis=0, return_counts=True)
    (s, p), count = key[0].T, key[1]
    first = np.r_[True, s[1:] != s[:-1]]                 # rows sorted by (s, p): p ascending
    top = np.maximum.accumulate(np.where(first, np.arange(s.shape[0]), 0))
    other = ~first                                       # every peak but the saddle's highest
    vertex, a, b = s[other], p[top][other], p[other]
    support = count[top][other] + count[other]

    order = np.lexsort((b, -rank[vertex]))               # highest saddle first
    vertex, a, b, support = vertex[order], a[order], b[order], support[order]

    # Only the highest saddle of each pair of peaks (the user's choice,
    # 2026-11-27): the lower ones are mostly kNN artefacts away from any true
    # saddle, and the merge sweep never uses them anyway.
    _, first_of_pair = np.unique(a * (int(region.max()) + 1) + b, return_index=True)
    keep = np.sort(first_of_pair)
    return Passes(vertex=vertex[keep], peaks=np.stack([a, b], axis=1)[keep],
                  support=support[keep])


def _device_for(backend: ComputeBackend, n: int) -> Optional[str]:
    """Where AnyEG's kernels run: the GPU wherever our backend would use it."""
    if isinstance(backend, TorchBackend):
        return str(backend.device)
    if isinstance(backend, AutoBackend) and n >= backend.min_gpu_size and backend.gpu is not None:
        return str(backend.gpu.device)
    return "cpu"
