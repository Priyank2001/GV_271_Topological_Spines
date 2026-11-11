"""
Turning the kernel output into arcs: the CPU part of AnyEG.

This is a port of the second half of ``ttkExtremeGraphDev::RequestData``
(ttkExtremeGraphDev.cpp), which runs after ``compute()``:

1. **Persistence of each saddle**: ``|f(s) - f(maximum of s's own basin)|``.
   AnyEG computes percentile thresholds from these, then overwrites them
   with the constants ``0.0`` and ``20000.0``; those constants are the
   defaults of ``AssemblyOptions.persistence_range``.
2. **Optional edge bundling** (``use_edge_bundling``, switched off in the
   code): keep only saddles whose neighbours reach three or more maxima, plus
   for every pair of maxima the highest saddle that reaches exactly those two.
3. **Arcs.** For every saddle, group its neighbours by the maximum they
   ascend to and take the highest neighbour of each group (the first one in
   list order on a tie). This includes neighbours *below* the saddle, so a
   saddle is also joined to the maximum of its own basin. For each group
   whose ``|f(s) - f(maximum)|`` lies in ``[low, high)``, the arc is
   ``saddle -> that neighbour -> first_link -> ... -> maximum``.

AnyEG keeps the groups in a hash map, so the order of a saddle's arcs is not
defined there; here they are sorted by the maximum's vertex id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .graph import AnyEGGraph
from .kernels.base import KernelOutput
from .neighborhood import NeighborTable


@dataclass
class AssemblyOptions:
    persistence_range: Tuple[float, float] = (0.0, 20000.0)   # AnyEG's [threshold_lower, threshold_upper)
    bundle_edges: bool = False                                 # AnyEG's use_edge_bundling


class ArcAssembler:
    """Builds the ``AnyEGGraph`` from the kernel output (vectorised NumPy)."""

    def __init__(self, options: AssemblyOptions = AssemblyOptions()):
        self.options = options

    def assemble(self, values: np.ndarray, neighbors: NeighborTable, k: KernelOutput) -> AnyEGGraph:
        f = np.asarray(values, dtype=np.float32).astype(np.float64)   # AnyEG: float -> double
        saddles = k.saddles
        persistence = np.abs(f[saddles] - f[k.final_link[saddles]])

        drawn = self._bundled(f, neighbors, k) if self.options.bundle_edges else saddles
        arc_saddle, arc_start, arc_max = self._arc_starts(f, neighbors, k.final_link, drawn)

        low, high = self.options.persistence_range
        d = np.abs(f[arc_saddle] - f[arc_max])
        keep = (d >= low) & (d < high)
        arc_saddle, arc_start, arc_max = arc_saddle[keep], arc_start[keep], arc_max[keep]

        ptr, verts = _trace_arcs(arc_saddle, arc_start, k.first_link)
        return AnyEGGraph(
            num_points=f.shape[0], maxima=k.extrema, saddles=saddles,
            saddle_persistence=persistence, region=k.final_link, first_link=k.first_link,
            arc_saddle=arc_saddle, arc_maximum=arc_max, arc_ptr=ptr, arc_vertices=verts,
            num_candidates=k.num_candidates, seconds=dict(k.seconds),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _neighbor_entries(neighbors: NeighborTable, saddles: np.ndarray):
        """Flat (saddle index, position in list, neighbour) for every neighbour of every saddle."""
        rows = neighbors.table[saddles].astype(np.int64)            # (S, M)
        valid = rows >= 0
        s_idx, pos = np.nonzero(valid)
        return s_idx, pos, rows[s_idx, pos]

    def _arc_starts(self, f, neighbors, final_link, saddles):
        """For each (saddle, maximum reached by a neighbour): the highest such neighbour."""
        if saddles.shape[0] == 0:
            e = np.empty(0, dtype=np.int64)
            return e, e, e
        s_idx, pos, nb = self._neighbor_entries(neighbors, saddles)
        root = final_link[nb]
        # Sort by saddle, then maximum, then value (high first), then list position.
        order = np.lexsort((pos, -f[nb], root, s_idx))
        s_idx, nb, root = s_idx[order], nb[order], root[order]
        first = np.r_[True, (s_idx[1:] != s_idx[:-1]) | (root[1:] != root[:-1])]
        return saddles[s_idx[first]], nb[first], root[first]

    def _bundled(self, f, neighbors, k: KernelOutput) -> np.ndarray:
        """AnyEG's edge bundling: which saddles survive (kept in their original order)."""
        saddles = k.saddles
        if saddles.shape[0] == 0:
            return saddles
        s_idx, _, nb = self._neighbor_entries(neighbors, saddles)
        reached = {}
        for i, m in zip(s_idx.tolist(), k.final_link[nb].tolist()):
            reached.setdefault(i, set()).add(m)
        survive = set()
        best_for_pair = {}
        for i in range(saddles.shape[0]):
            ms = reached.get(i, set())
            if len(ms) >= 3:
                survive.add(i)
            elif len(ms) == 2:
                pair = tuple(sorted(ms))
                j = best_for_pair.get(pair)
                if j is None or f[saddles[i]] > f[saddles[j]]:
                    best_for_pair[pair] = i
        survive.update(best_for_pair.values())
        return saddles[sorted(survive)]


def _trace_arcs(arc_saddle: np.ndarray, arc_start: np.ndarray, first_link: np.ndarray):
    """Follow ``first_link`` from every arc's start to its maximum, all arcs at once.

    Returns CSR (ptr, vertices); each arc is [saddle, start, ..., maximum].
    """
    a = arc_saddle.shape[0]
    columns = [arc_saddle, arc_start]
    alive = np.ones(a, dtype=bool)
    cur = arc_start.copy()
    lengths = np.full(a, 2, dtype=np.int64)
    while alive.any():
        nxt = first_link[cur]
        alive &= nxt != cur                     # a maximum links to itself: the arc ends
        lengths += alive
        columns.append(np.where(alive, nxt, -1))
        cur = np.where(alive, nxt, cur)
    table = np.stack(columns, axis=1) if a else np.empty((0, 2), dtype=np.int64)
    ptr = np.zeros(a + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    return ptr, table[table >= 0]
