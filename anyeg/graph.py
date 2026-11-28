"""
The result of the AnyEG pipeline: maxima, saddles, and the arcs between them.

An **arc** starts at a saddle, steps to one of its neighbours, and then
follows the steepest-ascent links (``first_link``) up to a maximum. AnyEG
draws every arc as a chain of line segments in its ``.vtp`` output; the
extremum graph itself is "saddle s is linked to maximum m" for every arc.

Arcs are stored in CSR form: the vertices of arc ``i`` are
``arc_vertices[arc_ptr[i]:arc_ptr[i + 1]]``, starting with the saddle and
ending with the maximum.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Set, Tuple

import numpy as np

from .field import ScalarField
from .vtk_io import PolyLines

# NodeType values of AnyEG's .vtp output
REGULAR, SADDLE, MAXIMUM = 0, 1, 2


@dataclass
class AnyEGGraph:
    num_points: int
    maxima: np.ndarray               # (P,) vertex ids, ascending
    saddles: np.ndarray              # (S,) vertex ids, ascending (all saddles, drawn or not)
    saddle_persistence: np.ndarray   # (S,) |f(saddle) - f(maximum of its own basin)|
    region: np.ndarray               # (N,) maximum (vertex id) each vertex ascends to
    first_link: np.ndarray           # (N,) steepest-ascent neighbour of each vertex

    arc_saddle: np.ndarray           # (A,) saddle vertex of each arc
    arc_maximum: np.ndarray          # (A,) maximum vertex it reaches
    arc_ptr: np.ndarray              # (A+1,) CSR pointers into arc_vertices
    arc_vertices: np.ndarray         # vertices of all arcs, saddle first
    num_candidates: int = 0
    seconds: Dict[str, float] = field(default_factory=dict)

    # -- the extremum graph ----------------------------------------------------
    @property
    def num_maxima(self) -> int:
        return self.maxima.shape[0]

    @property
    def num_saddles(self) -> int:
        return self.saddles.shape[0]

    @property
    def num_arcs(self) -> int:
        return self.arc_saddle.shape[0]

    def links(self) -> Set[Tuple[int, int]]:
        """{(saddle vertex, maximum vertex)} for every arc."""
        return set(zip(self.arc_saddle.tolist(), self.arc_maximum.tolist()))

    def maximum_pairs(self) -> Set[Tuple[int, int]]:
        """{(m1, m2)}, m1 < m2: maxima joined through a common saddle.

        This is the graph on the maxima that the extremum graph describes; it
        can be compared with any other method's.
        """
        reached: Dict[int, Set[int]] = {}
        for s, m in zip(self.arc_saddle.tolist(), self.arc_maximum.tolist()):
            reached.setdefault(s, set()).add(m)
        pairs = set()
        for ms in reached.values():
            ms = sorted(ms)
            pairs.update((a, b) for i, a in enumerate(ms) for b in ms[i + 1:])
        return pairs

    def arc(self, i: int) -> np.ndarray:
        return self.arc_vertices[self.arc_ptr[i]:self.arc_ptr[i + 1]]

    def uphill_only(self, order: np.ndarray) -> "AnyEGGraph":
        """The graph without the arcs whose first step from the saddle goes down.

        Not AnyEG's behaviour. AnyEG joins a saddle to the basin of *every*
        neighbour, lower ones too, and the maximum a lower neighbour ascends
        to need not be connected to the saddle above the saddle's height
        (the fair comparison's false links). ``order``: the values' total
        order (higher = larger), e.g. the ranks AnyEG was run on.
        """
        order = np.asarray(order)
        start = self.arc_vertices[self.arc_ptr[:-1] + 1]
        keep = order[start] > order[self.arc_saddle]
        lengths = np.diff(self.arc_ptr)[keep]
        verts = (np.concatenate([self.arc(i) for i in np.flatnonzero(keep)]) if keep.any()
                 else self.arc_vertices[:0])
        return replace(self, arc_saddle=self.arc_saddle[keep], arc_maximum=self.arc_maximum[keep],
                       arc_ptr=np.r_[0, np.cumsum(lengths)].astype(self.arc_ptr.dtype),
                       arc_vertices=verts, seconds=dict(self.seconds))

    # -- the .vtp output ----------------------------------------------------------
    def to_polylines(self, field: ScalarField) -> PolyLines:
        """The geometry AnyEG writes: one 2-point line per arc step.

        As in AnyEG, points are numbered in order of first use, and a step
        shared by two arcs is written twice. Point arrays: ``NodeType``
        (0 regular, 1 saddle, 2 maximum) and the field value.
        """
        if self.num_arcs == 0:
            empty = np.empty((0, 3))
            return PolyLines.from_segments(empty, np.empty((0, 2), np.int64),
                                           NodeType=np.empty(0, np.int32))
        ends = self.arc_ptr[1:]
        steps = np.ones(self.arc_vertices.shape[0], dtype=bool)
        steps[ends - 1] = False                       # no step out of an arc's last vertex
        a = self.arc_vertices[steps]
        b = self.arc_vertices[np.flatnonzero(steps) + 1]
        ends_flat = np.stack([a, b], axis=1).reshape(-1)
        uniq, first, inverse = np.unique(ends_flat, return_index=True, return_inverse=True)
        by_first_use = np.argsort(first)               # AnyEG numbers points in order of first use
        new_id = np.empty_like(by_first_use)
        new_id[by_first_use] = np.arange(by_first_use.shape[0])
        used = uniq[by_first_use]
        segments = new_id[inverse].reshape(-1, 2)

        node_type = np.full(used.shape[0], REGULAR, dtype=np.int32)
        node_type[np.isin(used, self.saddles)] = SADDLE
        node_type[np.isin(used, self.maxima)] = MAXIMUM
        return PolyLines.from_segments(field.coordinates(used), segments, NodeType=node_type,
                                       **{field.array_name: field.values[used]},
                                       VertexId=used.astype(np.int64))

    def __repr__(self) -> str:
        return (f"AnyEGGraph({self.num_maxima:,} maxima, {self.num_saddles:,} saddles, "
                f"{self.num_arcs:,} arcs, N={self.num_points:,})")
