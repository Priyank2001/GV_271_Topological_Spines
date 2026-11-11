"""
A common description of an extremum graph, so different methods can be compared.

Our pipeline and AnyEG describe their results differently:

* ours (``pipeline.extremum_graph.ExtremumGraph``): peaks numbered 0..P-1,
  each saddle joins exactly two peaks, regions are peak ids;
* AnyEG (``anyeg.AnyEGGraph``): maxima and saddles are vertex ids, a saddle has
  one arc per basin among its neighbours (often two, sometimes more), regions
  are the vertex id of the maximum.

``CommonGraph`` uses vertex ids everywhere:

* ``maxima``         vertex ids of the maxima
* ``saddles``        vertex ids of the saddles that link maxima
* ``region``         for every vertex, the vertex id of its maximum
* ``maximum_pairs``  {(m1, m2)}: maxima joined through some saddle, the
                     structure the extremum graph shows (and the spine draws)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import numpy as np


@dataclass
class CommonGraph:
    method: str
    maxima: np.ndarray
    saddles: np.ndarray
    region: np.ndarray
    maximum_pairs: Set[Tuple[int, int]]
    seconds: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, object] = field(default_factory=dict)

    @property
    def num_maxima(self) -> int:
        return int(self.maxima.shape[0])

    @property
    def num_saddles(self) -> int:
        return int(self.saddles.shape[0])

    @property
    def total_seconds(self) -> float:
        return float(self.seconds.get("total", sum(self.seconds.values())))


def from_anyeg(graph, method: str, seconds: Dict[str, float]) -> CommonGraph:
    """Convert an ``anyeg.AnyEGGraph``."""
    linked = np.unique(graph.arc_saddle)
    return CommonGraph(
        method=method,
        maxima=np.sort(graph.maxima),
        saddles=linked,
        region=graph.region,
        maximum_pairs=graph.maximum_pairs(),
        seconds=seconds,
        notes={"candidates": graph.num_candidates, "upper_link_saddles": graph.num_saddles,
               "arcs": graph.num_arcs},
    )


def from_ours(eg, method: str, seconds: Dict[str, float]) -> CommonGraph:
    """Convert a ``pipeline.extremum_graph.ExtremumGraph``."""
    peaks = eg.peaks
    pairs = {tuple(sorted(p)) for p in peaks[eg.saddle_peaks].tolist()}
    return CommonGraph(
        method=method,
        maxima=np.sort(peaks),
        saddles=np.unique(eg.saddles),
        region=peaks[eg.region],
        maximum_pairs=pairs,
        seconds=seconds,
        notes={"loop_saddles": int((~eg.saddle_is_merge).sum())},
    )
