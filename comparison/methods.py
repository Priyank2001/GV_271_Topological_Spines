"""
The methods being compared, behind one interface: ``Method.run(field) -> CommonGraph``.

* ``AnyEGMethod``  the separate AnyEG pipeline (``anyeg``), with either the
                   original CUDA kernels or the PyTorch port.
* ``OursMethod``   our pipeline (``pipeline.extremum_graph``) on either our
                   usual 6-neighbour grid graph or AnyEG's 14-neighbour one,
                   with the NumPy or the torch backend.

Running ours on the 14-neighbour graph separates the two reasons the results
differ: the neighbourhood, and the algorithm itself.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from anyeg import AnyEGPipeline, FreudenthalGridNeighborhood, ScalarField
from anyeg.kernels import ExtremumGraphKernels
from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import NeighborGraph, grid_graph

from .common import CommonGraph, from_anyeg, from_ours


class Method(ABC):
    name: str

    @abstractmethod
    def run(self, field: ScalarField) -> CommonGraph:
        ...


class AnyEGMethod(Method):
    def __init__(self, name: str, kernels: ExtremumGraphKernels):
        self.name = name
        self.pipeline = AnyEGPipeline(neighborhood=FreudenthalGridNeighborhood(), kernels=kernels)

    def run(self, field: ScalarField) -> CommonGraph:
        graph = self.pipeline.run(field)
        s = graph.seconds
        modules = sum(v for k, v in s.items() if k.startswith("module"))
        seconds = {"graph": s["neighborhood"], "core": modules, "post": s["assembly"],
                   "total": s["total"]}
        if "compute_call" in s:          # the original: compute() includes its own copies
            seconds["compute_call"] = s["compute_call"]
        return from_anyeg(graph, self.name, seconds)


class OursMethod(Method):
    """Our extremum graph; ``neighborhood`` is "axis-6" (ours) or "freudenthal-14" (AnyEG's)."""

    def __init__(self, name: str, neighborhood: str = "axis-6", backend: str = "auto"):
        if neighborhood not in ("axis-6", "freudenthal-14"):
            raise ValueError(neighborhood)
        self.name = name
        self.neighborhood = neighborhood
        self.backend = backend

    def _graph(self, field: ScalarField) -> NeighborGraph:
        nx, ny, nz = field.dims
        if self.neighborhood == "axis-6":
            return grid_graph((nz, ny, nx), backend=self.backend)      # same vertex numbering
        table = FreudenthalGridNeighborhood().build(field)
        return NeighborGraph.from_edges(field.num_points, table.edges(), backend=self.backend)

    def run(self, field: ScalarField) -> CommonGraph:
        t0 = time.perf_counter()
        g = self._graph(field)
        t1 = time.perf_counter()
        eg = extremum_graph(field.values.astype(np.float64), g, backend=self.backend)
        t2 = time.perf_counter()
        return from_ours(eg, self.name, {"graph": t1 - t0, "core": t2 - t1, "post": 0.0,
                                         "total": t2 - t0})
