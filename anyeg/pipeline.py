"""
The AnyEG pipeline: neighbourhood -> GPU modules -> arcs.

``AnyEGPipeline`` only wires three replaceable parts together, each behind
an interface (dependency inversion):

* a ``NeighborhoodBuilder``      (default: TTK's 14-neighbour grid, as AnyEG)
* an ``ExtremumGraphKernels``    (default: the PyTorch port on the GPU;
                                  ``OriginalCudaKernels`` runs AnyEG's own code)
* an ``ArcAssembler``            (port of AnyEG's CPU post-processing)

Usage
-----
    from anyeg import AnyEGPipeline, read_vti, write_vtp

    field = read_vti("third_party/AnyEG/data/scalar_field_v1.vti")
    graph = AnyEGPipeline().run(field)
    write_vtp("results/scalar_field_v1.vtp", graph.to_polylines(field))
"""

from __future__ import annotations

import time
from typing import Optional

from .assembly import ArcAssembler
from .field import ScalarField
from .graph import AnyEGGraph
from .kernels import ExtremumGraphKernels, TorchKernels
from .neighborhood import NeighborhoodBuilder, NeighborTable, default_neighborhood


class AnyEGPipeline:
    def __init__(self, neighborhood: Optional[NeighborhoodBuilder] = None,
                 kernels: Optional[ExtremumGraphKernels] = None,
                 assembler: Optional[ArcAssembler] = None):
        self.neighborhood = neighborhood
        self.kernels = kernels or TorchKernels()
        self.assembler = assembler or ArcAssembler()

    def __repr__(self) -> str:
        nb = self.neighborhood.name if self.neighborhood else "default"
        return f"AnyEGPipeline(neighborhood={nb}, kernels={self.kernels!r})"

    def build_neighbors(self, field: ScalarField) -> NeighborTable:
        builder = self.neighborhood or default_neighborhood(field)
        return builder.build(field)

    def run(self, field: ScalarField, neighbors: Optional[NeighborTable] = None) -> AnyEGGraph:
        """Compute the extremum graph; pass ``neighbors`` to reuse a table."""
        t0 = time.perf_counter()
        if neighbors is None:
            neighbors = self.build_neighbors(field)
        t1 = time.perf_counter()
        out = self.kernels.run(field.values, neighbors)
        t2 = time.perf_counter()
        graph = self.assembler.assemble(field.values, neighbors, out)
        t3 = time.perf_counter()
        graph.seconds.update(neighborhood=t1 - t0, kernels_total=t2 - t1, assembly=t3 - t2,
                             total=t3 - t0)
        return graph
