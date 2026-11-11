"""
Neighbourhoods: which vertices each vertex is compared with.

AnyEG keeps the neighbours in a padded table, ``Neighbors[v * MaxNeighborNum + i]``
with ``-1`` after the last neighbour (``NeighborTable`` below). The table is
filled by the TTK filter:

* **Grids** (``vtkImageData``): TTK's implicit triangulation, i.e. the
  Freudenthal (Kuhn) split of each cube into 6 tetrahedra. An interior vertex
  has 14 neighbours: 6 along the axes and 8 along diagonals. AnyEG's own
  ``checkNeighborsWithOffsets`` (cudaHelper.cu) confirms that TTK's lists equal
  the 14 offsets in ``FREUDENTHAL_OFFSETS``. TTK's order inside each list is
  not known without TTK; the order of the offsets is used here. Order only
  matters when two neighbours have exactly the same value.
* **Unstructured meshes**: all vertices that share a cell with the vertex.
* **Point clouds** are not handled by AnyEG; ``KnnNeighborhood`` adds them
  so the method can be run on the same data as our pipeline.

Every builder implements ``NeighborhoodBuilder.build(field) -> NeighborTable``,
so the pipeline can be given any of them (dependency inversion).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .field import ScalarField

# (dx, dy, dz) in the order AnyEG lists them (h_dx/h_dy/h_dz in cudaHelper.cu).
FREUDENTHAL_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (0, -1, -1), (1, -1, -1), (0, 0, -1), (1, 0, -1),
    (0, -1, 0), (1, -1, 0), (1, 0, 0), (-1, 0, 0), (-1, 1, 0), (0, 1, 0),
    (-1, 0, 1), (0, 0, 1), (0, 1, 1), (-1, 1, 1),
)
AXIS_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
)


@dataclass
class NeighborTable:
    """AnyEG's neighbour layout: an (N, M) int32 table, rows padded with -1."""

    table: np.ndarray     # (N, M) int32
    counts: np.ndarray    # (N,) int32, valid entries per row (they come first)

    @property
    def num_vertices(self) -> int:
        return self.table.shape[0]

    @property
    def max_degree(self) -> int:
        return self.table.shape[1]

    @property
    def num_entries(self) -> int:
        return int(self.counts.sum())

    def neighbors(self, v: int) -> np.ndarray:
        return self.table[v, :self.counts[v]]

    def flat(self) -> np.ndarray:
        """All valid entries, row after row (the per-vertex lists concatenated)."""
        return self.table[self.table >= 0]

    def edges(self) -> np.ndarray:
        """(E, 2) undirected edges i < j, each once."""
        rows = np.repeat(np.arange(self.num_vertices, dtype=np.int64), self.counts)
        cols = self.flat().astype(np.int64)
        keep = rows < cols
        return np.stack([rows[keep], cols[keep]], axis=1)

    @classmethod
    def from_csr(cls, indptr: np.ndarray, indices: np.ndarray) -> "NeighborTable":
        counts = np.diff(indptr).astype(np.int32)
        n, m = counts.shape[0], int(counts.max(initial=0))
        table = np.full((n, max(m, 1)), -1, dtype=np.int32)
        rows = np.repeat(np.arange(n), counts)
        cols = np.arange(indices.shape[0]) - np.repeat(indptr[:-1], counts)
        table[rows, cols] = indices
        return cls(table, counts)


class NeighborhoodBuilder(ABC):
    """Builds the neighbour table of a field."""

    name: str = "abstract"

    @abstractmethod
    def build(self, field: ScalarField) -> NeighborTable:
        ...


class StencilGridNeighborhood(NeighborhoodBuilder):
    """Neighbours of grid vertices given by a fixed list of (dx, dy, dz) offsets.

    Offsets that leave the grid are skipped; the remaining neighbours keep the
    order of the offsets.
    """

    def __init__(self, offsets: Sequence[Tuple[int, int, int]], name: str):
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.name = name

    def build(self, field: ScalarField) -> NeighborTable:
        if not field.is_grid:
            raise ValueError(f"{self.name} needs a grid field")
        nx, ny, nz = field.dims
        ids = np.arange(field.num_points, dtype=np.int64)
        x, y, z = ids % nx, (ids // nx) % ny, ids // (nx * ny)      # VTK numbering, x fastest

        k = self.offsets.shape[0]
        table = np.full((field.num_points, k), -1, dtype=np.int32)
        for j, (dx, dy, dz) in enumerate(self.offsets):
            xn, yn, zn = x + dx, y + dy, z + dz
            inside = (xn >= 0) & (xn < nx) & (yn >= 0) & (yn < ny) & (zn >= 0) & (zn < nz)
            table[inside, j] = (xn + nx * (yn + ny * zn))[inside]

        # Move the valid entries to the front of each row, keeping their order.
        valid = table >= 0
        order = np.argsort(~valid, axis=1, kind="stable")
        table = np.take_along_axis(table, order, axis=1)
        counts = valid.sum(axis=1).astype(np.int32)
        width = int(counts.max(initial=1))
        return NeighborTable(np.ascontiguousarray(table[:, :width]), counts)


class FreudenthalGridNeighborhood(StencilGridNeighborhood):
    """TTK's grid triangulation: 14 neighbours per interior vertex (AnyEG's default)."""

    def __init__(self):
        super().__init__(FREUDENTHAL_OFFSETS, "freudenthal-14")


class AxisGridNeighborhood(StencilGridNeighborhood):
    """Only the 6 axis neighbours (what our pipeline's ``grid_graph`` uses)."""

    def __init__(self):
        super().__init__(AXIS_OFFSETS, "axis-6")


class KnnNeighborhood(NeighborhoodBuilder):
    """Symmetric k-nearest-neighbour graph of a point cloud (union of the kNN relations)."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"knn-{k}"

    def build(self, field: ScalarField) -> NeighborTable:
        from scipy.sparse import coo_matrix
        from scipy.spatial import cKDTree

        if field.positions is None:
            raise ValueError("KnnNeighborhood needs point positions")
        pts = np.asarray(field.positions, dtype=np.float64)
        n = pts.shape[0]
        _, idx = cKDTree(pts).query(pts, k=self.k + 1, workers=-1)
        rows = np.repeat(np.arange(n), self.k + 1)
        cols = idx.reshape(-1)
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
        adj = coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n, n)).tocsr()
        adj = (adj + adj.T).tocsr()
        adj.sort_indices()
        return NeighborTable.from_csr(adj.indptr, adj.indices.astype(np.int32))


def default_neighborhood(field: ScalarField) -> NeighborhoodBuilder:
    """AnyEG's choice for grids; point clouds need an explicit builder."""
    if field.is_grid:
        return FreudenthalGridNeighborhood()
    raise ValueError("point clouds need a neighbourhood, e.g. KnnNeighborhood(k=10)")
