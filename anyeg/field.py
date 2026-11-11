"""
The input of the AnyEG pipeline: a scalar field sampled at points.

AnyEG reads one scalar per point as a 32-bit float (``vtkDataArray::GetTuple1``
cast to ``float``), so ``ScalarField.values`` is always float32. The points are
either a regular grid (a VTK ImageData: ``dims`` and ``spacing``) or arbitrary
positions (a point cloud or unstructured mesh: ``positions``).

Grid vertices are numbered the VTK way, x fastest:
``id = x + nx * (y + ny * z)``. This is the same numbering as
``values.reshape(-1)`` of an array of shape ``(nz, ny, nx)``, which is how the
Open SciVis ``.raw`` volumes are stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ScalarField:
    values: np.ndarray                                  # (N,) float32
    name: str = "field"
    array_name: str = "Scalars"
    dims: Optional[Tuple[int, int, int]] = None         # (nx, ny, nz) for grids
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    positions: Optional[np.ndarray] = None              # (N, d) for point clouds
    extra: dict = field(default_factory=dict)           # e.g. other point arrays

    def __post_init__(self):
        self.values = np.ascontiguousarray(self.values, dtype=np.float32).reshape(-1)
        if self.is_grid and int(np.prod(self.dims)) != self.num_points:
            raise ValueError(f"dims {self.dims} do not match {self.num_points} values")
        if not self.is_grid:
            if self.positions is None:
                raise ValueError("a field needs either grid dims or point positions")
            if len(self.positions) != self.num_points:
                raise ValueError("positions and values have different lengths")

    # -- constructors ----------------------------------------------------------
    @classmethod
    def from_grid(cls, values_zyx: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                  name: str = "field", array_name: str = "Scalars") -> "ScalarField":
        """From an array of shape (nz, ny, nx) (or (ny, nx) for 2D)."""
        a = np.asarray(values_zyx)
        shape = (1,) * (3 - a.ndim) + a.shape            # 2D -> one z slice
        nz, ny, nx = shape
        return cls(a.reshape(-1), name=name, array_name=array_name, dims=(nx, ny, nz),
                   spacing=tuple(map(float, spacing)), origin=tuple(map(float, origin)))

    @classmethod
    def from_points(cls, positions: np.ndarray, values: np.ndarray,
                    name: str = "points") -> "ScalarField":
        return cls(values, name=name, positions=np.asarray(positions, dtype=np.float64))

    # -- properties ------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.values.shape[0]

    @property
    def is_grid(self) -> bool:
        return self.dims is not None

    def grid_index(self, vertices: np.ndarray) -> np.ndarray:
        """(k, 3) integer grid coordinates (x, y, z) of grid vertices."""
        if not self.is_grid:
            raise ValueError("not a grid")
        v = np.asarray(vertices, dtype=np.int64)
        nx, ny, _ = self.dims
        return np.stack([v % nx, (v // nx) % ny, v // (nx * ny)], axis=1)

    def coordinates(self, vertices: np.ndarray) -> np.ndarray:
        """(k, 3) positions of the given vertices (point clouds: first 3 coordinates, zero-padded)."""
        if self.is_grid:
            return self.grid_index(vertices) * np.asarray(self.spacing) + np.asarray(self.origin)
        p = self.positions[np.asarray(vertices, dtype=np.int64)]
        out = np.zeros((p.shape[0], 3))
        out[:, :min(3, p.shape[1])] = p[:, :3]
        return out

    def __repr__(self) -> str:
        where = f"grid {self.dims}" if self.is_grid else f"{self.positions.shape[1]}-d points"
        return f"ScalarField({self.name!r}, {self.num_points:,} values on {where})"
