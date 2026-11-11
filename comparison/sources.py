"""
One field as every extraction method sees it.

The viewer's files describe one field each (a generator point cloud or an
Open SciVis volume). ``FieldSource`` loads it once and hands every method the
same things:

* ``values``         the values the viewer shows (volumes scaled to [0, 1]);
* ``order``          the strict total order by (value, index): ``order[i]`` is
                     the rank of vertex i. Methods that do not break ties
                     themselves (AnyEG) run on it (see docs/extremum_graph_variants.md);
* ``coords`` / ``spacing``  positions of vertices, for the spine layout;
* ``base_graph``     our usual neighbour graph (6-neighbour grid or kNN);
* ``raw_values``     the values as stored on disk (to report what ties do).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Optional, Tuple

import numpy as np

from pipeline.neighbor_graph import NeighborGraph, grid_graph, knn_graph


@dataclass
class FieldSource:
    name: str
    kind: str                                   # "grid" or "points"
    values: np.ndarray                          # (N,) float64, as in the viewer
    raw_values: np.ndarray                      # (N,) as stored
    coords: Callable[[np.ndarray], np.ndarray]  # vertex ids -> (n, d) positions
    spacing: Optional[np.ndarray] = None        # voxel size (grids)
    grid_shape: Optional[Tuple[int, int, int]] = None   # (nz, ny, nx), grids only
    k: Optional[int] = None                     # kNN size, point clouds only
    points: Optional[np.ndarray] = field(default=None, repr=False)

    # -- constructors ---------------------------------------------------------
    @classmethod
    def volume(cls, name: str) -> "FieldSource":
        from pipeline.volume_loader import download, load_volume

        download(name)
        vol = load_volume(name)
        raw = load_volume(name, raw=True)
        return cls(name=name, kind="grid", values=vol.values.reshape(-1).astype(np.float64),
                   raw_values=raw.values.reshape(-1), coords=vol.coords,
                   spacing=np.asarray(vol.spacing, dtype=np.float64),
                   grid_shape=tuple(int(s) for s in vol.shape))

    @classmethod
    def dataset(cls, name: str, k: int) -> "FieldSource":
        from pipeline.data_loader import load_dataset

        ds = load_dataset(name)
        points = np.asarray(ds.points)
        values = np.asarray(ds.values, dtype=np.float64)
        return cls(name=name, kind="points", values=values, raw_values=values,
                   coords=lambda v: points[v], k=k, points=points)

    # -- derived (computed once) ------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.values.shape[0]

    @cached_property
    def order(self) -> np.ndarray:
        """Rank of every vertex in the (value, index) order: all distinct."""
        rank = np.empty(self.num_points, dtype=np.int64)
        rank[np.argsort(self.values, kind="stable")] = np.arange(self.num_points)
        return rank

    @cached_property
    def base_graph(self) -> NeighborGraph:
        """The neighbour graph our pipeline uses for this field."""
        if self.kind == "grid":
            return grid_graph(self.grid_shape)
        return knn_graph(self.points, self.k)
