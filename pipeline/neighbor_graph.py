"""
Neighbour graphs over the samples of a scalar field.

The generator stores only samples (points + values). The pipeline builds
the graph that connects them here, after loading:

  * point clouds -> k-nearest-neighbour graph, any k   (``knn_graph``)
  * regular grids -> axis-aligned grid adjacency          (``grid_graph``)

Both return a ``NeighborGraph``: an undirected graph stored as a unique edge
list plus CSR adjacency, which is what extremum-graph extraction needs
(for every vertex, look at all neighbours to see which ones are higher).

Usage
-----
    from pipeline.data_loader import load_dataset
    from pipeline.neighbor_graph import knn_graph

    ds = load_dataset("cyclic_3d")
    g = knn_graph(ds.points, k=12)
    g.neighbors(0)           # neighbour indices of vertex 0
    g.edges                  # (E, 2), i < j, each edge once
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NeighborGraph:
    """Undirected graph on ``num_nodes`` vertices."""

    num_nodes: int
    edges: np.ndarray     # (E, 2) int64, i < j, unique, sorted
    indptr: np.ndarray    # (N+1,) CSR row pointers
    indices: np.ndarray   # (2E,) CSR column indices (neighbours)

    @classmethod
    def from_edges(cls, num_nodes: int, edges: np.ndarray) -> "NeighborGraph":
        """Build from any (E, 2) edge list. Self loops and duplicates are dropped."""
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        e = np.sort(e, axis=1)
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(e, axis=0)

        # CSR over both directions
        src = np.concatenate([e[:, 0], e[:, 1]])
        dst = np.concatenate([e[:, 1], e[:, 0]])
        order = np.lexsort((dst, src))
        src, dst = src[order], dst[order]
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=num_nodes), out=indptr[1:])
        return cls(num_nodes, e, indptr, dst)

    @property
    def num_edges(self) -> int:
        return self.edges.shape[0]

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def neighbors(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i]:self.indptr[i + 1]]

    def num_components(self) -> int:
        """Number of connected components (a disconnected graph splits peaks)."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        adj = csr_matrix((np.ones_like(self.indices), self.indices, self.indptr),
                         shape=(self.num_nodes, self.num_nodes))
        return connected_components(adj, directed=False)[0]

    def to_torch(self, device: str = "cpu") -> dict:
        import torch

        return {
            "edges": torch.from_numpy(self.edges).to(device),
            "indptr": torch.from_numpy(self.indptr).to(device),
            "indices": torch.from_numpy(self.indices).to(device),
        }

    def __repr__(self) -> str:
        deg = self.degree
        return (f"NeighborGraph(N={self.num_nodes}, E={self.num_edges}, "
                f"degree min/mean/max={deg.min()}/{deg.mean():.1f}/{deg.max()})")


# ---------------------------------------------------------------------------
# Point clouds: k-nearest neighbours
# ---------------------------------------------------------------------------
def knn_indices(points: np.ndarray, k: int, device: str | None = None,
                chunk_size: int = 4096) -> np.ndarray:
    """(N, k) int64 indices of each point's k nearest neighbours, self excluded.

    Uses a KD-tree for d <= 10 and chunked brute force above that (KD-trees
    degrade to brute force in high dimensions anyway). Brute force runs on
    the GPU when torch + CUDA are available, unless ``device`` says otherwise.
    """
    points = np.ascontiguousarray(points, dtype=np.float32)
    n, d = points.shape
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1, {n - 1}], got {k}")

    if d <= 10:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(points).query(points, k=k + 1, workers=-1)
        return _drop_self(idx, k)

    try:
        import torch
    except ImportError:
        torch = None

    if torch is None:  # pure numpy fallback
        sq = np.einsum("ij,ij->i", points, points)
        out = np.empty((n, k + 1), dtype=np.int64)
        for s in range(0, n, chunk_size):
            q = points[s:s + chunk_size]
            dist = sq[s:s + chunk_size, None] - 2 * q @ points.T + sq[None, :]
            part = np.argpartition(dist, k, axis=1)[:, :k + 1]
            order = np.take_along_axis(dist, part, 1).argsort(axis=1)
            out[s:s + chunk_size] = np.take_along_axis(part, order, 1)
        return _drop_self(out, k)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    X = torch.from_numpy(points).to(dev)
    sq = (X * X).sum(1)
    out = torch.empty((n, k + 1), dtype=torch.int64)
    with torch.no_grad():
        for s in range(0, n, chunk_size):
            q = X[s:s + chunk_size]
            dist = sq[s:s + chunk_size, None] - 2 * q @ X.T + sq[None, :]
            out[s:s + chunk_size] = dist.topk(k + 1, dim=1, largest=False).indices.cpu()
    return _drop_self(out.numpy(), k)


def _drop_self(idx: np.ndarray, k: int) -> np.ndarray:
    """Remove each row's own index from (N, k+1) neighbour lists.

    Self is normally in column 0, but with duplicate points it can land
    elsewhere, so mask it explicitly rather than slicing ``idx[:, 1:]``.
    """
    n = idx.shape[0]
    is_self = idx == np.arange(n)[:, None]
    no_self = ~is_self.any(axis=1)
    is_self[no_self, -1] = True  # self not found: drop the farthest instead
    return idx[~is_self].reshape(n, k).astype(np.int64)


def knn_graph(points: np.ndarray, k: int,
              mode: Literal["union", "mutual"] = "union",
              device: str | None = None) -> NeighborGraph:
    """Undirected k-nearest-neighbour graph.

    The raw kNN relation is not symmetric (j can be among i's neighbours
    without i being among j's), so it is made undirected:

    * ``"union"``  : edge if i is in kNN(j) OR j is in kNN(i). Every vertex
                     has degree >= k; the usual choice.
    * ``"mutual"`` : edge only if both hold. Sparser and can disconnect.
    """
    idx = knn_indices(points, k, device=device)
    n = idx.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = idx.reshape(-1)
    edges = np.stack([src, dst], axis=1)

    if mode == "mutual":
        e = np.sort(edges, axis=1)
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        edges = uniq[counts == 2]
    elif mode != "union":
        raise ValueError(f"mode must be 'union' or 'mutual', got {mode!r}")

    return NeighborGraph.from_edges(n, edges)


# ---------------------------------------------------------------------------
# Regular grids
# ---------------------------------------------------------------------------
def grid_graph(shape: tuple[int, ...]) -> NeighborGraph:
    """Axis-aligned adjacency of a regular grid (2d neighbours per vertex).

    Vertices are numbered in C order, matching ``values.reshape(-1)`` of a
    grid array with this ``shape``.
    """
    ids = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    parts = []
    for axis in range(len(shape)):
        a = np.take(ids, np.arange(shape[axis] - 1), axis=axis).reshape(-1)
        b = np.take(ids, np.arange(1, shape[axis]), axis=axis).reshape(-1)
        parts.append(np.stack([a, b], axis=1))
    edges = np.concatenate(parts) if parts else np.empty((0, 2), np.int64)
    return NeighborGraph.from_edges(ids.size, edges)
