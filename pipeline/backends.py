"""
Compute backends: where the pipeline's heavy, vectorised graph steps run.

The extremum-graph code (``extremum_graph.py``) and the graph builders
(``neighbor_graph.py``) do two kinds of work:

* **bulk array work** over every vertex or edge: sorting the field into a
  total order, steepest ascent with pointer jumping, finding the highest
  pass between adjacent regions, and turning an edge list into CSR;
* **small sequential sweeps** over the saddles (union-find, the paper's
  Algorithm 1), which touch only a few thousand items.

Only the first kind is worth moving to a GPU, and it is exactly what a
``ComputeBackend`` does. Every backend returns plain NumPy arrays, so the rest
of the pipeline does not know or care where the work ran, and every backend
gives **bit-identical results** (``tests/test_backends.py`` checks this).

Backends
--------
``NumpyBackend``   CPU, always available (the original implementation).
``TorchBackend``   PyTorch on a CUDA device (or on the CPU, for testing).
``AutoBackend``    Picks torch on the GPU for large inputs and NumPy for
                   small ones (where copying to the GPU costs more than it
                   saves), and falls back to NumPy if the GPU runs out of
                   memory. This is the default everywhere.

Usage
-----
    from pipeline.backends import get_backend
    from pipeline.neighbor_graph import grid_graph
    from pipeline.extremum_graph import extremum_graph

    gpu = get_backend("torch")                 # or "cuda:1", "numpy", "auto"
    g = grid_graph(vol.shape, backend=gpu)
    eg = extremum_graph(values, g, backend=gpu)

Adding a backend (e.g. CuPy) means writing one subclass of ``ComputeBackend``;
nothing else changes (open/closed principle).
"""

from __future__ import annotations

import warnings
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import numpy as np

if TYPE_CHECKING:  # only for type hints; avoids a circular import
    from .neighbor_graph import NeighborGraph


# ---------------------------------------------------------------------------
# Result types shared by all backends
# ---------------------------------------------------------------------------
class Passes(NamedTuple):
    """Highest pass between each pair of adjacent regions, highest first."""

    vertex: np.ndarray    # (S,)   the pass (saddle) vertex
    peaks: np.ndarray     # (S, 2) region ids it joins, [smaller id, larger id]
    support: np.ndarray   # (S,)   number of graph edges between the two regions


class CSR(NamedTuple):
    """An undirected graph as a unique edge list plus CSR adjacency."""

    edges: np.ndarray     # (E, 2) int64, i < j, unique, sorted
    indptr: np.ndarray    # (N+1,) row pointers
    indices: np.ndarray   # (2E,) neighbours, sorted within each row


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
class ComputeBackend(ABC):
    """The bulk graph operations of the pipeline. All inputs and outputs are NumPy."""

    name: str = "abstract"

    @abstractmethod
    def total_order(self, values: np.ndarray) -> np.ndarray:
        """rank[i] = position of vertex i when sorted by (value, index), ascending."""

    @abstractmethod
    def steepest_ascent(self, rank: np.ndarray, graph: "NeighborGraph") -> np.ndarray:
        """For each vertex, the maximum reached by always stepping to the highest neighbour."""

    @abstractmethod
    def passes(self, rank: np.ndarray, region: np.ndarray, graph: "NeighborGraph",
               num_regions: int) -> Passes:
        """Highest pass between every pair of adjacent regions.

        Among all edges between regions A and B, the pass is the edge whose
        lower endpoint is highest; that endpoint is the pass vertex. Results
        are sorted from the highest pass down (ties keep the (A, B) order).
        """

    @abstractmethod
    def build_csr(self, num_nodes: int, edges: np.ndarray) -> CSR:
        """Undirected graph from any (E, 2) edge list; self loops and duplicates dropped."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# CPU: NumPy
# ---------------------------------------------------------------------------
class NumpyBackend(ComputeBackend):
    """Reference implementation on the CPU."""

    name = "numpy"

    def total_order(self, values):
        n = values.shape[0]
        order = np.argsort(values, kind="stable")          # ties keep index order
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        return rank

    def steepest_ascent(self, rank, graph):
        n = graph.num_nodes
        up = np.arange(n, dtype=np.int64)
        has_nbr = graph.degree > 0
        if has_nbr.any():
            starts = graph.indptr[:-1][has_nbr]
            best_rank = np.maximum.reduceat(rank[graph.indices], starts)
            vertex_at_rank = np.empty(n, dtype=np.int64)
            vertex_at_rank[rank] = np.arange(n)
            v = np.flatnonzero(has_nbr)
            higher = best_rank > rank[v]
            up[v[higher]] = vertex_at_rank[best_rank[higher]]
        return _pointer_jump(up, np.array_equal)

    def passes(self, rank, region, graph, num_regions):
        e = graph.edges
        ra, rb = region[e[:, 0]], region[e[:, 1]]
        cross = ra != rb
        e, ra, rb = e[cross], ra[cross], rb[cross]
        if e.shape[0] == 0:
            return _no_passes()

        r0, r1 = rank[e[:, 0]], rank[e[:, 1]]
        level = np.minimum(r0, r1)                           # pass height of each edge
        low_vertex = np.where(r0 < r1, e[:, 0], e[:, 1])     # its lower endpoint
        a, b = np.minimum(ra, rb), np.maximum(ra, rb)
        key = a * num_regions + b

        order = np.lexsort((level, key))                     # by pair, then by height
        key, a, b, level, low_vertex = key[order], a[order], b[order], level[order], low_vertex[order]
        group_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        group_end = np.r_[group_start[1:], key.shape[0]] - 1  # highest edge of each pair

        top_down = np.argsort(-level[group_end], kind="stable")
        pick = group_end[top_down]
        return Passes(vertex=low_vertex[pick],
                      peaks=np.stack([a[pick], b[pick]], axis=1),
                      support=(group_end - group_start + 1)[top_down])

    def build_csr(self, num_nodes, edges):
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        lo, hi = e.min(axis=1), e.max(axis=1)
        key = np.unique(_pair_key(lo[lo != hi], hi[lo != hi], num_nodes))   # sorted, unique
        lo, hi = key // num_nodes, key % num_nodes

        src, dst = np.concatenate([lo, hi]), np.concatenate([hi, lo])
        order = np.argsort(_pair_key(src, dst, num_nodes))
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=num_nodes), out=indptr[1:])
        return CSR(np.stack([lo, hi], axis=1), indptr, dst[order])


# ---------------------------------------------------------------------------
# GPU: PyTorch
# ---------------------------------------------------------------------------
class TorchBackend(ComputeBackend):
    """The same operations as ``NumpyBackend``, written with PyTorch tensors.

    Each method copies its inputs to ``device``, works there, and copies the
    result back. The graph's arrays are copied once and kept while the same
    graph object is used again (``extremum_graph`` uses it four times).
    """

    name = "torch"

    def __init__(self, device: Optional[str] = None):
        import torch

        self._torch = torch
        self.device = torch.device(device) if device else pick_cuda_device()
        self._graph_cache = _DeviceGraphCache(self)

    def __repr__(self) -> str:
        return f"TorchBackend(device='{self.device}')"

    # -- helpers -----------------------------------------------------------
    def to_device(self, array: np.ndarray):
        return self._torch.as_tensor(np.ascontiguousarray(array), device=self.device)

    @staticmethod
    def to_numpy(tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def _arange(self, n: int):
        return self._torch.arange(n, dtype=self._torch.int64, device=self.device)

    # -- operations ----------------------------------------------------------
    def total_order(self, values):
        torch = self._torch
        v = self.to_device(values)
        order = torch.sort(v, stable=True).indices
        rank = torch.empty_like(order)
        rank[order] = self._arange(order.shape[0])
        return self.to_numpy(rank)

    def steepest_ascent(self, rank, graph):
        torch = self._torch
        g = self._graph_cache.get(graph)
        n = graph.num_nodes
        r = self.to_device(rank)

        # Highest neighbour rank of every vertex: a segmented max over the CSR rows.
        best = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        best.scatter_reduce_(0, g.row, r[g.indices], reduce="amax", include_self=True)

        vertex_at_rank = torch.empty_like(r)
        vertex_at_rank[r] = self._arange(n)
        up = self._arange(n)
        higher = best > r
        up[higher] = vertex_at_rank[best[higher]]
        return self.to_numpy(_pointer_jump(up, torch.equal))

    def passes(self, rank, region, graph, num_regions):
        torch = self._torch
        g = self._graph_cache.get(graph)
        e0, e1 = g.edges[:, 0], g.edges[:, 1]
        reg = self.to_device(region)
        ra, rb = reg[e0], reg[e1]
        cross = ra != rb
        if not bool(cross.any()):
            return _no_passes()
        e0, e1, ra, rb = e0[cross], e1[cross], ra[cross], rb[cross]

        r = self.to_device(rank)
        r0, r1 = r[e0], r[e1]
        level = torch.minimum(r0, r1)
        low_vertex = torch.where(r0 < r1, e0, e1)
        a, b = torch.minimum(ra, rb), torch.maximum(ra, rb)
        key = a * num_regions + b

        # Sort by pair, then by height: two stable sorts, the minor key first.
        order = torch.sort(level, stable=True).indices
        order = order[torch.sort(key[order], stable=True).indices]
        key, a, b, level, low_vertex = key[order], a[order], b[order], level[order], low_vertex[order]
        _, counts = torch.unique_consecutive(key, return_counts=True)
        group_end = torch.cumsum(counts, 0) - 1                       # highest edge of each pair

        top_down = torch.sort(-level[group_end], stable=True).indices
        pick = group_end[top_down]
        return Passes(vertex=self.to_numpy(low_vertex[pick]),
                      peaks=self.to_numpy(torch.stack([a[pick], b[pick]], dim=1)),
                      support=self.to_numpy(counts[top_down]))

    def build_csr(self, num_nodes, edges):
        torch = self._torch
        e = self.to_device(np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        lo, hi = torch.minimum(e[:, 0], e[:, 1]), torch.maximum(e[:, 0], e[:, 1])
        proper = lo != hi
        key = torch.unique(_pair_key(lo[proper], hi[proper], num_nodes))   # sorted, unique
        lo, hi = key // num_nodes, key % num_nodes

        src, dst = torch.cat([lo, hi]), torch.cat([hi, lo])
        order = torch.argsort(_pair_key(src, dst, num_nodes))
        indptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=self.device)
        indptr[1:] = torch.cumsum(torch.bincount(src, minlength=num_nodes), 0)
        return CSR(self.to_numpy(torch.stack([lo, hi], dim=1)),
                   self.to_numpy(indptr), self.to_numpy(dst[order]))


class _DeviceGraph(NamedTuple):
    edges: object     # (E, 2) tensor
    indices: object   # (2E,) tensor, CSR neighbours
    row: object       # (2E,) tensor, the CSR row (source vertex) of each entry


class _DeviceGraphCache:
    """Keeps the device copy of the most recently used graph.

    Keyed by object identity and checked with a weak reference, so a new
    graph that happens to reuse a freed object's id is never confused with it.
    """

    def __init__(self, backend: TorchBackend):
        self._backend = backend
        self._ref = None
        self._tensors: Optional[_DeviceGraph] = None

    def get(self, graph: "NeighborGraph") -> _DeviceGraph:
        if self._ref is not None and self._ref() is graph:
            return self._tensors
        b = self._backend
        degree = b.to_device(graph.degree)
        self._tensors = _DeviceGraph(
            edges=b.to_device(graph.edges),
            indices=b.to_device(graph.indices),
            row=b._torch.repeat_interleave(b._arange(graph.num_nodes), degree),
        )
        self._ref = weakref.ref(graph)
        return self._tensors


# ---------------------------------------------------------------------------
# Automatic choice
# ---------------------------------------------------------------------------
class AutoBackend(ComputeBackend):
    """GPU for large inputs, CPU for small ones; falls back to CPU on GPU failure.

    ``min_gpu_size`` is the number of vertices (or edges, for ``build_csr``)
    from which the GPU is used. Below about 10^5 the copies to and from the
    GPU take longer than NumPy needs for the whole job (see the benchmark in
    ``benchmarks/backend_benchmark.py``).
    """

    name = "auto"

    def __init__(self, min_gpu_size: int = 100_000, device: Optional[str] = None):
        self.cpu = NumpyBackend()
        self.min_gpu_size = min_gpu_size
        self._device = device
        self._gpu: Optional[ComputeBackend] = None
        self._gpu_checked = False

    def __repr__(self) -> str:
        return f"AutoBackend(min_gpu_size={self.min_gpu_size}, gpu={self.gpu})"

    @property
    def gpu(self) -> Optional[ComputeBackend]:
        """The torch backend, or None if torch or CUDA is missing."""
        if not self._gpu_checked:
            self._gpu_checked = True
            if cuda_available():
                self._gpu = TorchBackend(self._device)
        return self._gpu

    def _run(self, size: int, method: str, *args):
        if size >= self.min_gpu_size and self.gpu is not None:
            try:
                return getattr(self.gpu, method)(*args)
            except RuntimeError as err:          # includes torch.cuda.OutOfMemoryError
                warnings.warn(f"GPU {method} failed ({err}); using NumPy instead.")
        return getattr(self.cpu, method)(*args)

    def total_order(self, values):
        return self._run(values.shape[0], "total_order", values)

    def steepest_ascent(self, rank, graph):
        return self._run(graph.num_nodes, "steepest_ascent", rank, graph)

    def passes(self, rank, region, graph, num_regions):
        return self._run(graph.num_nodes, "passes", rank, region, graph, num_regions)

    def build_csr(self, num_nodes, edges):
        return self._run(len(edges), "build_csr", num_nodes, edges)


# ---------------------------------------------------------------------------
# Factory and device helpers
# ---------------------------------------------------------------------------
BackendSpec = Union[None, str, ComputeBackend]
_DEFAULT: Optional[ComputeBackend] = None


def get_backend(spec: BackendSpec = None) -> ComputeBackend:
    """Turn a name into a backend.

    ``None`` or ``"auto"`` -> the shared ``AutoBackend``; ``"numpy"``/``"cpu"``
    -> ``NumpyBackend``; ``"torch"`` -> ``TorchBackend`` on the GPU with the
    most free memory; ``"cuda:1"`` or ``"torch:cpu"`` -> ``TorchBackend`` on
    that device. A ``ComputeBackend`` instance is returned unchanged.
    """
    global _DEFAULT
    if isinstance(spec, ComputeBackend):
        return spec
    if spec is None or spec == "auto":
        if _DEFAULT is None:
            _DEFAULT = AutoBackend()
        return _DEFAULT
    if spec in ("numpy", "cpu"):
        return NumpyBackend()
    if spec == "torch":
        return TorchBackend()
    if spec.startswith("torch:"):
        return TorchBackend(spec.split(":", 1)[1])
    if spec.startswith("cuda"):
        return TorchBackend(spec)
    raise ValueError(f"unknown backend {spec!r}; use 'auto', 'numpy', 'torch', 'cuda:N' or 'torch:cpu'")


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pick_cuda_device() -> str:
    """The CUDA device with the most free memory ('cpu' if there is none).

    The lab GPUs are shared, so device 0 is often the fullest one.
    """
    if not cuda_available():
        return "cpu"
    import torch

    free = []
    for i in range(torch.cuda.device_count()):
        try:
            free.append((torch.cuda.mem_get_info(i)[0], -i))
        except RuntimeError:              # device busy or not usable
            continue
    if not free:
        return "cpu"
    return f"cuda:{-max(free)[1]}"


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _pointer_jump(up, equal):
    """Follow ``up`` pointers until every vertex points at a root (works for NumPy and torch)."""
    while True:
        nxt = up[up]
        if equal(nxt, up):
            return up
        up = nxt


def _pair_key(a, b, n: int):
    """One int64 per ordered pair (a, b) with a, b < n. Valid while n**2 < 2**63."""
    if n >= 3_037_000_499:
        raise ValueError(f"too many vertices ({n}) for 64-bit pair keys")
    return a * n + b


def _no_passes() -> Passes:
    empty = np.empty(0, dtype=np.int64)
    return Passes(empty, np.empty((0, 2), dtype=np.int64), empty)
