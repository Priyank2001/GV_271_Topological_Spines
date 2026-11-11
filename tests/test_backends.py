"""Tests for pipeline/backends.py: every backend must give exactly the NumPy result.

torch on the CPU always runs; the CUDA cases are skipped without a GPU.
Run from the repo root: python3 -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.backends import (AutoBackend, NumpyBackend, TorchBackend, cuda_available,  # noqa: E402
                               get_backend)
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import NeighborGraph, grid_graph, knn_graph  # noqa: E402

torch = pytest.importorskip("torch")

DEVICES = ["cpu"] + (["cuda"] if cuda_available() else [])
FIELDS = ["peaks", "peak_values", "peak_persistence", "peak_parent", "saddles", "saddle_peaks",
          "saddle_is_merge", "saddle_persistence", "saddle_support", "saddle_relief",
          "saddle_saturation", "region", "region_size"]


def graphs_and_fields():
    rng = np.random.default_rng(3)
    pts = rng.random((800, 3))
    yield "knn", knn_graph(pts, 8, backend="numpy"), np.sin(7 * pts).sum(1)
    yield "grid smooth", grid_graph((12, 13, 14), backend="numpy"), rng.random(12 * 13 * 14)
    # Many equal values: the total order must break ties the same way everywhere.
    yield "grid ties", grid_graph((30, 30), backend="numpy"), rng.integers(0, 3, 900).astype(float)
    yield "constant", grid_graph((6, 6), backend="numpy"), np.zeros(36)
    # Isolated vertices (degree 0) and two components.
    yield "isolated", NeighborGraph.from_edges(6, np.array([[0, 1], [3, 4]]), backend="numpy"), \
        np.array([1.0, 2.0, 5.0, 3.0, 1.0, 0.0])


@pytest.mark.parametrize("device", DEVICES)
def test_extremum_graph_identical_to_numpy(device):
    gpu = TorchBackend(device)
    for name, g, f in graphs_and_fields():
        a = extremum_graph(f, g, backend="numpy")
        b = extremum_graph(f, g, backend=gpu)
        for field in FIELDS:
            assert np.array_equal(getattr(a, field), getattr(b, field)), (name, field)


@pytest.mark.parametrize("device", DEVICES)
def test_build_csr_identical_to_numpy(device):
    rng = np.random.default_rng(0)
    edges = rng.integers(0, 50, (400, 2))               # duplicates, both orientations, self loops
    a = NumpyBackend().build_csr(60, edges)              # vertices 50..59 have no edges
    b = TorchBackend(device).build_csr(60, edges)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert (a.edges[:, 0] < a.edges[:, 1]).all()
    assert np.array_equal(a.indptr[51:], np.full(10, a.indptr[50]))


def test_build_csr_matches_textbook_definition():
    edges = np.array([[2, 1], [1, 2], [0, 0], [0, 3], [3, 1]])
    csr = NumpyBackend().build_csr(4, edges)
    assert csr.edges.tolist() == [[0, 3], [1, 2], [1, 3]]
    assert csr.indptr.tolist() == [0, 1, 3, 4, 6]
    assert csr.indices.tolist() == [3, 2, 3, 1, 0, 1]


def test_graph_builders_accept_a_backend():
    g1 = grid_graph((5, 6, 7), backend="numpy")
    g2 = grid_graph((5, 6, 7), backend="torch:cpu")
    assert np.array_equal(g1.indices, g2.indices) and np.array_equal(g1.edges, g2.edges)


def test_device_graph_cache_is_not_fooled_by_a_new_graph():
    gpu = TorchBackend("cpu")
    f = np.arange(10.0)
    path = NeighborGraph.from_edges(10, np.array([[i, i + 1] for i in range(9)]))
    assert extremum_graph(f, path, backend=gpu).num_peaks == 1
    split = NeighborGraph.from_edges(10, np.array([[i, i + 1] for i in range(9) if i != 4]))
    assert extremum_graph(f, split, backend=gpu).num_peaks == 2


def test_auto_backend_uses_cpu_for_small_inputs_and_falls_back_on_gpu_errors():
    auto = AutoBackend(min_gpu_size=10)

    class BrokenGPU(NumpyBackend):
        def total_order(self, values):
            raise RuntimeError("CUDA out of memory (simulated)")

    auto._gpu, auto._gpu_checked = BrokenGPU(), True
    with pytest.warns(UserWarning, match="using NumPy"):
        rank = auto.total_order(np.array([3.0, 1.0, 2.0] * 5))
    assert rank[:3].tolist() == [10, 0, 5]          # five 1s, then five 2s, then the 3s


def test_get_backend_names():
    assert isinstance(get_backend("numpy"), NumpyBackend)
    assert isinstance(get_backend("torch:cpu"), TorchBackend)
    assert get_backend() is get_backend("auto")
    with pytest.raises(ValueError):
        get_backend("tpu")
