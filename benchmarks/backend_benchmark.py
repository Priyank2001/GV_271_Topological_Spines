"""
Benchmark the compute backends of our pipeline (NumPy on the CPU vs torch on the GPU).

For each input it times building the neighbour graph and computing the
extremum graph with every backend, checks that the results are identical,
and prints a Markdown table.

    python3 benchmarks/backend_benchmark.py                 # default inputs
    python3 benchmarks/backend_benchmark.py --sizes 64 128 256 --repeat 3

Inputs: synthetic smooth fields on n^3 grids (sum of random Gaussians plus a
little noise, so there are many peaks), the downloaded Open SciVis volumes,
and kNN graphs of random point clouds. Timings are wall-clock seconds, best of
``--repeat`` runs, after one warm-up run (CUDA start-up is not counted).
The GPUs on the lab server are shared, so GPU timings vary with their load.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.backends import ComputeBackend, NumpyBackend, TorchBackend, cuda_available  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import NeighborGraph, grid_graph, knn_graph  # noqa: E402


@dataclass
class Case:
    """One benchmark input: a field and a function that builds its graph."""

    name: str
    values: np.ndarray
    build_graph: Callable[[ComputeBackend], NeighborGraph]


def gaussian_field(n: int, bumps: int = 200, seed: int = 0) -> np.ndarray:
    """A smooth n^3 field with about ``bumps`` peaks, flattened in C order."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 1, n, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    f = np.zeros((n, n, n), dtype=np.float32)
    for c, h, w in zip(rng.random((bumps, 3)), rng.random(bumps), 0.03 + 0.07 * rng.random(bumps)):
        f += h * np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * w * w))
    f += 1e-3 * rng.standard_normal(f.shape).astype(np.float32)
    return f.reshape(-1).astype(np.float64)


def grid_case(n: int) -> Case:
    return Case(f"gaussians {n}^3", gaussian_field(n), lambda b: grid_graph((n, n, n), backend=b))


def volume_case(name: str) -> Case:
    from pipeline.volume_loader import load_volume

    vol = load_volume(name)
    return Case(f"{name} {'x'.join(map(str, vol.shape[::-1]))}", vol.values.reshape(-1),
                lambda b: grid_graph(vol.shape, backend=b))


def knn_case(n: int, d: int, k: int) -> Case:
    rng = np.random.default_rng(1)
    pts = rng.random((n, d)).astype(np.float32)
    values = np.sin(6 * pts).sum(1).astype(np.float64)
    return Case(f"kNN N={n:,} d={d} k={k}", values, lambda b: knn_graph(pts, k, backend=b))


def best_time(fn: Callable[[], object], repeat: int):
    fn()                                                  # warm-up
    times, result = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return min(times), result


def same_graph(a, b) -> bool:
    fields = ["peaks", "peak_persistence", "peak_parent", "saddles", "saddle_peaks",
              "saddle_persistence", "saddle_saturation", "region"]
    return all(np.array_equal(getattr(a, f), getattr(b, f)) for f in fields)


def run(cases: List[Case], backends: List[ComputeBackend], repeat: int) -> None:
    print("| input | N | backend | graph build (s) | extremum graph (s) | total (s) | speed-up | same result |")
    print("|---|---|---|---|---|---|---|---|")
    for case in cases:
        reference = None
        for backend in backends:
            t_graph, g = best_time(lambda: case.build_graph(backend), repeat)
            t_eg, eg = best_time(lambda: extremum_graph(case.values, g, backend=backend), repeat)
            total = t_graph + t_eg
            if reference is None:
                reference = (total, eg)
            speedup = reference[0] / total
            same = "yes" if same_graph(reference[1], eg) else "**NO**"
            print(f"| {case.name} | {g.num_nodes:,} | {backend!r} | {t_graph:.3f} | {t_eg:.3f} | "
                  f"{total:.3f} | {speedup:.1f}x | {same} |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="*", default=[32, 64, 128, 256], help="n for n^3 grids")
    ap.add_argument("--volumes", nargs="*", default=["fuel", "hydrogen_atom"])
    ap.add_argument("--knn", type=int, nargs="*", default=[100_000, 1_000_000], help="point counts")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", default=None, help="CUDA device for torch (default: most free memory)")
    args = ap.parse_args()

    backends: List[ComputeBackend] = [NumpyBackend()]
    if cuda_available():
        backends.append(TorchBackend(args.device))
    else:
        print("No CUDA device: only NumPy is timed.")

    cases = ([grid_case(n) for n in args.sizes] + [volume_case(v) for v in args.volumes]
             + [knn_case(n, 3, 10) for n in args.knn])
    run(cases, backends, args.repeat)


if __name__ == "__main__":
    main()
