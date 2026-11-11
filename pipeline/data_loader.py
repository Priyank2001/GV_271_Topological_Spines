"""
Load the scalar-field datasets produced by ``generator/``.

Each dataset lives in ``generator/datasets/<name>/`` and contains only
samples and ground truth:

    points.npy   (N, d) float32  sample positions
    values.npy   (N,)   float32  scalar value at each point
    truth.json   metadata and, where known, ground-truth critical points

The neighbour graph is NOT stored; build it after loading with
``pipeline.neighbor_graph.knn_graph(ds.points, k)``.

Ground truth comes in two layouts, both normalised here to a flat list of
``CriticalPoint`` objects:

  * ``controlled_extrema`` datasets store ``extrema = {maxima, minima, saddles}``
    with centers (and amplitudes / saddle signs).
  * ``cyclic_ring`` datasets store ``critical_points = [{type, id, center}]``
    plus the expected extremum graph in ``extremum_graph_edges``.

Datasets without labels (``mathematical``, ``smooth_random``) get an empty list.

Usage
-----
    from pipeline.data_loader import load_dataset, list_datasets
    from pipeline.neighbor_graph import knn_graph

    ds = load_dataset("ext_2d_small")
    ds.points, ds.values           # numpy arrays
    ds.maxima, ds.saddles          # ground-truth critical points
    g = knn_graph(ds.points, k=15) # neighbour graph, any k
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "generator" / "datasets"

_REQUIRED_FILES = ("points.npy", "values.npy", "truth.json")
_TYPE_ALIASES = {"maxima": "max", "minima": "min", "saddles": "saddle"}


@dataclass(frozen=True)
class CriticalPoint:
    """One ground-truth critical point."""

    type: str                 # "max", "min" or "saddle"
    id: str                   # e.g. "max_0"
    center: np.ndarray        # (d,) float64 position
    extra: dict[str, Any] = field(default_factory=dict)  # amplitude, signs, ...


@dataclass
class Dataset:
    """A loaded scalar field sampled at points, with its ground truth."""

    name: str
    path: Path
    points: np.ndarray        # (N, d)
    values: np.ndarray        # (N,)
    truth: dict[str, Any]     # raw truth.json
    critical_points: list[CriticalPoint]
    extremum_graph_edges: list[tuple[str, str]]

    # ---- basic properties -------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def dimensions(self) -> int:
        return self.points.shape[1]

    @property
    def mode(self) -> str:
        return self.truth.get("mode", "unknown")

    @property
    def has_labels(self) -> bool:
        return bool(self.truth.get("has_extrema_labels", False))

    # ---- ground-truth accessors ------------------------------------------
    def _of_type(self, t: str) -> list[CriticalPoint]:
        return [c for c in self.critical_points if c.type == t]

    @property
    def maxima(self) -> list[CriticalPoint]:
        return self._of_type("max")

    @property
    def minima(self) -> list[CriticalPoint]:
        return self._of_type("min")

    @property
    def saddles(self) -> list[CriticalPoint]:
        return self._of_type("saddle")

    def centers(self, t: str | None = None) -> np.ndarray:
        """(M, d) array of ground-truth centers, optionally filtered by type."""
        cps = self.critical_points if t is None else self._of_type(t)
        if not cps:
            return np.empty((0, self.dimensions))
        return np.stack([c.center for c in cps])

    def nearest_point(self, center: np.ndarray) -> int:
        """Index of the sample closest to ``center``."""
        return int(np.argmin(np.sum((self.points - center) ** 2, axis=1)))

    def to_torch(self, device: str = "cpu") -> dict[str, Any]:
        """Return points/values as torch tensors."""
        import torch

        # np.array(...) copies, so memory-mapped (read-only) arrays work too.
        return {
            "points": torch.from_numpy(np.array(self.points)).to(device),
            "values": torch.from_numpy(np.array(self.values)).to(device),
        }

    def __repr__(self) -> str:
        counts = {t: len(self._of_type(t)) for t in ("max", "min", "saddle")}
        gt = ", ".join(f"{v} {k}" for k, v in counts.items() if v) or "none"
        return (
            f"Dataset({self.name!r}, mode={self.mode}, N={self.num_points}, "
            f"d={self.dimensions}, "
            f"values=[{float(self.values.min()):.3f}, {float(self.values.max()):.3f}], "
            f"ground_truth: {gt})"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _parse_critical_points(truth: dict[str, Any]) -> list[CriticalPoint]:
    """Normalise both truth.json layouts into a list of CriticalPoint."""
    out: list[CriticalPoint] = []

    # Layout 1 (cyclic_ring): flat list with explicit type and id.
    for cp in truth.get("critical_points", []):
        extra = {k: v for k, v in cp.items() if k not in ("type", "id", "center")}
        out.append(CriticalPoint(cp["type"], cp["id"], np.asarray(cp["center"], float), extra))

    # Layout 2 (controlled_extrema): dict of maxima / minima / saddles.
    extrema = truth.get("extrema") or truth.get("generator_metadata", {}).get("extrema")
    if extrema and not out:
        for key, items in extrema.items():
            t = _TYPE_ALIASES.get(key, key)
            for i, cp in enumerate(items):
                extra = {k: v for k, v in cp.items() if k != "center"}
                out.append(CriticalPoint(t, f"{t}_{i}", np.asarray(cp["center"], float), extra))

    return out


def list_datasets(root: str | Path = DEFAULT_ROOT) -> list[str]:
    """Names of all complete dataset folders under ``root`` (sorted)."""
    root = Path(root)
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and all((p / f).exists() for f in _REQUIRED_FILES)
    )


def load_dataset(name: str, root: str | Path = DEFAULT_ROOT, mmap: bool = False) -> Dataset:
    """Load one dataset by folder name.

    ``mmap=True`` memory-maps the arrays instead of reading them into RAM,
    useful for the large high-dimensional sets (ext_50d, ext_100d).
    """
    path = Path(root) / name
    missing = [f for f in _REQUIRED_FILES if not (path / f).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing {missing}")

    mode = "r" if mmap else None
    points = np.load(path / "points.npy", mmap_mode=mode)
    values = np.load(path / "values.npy", mmap_mode=mode)
    with open(path / "truth.json") as f:
        truth = json.load(f)

    _validate(name, points, values, truth)

    return Dataset(
        name=name,
        path=path,
        points=points,
        values=values,
        truth=truth,
        critical_points=_parse_critical_points(truth),
        extremum_graph_edges=[tuple(e) for e in truth.get("extremum_graph_edges", [])],
    )


def iter_datasets(root: str | Path = DEFAULT_ROOT, **kwargs) -> Iterator[Dataset]:
    """Yield every dataset under ``root``."""
    for name in list_datasets(root):
        yield load_dataset(name, root, **kwargs)


def _validate(name, points, values, truth) -> None:
    if points.ndim != 2:
        raise ValueError(f"{name}: points must be (N, d), got {points.shape}")
    n, d = points.shape
    if values.shape != (n,):
        raise ValueError(f"{name}: values shape {values.shape} != ({n},)")
    if truth.get("dimensions", d) != d:
        raise ValueError(f"{name}: truth says d={truth['dimensions']}, points have d={d}")
    if truth.get("num_points", n) != n:
        raise ValueError(f"{name}: truth says N={truth['num_points']}, points have N={n}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load datasets and (optionally) build their kNN graphs.")
    parser.add_argument("names", nargs="*", help="dataset names (default: all)")
    parser.add_argument("--root", default=DEFAULT_ROOT, type=Path)
    parser.add_argument("-k", type=int, default=None,
                        help="also build the k-nearest-neighbour graph")
    parser.add_argument("--mmap", action="store_true", help="memory-map arrays")
    args = parser.parse_args()

    for name in args.names or list_datasets(args.root):
        ds = load_dataset(name, args.root, mmap=args.mmap)
        print(ds)
        if ds.extremum_graph_edges:
            print(f"    expected extremum graph: {len(ds.extremum_graph_edges)} edges")
        if args.k:
            try:
                from pipeline.neighbor_graph import knn_graph
            except ImportError:  # run as a script: pipeline/ is on sys.path
                from neighbor_graph import knn_graph

            g = knn_graph(ds.points, args.k)
            print(f"    {g}, components={g.num_components()}")
