"""Where the landscape is sampled: an axis-aligned box in subspace coordinates, and samplers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Box:
    lo: np.ndarray    # (d,)
    hi: np.ndarray    # (d,)

    @property
    def dim(self) -> int:
        return len(self.lo)

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    @classmethod
    def around(cls, anchors: np.ndarray, margin: float = 0.3, min_half_width: float = 0.0) -> "Box":
        """Bounding box of ``anchors`` grown by ``margin`` x its size on every side.
        Axes along which the anchors do not vary (e.g. random directions) get
        ``min_half_width`` on each side of the anchors' centre."""
        lo, hi = anchors.min(0), anchors.max(0)
        pad = margin * (hi - lo)
        lo, hi = lo - pad, hi + pad
        mid = 0.5 * (lo + hi)
        half = np.maximum(0.5 * (hi - lo), min_half_width)
        return cls(mid - half, mid + half)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Coordinates scaled so the box is the unit cube (for distances between results)."""
        return (x - self.lo) / self.size

    def to_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


def grid(box: Box, shape: Tuple[int, ...]) -> np.ndarray:
    """Points of a regular grid with ``shape`` vertices per axis, in C order (the order of
    ``pipeline.neighbor_graph.grid_graph`` / ``freudenthal_grid_graph``)."""
    axes = [np.linspace(l, h, n) for l, h, n in zip(box.lo, box.hi, shape)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], 1)


def uniform(box: Box, n: int, seed: int) -> np.ndarray:
    u = np.random.default_rng(seed).random((n, box.dim))
    return box.lo + u * box.size


def sobol(box: Box, n: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points; any prefix of length 2^m is itself balanced."""
    from scipy.stats import qmc
    u = qmc.Sobol(box.dim, scramble=True, seed=seed).random(n)
    return box.lo + u * box.size


SAMPLERS = {"uniform": uniform, "sobol": sobol}
