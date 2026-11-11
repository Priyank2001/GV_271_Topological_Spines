#!/usr/bin/env python3
"""
Grid-free synthetic scalar fields on point clouds.

A grid with `r` samples per axis in `d` dimensions needs r**d values
(r=10, d=100 is already 1e100). This module instead stores `N` sample
points in `d` dimensions and one scalar value per point, so memory grows
as N*d. Every field is an analytic function (or a smooth random function)
that can be evaluated at arbitrary points, so no grid is ever built.

Storage: float32 by default (halves memory); pass dtype=np.float64 if needed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch


@dataclass
class PointCloudField:
    """Scalar values attached to a point cloud."""
    points: np.ndarray            # (N, d)
    values: np.ndarray            # (N,)
    mode: str
    metadata: Dict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    @property
    def dimensions(self) -> int:
        return self.points.shape[1]


class PointCloudScalarFieldGenerator:
    """Generate scalar fields on point clouds in arbitrary dimensions."""

    def __init__(self, seed: Optional[int] = None, dtype=np.float32):
        self.rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.seed = seed

    # ------------------------------------------------------------------
    # Point sampling
    # ------------------------------------------------------------------
    def sample_points(
        self,
        num_points: int,
        dimensions: int,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        method: Literal["uniform", "gaussian"] = "uniform",
    ) -> np.ndarray:
        """Sample N points in d dimensions. Returns array of shape (N, d)."""
        lo, hi = bounds
        if method == "uniform":
            pts = self.rng.uniform(lo, hi, size=(num_points, dimensions))
        elif method == "gaussian":
            mid, half = (lo + hi) / 2, (hi - lo) / 4
            pts = self.rng.normal(mid, half, size=(num_points, dimensions))
            pts = np.clip(pts, lo, hi)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        return pts.astype(self.dtype, copy=False)

    # ------------------------------------------------------------------
    # Field generators (all evaluate f(x) at each point, no grid)
    # ------------------------------------------------------------------
    def random_values(
        self,
        points: np.ndarray,
        distribution: Literal["uniform", "gaussian", "exponential"] = "gaussian",
        value_range: Tuple[float, float] = (0.0, 1.0),
    ) -> PointCloudField:
        """Independent random value per point (no spatial correlation)."""
        n = points.shape[0]
        lo, hi = value_range
        if distribution == "uniform":
            vals = self.rng.uniform(lo, hi, n)
        elif distribution == "gaussian":
            vals = np.clip(self.rng.normal((lo + hi) / 2, (hi - lo) / 4, n), lo, hi)
        elif distribution == "exponential":
            vals = self.rng.exponential(1.0, n)
            vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
            vals = vals * (hi - lo) + lo
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        return self._make(points, vals, "random",
                          {"distribution": distribution, "range": value_range})

    def analytic(
        self,
        points: np.ndarray,
        function: Literal["gaussian_peaks", "rastrigin", "schwefel", "sphere"] = "gaussian_peaks",
        num_peaks: int = 5,
        domain: Tuple[float, float] = (-5.0, 5.0),
    ) -> PointCloudField:
        """
        Evaluate a benchmark function at the points.
        Points are assumed to lie in `domain`; they are used as-is.
        """
        x = points.astype(np.float64, copy=False)
        if function == "gaussian_peaks":
            vals = self._gaussian_peaks(x, num_peaks, domain)
        elif function == "rastrigin":
            vals = self._rastrigin(x)
        elif function == "schwefel":
            vals = self._schwefel(x)
        elif function == "sphere":
            vals = np.sum(x ** 2, axis=1)
        else:
            raise ValueError(f"Unknown function: {function}")
        vals = self._normalize(vals)
        return self._make(points, vals, "mathematical",
                          {"function": function, "num_peaks": num_peaks, "domain": domain})

    def smooth_random(
        self,
        points: np.ndarray,
        num_features: int = 256,
        length_scale: float = 1.0,
        chunk: int = 4096,
    ) -> PointCloudField:
        """
        Smooth random field via random Fourier features (approximate
        Gaussian process with RBF kernel of width `length_scale`).
        Memory: O(N * num_features), processed in chunks of points.
        """
        d = points.shape[1]
        w = self.rng.normal(0.0, 1.0 / length_scale, size=(num_features, d))
        b = self.rng.uniform(0.0, 2 * np.pi, size=num_features)
        a = self.rng.normal(0.0, 1.0, size=num_features)
        scale = np.sqrt(2.0 / num_features)

        vals = np.empty(points.shape[0], dtype=np.float64)
        for start in range(0, points.shape[0], chunk):
            block = points[start:start + chunk].astype(np.float64, copy=False)
            vals[start:start + chunk] = scale * (np.cos(block @ w.T + b) @ a)

        vals = self._normalize(vals)
        return self._make(points, vals, "smooth_random",
                          {"num_features": num_features, "length_scale": length_scale})

    def controlled_extrema(
        self,
        points: np.ndarray,
        num_maxima: int = 3,
        num_minima: int = 3,
        num_saddles: int = 2,
        width: float = 1.0,
        background: Optional[PointCloudField] = None,
    ) -> PointCloudField:
        """
        Add Gaussian bumps (maxima), dips (minima) and mixed-sign quadratic
        saddles at random centers. Centers are drawn from the points' bounding
        box. Each extremum is recorded with its center and value.

        A saddle is f = sum_k s_k (x_k - c_k)^2 with both signs present, which
        has a true saddle at c in any dimension d >= 2, windowed by a Gaussian
        so its influence stays local.
        """
        x = points.astype(np.float64, copy=False)
        n, d = x.shape
        lo, hi = x.min(axis=0), x.max(axis=0)

        base = background.values.astype(np.float64) if background is not None \
            else np.zeros(n)
        vals = base.copy()
        info = {"maxima": [], "minima": [], "saddles": []}

        def centers(k):
            return self.rng.uniform(lo, hi, size=(k, d))

        for c in centers(num_maxima):
            amp = self.rng.uniform(0.5, 2.0)
            vals += amp * self._bump(x, c, width)
            info["maxima"].append({"center": c.tolist(), "amplitude": float(amp)})

        for c in centers(num_minima):
            amp = self.rng.uniform(0.5, 2.0)
            vals -= amp * self._bump(x, c, width)
            info["minima"].append({"center": c.tolist(), "amplitude": float(amp)})

        for c in centers(num_saddles):
            signs = np.ones(d)
            signs[: d // 2] = -1.0           # half negative, half positive
            self.rng.shuffle(signs)
            quad = np.sum(signs * (x - c) ** 2, axis=1)
            vals += 0.5 * quad * self._bump(x, c, width * 2)
            info["saddles"].append({"center": c.tolist(), "signs": signs.tolist()})

        vals = self._normalize(vals)
        return self._make(points, vals, "controlled_extrema",
                          {"num_maxima": num_maxima, "num_minima": num_minima,
                           "num_saddles": num_saddles, "width": width,
                           "extrema": info})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bump(x: np.ndarray, c: np.ndarray, width: float) -> np.ndarray:
        return np.exp(-np.sum((x - c) ** 2, axis=1) / (2 * width ** 2))

    def _gaussian_peaks(self, x, num_peaks, domain):
        n, d = x.shape
        vals = np.zeros(n)
        for _ in range(num_peaks):
            c = self.rng.uniform(domain[0] + 1, domain[1] - 1, size=d)
            amp = self.rng.uniform(0.5, 2.0)
            var = self.rng.uniform(0.1, 1.0)
            vals += amp * self._bump(x, c, var)
        return vals / num_peaks

    @staticmethod
    def _rastrigin(x):
        A = 10.0
        d = x.shape[1]
        return A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=1)

    @staticmethod
    def _schwefel(x):
        d = x.shape[1]
        return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float64, copy=False)
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    def _make(self, points, values, mode, metadata):
        return PointCloudField(
            points=points,
            values=values.astype(self.dtype, copy=False),
            mode=mode,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # I/O and conversion
    # ------------------------------------------------------------------
    def save(self, field_: PointCloudField, output_dir: Path) -> Path:
        """Save as compressed NPZ: points, values, and metadata string."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"pc_dim{field_.dimensions}_n{field_.num_points}_{field_.mode}.npz"
        np.savez_compressed(path, points=field_.points, values=field_.values,
                            mode=field_.mode, metadata=str(field_.metadata))
        print(f"Saved to {path}")
        return path

    @staticmethod
    def to_torch(field_: PointCloudField) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (points, values) as float32 tensors."""
        return (torch.from_numpy(np.ascontiguousarray(field_.points)).float(),
                torch.from_numpy(np.ascontiguousarray(field_.values)).float())


def main():
    gen = PointCloudScalarFieldGenerator(seed=42)

    print("1) 100k points in 100D, Gaussian peaks")
    pts = gen.sample_points(100_000, 100, bounds=(-5, 5))
    f1 = gen.analytic(pts, "gaussian_peaks", num_peaks=5)
    print(f"   points {pts.shape}, values {f1.values.shape}, "
          f"range [{f1.values.min():.3f}, {f1.values.max():.3f}]")

    print("2) 20k points in 200D, smooth random field")
    pts2 = gen.sample_points(20_000, 200)
    f2 = gen.smooth_random(pts2, num_features=256, length_scale=3.0)
    print(f"   points {pts2.shape}, range [{f2.values.min():.3f}, {f2.values.max():.3f}]")

    print("3) 50k points in 50D, controlled extrema")
    pts3 = gen.sample_points(50_000, 50)
    f3 = gen.controlled_extrema(pts3, num_maxima=3, num_minima=3, num_saddles=2)
    print(f"   extrema: {len(f3.metadata['extrema']['maxima'])} max, "
          f"{len(f3.metadata['extrema']['minima'])} min, "
          f"{len(f3.metadata['extrema']['saddles'])} saddles")

    gen.save(f1, Path("output"))
    t_pts, t_vals = gen.to_torch(f1)
    print(f"Torch: points {tuple(t_pts.shape)}, values {tuple(t_vals.shape)}")


if __name__ == "__main__":
    main()
