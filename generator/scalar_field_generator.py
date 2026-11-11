#!/usr/bin/env python3
"""
Synthetic scalar field generator for extreme graph visualization.
Supports multiple generation modes: random, Perlin noise, mathematical functions,
controllable extrema, and saddle points.
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Optional, Literal
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScalarField:
    """Container for scalar field data with metadata."""
    data: np.ndarray
    dimensions: int
    resolution: Tuple[int, ...]
    mode: str
    extrema: Optional[Dict] = None
    metadata: Optional[Dict] = None


class ScalarFieldGenerator:
    """Generate synthetic scalar fields in arbitrary dimensions."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.seed = seed

    def generate_random(
        self,
        dimensions: int,
        resolution: Tuple[int, ...],
        distribution: Literal["uniform", "gaussian", "exponential"] = "gaussian",
        range_: Tuple[float, float] = (0.0, 1.0),
    ) -> ScalarField:
        """
        Generate random scalar field.

        Args:
            dimensions: Number of dimensions
            resolution: Resolution per dimension (e.g., (100, 200) for 2D)
            distribution: Type of random distribution
            range_: Value range (min, max)
        """
        if len(resolution) != dimensions:
            raise ValueError(f"Resolution length {len(resolution)} != dimensions {dimensions}")

        if distribution == "uniform":
            data = np.random.uniform(range_[0], range_[1], resolution)
        elif distribution == "gaussian":
            data = np.random.normal((range_[0] + range_[1]) / 2,
                                   (range_[1] - range_[0]) / 4, resolution)
            data = np.clip(data, range_[0], range_[1])
        elif distribution == "exponential":
            data = np.random.exponential(scale=1.0, size=resolution)
            data = (data - data.min()) / (data.max() - data.min())
            data = data * (range_[1] - range_[0]) + range_[0]
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return ScalarField(
            data=data,
            dimensions=dimensions,
            resolution=resolution,
            mode="random",
            metadata={"distribution": distribution, "range": range_},
        )

    def generate_perlin_noise(
        self,
        dimensions: int,
        resolution: Tuple[int, ...],
        scale: float = 1.0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ) -> ScalarField:
        """
        Generate Perlin-like noise using fractional Brownian motion.

        Args:
            dimensions: Number of dimensions
            resolution: Resolution per dimension
            scale: Noise scale
            octaves: Number of noise layers
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
        """
        if len(resolution) != dimensions:
            raise ValueError(f"Resolution length {len(resolution)} != dimensions {dimensions}")

        # Generate base noise with FFT-based method (works in any dimension)
        data = np.zeros(resolution)
        amplitude = 1.0
        frequency = scale
        max_value = 0.0

        for _ in range(octaves):
            # Create random phase for this octave
            noise = np.random.randn(*resolution)

            # Apply frequency scaling via coordinate transformation
            grid_coords = np.ogrid[tuple(slice(0, res) for res in resolution)]
            scaled_coords = [c * frequency / res for c, res in zip(grid_coords, resolution)]

            # Simple frequency-scaled noise
            freq_noise = noise * amplitude
            data += freq_noise
            max_value += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        data = data / max_value  # Normalize
        data = (data - data.min()) / (data.max() - data.min())

        return ScalarField(
            data=data,
            dimensions=dimensions,
            resolution=resolution,
            mode="perlin_noise",
            metadata={"scale": scale, "octaves": octaves,
                     "persistence": persistence, "lacunarity": lacunarity},
        )

    def generate_mathematical(
        self,
        dimensions: int,
        resolution: Tuple[int, ...],
        function: Literal["gaussian_peaks", "rastrigin", "schwefel", "sphere"] = "gaussian_peaks",
        num_peaks: int = 5,
    ) -> ScalarField:
        """
        Generate scalar field from mathematical functions.

        Args:
            dimensions: Number of dimensions
            resolution: Resolution per dimension
            function: Type of mathematical function
            num_peaks: Number of peaks (for gaussian_peaks)
        """
        if len(resolution) != dimensions:
            raise ValueError(f"Resolution length {len(resolution)} != dimensions {dimensions}")

        # Create coordinate grids normalized to [-5, 5] range
        grids = np.meshgrid(
            *[np.linspace(-5, 5, res) for res in resolution],
            indexing="ij"
        )
        coords = np.stack(grids, axis=0)

        if function == "gaussian_peaks":
            data = self._gaussian_peaks(coords, num_peaks)
        elif function == "rastrigin":
            data = self._rastrigin(coords)
        elif function == "schwefel":
            data = self._schwefel(coords)
        elif function == "sphere":
            data = self._sphere(coords)
        else:
            raise ValueError(f"Unknown function: {function}")

        return ScalarField(
            data=data,
            dimensions=dimensions,
            resolution=resolution,
            mode="mathematical",
            metadata={"function": function, "num_peaks": num_peaks},
        )

    @staticmethod
    def _gaussian_peaks(coords: np.ndarray, num_peaks: int) -> np.ndarray:
        """Gaussian peaks function."""
        data = np.zeros(coords.shape[1:])
        for _ in range(num_peaks):
            center = np.random.uniform(-4, 4, size=coords.shape[0])
            amplitude = np.random.uniform(0.5, 2.0)
            variance = np.random.uniform(0.1, 1.0)

            # L2 distance from center
            dist_sq = np.sum((coords - center[:, None, None]) ** 2, axis=0)
            data += amplitude * np.exp(-dist_sq / (2 * variance ** 2))

        return data / num_peaks

    @staticmethod
    def _rastrigin(coords: np.ndarray) -> np.ndarray:
        """Rastrigin function (many local minima)."""
        A = 10
        n = coords.shape[0]
        data = A * n + np.sum(coords ** 2 - A * np.cos(2 * np.pi * coords), axis=0)
        return data / (data.max() - data.min() + 1e-8)

    @staticmethod
    def _schwefel(coords: np.ndarray) -> np.ndarray:
        """Schwefel function."""
        n = coords.shape[0]
        data = 418.9829 * n - np.sum(coords * np.sin(np.sqrt(np.abs(coords))), axis=0)
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    @staticmethod
    def _sphere(coords: np.ndarray) -> np.ndarray:
        """Simple sphere function."""
        data = np.sum(coords ** 2, axis=0)
        return data / (data.max() - data.min() + 1e-8)

    def generate_with_extrema(
        self,
        dimensions: int,
        resolution: Tuple[int, ...],
        num_maxima: int = 3,
        num_minima: int = 3,
        num_saddles: int = 2,
        background_mode: str = "smooth",
    ) -> ScalarField:
        """
        Generate scalar field with controllable extrema and saddle points.

        Args:
            dimensions: Number of dimensions
            resolution: Resolution per dimension
            num_maxima: Number of maximum points
            num_minima: Number of minimum points
            num_saddles: Number of saddle points
            background_mode: "smooth" or "noisy"
        """
        if len(resolution) != dimensions:
            raise ValueError(f"Resolution length {len(resolution)} != dimensions {dimensions}")

        # Start with smooth background
        if background_mode == "smooth":
            data = self._create_smooth_background(resolution)
        else:
            data = np.random.randn(*resolution)

        # Place extrema
        extrema_info = {"maxima": [], "minima": [], "saddles": []}

        for _ in range(num_maxima):
            idx = tuple(np.random.randint(0, res) for res in resolution)
            amplitude = np.random.uniform(0.5, 2.0)
            data[idx] = max(data[idx], amplitude)
            extrema_info["maxima"].append({"index": idx, "value": float(data[idx])})

        for _ in range(num_minima):
            idx = tuple(np.random.randint(0, res) for res in resolution)
            amplitude = np.random.uniform(-2.0, -0.5)
            data[idx] = min(data[idx], amplitude)
            extrema_info["minima"].append({"index": idx, "value": float(data[idx])})

        # Saddle points (approximated)
        for _ in range(num_saddles):
            idx = tuple(np.random.randint(0, res) for res in resolution)
            extrema_info["saddles"].append({"index": idx, "value": float(data[idx])})

        # Normalize
        data = (data - data.min()) / (data.max() - data.min())

        return ScalarField(
            data=data,
            dimensions=dimensions,
            resolution=resolution,
            mode="controlled_extrema",
            extrema=extrema_info,
            metadata={"num_maxima": num_maxima, "num_minima": num_minima,
                     "num_saddles": num_saddles, "background": background_mode},
        )

    @staticmethod
    def _create_smooth_background(resolution: Tuple[int, ...]) -> np.ndarray:
        """Create smooth background using interpolated random points."""
        grids = np.meshgrid(
            *[np.linspace(0, 1, res) for res in resolution],
            indexing="ij"
        )
        coords = np.stack(grids, axis=0)

        # Smooth function combining coordinates
        data = np.sum(np.sin(3 * np.pi * coords) * coords, axis=0)
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    def save(self, field: ScalarField, output_dir: Path, format: Literal["npz", "npy", "json"] = "npz"):
        """Save scalar field to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"field_dim{field.dimensions}_{field.mode}"

        if format == "npz":
            filepath = output_dir / f"{filename}.npz"
            np.savez_compressed(
                filepath,
                data=field.data,
                dimensions=field.dimensions,
                resolution=field.resolution,
                mode=field.mode,
                extrema=str(field.extrema),
                metadata=str(field.metadata),
            )
        elif format == "npy":
            filepath = output_dir / f"{filename}.npy"
            np.save(filepath, field.data)
        elif format == "json":
            import json
            filepath = output_dir / f"{filename}.json"
            save_data = {
                "dimensions": field.dimensions,
                "resolution": field.resolution,
                "mode": field.mode,
                "extrema": field.extrema,
                "metadata": field.metadata,
                "data_shape": field.data.shape,
                "data_min": float(field.data.min()),
                "data_max": float(field.data.max()),
                "data_mean": float(field.data.mean()),
            }
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)

        print(f"Saved to {filepath}")
        return filepath

    def to_torch(self, field: ScalarField) -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.from_numpy(field.data).float()


def main():
    """Example usage of the generator."""
    import sys

    gen = ScalarFieldGenerator(seed=42)

    # Example 1: Random field
    print("Generating random field...")
    field1 = gen.generate_random(dimensions=2, resolution=(128, 128), distribution="gaussian")
    print(f"  Shape: {field1.data.shape}, Range: [{field1.data.min():.3f}, {field1.data.max():.3f}]")

    # Example 2: Perlin noise
    print("Generating Perlin noise...")
    field2 = gen.generate_perlin_noise(dimensions=2, resolution=(128, 128), octaves=5)
    print(f"  Shape: {field2.data.shape}, Range: [{field2.data.min():.3f}, {field2.data.max():.3f}]")

    # Example 3: Mathematical function
    print("Generating mathematical function...")
    field3 = gen.generate_mathematical(dimensions=2, resolution=(128, 128),
                                       function="gaussian_peaks", num_peaks=5)
    print(f"  Shape: {field3.data.shape}, Range: [{field3.data.min():.3f}, {field3.data.max():.3f}]")

    # Example 4: Controlled extrema
    print("Generating with controlled extrema...")
    field4 = gen.generate_with_extrema(dimensions=2, resolution=(100, 100),
                                       num_maxima=3, num_minima=3, num_saddles=2)
    print(f"  Shape: {field4.data.shape}, Range: [{field4.data.min():.3f}, {field4.data.max():.3f}]")
    print(f"  Extrema: {field4.extrema}")

    # Example 5: High-dimensional field
    print("Generating high-dimensional field...")
    field5 = gen.generate_random(dimensions=5, resolution=(50, 50, 50, 50, 50))
    print(f"  Shape: {field5.data.shape}, Range: [{field5.data.min():.3f}, {field5.data.max():.3f}]")

    # Save examples
    output_dir = Path("output")
    print("\nSaving examples...")
    gen.save(field1, output_dir, format="npz")
    gen.save(field3, output_dir, format="json")

    # Convert to torch
    print("\nConverting to PyTorch...")
    torch_field = gen.to_torch(field1)
    print(f"  PyTorch tensor shape: {torch_field.shape}, dtype: {torch_field.dtype}")


if __name__ == "__main__":
    main()
