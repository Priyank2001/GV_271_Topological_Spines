# Scalar Field Generator

Synthetic scalar field generator for extreme graph visualization in ISC graphics coursework.

## Setup

Activate the conda environment:
```bash
conda activate gv271
```

Or use the convenience script:
```bash
source activate.sh
```

## Features

The generator supports multiple modes for creating synthetic scalar fields in arbitrary dimensions (tested up to 200D):

### 1. **Random Fields**
- Uniform, Gaussian, or Exponential distributions
- Configurable value ranges

### 2. **Perlin Noise**
- Fractional Brownian motion (fBm)
- Configurable octaves, persistence, lacunarity
- Works in arbitrary dimensions

### 3. **Mathematical Functions**
- **Gaussian Peaks**: Configurable peaks at random locations
- **Rastrigin**: Many local minima (optimization benchmark)
- **Schwefel**: Complex landscape
- **Sphere**: Simple convex function

### 4. **Controlled Extrema**
- Place maxima, minima, and saddle points explicitly
- Smooth or noisy background
- Full metadata tracking

## Usage

### Basic Example
```python
from scalar_field_generator import ScalarFieldGenerator
import numpy as np

gen = ScalarFieldGenerator(seed=42)

# Generate 2D field
field = gen.generate_random(
    dimensions=2,
    resolution=(100, 200),
    distribution="gaussian"
)

print(field.data.shape)  # (100, 200)
print(field.data.min(), field.data.max())

# Convert to PyTorch tensor
import torch
tensor = gen.to_torch(field)
```

### Run Examples
```bash
python scalar_field_generator.py
```

This generates multiple fields and saves them to `output/` directory.

### Generate High-Dimensional Fields
```python
# 100D field with 50 points per dimension
field = gen.generate_random(
    dimensions=100,
    resolution=(50,) * 100  # or tuple([50] * 100)
)

# Save as compressed NPZ (efficient storage)
gen.save(field, "output", format="npz")
```

### Save Formats
- **NPZ**: Compressed, preserves all metadata (recommended for large fields)
- **NPY**: Raw numpy array only
- **JSON**: Metadata only (useful for inspection)

## Next Steps

Once testing is complete, the next phases are:
1. Extract extrema (maxima, minima, saddle points) using PyTorch
2. Build web visualization (likely Three.js or D3)
3. Handle TTK/Taki replacement for high-dimensional topology

## Environment Contents

```
gv271/  (conda environment)
├── numpy 1.24+
├── scipy 1.10+
├── torch 2.0+
├── scikit-learn 1.3+
├── matplotlib 3.7+
└── pillow 10.0+
```

## Dependencies

All dependencies are pre-installed in the `gv271` conda environment. To add more:
```bash
conda install -n gv271 <package_name>
```
