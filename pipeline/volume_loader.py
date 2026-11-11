"""
Download and load volumes from the Open SciVis Datasets collection.

The collection is at http://klacansky.com/open-scivis-datasets/ (maintained
by Pavol Klacansky). Every dataset is one headerless ``.raw`` file. Its grid
size, value type and voxel spacing are in the collection's ``datasets.json``,
which is cached in ``data/klacansky/datasets.json``.

Layout of a ``.raw`` file: ``size = [nx, ny, nz]`` with x varying fastest,
so the array is read with shape ``(nz, ny, nx)``. ``Volume.values`` keeps
that (z, y, x) order, and the flat index of a voxel matches
``pipeline.neighbor_graph.grid_graph(volume.shape)``.

Note: on 2026-11-09 the server refused HTTPS connections but served plain
HTTP, so the URLs from ``datasets.json`` (which are ``http://``) are used as
they are.

Usage
-----
    from pipeline.volume_loader import download, load_volume
    from pipeline.neighbor_graph import grid_graph
    from pipeline.extremum_graph import extremum_graph

    download("fuel")                       # no-op if the file is already there
    vol = load_volume("fuel")              # values scaled to [0, 1]
    eg = extremum_graph(vol.values.reshape(-1), grid_graph(vol.shape))
    vol.coords(eg.peaks)                   # (P, 3) voxel positions as (x, y, z)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "klacansky"
INDEX_URL = "http://klacansky.com/open-scivis-datasets/datasets.json"

# Volumes that are not in the Open SciVis index. Same fields as its datasets.json entries
# (``name`` is the key). ``bucky`` is the C60 (Buckminsterfullerene) electron density from the
# Erlangen Volume Library (originally AVS), the "Bucky Ball" of Correa et al. 2011; the copy is the
# one a UMBC course page serves with a .mhd header (DimSize 32 32 32, MET_UCHAR).
EXTRA_VOLUMES = {
    "bucky": {"name": "Bucky Ball", "type": "uint8", "size": [32, 32, 32], "spacing": [1, 1, 1],
              "description": "Simulated electron density of a C60 (Buckminster-fullerene) molecule.",
              "url": "https://userpages.cs.umbc.edu/olano/class/491-19-8/bucky.raw"},
}

# Small volumes (at most 128^3) used by notebooks/klacansky_extremum_graphs.ipynb.
SMALL_DATASETS = ("nucleon", "marschner_lobb", "silicium", "fuel", "neghip", "hydrogen_atom")


@dataclass
class Volume:
    name: str
    values: np.ndarray        # (nz, ny, nx) float64, scaled to [0, 1] unless raw=True
    spacing: np.ndarray       # (3,) voxel size along (x, y, z)
    dtype: str                # value type of the raw file
    raw_range: tuple          # (min, max) of the raw values before scaling
    description: str = ""

    @property
    def shape(self) -> tuple:
        return self.values.shape

    @property
    def num_voxels(self) -> int:
        return int(self.values.size)

    def coords(self, vertices: np.ndarray) -> np.ndarray:
        """(n, 3) positions (x, y, z) in voxel units of flat vertex indices."""
        z, y, x = np.unravel_index(np.asarray(vertices), self.shape)
        return np.stack([x, y, z], axis=-1).astype(np.float64)

    def __repr__(self) -> str:
        nz, ny, nx = self.shape
        return (f"Volume({self.name!r}, {nx}x{ny}x{nz} {self.dtype}, "
                f"raw range {self.raw_range[0]:g}..{self.raw_range[1]:g})")


def _index(root: Path) -> dict:
    path = root / "datasets.json"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(INDEX_URL, path)
    with open(path) as f:
        return {**json.load(f), **EXTRA_VOLUMES}


def list_volumes(root: str | Path = DEFAULT_ROOT, max_voxels: int | None = None) -> list[str]:
    """Names in the collection, optionally only those with at most ``max_voxels`` voxels."""
    index = _index(Path(root))
    return sorted(n for n, m in index.items()
                  if max_voxels is None or int(np.prod(m["size"])) <= max_voxels)


def _raw_path(meta: dict, root: Path) -> Path:
    return root / meta["url"].rsplit("/", 1)[-1]


def download(name: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Download the ``.raw`` file of ``name`` into ``root`` unless it is already there."""
    root = Path(root)
    meta = _index(root)[name]
    path = _raw_path(meta, root)
    expected = int(np.prod(meta["size"])) * np.dtype(meta["type"]).itemsize
    if path.exists() and path.stat().st_size == expected:
        return path
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(meta["url"], tmp)
    if tmp.stat().st_size != expected:
        raise IOError(f"{name}: got {tmp.stat().st_size} bytes, expected {expected}")
    tmp.rename(path)
    return path


def load_volume(name: str, root: str | Path = DEFAULT_ROOT, raw: bool = False) -> Volume:
    """Read a downloaded volume. Values are scaled to [0, 1] unless ``raw``."""
    root = Path(root)
    meta = _index(root)[name]
    path = _raw_path(meta, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; call download({name!r}) first")
    nx_, ny_, nz_ = meta["size"]
    data = np.fromfile(path, dtype=meta["type"]).reshape(nz_, ny_, nx_)
    lo, hi = float(data.min()), float(data.max())
    values = data.astype(np.float64)
    if not raw and hi > lo:
        values = (values - lo) / (hi - lo)
    return Volume(name=name, values=values, spacing=np.asarray(meta["spacing"], float),
                  dtype=meta["type"], raw_range=(lo, hi),
                  description=meta.get("description", ""))
