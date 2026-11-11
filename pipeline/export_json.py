"""
Export an extremum graph to JSON for the Three.js front end.

One file describes one field. It holds the graph, the region of every point
and (optionally) what the data view needs to draw the field. JSON cannot
store ``Infinity``, so the persistence of the global maximum is ``null``.

Schema (``"version": 1``)
-------------------------
    {
      "version": 1,
      "meta": {
        "name": "cyclic_3d",
        "kind": "points" | "grid",       # point cloud or regular grid
        "num_points": N,
        "dimensions": d,
        "grid_shape": [nz, ny, nx],      # grids only, flat index = (z, y, x) C order
        "value_range": [lo, hi],
        "k": 20,                         # kNN graphs only
        "simplify_threshold": 0.04,      # null if the graph is not simplified
        "initial_threshold": 0.1         # persistence the viewer opens at (>= simplify_threshold)
      },
      "peaks": [                         # id = list index, 0 = global maximum
        {"id": 0, "vertex": 17, "value": 1.0, "persistence": null,
         "parent": -1, "size": 5120, "position": [x, y, ...]}, ...
      ],
      "saddles": [                       # highest first
        {"id": 0, "vertex": 903, "value": 0.95, "peaks": [0, 3],
         "persistence": 0.9, "is_merge": true, "support": 14, "relief": 1.0,
         "saturation": 0.9,              # saturated persistence p* >= persistence
         "position": [x, y, ...]}, ...
      ],
      "region": [peak id of every point],            # length N, omitted if not requested
      "persistence_curve": [descending persistence of every peak of the
                            unsimplified graph, without the global maximum],
      "data": { ... },                               # see points_data / volume_data
                                                     # point clouds: "values" plus "embeddings",
                                                     # {reducer name: flat [x0, y0, x1, y1, ...]}
                                                     # scaled into [-1, 1], for the scatterplot
      "spine": {                                     # 2D layout, see pipeline/spine_layout.py
        "method": "pca+stress",
        "peaks": [[x, y], ...], "saddles": [[x, y], ...],   # fitted into [-1, 1]
        "scale": 3.1,                  # data distance per layout unit
        "stress": 0.04, "pca_stress": 0.09,   # normalised, lower is better
        "crossings": 0, "iterations": 31
      }
    }

``position`` is in the original space: sample coordinates for point
clouds, voxel coordinates ``(x, y, z)`` for grids. ``peaks[i].size`` is the
number of points in the region of peak ``i``. A saddle's ``peaks`` are
``[higher, lower]``.

Command line (from the repo root):

    python3 -m pipeline.export_json cyclic_3d -k 20 --simplify 0.04 -o out/cyclic_3d.json
    python3 -m pipeline.export_json --volume fuel --simplify 0.1 -o out/fuel.json
    python3 -m pipeline.export_json ext_50d -k 20 --embed pca --simplify 0.01 -o out/e.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from .extremum_graph import ExtremumGraph
from .spine_layout import SpineLayout

SCHEMA_VERSION = 1

Coords = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


# ---------------------------------------------------------------------------
# Building the dict
# ---------------------------------------------------------------------------
def extremum_graph_to_dict(
    eg: ExtremumGraph,
    *,
    coords: Optional[Coords] = None,
    meta: Optional[dict] = None,
    include_region: bool = True,
    full_peak_persistence: Optional[np.ndarray] = None,
    data: Optional[dict] = None,
    spine: Optional[SpineLayout] = None,
    saddle_extra: Optional[dict] = None,
    decimals: int = 6,
) -> dict:
    """Turn ``eg`` into a JSON-ready dict (see the module docstring).

    coords:  ``(N, d)`` array of point positions, or a function mapping vertex
             indices to ``(n, d)`` positions (use this for large grids).
             Without it, nodes have no ``position``.
    meta:    extra entries for ``meta`` (name, kind, grid_shape, k, ...).
    full_peak_persistence: ``peak_persistence`` of the unsimplified graph;
             becomes ``persistence_curve`` so the browser can draw the
             persistence plot even when ``eg`` is already simplified.
    data:    the ``data`` section, from ``points_data`` or ``volume_data``.
    spine:   a ``SpineLayout`` of this same graph; becomes the ``spine`` section.
    saddle_extra: ``{field name: one JSON value per saddle}``, extra fields
             added to every saddle (e.g. what another extraction method knows).
    """
    if spine is not None and (len(spine.peak_xy) != eg.num_peaks
                              or len(spine.saddle_xy) != eg.num_saddles):
        raise ValueError("spine layout was computed for a different graph")

    def pos(vertices: np.ndarray) -> Optional[np.ndarray]:
        if coords is None:
            return None
        xyz = coords(vertices) if callable(coords) else np.asarray(coords)[vertices]
        return np.round(np.asarray(xyz, dtype=np.float64), 4)

    def r(a: np.ndarray) -> list:
        return _finite_or_none(np.round(np.asarray(a, dtype=np.float64), decimals))

    peak_pos, saddle_pos = pos(eg.peaks), pos(eg.saddles)
    peak_persistence = r(eg.peak_persistence)
    saddle_persistence = r(eg.saddle_persistence)
    peak_values, saddle_values = r(eg.peak_values), r(eg.saddle_values)
    relief = r(eg.saddle_relief)
    saturation = r(eg.saddle_saturation)

    peaks = []
    for i in range(eg.num_peaks):
        node = {"id": i, "vertex": int(eg.peaks[i]), "value": peak_values[i],
                "persistence": peak_persistence[i], "parent": int(eg.peak_parent[i]),
                "size": int(eg.region_size[i])}
        if peak_pos is not None:
            node["position"] = peak_pos[i].tolist()
        peaks.append(node)

    saddles = []
    for i in range(eg.num_saddles):
        node = {"id": i, "vertex": int(eg.saddles[i]), "value": saddle_values[i],
                "peaks": [int(eg.saddle_peaks[i, 0]), int(eg.saddle_peaks[i, 1])],
                "persistence": saddle_persistence[i], "is_merge": bool(eg.saddle_is_merge[i]),
                "support": int(eg.saddle_support[i]), "relief": relief[i],
                "saturation": saturation[i]}
        if saddle_pos is not None:
            node["position"] = saddle_pos[i].tolist()
        for name, column in (saddle_extra or {}).items():
            node[name] = column[i]
        saddles.append(node)

    values = np.asarray(eg.values, dtype=np.float64)
    out_meta: dict[str, Any] = {
        "kind": "points",
        "num_points": int(eg.num_points),
        "value_range": [float(values.min()), float(values.max())],
        "simplify_threshold": None,
    }
    out_meta.update(meta or {})

    out: dict[str, Any] = {"version": SCHEMA_VERSION, "meta": out_meta,
                           "peaks": peaks, "saddles": saddles}
    if include_region:
        out["region"] = eg.region.tolist()
    if full_peak_persistence is not None:
        pers = np.sort(np.asarray(full_peak_persistence, dtype=np.float64))[::-1]
        out["persistence_curve"] = np.round(pers[np.isfinite(pers)], decimals).tolist()
    if data is not None:
        out["data"] = data
    if spine is not None:
        out["spine"] = spine.to_dict()
    return out


def points_data(values: np.ndarray, embedding: Optional[np.ndarray] = None,
                decimals: int = 4, embeddings: Optional[dict] = None) -> dict:
    """``data`` section for a point cloud: the field values and 2D embeddings
    for the scatterplot.

    ``embedding`` is one ``(N, m)`` array, kept as ``data.embedding``.
    ``embeddings`` maps a reducer name to an ``(N, 2)`` array; each is scaled
    into [-1, 1] (one factor for both axes, so shapes are not distorted),
    rounded to 3 decimals and stored flat as ``data.embeddings[name]``.
    The first name is the viewer's default (``data.embedding_default``).
    """
    out: dict[str, Any] = {"values": np.round(np.asarray(values, np.float64), decimals).tolist()}
    if embedding is not None:
        embedding = np.asarray(embedding)
        if embedding.ndim != 2 or embedding.shape[0] != len(values):
            raise ValueError(f"embedding must be (N, m), got {embedding.shape}")
        out["embedding"] = np.round(embedding.astype(np.float64), decimals).tolist()
    if embeddings:
        out["embeddings"] = {}
        for name, e in embeddings.items():
            e = np.asarray(e, np.float64)
            if e.shape != (len(values), 2):
                raise ValueError(f"embedding {name!r} must be (N, 2), got {e.shape}")
            e = e - (e.min(axis=0) + e.max(axis=0)) / 2
            half = float(np.abs(e).max()) or 1.0
            out["embeddings"][name] = np.round(e / half, 3).reshape(-1).tolist()
        out["embedding_default"] = next(iter(embeddings))
    return out


# Slow reducers are fitted on a subset and the rest of the points are placed afterwards.
_FIT_SAMPLE = {"isomap": 3000, "tsne": 5000, "kernel_pca": 3000, "umap": 20000}


def embed_points(points: np.ndarray, method: str, seed: int = 0) -> np.ndarray:
    """2D embedding of ``points`` with the named reducer (see ``dim_reduction``)."""
    from .dim_reduction import make_reducer

    kwargs: dict[str, Any] = {"n_components": 2}
    key = {"t-sne": "tsne", "kpca": "kernel_pca"}.get(method, method)
    if key in _FIT_SAMPLE and len(points) > _FIT_SAMPLE[key]:
        kwargs["fit_sample_size"] = _FIT_SAMPLE[key]
    return make_reducer(method, **kwargs).fit_transform(points)


def volume_data(volume, decimals: int = 4) -> dict:
    """``data`` section for a regular grid (a ``pipeline.volume_loader.Volume``):
    shape, spacing and the values scaled to [0, 1], flat in (z, y, x) order."""
    return {"shape": [int(s) for s in volume.shape],
            "spacing": [float(s) for s in volume.spacing],
            "values": np.round(volume.values.reshape(-1), decimals).tolist()}


def _finite_or_none(a: np.ndarray) -> list:
    """List of floats with ``inf`` replaced by ``None`` (JSON has no Infinity)."""
    return [float(x) if np.isfinite(x) else None for x in a]


# ---------------------------------------------------------------------------
# Writing / reading
# ---------------------------------------------------------------------------
def write_json(obj: dict, path: Union[str, Path], indent: Optional[int] = None) -> Path:
    """Write ``obj`` to ``path`` (parent folders are created). Compact by default.

    ``allow_nan=False`` makes a stray NaN or Infinity an error here instead
    of a file the browser cannot parse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, allow_nan=False, indent=indent,
                  separators=(",", ":") if indent is None else None)
    return path


def export_extremum_graph(eg: ExtremumGraph, path: Union[str, Path], *,
                          indent: Optional[int] = None, **kwargs) -> Path:
    """``extremum_graph_to_dict(eg, **kwargs)`` written to ``path``."""
    return write_json(extremum_graph_to_dict(eg, **kwargs), path, indent=indent)


def check_export(doc: dict) -> None:
    """Raise ``ValueError`` if ``doc`` breaks the schema's internal rules.

    Cheap consistency checks, used by the tests and the command line.
    """
    if doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"unknown version {doc.get('version')!r}")
    peaks, saddles, n = doc["peaks"], doc["saddles"], doc["meta"]["num_points"]
    for i, p in enumerate(peaks):
        if p["id"] != i:
            raise ValueError(f"peak {i} has id {p['id']}")
        if not -1 <= p["parent"] < i:
            raise ValueError(f"peak {i}: parent {p['parent']} must be -1 or a higher peak")
    if sum(p["size"] for p in peaks) != n:
        raise ValueError("peak sizes do not add up to num_points")
    for i, s in enumerate(saddles):
        a, b = s["peaks"]
        if s["id"] != i or not (0 <= a < b < len(peaks)):
            raise ValueError(f"saddle {i}: bad peak pair {s['peaks']}")
        if s["value"] > peaks[b]["value"] + 1e-9:
            raise ValueError(f"saddle {i} is above its lower peak")
        if "saturation" in s and s["saturation"] < s["persistence"] - 1e-9:
            raise ValueError(f"saddle {i}: saturated persistence below its persistence")
    if "spine" in doc:
        sp = doc["spine"]
        if len(sp["peaks"]) != len(peaks) or len(sp["saddles"]) != len(saddles):
            raise ValueError("spine has a different number of nodes than the graph")
        if any(len(xy) != 2 for xy in sp["peaks"] + sp["saddles"]):
            raise ValueError("spine nodes must be [x, y] pairs")
    if "region" in doc:
        region = np.asarray(doc["region"])
        if region.shape != (n,) or region.min() < 0 or region.max() >= len(peaks):
            raise ValueError("region has the wrong length or out-of-range peak ids")
        if not np.array_equal(np.bincount(region, minlength=len(peaks)),
                              [p["size"] for p in peaks]):
            raise ValueError("region does not match the peak sizes")


# ---------------------------------------------------------------------------
# Whole pipeline, and the command line
# ---------------------------------------------------------------------------
def build_document(dataset: Optional[str] = None, volume: Optional[str] = None, *,
                   k: int = 20, simplify: Optional[float] = None,
                   initial_threshold: Optional[float] = None,
                   embed: Union[None, str, list] = None,
                   spine: bool = True, region: bool = True, data: bool = True,
                   verbose: bool = False) -> dict:
    """Load a generator dataset or an Open SciVis volume, compute its extremum
    graph, and return the export dict.

    simplify:          drop peaks below this persistence (the file's base graph).
                       The browser can only simplify further, never restore.
    initial_threshold: persistence the viewer opens at (``meta.initial_threshold``);
                       defaults to ``simplify``.
    embed:             point clouds: reducer name, or a list of names, for 2D embeddings
                       of the points (the viewer's dimensionality-reduction choice).
                       They are for drawing only; the graph is built in the original space.
    """
    from .extremum_graph import extremum_graph
    from .neighbor_graph import grid_graph, knn_graph
    from .spine_layout import spine_layout

    if bool(dataset) == bool(volume):
        raise ValueError("give exactly one of dataset or volume")

    if volume:
        from .volume_loader import download, load_volume

        download(volume)
        vol = load_volume(volume)
        raw = extremum_graph(vol.values.reshape(-1), grid_graph(vol.shape))
        coords: Coords = vol.coords
        meta = {"name": volume, "kind": "grid", "dimensions": 3,
                "grid_shape": [int(s) for s in vol.shape],
                "spacing": [float(s) for s in vol.spacing]}
        section = volume_data(vol) if data else None
    else:
        from .data_loader import load_dataset

        ds = load_dataset(dataset)
        values = np.asarray(ds.values)
        raw = extremum_graph(values, knn_graph(ds.points, k))
        coords = np.asarray(ds.points)
        meta = {"name": ds.name, "kind": "points", "dimensions": int(ds.dimensions), "k": k}
        embeddings = {}
        for name in ([embed] if isinstance(embed, str) else embed or []):
            embeddings[name] = embed_points(coords, name)
        section = points_data(values, embeddings=embeddings) if data else None

    eg = raw.simplify(simplify) if simplify is not None else raw
    meta["simplify_threshold"] = simplify
    meta["initial_threshold"] = initial_threshold if initial_threshold is not None else simplify

    layout = None
    if spine:
        layout = spine_layout(eg, coords,
                              spacing=np.asarray(meta["spacing"]) if "spacing" in meta else None)
        if verbose:
            print(layout)

    doc = extremum_graph_to_dict(eg, coords=coords, meta=meta, include_region=region,
                                 full_peak_persistence=raw.peak_persistence,
                                 data=section, spine=layout)
    check_export(doc)
    if verbose:
        print(eg)
    return doc


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute an extremum graph and write it as JSON for the web viewer.")
    ap.add_argument("dataset", nargs="?", help="generator dataset name (e.g. cyclic_3d)")
    ap.add_argument("--volume", help="Open SciVis volume name (e.g. fuel) instead of a dataset")
    ap.add_argument("-k", type=int, default=20, help="neighbours per point (point clouds only)")
    ap.add_argument("--simplify", type=float, default=None,
                    help="drop peaks with persistence below this value")
    ap.add_argument("--initial", type=float, default=None,
                    help="persistence the viewer opens at (default: --simplify)")
    ap.add_argument("--embed", default=None, nargs="+",
                    help="point clouds: 2D embeddings for the scatterplot, one per reducer "
                         "name (e.g. pca tsne isomap); the viewer offers them as a choice")
    ap.add_argument("--no-spine", action="store_true", help="leave out the 2D spine layout")
    ap.add_argument("--no-region", action="store_true", help="leave out the per-point regions")
    ap.add_argument("--no-data", action="store_true",
                    help="leave out the data section (values / embedding / volume)")
    ap.add_argument("--indent", type=int, default=None, help="pretty-print with this indent")
    ap.add_argument("-o", "--output", required=True, type=Path)
    args = ap.parse_args()
    if bool(args.dataset) == bool(args.volume):
        ap.error("give exactly one of DATASET or --volume")

    doc = build_document(args.dataset, args.volume, k=args.k, simplify=args.simplify,
                         initial_threshold=args.initial, embed=args.embed,
                         spine=not args.no_spine, region=not args.no_region,
                         data=not args.no_data, verbose=True)
    path = write_json(doc, args.output, indent=args.indent)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    _main()
