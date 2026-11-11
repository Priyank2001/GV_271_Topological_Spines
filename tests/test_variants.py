"""Tests for the extraction variants and their export for the viewer's compare mode
(comparison/variants.py, comparison/viewer_export.py, pipeline.spine_layout.anchored_layout).

Run from the repo root: python3 -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("torch")

from comparison.sources import FieldSource  # noqa: E402
from comparison.variants import (TRAIT_LABELS, VARIANTS, AnyEGVariant, OursVariant,  # noqa: E402
                                 anyeg_to_extremum_graph)
from comparison.viewer_export import PositionRegistry, export_variants  # noqa: E402
from pipeline.export_json import check_export, extremum_graph_to_dict  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import grid_graph  # noqa: E402
from pipeline.spine_layout import anchored_layout, spine_layout  # noqa: E402


def bumpy_grid(shape=(10, 12, 14), levels=None, seed=0) -> FieldSource:
    """A small smooth volume with a few bumps; ``levels`` quantises it (ties, like 8-bit data)."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*(np.linspace(0, 1, n) for n in shape), indexing="ij")
    f = np.zeros(shape)
    for c, w in zip(rng.random((6, 3)), 0.12 + 0.1 * rng.random(6)):
        f += np.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) / (2 * w * w))
    f = (f - f.min()) / (f.max() - f.min())
    if levels:
        f = np.round(f * levels) / levels
    nz, ny, nx = shape

    def coords(v):
        v = np.asarray(v)
        return np.stack([v % nx, (v // nx) % ny, v // (nx * ny)], axis=1).astype(float)

    values = f.reshape(-1)
    return FieldSource(name="bumps", kind="grid", values=values, raw_values=values, coords=coords,
                       spacing=np.ones(3), grid_shape=shape)


def main_document(src: FieldSource, base: float = 0.01) -> dict:
    raw = extremum_graph(src.values, grid_graph(src.grid_shape))
    eg = raw.simplify(base)
    meta = {"name": src.name, "kind": "grid", "dimensions": 3, "grid_shape": list(src.grid_shape),
            "spacing": [1.0, 1.0, 1.0], "simplify_threshold": base, "initial_threshold": base}
    return extremum_graph_to_dict(eg, coords=src.coords, meta=meta, spine=spine_layout(eg, src.coords),
                                  full_peak_persistence=raw.peak_persistence)


# ---------------------------------------------------------------------------
# anchored layout
# ---------------------------------------------------------------------------
def test_anchored_layout_keeps_anchors_and_places_the_rest():
    src = bumpy_grid()
    eg = extremum_graph(src.values, grid_graph(src.grid_shape))
    first = spine_layout(eg, src.coords)
    anchors = {int(v): first.peak_xy[i] for i, v in enumerate(eg.peaks[:3])}
    second = anchored_layout(eg, src.coords, anchors, first.scale)
    assert np.allclose(second.peak_xy[:3], first.peak_xy[:3])           # anchors exactly in place
    assert np.isfinite(second.peak_xy).all() and np.isfinite(second.saddle_xy).all()
    assert second.stress <= second.pca_stress + 1e-12                    # SMACOF never makes it worse


def test_anchored_layout_gives_one_position_per_vertex():
    src = bumpy_grid()
    eg = extremum_graph(src.values, grid_graph(src.grid_shape))
    lay = anchored_layout(eg, src.coords, {}, 1.0)
    pos = {}
    for v, xy in zip(np.concatenate([eg.peaks, eg.saddles]), np.concatenate([lay.peak_xy, lay.saddle_xy])):
        assert np.allclose(pos.setdefault(int(v), xy), xy)


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
def test_registry_and_traits():
    assert list(VARIANTS)[:3] == ["ours", "anyeg", "ours14"]
    src = bumpy_grid()
    ours, anyeg, ours14 = (VARIANTS[k].traits(src) for k in ("ours", "anyeg", "ours14"))
    assert set(ours) == set(TRAIT_LABELS)
    assert [k for k in TRAIT_LABELS if ours[k] != ours14[k]] == ["neighbourhood"]
    assert all(ours[k] != anyeg[k] for k in TRAIT_LABELS)
    assert not VARIANTS["ours14"].applies_to(FieldSource(
        "p", "points", np.zeros(3), np.zeros(3), coords=lambda v: v, k=2))


def test_anyeg_variant_in_our_format_adds_nothing_of_ours():
    src = bumpy_grid(levels=12)                     # ties: the raw counts must be reported
    res = AnyEGVariant().extract(src)
    eg = res.graph
    assert np.isinf(eg.peak_persistence).all() and (eg.peak_parent == -1).all()   # no simplification
    assert np.all(np.diff(src.order[eg.peaks]) < 0)                               # highest first
    assert np.array_equal(np.sort(eg.peaks), np.sort(res.common.maxima))
    assert res.notes["raw_maxima"] >= res.notes["maxima"]
    # a saddle reaching k maxima is stored as k - 1 entries at one vertex
    groups = np.asarray(res.saddle_extra["anyeg_saddle"])
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        assert len(set(eg.saddles[rows].tolist())) == 1
        assert len(rows) == len(res.saddle_extra["reaches"][rows[0]]) - 1
    doc = extremum_graph_to_dict(eg, coords=src.coords, saddle_extra=res.saddle_extra,
                                 spine=anchored_layout(eg, src.coords, {}, 1.0))
    check_export(doc)
    assert all(p["persistence"] is None for p in doc["peaks"])


def test_anyeg_and_ours_on_the_same_14_neighbour_grid_share_maxima_and_basins():
    src = bumpy_grid(levels=20)
    a = AnyEGVariant().extract(src)
    b = OursVariant("o14", "ours14", neighbourhood="freudenthal-14").extract(src)
    assert np.array_equal(np.sort(a.graph.peaks), np.sort(b.graph.peaks))
    assert np.array_equal(a.graph.peaks[a.graph.region], b.graph.peaks[b.graph.region])


# ---------------------------------------------------------------------------
# export for the viewer
# ---------------------------------------------------------------------------
def test_export_variants_writes_every_variant_in_one_frame():
    src = bumpy_grid(levels=30)
    main = main_document(src)
    written = {}
    entries, comparisons = export_variants(src, main, "bumps.json",
                                           write=lambda d, f: written.__setitem__(f, d), log=lambda *_: None)
    assert [e["id"] for e in entries] == ["ours", "anyeg", "ours14"]
    assert set(written) == {"bumps.anyeg.json", "bumps.ours14.json"}
    assert len(comparisons) == 3
    assert main["meta"]["variant"]["id"] == "ours" and main["meta"]["all_maxima"]

    reg = PositionRegistry.from_document(main)
    for name, doc in written.items():
        check_export(doc)
        assert "data" not in doc and doc["meta"]["data_file"] == "bumps.json"
        for node, xy in zip(doc["peaks"], doc["spine"]["peaks"]):    # shared vertices: same place
            if node["vertex"] in reg.xy:
                assert np.allclose(reg.xy[node["vertex"]], xy), name
    anyeg = written["bumps.anyeg.json"]
    assert anyeg["meta"]["variant"]["simplifiable"] is False and anyeg["meta"]["simplify_threshold"] is None
    ag = next(c for c in comparisons if {c["a"], c["b"]} == {"anyeg", "ours14"})
    assert ag["maxima_jaccard"] == 1.0 and ag["region_agreement"] == 1.0
