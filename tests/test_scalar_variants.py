"""Datasets with several scalar functions on the same points (truth.json "scalars")."""
import numpy as np

from comparison.sources import FieldSource
from comparison.variants import scalar_id, scalar_values, variants_for
from comparison.viewer_export import _related
from pipeline.neighbor_graph import knn_graph


def _source(n=120, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 5)).astype(np.float32)
    a = np.exp(-((pts - 1.5) ** 2).sum(1)) + np.exp(-((pts + 1.5) ** 2).sum(1))
    b = pts[:, 0].astype(np.float64)
    main = scalar_values(a, knn_graph(pts, 10), 2)
    sc = {"functions": [{"id": "a", "label": "A", "values": a}, {"id": "b", "label": "B", "values": b}],
          "smooths": [0, 2], "ks": [6, 20], "main": {"id": "a", "smooth": 2}}
    return FieldSource(name="t", kind="points", values=main, raw_values=main,
                       coords=lambda v: pts[v], k=10, points=pts, scalars=sc)


def test_ids_and_family():
    src = _source()
    ids = [v.id for v in variants_for(src)]
    assert ids[0] == "ours" and "ours_smooth" not in ids
    assert {"ours_k6", "ours_k20", "ours_f_a_s0", "ours_f_b_s2", "ours_f_b_s0_k20"} <= set(ids)
    assert "ours_f_a_s2" not in ids                     # that is the main file
    assert scalar_id("a", 2, 10, src) == "ours" and scalar_id("b", 0, 6, src) == "ours_f_b_s0_k6"


def test_main_function_variant_matches_main_values():
    src = _source()
    v = {x.id: x for x in variants_for(src)}["ours_k20"]
    g = v.graph_for(src)
    assert np.allclose(v.values_for(src, g), scalar_values(src.scalars["functions"][0]["values"], g, 2))
    w = {x.id: x for x in variants_for(src)}["ours_f_b_s0"]
    vals = w.values_for(src, w.graph_for(src))
    assert vals.min() == 0 and vals.max() == 1 and np.argmax(vals) == np.argmax(src.points[:, 0])


def test_related_pairs():
    assert _related("ours_f_b_s0", "ours_f_b_s0_k20")
    assert _related("ours_f_b_s0_k20", "ours_k20")
    assert not _related("ours_f_b_s0", "ours_f_a_s0_k6")
