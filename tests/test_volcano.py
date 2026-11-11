"""The paper's volcano (Fig. 5): what our spine does when the rim is breached.

Documents current behaviour, including the known gap: with p_hi off a breached
rim still gives a ring (the paper cuts it automatically); p_hi has to be set.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generator"))
sys.path.insert(0, str(ROOT / "tests"))

from make_volcano_dataset import SPECS, build  # noqa: E402
from check_volcano import shape  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402


def spine(name, variation=None):
    spec = next(s for s in SPECS if s["name"] == name)
    pts, vals, hills, sads, floor, edges = build(spec)
    eg = extremum_graph(vals, knn_graph(pts, 15)).simplify(0.02)
    return shape(eg.simplify(0.02, variation=variation) if variation else eg), sads


def test_intact_and_notched_rims_are_rings_for_any_reasonable_p_hi():
    for name in ("volcano_intact", "volcano_notched"):
        got, sads = spine(name)
        assert got == "ring" and all(s["paper_class"] == "ridge" for s in sads)
        assert spine(name, variation=0.3)[0] == "ring"


def test_breached_rim_needs_p_hi_to_open_the_ring():
    got, sads = spine("volcano_breached")
    assert sum(s["paper_class"] == "valley" for s in sads) == 1
    assert got == "ring"                                    # the gap: the paper would give a chain here
    assert spine("volcano_breached", variation=0.3)[0] == "chain"
