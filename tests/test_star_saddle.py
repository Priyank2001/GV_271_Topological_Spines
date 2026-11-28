"""star_saddle_2d: one saddle (the hub, vertex 0) shared by all 10 peaks."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generator"))

from make_star_saddle_dataset import M, field, sample  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402


@pytest.fixture(scope="module")
def data():
    pts = sample()
    return pts, field(pts)


@pytest.mark.parametrize("k", [20, 30, 50])
def test_one_saddle_joins_all_peaks(data, k):
    pts, vals = data
    eg = extremum_graph(vals, knn_graph(pts, k))              # exact: no noise peaks to simplify away
    assert eg.num_peaks == M
    assert set(eg.saddles.tolist()) == {0}                      # every saddle row is the hub
    joined = set(eg.saddle_peaks.ravel().tolist())
    assert joined == set(range(M))                              # ... and it reaches all 10 peaks
    assert eg.saddle_is_merge.all()


def test_hub_is_a_monkey_saddle(data):
    """10 higher and 10 lower neighbours: the 20 arm tips, ridges and valleys alternating."""
    pts, vals = data
    g = knn_graph(pts, 2 * M)
    nb = g.indices[g.indptr[0]:g.indptr[1]]
    assert (vals[nb] > vals[0]).sum() == M and (vals[nb] < vals[0]).sum() == M
