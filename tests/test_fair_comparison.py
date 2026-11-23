"""Tests for the fair AnyEG-vs-ours comparison (comparison/fair.py, comparison/reference.py).

Run from the repo root: python3 -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from anyeg import ArcAssembler, FreudenthalGridNeighborhood, ScalarField, TorchKernels  # noqa: E402
from anyeg.neighborhood import FREUDENTHAL_OFFSETS  # noqa: E402
from comparison.common import from_ours  # noqa: E402
from comparison.fair import AnyEGContender, FairStudy, OursContender, make_case  # noqa: E402
from comparison.inputs import FieldInput, RankValues  # noqa: E402
from comparison.reference import (ROOT as NEVER, graph_persistence, merge_tree_fidelity,  # noqa: E402
                                  merge_tree_pairs, pl_saddle_valid)
from pipeline.backends import NumpyBackend  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402

CPU = TorchKernels("cpu")


def brute_force_pairs(order, edges):
    """Textbook vertex sweep from the top, elder rule: {maximum: death rank}."""
    n = order.shape[0]
    nbrs = [[] for _ in range(n)]
    for a, b in edges.tolist():
        nbrs[a].append(b)
        nbrs[b].append(a)
    parent, top, death = list(range(n)), list(range(n)), {}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for v in np.argsort(-order).tolist():
        roots = {find(u) for u in nbrs[v] if order[u] > order[v]}
        if not roots:
            death[v] = NEVER
            continue
        roots = sorted(roots, key=lambda r: -order[top[r]])
        for r in roots[1:]:
            death[top[r]] = int(order[v])
            parent[r] = roots[0]
        parent[v] = roots[0]
    return death


def random_case(shape, neighborhood, seed):
    rng = np.random.default_rng(seed)
    base = ScalarField.from_grid(rng.integers(0, 6, size=shape).astype(np.float32))  # many ties
    return make_case(FieldInput("t", "random", lambda: base), base, neighborhood)


# ---------------------------------------------------------------------------
def test_merge_tree_on_a_path():
    order = np.array([0, 4, 1, 3, 2])          # maxima at 1 (global) and 3; they meet at 2
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert merge_tree_pairs(order, edges) == {1: NEVER, 3: 1}


@pytest.mark.parametrize("nb", ["14-nbr", "6-nbr"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_merge_tree_matches_brute_force(nb, seed):
    case = random_case((5, 6, 7), nb, seed)
    assert merge_tree_pairs(case.order, case.graph.edges) == brute_force_pairs(case.order, case.graph.edges)


@pytest.mark.parametrize("nb", ["14-nbr", "6-nbr"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ours_reproduces_the_merge_tree(nb, seed):
    """A theorem for ours: the highest pass between basins is the merge saddle."""
    case = random_case((6, 7, 8), nb, seed)
    for relief in (0.0, 0.5):
        eg = extremum_graph(case.heights, case.graph, min_loop_relief=relief, backend=NumpyBackend())
        g = from_ours(eg, "ours", {})
        ref = merge_tree_pairs(case.order, case.graph.edges)
        fid = merge_tree_fidelity(ref, graph_persistence(g.maxima, g.links, case.order),
                                  case.order, case.heights, 0.0)
        assert fid.exact == fid.reference_pairs and fid.missing_maxima == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pl_test_agrees_with_anyeg_module4(seed):
    """Every AnyEG saddle passes the independent upper-link test, and the
    candidates AnyEG rejects fail it."""
    case = random_case((6, 7, 8), "14-nbr", seed)
    out = CPU.run(case.field.values, case.table)
    ok = pl_saddle_valid(out.saddles, case.table.table, case.order, case.field.dims,
                         FREUDENTHAL_OFFSETS)
    assert ok.all()
    everything = np.arange(case.num_points)
    valid_all = pl_saddle_valid(everything, case.table.table, case.order, case.field.dims,
                                FREUDENTHAL_OFFSETS)
    has_higher_other_basin = np.zeros(case.num_points, dtype=bool)
    for a, b in case.graph.edges.tolist():
        if out.final_link[a] != out.final_link[b]:
            hi, lo = (a, b) if case.order[a] > case.order[b] else (b, a)
            has_higher_other_basin[lo] = True
    expected = np.flatnonzero(valid_all & has_higher_other_basin)
    assert set(out.saddles.tolist()) == set(expected.tolist())


def test_arc_filter_uses_heights_not_ranks():
    """Ranks spread far beyond AnyEG's [0, 20000) range must not drop arcs."""
    rng = np.random.default_rng(3)
    values = rng.random((8, 9, 10)).astype(np.float32).reshape(-1)
    field = ScalarField(values, dims=(10, 9, 8))
    table = FreudenthalGridNeighborhood().build(field)
    ranks = RankValues().apply(values) * 1000.0          # order kept, gaps > 20000
    plain = ArcAssembler().assemble(values, table, CPU.run(values, table))
    on_ranks = ArcAssembler().assemble(ranks, table, CPU.run(ranks, table))
    with_heights = ArcAssembler().assemble(ranks, table, CPU.run(ranks, table), heights=values)
    assert on_ranks.num_arcs < plain.num_arcs                      # the bias this fixes
    assert with_heights.links() == plain.links()


def test_rank_variant_hands_heights_to_anyeg():
    base = ScalarField.from_grid(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    f = RankValues().field(base)
    assert np.array_equal(f.heights, base.values)


def test_small_study_runs_and_backends_agree():
    rng = np.random.default_rng(5)
    base = ScalarField.from_grid(rng.random((6, 7, 8)))
    inputs = [FieldInput("tiny", "random", lambda: base)]
    contenders = [AnyEGContender("AnyEG port CPU", TorchKernels("cpu"), "CPU", ""),
                  OursContender("Ours CPU", NumpyBackend, "CPU", "")]
    res = FairStudy(inputs, ["14-nbr", "6-nbr"], contenders, repeats=2).run(log=lambda s: None)
    assert len(res) == 2
    for r in res:
        ours = next(c for c in r.contenders if c.name == "Ours CPU")
        anyeg = next(c for c in r.contenders if c.name == "AnyEG port CPU")
        assert ours.maxima == anyeg.maxima == r.reference_maxima
        assert ours.fidelity.exact_fraction == 1.0
        assert ours.timing.as_dict()["detect"]["runs"] == 2
    assert res[0].contenders[0].pl_valid is not None and res[1].contenders[0].pl_valid is None
