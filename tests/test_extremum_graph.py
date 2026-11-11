"""Tests for pipeline/extremum_graph.py. Run from the repo root: python3 -m pytest tests"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import NeighborGraph, grid_graph, knn_graph


def path_graph(n):
    return NeighborGraph.from_edges(n, np.array([[i, i + 1] for i in range(n - 1)]))


def test_five_point_path():
    # A B C D E = 9 4 7 3 8: peaks A, E, C; saddles B (A-C) and D (C-E).
    eg = extremum_graph(np.array([9, 4, 7, 3, 8.0]), path_graph(5))
    assert eg.peaks.tolist() == [0, 4, 2]
    assert eg.saddles.tolist() == [1, 3]
    assert eg.saddle_peaks.tolist() == [[0, 2], [1, 2]]
    assert eg.peak_persistence[2] == 3 and eg.peak_persistence[1] == 5
    assert np.isinf(eg.peak_persistence[0])
    assert eg.saddle_is_merge.all()
    assert eg.region.tolist() == [0, 0, 2, 1, 1]
    assert eg.region_size.tolist() == [2, 2, 1]


def test_monotone_field_has_one_peak_and_no_saddle():
    eg = extremum_graph(np.arange(10.0), path_graph(10))
    assert eg.peaks.tolist() == [9] and eg.num_saddles == 0


def test_constant_field_ties_are_broken_by_index():
    eg = extremum_graph(np.zeros(25), grid_graph((5, 5)))
    assert eg.num_peaks == 1 and eg.num_saddles == 0


def test_disconnected_graph_gives_one_peak_per_piece_and_no_saddle():
    g = NeighborGraph.from_edges(4, np.array([[0, 1], [2, 3]]))
    eg = extremum_graph(np.array([1.0, 2.0, 3.0, 4.0]), g)
    assert sorted(eg.peaks.tolist()) == [1, 3] and eg.num_saddles == 0


def test_two_bumps_on_a_grid():
    x, y = np.meshgrid(np.linspace(-3, 3, 41), np.linspace(-2, 2, 31), indexing="ij")
    f = (np.exp(-((x - 1.5) ** 2 + y ** 2)) + 0.8 * np.exp(-((x + 1.5) ** 2 + y ** 2))).ravel()
    eg = extremum_graph(f, grid_graph(x.shape))
    assert eg.num_peaks == 2 and eg.num_saddles == 1
    sx = x.ravel()[eg.saddles[0]]
    assert abs(sx) < 0.3                                   # saddle sits between the bumps
    assert eg.peak_persistence[1] == pytest.approx(0.8 - f[eg.saddles[0]], abs=0.02)


def test_three_peaks_in_a_ring_need_a_loop_closing_saddle():
    # Cycle of 6 points: peaks at 0, 2, 4 (9, 8, 7), valleys at 1, 3, 5 (3, 3.5, 3.2).
    # Two valleys merge the three peaks into one group; the third closes the loop.
    n = 6
    g = NeighborGraph.from_edges(n, np.array([[i, (i + 1) % n] for i in range(n)]))
    # min_loop_relief=0: this toy ring has no points inside it, so no valley
    # below the saddles, and the default filter would drop the loop.
    eg = extremum_graph(np.array([9.0, 3.0, 8.0, 3.5, 7.0, 3.2]), g, min_loop_relief=0.0)
    assert eg.peaks.tolist() == [0, 2, 4]
    assert eg.saddles.tolist() == [3, 5, 1]                # highest to lowest
    assert eg.saddle_is_merge.tolist() == [True, True, False]
    assert eg.peak_persistence[2] == pytest.approx(7 - 3.5)
    assert eg.peak_persistence[1] == pytest.approx(8 - 3.2)


def test_only_the_highest_pass_between_two_regions_is_kept():
    # Both valleys (1 and 3) climb to peak 0, so peaks 0 and 2 touch twice;
    # the lower pass (vertex 1) is not reported as a second saddle.
    g = NeighborGraph.from_edges(4, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
    eg = extremum_graph(np.array([9.0, 3.0, 8.0, 4.0]), g, min_loop_relief=0.0)
    assert eg.saddles.tolist() == [3]


def test_simplify_keeps_every_point_in_exactly_one_region():
    rng = np.random.default_rng(1)
    pts = rng.uniform(size=(2000, 2))
    f = np.sin(6 * pts[:, 0]) * np.cos(6 * pts[:, 1]) + 0.05 * rng.normal(size=2000)
    eg = extremum_graph(f, knn_graph(pts, 8))
    for t in (0.0, 0.05, 0.3, 10.0):
        s = eg.simplify(t)
        assert s.region_size.sum() == len(pts)
        assert np.array_equal(np.bincount(s.region, minlength=s.num_peaks), s.region_size)
        assert (s.saddle_peaks[:, 0] < s.saddle_peaks[:, 1]).all()   # [higher, lower]
    assert eg.simplify(10.0).num_peaks == 1                # only the global maximum survives


def test_every_saddle_lies_below_both_of_its_peaks():
    rng = np.random.default_rng(2)
    pts = rng.uniform(size=(3000, 3))
    f = np.sin(5 * pts[:, 0]) + np.cos(4 * pts[:, 1]) * pts[:, 2]
    eg = extremum_graph(f, knn_graph(pts, 10))
    assert (eg.saddle_values[:, None] <= eg.peak_values[eg.saddle_peaks]).all()


def test_values_shape_is_checked():
    with pytest.raises(ValueError):
        extremum_graph(np.zeros(3), path_graph(5))


@pytest.mark.parametrize("name,k", [("cyclic_2d", 20), ("cyclic_3d", 20)])
def test_ring_datasets_recover_the_six_cycle(name, k):
    from pipeline.data_loader import load_dataset

    ds = load_dataset(name)
    eg = extremum_graph(ds.values, knn_graph(ds.points, k)).simplify(0.04)
    assert eg.num_peaks == 6 and eg.num_saddles == 6

    tmax, tsad = ds.centers("max"), ds.centers("saddle")
    near = lambda c, q: int(np.argmin(np.linalg.norm(c - q, axis=1)))
    found = {(ds.maxima[near(tmax, ds.points[eg.peaks[p]])].id,
              ds.saddles[near(tsad, ds.points[eg.saddles[s]])].id) for p, s in eg.edges}
    truth = {(b, a) if a.startswith("saddle") else (a, b) for a, b in ds.extremum_graph_edges}
    assert found == truth


def test_variation_threshold_drops_only_deep_saddles_and_never_peaks():
    # Three peaks in a row: 9 -- saddle 6 -- 8 -- saddle 1 -- 7 (persistences 2 and 6).
    g = NeighborGraph.from_edges(5, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
    eg = extremum_graph(np.array([9.0, 6.0, 8.0, 1.0, 7.0]), g)
    assert eg.saddle_persistence.tolist() == [2.0, 6.0]
    assert eg.simplify(0.0) is eg and eg.simplify(0.0, variation=None) is eg
    assert eg.simplify(0.0, variation=6.0) is eg                 # nothing is above 6
    cut = eg.simplify(0.0, variation=3.0)                        # the deep saddle goes
    assert cut.num_peaks == 3 and cut.saddles.tolist() == [1]
    assert cut.saddle_peaks.tolist() == [[0, 1]]                 # the graph is now two components
    assert eg.simplify(0.0, variation=0.0).num_saddles == 0
    # the noise threshold still merges peaks, and then the variation threshold applies to what is left
    both = eg.simplify(3.0, variation=100.0)
    assert both.num_peaks == 2 and both.saddles.tolist() == [3]  # peak 8 (persistence 2) merged into 9; the link left is the deep one


def test_variation_threshold_keeps_regions_complete():
    rng = np.random.default_rng(3)
    pts = rng.uniform(size=(2000, 2))
    f = np.sin(6 * pts[:, 0]) * np.cos(6 * pts[:, 1]) + 0.05 * rng.normal(size=2000)
    eg = extremum_graph(f, knn_graph(pts, 8))
    for hi in (0.2, 0.8, 5.0):
        s = eg.simplify(0.05, variation=hi)
        assert s.region_size.sum() == len(pts)
        assert (s.saddle_persistence <= hi).all()
        assert s.num_peaks == eg.simplify(0.05).num_peaks


# ---- saturated persistence (Correa et al. 2011, Algorithm 1) -----------------
def test_saturation_keeps_the_ridge_through_a_merged_peak():
    # A B C D E = 9 4 7 3 8. B cancels C first (p=3); then D's path to the top
    # runs E - D - C - B - A, so B is needed up to D's persistence (5).
    eg = extremum_graph(np.array([9, 4, 7, 3, 8.0]), path_graph(5))
    assert eg.saddle_persistence.tolist() == [3.0, 5.0]
    assert eg.saddle_saturation.tolist() == [5.0, 5.0]
    # At t=4 peak C is merged away and B is no longer a saddle of the graph,
    # but both of its arcs are still drawn: they carry D's link to A.
    assert eg.simplify(4.0).saddles.tolist() == [3]
    assert eg.arc_mask(4.0).tolist() == [True, True]
    assert eg.arc_mask(5.5).tolist() == [False, False]


def test_saturation_keeps_the_rim_of_a_volcano():
    # The ring of the paper's Fig. 5: three peaks (9, 8, 7) on a rim. Once the
    # rim is a single peak, the loop saddle still shows the whole rim (Fig. 5b);
    # cutting it with the variation threshold opens the ring (Fig. 5a).
    n = 6
    g = NeighborGraph.from_edges(n, np.array([[i, (i + 1) % n] for i in range(n)]))
    eg = extremum_graph(np.array([9.0, 3.0, 8.0, 3.5, 7.0, 3.2]), g, min_loop_relief=0.0)
    assert eg.saddle_persistence.tolist() == pytest.approx([3.5, 4.8, 5.0])
    assert eg.saddle_saturation.tolist() == pytest.approx([5.0, 5.0, 5.0])
    rim = eg.simplify(4.9)
    assert rim.num_peaks == 1 and rim.num_saddles == 0      # no link between surviving peaks ...
    assert eg.arc_mask(4.9).all()                           # ... but the whole rim is drawn
    assert eg.arc_mask(4.9, variation=4.9).tolist() == [True, True, False]


def test_saturation_is_at_least_persistence_and_survives_simplify():
    rng = np.random.default_rng(5)
    pts = rng.uniform(size=(1500, 2))
    f = np.sin(5 * pts[:, 0]) * np.cos(5 * pts[:, 1]) + 0.05 * rng.normal(size=1500)
    eg = extremum_graph(f, knn_graph(pts, 10))
    assert (eg.saddle_saturation >= eg.saddle_persistence).all()
    s = eg.simplify(0.1)
    assert (s.saddle_saturation >= s.saddle_persistence).all()
    assert eg.arc_mask(0.1).sum() >= s.num_saddles


@pytest.mark.parametrize("seed", [0, 1])
def test_saturation_arcs_lead_through_merged_peaks_to_the_right_peak(seed):
    # The point of p*: at any threshold, the drawn arcs lead from each saddle of
    # the simplified graph, through merged-away peaks only, to the surviving
    # peaks it joins. Checked on the raw graph and on a simplified one.
    from collections import deque

    rng = np.random.default_rng(seed)
    pts = rng.uniform(size=(3000, 3))
    f = (np.sin(4 * pts[:, 0]) + np.cos(5 * pts[:, 1]) * np.sin(3 * pts[:, 2])
         + 0.1 * rng.normal(size=3000))
    raw = extremum_graph(f, knn_graph(pts, 12))
    for eg in (raw, raw.simplify(0.05)):
        pers = eg.peak_persistence[np.isfinite(eg.peak_persistence)]
        for t in np.quantile(pers, [0.2, 0.5, 0.8, 0.95]):
            survives = eg.peak_persistence >= t
            root = np.arange(eg.num_peaks)
            for p in range(eg.num_peaks):
                if not survives[p]:
                    root[p] = root[eg.peak_parent[p]]
            adj = {}
            for a, b in eg.saddle_peaks[eg.arc_mask(t)].tolist():
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            kept = (eg.saddle_persistence >= t) & (root[eg.saddle_peaks[:, 0]] != root[eg.saddle_peaks[:, 1]])
            for e in np.unique(eg.saddle_peaks[kept]).tolist():
                seen, queue = {e}, deque([e])
                while queue:
                    u = queue.popleft()
                    if u == root[e] or (u != e and survives[u]):
                        continue                       # stop at surviving peaks
                    for w in adj.get(u, []):
                        if w not in seen:
                            seen.add(w)
                            queue.append(w)
                assert root[e] in seen, f"t={t}: peak {e} has no ridge to peak {root[e]}"


def test_a_cancelled_peak_joins_the_peak_across_its_saddle():
    # T m y s p = 10 6 9.5 2 5 on a path. The merge tree's elder rule would hand p
    # to T (the top of the group it joins at s), but p only touches y. The
    # paper's cancellation (Algorithm 1) merges p into y, its neighbour.
    eg = extremum_graph(np.array([10, 6, 9.5, 2, 5.0]), path_graph(5))
    assert eg.peaks.tolist() == [0, 2, 4]
    assert eg.peak_persistence[1:].tolist() == [3.5, 3.0]
    assert eg.peak_parent.tolist() == [-1, 0, 1]
    s = eg.simplify(3.2)                                     # p goes, y stays
    assert s.region.tolist() == [0, 0, 1, 1, 1]              # p's points join y, not T
    assert s.saddle_peaks.tolist() == [[0, 1]]


def test_ackley_on_a_knn_graph_has_the_lattice_peaks_and_links():
    # tests/check_ackley_knn.py, smallest case: 2D, 20k random points, k = 15.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_ackley_knn import check

    for seed in (0, 1):
        r = check(d=2, n=20_000, k=15, seed=seed, min_loop_relief=0.5)
        assert r["lattice"] and r["err"] < 0.1
        assert (r["true"], r["diag"], r["centre"]) == (12, 0, 4)
