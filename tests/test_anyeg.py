"""Tests for the separate AnyEG pipeline (anyeg/) and the comparison (comparison/).

Tests that need AnyEG's sample files (third_party/AnyEG, from anyeg/native/build.sh)
or the original CUDA build are skipped when those are missing.
Run from the repo root: python3 -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from anyeg import (AnyEGPipeline, ArcAssembler, AssemblyOptions, FreudenthalGridNeighborhood,  # noqa: E402
                   NeighborTable, OriginalCudaKernels, PolyLines, ScalarField, TorchKernels,
                   read_vti, read_vtp, write_vti, write_vtp)
from anyeg.neighborhood import FREUDENTHAL_OFFSETS  # noqa: E402

SAMPLES = ROOT / "third_party" / "AnyEG" / "data"
needs_samples = pytest.mark.skipif(not (SAMPLES / "scalar_field_v1.vti").exists(),
                                   reason="AnyEG samples not cloned (anyeg/native/build.sh)")
needs_original = pytest.mark.skipif(not OriginalCudaKernels.is_built(),
                                    reason="original AnyEG kernels not built (anyeg/native/build.sh)")
CPU = TorchKernels("cpu")


def path_table(n):
    """Neighbour table of the path 0 - 1 - ... - n-1."""
    indptr = np.r_[0, np.cumsum([1] + [2] * (n - 2) + [1])]
    indices = np.concatenate([[1]] + [[i - 1, i + 1] for i in range(1, n - 1)] + [[n - 2]])
    return NeighborTable.from_csr(indptr, indices.astype(np.int32))


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("compress", [True, False])
def test_vti_round_trip(tmp_path, compress):
    rng = np.random.default_rng(0)
    f = ScalarField.from_grid(rng.random((3, 4, 5)), spacing=(1, 2, 3), origin=(0, 0, 1), array_name="T")
    write_vti(tmp_path / "f.vti", f, compress=compress)
    g = read_vti(tmp_path / "f.vti")
    assert g.dims == (5, 4, 3) and g.spacing == (1, 2, 3) and g.array_name == "T"
    assert np.array_equal(g.values, f.values)


def test_vtp_round_trip(tmp_path):
    lines = PolyLines.from_segments(np.arange(12.0).reshape(4, 3), [[0, 1], [1, 2], [2, 3]],
                                    NodeType=np.array([1, 0, 0, 2], dtype=np.int32))
    write_vtp(tmp_path / "g.vtp", lines)
    back = read_vtp(tmp_path / "g.vtp")
    assert np.allclose(back.points, lines.points)
    assert back.segments().tolist() == [[0, 1], [1, 2], [2, 3]]
    assert back.point_data["NodeType"].tolist() == [1, 0, 0, 2]


@needs_samples
def test_reads_anyeg_sample_with_its_designed_maxima():
    # testdata_create_withsaddlecheck.py puts maxima at (1,1,1), (5,5,2), (7,7,8) but
    # writes the array in C order, so in the file x and z are swapped.
    f = read_vti(SAMPLES / "scalar_field_v1.vti")
    assert f.dims == (10, 10, 10) and f.array_name == "ScalarField"
    top3 = np.argsort(-f.values)[:3]
    assert sorted(f.grid_index(top3).tolist()) == [[1, 1, 1], [2, 5, 5], [8, 7, 7]]


# ---------------------------------------------------------------------------
# Neighbourhood
# ---------------------------------------------------------------------------
def test_freudenthal_neighbourhood_is_the_14_stencil_and_symmetric():
    f = ScalarField.from_grid(np.zeros((4, 5, 6)))
    t = FreudenthalGridNeighborhood().build(f)
    centre = 1 + 6 * (2 + 5 * 1)                       # (x, y, z) = (1, 2, 1), interior
    assert t.counts[centre] == 14
    expected = [centre + dx + 6 * (dy + 5 * dz) for dx, dy, dz in FREUDENTHAL_OFFSETS]
    assert t.neighbors(centre).tolist() == expected     # offset order is kept
    assert sorted(t.neighbors(0).tolist()) == [1, 6, 30, 36]   # corner: +x, +y, +z, (0, 1, 1)
    e = t.edges()
    pairs = {(a, b) for a, b in e.tolist()}
    for v in range(t.num_vertices):                      # every neighbour relation goes both ways
        for u in t.neighbors(v):
            assert (min(v, u), max(v, u)) in pairs


# ---------------------------------------------------------------------------
# Kernels and assembly
# ---------------------------------------------------------------------------
def test_kernels_on_a_path():
    # 9 4 7 3 8: maxima 0, 2, 4; saddles 1 and 3 (both neighbours higher, not adjacent).
    out = CPU.run(np.array([9, 4, 7, 3, 8], np.float32), path_table(5))
    assert out.extrema.tolist() == [0, 2, 4]
    assert out.saddles.tolist() == [1, 3]
    assert out.first_link.tolist() == [0, 0, 2, 4, 4]   # 4 rises to 9 (+5), not 7 (+3)
    assert out.final_link.tolist() == [0, 0, 2, 4, 4]


def test_steepest_rise_wins_not_highest_neighbour_order():
    # Vertex 1 has neighbours 0 (value 9) and 2 (value 7): rise 5 vs 3, so it links to 0.
    out = CPU.run(np.array([9, 4, 7], np.float32), path_table(3))
    assert out.first_link[1] == 0


def test_equal_values_are_not_higher_so_plateaus_give_several_maxima():
    out = CPU.run(np.zeros(6, np.float32), path_table(6))
    assert out.extrema.tolist() == list(range(6)) and out.saddles.size == 0


def test_assembly_arcs_on_a_path():
    values = np.array([9, 4, 7, 3, 8], np.float32)
    table = path_table(5)
    g = ArcAssembler().assemble(values, table, CPU.run(values, table))
    assert sorted(g.links()) == [(1, 0), (1, 2), (3, 2), (3, 4)]
    assert sorted(g.maximum_pairs()) == [(0, 2), (2, 4)]
    assert [g.arc(i).tolist() for i in range(g.num_arcs)] == [[1, 0], [1, 2], [3, 2], [3, 4]]
    narrow = ArcAssembler(AssemblyOptions(persistence_range=(0.0, 4.5))).assemble(
        values, table, CPU.run(values, table))
    assert sorted(narrow.links()) == [(1, 2), (3, 2)]   # |4 - 9| = |3 - 8| = 5 are cut


def test_polylines_mark_node_types():
    f = ScalarField.from_points(np.arange(5.0)[:, None], [9, 4, 7, 3, 8])
    g = AnyEGPipeline(kernels=CPU).run(f, path_table(5))
    lines = g.to_polylines(f)
    by_vertex = dict(zip(lines.point_data["VertexId"].tolist(), lines.point_data["NodeType"].tolist()))
    assert by_vertex == {0: 2, 1: 1, 2: 2, 3: 1, 4: 2}


@needs_original
@pytest.mark.parametrize("levels", [None, 4])
def test_torch_port_matches_original_cuda(levels):
    rng = np.random.default_rng(7)
    v = rng.random((9, 10, 11))
    if levels:                                             # many equal values, like 8-bit data
        v = np.floor(v * levels)
    f = ScalarField.from_grid(v)
    table = FreudenthalGridNeighborhood().build(f)
    a = OriginalCudaKernels().run(f.values, table)
    b = CPU.run(f.values, table)
    assert a.same_as(b), a.differences(b)


# ---------------------------------------------------------------------------
# Comparison with our pipeline
# ---------------------------------------------------------------------------
def test_same_neighbourhood_and_distinct_values_give_the_same_maxima_and_basins():
    """AnyEG's steepest ascent and ours agree exactly when they see the same order."""
    from comparison.inputs import RankValues
    from comparison.methods import AnyEGMethod, OursMethod
    from comparison.metrics import compare

    rng = np.random.default_rng(1)
    f = RankValues().field(ScalarField.from_grid(np.floor(rng.random((12, 13, 14)) * 6)))
    a = AnyEGMethod("AnyEG", CPU).run(f)
    b = OursMethod("Ours", "freudenthal-14", backend="numpy").run(f)
    agreement = compare(a, b)
    assert agreement.maxima_jaccard == 1.0 and agreement.region_agreement == 1.0
