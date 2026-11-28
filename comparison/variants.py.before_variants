"""
Extraction variants: the different ways of computing an extremum graph that
the viewer can show side by side.

Every variant implements ``ExtractionVariant``:

* ``extract(source) -> VariantResult``: the graph in *our* ``ExtremumGraph``
  form, so the existing JSON export and spine layout work unchanged, plus a
  ``CommonGraph`` for the agreement numbers;
* ``traits(source) -> {trait: short text}``: how the variant makes each design
  choice. The viewer's comparison box lists the traits on which two variants
  differ; that list is the short reason why their graphs differ;
* ``simplifiable``: False for methods without persistence (AnyEG). The viewer
  then shows their full graph and the persistence slider leaves them alone.

To add a variant, write a subclass and ``register`` it; ``web/build_data.py``
exports every registered variant for every dataset it applies to. Design
notes: docs/extremum_graph_variants.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from anyeg import ArcAssembler, FreudenthalGridNeighborhood, NeighborTable, ScalarField, TorchKernels
from anyeg.graph import AnyEGGraph
from anyeg.kernels import ExtremumGraphKernels
from pipeline.extremum_graph import ExtremumGraph, extremum_graph
from pipeline.neighbor_graph import NeighborGraph

from .common import CommonGraph, from_anyeg, from_ours
from .sources import FieldSource

# Order and wording of the traits in the viewer's comparison box.
TRAIT_LABELS = {
    "neighbourhood": "Neighbourhood",
    "ties": "Equal values",
    "saddles": "Saddles",
    "links": "Links",
    "loops": "Loop saddles",
    "simplification": "Simplification",
}


@dataclass
class VariantResult:
    graph: ExtremumGraph                  # full (unsimplified) graph in our format
    common: CommonGraph                   # for agreement numbers between variants
    saddle_extra: Optional[Dict[str, list]] = None   # extra JSON fields per saddle
    notes: Dict[str, object] = field(default_factory=dict)   # facts for the comparison box


class ExtractionVariant(ABC):
    id: str
    label: str
    summary: str
    simplifiable: bool = True

    def applies_to(self, source: FieldSource) -> bool:
        return True

    @abstractmethod
    def traits(self, source: FieldSource) -> Dict[str, str]:
        ...

    @abstractmethod
    def extract(self, source: FieldSource) -> VariantResult:
        ...

    def describe(self, source: FieldSource) -> dict:
        """What the viewer needs to know about the variant (stored in index.json)."""
        return {"id": self.id, "label": self.label, "summary": self.summary,
                "simplifiable": self.simplifiable, "traits": self.traits(source)}


# ---------------------------------------------------------------------------
# Our pipeline
# ---------------------------------------------------------------------------
class OursVariant(ExtractionVariant):
    """``pipeline.extremum_graph`` on our usual graph, or on TTK's 14-neighbour grid."""

    def __init__(self, id: str, label: str, neighbourhood: str = "base"):
        if neighbourhood not in ("base", "freudenthal-14"):
            raise ValueError(neighbourhood)
        self.id, self.label, self.neighbourhood = id, label, neighbourhood
        self.summary = ("Our pipeline: steepest ascent, the highest pass between touching basins, "
                        "then persistence simplification.")
        if neighbourhood == "freudenthal-14":
            self.summary = "Our algorithm on AnyEG's (TTK's) 14-neighbour grid."

    def applies_to(self, source: FieldSource) -> bool:
        return self.neighbourhood == "base" or source.kind == "grid"

    def traits(self, source: FieldSource) -> Dict[str, str]:
        if self.neighbourhood == "freudenthal-14":
            nb = "grid with 14 neighbours (TTK's triangulation, as AnyEG)"
        elif source.kind == "grid":
            nb = "grid with 6 axis neighbours (a diagonal ridge can split into several maxima)"
        else:
            nb = f"kNN graph, k = {source.k}"
        return {
            "neighbourhood": nb,
            "ties": "broken by vertex index (a strict total order)",
            "saddles": "the highest pass between every two basins that touch",
            "links": "each saddle joins exactly two maxima",
            "loops": "a loop-closing saddle is kept only if the valley below it runs deep (relief >= 0.5)",
            "simplification": "persistence: the noise slider (p_lo) merges small peaks, p_hi cuts deep links",
        }

    def graph_for(self, source: FieldSource) -> NeighborGraph:
        if self.neighbourhood == "base":
            return source.base_graph
        nz, ny, nx = source.grid_shape
        dummy = ScalarField(np.zeros(source.num_points, np.float32), dims=(nx, ny, nz))
        table = FreudenthalGridNeighborhood().build(dummy)
        return NeighborGraph.from_edges(source.num_points, table.edges())

    def extract(self, source: FieldSource) -> VariantResult:
        eg = extremum_graph(source.values, self.graph_for(source))
        return VariantResult(graph=eg, common=from_ours(eg, self.id, {}),
                             notes={"maxima": eg.num_peaks, "saddles": eg.num_saddles,
                                    "loop_saddles": int((~eg.saddle_is_merge).sum())})


# ---------------------------------------------------------------------------
# AnyEG
# ---------------------------------------------------------------------------
class AnyEGVariant(ExtractionVariant):
    """The separate AnyEG pipeline (``anyeg``), on values made distinct."""

    simplifiable = False

    def __init__(self, kernels: Optional[ExtremumGraphKernels] = None):
        self.id, self.label = "anyeg", "AnyEG"
        self.summary = ("AnyEG (BuptCIAGroup): steepest ascent on the GPU, true saddles by an "
                        "upper-link test, arcs to every neighbouring basin; no simplification.")
        self._kernels = kernels

    @property
    def kernels(self) -> ExtremumGraphKernels:
        if self._kernels is None:
            self._kernels = TorchKernels()
        return self._kernels

    def traits(self, source: FieldSource) -> Dict[str, str]:
        nb = ("grid with 14 neighbours (TTK's triangulation)" if source.kind == "grid" else
              f"the same kNN graph as ours, k = {source.k} (AnyEG itself takes meshes only)")
        return {
            "neighbourhood": nb,
            "ties": "AnyEG breaks none; here values are replaced by their (value, index) rank first",
            "saddles": "vertices whose higher neighbours fall into two or more separate groups (true saddles)",
            "links": "a saddle gets an arc to every basin among its neighbours, also through lower "
                     "neighbours, so it can join three or more maxima",
            "loops": "no filter: every saddle is kept",
            "simplification": "none: every maximum and saddle is shown, the slider does not apply",
        }

    def neighbor_table(self, source: FieldSource) -> NeighborTable:
        if source.kind == "grid":
            nz, ny, nx = source.grid_shape
            dummy = ScalarField(np.zeros(source.num_points, np.float32), dims=(nx, ny, nz))
            return FreudenthalGridNeighborhood().build(dummy)
        g = source.base_graph
        return NeighborTable.from_csr(g.indptr, g.indices.astype(np.int32))

    def extract(self, source: FieldSource) -> VariantResult:
        table = self.neighbor_table(source)
        if source.num_points > 2 ** 24:
            raise ValueError("ranks above 2^24 are not exact in float32")
        distinct = source.order.astype(np.float32)          # only the order, never the heights
        out = self.kernels.run(distinct, table)
        graph = ArcAssembler().assemble(distinct, table, out)
        eg, extra = anyeg_to_extremum_graph(graph, source.values, source.order)
        notes = {
            "maxima": graph.num_maxima,
            "saddles": int(np.unique(graph.arc_saddle).shape[0]),
            "saddles_reaching_3plus": int(sum(len(r) > 2 for r in _reached(graph).values())),
            "pieces": _pieces(graph),
        }
        if np.unique(source.raw_values).shape[0] < source.num_points:      # only ties make these differ
            notes["raw_maxima"] = int(self.kernels.run(source.raw_values.astype(np.float32), table)
                                      .extrema.shape[0])
            notes["preprocess_maxima"] = int(self.kernels.run(_anyeg_preprocess(source.raw_values), table)
                                             .extrema.shape[0])
        return VariantResult(graph=eg, common=from_anyeg(graph, self.id, {}),
                             saddle_extra=extra, notes=notes)


def anyeg_to_extremum_graph(g: AnyEGGraph, values: np.ndarray, order: np.ndarray):
    """AnyEG's result in our ``ExtremumGraph`` form, *without* adding anything of ours.

    * peaks: AnyEG's maxima, numbered from the highest down (our convention);
      persistence infinite (= not computed), no parents: nothing is merged;
    * saddles: a saddle that reaches k maxima is stored as k - 1 entries
      ``(highest maximum, other maximum)`` at the same vertex, because our
      format joins exactly two peaks per saddle. Drawn, they are the same k
      lines from one saddle position. ``saddle_extra`` keeps the full list
      (``reaches``) and which AnyEG saddle an entry belongs to (``anyeg_saddle``);
    * saddle persistence: AnyEG's own number, |f(saddle) - f(maximum of its basin)|;
      relief: not defined (NaN, exported as null).
    """
    peaks = g.maxima[np.argsort(-order[g.maxima], kind="stable")]
    peak_id = np.full(g.num_points, -1, dtype=np.int64)
    peak_id[peaks] = np.arange(peaks.shape[0])

    reached = _reached(g)
    saddle_vertex = sorted(reached, key=lambda s: -order[s])         # highest first
    rows, groups, reaches = [], [], []
    for k, s in enumerate(saddle_vertex):
        ids = sorted(peak_id[list(reached[s])].tolist())
        for other in ids[1:]:
            rows.append((s, ids[0], other))
            groups.append(k)
            reaches.append(ids)
    sv = np.array([r[0] for r in rows], dtype=np.int64)
    sp = np.array([[r[1], r[2]] for r in rows], dtype=np.int64).reshape(-1, 2)
    region = peak_id[g.region]
    pers = np.abs(values[sv] - values[g.region[sv]]) if len(sv) else np.empty(0)

    eg = ExtremumGraph(
        num_points=g.num_points, values=values,
        peaks=peaks, peak_values=values[peaks],
        peak_persistence=np.full(peaks.shape[0], np.inf),
        peak_parent=np.full(peaks.shape[0], -1, dtype=np.int64),
        saddles=sv, saddle_values=values[sv], saddle_peaks=sp,
        saddle_is_merge=np.ones(len(sv), dtype=bool),
        saddle_persistence=pers, saddle_support=np.zeros(len(sv), dtype=np.int64),
        saddle_relief=np.full(len(sv), np.nan), saddle_saturation=pers.copy(),
        region=region, region_size=np.bincount(region, minlength=peaks.shape[0]),
    )
    return eg, {"reaches": reaches, "anyeg_saddle": groups}


def _reached(g: AnyEGGraph) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for s, m in zip(g.arc_saddle.tolist(), g.arc_maximum.tolist()):
        out.setdefault(s, set()).add(m)
    return out


def _pieces(g: AnyEGGraph) -> int:
    """Connected pieces of the graph on the maxima (saddles join what they reach)."""
    parent = {int(m): int(m) for m in g.maxima.tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ms in _reached(g).values():
        ms = list(ms)
        for m in ms[1:]:
            parent[find(m)] = find(ms[0])
    return len({find(m) for m in parent})


def _anyeg_preprocess(values: np.ndarray) -> np.ndarray:
    """AnyEG's preprocess_vti.py: v + 1e-8 * rank in float64, stored as float32."""
    v = np.asarray(values, dtype=np.float64)
    rank = np.empty(v.shape[0])
    rank[np.argsort(v, kind="stable")] = np.arange(v.shape[0])
    return (v + 1e-8 * rank).astype(np.float32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
VARIANTS: Dict[str, ExtractionVariant] = {}
MAIN_VARIANT = "ours"          # its file is the dataset's main file (with the data section)


def register(variant: ExtractionVariant) -> ExtractionVariant:
    if variant.id in VARIANTS:
        raise ValueError(f"variant {variant.id!r} is already registered")
    VARIANTS[variant.id] = variant
    return variant


def variants_for(source: FieldSource) -> List[ExtractionVariant]:
    """Registered variants that apply to ``source``, the main one first."""
    return [v for v in VARIANTS.values() if v.applies_to(source)]


register(OursVariant(MAIN_VARIANT, "Ours"))
register(AnyEGVariant())
register(OursVariant("ours14", "Ours, 14-neighbour grid", neighbourhood="freudenthal-14"))
