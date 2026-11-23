"""
A fair comparison of AnyEG and our pipeline: same inputs, same graph, same
order, same GPU, matched stages, repeated timings, neutral yardsticks.

    python3 -m comparison.fair                      # all inputs, writes results/fair_comparison.*
    python3 -m comparison.fair --volumes fuel --no-samples --repeats 3

What is held equal (and why each matters) is listed in ``PROTOCOL`` below and
repeated at the top of the report. In short:

* **Values.** Both methods see the same strict order: AnyEG gets the
  (value, index) ranks as float32 (it breaks no ties itself), ours gets the
  real values and breaks ties the same way. AnyEG's arc filter, which measures
  heights, gets the real values (``ArcAssembler.assemble(heights=...)``).
* **Graph.** One neighbour structure per case, built once, outside the clock,
  and given to both: TTK's 14-neighbour Freudenthal grid (AnyEG's) and the
  6-neighbour axis grid (ours), so every method runs on both (a 2 x 2 design).
* **Hardware.** One GPU, chosen once, for every GPU run; CPU runs use one
  thread on both sides (NumPy's sort is single-threaded).
* **Timing.** A warm-up run, then ``repeats`` runs; medians are reported.
  Stages are matched: *detect* = maxima, basins and saddles, from host arrays
  to host arrays (AnyEG modules 1-4; our order, ascent and passes), *post* =
  what each method does after that (AnyEG: arcs; ours: persistence, loop
  filter, simplification), which is different work and labelled so.
* **Correctness.** Agreement between the two is reported, but also two
  yardsticks that need neither method (``comparison/reference.py``).
* **Post-filters as separate rows.** Ours without its loop filter, and AnyEG
  with its own (off by default) edge bundling, so the core algorithms can be
  told apart from the filters on top.
"""

from __future__ import annotations

import argparse
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from anyeg import (ArcAssembler, AssemblyOptions, AxisGridNeighborhood, FreudenthalGridNeighborhood,
                   NeighborTable, ScalarField)
from anyeg.kernels import ExtremumGraphKernels, OriginalCudaKernels, TorchKernels
from anyeg.neighborhood import AXIS_OFFSETS, FREUDENTHAL_OFFSETS
from pipeline.backends import NumpyBackend, TorchBackend, pick_cuda_device
from pipeline.extremum_graph import extremum_graph
from pipeline.neighbor_graph import NeighborGraph

from .common import CommonGraph, from_anyeg, from_ours
from .inputs import FieldInput, RankValues, anyeg_samples, klacansky_volumes
from .metrics import Agreement, compare
from .reference import (MergeTreeFidelity, graph_persistence, merge_tree_fidelity, merge_tree_pairs,
                        pl_saddle_valid)

PROTOCOL = [
    ("Same order", "AnyEG gets (value, index) ranks as float32; ours gets the real values and "
                   "breaks ties by index, which is the same strict order. Neither method sees ties."),
    ("Heights for height tests", "AnyEG's arc filter `[0, 20000)` measures heights, so it gets the "
                                 "real values, not ranks (on ranks it dropped up to 2/3 of the arcs)."),
    ("Same graph", "One neighbour structure per case, built once outside the clock and given to "
                   "both. Both methods run on both grids: 14-neighbour (TTK, AnyEG's) and "
                   "6-neighbour (ours)."),
    ("Same GPU", "One device for every GPU run; CUDA is initialised before any clock starts "
                 "(also inside AnyEG's own driver process)."),
    ("Same CPU budget", "CPU runs use one thread on both sides."),
    ("Repeated timing", "One warm-up run, then the median of the repeats. Clocks stop only after "
                        "results are back on the host."),
    ("Matched stages", "*detect*: maxima + basins + saddles, host to host (AnyEG modules 1-4 incl. "
                       "copies; ours: order + ascent + passes incl. copies). *post*: different work "
                       "per method (AnyEG: arcs; ours: persistence, loop filter, p*)."),
    ("Neutral yardsticks", "Merge-tree fidelity and PL saddle validity need neither method; each "
                           "is native to one of them (see below), so both are shown."),
    ("Filters separated", "Ours without the loop filter and AnyEG with its own edge bundling are "
                          "extra rows, so the core algorithms are compared apart from the filters."),
]

NEIGHBORHOODS = {
    "14-nbr": (FreudenthalGridNeighborhood, FREUDENTHAL_OFFSETS, True),
    "6-nbr": (AxisGridNeighborhood, AXIS_OFFSETS, False),
}


# ---------------------------------------------------------------------------
# One case: a field on one neighbourhood, shared by every contender
# ---------------------------------------------------------------------------
@dataclass
class FairCase:
    input: str
    source: str
    neighborhood: str
    field: ScalarField            # float32 ranks; ``field.heights`` = real values
    order: np.ndarray             # (N,) int64 rank
    table: NeighborTable          # AnyEG's layout of the graph
    graph: NeighborGraph          # ours, same edges
    offsets: Sequence[Tuple[int, int, int]]
    triangulated: bool
    build_seconds: float

    @property
    def heights(self) -> np.ndarray:
        return self.field.heights

    @property
    def num_points(self) -> int:
        return self.field.num_points


def make_case(inp: FieldInput, base: ScalarField, neighborhood: str) -> FairCase:
    builder_cls, offsets, triangulated = NEIGHBORHOODS[neighborhood]
    f = RankValues().field(base)
    t0 = time.perf_counter()
    table = builder_cls().build(f)
    graph = NeighborGraph.from_edges(f.num_points, table.edges(), backend="numpy")
    build = time.perf_counter() - t0
    order = f.values.astype(np.int64)
    return FairCase(inp.name, inp.source, neighborhood, f, order, table, graph, offsets,
                    triangulated, build)


# ---------------------------------------------------------------------------
# Contenders
# ---------------------------------------------------------------------------
@dataclass
class Run:
    graph: CommonGraph
    detect: float                 # seconds, host to host
    post: float
    extra: Dict[str, float] = field(default_factory=dict)
    merge_saddles: Optional[np.ndarray] = None    # ours: vertices of merge (tree) saddles


class Contender(ABC):
    name: str
    family: str                   # "AnyEG" or "Ours"
    device: str                   # "GPU" or "CPU"
    timed: bool = True            # False: output only (a filter variant of a timed row)
    description: str = ""

    @abstractmethod
    def run(self, case: FairCase) -> Run:
        ...


class AnyEGContender(Contender):
    family = "AnyEG"

    def __init__(self, name: str, kernels: ExtremumGraphKernels, device: str, description: str,
                 options: AssemblyOptions = AssemblyOptions(), timed: bool = True,
                 uphill_only: bool = False):
        """``uphill_only``: a diagnostic, NOT AnyEG's behaviour: drop the arcs that
        leave a saddle through a lower neighbour (see ``_uphill_only``)."""
        self.name, self.kernels, self.device, self.description = name, kernels, device, description
        self.assembler = ArcAssembler(options)
        self.timed = timed
        self.uphill_only = uphill_only

    def run(self, case: FairCase) -> Run:
        t0 = time.perf_counter()
        out = self.kernels.run(case.field.values, case.table)
        t1 = time.perf_counter()
        g = self.assembler.assemble(case.field.values, case.table, out, heights=case.heights)
        t2 = time.perf_counter()
        if self.uphill_only:
            g = _uphill_only(g, case.order)
        extra = {"kernels_only": sum(v for k, v in out.seconds.items() if k.startswith("module"))}
        detect = t1 - t0
        if "compute_call" in out.seconds:      # original: time inside its process, not the spawn
            detect = out.seconds["compute_call"]
            extra["process_and_files"] = (t1 - t0) - detect
        return Run(from_anyeg(g, self.name, {}), detect, t2 - t1, extra)


def _uphill_only(g, order: np.ndarray):
    """The graph without arcs whose first step from the saddle goes down.

    AnyEG joins a saddle to the basin of *every* neighbour, also lower ones;
    the maximum such a lower neighbour ascends to need not be connected to the
    saddle above the saddle's height. Used only to show where AnyEG's merge-tree
    errors come from.
    """
    from dataclasses import replace

    start = g.arc_vertices[g.arc_ptr[:-1] + 1]
    keep = order[start] > order[g.arc_saddle]
    lengths = np.diff(g.arc_ptr)[keep]
    verts = np.concatenate([g.arc(i) for i in np.flatnonzero(keep)]) if keep.any() \
        else g.arc_vertices[:0]
    return replace(g, arc_saddle=g.arc_saddle[keep], arc_maximum=g.arc_maximum[keep],
                   arc_ptr=np.r_[0, np.cumsum(lengths)], arc_vertices=verts)


class OursContender(Contender):
    family = "Ours"

    def __init__(self, name: str, backend: Callable[[], object], device: str, description: str,
                 min_loop_relief: float = 0.5, timed: bool = True):
        self.name, self.backend, self.device, self.description = name, backend, device, description
        self.min_loop_relief = min_loop_relief
        self.timed = timed

    def run(self, case: FairCase) -> Run:
        t = {}
        # A new backend per run: the graph is copied to the GPU every time, as AnyEG's table is.
        eg = extremum_graph(case.heights, case.graph, min_loop_relief=self.min_loop_relief,
                            backend=self.backend(), timings=t)
        detect = t["order"] + t["ascent"] + t["passes"]
        post = t["persistence"] + t["relief"] + t["simplification"]
        return Run(from_ours(eg, self.name, {}), detect, post, dict(t),
                   merge_saddles=np.unique(eg.saddles[eg.saddle_is_merge]))


class _OneThread:
    """Limit torch to one CPU thread inside the block (NumPy's sort uses one)."""

    def __enter__(self):
        import torch
        self.torch, self.n = torch, torch.get_num_threads()
        torch.set_num_threads(1)

    def __exit__(self, *exc):
        self.torch.set_num_threads(self.n)


def default_contenders(gpu: Optional[str], cpu: bool = True) -> List[Contender]:
    out: List[Contender] = []
    if gpu is not None:
        idx = int(gpu.split(":")[1])
        if OriginalCudaKernels.is_built():
            out.append(AnyEGContender(
                "AnyEG original", OriginalCudaKernels(gpu=idx), "GPU",
                "AnyEG's unchanged CUDA `compute()` (cudaHelper.cu) in its driver process; "
                "*detect* is the time of the `compute()` call inside that process"))
        out.append(AnyEGContender(
            "AnyEG port", TorchKernels(device=gpu), "GPU",
            "`anyeg.TorchKernels`, the PyTorch port of the same four modules (checked equal to the "
            "original)"))
        out.append(AnyEGContender(
            "AnyEG +bundling", TorchKernels(device=gpu), "GPU",
            "the port with AnyEG's own edge bundling switched on (one saddle per pair of maxima, "
            "like ours); output only", AssemblyOptions(bundle_edges=True), timed=False))
        out.append(AnyEGContender(
            "AnyEG uphill arcs (diagnostic)", TorchKernels(device=gpu), "GPU",
            "NOT AnyEG: the port with the arcs that leave a saddle downhill removed, to show "
            "where AnyEG's merge-tree errors come from; output only", timed=False,
            uphill_only=True))
        out.append(OursContender(
            "Ours", lambda: TorchBackend(gpu), "GPU",
            "`pipeline.extremum_graph`, torch backend, as used (loop filter on, relief >= 0.5)"))
        out.append(OursContender(
            "Ours no loop filter", lambda: TorchBackend(gpu), "GPU",
            "the same with `min_loop_relief=0`: every pass between adjacent basins; output only",
            min_loop_relief=0.0, timed=False))
    if cpu:
        out.append(AnyEGContender("AnyEG port CPU", TorchKernels(device="cpu"), "CPU",
                                  "the PyTorch port on the CPU, one thread"))
        out.append(OursContender("Ours CPU", NumpyBackend, "CPU",
                                 "`pipeline.extremum_graph`, NumPy backend, one thread"))
    return out


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------
@dataclass
class Timing:
    detect: List[float]
    post: List[float]
    extra: Dict[str, List[float]]

    @staticmethod
    def _summary(xs: List[float]) -> dict:
        return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "runs": len(xs)}

    def as_dict(self) -> dict:
        tot = [a + b for a, b in zip(self.detect, self.post)]
        d = {"detect": self._summary(self.detect), "post": self._summary(self.post),
             "total": self._summary(tot)}
        d.update({k: self._summary(v) for k, v in self.extra.items()})
        return d


@dataclass
class ContenderResult:
    name: str
    family: str
    device: str
    timed: bool
    maxima: int
    saddles: int
    pairs: int
    fidelity: MergeTreeFidelity
    pl_valid: Optional[Tuple[int, int]]           # (valid, all) saddles; None if not triangulated
    pl_valid_merge: Optional[Tuple[int, int]]     # ours: merge saddles only
    multi_saddles: int                            # saddles reaching 3+ maxima
    timing: Optional[Timing]

    def as_dict(self) -> dict:
        return {"name": self.name, "family": self.family, "device": self.device,
                "maxima": self.maxima, "saddles": self.saddles, "pairs": self.pairs,
                "merge_tree": self.fidelity.as_dict(), "pl_valid": self.pl_valid,
                "pl_valid_merge": self.pl_valid_merge, "multi_saddles": self.multi_saddles,
                "timing": self.timing.as_dict() if self.timing else None}


@dataclass
class CaseResult:
    input: str
    source: str
    neighborhood: str
    num_points: int
    num_edges: int
    build_seconds: float
    reference_seconds: float
    reference_maxima: int
    significant: float
    contenders: List[ContenderResult] = field(default_factory=list)
    agreements: List[Agreement] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"input": self.input, "source": self.source, "neighborhood": self.neighborhood,
                "num_points": self.num_points, "num_edges": self.num_edges,
                "build_seconds": self.build_seconds, "reference_seconds": self.reference_seconds,
                "reference_maxima": self.reference_maxima, "significant": self.significant,
                "contenders": [c.as_dict() for c in self.contenders],
                "agreements": [a.as_dict() for a in self.agreements]}


AGREEMENT_PAIRS = [
    ("AnyEG original", "AnyEG port"),          # is the port exact? (must be identical)
    ("Ours CPU", "Ours"),                      # are our backends exact? (must be identical)
    ("AnyEG port", "Ours"),                    # each as used, same graph and order
    ("AnyEG +bundling", "Ours"),               # both one saddle per pair
    ("AnyEG port", "Ours no loop filter"),     # cores without our filter
]


class FairStudy:
    def __init__(self, inputs: Sequence[FieldInput], neighborhoods: Sequence[str],
                 contenders: Sequence[Contender], repeats: int = 5,
                 significant_fraction: float = 0.01):
        self.inputs, self.neighborhoods, self.contenders = inputs, neighborhoods, contenders
        self.repeats, self.significant_fraction = repeats, significant_fraction

    def run(self, log=print) -> List[CaseResult]:
        results = []
        for inp in self.inputs:
            base = inp.load()
            for nb in self.neighborhoods:
                case = make_case(inp, base, nb)
                results.append(self.run_case(case))
                log(_one_line(results[-1]))
        return results

    def run_case(self, case: FairCase) -> CaseResult:
        t0 = time.perf_counter()
        reference = merge_tree_pairs(case.order, case.graph.edges)
        ref_seconds = time.perf_counter() - t0
        h = case.heights
        significant = self.significant_fraction * float(h.max() - h.min())
        res = CaseResult(case.input, case.source, case.neighborhood, case.num_points,
                         case.graph.num_edges, case.build_seconds, ref_seconds, len(reference),
                         significant)
        graphs: Dict[str, CommonGraph] = {}
        for c in self.contenders:
            run, timing = self._run(c, case)
            graphs[c.name] = run.graph
            res.contenders.append(self._evaluate(c, case, run, timing, reference, significant))
        for a, b in AGREEMENT_PAIRS:
            if a in graphs and b in graphs:
                res.agreements.append(compare(graphs[a], graphs[b]))
        return res

    def _run(self, c: Contender, case: FairCase) -> Tuple[Run, Optional[Timing]]:
        guard = _OneThread() if c.device == "CPU" else _NoGuard()
        with guard:
            first = c.run(case)                       # warm-up; also the output
            if not c.timed:
                return first, None
            runs = [c.run(case) for _ in range(self.repeats)]
        keys = sorted(set().union(*(r.extra for r in runs)))
        return first, Timing([r.detect for r in runs], [r.post for r in runs],
                             {k: [r.extra[k] for r in runs if k in r.extra] for k in keys})

    def _evaluate(self, c: Contender, case: FairCase, run: Run, timing, reference,
                  significant) -> ContenderResult:
        g = run.graph
        derived = graph_persistence(g.maxima, g.links, case.order)
        fid = merge_tree_fidelity(reference, derived, case.order, case.heights, significant)
        pl = pl_merge = None
        if case.triangulated:
            ok = pl_saddle_valid(g.saddles, case.table.table, case.order, case.field.dims,
                                 case.offsets)
            pl = (int(ok.sum()), int(ok.shape[0]))
            if run.merge_saddles is not None:
                okm = pl_saddle_valid(run.merge_saddles, case.table.table, case.order,
                                      case.field.dims, case.offsets)
                pl_merge = (int(okm.sum()), int(okm.shape[0]))
        reach: Dict[int, set] = {}
        for s, m1, m2 in g.links.tolist():
            reach.setdefault(s, set()).update((m1, m2))
        return ContenderResult(
            name=c.name, family=c.family, device=c.device, timed=c.timed,
            maxima=g.num_maxima, saddles=g.num_saddles, pairs=len(g.maximum_pairs),
            fidelity=fid, pl_valid=pl, pl_valid_merge=pl_merge,
            multi_saddles=sum(len(v) > 2 for v in reach.values()), timing=timing)


class _NoGuard:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _one_line(r: CaseResult) -> str:
    parts = []
    for c in r.contenders:
        t = f" {statistics.median(c.timing.detect):.3f}s" if c.timing else ""
        parts.append(f"{c.name}: {c.maxima}/{c.saddles} MT {c.fidelity.exact_fraction:.3f}{t}")
    return f"{r.input} [{r.neighborhood}] " + "; ".join(parts)


# ---------------------------------------------------------------------------
DEFAULT_VOLUMES = ["nucleon", "marschner_lobb", "silicium", "fuel", "neghip", "hydrogen_atom"]


def main(argv=None) -> None:
    from .fair_report import write_fair_report

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", nargs="*", default=DEFAULT_VOLUMES)
    ap.add_argument("--no-samples", action="store_true", help="skip AnyEG's sample .vti files")
    ap.add_argument("--neighborhoods", nargs="*", default=list(NEIGHBORHOODS),
                    choices=list(NEIGHBORHOODS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-cpu", action="store_true", help="skip the one-thread CPU runs")
    ap.add_argument("--gpu", default=None, help="cuda:N (default: the one with most free memory)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args(argv)

    gpu = args.gpu or pick_cuda_device()
    gpu = None if gpu == "cpu" else gpu
    contenders = default_contenders(gpu, cpu=not args.no_cpu)
    _warm_up(contenders)
    inputs = ([] if args.no_samples else anyeg_samples()) + klacansky_volumes(args.volumes)
    study = FairStudy(inputs, args.neighborhoods, contenders, repeats=args.repeats)
    results = study.run()
    md, js = write_fair_report(results, contenders, Path(args.out), gpu, args.repeats)
    print("wrote", md, "and", js)


def _warm_up(contenders: Sequence[Contender]) -> None:
    """One tiny case per contender, so CUDA/torch start-up is outside every timed run."""
    rng = np.random.default_rng(0)
    base = ScalarField.from_grid(rng.random((6, 7, 8)))
    inp = FieldInput("warm-up", "random", lambda: base)
    for nb in NEIGHBORHOODS:
        case = make_case(inp, base, nb)
        for c in contenders:
            c.run(case)


if __name__ == "__main__":
    main()
