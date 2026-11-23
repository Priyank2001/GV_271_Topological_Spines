"""
Step 2: is the kNN extremum graph of scattered samples right on a loss landscape?

    python3 -m landscape.accuracy --run cnn_m4_e30            # d = 3 (4 models), ~10 min on one GPU

Reference. In low dimension a dense grid is affordable. On the Freudenthal triangulation of the grid
(``pipeline.neighbor_graph.freudenthal_grid_graph``) the sampled loss interpolates to a piecewise-linear
field, whose maxima and merge saddles our sweep computes exactly. Grids of several resolutions show
that the reference has converged.

Two checks that use no topology method at all:
  1. every trained model must sit at a reference minimum (they are minima of the training objective);
  2. the loss along the straight segment between two models is an upper bound on their barrier (the
     barrier is the lowest possible highest point over all paths), so the reference barrier may never
     exceed it. How much lower it is shows how far from straight the low path runs.

Under test: our extremum graph on a kNN graph (k = 8, 15, 30) of N scattered samples (Sobol and uniform,
N = 1,024 .. 32,768, three seeds). Both graphs see f = -log L scaled by the reference's range and are
simplified at the same persistence threshold tau. Scores (peaks are the minima of L):
  * minima: Hungarian matching of positions (box scaled to the unit cube), accepted within ``radius``;
    recall (reference minima found), precision (found minima that are real), position error;
  * barriers: for every pair of matched minima, the minimax barrier (the saddle level at which their
    basins join, over all saddles of the graph) in loss units, relative error vs the reference;
  * links: which pairs of minima are joined directly by a saddle in the simplified graphs.

Writes comparison/results/loss_landscape_accuracy.{md,json,png}.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.extremum_graph import ExtremumGraph, extremum_graph
from pipeline.neighbor_graph import freudenthal_grid_graph, knn_graph

from . import ROOT
from .run import LandscapeRun

RESULTS = ROOT / "comparison" / "results"


# ---------------------------------------------------------------------------------------------
# graph summaries
# ---------------------------------------------------------------------------------------------
def minimax_barriers(eg: ExtremumGraph) -> np.ndarray:
    """(P, P) array: the highest level at which peaks i and j are joined by a path of saddles
    (= where their superlevel components merge); -inf if never. Uses every saddle of ``eg``."""
    p = eg.num_peaks
    parent = np.arange(p)
    members = {i: [i] for i in range(p)}
    out = np.full((p, p), -np.inf)
    np.fill_diagonal(out, np.inf)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in np.argsort(-eg.saddle_values, kind="stable"):
        a, b = find(eg.saddle_peaks[s, 0]), find(eg.saddle_peaks[s, 1])
        if a == b:
            continue
        ma, mb = members[a], members[b]
        out[np.ix_(ma, mb)] = eg.saddle_values[s]
        out[np.ix_(mb, ma)] = eg.saddle_values[s]
        if len(ma) < len(mb):
            a, b, ma, mb = b, a, mb, ma
        parent[b] = a
        ma.extend(mb)
        del members[b]
    return out


@dataclass
class Summary:
    """What is compared: surviving peaks (positions, values), barriers between them, links."""
    pos: np.ndarray            # (P, d) unit-cube coordinates of the surviving peaks
    value: np.ndarray          # (P,) f at the peaks
    barrier: np.ndarray        # (P, P) minimax barrier in f units, over all saddles of the full graph
    links: set                 # {(i, j)} pairs joined directly in the simplified graph
    num_peaks_raw: int

    @classmethod
    def of(cls, eg: ExtremumGraph, unit_points: np.ndarray, tau: float) -> "Summary":
        simple = eg.simplify(tau)
        keep = np.flatnonzero(eg.peak_persistence >= tau)      # simplify() keeps exactly these, in order
        assert np.array_equal(eg.peaks[keep], simple.peaks)
        bar = minimax_barriers(eg)[np.ix_(keep, keep)]
        links = {tuple(sorted(map(int, pr))) for pr in simple.saddle_peaks}
        return cls(unit_points[simple.peaks], simple.peak_values, bar, links, eg.num_peaks)


def compare(ref: Summary, got: Summary, f_to_loss, radius: float) -> dict:
    d = np.linalg.norm(ref.pos[:, None] - got.pos[None], axis=-1)
    r, c = linear_sum_assignment(d)
    ok = d[r, c] <= radius
    r, c = r[ok], c[ok]
    match = dict(zip(r.tolist(), c.tolist()))
    out = {"ref_minima": len(ref.pos), "found_minima": len(got.pos), "matched": len(r),
           "recall": len(r) / max(len(ref.pos), 1), "precision": len(r) / max(len(got.pos), 1),
           "pos_err_median": float(np.median(d[r, c])) if len(r) else None,
           "minimum_loss_rel_err_max": (float(np.max(np.abs(f_to_loss(got.value[c]) / f_to_loss(ref.value[r]) - 1)))
                                        if len(r) else None)}
    # barriers between matched minima
    errs = []
    for a in range(len(r)):
        for b in range(a + 1, len(r)):
            br, bg = ref.barrier[r[a], r[b]], got.barrier[c[a], c[b]]
            if np.isfinite(br) and np.isfinite(bg):
                errs.append(abs(f_to_loss(bg) / f_to_loss(br) - 1))
            else:
                errs.append(np.inf if np.isfinite(br) != np.isfinite(bg) else 0.0)
    out["barrier_rel_err_median"] = float(np.median(errs)) if errs else None
    out["barrier_rel_err_max"] = float(np.max(errs)) if errs else None
    # direct links among matched minima
    ref_links = {pr for pr in ref.links if pr[0] in match and pr[1] in match}
    got_links = {tuple(sorted((match[a], match[b]))) for a, b in ref_links}
    inv = {v: k for k, v in match.items()}
    got_matched = {pr for pr in got.links if pr[0] in inv and pr[1] in inv}
    out["link_recall"] = len(got_links & got_matched) / len(ref_links) if ref_links else 1.0
    out["link_precision"] = len(got_links & got_matched) / len(got_matched) if got_matched else 1.0
    out["raw_peaks"] = got.num_peaks_raw
    return out


# ---------------------------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------------------------
class AccuracyStudy:
    def __init__(self, run: LandscapeRun, grid_sizes: List[int], ns: List[int], ks: List[int],
                 samplers: List[str], seeds: List[int], taus: List[float], radius: float):
        self.run, self.grid_sizes, self.ns, self.ks = run, grid_sizes, ns, ks
        self.samplers, self.seeds, self.taus, self.radius = samplers, seeds, taus, radius
        self.timings: Dict[str, float] = {}

    # f = -log L scaled with the finest grid's range
    def _scale(self, loss: np.ndarray) -> np.ndarray:
        return (-np.log(loss) - self.lo) / (self.hi - self.lo)

    def f_to_loss(self, f):
        return np.exp(-(self.lo + np.asarray(f) * (self.hi - self.lo)))

    def reference(self) -> Tuple[dict, Dict[float, Summary], dict]:
        fine = max(self.grid_sizes)
        g, _ = self.run.grid(fine)
        self.lo, self.hi = float(-np.log(g["loss"]).max()), float(-np.log(g["loss"]).min())
        self.lo, self.hi = min(self.lo, self.hi), max(self.lo, self.hi)
        conv, refs, eg_fine = {}, {}, None
        for n in sorted(self.grid_sizes):
            g, shape = self.run.grid(n, finer=fine)
            t0 = time.time()
            eg = extremum_graph(self._scale(g["loss"]), freudenthal_grid_graph(shape))
            self.timings[f"grid{n}_graph_s"] = round(time.time() - t0, 2)
            unit = self.run.box.normalise(g["x"])
            conv[n] = {tau: Summary.of(eg, unit, tau) for tau in self.taus}
            if n == fine:
                refs, eg_fine, gfine = conv[n], eg, g
        # convergence: each coarser grid scored against the finest
        convergence = {n: {tau: compare(refs[tau], conv[n][tau], self.f_to_loss, self.radius)
                           for tau in self.taus} for n in self.grid_sizes if n != fine}
        return {"eg": eg_fine, "grid": gfine}, refs, convergence

    def ground_truth_checks(self, ref_eg: ExtremumGraph, grid: dict) -> dict:
        """The two method-free checks (module docstring), on the UNSIMPLIFIED reference graph: a
        model is checked against the nearest reference minimum of any persistence, and two models'
        barrier is the minimax barrier between the basins of the grid vertices nearest to them."""
        run, box = self.run, self.run.box
        anchors = run.anchors()
        unit_grid = box.normalise(grid["x"])
        unit_anchor = box.normalise(anchors["x"])
        pk = unit_grid[ref_eg.peaks]
        models = []
        for i, a in enumerate(unit_anchor):
            dist = np.linalg.norm(pk - a, axis=1)
            j = int(np.argmin(dist))
            basin = int(ref_eg.region[np.argmin(np.linalg.norm(unit_grid - a, axis=1))])
            models.append({"model": i, "loss": float(anchors["loss"][i]), "nearest_minimum": j,
                           "distance": float(dist[j]), "basin": basin, "in_own_basin": basin == j,
                           "minimum_loss": float(self.f_to_loss(ref_eg.peak_values[j])),
                           "minimum_persistence": float(ref_eg.peak_persistence[j])})
        spacing = float(1 / (max(self.grid_sizes) - 1))
        lin = run.linear_paths()
        bar = minimax_barriers(ref_eg)
        pairs = []
        steps = len(lin["t"])
        for p, (i, j) in enumerate(lin["pairs"]):
            seg = lin["loss"][p * steps:(p + 1) * steps]
            bi, bj = models[i]["basin"], models[j]["basin"]
            ref_bar = float(self.f_to_loss(bar[bi, bj])) if bi != bj else None
            pairs.append({"pair": [int(i), int(j)], "linear_barrier_loss": float(seg.max()),
                          "reference_barrier_loss": ref_bar, "same_basin": bi == bj,
                          "excess": None if ref_bar is None else ref_bar / float(seg.max()) - 1})
        return {"grid_spacing_unit": spacing, "models": models, "pairs": pairs,
                "distinct_minima_for_models": len({m["basin"] for m in models})}

    def scattered(self, refs: Dict[float, Summary]) -> List[dict]:
        rows = []
        nmax = max(self.ns)
        for sampler in self.samplers:
            for seed in self.seeds:
                s = self.run.scattered(sampler, nmax, seed)
                for n in self.ns:
                    x, f = s["x"][:n], self._scale(s["loss"][:n])
                    unit = self.run.box.normalise(x)
                    for k in self.ks:
                        t0 = time.time()
                        eg = extremum_graph(f, knn_graph(x, k))
                        secs = time.time() - t0
                        for tau in self.taus:
                            row = {"sampler": sampler, "seed": seed, "n": n, "k": k, "tau": tau,
                                   "graph_s": round(secs, 3)}
                            row.update(compare(refs[tau], Summary.of(eg, unit, tau), self.f_to_loss, self.radius))
                            rows.append(row)
                print(f"  {sampler} seed {seed} done", flush=True)
        return rows


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def _agg(rows, keys, fields):
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        rec = dict(zip(keys, g))
        for f in fields:
            v = np.array([r[f] for r in rs if r[f] is not None], float)
            if not len(v):
                rec[f] = None
                continue
            rec[f] = (float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else float("inf"),
                      float(np.std(v[np.isfinite(v)])) if np.isfinite(v).any() else 0.0,
                      int((~np.isfinite(v)).sum()))
        out.append(rec)
    return out


def _f(v, p=3) -> str:
    return "–" if v is None else ("∞" if not np.isfinite(v) else f"{v:.{p}f}")


def report(study: AccuracyStudy, refs, convergence, checks, rows, path_md, path_json) -> None:
    run = study.run
    fine = max(study.grid_sizes)
    models = run.meta["models"]
    acc = f"{min(m['test']['acc'] for m in models):.1%}–{max(m['test']['acc'] for m in models):.1%}"
    origin = (f"children of one parent trained {run.meta['spawn_epoch']} epochs" if run.meta["spawn_epoch"]
              else "independent initialisations")
    L = [f"# Loss-landscape accuracy: scattered kNN extremum graph vs a dense PL grid ({run.name})", "",
         f"Landscape: training objective (cross-entropy on {run.n_eval} fixed CIFAR-10 training images + weight decay "
         f"{run.l2:g}/2·‖θ‖²) of {len(models)} SmallCNNs ({origin}; test accuracy {acc}) in the {run.box.dim}-D "
         "affine subspace through their weights.", "",
         "f = −log L, scaled with the reference's range; minima of L are the peaks. Box: the models' bounding box "
         f"+30% per side. Positions are compared in the box scaled to the unit cube; match radius {study.radius}.", ""]
    L += [f"## Reference: {fine}^{run.box.dim} Freudenthal grid", ""]
    for tau in study.taus:
        r = refs[tau]
        L.append(f"* τ = {tau}: **{len(r.pos)} minima**, {len(r.links)} links "
                 f"(unsimplified: {r.num_peaks_raw} peaks); minimum losses "
                 + ", ".join(f"{study.f_to_loss(v):.3f}" for v in r.value))
    L += ["", "Convergence (coarser grid scored against the finest):", "",
          "| grid | τ | minima | recall | precision | position error | barrier rel. error (median / max) |",
          "|---|---|---|---|---|---|---|"]
    for n, by_tau in sorted(convergence.items()):
        for tau, c in by_tau.items():
            L.append(f"| {n}^{run.box.dim} | {tau} | {c['found_minima']} | {c['recall']:.2f} | {c['precision']:.2f} | "
                     f"{_f(c['pos_err_median'])} | {_f(c['barrier_rel_err_median'])} / {_f(c['barrier_rel_err_max'])} |")
    L += ["", "## Method-free checks", "",
          f"1. Trained models vs reference minima (unsimplified graph; grid spacing {checks['grid_spacing_unit']:.3f} "
          "of the box):", "",
          "| model | its loss | nearest reference minimum | distance | loss there | its persistence | model inside that minimum's basin |",
          "|---|---|---|---|---|---|---|"]
    for m in checks["models"]:
        L.append(f"| {m['model']} | {m['loss']:.4f} | #{m['nearest_minimum']} | {m['distance']:.3f} | {m['minimum_loss']:.4f} | "
                 f"{_f(m['minimum_persistence'])} | {'yes' if m['in_own_basin'] else 'no'} |")
    L += ["", f"The {len(checks['models'])} models lie in {checks['distinct_minima_for_models']} distinct reference basins.", "",
          "2. Barrier between two models: straight segment (an upper bound) vs reference (lowest path between their basins "
          "in the subspace):", "",
          "| models | straight-line barrier (loss) | reference barrier (loss) | bound holds |", "|---|---|---|---|"]
    for p in checks["pairs"]:
        if p["same_basin"]:
            verdict, rb = "same basin", "–"
        else:
            rb = f"{p['reference_barrier_loss']:.4f}"
            verdict = "yes" if p["excess"] <= 1e-6 else f"**no** (+{100 * p['excess']:.2f}%)"
        L.append(f"| {p['pair'][0]}–{p['pair'][1]} | {p['linear_barrier_loss']:.4f} | {rb} | {verdict} |")
    fields = ["recall", "precision", "pos_err_median", "barrier_rel_err_median", "barrier_rel_err_max",
              "link_recall", "link_precision", "found_minima"]
    agg = _agg(rows, ["tau", "sampler", "k", "n"], fields)
    for tau in study.taus:
        L += ["", f"## Scattered samples vs reference, τ = {tau} (mean ± sd over {len(study.seeds)} seeds)", "",
              "| sampler | k | N | minima found | recall | precision | position error | barrier err. median | barrier err. max | link recall | link precision |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in agg:
            if a["tau"] != tau:
                continue

            def ms(f, p=2):
                if a[f] is None:
                    return "–"
                m, s, ninf = a[f]
                return ("∞" if not np.isfinite(m) else f"{m:.{p}f} ± {s:.{p}f}") + (f" ({ninf}×∞)" if ninf else "")
            L.append(f"| {a['sampler']} | {a['k']} | {a['n']:,} | {ms('found_minima', 1)} | {ms('recall')} | {ms('precision')} | "
                     f"{ms('pos_err_median', 3)} | {ms('barrier_rel_err_median', 3)} | {ms('barrier_rel_err_max', 3)} | "
                     f"{ms('link_recall')} | {ms('link_precision')} |")
    L += ["", "Timings (s): " + ", ".join(f"{k} {v}" for k, v in study.timings.items()), ""]
    path_md.write_text("\n".join(L) + "\n")
    path_json.write_text(json.dumps({"run": run.name, "box": run.box.to_dict(), "taus": study.taus,
                                     "reference": {str(t): {"minima": len(r.pos), "links": sorted(r.links)}
                                                   for t, r in refs.items()},
                                     "convergence": {str(n): {str(t): c for t, c in v.items()} for n, v in convergence.items()},
                                     "checks": checks, "rows": rows, "timings": study.timings}, indent=1, default=float))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="cnn_m4_e30")
    ap.add_argument("--grids", type=int, nargs="+", default=[13, 25, 49])
    ap.add_argument("--ns", type=int, nargs="+", default=[1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 15, 30])
    ap.add_argument("--samplers", nargs="+", default=["sobol", "uniform"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.005, 0.02])
    ap.add_argument("--radius", type=float, default=0.1)
    args = ap.parse_args()

    run = LandscapeRun.load(args.run)
    study = AccuracyStudy(run, args.grids, args.ns, args.ks, args.samplers, args.seeds, args.taus, args.radius)
    ref, refs, convergence = study.reference()
    checks = study.ground_truth_checks(ref["eg"], ref["grid"])
    rows = study.scattered(refs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"loss_landscape_accuracy_{args.run}"
    report(study, refs, convergence, checks, rows, RESULTS / f"{stem}.md", RESULTS / f"{stem}.json")


if __name__ == "__main__":
    main()
