"""
Step 3: what do 2-D slices of a loss landscape show, compared with the extremum graph of the whole subspace?

    python3 -m landscape.slices --run cnn_m4_e30          # d = 3: full view = the 49^3 PL grid of step 2
    python3 -m landscape.slices --run cnn_m6_e30          # d = 5: full view = kNN graph of 65,536 Sobol samples

The usual pictures of a loss landscape are 2-D:
  * the plane through three trained models (mode-connectivity papers; Garipov et al. 2018);
  * Li et al.'s (2018) plane around one model spanned by two filter-normalised random directions.
Two facts hold whatever method is used: a slice shows only the minima it passes through, and the barrier
between two minima measured inside a slice can never be lower than in the full space (fewer paths).
This study measures how much that matters, slice by slice:

  * every slice minimum is placed in the full landscape: "true" if it lies within ``radius`` of the minimum
    of the full basin it falls in, "displaced" if it lies in a full basin but away from its minimum (the
    slice cuts the basin's wall, so the picture shows a minimum where the full space has none); a basin
    shown by two or more slice minima is "split";
  * full minima that no slice minimum represents are "missed";
  * for two slice minima in different full basins, the slice barrier is compared with the full barrier
    between those basins (both in loss units; excess = how much the slice overstates it);
  * links and loops: the full graph's saddles between minima vs the slice's.

Two method-free checks keep the full view honest where no grid reference exists (d = 5):
  * barriers: the full graph's minimax path between two models' basins is evaluated densely; its maximum
    is the barrier of a real path, so it bounds the true barrier from above. A slice barrier above it
    overstates the truth by at least that much, whoever computed the path;
  * minima no model sits in: gradient descent (finite differences) from them shows whether they are basins
    of their own or sampling artefacts that slide into a model's basin.
The full kNN view uses k = 30 for d >= 4 (k = 15 gives false minima in 5-D; see the robustness table).

Both views are computed with ``pipeline.extremum_graph``; slices on the Freudenthal triangulation of a
65 x 65 grid (exact for the PL slice). f = -log L scaled with the full view's range; persistence threshold
``tau`` in those units. Writes comparison/results/loss_landscape_slices_<run>.{md,json,png}.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

from pipeline.extremum_graph import ExtremumGraph, extremum_graph
from pipeline.neighbor_graph import freudenthal_grid_graph, knn_graph

from .accuracy import RESULTS, minimax_barriers
from .run import LandscapeRun
from .sampling import Box


# ---------------------------------------------------------------------------------------------
# planes
# ---------------------------------------------------------------------------------------------
@dataclass
class Plane:
    name: str
    origin: np.ndarray        # (d,) subspace coordinates of the plane's origin
    basis: np.ndarray         # (2, d) orthonormal
    box: Box                  # 2-D extent in plane coordinates
    models: tuple             # models lying in the plane

    def to_subspace(self, u: np.ndarray) -> np.ndarray:
        return self.origin + u @ self.basis


def triple_planes(anchors: np.ndarray, margin: float = 0.3) -> List[Plane]:
    """The plane through every three models, extent = their bounding box in the plane + ``margin``."""
    out = []
    for tri in itertools.combinations(range(len(anchors)), 3):
        a, b, c = anchors[list(tri)]
        o = (a + b + c) / 3
        q, _ = np.linalg.qr(np.stack([b - a, c - a], 1))
        basis = q.T
        uv = (anchors[list(tri)] - o) @ basis.T
        out.append(Plane("models" + "".join(map(str, tri)), o, basis, Box.around(uv, margin), tri))
    return out


# ---------------------------------------------------------------------------------------------
# the full view (d-dimensional extremum graph)
# ---------------------------------------------------------------------------------------------
@dataclass
class FullView:
    kind: str
    unit: np.ndarray          # (N, d) samples in unit-box coordinates
    eg: ExtremumGraph         # unsimplified
    keep: np.ndarray          # peak ids of eg that survive tau
    barrier: np.ndarray       # (M, M) minimax barrier (f units) between surviving minima
    lo: float
    hi: float
    x: np.ndarray = None      # (N, d) samples in subspace coordinates
    f: np.ndarray = None      # (N,) scaled f
    graph: object = None      # the NeighborGraph
    ref_unit: np.ndarray = None   # (M, d) reference position per minimum (set by ``use_models``)

    def use_models(self, run: LandscapeRun, tau: float) -> np.ndarray:
        """Reference position of each minimum: the trained model lying in its basin (an exact minimum of the
        objective) if there is one, else the best sample. Returns the basin of every model."""
        model_unit = run.box.normalise(run.anchors()["x"])
        basins = self.basin(model_unit, tau)
        ref = self.minima_unit.copy()
        for m, b in enumerate(basins):
            ref[b] = model_unit[m]
        self.ref_unit = ref
        return basins

    def bottleneck_path(self, a: int, b: int) -> np.ndarray:
        """Vertices of the path between minima ``a`` and ``b`` whose lowest f is highest (the path through the
        minimax saddle), via the maximum spanning tree of the graph (edge level = lower endpoint's f)."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
        if not hasattr(self, "_mst"):
            e = self.graph.edges
            level = np.minimum(self.f[e[:, 0]], self.f[e[:, 1]])
            n = len(self.f)
            w = (2.0 - level)                           # > 0; small weight = high level
            t = minimum_spanning_tree(coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr())
            self._mst = (t + t.T).tocsr()
        src, dst = self.eg.peaks[self.keep[a]], self.eg.peaks[self.keep[b]]
        _, pred = breadth_first_order(self._mst, src, directed=False, return_predecessors=True)
        path = [dst]
        while path[-1] != src:
            path.append(pred[path[-1]])
            if path[-1] < 0:
                raise ValueError("minima not connected")
        return np.array(path[::-1])

    @property
    def num_minima(self) -> int:
        return len(self.keep)

    @property
    def minima_unit(self) -> np.ndarray:
        return self.unit[self.eg.peaks[self.keep]]

    def basin(self, unit_pts: np.ndarray, tau: float) -> np.ndarray:
        """Surviving minimum (0..M-1) whose simplified basin contains the nearest sample of each point."""
        if not hasattr(self, "_tree"):
            self._tree = cKDTree(self.unit)
            simple = self.eg.simplify(tau)
            self._region = simple.region
        _, idx = self._tree.query(unit_pts)
        return self._region[idx]

    def scale(self, loss: np.ndarray) -> np.ndarray:
        return (-np.log(loss) - self.lo) / (self.hi - self.lo)

    def to_loss(self, f) -> np.ndarray:
        return np.exp(-(self.lo + np.asarray(f) * (self.hi - self.lo)))

    @classmethod
    def build(cls, run: LandscapeRun, kind: str, tau: float, n: int, k: int) -> "FullView":
        if kind == "grid":
            s, shape = run.grid(n)
            graph = freudenthal_grid_graph(shape)
        else:
            s = run.scattered("sobol", n, 0)
            graph = knn_graph(s["x"], k)
        g = -np.log(s["loss"])
        lo, hi = float(g.min()), float(g.max())
        f = (g - lo) / (hi - lo)
        eg = extremum_graph(f, graph)
        keep = np.flatnonzero(eg.peak_persistence >= tau)
        return cls(f"{kind} ({n}{'^' + str(run.box.dim) if kind == 'grid' else f' Sobol samples, k = {k}'})",
                   run.box.normalise(s["x"]), eg, keep, minimax_barriers(eg)[np.ix_(keep, keep)], lo, hi,
                   x=s["x"], f=f, graph=graph)


# ---------------------------------------------------------------------------------------------
# one slice
# ---------------------------------------------------------------------------------------------
def analyse_slice(name: str, models: tuple, u: np.ndarray, x_sub: Optional[np.ndarray], loss: np.ndarray,
                  n: int, full: FullView, run: LandscapeRun, tau: float, radius: float) -> dict:
    """``x_sub``: subspace coordinates of the slice samples (None for slices outside the subspace)."""
    f = full.scale(loss)
    eg = extremum_graph(f, freudenthal_grid_graph((n, n)))
    simple = eg.simplify(tau)
    keep = np.flatnonzero(eg.peak_persistence >= tau)
    bar = minimax_barriers(eg)[np.ix_(keep, keep)]
    rec = {"slice": name, "models": list(models), "minima": int(simple.num_peaks),
           "saddles": int(simple.num_saddles), "loops": int((~simple.saddle_is_merge).sum()),
           "loss_min": float(loss.min()), "loss_max": float(loss.max()),
           "minimum_losses": [float(v) for v in full.to_loss(simple.peak_values)]}
    if x_sub is None:                           # outside the subspace: no placement in the full view
        rec.update({"true": None, "displaced": None, "split_basins": None, "missed": None})
        return rec
    unit = run.box.normalise(x_sub[simple.peaks])
    basins = full.basin(unit, tau)
    ref = full.ref_unit if full.ref_unit is not None else full.minima_unit
    dist = np.linalg.norm(unit - ref[basins], axis=1)
    true = dist <= radius
    rec["placement"] = [{"basin": int(b), "distance": float(d), "true": bool(t),
                         "loss_excess": float(full.to_loss(simple.peak_values[i]) /
                                              full.to_loss(full.eg.peak_values[full.keep[b]]) - 1)}
                        for i, (b, d, t) in enumerate(zip(basins, dist, true))]
    counts = np.bincount(basins, minlength=full.num_minima)
    rec["true"] = int(len(set(basins[true].tolist())))
    rec["displaced"] = int((~true).sum())
    rec["split_basins"] = int((counts > 1).sum())
    rec["shown_basins"] = sorted(set(basins.tolist()))
    rec["missed"] = int(full.num_minima - len(set(basins[true].tolist())))
    # barriers between slice minima in different full basins
    pairs = []
    for i, j in itertools.combinations(range(simple.num_peaks), 2):
        bi, bj = basins[i], basins[j]
        if bi == bj or not np.isfinite(bar[i, j]):
            continue
        ls, lf = float(full.to_loss(bar[i, j])), float(full.to_loss(full.barrier[bi, bj]))
        pairs.append({"basins": [int(bi), int(bj)], "slice_barrier": ls, "full_barrier": lf, "excess": ls / lf - 1})
    rec["pairs"] = pairs
    rec["barrier_excess_max"] = max((p["excess"] for p in pairs), default=None)
    rec["barrier_excess_median"] = float(np.median([p["excess"] for p in pairs])) if pairs else None
    return rec


# ---------------------------------------------------------------------------------------------
# study + report
# ---------------------------------------------------------------------------------------------
def full_checks(run: LandscapeRun, full: FullView, tau: float) -> dict:
    """Method-free checks of the full view: models in distinct basins; full barrier <= straight-line barrier."""
    anchors = run.anchors()
    basins = full.basin(run.box.normalise(anchors["x"]), tau)
    lin = run.linear_paths()
    steps = len(lin["t"])
    pairs = []
    for p, (i, j) in enumerate(lin["pairs"]):
        seg = lin["loss"][p * steps:(p + 1) * steps]
        bi, bj = basins[i], basins[j]
        fb = None if bi == bj else float(full.to_loss(full.barrier[bi, bj]))
        pairs.append({"pair": [int(i), int(j)], "straight": float(seg.max()), "full": fb,
                      "holds": None if fb is None else bool(fb <= seg.max() * (1 + 1e-6))})
    return {"model_basins": basins.tolist(), "distinct": int(len(set(basins.tolist()))), "pairs": pairs}


def certified_paths(run: LandscapeRun, full: FullView, basins: np.ndarray, per_edge: int = 8) -> dict:
    """For every two models in different basins: the full graph's minimax path between their basins' minima,
    with the loss evaluated densely along its straight pieces. Its maximum is the barrier of a real path in
    the subspace, hence an upper bound on the true barrier that does not depend on any topology method."""
    import hashlib

    t = np.linspace(0, 1, per_edge + 1)[:-1]
    pairs, chunks = [], []
    for i, j in itertools.combinations(range(len(basins)), 2):
        if basins[i] == basins[j]:
            continue
        path = full.bottleneck_path(int(basins[i]), int(basins[j]))
        x = full.x[path]                                                    # (V, d)
        inner = (1 - t)[None, :, None] * x[:-1, None, :] + t[None, :, None] * x[1:, None, :]
        chunks.append(np.concatenate([inner.reshape(-1, x.shape[1]), x[-1:]]))
        pairs.append((i, j, len(path)))
    if not chunks:
        return {}
    allpts = np.concatenate(chunks)
    key = "paths_" + hashlib.sha1(allpts.tobytes()).hexdigest()[:12]
    loss = run._cached(key, lambda: allpts)["loss"]
    out, start = {}, 0
    for (i, j, nv), c in zip(pairs, chunks):
        seg = loss[start:start + len(c)]
        start += len(c)
        out[f"{i}-{j}"] = {"models": [i, j], "basins": [int(basins[i]), int(basins[j])], "path_vertices": int(nv),
                           "certified": float(seg.max()),
                           "graph_estimate": float(full.to_loss(full.barrier[basins[i], basins[j]]))}
    return out


def descend(run: LandscapeRun, x0: np.ndarray, iters: int = 150, h_frac: float = 2e-3) -> dict:
    """Gradient descent in the subspace from ``x0``, with central-difference gradients and a line search over
    8 step sizes per iteration (all points of an iteration in one batch). Uses no topology method: it tells
    whether a minimum of the graph is a basin of its own or slides into another (e.g. a model's)."""
    h = h_frac * run.box.size
    d = len(x0)
    x = np.asarray(x0, float)
    lx = float(run.evaluator(x[None])["loss"][0])
    steps = np.geomspace(1e-4, 0.3, 8)
    for _ in range(iters):
        probes = np.concatenate([x + np.diag(h), x - np.diag(h)])
        lp = run.evaluator(probes)["loss"]
        g = (lp[:d] - lp[d:]) / (2 * h)                     # per-axis derivative
        gn = g * run.box.size                               # in unit-box coordinates
        if not np.linalg.norm(gn) > 0:
            break
        cand = x - steps[:, None] * (gn / np.linalg.norm(gn)) * run.box.size
        lc = run.evaluator(cand)["loss"]
        if lc.min() >= lx:
            break
        x, lx = cand[int(np.argmin(lc))], float(lc.min())
    return {"x": x, "loss": lx}


def extra_minima(run: LandscapeRun, full: FullView, basins: np.ndarray, tau: float) -> list:
    """Minima of the full view that contain no model: descend from each and report where it ends."""
    out = []
    model_unit = run.box.normalise(run.anchors()["x"])
    for b in range(full.num_minima):
        if b in set(basins.tolist()):
            continue
        v = full.eg.peaks[full.keep[b]]
        start_loss = float(full.to_loss(full.f[v]))
        end = descend(run, full.x[v])
        eu = run.box.normalise(end["x"][None])[0]
        dist_models = np.linalg.norm(model_unit - eu, axis=1)
        end_basin = int(full.basin(eu[None], tau)[0])
        out.append({"minimum": b, "start_loss": start_loss, "end_loss": end["loss"],
                    "moved": float(np.linalg.norm(eu - full.unit[v])),
                    "nearest_model": int(np.argmin(dist_models)), "distance_to_nearest_model": float(dist_models.min()),
                    "end_basin": end_basin, "stays_in_own_basin": end_basin == b})
    return out


def robustness(run: LandscapeRun, kind: str, n_full: int, tau: float, ks=(15, 30, 50)) -> list:
    """Minima of the full kNN view for several k (grid views have no k)."""
    if kind == "grid":
        return []
    rows = []
    for k in ks:
        fv = FullView.build(run, kind, tau, n_full, k)
        b = fv.basin(run.box.normalise(run.anchors()["x"]), tau)
        pers = np.sort(fv.eg.peak_persistence[fv.keep])[::-1]
        rows.append({"k": k, "minima": fv.num_minima, "model_basins_distinct": int(len(set(b.tolist()))),
                     "weakest_persistence": float(pers[-1])})
    return rows


def run_study(run: LandscapeRun, kind: str, n_full: int, k: int, n_slice: int, tau: float, radius: float,
              li: bool) -> dict:
    full = FullView.build(run, kind, tau, n_full, k)
    model_basins = full.use_models(run, tau)
    checks = full_checks(run, full, tau)
    certified = certified_paths(run, full, model_basins)
    extras = extra_minima(run, full, model_basins, tau)
    rows = []
    for plane in triple_planes(run.subspace.anchors):
        s = run.plane(plane, n_slice)
        rows.append(analyse_slice(plane.name, plane.models, s["u"], s["x"], s["loss"], n_slice, full, run, tau, radius))
    if li:
        for m in range(len(run.meta["models"])):
            s = run.li_plane(m, n_slice)
            rows.append(analyse_slice(f"li_model{m}", (m,), s["u"], None, s["loss"], n_slice, full, run, tau, radius))
    # best barrier any slice gives for each pair of full minima
    best: Dict[tuple, float] = {}
    for r in rows:
        for p in r.get("pairs") or []:
            key = tuple(sorted(p["basins"]))
            best[key] = min(best.get(key, np.inf), p["excess"])
    m = full.num_minima
    full_links = {tuple(sorted(map(int, full.eg.simplify(tau).saddle_peaks[s])))
                  for s in range(full.eg.simplify(tau).num_saddles)}
    simple = full.eg.simplify(tau)
    return {"run": run.name, "dim": run.box.dim, "full": {
                "kind": full.kind, "minima": m, "saddles": int(simple.num_saddles),
                "loops": int((~simple.saddle_is_merge).sum()), "links": sorted(full_links),
                "minimum_losses": [float(v) for v in full.to_loss(simple.peak_values)],
                "barriers": [[float(full.to_loss(full.barrier[i, j])) if i != j else None for j in range(m)]
                             for i in range(m)]},
            "checks": checks, "slices": rows, "certified": certified, "extra_minima": extras,
            "robustness": robustness(run, kind, n_full, tau),
            "best_slice_excess": {f"{a}-{b}": v for (a, b), v in sorted(best.items())},
            "pairs_never_in_a_slice": int(m * (m - 1) // 2 - len(best)),
            "settings": {"n_full": n_full, "k": k, "n_slice": n_slice, "tau": tau, "radius": radius}}


def _f(v, p=1, pct=True) -> str:
    if v is None:
        return "–"
    return f"{100 * v:+.{p}f}%" if pct else f"{v:.{p}f}"


def report(res: dict, path: "Path") -> None:
    full, checks, st = res["full"], res["checks"], res["settings"]
    models = len(checks["model_basins"])
    L = [f"# Loss landscape, 2-D slices vs the full {res['dim']}-D subspace ({res['run']})", "",
         f"Full view: our extremum graph on the {full['kind']}; τ = {st['tau']} (f = −log L scaled to the full "
         f"view's range). Slices: {st['n_slice']}×{st['n_slice']} Freudenthal grids. A slice minimum is *true* when it "
         f"lies within {st['radius']} (box scaled to the unit cube) of the minimum of the full basin it falls in, "
         "*displaced* otherwise.", "",
         f"## Full view: {full['minima']} minima, {full['saddles']} saddles ({full['loops']} close loops)", "",
         f"Minimum losses: " + ", ".join(f"{v:.3f}" for v in full["minimum_losses"]) + ".", "",
         f"Method-free checks: the {models} models lie in **{checks['distinct']} distinct basins**; full barrier ≤ "
         f"straight-line barrier for {sum(1 for p in checks['pairs'] if p['holds'])} of "
         f"{sum(1 for p in checks['pairs'] if p['holds'] is not None)} pairs"
         + ("" if all(p["holds"] in (True, None) for p in checks["pairs"]) else
            " (**violations:** " + ", ".join(f"{p['pair']} {p['full']:.4f} > {p['straight']:.4f}"
                                             for p in checks["pairs"] if p["holds"] is False) + ")") + ".", "",
         "Barriers between full minima (loss):", "",
         "| | " + " | ".join(f"#{j}" for j in range(full["minima"])) + " |",
         "|---" * (full["minima"] + 1) + "|"]
    for i, row in enumerate(full["barriers"]):
        L.append(f"| #{i} | " + " | ".join("–" if v is None else f"{v:.3f}" for v in row) + " |")
    L += ["", "## Slices", "",
          "| slice | models in it | minima shown | true | displaced | split basins | full minima missed | loops | barrier excess (median / max) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in res["slices"]:
        L.append(f"| {r['slice']} | {', '.join(map(str, r['models']))} | {r['minima']} | {_f(r['true'], 0, False)} | "
                 f"{_f(r['displaced'], 0, False)} | {_f(r['split_basins'], 0, False)} | {_f(r['missed'], 0, False)} | "
                 f"{r['loops']} | {_f(r.get('barrier_excess_median'))} / {_f(r.get('barrier_excess_max'))} |")
    tri = [r for r in res["slices"] if r["slice"].startswith("models")]
    li = [r for r in res["slices"] if r["slice"].startswith("li_")]
    exc = [p["excess"] for r in tri for p in r["pairs"]]
    disp = [pl for r in tri for pl in r["placement"] if not pl["true"]]
    L += ["", "## Summary", "",
          f"* Planes through three models (**{len(tri)}**): each shows on average {np.mean([r['true'] for r in tri]):.2f} of the "
          f"{full['minima']} full minima (at most {max(r['true'] for r in tri)}); {sum(r['displaced'] for r in tri)} displaced minima "
          f"in total" + (f" (loss {min(d['loss_excess'] for d in disp):+.1%} to {max(d['loss_excess'] for d in disp):+.1%} "
                         "above the true minimum of their basin)" if disp else "") + "; "
          f"{sum(r['split_basins'] for r in tri)} split basins.",
          f"* Barriers seen in those planes vs the full graph's own estimate: median {_f(float(np.median(exc)) if exc else None)}, "
          f"max {_f(max(exc) if exc else None)} (over {len(exc)} slice pairs). The best plane for each pair of full minima: "
          + ", ".join(f"{k} {_f(v)}" for k, v in res["best_slice_excess"].items())
          + f"; {res['pairs_never_in_a_slice']} of {full['minima'] * (full['minima'] - 1) // 2} pairs never appear together "
          "in any plane. "
          + (f"The loss measured densely along the graph's own path is {100 * float(np.median([c['certified'] / c['graph_estimate'] - 1 for c in res['certified'].values()])):.1f}% "
             "(median) above the graph's estimate, so the method-free comparison is the path table below."
             if res.get("certified") else ""),
          f"* Full graph: {full['saddles']} saddles with {full['loops']} independent loops; the planes show "
          f"{sum(r['loops'] for r in tri)} loop saddles in total."]
    if li:
        L.append(f"* Li et al. random planes around each model (±1 filter-normalised units, {len(li)}): minima shown "
                 + ", ".join(str(r["minima"]) for r in li) + "; loss "
                 + ", ".join(f"{r['loss_min']:.2f}..{r['loss_max']:.1f}" for r in li) + ".")
    if res.get("extra_minima"):
        L += ["", "## Minima of the full view that no model sits in", "",
              "Checked by gradient descent in the subspace (finite differences, no topology method) from the minimum's sample.", "",
              "| minimum | loss at the sample | loss after descent | moved (unit box) | nearest model (distance) | stays in its own basin |",
              "|---|---|---|---|---|---|"]
        for e in res["extra_minima"]:
            L.append(f"| #{e['minimum']} | {e['start_loss']:.4f} | {e['end_loss']:.4f} | {e['moved']:.3f} | "
                     f"{e['nearest_model']} ({e['distance_to_nearest_model']:.3f}) | {'yes' if e['stays_in_own_basin'] else 'no'} |")
    if res.get("robustness"):
        L += ["", "## Full view: robustness to k", "",
              "| k | minima | models in distinct basins | weakest kept persistence |", "|---|---|---|---|"]
        for r in res["robustness"]:
            L.append(f"| {r['k']} | {r['minima']} | {r['model_basins_distinct']} of {models} | {r['weakest_persistence']:.3f} |")
    if res.get("certified"):
        straight = {tuple(p["pair"]): p["straight"] for p in checks["pairs"]}
        L += ["", "## Barriers between models: slices vs a real path in the full subspace", "",
              "*Path* = the full graph's minimax path between the two basins, with the loss evaluated densely along it: "
              "a real path, so the true barrier is at most its maximum. *Best slice* = the lowest barrier any plane "
              "containing both models shows. A positive last column means every plane overstates the true barrier "
              "by at least that much, whatever method computed the full view.", "",
              "| models | straight line | full graph estimate | path (upper bound) | best slice | best slice vs path |",
              "|---|---|---|---|---|---|"]
        over = []
        for key, c in res["certified"].items():
            i, j = c["models"]
            bi, bj = sorted(c["basins"])
            sl = [pp["slice_barrier"] for r in res["slices"] if r["slice"].startswith("models")
                  and i in r["models"] and j in r["models"] for pp in r["pairs"] if sorted(pp["basins"]) == [bi, bj]]
            best_slice = min(sl) if sl else None
            rel = None if best_slice is None else best_slice / c["certified"] - 1
            if rel is not None:
                over.append(rel)
            L.append(f"| {i}–{j} | {straight[(i, j)]:.4f} | {c['graph_estimate']:.4f} | {c['certified']:.4f} | "
                     f"{'–' if best_slice is None else f'{best_slice:.4f}'} | {_f(rel)} |")
        if over:
            L += ["", f"Best slice vs path over {len(over)} model pairs: median {_f(float(np.median(over)))}, "
                      f"min {_f(min(over))}, max {_f(max(over))}; the path is lower in {sum(o > 0 for o in over)} of {len(over)}."]
    path.with_suffix(".md").write_text("\n".join(L) + "\n")
    path.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="cnn_m4_e30")
    ap.add_argument("--full", choices=["auto", "grid", "knn"], default="auto",
                    help="auto: the 49^3 grid for d = 3, kNN of Sobol samples otherwise")
    ap.add_argument("--grid", type=int, default=49)
    ap.add_argument("--n-full", type=int, default=65536, help="Sobol samples for the kNN full view")
    ap.add_argument("--k", type=int, default=None, help="kNN k of the full view (default 15 for d = 3, 30 above)")
    ap.add_argument("--n-slice", type=int, default=65)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--no-li", action="store_true")
    args = ap.parse_args()
    run = LandscapeRun.load(args.run)
    kind = args.full if args.full != "auto" else ("grid" if run.box.dim == 3 else "knn")
    args.k = args.k or (15 if run.box.dim <= 3 else 30)
    res = run_study(run, kind, args.grid if kind == "grid" else args.n_full, args.k, args.n_slice, args.tau,
                    args.radius, not args.no_li)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report(res, RESULTS / f"loss_landscape_slices_{args.run}")


if __name__ == "__main__":
    main()
