"""
Step 4: are better models' loss landscapes topologically simpler?

    python3 -m landscape.family train --group stage --seed 0      # also: width, lr; seeds 0 and 1
    python3 -m landscape.family sample                            # every trained model without a landscape yet
    python3 -m landscape.family analyse                           # report + figure

Claim under test (Geniesse et al., NeurIPS 2024; Landscaper, 2026): the loss landscape of a better
model is simpler (fewer minima and saddles in its merge tree, smoother). We vary one factor at a time:

  stage  one run (width 32, lr 0.05) checkpointed after 1, 2, 4, 8, 16, 30 epochs
  width  8, 16, 32, 64 channels, 30 epochs
  lr     0.005, 0.02, 0.05, 0.15 (peak of the cosine schedule), 30 epochs

Landscape of a model (the papers' setting, not the subspace through several models of steps 1-3): its
training objective (CE on 2,000 fixed training images + weight decay) at theta + sum_i a_i d_i, a in
[-1, 1]^4, with four filter-normalised random directions d_i (Li et al. 2018: the normalisation that makes
landscapes of different models comparable; the same random draws for every model of one width).
Sampled with 16,384 Sobol points; our extremum graph on the kNN graph (k = 30, the step-3 lesson for
d >= 4). The 2-D slice a_3 = a_4 = 0 (65 x 65 Freudenthal grid) is the papers' 2-D view of the same box.

Measures, in log-loss units (f = -log L, so a persistence of 0.05 is a 5% loss ratio), all from the graph:
  minima@tau       minima with persistence >= tau (tau = 0.01, 0.03, 0.1)
  saddles@tau      saddles of the simplified graph
  total_pers       sum of the finite persistences (merge tree "size")
  basin_frac       share of the box in the basin of the minimum that contains the trained model
  escape           lowest barrier out of that basin (log-loss above its minimum), inf if no saddle in the box
  smd              mean distance (in a units) from each merge saddle to its two minima: our reading of
                   Landscaper's SMAD (Saddle-Minimum Average Distance); NaN without saddles
  loss_mean        mean loss over the box (a flatness measure that needs no topology)
Each is compared with test accuracy (Spearman rank correlation over all models, and the trend within
each factor). A k robustness check (minima@0.03 for k = 15, 30, 50) is reported per model.

Writes data/loss_landscape/family/ (models, landscapes) and comparison/results/loss_landscape_family.{md,json,png}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import WORK, cifar
from .models import SmallCNN
from .run import N_EVAL, LandscapeRun
from .sampling import Box
from .subspace import Subspace
from .train import Trainer, TrainConfig

FAMILY = WORK / "family"
DIM, N_SAMPLES, K, N_SLICE, DIR_SEED = 4, 16384, 30, 65, 0
TAUS = (0.01, 0.03, 0.1)
GROUPS = {
    "stage": [dict(width=32, lr=0.05, epochs=30, checkpoints=(1, 2, 4, 8, 16, 30))],
    "width": [dict(width=w, lr=0.05, epochs=30) for w in (8, 16, 64)],          # 32 comes from "stage"
    "lr": [dict(width=32, lr=lr, epochs=30) for lr in (0.005, 0.02, 0.15)],      # 0.05 comes from "stage"
}


def model_name(width: int, lr: float, epoch: int, seed: int) -> str:
    return f"w{width}_lr{lr:g}_e{epoch}_s{seed}"


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def train_group(group: str, seed: int, device: str) -> None:
    out = FAMILY / "models"
    out.mkdir(parents=True, exist_ok=True)
    train, test = cifar.load()
    for spec in GROUPS[group]:
        cfg = TrainConfig(epochs=spec["epochs"], lr=spec["lr"], width=spec["width"])
        trainer = Trainer(cfg, train, test, device)
        arch = {"name": "SmallCNN", "width": cfg.width}
        keep = set(spec.get("checkpoints", (spec["epochs"],)))
        t0 = time.time()

        def save(epoch: int, model: torch.nn.Module) -> None:
            if epoch not in keep:
                return
            name = model_name(cfg.width, cfg.lr, epoch, seed)
            rec = {"name": name, "file": f"{name}.pt", "group": group, "seed": seed, "width": cfg.width,
                   "lr": cfg.lr, "epoch": epoch, "epochs_planned": cfg.epochs, "arch": arch,
                   "config": {"weight_decay": cfg.weight_decay, "momentum": cfg.momentum, "batch_size": cfg.batch_size},
                   "train": trainer.evaluate(model, trainer.train), "test": trainer.evaluate(model, trainer.test),
                   "seconds": round(time.time() - t0, 1)}
            torch.save({"arch": arch, "state_dict": model.state_dict()}, out / rec["file"])
            (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(f"{name}: test acc {rec['test']['acc']:.3f}, train loss {rec['train']['loss']:.3f}", flush=True)

        torch.manual_seed(seed)
        try:
            trainer.run(SmallCNN(cfg.width), seed=seed, epochs=cfg.epochs, on_epoch=save)
        except FloatingPointError as e:
            name = model_name(cfg.width, cfg.lr, cfg.epochs, seed)
            (out / f"{name}.diverged.json").write_text(json.dumps({"name": name, "error": str(e)}))
            print(f"{name}: {e}", flush=True)


SINGLE_SEEDS = (0, 1)          # single-model landscapes
SUB_SEEDS = (0, 1, 2, 3)       # multi-solution subspaces (one per configuration)
SUB_N, SUB_K = 8192, 15        # 3-D: the setting step 2 validated


def records(seeds=SINGLE_SEEDS) -> List[dict]:
    recs = (json.loads(f.read_text()) for f in (FAMILY / "models").glob("*_s[0-9].json"))
    return sorted((r for r in recs if r["seed"] in seeds), key=lambda r: (r["width"], r["lr"], r["epoch"], r["seed"]))


def configurations() -> Dict[tuple, List[dict]]:
    """(width, lr, epoch) -> its models for every seed in SUB_SEEDS (configurations missing a seed, e.g. a
    diverged run, are left out)."""
    by: Dict[tuple, List[dict]] = {}
    for r in records(SUB_SEEDS):
        by.setdefault((r["width"], r["lr"], r["epoch"]), []).append(r)
    return {c: sorted(rs, key=lambda r: r["seed"]) for c, rs in sorted(by.items()) if len(rs) == len(SUB_SEEDS)}


def config_run(recs: List[dict], device: str | None = None) -> LandscapeRun:
    """The affine subspace through one configuration's solutions (as in steps 1-3)."""
    states = [torch.load(FAMILY / "models" / r["file"], map_location="cpu", weights_only=True)["state_dict"]
              for r in recs]
    sub = Subspace.through(states)
    r0 = recs[0]
    name = f"w{r0['width']}_lr{r0['lr']:g}_e{r0['epoch']}"
    meta = {"arch": r0["arch"], "config": r0["config"], "models": recs, "spawn_epoch": 0}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return LandscapeRun(name, FAMILY / "subspaces" / name, meta, sub, Box.around(sub.anchors, 0.3), N_EVAL, device,
                        param_batch=64 * 32 // max(r0["width"], 32))


def subspace_view(run: LandscapeRun):
    """Full view, model basins and certified paths of a configuration's subspace (evaluations cached)."""
    from .slices import FullView, certified_paths
    run.anchors()
    run.linear_paths()
    full = FullView.build(run, "knn", 1e-9, SUB_N, SUB_K)
    tau = 0.01 / (full.hi - full.lo)                       # 0.01 in log-loss units
    full.keep = np.flatnonzero(full.eg.peak_persistence >= tau)
    from .accuracy import minimax_barriers
    full.barrier = minimax_barriers(full.eg)[np.ix_(full.keep, full.keep)]
    basins = full.use_models(run, tau)
    return full, basins, certified_paths(run, full, basins)


def sample_subspaces(device: str | None = None, part: str = "0/1") -> None:
    i, n = map(int, part.split("/"))
    for c, recs in list(configurations().items())[i::n]:
        run = config_run(recs, device)
        t0 = time.time()
        subspace_view(run)
        print(f"{run.name}: subspace sampled ({time.time() - t0:.0f}s)", flush=True)


def verified_minima(run: LandscapeRun, full, iters: int = 300, merge: float = 0.02) -> dict:
    """Method-free count of the minima in the box: gradient descent (finite differences) from every graph minimum
    and from every model; end points closer than ``merge`` (unit box) are the same minimum. Early checkpoints are
    not minima themselves, so the models alone are no ground truth there."""
    from .slices import descend
    starts = [full.x[full.eg.peaks[full.keep[b]]] for b in range(full.num_minima)] + list(run.anchors()["x"])
    reps, labels = [], []
    for x0 in starts:
        e = descend(run, x0, iters=iters)
        u = run.box.normalise(e["x"][None])[0]
        for i, (ru, _) in enumerate(reps):
            if np.linalg.norm(u - ru) < merge:
                labels.append(i)
                break
        else:
            reps.append((u, e["loss"]))
            labels.append(len(reps) - 1)
    return {"count": len(reps), "losses": [float(l) for _, l in reps], "graph_minimum_to": labels[:full.num_minima],
            "model_to": labels[full.num_minima:]}


def analyse_config(recs: List[dict]) -> dict:
    run = config_run(recs)
    full, basins, cert = subspace_view(run)
    # descent check only where the graph has more minima than models (checked once on a converged control too)
    check = verified_minima(run, full) if full.num_minima > len(recs) else None
    scale = full.hi - full.lo                              # f units -> log-loss units
    pers = full.eg.peak_persistence * scale
    simple = full.eg.simplify(0.01 / scale)
    a = run.anchors()["loss"]
    lin = run.linear_paths()
    steps = len(lin["t"])
    straight, low = [], []
    for p, (i, j) in enumerate(lin["pairs"]):
        base = 0.5 * (a[i] + a[j])
        straight.append(float(lin["loss"][p * steps:(p + 1) * steps].max() - base))
        c = cert.get(f"{i}-{j}")
        low.append(0.0 if c is None else float(c["certified"] - base))   # same basin: no barrier
    r0 = recs[0]
    return {"config": run.name, "width": r0["width"], "lr": r0["lr"], "epoch": r0["epoch"],
            "test_acc": float(np.mean([r["test"]["acc"] for r in recs])),
            "minima@0.01": int((pers >= 0.01).sum()), "minima@0.03": int((pers >= 0.03).sum()),
            "descent_check": check, "verified_minima": check["count"] if check else int((pers >= 0.01).sum()),
            "minima@0.1": int((pers >= 0.1).sum()), "loops@0.01": int((~simple.saddle_is_merge).sum()),
            "model_basins": int(len(set(basins.tolist()))),
            "barrier_straight_mean": float(np.mean(straight)), "barrier_low_mean": float(np.mean(low)),
            "barrier_low_max": float(np.max(low)), "minimum_loss_mean": float(np.mean(a)),
            "path_top_mean": float(np.mean([c["certified"] for c in cert.values()])) if cert else float("nan"),
            "low_vs_straight": float(np.mean(low) / np.mean(straight)) if np.mean(straight) > 0 else float("nan")}



# ---------------------------------------------------------------------------------------------
# landscapes
# ---------------------------------------------------------------------------------------------
def landscape_of(rec: dict, device: str | None = None) -> LandscapeRun:
    state = torch.load(FAMILY / "models" / rec["file"], map_location="cpu", weights_only=True)["state_dict"]
    sub = Subspace.filter_normalised(state, DIM, DIR_SEED)
    meta = {"arch": rec["arch"], "config": rec["config"], "models": [rec], "spawn_epoch": 0}
    box = Box(-np.ones(DIM), np.ones(DIM))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return LandscapeRun(rec["name"], FAMILY / "landscapes" / rec["name"], meta, sub, box, N_EVAL, device,
                        param_batch=64 * 32 // max(rec["width"], 32))          # same activation memory per batch


def slice_points(n: int = N_SLICE) -> np.ndarray:
    from .sampling import grid
    uv = grid(Box(-np.ones(2), np.ones(2)), (n, n))
    return np.concatenate([uv, np.zeros((len(uv), DIM - 2))], 1)


def sample_all(device: str | None = None, part: str = "0/1") -> None:
    i, n = map(int, part.split("/"))
    for rec in records()[i::n]:
        run = landscape_of(rec, device)
        t0 = time.time()
        run.scattered("sobol", N_SAMPLES, 0)
        run._cached(f"slice12_{N_SLICE}", slice_points)
        print(f"{rec['name']}: sampled ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------------------------
def measures(f: np.ndarray, x: np.ndarray, graph, centre_vertex: int) -> dict:
    """Topology measures of f = -log L on ``graph`` (see module docstring)."""
    from pipeline.extremum_graph import extremum_graph
    eg = extremum_graph(f, graph)
    pers = eg.peak_persistence
    out = {f"minima@{t:g}": int((pers >= t).sum()) for t in TAUS}
    for t in TAUS:
        out[f"saddles@{t:g}"] = int(eg.simplify(t).num_saddles)
    out["total_pers"] = float(pers[np.isfinite(pers)].sum())
    s = eg.simplify(TAUS[0])
    c = int(s.region[centre_vertex])
    out["basin_frac"] = float(s.region_size[c] / len(f))
    inc = [i for i in range(s.num_saddles) if c in s.saddle_peaks[i]]
    out["escape"] = float(s.peak_values[c] - max(s.saddle_values[i] for i in inc)) if inc else float("inf")
    d = [np.linalg.norm(x[s.saddles[i]] - x[s.peaks[p]]) for i in range(s.num_saddles) if s.saddle_is_merge[i]
         for p in s.saddle_peaks[i]]
    out["smd"] = float(np.mean(d)) if d else float("nan")
    out["centre_is_global"] = bool(c == 0)
    return out


def analyse_model(rec: dict) -> dict:
    from pipeline.extremum_graph import extremum_graph
    from pipeline.neighbor_graph import freudenthal_grid_graph, knn_graph
    run = landscape_of(rec, "cpu")
    s = run.scattered("sobol", N_SAMPLES, 0)
    x, f = s["x"], -np.log(s["loss"])
    centre = int(np.argmin(np.linalg.norm(x, axis=1)))
    row = {k: rec[k] for k in ("name", "group", "seed", "width", "lr", "epoch")}
    row.update({"test_acc": rec["test"]["acc"], "test_loss": rec["test"]["loss"], "train_loss": rec["train"]["loss"],
                "loss_mean": float(s["loss"].mean()), "loss_centre": float(run.anchors()["loss"][0])})
    row["d4"] = measures(f, x, knn_graph(x, K), centre)
    row["d4_k"] = {k: int((extremum_graph(f, knn_graph(x, k)).peak_persistence >= 0.03).sum()) for k in (15, 30, 50)}
    sl = run._cached(f"slice12_{N_SLICE}", slice_points)
    xs, fs = sl["x"], -np.log(sl["loss"])
    row["d2"] = measures(fs, xs[:, :2], freudenthal_grid_graph((N_SLICE, N_SLICE)), int(np.argmin(np.linalg.norm(xs, axis=1))))
    row["d2"]["loss_mean"] = float(sl["loss"].mean())
    return row


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


METRICS = ["minima@0.01", "minima@0.03", "minima@0.1", "saddles@0.03", "total_pers", "basin_frac", "escape", "smd"]


def report(rows: List[dict], diverged: List[str], out: Path, configs: List[dict] | None = None) -> None:
    acc = [r["test_acc"] for r in rows]
    L = ["# Loss landscape topology vs model quality (step 4)", "",
         "Two settings. **A. Single model** (the papers' setting): a box of filter-normalised random directions "
         "around one model. **B. Several solutions** of one configuration (4 seeds): the 3-D subspace through them, "
         "as in steps 1–3, where basins, saddles and barriers between solutions exist.", "",
         "## A. Single-model landscapes", "",
         f"{len(rows)} SmallCNNs on CIFAR-10 (two seeds per setting). Landscape per model: training objective at "
         f"θ + Σ aᵢdᵢ, a ∈ [−1, 1]^{DIM}, four filter-normalised random directions; {N_SAMPLES:,} Sobol samples, our "
         f"extremum graph on the kNN graph (k = {K}); the 2-D column is the slice a₃ = a₄ = 0 on a {N_SLICE}² PL grid. "
         "Persistence in log-loss units (0.03 = a 3% loss ratio). Minima of the loss are the peaks of f = −log L.", ""]
    if diverged:
        L += [f"Diverged during training (no landscape): {', '.join(diverged)}.", ""]
    L += ["## Spearman correlation with test accuracy (all models)", "",
          "| measure | 4-D landscape | 2-D slice | simpler for better models would mean |", "|---|---|---|---|"]
    expect = {"minima@0.01": "negative", "minima@0.03": "negative", "minima@0.1": "negative", "saddles@0.03": "negative",
              "total_pers": "negative", "basin_frac": "positive", "escape": "positive (deeper own basin)",
              "smd": "per Landscaper: larger = smoother"}
    corr = {}
    for m in METRICS:
        c4 = spearman([r["d4"][m] for r in rows], acc)
        c2 = spearman([r["d2"][m] for r in rows], acc)
        corr[m] = (c4, c2)
        cell = lambda c: "constant" if np.isnan(c) else f"{c:+.2f}"
        L.append(f"| {m} | {cell(c4)} | {cell(c2)} | {expect[m]} |")
    c4 = spearman([r["loss_mean"] for r in rows], acc)
    c2 = spearman([r["d2"]["loss_mean"] for r in rows], acc)
    corr["loss_mean"] = (c4, c2)
    L.append(f"| loss_mean (no topology) | {c4:+.2f} | {c2:+.2f} | negative (flatter) |")
    L += ["", f"With {len(rows)} models, |ρ| ≥ {0.40 if len(rows) >= 24 else 0.45:.2f} is roughly the 5% significance level "
          "(two-sided, no correction for testing several measures).", ""]
    for group, key, label in (("stage", "epoch", "epochs"), ("width", "width", "width"), ("lr", "lr", "learning rate")):
        if group == "stage":
            sel = [r for r in rows if r["width"] == 32 and r["lr"] == 0.05]
        elif group == "width":
            sel = [r for r in rows if r["lr"] == 0.05 and r["epoch"] == 30]
        else:
            sel = [r for r in rows if r["width"] == 32 and r["epoch"] == 30]
        L += [f"## Factor: {label} (mean over seeds)", "",
              f"| {label} | test acc | minima@0.01 (4-D / 2-D) | minima@0.03 | minima@0.1 | saddles@0.03 | total pers. | own basin share | escape | SMD | mean loss | minima@0.03 for k = 15/30/50 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted({r[key] for r in sel}):
            rs = [r for r in sel if r[key] == v]

            def mm(d, m, p=1):
                a = np.array([r[d][m] for r in rs], float)
                return "∞" if np.isinf(a).all() else f"{np.nanmean(a[np.isfinite(a)]):.{p}f}"
            kk = " / ".join(f"{np.mean([r['d4_k'][k] for r in rs]):.1f}" for k in (15, 30, 50))
            L.append(f"| {v:g} | {np.mean([r['test_acc'] for r in rs]):.3f} | {mm('d4', 'minima@0.01')} / {mm('d2', 'minima@0.01')} | "
                     f"{mm('d4', 'minima@0.03')} / {mm('d2', 'minima@0.03')} | {mm('d4', 'minima@0.1')} / {mm('d2', 'minima@0.1')} | "
                     f"{mm('d4', 'saddles@0.03')} / {mm('d2', 'saddles@0.03')} | {mm('d4', 'total_pers', 2)} / {mm('d2', 'total_pers', 2)} | "
                     f"{mm('d4', 'basin_frac', 2)} / {mm('d2', 'basin_frac', 2)} | {mm('d4', 'escape', 2)} / {mm('d2', 'escape', 2)} | "
                     f"{mm('d4', 'smd', 2)} / {mm('d2', 'smd', 2)} | {np.mean([r['loss_mean'] for r in rs]):.1f} | {kk} |")
            corr.setdefault("_groups", {})
        L.append("")
    ccorr = {}
    if configs:
        cacc = [c["test_acc"] for c in configs]
        L += ["## B. Subspace through four solutions of each configuration", "",
              f"For each (width, lr, epochs) with all four seeds: the 3-D affine subspace through the four trained models, "
              f"{SUB_N:,} Sobol samples, our extremum graph on the kNN graph (k = {SUB_K}). Barrier = highest loss on the "
              "path minus the mean loss of its two models (the usual definition), averaged over the 6 pairs: along the "
              "straight segment (linear mode connectivity), and along the full graph's lowest path, evaluated densely "
              "(a real path, so an upper bound on the true barrier). Persistence in log-loss units.", "",
              "| configuration | test acc | models in distinct basins | graph minima@0.01 | real minima (descent) | loops@0.01 | model loss | barrier, straight | barrier, lowest path | top of lowest path (loss) | lowest / straight |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(configs, key=lambda c: (c["width"] != 32 or c["lr"] != 0.05, c["width"], c["lr"], c["epoch"])):
            L.append(f"| {c['config']} | {c['test_acc']:.3f} | {c['model_basins']} of 4 | {c['minima@0.01']} | {c['verified_minima']} | "
                     f"{c['loops@0.01']} | {c['minimum_loss_mean']:.3f} | {c['barrier_straight_mean']:.3f} | {c['barrier_low_mean']:.3f} | "
                     f"{c['path_top_mean']:.3f} | {c['low_vs_straight']:.2f} |")
        L += ["", f"Spearman correlation with test accuracy over the {len(configs)} configurations "
                  f"(|ρ| ≥ {0.62 if len(configs) >= 10 else 0.7:.2f} ≈ 5% level):", ""]
        for m in ("minima@0.01", "verified_minima", "loops@0.01", "minimum_loss_mean", "barrier_straight_mean",
                  "barrier_low_mean", "path_top_mean", "low_vs_straight"):
            ccorr[m] = spearman([c[m] for c in configs], cacc)
            L.append(f"* {m}: " + ("constant (no correlation to compute)" if np.isnan(ccorr[m]) else f"ρ = {ccorr[m]:+.2f}"))
        L.append("")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "diverged": diverged, "spearman": corr,
                                                    "configs": configs, "config_spearman": ccorr}, indent=1, default=float))
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--group", choices=list(GROUPS), required=True)
    t.add_argument("--seed", type=int, default=0)
    sm = sub.add_parser("sample")
    sm.add_argument("--part", default="0/1", help="i/n: only every n-th model starting at i (split over GPUs)")
    ss = sub.add_parser("sample-subspaces")
    ss.add_argument("--part", default="0/1")
    sub.add_parser("analyse")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "train":
        train_group(args.group, args.seed, device)
    elif args.cmd == "sample":
        sample_all(device, args.part)
    elif args.cmd == "sample-subspaces":
        sample_subspaces(device, args.part)
    else:
        from .accuracy import RESULTS
        rows = [analyse_model(r) for r in records()]
        configs = [analyse_config(rs) for rs in configurations().values()]
        diverged = sorted(json.loads(f.read_text())["name"] for f in (FAMILY / "models").glob("*.diverged.json"))
        RESULTS.mkdir(parents=True, exist_ok=True)
        report(rows, diverged, RESULTS / "loss_landscape_family", configs)


if __name__ == "__main__":
    main()
