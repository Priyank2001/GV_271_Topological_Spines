"""
Write a loss landscape as a pipeline dataset (generator/datasets/<name>/).

    python3 -m landscape.make_dataset --run cnn_m4_e30 --sampler sobol --n 16384
    python3 web/build_data.py <name>          # after adding the entry to web/build_data.py

Points: subspace coordinates x (d = models - 1). f = -log L(x), scaled to 0..1, so the minima of
the training objective L are the maxima of f (a monotone change: the same critical points and
basins as L, persistence measured in log-loss). Ground truth (``extrema.maxima``): the trained
models' coordinates, with their loss. ``scalars`` adds test-free extras for the viewer: the
accuracy on the evaluation images. Raw ``loss.npy`` / ``ce.npy`` / ``acc.npy`` are saved too.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import ROOT
from .run import LandscapeRun


def f_from_loss(loss: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """-log loss scaled with the given range of -log loss to 0..1."""
    return (-np.log(loss) - lo) / (hi - lo)


def write(run: LandscapeRun, samples: dict, name: str, description: str, k: int = 15) -> dict:
    out = ROOT / "generator" / "datasets" / name
    (out / "scalars").mkdir(parents=True, exist_ok=True)
    x, loss = samples["x"].astype(np.float32), samples["loss"]
    g = -np.log(loss)
    lo, hi = float(g.min()), float(g.max())
    np.save(out / "points.npy", x)
    np.save(out / "values.npy", f_from_loss(loss, lo, hi))
    for key in ("loss", "ce", "acc"):
        np.save(out / f"{key}.npy", samples[key])
    np.save(out / "scalars" / "acc.npy", samples["acc"])
    np.save(out / "scalars" / "neg_log_loss.npy", g)
    anchors = run.anchors()
    models = run.meta["models"]
    d = x.shape[1]
    # known labels for the viewer: the trained model each sample is closest to (box scaled to the unit cube).
    # Purely geometric, no topology, so it can be compared with the peak regions.
    unit, unit_models = run.box.normalise(x), run.box.normalise(anchors["x"])
    nearest = np.argmin(((unit[:, None, :] - unit_models[None]) ** 2).sum(-1), axis=1)
    truth = {
        "name": name, "dimensions": d, "num_points": int(len(x)), "mode": "loss_landscape",
        "has_extrema_labels": True,
        "description": description,
        "formula": "f = -log L(x), scaled to 0..1; L = CE on the evaluation images + (wd/2)||theta||^2",
        "neg_log_loss_range": [lo, hi],
        "run": run.name, "box": run.box.to_dict(), "n_eval": run.n_eval, "weight_decay": run.l2,
        "dimension_names": [f"u{i + 1}" for i in range(d)],
        "extrema": {"maxima": [
            {"center": a.tolist(), "model": m["file"], "loss": float(l), "test_acc": m["test"]["acc"]}
            for a, l, m in zip(anchors["x"], anchors["loss"], models)]},
        "scalars": {
            "functions": [
                {"id": "neg_log_loss", "label": "-log training loss", "file": "scalars/neg_log_loss.npy",
                 "summary": "The main function, unscaled."},
                {"id": "acc", "label": "accuracy on the evaluation images", "file": "scalars/acc.npy",
                 "summary": f"Fraction of the {run.n_eval} evaluation images classified right."},
            ],
            "smooths": [0], "ks": [8, 30] if k == 15 else [15, 50],      # compare-mode k around the entry's k
            "main": {"id": "neg_log_loss", "smooth": 0},
        },
        "point_classes": [{"name": "Nearest trained model", "names": [f"model {i}" for i in range(len(models))],
                           "values": nearest.astype(int).tolist()}],
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {out}: N={len(x)}, d={d}, loss {loss.min():.3f}..{loss.max():.3f}")
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="cnn_m4_e30")
    ap.add_argument("--sampler", default="sobol", choices=["sobol", "uniform"])
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=None, help="kNN k the viewer entry uses (default 15 for d = 3, 30 above)")
    args = ap.parse_args()
    run = LandscapeRun.load(args.run)
    s = run.scattered(args.sampler, args.n, args.seed)
    name = args.name or f"loss_{args.run}"
    desc = (f"Training loss of {len(run.meta['models'])} CIFAR-10 CNNs ({args.run}) in the {run.box.dim}-D affine "
            f"subspace through their weights; {args.n:,} {args.sampler} samples (seed {args.seed}); "
            f"loss on {run.n_eval} fixed training images + weight decay.")
    write(run, s, name, desc, k=args.k or (15 if run.box.dim <= 3 else 30))


if __name__ == "__main__":
    main()
