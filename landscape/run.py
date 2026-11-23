"""
``LandscapeRun``: one set of trained models with its subspace, box, evaluator and sample cache.

Evaluating the loss is the only expensive step, so every sample set is cached once in
``data/loss_landscape/<run>/samples/<key>.npz`` (coordinates, loss, ce, acc); the dataset writer
and the accuracy study read the cache.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from . import WORK, cifar
from .evaluator import LossEvaluator
from .models import build
from .sampling import SAMPLERS, Box, grid
from .subspace import Subspace

N_EVAL, EVAL_SEED, MARGIN = 2000, 0, 0.3


@dataclass
class LandscapeRun:
    name: str
    path: Path
    meta: dict
    subspace: Subspace
    box: Box
    n_eval: int
    device: str
    _evaluator: Optional[LossEvaluator] = None

    @classmethod
    def load(cls, name: str, n_eval: int = N_EVAL, margin: float = MARGIN,
             device: Optional[str] = None) -> "LandscapeRun":
        path = WORK / name
        meta = json.loads((path / "models.json").read_text())
        states = [torch.load(path / m["file"], map_location="cpu", weights_only=True)["state_dict"] for m in meta["models"]]
        sub = Subspace.through(states)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(name, path, meta, sub, Box.around(sub.anchors, margin), n_eval, device)

    @property
    def l2(self) -> float:
        return float(self.meta["config"]["weight_decay"])

    @property
    def evaluator(self) -> LossEvaluator:
        if self._evaluator is None:
            train, _ = cifar.load()
            data = train.subset(self.n_eval, EVAL_SEED)
            self._evaluator = LossEvaluator(build(self.meta["arch"]), data, self.subspace, self.l2, self.device)
        return self._evaluator

    # ---- samples ---------------------------------------------------------------------------
    def _cached(self, key: str, make_points, evaluator: Optional[LossEvaluator] = None) -> dict:
        f = self.path / "samples" / f"{key}_eval{self.n_eval}.npz"
        if f.exists():
            z = np.load(f)
            return {k: z[k] for k in z.files}
        x = make_points()
        print(f"{self.name}: evaluating {len(x):,} points ({key})")
        out = {"x": x, **(evaluator or self.evaluator).chunked(x)}
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **out)
        return out

    def grid(self, n: int, finer: Optional[int] = None) -> Tuple[dict, Tuple[int, ...]]:
        """Grid with ``n`` vertices per axis. If ``finer`` is given and its grid contains this one
        ((finer - 1) divisible by (n - 1)), it is cut out of the finer grid's samples (no new evaluations)."""
        shape = (n,) * self.box.dim
        if finer and finer != n and (finer - 1) % (n - 1) == 0:
            fine, fshape = self.grid(finer)
            step = (finer - 1) // (n - 1)
            idx = np.arange(int(np.prod(fshape))).reshape(fshape)[(slice(None, None, step),) * self.box.dim]
            return {k: v[idx.reshape(-1)] for k, v in fine.items()}, shape
        return self._cached(f"grid{n}", lambda: grid(self.box, shape)), shape

    def scattered(self, sampler: str, n: int, seed: int) -> dict:
        return self._cached(f"{sampler}{n}_s{seed}", lambda: SAMPLERS[sampler](self.box, n, seed))

    def plane(self, plane, n: int) -> dict:
        """Loss on an n x n grid of a 2-D plane of the subspace (``slices.Plane``); ``u`` = plane coordinates."""
        out = self._cached(f"plane_{plane.name}_{n}", lambda: plane.to_subspace(grid(plane.box, (n, n))))
        out["u"] = grid(plane.box, (n, n))
        return out

    def states(self) -> list:
        return [torch.load(self.path / m["file"], map_location="cpu", weights_only=True)["state_dict"]
                for m in self.meta["models"]]

    def li_plane(self, model: int, n: int, seed: int = 0, extent: float = 1.0) -> dict:
        """Loss on an n x n grid of Li et al.'s random-direction plane around ``model`` ((a, b) in
        [-extent, extent]^2). These points are mostly outside the subspace."""
        sub = Subspace.li_plane(self.states()[model], seed=seed + 100 * model)
        box = Box(np.array([-extent, -extent]), np.array([extent, extent]))
        out = self._cached(f"li_m{model}_s{seed}_e{extent:g}_{n}", lambda: grid(box, (n, n)),
                           evaluator=self.evaluator.with_subspace(sub))
        out["u"] = out["x"]
        return out

    def anchors(self) -> dict:
        """The trained models themselves (the known minima)."""
        return self._cached("anchors", lambda: self.subspace.anchors.copy())

    def linear_paths(self, steps: int = 101) -> dict:
        """Loss along the straight segment between every two models (an upper bound on the barrier)."""
        m = len(self.subspace.anchors)
        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        t = np.linspace(0, 1, steps)

        def pts():
            a = self.subspace.anchors
            return np.concatenate([(1 - t)[:, None] * a[i] + t[:, None] * a[j] for i, j in pairs])

        out = self._cached(f"linear{steps}", pts)
        out["pairs"], out["t"] = np.array(pairs), t
        return out
