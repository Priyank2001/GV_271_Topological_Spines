"""
Training objective at many points of a subspace at once.

    L(x) = mean cross-entropy of theta(x) on a fixed set of training images + (l2 / 2) ||theta(x)||^2

The L2 term is the weight decay the models were trained with (SGD's ``weight_decay=l2`` adds
``l2 * theta`` to the gradient), so a trained model is a stationary point of exactly this L.
Without it, the cross-entropy alone would keep decreasing past the trained weights.

Many weight vectors are evaluated in one call with ``torch.func.vmap`` over
``functional_call``: a batch of ``param_batch`` networks sees each chunk of images together.
Plain fp32 (TF32 off), so grid and scattered samples are computed with identical arithmetic.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap

from .cifar import Split
from .subspace import Subspace


class LossEvaluator:
    def __init__(self, model: torch.nn.Module, data: Split, subspace: Subspace, l2: float,
                 device: str = "cuda", param_batch: int = 64, image_batch: int = 500):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = device
        self.model = model.to(device).eval()
        self.data = data.to(device)
        self.sub = subspace.to(device)
        self.l2 = l2
        self.param_batch, self.image_batch = param_batch, image_batch
        layout = self.sub.layout

        def logits(flat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.model, layout.unflatten(flat), (x,))

        self._logits = vmap(logits, in_dims=(0, None))

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> dict:
        """(B, d) subspace coordinates -> {"loss": (B,), "ce": (B,), "acc": (B,)} as float64 arrays."""
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        ce, acc, sq = [], [], []
        n = len(self.data)
        for i in range(0, len(x), self.param_batch):
            theta = self.sub.params(x[i:i + self.param_batch])                  # (b, P)
            tot = torch.zeros(len(theta), device=self.device, dtype=torch.float64)
            right = torch.zeros_like(tot)
            for j in range(0, n, self.image_batch):
                out = self._logits(theta, self.data.x[j:j + self.image_batch])  # (b, m, C)
                y = self.data.y[j:j + self.image_batch].expand(len(theta), -1)
                tot += F.cross_entropy(out.transpose(1, 2), y, reduction="none").sum(1).double()
                right += (out.argmax(-1) == y).sum(1).double()
            ce.append(tot / n)
            acc.append(right / n)
            sq.append((theta.double() ** 2).sum(1))
        ce, acc, sq = (torch.cat(v).cpu().numpy() for v in (ce, acc, sq))
        return {"loss": ce + 0.5 * self.l2 * sq, "ce": ce, "acc": acc}

    def chunked(self, x: np.ndarray, chunk: int = 4096, progress: bool = True) -> dict:
        """Same as calling, in chunks of ``chunk`` points with a progress line."""
        parts = []
        for i in range(0, len(x), chunk):
            parts.append(self(x[i:i + chunk]))
            if progress:
                print(f"\r  evaluated {min(i + chunk, len(x)):,} / {len(x):,}", end="", flush=True)
        if progress:
            print()
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
