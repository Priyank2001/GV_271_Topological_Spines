"""CIFAR-10 read straight from the python pickles (``cifar-10-batches-py``), as float tensors."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DEFAULT_ROOT = Path(os.environ.get("CIFAR10_ROOT", "/home/priyank1/aditya/datasets/vision/cifar-10-batches-py"))
# per-channel mean / std of the CIFAR-10 training set
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@dataclass
class Split:
    x: torch.Tensor   # (N, 3, 32, 32) float32, normalised
    y: torch.Tensor   # (N,) int64

    def __len__(self) -> int:
        return len(self.y)

    def to(self, device) -> "Split":
        return Split(self.x.to(device), self.y.to(device))

    def subset(self, n: int, seed: int) -> "Split":
        """``n`` images drawn without replacement, the same ones for the same seed."""
        idx = np.random.default_rng(seed).choice(len(self), n, replace=False)
        idx = torch.as_tensor(np.sort(idx))
        return Split(self.x[idx], self.y[idx])


def _read(files, root: Path) -> Split:
    xs, ys = [], []
    for f in files:
        with open(root / f, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        xs.append(np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32))
        ys.append(np.asarray(d[b"labels"], np.int64))
    x = torch.from_numpy(np.concatenate(xs)).float().div_(255)
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    return Split((x - mean) / std, torch.from_numpy(np.concatenate(ys)))


def load(root: Path = DEFAULT_ROOT) -> tuple[Split, Split]:
    """(train 50,000, test 10,000)."""
    root = Path(root)
    if not (root / "data_batch_1").exists():
        raise FileNotFoundError(f"CIFAR-10 pickles not found in {root} (set CIFAR10_ROOT)")
    return _read([f"data_batch_{i}" for i in range(1, 6)], root), _read(["test_batch"], root)


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random horizontal flip and 4-pixel padded random crop, on the tensor's device."""
    n = x.shape[0]
    flip = torch.rand(n, generator=generator, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    pad = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")       # (n, 3, 40, 40)
    dy = torch.randint(0, 9, (n,), generator=generator, device=x.device)
    dx = torch.randint(0, 9, (n,), generator=generator, device=x.device)
    return crop(pad, dy, dx)


def crop(pad: torch.Tensor, dy: torch.Tensor, dx: torch.Tensor, size: int = 32) -> torch.Tensor:
    """``out[i] = pad[i, :, dy[i]:dy[i]+size, dx[i]:dx[i]+size]`` for every image, without a loop."""
    n, ch, _, w = pad.shape
    ar = torch.arange(size, device=pad.device)
    rows = (dy.view(-1, 1) + ar).view(n, 1, size, 1).expand(n, ch, size, w)
    out = pad.gather(2, rows)                                            # (n, ch, size, w)
    cols = (dx.view(-1, 1) + ar).view(n, 1, 1, size).expand(n, ch, size, size)
    return out.gather(3, cols)                                           # (n, ch, size, size)
