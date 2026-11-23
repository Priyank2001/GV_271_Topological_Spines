"""The network whose loss landscape is studied: a small CNN without normalisation layers.

No BatchNorm on purpose: with BatchNorm the loss at theta(x) would also depend on running
statistics that are not part of the weight vector, so a point of the subspace would not fix
the function. Three conv blocks + two linear layers, 0.19 M parameters, ~16 MFLOP per image,
so the loss at hundreds of thousands of weight vectors can be evaluated in minutes.
"""
from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, width: int = 32, num_classes: int = 10):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),            # 16x16
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 8x8
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 4x4
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * w * 16, 4 * w), nn.ReLU(),
                                  nn.Linear(4 * w, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build(arch: dict) -> nn.Module:
    """Model from its saved description, e.g. ``{"name": "SmallCNN", "width": 32}``."""
    arch = dict(arch)
    name = arch.pop("name")
    if name != "SmallCNN":
        raise ValueError(f"unknown architecture {name!r}")
    return SmallCNN(**arch)
