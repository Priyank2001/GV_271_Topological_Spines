"""
An affine subspace of weight space: theta(x) = origin + x @ basis, x in R^d.

``Subspace.through(models)``: the affine hull of m trained weight vectors (d = m - 1), with an
orthonormal basis (Gram-Schmidt of theta_i - centroid). Each model then sits at a known point
x_i of R^d, and every trained model is a minimum of the training objective, so the landscape
has known minima: ground truth that no topology method was involved in.

Extra directions (``with_random``) are filter-normalised random vectors (Li et al. 2018: each
conv filter / linear row of the random direction gets the norm of the matching filter of the
reference weights), orthogonalised against the basis and normalised to unit length. How far
to sample along them is the sampling box's business (``sampling.Box.around``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch


def flatten(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()])


@dataclass
class ParamLayout:
    """Names and shapes of a model's parameters, to go between a flat vector and a dict."""
    names: List[str]
    shapes: List[torch.Size]

    @classmethod
    def of(cls, state: Dict[str, torch.Tensor]) -> "ParamLayout":
        return cls(list(state), [v.shape for v in state.values()])

    @property
    def sizes(self) -> List[int]:
        return [int(np.prod(s)) for s in self.shapes]

    def unflatten(self, flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``flat`` (..., P) -> {name: (..., *shape)} (views, no copy)."""
        lead = flat.shape[:-1]
        parts = torch.split(flat, self.sizes, dim=-1)
        return {n: p.reshape(*lead, *s) for n, p, s in zip(self.names, parts, self.shapes)}


@dataclass
class Subspace:
    origin: torch.Tensor          # (P,)
    basis: torch.Tensor           # (d, P), orthonormal rows
    layout: ParamLayout
    anchors: np.ndarray           # (m, d) coordinates of the models that define it (may be empty)

    @property
    def dim(self) -> int:
        return self.basis.shape[0]

    def params(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d) coordinates -> (B, P) flat weight vectors."""
        return self.origin + x.to(self.basis) @ self.basis

    def coords(self, theta: torch.Tensor) -> torch.Tensor:
        """Projection of flat weight vectors (..., P) onto the subspace: (..., d)."""
        return (theta.to(self.basis) - self.origin) @ self.basis.T

    def to(self, device) -> "Subspace":
        return Subspace(self.origin.to(device), self.basis.to(device), self.layout, self.anchors)

    # ---- construction --------------------------------------------------------------------
    @classmethod
    def through(cls, states: Sequence[Dict[str, torch.Tensor]]) -> "Subspace":
        """Affine hull of the given models' weights (needs at least two)."""
        if len(states) < 2:
            raise ValueError("need at least two models")
        thetas = torch.stack([flatten(s).double() for s in states])          # (m, P)
        origin = thetas.mean(0)
        q, r = torch.linalg.qr((thetas - origin)[:-1].T)                     # m-1 independent differences
        if (r.diagonal().abs() < 1e-8 * r.diagonal().abs().max()).any():
            raise ValueError("the models are affinely dependent")
        basis = q.T.contiguous()                                             # (m-1, P)
        anchors = ((thetas - origin) @ basis.T).numpy()
        return cls(origin.float(), basis.float(), ParamLayout.of(states[0]), anchors)

    def with_random(self, n: int, reference: Dict[str, torch.Tensor], seed: int) -> "Subspace":
        """This subspace plus ``n`` filter-normalised random directions (see module docstring)."""
        g = torch.Generator().manual_seed(seed)
        basis = self.basis.double()
        new = []
        for _ in range(n):
            parts = []
            for v in reference.values():
                r = torch.randn(v.shape, generator=g, dtype=torch.float64)
                if v.dim() > 1:                                 # per output filter / row
                    r *= v.double().flatten(1).norm(dim=1).view(-1, *[1] * (v.dim() - 1)) / \
                        r.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *[1] * (v.dim() - 1))
                else:                                           # biases: Li et al. set them to 0
                    r.zero_()
                parts.append(r.reshape(-1))
            r = torch.cat(parts)
            for b in list(basis) + new:                        # Gram-Schmidt
                r -= (r @ b) * b
            new.append(r / r.norm())
        basis = torch.cat([basis, torch.stack(new)]) if new else basis
        anchors = np.concatenate([self.anchors, np.zeros((len(self.anchors), n))], 1)
        return Subspace(self.origin, basis.float(), self.layout, anchors)

    @classmethod
    def li_plane(cls, state: Dict[str, torch.Tensor], seed: int) -> "Subspace":
        """The usual loss-landscape picture (Li et al. 2018): theta + a d1 + b d2 around one model, with two
        filter-normalised random directions d1, d2 (each filter of d_i has the norm of the model's filter,
        biases 0). The basis is NOT orthonormal (``coords`` does not apply); (a, b) in [-1, 1]^2 is the
        customary range."""
        g = torch.Generator().manual_seed(seed)
        dirs = []
        for _ in range(2):
            parts = []
            for v in state.values():
                r = torch.randn(v.shape, generator=g, dtype=torch.float64)
                if v.dim() > 1:
                    shape = (-1,) + (1,) * (v.dim() - 1)
                    r *= (v.double().flatten(1).norm(dim=1) / r.flatten(1).norm(dim=1).clamp_min(1e-12)).view(shape)
                else:
                    r.zero_()
                parts.append(r.reshape(-1))
            dirs.append(torch.cat(parts))
        return cls(flatten(state), torch.stack(dirs).float(), ParamLayout.of(state), np.zeros((1, 2)))
