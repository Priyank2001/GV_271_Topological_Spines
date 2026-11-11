"""
The GPU part of AnyEG: four "modules" that find maxima, basins and saddles.

AnyEG's ``compute()`` (cudaHelper.cu) runs these steps, each a CUDA kernel:

1. **Maxima and steepest ascent** (``judgeExtremePoints_New``). A vertex is a
   maximum if no neighbour is *strictly* higher. Otherwise it points to the
   neighbour with the largest rise ``f(n) - f(v)`` (float32), the first one in
   list order on a tie. These one-step pointers are ``first_link``.
2. **Pointer doubling** (``updateFa``). ``link = link[link]`` until every
   vertex points at the maximum its ascent ends at: ``final_link`` (the
   vertex's basin, or region).
3. **Candidate saddles** (``judgeCandidateSaddlePoints_New``). A vertex with a
   strictly higher neighbour that lies in a different basin.
4. **Upper-link test** (``judgeCandidateSaddlePoints_ByMorse``). Take the
   candidate's strictly higher neighbours and join two of them when they are
   neighbours of each other. The candidate is a saddle if this "upper link"
   has two or more connected pieces.

Note what AnyEG does *not* do: values are not made unique (equal neighbours
are neither higher nor lower, so a flat top gives several maxima), and there
is no persistence or simplification at this stage.

``ExtremumGraphKernels`` is the interface; ``TorchKernels`` (a PyTorch port)
and ``OriginalCudaKernels`` (the unchanged AnyEG code) implement it, and both
return a ``KernelOutput``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..neighborhood import NeighborTable


@dataclass
class KernelOutput:
    extrema: np.ndarray          # (P,) maxima, ascending vertex id
    saddles: np.ndarray          # (S,) candidates that pass the upper-link test, ascending
    first_link: np.ndarray       # (N,) steepest-ascent neighbour (itself for maxima)
    final_link: np.ndarray       # (N,) the maximum each vertex ascends to
    num_candidates: int          # candidates after module 3 (before the upper-link test)
    seconds: Dict[str, float] = field(default_factory=dict)   # per-module time

    def same_as(self, other: "KernelOutput") -> bool:
        """Exactly the same maxima, saddles and links."""
        return (self.num_candidates == other.num_candidates
                and all(np.array_equal(getattr(self, f), getattr(other, f))
                        for f in ("extrema", "saddles", "first_link", "final_link")))

    def differences(self, other: "KernelOutput") -> Dict[str, int]:
        """Number of differing entries per field (0 everywhere when ``same_as``)."""
        out = {}
        for f in ("extrema", "saddles"):
            a, b = set(getattr(self, f).tolist()), set(getattr(other, f).tolist())
            out[f] = len(a ^ b)
        for f in ("first_link", "final_link"):
            out[f] = int((getattr(self, f) != getattr(other, f)).sum())
        out["num_candidates"] = abs(self.num_candidates - other.num_candidates)
        return out


class ExtremumGraphKernels(ABC):
    """Runs AnyEG's modules 1-4 on float32 values and a neighbour table."""

    name: str = "abstract"

    @abstractmethod
    def run(self, values: np.ndarray, neighbors: NeighborTable) -> KernelOutput:
        ...
