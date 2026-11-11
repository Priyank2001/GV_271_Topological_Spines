"""
AnyEG's four GPU modules written with PyTorch (see ``base.py`` for the algorithm).

Each CUDA kernel of AnyEG handles one vertex per thread. Here every module is
a few whole-array tensor operations over the padded neighbour table, done in
chunks of vertices so the memory use stays bounded. The arithmetic is the
same as in the CUDA code (float32 values, float32 differences, strict
comparisons, first neighbour wins a tie), so the output is identical to the
original's; ``tests/test_anyeg.py`` and ``python3 -m comparison`` check this.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from ..neighborhood import NeighborTable
from .base import ExtremumGraphKernels, KernelOutput


class TorchKernels(ExtremumGraphKernels):
    """PyTorch port of AnyEG's modules 1-4, on a GPU or the CPU."""

    name = "torch"

    def __init__(self, device: Optional[str] = None, chunk_entries: int = 1 << 25):
        """``chunk_entries`` bounds the size of the per-chunk temporaries
        (vertices x neighbours, or candidates x neighbours^3 in module 4)."""
        import torch

        self.torch = torch
        self.device = torch.device(device or _most_free_cuda_device())
        self.chunk_entries = chunk_entries

    def __repr__(self) -> str:
        return f"TorchKernels(device='{self.device}')"

    # ------------------------------------------------------------------
    def run(self, values: np.ndarray, neighbors: NeighborTable) -> KernelOutput:
        torch = self.torch
        vals = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float32), device=self.device)
        nbr = torch.as_tensor(neighbors.table, device=self.device)        # int32, -1 = none
        seconds = {}

        with torch.no_grad():
            t = self._tic()
            first_link, is_max = self._module1_steepest_ascent(vals, nbr)
            seconds["module1_extrema"] = self._toc(t)

            t = self._tic()
            final_link = self._module2_pointer_doubling(first_link)
            seconds["module2_pointer_doubling"] = self._toc(t)

            t = self._tic()
            candidates = self._module3_candidates(vals, nbr, final_link)
            seconds["module3_candidates"] = self._toc(t)

            t = self._tic()
            saddles = self._module4_upper_link_test(vals, nbr, candidates)
            seconds["module4_upper_link"] = self._toc(t)

        return KernelOutput(
            extrema=_np(torch.nonzero(is_max).flatten()),
            saddles=_np(saddles),
            first_link=_np(first_link),
            final_link=_np(final_link),
            num_candidates=int(candidates.shape[0]),
            seconds=seconds,
        )

    # -- module 1 ----------------------------------------------------------
    def _module1_steepest_ascent(self, vals, nbr):
        """Each vertex's steepest-ascent neighbour (itself for maxima)."""
        torch = self.torch
        n, m = nbr.shape
        first_link = torch.arange(n, device=self.device, dtype=torch.int64)
        is_max = torch.zeros(n, dtype=torch.bool, device=self.device)
        for lo, hi in self._chunks(n, m):
            ids = nbr[lo:hi].long()
            valid = ids >= 0
            v = vals[lo:hi, None]
            nv = vals[ids.clamp(min=0)]
            higher = valid & (v < nv)
            # The rise f(n) - f(v) in float32, as the kernel computes it; -inf
            # where the neighbour is not higher. argmax returns the first
            # maximum, like the kernel's strict ">" scan.
            rise = torch.where(higher, nv - v, torch.full_like(nv, -float("inf")))
            best = rise.argmax(dim=1)
            has_up = higher.any(dim=1)
            chosen = ids.gather(1, best[:, None]).squeeze(1)
            first_link[lo:hi] = torch.where(has_up, chosen, first_link[lo:hi])
            is_max[lo:hi] = ~has_up
        return first_link, is_max

    # -- module 2 ----------------------------------------------------------
    def _module2_pointer_doubling(self, first_link):
        """Follow the pointers until each vertex points at its maximum."""
        link = first_link.clone()
        while True:
            nxt = link[link]
            if self.torch.equal(nxt, link):
                return link
            link = nxt

    # -- module 3 ----------------------------------------------------------
    def _module3_candidates(self, vals, nbr, final_link):
        """Vertices with a strictly higher neighbour in a different basin."""
        torch = self.torch
        n, m = nbr.shape
        found = []
        for lo, hi in self._chunks(n, m):
            ids = nbr[lo:hi].long()
            safe = ids.clamp(min=0)
            hit = ((ids >= 0) & (vals[lo:hi, None] < vals[safe])
                   & (final_link[lo:hi, None] != final_link[safe]))
            found.append(torch.nonzero(hit.any(dim=1)).flatten() + lo)
        return torch.cat(found) if found else torch.empty(0, dtype=torch.int64, device=self.device)

    # -- module 4 ----------------------------------------------------------
    def _module4_upper_link_test(self, vals, nbr, candidates):
        """Keep candidates whose higher neighbours form >= 2 connected pieces."""
        torch = self.torch
        m = nbr.shape[1]
        kept = []
        for lo, hi in self._chunks(candidates.shape[0], m ** 3):
            a = candidates[lo:hi]
            up = nbr[a].long()                                          # (c, m) neighbours of a
            safe = up.clamp(min=0)
            is_up = (up >= 0) & (vals[safe] > vals[a, None])            # the "upper" neighbours

            # adjacent[c, i, j]: upper neighbour j is in the neighbour list of upper neighbour i.
            lists = nbr[safe].long()                                    # (c, m, m)
            adjacent = (lists[:, :, None, :] == up[:, None, :, None]).any(dim=-1)
            adjacent &= is_up[:, :, None] & is_up[:, None, :]
            adjacent = adjacent | adjacent.transpose(1, 2)              # union-find is symmetric

            pieces = _count_components(torch, adjacent, is_up)
            kept.append(a[pieces >= 2])
        return torch.cat(kept) if kept else candidates[:0]

    # -- helpers -------------------------------------------------------------
    def _chunks(self, n: int, per_item: int) -> Iterator:
        step = max(1, self.chunk_entries // max(per_item, 1))
        for lo in range(0, n, step):
            yield lo, min(n, lo + step)

    def _tic(self) -> float:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _toc(self, t0: float) -> float:
        return self._tic() - t0


def _count_components(torch, adjacent, members):
    """Connected pieces of each small graph in a batch.

    ``adjacent`` is (c, m, m) boolean, ``members`` (c, m) says which of the m
    nodes take part. Reachability is found by squaring (A + I) until it stops
    changing (at most log2(m) times); a node is the representative of its
    piece if no smaller-numbered member reaches it.
    """
    c, m, _ = adjacent.shape
    eye = torch.eye(m, dtype=torch.bool, device=adjacent.device)
    reach = adjacent | (eye & members[:, :, None])
    for _ in range(max(1, int(np.ceil(np.log2(max(m, 2)))))):
        r = reach.float()
        reach = torch.bmm(r, r) > 0
    smaller = torch.tril(torch.ones(m, m, dtype=torch.bool, device=adjacent.device), diagonal=-1)
    reached_by_smaller = (reach & smaller[None]).any(dim=2)                 # j < i reaches i
    return (members & ~reached_by_smaller).sum(dim=1)


def _np(t) -> np.ndarray:
    return t.cpu().numpy().astype(np.int64)


def _most_free_cuda_device() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], -i) for i in range(torch.cuda.device_count())]
    return f"cuda:{-max(free)[1]}"
