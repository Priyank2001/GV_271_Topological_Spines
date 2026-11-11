"""
2D layout of the extremum graph (the "spine" of Correa et al., 2011).

Peaks and saddles are placed in the plane so that their distances on screen
roughly match their distances in the data space. Two steps:

1. *PCA start.* Project the node positions (peaks and saddles, in the
   original d-dimensional space) onto their first two principal axes. This
   is the best flat map in the least-squares sense, but it can squash nodes
   that differ mainly along other axes.
2. *Stress refinement.* Minimise the weighted stress

       stress(X) = sum_{i<j} w_ij (|x_i - x_j| - d_ij)^2,   w_ij = 1 / d_ij^2

   with SMACOF (stress majorisation, Gansner et al. 2004), starting from the
   PCA map. ``d_ij`` is the distance between nodes i and j in the original
   space. The weights make errors on close pairs count more than errors on
   distant pairs, so neighbouring peaks stay readable. Each SMACOF step
   cannot increase the stress, so the result is never worse than the PCA
   start.

Nodes are numbered peaks first, then saddles: node ``i < P`` is peak ``i``
and node ``P + s`` is saddle ``s``. The links of the drawn graph are the
``(peak, P + saddle)`` pairs of ``ExtremumGraph.edges``.

Limits
------
* The distance matrix has one row per node, so the cost grows with the
  square of the node count. ``spine_layout`` refuses more than
  ``MAX_NODES`` nodes; simplify the graph first.
* Straight-line distances in a high-dimensional space are a crude measure
  (distances concentrate as d grows). The layout is only as meaningful as
  that.
* Nothing here avoids crossing links; ``count_crossings`` measures them.

Usage
-----
    from pipeline.spine_layout import spine_layout

    layout = spine_layout(eg, coords=ds.points)    # coords: (N, d) or vertices -> (n, d)
    layout.peak_xy, layout.saddle_xy               # (P, 2), (S, 2), fitted to [-1, 1]
    layout.stress, layout.pca_stress               # normalised, lower is better
    layout.crossings                               # number of crossing link pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .extremum_graph import ExtremumGraph

MAX_NODES = 3000
_CHUNK_PAIRS = 4_000_000     # link pairs tested at once by count_crossings

Coords = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass
class SpineLayout:
    peak_xy: np.ndarray       # (P, 2) fitted into [-1, 1] x [-1, 1]
    saddle_xy: np.ndarray     # (S, 2)
    scale: float              # data-space distance per unit of layout distance
    stress: float             # normalised stress of the final layout, see ``stress``
    pca_stress: float         # same for the PCA start
    crossings: int            # pairs of links that cross
    iterations: int

    def to_dict(self, decimals: int = 5) -> dict:
        """JSON-ready form (the ``spine`` entry of an export)."""
        return {"method": "pca+stress",
                "peaks": np.round(self.peak_xy, decimals).tolist(),
                "saddles": np.round(self.saddle_xy, decimals).tolist(),
                "scale": float(self.scale),
                "stress": float(self.stress),
                "pca_stress": float(self.pca_stress),
                "crossings": int(self.crossings),
                "iterations": int(self.iterations)}

    def __repr__(self) -> str:
        return (f"SpineLayout({len(self.peak_xy)} peaks, {len(self.saddle_xy)} saddles, "
                f"stress {self.stress:.4f} (PCA {self.pca_stress:.4f}), "
                f"{self.crossings} crossings)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def spine_layout(eg: ExtremumGraph, coords: Coords, *,
                 spacing: Optional[np.ndarray] = None,
                 max_iter: int = 300, tol: float = 1e-6) -> SpineLayout:
    """Lay out the peaks and saddles of ``eg`` in 2D.

    coords:  ``(N, d)`` positions of all points, or a function mapping vertex
             indices to ``(n, d)`` positions (use it for large grids).
    spacing: per-axis scale applied to the positions first (voxel size of a
             volume, so distances are physical).
    """
    vertices = np.concatenate([eg.peaks, eg.saddles])
    n = len(vertices)
    if n > MAX_NODES:
        raise ValueError(f"{n} nodes is more than MAX_NODES={MAX_NODES}; simplify the graph first")

    X = coords(vertices) if callable(coords) else np.asarray(coords)[vertices]
    X = np.asarray(X, dtype=np.float64)
    if spacing is not None:
        X = X * np.asarray(spacing, dtype=np.float64)

    D = _pairwise_distances(X)
    start = pca_layout(X)
    pca_stress = stress(start, D)
    xy, iterations = _smacof(start, D, max_iter=max_iter, tol=tol)
    final_stress = stress(xy, D)
    if final_stress > pca_stress:        # cannot happen in exact arithmetic
        xy, final_stress = start, pca_stress

    xy, scale = _fit_to_unit_box(xy)
    p = eg.num_peaks
    return SpineLayout(
        peak_xy=xy[:p], saddle_xy=xy[p:], scale=scale,
        stress=final_stress, pca_stress=pca_stress,
        crossings=count_crossings(xy, _links(eg)), iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Pieces (public so they can be tested and reused)
# ---------------------------------------------------------------------------
def pca_layout(X: np.ndarray) -> np.ndarray:
    """First two principal coordinates of ``X`` ``(n, d)`` as an ``(n, 2)`` array.

    Columns that carry no variance (fewer than two axes, or all points equal)
    are zero.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out = np.zeros((n, 2))
    if n < 2:
        return out
    centred = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    m = min(2, len(s))
    out[:, :m] = centred @ vt[:m].T
    return out


def stress(xy: np.ndarray, D: np.ndarray) -> float:
    """Normalised weighted stress, ``sqrt(sum w (|xi-xj| - dij)^2 / sum w dij^2)``,
    with ``w = 1/d^2``. 0 is a perfect map; a map that puts every node at one
    point scores 1."""
    n = len(xy)
    if n < 2:
        return 0.0
    W, Dc = _weights(D)
    low = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    num = (W * (low - Dc) ** 2).sum()
    den = (W * Dc ** 2).sum()
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def count_crossings(xy: np.ndarray, links: np.ndarray) -> int:
    """Number of pairs of straight links ``(a, b)`` that properly cross.

    ``links`` is ``(L, 2)`` node indices. Links that share a node never count,
    and neither do touching or overlapping collinear links.
    """
    links = np.asarray(links, dtype=np.int64).reshape(-1, 2)
    num = len(links)
    if num < 2:
        return 0
    a, b = xy[links[:, 0]], xy[links[:, 1]]
    total = 0
    step = max(1, _CHUNK_PAIRS // num)       # rows per chunk, bounds memory
    for lo in range(0, num, step):
        hi = min(lo + step, num)
        p, q = a[lo:hi, None, :], b[lo:hi, None, :]
        r, s = a[None, :, :], b[None, :, :]
        o1, o2 = _orient(p, q, r), _orient(p, q, s)
        o3, o4 = _orient(r, s, p), _orient(r, s, q)
        cross = (o1 * o2 < 0) & (o3 * o4 < 0)
        shared = ((links[lo:hi, None, 0] == links[None, :, 0]) |
                  (links[lo:hi, None, 0] == links[None, :, 1]) |
                  (links[lo:hi, None, 1] == links[None, :, 0]) |
                  (links[lo:hi, None, 1] == links[None, :, 1]))
        total += int((cross & ~shared).sum())
    return total // 2                        # every pair was counted in both orders


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _links(eg: ExtremumGraph) -> np.ndarray:
    """(2S, 2) node pairs ``(peak, P + saddle)``, one per link of the drawn graph."""
    e = eg.edges
    return np.stack([e[:, 0], eg.num_peaks + e[:, 1]], axis=1)


def _orient(a, b, c) -> np.ndarray:
    """Sign-carrying area of the triangle (a, b, c); broadcasts over leading axes."""
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
           (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    D = np.sqrt(np.maximum(d2, 0.0))
    np.fill_diagonal(D, 0.0)
    return D


def _weights(D: np.ndarray):
    """Weights 1/d^2 and the distances with zeros lifted off the floor.

    Two nodes at the same place would get an infinite weight. Their distance
    is raised to a small fraction of the typical distance instead.
    """
    n = len(D)
    off = ~np.eye(n, dtype=bool)
    typical = np.median(D[off & (D > 0)]) if (D[off] > 0).any() else 1.0
    Dc = np.where(off, np.maximum(D, 1e-3 * typical), 0.0)
    W = np.zeros_like(Dc)
    W[off] = 1.0 / Dc[off] ** 2
    return W, Dc


def _smacof(start: np.ndarray, D: np.ndarray, max_iter: int, tol: float):
    """Weighted SMACOF from ``start``. Returns (layout, iterations used)."""
    n = len(start)
    if n < 3:
        # One node is a point; two nodes can be placed exactly on a line.
        if n == 2:
            d = D[0, 1]
            return np.array([[-d / 2, 0.0], [d / 2, 0.0]]), 0
        return start.copy(), 0

    W, Dc = _weights(D)
    L = np.diag(W.sum(axis=1)) - W                       # weighted Laplacian
    L_pinv = np.linalg.pinv(L)                           # fixed across iterations
    X = start.copy()
    if np.allclose(X, X[0]):                             # all nodes equal: break the tie
        X = np.random.default_rng(0).normal(scale=1e-3, size=X.shape) * Dc.max()

    prev = _raw_stress(X, W, Dc)
    for it in range(1, max_iter + 1):
        low = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            B = np.where(low > 1e-12, -W * Dc / low, 0.0)
        np.fill_diagonal(B, 0.0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = L_pinv @ (B @ X)
        cur = _raw_stress(X, W, Dc)
        if prev - cur <= tol * max(prev, 1e-30):
            return X, it
        prev = cur
    return X, max_iter


def _raw_stress(X, W, Dc) -> float:
    low = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return float((W * (low - Dc) ** 2).sum() / 2.0)


def _fit_to_unit_box(xy: np.ndarray):
    """Centre ``xy`` and scale it uniformly so its larger extent is [-1, 1].

    Returns the new layout and the scale (data distance per layout unit).
    """
    centred = xy - (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    half = float(np.abs(centred).max())
    if half <= 0.0:
        return centred, 1.0
    return centred / half, half
