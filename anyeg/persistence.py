"""
Persistence and saturated persistence of AnyEG's extremum graph.

AnyEG itself gives every saddle only ``|f(s) - f(maximum of s's own basin)|``
(``AnyEGGraph.saddle_persistence``) and never simplifies. This module adds
the two measures a topological spine needs, computed on AnyEG's own graph
(its maxima, saddles and arcs) and nothing else:

* **persistence** of every peak and saddle (0-dimensional superlevel-set
  persistence of the graph: the elder rule);
* **saturated persistence** p* of every saddle (Correa et al. 2011, section
  3.4): the largest noise threshold at which its arcs are still needed to
  connect a drawn saddle to the peak it now leads to.

Saddles with k maxima
---------------------
An AnyEG saddle can reach three or more maxima. Persistence and Correa's
cancellation (Algorithm 1) are defined for saddles that join two peaks, so a
saddle that reaches peaks ``p0 < p1 < ... < p(k-1)`` (peak ids, smaller =
higher) becomes k - 1 *rows* ``(s, p0, pi)``, all at the height of s. The
rows of one saddle span its k peaks, so the union-find sweep joins exactly
the groups the saddle joins, whatever the order of its rows. This is the
convention of ``comparison.variants.anyeg_to_extremum_graph`` and of
``pipeline/anyeg_saddles.py``.

Which arcs (``links=``)
-----------------------
* ``"all"`` (default): AnyEG's arcs as they are. AnyEG also joins a saddle
  to the basins of its *lower* neighbours, and such a basin's maximum need
  not be connected to the saddle above the saddle's height. Those false
  links merge groups too early, so a peak can die at a saddle above its true
  one: every peak's persistence is <= its true value.
* ``"uphill"``: only the arcs that leave the saddle through a higher
  neighbour (``AnyEGGraph.uphill_only``). Every such link is real (saddle
  and maximum are connected above the saddle), and every merge saddle of the
  merge tree passes AnyEG's upper-link test, so the peak persistence is
  exactly the merge tree's (``tests/test_anyeg_persistence.py``).

Persistence of a row
--------------------
Rows are visited from the highest saddle down. A row that joins two groups
is a *merge* row: the group whose top peak is lower dies, with persistence
``f(top) - f(s)``. A row whose two peaks are already in one group closes a
loop; its persistence is ``min(f(a), f(b)) - f(s)``. The cancellation order,
the parents (``peak_parent``: the peak across the cancelling saddle, as in
Algorithm 1, not the elder rule's) and p* follow
``pipeline/extremum_graph.py`` step 6 exactly. The code is repeated here
because this package does not import ``pipeline/``; a test checks the two
give the same numbers.

Usage
-----
    graph = AnyEGPipeline().run(field)
    pers = compute_persistence(graph, heights=field.heights)    # or field.values
    pers.peak_persistence          # per peak, highest first
    pers.arc_mask(0.1)             # rows still drawn at noise threshold 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .graph import AnyEGGraph

LINKS = ("all", "uphill")


@dataclass
class AnyEGPersistence:
    """Persistence of an ``AnyEGGraph``. Peak ids: position in ``peaks`` (0 = highest)."""

    links: str                     # which arcs were used: "all" or "uphill"
    peaks: np.ndarray              # (P,) maximum vertex ids, highest first
    peak_values: np.ndarray        # (P,) heights
    peak_persistence: np.ndarray   # (P,) inf for the top peak of every connected piece
    peak_parent: np.ndarray        # (P,) peak id it is merged into (Algorithm 1), -1 if never
    saddles: np.ndarray            # (R,) saddle vertex of each row, highest first
    saddle_values: np.ndarray      # (R,) heights
    saddle_peaks: np.ndarray       # (R, 2) [highest peak the saddle reaches, other peak]
    saddle_is_merge: np.ndarray    # (R,) False if the row closes a loop
    saddle_persistence: np.ndarray  # (R,) see the module docstring
    saddle_saturation: np.ndarray  # (R,) saturated persistence p* >= persistence

    @property
    def num_peaks(self) -> int:
        return self.peaks.shape[0]

    @property
    def num_rows(self) -> int:
        return self.saddles.shape[0]

    def peak_id(self, num_points: int) -> np.ndarray:
        """(N,) peak id of every maximum vertex, -1 elsewhere."""
        out = np.full(num_points, -1, dtype=np.int64)
        out[self.peaks] = np.arange(self.num_peaks)
        return out

    def surviving_peaks(self, threshold: float) -> np.ndarray:
        """(P,) bool: peaks not merged away at noise threshold ``threshold``."""
        return self.peak_persistence >= threshold

    def arc_mask(self, threshold: float) -> np.ndarray:
        """(R,) bool: rows whose arcs are still drawn at ``threshold`` (p* >= threshold)."""
        return self.saddle_saturation >= threshold

    def __repr__(self) -> str:
        return (f"AnyEGPersistence(links={self.links!r}, {self.num_peaks:,} peaks, "
                f"{self.num_rows:,} rows, {int(self.saddle_is_merge.sum()):,} merge)")


def compute_persistence(graph: AnyEGGraph, heights: np.ndarray,
                        order: Optional[np.ndarray] = None,
                        links: str = "all") -> AnyEGPersistence:
    """Persistence and saturated persistence of ``graph``.

    ``heights``: the field values (persistence is measured in them).
    ``order``: the total order of the vertices (larger = higher), e.g. the
    ranks AnyEG was run on; it sorts peaks and saddles and decides which of
    two equal heights is higher. Default: the (height, index) rank.
    ``links``: ``"all"`` or ``"uphill"``, see the module docstring.
    """
    if links not in LINKS:
        raise ValueError(f"links must be one of {LINKS}, got {links!r}")
    h = np.asarray(heights, dtype=np.float64).reshape(-1)
    if h.shape[0] != graph.num_points:
        raise ValueError(f"{h.shape[0]} heights for a graph on {graph.num_points} points")
    order = _rank(h) if order is None else np.asarray(order).reshape(-1)
    if order.shape != h.shape:
        raise ValueError("order and heights have different lengths")
    if links == "uphill":
        graph = graph.uphill_only(order)

    peaks = graph.maxima[np.argsort(-order[graph.maxima], kind="stable")]
    pid = np.full(graph.num_points, -1, dtype=np.int64)
    pid[peaks] = np.arange(peaks.shape[0])
    sad, sp = _rows(graph.arc_saddle, pid[graph.arc_maximum], order)

    peak_vals, sad_vals = h[peaks], h[sad]
    sweep = _sweep(peak_vals, sad_vals, sp)
    sat = _saturation(peak_vals.shape[0], sp, sweep["saddle"], sweep["is_merge"])
    return AnyEGPersistence(
        links=links, peaks=peaks, peak_values=peak_vals,
        peak_persistence=sweep["peak"], peak_parent=sat["parent"],
        saddles=sad, saddle_values=sad_vals, saddle_peaks=sp,
        saddle_is_merge=sweep["is_merge"], saddle_persistence=sweep["saddle"],
        saddle_saturation=sat["saturation"],
    )


# ---------------------------------------------------------------------------
def _rank(h: np.ndarray) -> np.ndarray:
    rank = np.empty(h.shape[0], dtype=np.int64)
    rank[np.lexsort((np.arange(h.shape[0]), h))] = np.arange(h.shape[0])
    return rank


def _rows(arc_saddle: np.ndarray, arc_peak: np.ndarray, order: np.ndarray):
    """k - 1 rows ``(s, highest peak, other)`` per saddle reaching k peaks, highest saddle first."""
    if arc_saddle.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    pairs = np.unique(np.stack([arc_saddle, arc_peak], axis=1).astype(np.int64), axis=0)
    s, p = pairs[:, 0], pairs[:, 1]
    idx = np.lexsort((p, -order[s]))               # highest saddle first, its peaks highest first
    s, p = s[idx], p[idx]
    first = np.r_[True, s[1:] != s[:-1]]
    top = p[np.maximum.accumulate(np.where(first, np.arange(s.shape[0]), 0))]
    other = ~first
    return s[other], np.stack([top[other], p[other]], axis=1)


def _sweep(peak_vals: np.ndarray, sad_vals: np.ndarray, sad_peaks: np.ndarray) -> dict:
    """Union-find over the rows from the highest saddle down (the elder rule)."""
    num_peaks, num_rows = peak_vals.shape[0], sad_peaks.shape[0]
    parent = list(range(num_peaks))
    top = list(range(num_peaks))                   # top[root] = highest peak id of the group

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    peak_pers = np.full(num_peaks, np.inf)
    is_merge = np.zeros(num_rows, dtype=bool)
    sad_pers = np.zeros(num_rows)
    for r in range(num_rows):
        a, b = int(sad_peaks[r, 0]), int(sad_peaks[r, 1])
        ra, rb = find(a), find(b)
        if ra != rb:
            keep, dead = (ra, rb) if top[ra] < top[rb] else (rb, ra)
            peak_pers[top[dead]] = peak_vals[top[dead]] - sad_vals[r]
            sad_pers[r] = peak_pers[top[dead]]
            is_merge[r] = True
            parent[dead] = keep
        else:
            sad_pers[r] = min(peak_vals[a], peak_vals[b]) - sad_vals[r]
    return {"peak": peak_pers, "is_merge": is_merge, "saddle": sad_pers}


def _saturation(num_peaks: int, sad_peaks: np.ndarray, sad_pers: np.ndarray,
                is_merge: np.ndarray) -> dict:
    """Correa et al.'s cancellations, then p* of every row.

    Upwards: merge rows are cancelled in order of increasing persistence;
    each peak points along ``peak -> saddle -> peak ...`` to the uncancelled
    peak its flow ends at, and cancelling a row kills the lower of its two
    terminals and reverses the path to it (Algorithm 1, ``REVERSE``).
    Downwards: undo the cancellations from the most persistent one, and
    every row's two peaks become ports whose paths to their surviving peak
    are marked "needed up to this persistence". See ``pipeline/extremum_graph.py``
    (``_cancellations``, ``_saturation``) for why this replaces marking once.
    """
    num_rows = sad_pers.shape[0]
    out_saddle = np.full(num_peaks, -1, dtype=np.int64)
    out_peak = np.full(num_peaks, -1, dtype=np.int64)
    parent = np.full(num_peaks, -1, dtype=np.int64)
    order = np.lexsort((np.arange(num_rows), sad_pers))

    def path(u: int):
        nodes, rows = [u], []
        while out_saddle[u] >= 0:
            rows.append(int(out_saddle[u]))
            u = int(out_peak[u])
            nodes.append(u)
        return nodes, rows

    undo = {}
    for r in order:
        if not is_merge[r]:
            continue
        a, b = int(sad_peaks[r, 0]), int(sad_peaks[r, 1])
        nodes_a, rows_a = path(a)
        nodes_b, rows_b = path(b)
        ta, tb = nodes_a[-1], nodes_b[-1]
        if ta == tb:                               # merge rows form a forest; be safe
            continue
        if ta > tb:                                # the lower terminal dies
            nodes, rows, other, parent[ta] = nodes_a, rows_a, b, tb
        else:
            nodes, rows, other, parent[tb] = nodes_b, rows_b, a, ta
        for i in range(len(nodes) - 1, 0, -1):
            out_saddle[nodes[i]] = rows[i - 1]
            out_peak[nodes[i]] = nodes[i - 1]
        out_saddle[nodes[0]] = r
        out_peak[nodes[0]] = other
        undo[int(r)] = (nodes, rows)

    saturation = sad_pers.astype(np.float64).copy()
    marked = np.zeros(num_rows, dtype=bool)

    def mark_to_root(u: int, p: float, stop_at_marked: bool) -> None:
        while out_saddle[u] >= 0:
            e = int(out_saddle[u])
            if marked[e]:
                if stop_at_marked:
                    return
            else:
                marked[e] = True
                saturation[e] = max(saturation[e], p)
            u = int(out_peak[u])

    for r in order[::-1]:
        p = float(sad_pers[r])
        a, b = int(sad_peaks[r, 0]), int(sad_peaks[r, 1])
        if int(r) in undo:
            nodes, rows = undo[int(r)]
            for i in range(len(nodes) - 1):
                out_saddle[nodes[i]] = rows[i]
                out_peak[nodes[i]] = nodes[i + 1]
            out_saddle[nodes[-1]] = out_peak[nodes[-1]] = -1
            near = nodes[0]
            mark_to_root(near, p, stop_at_marked=False)
            mark_to_root(b if near == a else a, p, stop_at_marked=True)
        else:
            mark_to_root(a, p, stop_at_marked=True)
            mark_to_root(b, p, stop_at_marked=True)
    return {"parent": parent, "saturation": saturation}
