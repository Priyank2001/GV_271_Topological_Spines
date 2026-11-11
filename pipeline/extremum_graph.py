"""
Extremum graph of a scalar field sampled on a neighbour graph.

The extremum graph has the field's **maxima** as nodes and its **saddles** as
the links between neighbouring maxima. It is computed on any
``NeighborGraph`` (kNN graph of a point cloud, or a regular grid), so the
dimension of the data does not matter.

Algorithm (three vectorised steps plus one small sweep)
-------------------------------------------------------
1. *Order.* Points get a total order by (value, index), so equal values never
   make two points incomparable.
2. *Maxima and regions.* Every point points to its highest neighbour, if that
   neighbour is higher than the point itself. Following these pointers from
   any point ends at a **maximum** (a point with no higher neighbour). The
   maximum a point ends at is its **region** (its steepest-ascent basin).
3. *Saddles.* Two regions are adjacent if some graph edge joins them. Among
   all edges between regions A and B, the edge whose *lower* endpoint is
   highest is the pass between them: the highest level at which A and B are
   still directly linked. Its lower endpoint is the **saddle** of (A, B).
   Each saddle is linked to the two maxima it separates.
4. *Persistence.* Visit the saddles from highest to lowest and merge regions
   with union-find. A saddle that joins two different groups of peaks is a
   ``merge`` saddle: the lower of the two groups' top peaks dies there, with
   persistence ``f(peak) - f(saddle)``. A saddle that joins peaks already in
   the same group closes a loop: the merge tree has no entry for it, so its
   prominence is ``min(f(a), f(b)) - f(saddle)`` instead.
5. *Loop filter.* Why not only the merge-tree sweep? It gives a *tree*: N
   peaks need only N-1 merge saddles, but a ring of 6 peaks has 6 saddles
   and the sixth closes the loop. Step 3 finds all of them, and also some
   false passes: where several basins meet at the bottom of a bowl (around a
   minimum), neighbouring regions touch there even if their peaks are not
   neighbours. A loop-closing saddle is therefore kept only if its *relief*
   is high (see ``_relief``): a real saddle joins valleys that run down to
   low values, while a false pass sits inside a single bowl.
6. *Cancellation and saturated persistence* (``_cancellations``,
   ``_saturation``). The merge saddles are replayed in order of persistence
   as in the paper's Algorithm 1: a cancelled peak's region joins the peak
   across its saddle (``peak_parent``), and the paths the flow takes through
   cancelled peaks give each saddle its saturated persistence p*. Simplified
   saddles with p* above the threshold keep their arcs (``arc_mask``), so a
   link still runs along the ridge through the peaks it absorbed.

Known limits:

* At most one saddle is reported per pair of peaks (the highest pass). Two
  true saddles between the same two peaks show up as one link.
* With a small k (8 or less) a few false passes survive, mostly in sparse
  regions far from the data.
* Saddles are data points, so their position is only as precise as the
  sampling.

Usage
-----
    from pipeline.data_loader import load_dataset
    from pipeline.neighbor_graph import knn_graph
    from pipeline.extremum_graph import extremum_graph

    ds = load_dataset("cyclic_2d")
    g = knn_graph(ds.points, k=10)
    eg = extremum_graph(ds.values, g)
    eg.peaks, eg.saddles                 # vertex indices
    eg.saddle_peaks                      # (S, 2) peak ids joined by each saddle
    eg.simplify(0.05)                    # drop peaks with persistence < 0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .neighbor_graph import NeighborGraph


@dataclass
class ExtremumGraph:
    """Maxima, saddles and the regions of every sample point.

    "Peak id" means an index into ``peaks`` (0 is the global maximum);
    "vertex" means an index into the original points.
    """

    num_points: int
    values: np.ndarray            # (N,) the field values, kept for convenience

    # ---- maxima ----
    peaks: np.ndarray             # (P,) vertex of each maximum, highest first
    peak_values: np.ndarray       # (P,)
    peak_persistence: np.ndarray  # (P,) f(peak) - f(saddle that absorbs it); inf for roots
    peak_parent: np.ndarray       # (P,) peak id that absorbs this one, -1 for roots: the peak
                                  #      across the saddle that cancels it (paper's Algorithm 1)

    # ---- saddles ----
    saddles: np.ndarray           # (S,) vertex of each saddle, highest first
    saddle_values: np.ndarray     # (S,)
    saddle_peaks: np.ndarray      # (S, 2) peak ids joined by the saddle, [higher, lower]
    saddle_is_merge: np.ndarray   # (S,) bool, False if the saddle closes a loop
    saddle_persistence: np.ndarray  # (S,) see module docstring
    saddle_support: np.ndarray    # (S,) number of graph edges between the two regions
    saddle_relief: np.ndarray     # (S,) in [0, 1], how deep the valley below the saddle goes
    saddle_saturation: np.ndarray  # (S,) saturated persistence p* >= persistence, see _saturation

    # ---- regions ----
    region: np.ndarray            # (N,) peak id of every point
    region_size: np.ndarray       # (P,) number of points per region

    # ---- convenience ----
    @property
    def num_peaks(self) -> int:
        return self.peaks.shape[0]

    @property
    def num_saddles(self) -> int:
        return self.saddles.shape[0]

    @property
    def edges(self) -> np.ndarray:
        """(2S, 2) rows ``(peak id, saddle id)``: the links of the graph."""
        s = np.arange(self.num_saddles)
        return np.concatenate([
            np.stack([self.saddle_peaks[:, 0], s], axis=1),
            np.stack([self.saddle_peaks[:, 1], s], axis=1),
        ])

    def simplify(self, threshold: float, variation: Optional[float] = None) -> "ExtremumGraph":
        """Remove peaks whose persistence is below ``threshold``.

        A removed peak is merged into the peak that absorbs it (``peak_parent``),
        repeatedly if that peak is removed too. Its region joins the parent's
        region. A saddle is removed when both of its peaks end up as the same
        peak, or when its own persistence is below ``threshold``; the
        remaining saddles are re-attached to the surviving peaks.

        ``variation`` is the paper's second threshold, p_hi (Correa et al. 2011,
        section 3.4): a saddle whose persistence is *above* it is removed, so
        that deep, valley-like passes do not join peaks that are only loosely
        related. Unlike ``threshold`` it never removes peaks, only links, so
        the graph may fall apart into several components. ``None`` keeps every
        saddle. The saddle kept for a pair of peaks is the highest one that
        passes both thresholds.

        ``saddle_saturation`` is recomputed for the simplified graph, so that
        ``arc_mask`` stays exact for it (see ``_saturation``).
        """
        survives = self.peak_persistence >= threshold
        calm = (np.ones(self.num_saddles, bool) if variation is None
                else self.saddle_persistence <= variation)
        if survives.all() and (self.saddle_persistence >= threshold).all() and calm.all():
            return self                  # nothing to remove

        # Resolve each peak to the nearest surviving ancestor. Parents have
        # higher values, so processing peaks in id order sees parents first.
        root = np.arange(self.num_peaks)
        for p in range(self.num_peaks):
            if not survives[p]:
                root[p] = root[self.peak_parent[p]]

        new_id = np.cumsum(survives) - 1                  # old surviving id -> new id
        peak_map = new_id[root]                           # old peak id -> new peak id

        # Saddle ends in new ids; sorted so column 0 is the higher peak again
        # (a peak can be absorbed by one with a larger id than its partner's).
        sp = np.sort(peak_map[self.saddle_peaks], axis=1)
        keep = (sp[:, 0] != sp[:, 1]) & (self.saddle_persistence >= threshold) & calm

        # Two kept saddles can now join the same pair of surviving peaks.
        # Keep the highest one of each pair; saddles are already sorted high to low.
        lo, hi = sp.min(1), sp.max(1)
        pair = lo * max(int(survives.sum()), 1) + hi
        seen: set[int] = set()
        for i in np.flatnonzero(keep):
            if int(pair[i]) in seen:
                keep[i] = False
            else:
                seen.add(int(pair[i]))

        num_kept = int(survives.sum())
        size = np.bincount(peak_map, weights=self.region_size,
                           minlength=num_kept)[:num_kept].astype(np.int64)
        new_parent = np.where(self.peak_parent[survives] >= 0,
                              peak_map[np.maximum(self.peak_parent[survives], 0)], -1)
        return ExtremumGraph(
            num_points=self.num_points,
            values=self.values,
            peaks=self.peaks[survives],
            peak_values=self.peak_values[survives],
            peak_persistence=self.peak_persistence[survives],
            peak_parent=new_parent,
            saddles=self.saddles[keep],
            saddle_values=self.saddle_values[keep],
            saddle_peaks=sp[keep],
            saddle_is_merge=self.saddle_is_merge[keep],
            saddle_persistence=self.saddle_persistence[keep],
            saddle_support=self.saddle_support[keep],
            saddle_relief=self.saddle_relief[keep],
            saddle_saturation=_saturation(self.peak_values[survives], self.saddle_values[keep],
                                          sp[keep], self.saddle_persistence[keep],
                                          self.saddle_is_merge[keep]),
            region=peak_map[self.region],
            region_size=size,
        )

    def arc_mask(self, threshold: float, variation: Optional[float] = None) -> np.ndarray:
        """Saddles whose two arcs are drawn at ``[threshold, variation]``.

        The paper's equation (3): the arcs of saddle s are kept when
        ``persistence(s) <= variation`` and ``saturation(s) >= threshold``.
        This includes every saddle that ``simplify`` keeps, plus *simplified*
        saddles whose arcs are still needed to connect a kept saddle to the
        surviving peaks (the ridge through peaks that were merged away, e.g.
        the rim of the paper's Fig. 5b). The arcs go to the saddle's own two
        peaks (``saddle_peaks``), which may be peaks that ``threshold`` hides.
        Indices are into this graph's saddles.
        """
        mask = self.saddle_saturation >= threshold
        if variation is not None:
            mask &= self.saddle_persistence <= variation
        return mask

    def __repr__(self) -> str:
        n_cycle = int((~self.saddle_is_merge).sum())
        return (f"ExtremumGraph({self.num_peaks} peaks, {self.num_saddles} saddles "
                f"({n_cycle} close a loop), N={self.num_points})")


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def extremum_graph(values: np.ndarray, graph: NeighborGraph,
                   min_loop_relief: float = 0.5) -> ExtremumGraph:
    """Compute the extremum graph of ``values`` sampled on ``graph``.

    Saddles that join two separate groups of peaks (``merge``) are always
    kept. A saddle that only closes a loop is kept when its relief is at
    least ``min_loop_relief`` (see ``_relief``); lower values drop more
    loops. Use ``0`` to keep every pass between adjacent regions.
    """
    values = np.asarray(values)
    n = graph.num_nodes
    if values.shape != (n,):
        raise ValueError(f"values must have shape ({n},), got {values.shape}")

    rank = _total_order(values)
    up = _steepest_ascent(rank, graph)

    # Maxima, numbered from the highest down so peak id 0 is the global maximum.
    max_vertices = np.flatnonzero(up == np.arange(n))
    max_vertices = max_vertices[np.argsort(-rank[max_vertices])]
    num_peaks = max_vertices.shape[0]
    peak_id = np.full(n, -1, dtype=np.int64)
    peak_id[max_vertices] = np.arange(num_peaks)
    region = peak_id[up]

    sad_vertex, sad_peaks, sad_support = _passes(rank, region, graph, num_peaks)
    persistence = _sweep(values, max_vertices, sad_vertex, sad_peaks)
    relief = _relief(values, rank, graph, sad_vertex)

    is_merge, sad_pers = persistence["is_merge"], persistence["saddle"]
    keep = is_merge | (relief >= min_loop_relief)
    sad_vertex, sad_peaks, sad_support = sad_vertex[keep], sad_peaks[keep], sad_support[keep]
    is_merge, sad_pers, relief = is_merge[keep], sad_pers[keep], relief[keep]
    saturation = _saturation(values[max_vertices], values[sad_vertex], sad_peaks,
                             sad_pers, is_merge)
    # Which peak absorbs which: the paper's cancellation partner. The sweep's
    # elder-rule parent (the top of the group) gives the same persistence but
    # can be a peak that is not even adjacent (observations.md, 2026-11-10).
    parent = _cancellations(num_peaks, sad_peaks, sad_pers, is_merge)["parent"]

    return ExtremumGraph(
        num_points=n,
        values=values,
        peaks=max_vertices,
        peak_values=values[max_vertices],
        peak_persistence=persistence["peak"],
        peak_parent=parent,
        saddles=sad_vertex,
        saddle_values=values[sad_vertex],
        saddle_peaks=sad_peaks,
        saddle_is_merge=is_merge,
        saddle_persistence=sad_pers,
        saddle_support=sad_support,
        saddle_relief=relief,
        saddle_saturation=saturation,
        region=region,
        region_size=np.bincount(region, minlength=num_peaks),
    )


def _total_order(values: np.ndarray) -> np.ndarray:
    """rank[i] = position of point i when sorted by (value, index), ascending."""
    n = values.shape[0]
    order = np.lexsort((np.arange(n), values))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def _steepest_ascent(rank: np.ndarray, graph: NeighborGraph) -> np.ndarray:
    """For each vertex, the maximum reached by always stepping to the highest neighbour."""
    n = graph.num_nodes
    up = np.arange(n, dtype=np.int64)

    has_nbr = graph.degree > 0
    if has_nbr.any():
        starts = graph.indptr[:-1][has_nbr]
        best_rank = np.maximum.reduceat(rank[graph.indices], starts)
        vertex_at_rank = np.empty(n, dtype=np.int64)
        vertex_at_rank[rank] = np.arange(n)
        v = np.flatnonzero(has_nbr)
        higher = best_rank > rank[v]
        up[v[higher]] = vertex_at_rank[best_rank[higher]]

    # Pointer jumping: follow pointers until every vertex points at a maximum.
    while True:
        nxt = up[up]
        if np.array_equal(nxt, up):
            return up
        up = nxt


def _passes(rank: np.ndarray, region: np.ndarray, graph: NeighborGraph,
            num_peaks: int):
    """Highest pass between every pair of adjacent regions.

    Returns (saddle vertices, (S, 2) peak ids, (S,) boundary-edge counts),
    sorted from the highest saddle to the lowest.
    """
    e = graph.edges
    ra, rb = region[e[:, 0]], region[e[:, 1]]
    cross = ra != rb
    e, ra, rb = e[cross], ra[cross], rb[cross]
    if e.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty((0, 2), dtype=np.int64), empty

    r0, r1 = rank[e[:, 0]], rank[e[:, 1]]
    level = np.minimum(r0, r1)                           # pass height of each edge
    low_vertex = np.where(r0 < r1, e[:, 0], e[:, 1])     # its lower endpoint
    a, b = np.minimum(ra, rb), np.maximum(ra, rb)        # a < b: a is the higher peak
    key = a * num_peaks + b

    order = np.lexsort((level, key))                     # by pair, then by height
    key, a, b, level, low_vertex = key[order], a[order], b[order], level[order], low_vertex[order]
    first = np.r_[True, key[1:] != key[:-1]]
    group_start = np.flatnonzero(first)
    group_end = np.r_[group_start[1:], key.shape[0]] - 1  # highest edge of each pair
    support = group_end - group_start + 1

    sad_vertex = low_vertex[group_end]
    sad_peaks = np.stack([a[group_end], b[group_end]], axis=1)
    sad_level = level[group_end]

    top_down = np.argsort(-sad_level)
    return sad_vertex[top_down], sad_peaks[top_down], support[top_down]


def _sweep(values, max_vertices, sad_vertex, sad_peaks) -> dict:
    """Union-find over saddles from highest to lowest; fills in persistence."""
    num_peaks = max_vertices.shape[0]
    num_saddles = sad_vertex.shape[0]

    parent = list(range(num_peaks))   # union-find over peak ids
    top = list(range(num_peaks))      # top[root] = highest peak id in that group

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    peak_persistence = np.full(num_peaks, np.inf)
    peak_parent = np.full(num_peaks, -1, dtype=np.int64)
    is_merge = np.zeros(num_saddles, dtype=bool)
    saddle_persistence = np.zeros(num_saddles)

    peak_vals = values[max_vertices].astype(np.float64)
    sad_vals = values[sad_vertex].astype(np.float64)

    for s in range(num_saddles):
        a, b = int(sad_peaks[s, 0]), int(sad_peaks[s, 1])
        ra, rb = find(a), find(b)
        if ra != rb:
            # Smaller peak id = higher peak, so the group with the larger top id dies.
            ta, tb = top[ra], top[rb]
            keep_root, dead_root = (ra, rb) if ta < tb else (rb, ra)
            dead_top = top[dead_root]
            peak_persistence[dead_top] = peak_vals[dead_top] - sad_vals[s]
            peak_parent[dead_top] = top[keep_root]
            saddle_persistence[s] = peak_persistence[dead_top]
            is_merge[s] = True
            parent[dead_root] = keep_root
        else:
            saddle_persistence[s] = min(peak_vals[a], peak_vals[b]) - sad_vals[s]

    return {"peak": peak_persistence, "parent": peak_parent,
            "is_merge": is_merge, "saddle": saddle_persistence}


def _cancellations(num_peaks: int, sad_peaks: np.ndarray, sad_pers: np.ndarray,
                   is_merge: np.ndarray) -> dict:
    """Replay the persistence simplification as in Correa et al. 2011, Algorithm 1.

    Merge saddles are cancelled in order of increasing persistence. Every peak
    points along a path ``peak -> saddle -> peak -> ...`` to the not yet
    cancelled ("terminal") peak its gradient flow now ends at. Cancelling s
    kills the lower of the two terminals its peaks lead to, and reverses the
    path to it (``REVERSE(G, U)``), so that flow now runs through s to the
    other terminal: that terminal is the dead peak's ``parent``. It is the
    peak on the other side of the cancelling saddle, which need not be the
    highest peak of the group it joins (the elder rule of a merge tree).

    Peak ids follow the total order (smaller id = higher), which also breaks
    ties in value. Merge saddles form a forest, so the two terminals always
    differ. Returns the parents, the final paths (``out_saddle``/``out_peak``),
    and for each cancelled saddle the reversed path, so ``_saturation`` can undo it.
    """
    num_saddles = sad_pers.shape[0]
    out_saddle = np.full(num_peaks, -1, dtype=np.int64)   # next saddle on the path to the terminal
    out_peak = np.full(num_peaks, -1, dtype=np.int64)     # and the peak it leads to
    parent = np.full(num_peaks, -1, dtype=np.int64)
    order = np.lexsort((np.arange(num_saddles), sad_pers))

    def path(u: int):
        nodes, saddles = [u], []
        while out_saddle[u] >= 0:
            saddles.append(int(out_saddle[u]))
            u = int(out_peak[u])
            nodes.append(u)
        return nodes, saddles

    undo = {}
    for s in order:
        if not is_merge[s]:
            continue
        a, b = int(sad_peaks[s, 0]), int(sad_peaks[s, 1])
        nodes_a, sads_a = path(a)
        nodes_b, sads_b = path(b)
        ta, tb = nodes_a[-1], nodes_b[-1]
        if ta == tb:                     # cannot happen for a forest; be safe
            continue
        if ta > tb:                      # the lower terminal dies
            nodes, sads, other, parent[ta] = nodes_a, sads_a, b, tb
        else:
            nodes, sads, other, parent[tb] = nodes_b, sads_b, a, ta
        for i in range(len(nodes) - 1, 0, -1):          # reverse the path to the dead peak ...
            out_saddle[nodes[i]] = sads[i - 1]
            out_peak[nodes[i]] = nodes[i - 1]
        out_saddle[nodes[0]] = s                        # ... and leave through s
        out_peak[nodes[0]] = other
        undo[int(s)] = (nodes, sads)
    return {"parent": parent, "out_saddle": out_saddle, "out_peak": out_peak,
            "undo": undo, "order": order}


def _saturation(peak_vals: np.ndarray, sad_vals: np.ndarray, sad_peaks: np.ndarray,
                sad_pers: np.ndarray, is_merge: np.ndarray) -> np.ndarray:
    """Saturated persistence p* of every saddle (Correa et al. 2011, section 3.4).

    The paper defines p*(s) as "the persistence of the last (highest-persistence)
    extremum that needed [the arcs of s] to be connected to the extremum graph".
    At noise threshold t:

    * the merge saddles with persistence < t form a forest over the peaks
      (the cancellations done so far); each tree holds exactly one surviving
      peak, the one every other peak in it was merged into;
    * every saddle with persistence >= t is drawn, so its two peaks are
      *ports*: its links have to reach the surviving peaks from there;
    * a forest edge (a simplified saddle) is needed if it lies on the path
      from a port to its tree's surviving peak.

    p*(s) is the largest t at which s is needed (or its own persistence, if
    larger). Going down in t, ports are only added and trees are only split,
    and an edge that is needed stays needed, so one sweep from high to low
    persistence finds every p*: replay the cancellations upwards to build the
    forest oriented towards the surviving peaks (``_cancellations``), then undo them in reverse order, marking the paths from
    each saddle's two peaks as it becomes a port.

    Why not Algorithm 1 itself? It marks paths only once, when a saddle is
    cancelled, and later cancellations re-route them. On neghip that leaves
    about 5 % of the drawn links without a ridge to their surviving peak (see
    observations.md, 2026-11-10). Loop saddles are ordered by our own
    ``saddle_persistence``, which also differs from the paper (same note).
    """
    num_saddles = sad_pers.shape[0]
    c = _cancellations(peak_vals.shape[0], sad_peaks, sad_pers, is_merge)
    out_saddle, out_peak, undo, order = c["out_saddle"], c["out_peak"], c["undo"], c["order"]

    # Downward: undo each cancellation, then mark the paths from its two peaks.
    saturation = sad_pers.astype(np.float64).copy()
    marked = np.zeros(num_saddles, dtype=bool)

    def mark_to_root(u: int, p: float, stop_at_marked: bool) -> None:
        while out_saddle[u] >= 0:
            e = int(out_saddle[u])
            if marked[e]:
                if stop_at_marked:          # everything from here to the root is marked too
                    return
            else:
                marked[e] = True
                saturation[e] = max(saturation[e], p)
            u = int(out_peak[u])

    for s in order[::-1]:
        p = float(sad_pers[s])
        a, b = int(sad_peaks[s, 0]), int(sad_peaks[s, 1])
        if int(s) in undo:
            nodes, sads = undo[int(s)]
            for i in range(len(nodes) - 1):              # restore: the dead peak is a root again
                out_saddle[nodes[i]] = sads[i]
                out_peak[nodes[i]] = nodes[i + 1]
            out_saddle[nodes[-1]] = out_peak[nodes[-1]] = -1
            # In the split-off tree the paths now run the other way, so the
            # marks there say nothing yet: walk all the way to its new root.
            near = nodes[0]
            mark_to_root(near, p, stop_at_marked=False)
            mark_to_root(b if near == a else a, p, stop_at_marked=True)
        else:
            mark_to_root(a, p, stop_at_marked=True)
            mark_to_root(b, p, stop_at_marked=True)
    return saturation


def _relief(values: np.ndarray, rank: np.ndarray, graph: NeighborGraph,
            sad_vertex: np.ndarray) -> np.ndarray:
    """How deep the valley below each saddle goes, as a fraction in [0, 1].

    Take all points with value <= f(saddle) and the connected piece of them
    that contains the saddle. ``relief`` is

        (f(saddle) - lowest value in that piece) / (f(saddle) - lowest value overall)

    A real saddle between two peaks joins valleys that run down to low
    values (relief near 1). Where many basins meet at the bottom of one bowl
    (around a minimum), the piece is just that bowl and the relief is near 0.

    The pieces come from the same computation run upside down: the minima
    of ``values`` are the maxima of ``-values``, and the passes between
    neighbouring minima are merged in order of height with union-find.
    """
    n = values.shape[0]
    flipped = n - 1 - rank                      # rank of -values
    up = _steepest_ascent(flipped, graph)       # steepest descent of the field
    min_vertices = np.flatnonzero(up == np.arange(n))
    min_id = np.full(n, -1, dtype=np.int64)
    min_id[min_vertices] = np.arange(min_vertices.shape[0])
    min_region = min_id[up]

    # Passes between minima, already sorted from lowest to highest in `values`.
    pass_vertex, pass_minima, _ = _passes(flipped, min_region, graph, min_vertices.shape[0])

    parent = list(range(min_vertices.shape[0]))
    lowest = values[min_vertices].astype(np.float64).tolist()   # lowest value per group

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pass_rank = rank[pass_vertex]
    relief = np.empty(sad_vertex.shape[0])
    j = 0
    floor = float(values.min())
    for s in np.argsort(rank[sad_vertex]):      # saddles from lowest to highest
        v = sad_vertex[s]
        while j < pass_rank.shape[0] and pass_rank[j] <= rank[v]:
            a, b = find(int(pass_minima[j, 0])), find(int(pass_minima[j, 1]))
            if a != b:
                parent[a] = b
                lowest[b] = min(lowest[a], lowest[b])
            j += 1
        height = float(values[v])
        piece_low = lowest[find(int(min_region[v]))]
        relief[s] = (height - piece_low) / (height - floor) if height > floor else 0.0
    return relief
