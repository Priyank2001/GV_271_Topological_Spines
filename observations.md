# Observations

Things we noticed while building the pipeline that may affect results later.
Newest entries at the top. Each entry says what we saw, why it matters,
and what is still open.

---

## 2026-11-11 — AnyEG vs our pipeline, and our pipeline on the GPU

**What we did.** `anyeg/` is a separate pipeline that computes extremum
graphs the way AnyEG does (https://github.com/BuptCIAGroup/AnyEG, commit
08068c2). AnyEG is a TTK filter, and its CMake file does not build on its own,
so `anyeg/native/build.sh` compiles its CUDA source (`cudaHelper.cu`,
unchanged except one include path) with a small driver instead of TTK.
`anyeg.TorchKernels` is a PyTorch port of its four GPU modules, and
`anyeg/assembly.py` ports its CPU post-processing (arcs, `.vtp` output).
`python3 -m comparison` runs both pipelines on AnyEG's three sample files and
six Open SciVis volumes, each with three value variants (raw; AnyEG's own
`preprocess_vti.py` recipe; values replaced by their rank). Full tables:
`comparison/results/anyeg_vs_ours.md`.

**1. The PyTorch port is exact.** It gives exactly the original's output
(maxima, saddles, both link arrays) in all 27 cases, on the GPU and on the
CPU. So does our new torch backend against NumPy (27/27).

**2. AnyEG does not break ties, so 8-bit volumes give absurd counts.** A
vertex is a maximum when no neighbour is *strictly* higher, so every voxel of
a flat region is a maximum: 243,266 maxima on `fuel` (93 % of the voxels),
124,320 on `neghip`, 1,562,047 on `hydrogen_atom`. Its own preprocessing
(`v + 1e-8·rank`, then stored as float32) does not fix this, because the
offsets fall below float32 resolution above about 0.1. Leftover maxima:
`neghip` 3,904 (vs 175 with distinct values), `hydrogen_atom` 7,975 (vs 938).
AnyEG's own samples are fine because their generator uses 1e-5 on values up
to 100.

**3. Same neighbourhood and same order: same maxima and basins.** With values
made distinct (rank) and our algorithm run on AnyEG's 14-neighbour grid,
maxima and basins agree exactly in all 9 inputs (Jaccard 1.000, 100 % of
voxels in the same basin). Steepest ascent is the same method in both. The
differences are only in the saddles:

| (rank values, 14 neighbours) | linked pairs AnyEG | ours | both | only ours | only AnyEG |
|---|---|---|---|---|---|
| scalar_field_v1 | 3 | 5 | 3 | 2 | 0 |
| fuel | 156 | 331 | 147 | 184 | 9 |
| neghip | 595 | 903 | 430 | 473 | 165 |
| silicium | 263 | 703 | 263 | 440 | 0 |
| hydrogen_atom | 2,270 | 6,019 | 1,828 | 4,191 | 442 |

* *Only ours:* we link every two basins that touch, through their highest
  pass. AnyEG keeps a vertex only if its upper link falls into two or more
  pieces, a real piecewise-linear saddle. Most of our extra pairs are passes
  that fail that test (fuel: 165 of 184), and almost all are loop-closing
  saddles kept by the relief filter (fuel: 183 of 184), not merge saddles.
  On AnyEG's designed sample both extra links have persistence 0 (they join
  the background ramp), so `simplify` removes them. On `silicium` they have
  persistence about 0.8 of the value range: basins that touch in the zero
  background. Those are the deep valley passes the paper's p_hi threshold
  cuts.
* *Only AnyEG:* its arcs also start at neighbours *below* the saddle, and a
  saddle gets an arc to every basin among its neighbours, so degenerate
  saddles link three or more maxima (`hydrogen_atom`: 1,610 saddles). Of
  AnyEG's 442 extra pairs on `hydrogen_atom`, 315 exist only through such a
  downhill arc, and 74 are pairs our relief filter dropped.
* AnyEG has no persistence or simplification. Its two thresholds are
  hard-coded to `[0, 20000)`, so it keeps everything. It has no saturated
  persistence, no p_lo/p_hi, and no point clouds (meshes only).

**4. Our 6-neighbour grid makes false maxima.** Our `grid_graph` joins only
axis neighbours, so a ridge that runs diagonally is cut into separate peaks.
AnyEG's v1 sample (3 designed maxima on diagonal ridges, plus the background
ramp's corner maximum): 7 maxima with 6 neighbours, 4 with 14. With
rank values: `neghip` 900 vs 175, `hydrogen_atom` 6,184 vs 938, `nucleon` 301
vs 65. Basins agree for only 29–77 % of the voxels between the two grids
(`nucleon` 29 %, `fuel` 77 %).
TTK, TACHYON and AnyEG all use the 14-neighbour Freudenthal triangulation.
**Open:** switch `grid_graph` (and so the web files) to 14 neighbours? It
changes every volume result.

**5. GPU speed.** Our pipeline now has a `ComputeBackend` (`pipeline/backends.py`).
Sorting, steepest ascent, the passes and the CSR build run in torch, and the
union-find sweeps stay on the CPU. Results are bit-identical. Graph build
plus extremum graph, NumPy vs torch on a shared RTX 3090
(`benchmarks/backend_benchmark.py`): 32³ 0.03 vs 0.10 s (GPU slower), 64³ 0.30
vs 0.13 s, `hydrogen_atom` 128³ 2.9 vs 0.70 s, smooth 256³ field 43 vs 5.7 s,
kNN 10⁶ points 4.9 vs 1.2 s. The default `AutoBackend` therefore uses the GPU
from 100,000 vertices. AnyEG's own GPU modules take 0.13 s on
`hydrogen_atom`, and our full `extremum_graph` 0.35–0.55 s on the GPU. That
is not like for like: ours also computes persistence, relief and p* on the
CPU.

**Side effect of the backend change.** Saddles at the same height (one
vertex being the pass for several pairs of peaks) used to be ordered by
NumPy's unstable quicksort, which is arbitrary and cannot be reproduced on a
GPU. They are now kept in pair order (stable sort). Peaks, persistence and
regions are unchanged on all nine test inputs. `peak_parent` changed for
peaks whose two candidate parents have the same persistence (`ext_50d` 43 of
1,061, `neghip` 1 of 900), and `neghip` has one saddle row different
(2,627 in all). The files in `web/data/` were rebuilt with it on 2026-11-11. In
`neghip` this moved one label: two saddles sit at the same vertex (103487),
between peaks 6–21 and 21–23. The merge is now the one to the higher peak (6),
and the other closes a loop (its prominence went from 0.102 to 0.059). Both
labellings are valid for a tie. A browser check that assumed such a ridge at
threshold 0.1 now searches for a threshold with one.

**Compare mode in the viewer (2026-11-11).** AnyEG is exported for every
dataset where its graph fits the spine layout (≤ 3,000 nodes). On the kNN
point clouds its maxima and basins equal ours exactly (same graph, same order),
but its saddle test is far more permissive there. A kNN graph is not a
triangulation, so the "upper link in two or more pieces" test passes for most
candidates. Results: `cyclic_3d` 212 AnyEG saddles against our 23; `ext_10d`
13,328 saddles, 12,752 of them reaching three or more maxima (83k nodes, not
exported); `cyclic_10d` 27k nodes (not exported). On the volcano fields the linked
pairs are identical to ours (Jaccard 1.0); AnyEG just draws 48–57 saddles for
the 6 links. Design: `docs/extremum_graph_variants.md`.

---

## 2026-11-10 — the paper's volcano (Fig. 5): a breached rim still gives a ring

**What we did.** `generator/make_volcano_dataset.py` makes the "cycle of
hills" of the paper's Fig. 5 as 2D point clouds (30k points): a crater floor,
a rim with 6 hills, and a notch of depth b cut into the rim saddle at 30°.
The paper's rule (section 3.4): a saddle closer in value to its hills than to
the crater floor is ridge-like and keeps the ring (Fig. 5b); otherwise the rim
is really breached, the saddle is valley-like and is cut, and the ring becomes
a chain (Fig. 5a). `tests/check_volcano.py` runs our pipeline (k = 15, noise
threshold 0.02); the three datasets are also in the viewer.

| dataset | b | breach saddle: below its hills / above the floor | paper | ours, p_hi off | p_hi 0.25 | p_hi 0.30 | p_hi 0.45 |
|---|---|---|---|---|---|---|---|
| volcano_intact | 0 | 0.196 / 0.420 | ring | ring | ring | ring | ring |
| volcano_notched | 0.2 | 0.281 / 0.331 | ring | ring | **chain** | ring | ring |
| volcano_breached | 0.8 | 0.513 / 0.090 | chain | **ring** | chain | chain | chain |

Prototype sweep over b (same field): our persistence of the breach saddle is
0.196, 0.237, 0.279, 0.361, 0.440, 0.509, 0.551 for b = 0, 0.1, 0.2, 0.4, 0.6,
0.8, 0.95; the paper's rule switches from ridge to valley between b = 0.2 and
0.4. The relief of the breach saddle is 1.00 for every b.

**Why it matters.**
- **With p_hi off, our spine shows a closed ring even when the rim is breached
  almost to the crater floor.** This is the failure the paper's Fig. 5 is
  about. The paper decides ridge or valley per saddle from the minimum on the
  other side; we never look at the minima. Our loop-saddle persistence
  (min(f(a), f(b)) − f(s)) only says how far the saddle is below its hills, and
  our relief filter cannot help either: below the breach saddle the crater and
  the plain outside join, so its relief is 1.
- **p_hi fixes it only when set by hand and per dataset.** The breach's
  persistence (0.513) stands out against the other rim saddles (0.185–0.197),
  so any p_hi between about 0.28 and 0.51 gives the paper's answer for all
  three. But a p_hi just above the ordinary rim saddles (0.25) opens the
  notched rim, which the paper keeps: persistence alone cannot tell a deep
  notch from a breach, because only the crater floor separates them.

**Still open**
- [ ] The paper's automatic classification: per saddle, the persistence towards
      the minima (f(s) minus the higher of the two minima it separates below
      it, here the crater floor) compared with the persistence towards the
      maxima; a valley-like saddle is dropped from the maximum graph. The
      sublevel-set union-find of `_relief` already visits the saddles in the
      right order and could return it.

---

## 2026-11-10 — Ackley on kNN graphs: right peaks from k ≈ 10, extra diagonal links

**What we did.** The Ackley check below (`tests/check_ackley.py`) used grids
only. `tests/check_ackley_knn.py` repeats it on `knn_graph(points, k)` of N
uniform random points in [-1.5, 1.5]^d. Sampling makes spurious peaks, so
each graph is simplified at the middle of the persistence gap between the
3^d-th and the next peak (this uses the known answer, see "Still open").
Three seeds each; ranges are over the seeds.

| d | N | k | raw peaks | the 3^d peaks on the lattice | true links | diagonal links | centre degree |
|---|---|---|-----------|------------------------------|------------|----------------|---------------|
| 2 | 20k | 4 | 620–680 | no (no persistence gap) | 4–6 / 12 | 2–3 | – |
| 2 | 20k | 6 | 126–161 | yes | 11–12 / 12 | 0 | 4 |
| 2 | 20k | 10 | 16–24 | yes | 11–12 / 12 | 0 | 4 |
| 2 | 20k | 15, 25 | 9–12 | yes | 12 / 12 | 0 | 4 |
| 3 | 100k | 4 | ~3560 | no | 29–45 / 54 | 36–42 | – |
| 3 | 100k | 6 | 861–887 | yes | 54 / 54 | 60–65 | 6 |
| 3 | 100k | 10 | 88–96 | yes | 54 / 54 | 66–70 | 6 |
| 3 | 100k | 15 | 30–36 | yes | 54 / 54 | 72–74 | 6 |
| 3 | 100k | 25 | 27–28 | yes | 54 / 54 | 78–81 | 6 |
| 4 | 400k | 4, 6 | 4200–15700 | no | 179–211 / 216 | 409–574 | – |
| 4 | 400k | 10 | 539–558 | 2 of 3 seeds | 211–216 / 216 | 610–659 | 8 |
| 4 | 400k | 15, 25 | 86–157 | yes | 216 / 216 | 686–750 | 8 |

(Grid, for comparison: 3D 54 true + 36 diagonal, 4D 216 + 328.)

**Why it matters.**
- **Peaks are right once k is large enough**: k ≥ 6 in 2D and 3D, k ≥ 15 in 4D.
  Larger k also means fewer spurious peaks and a wider persistence gap. With
  k = 4 the spurious peaks are as persistent as the real ones and no
  threshold helps.
- **No true link is lost in 3D and 4D.** In 2D, 1 of 12 is missing in 2 of 15
  runs (k = 6 and 10). It is a loop saddle that the relief filter
  (`min_loop_relief = 0.5`) drops. With `--min-loop-relief 0` all 12 are
  always found, at the cost of 4 diagonal links.
- **The kNN graphs have more diagonal links than the grid, and p_hi cannot
  remove them all.** Grids only make edge–edge and face–face diagonals, whose
  persistence (0.21–0.22) is well above the true links' (≤ 0.112). kNN edges
  point in every direction, so they also make centre–edge, face–corner and
  centre–corner links. Breakdown for 3D, k = 15:

  | link type | count | persistence |
  |---|---|---|
  | true: centre–face, face–edge, edge–corner | 6, 24, 24 | 0.052–0.066, 0.087–0.098, 0.101–0.116 |
  | diagonal: centre–edge | 12 | 0.100–0.112 |
  | diagonal: face–corner, centre–corner | 22, 8 | 0.123–0.137, 0.127–0.135 |
  | diagonal: edge–edge, face–face | 22, 8 | 0.215–0.341, 0.225–0.325 |

  The 12 centre–edge links overlap the edge–corner ones, so the best single
  p_hi gets 12 links wrong in 3D for every k ≥ 10 (12–19 at k = 10) and
  38–88 in 4D.
- **`saddle_support` separates them much better.** Support is the number of
  kNN edges between the two regions. True links share a face of dimension
  d − 1, while diagonal basins only touch along a thinner set, so true links
  have far more edges across. The best single minimum support gets 0 wrong in
  3D for k ≥ 15 (one run: 1 at N = 400k), 2–7 at k = 10, and 2–14 in 4D for
  k ≥ 15. The cut is not a fixed number, though: it grows with N and k
  (3D, k = 15: about 160–230 at N = 25k, 600–750 at 100k, 2200–2800 at 400k).
- More points (3D, k = 10: N = 25k → 400k) give more raw peaks at a fixed k
  but do not change the link results much. k matters more than N.

**Still open**
- [ ] p_lo was set using the known answer (3^d). The largest jump in the
      persistence curve is at 3^d in only about half of the runs, so p_lo
      would have to be found by hand.
- [ ] A support-based link filter (support normalised by N and k, or by the
      region sizes) would remove most kNN diagonal links. It is not in the
      paper and has not been built.

---

## 2026-11-10 — saturated persistence p*, and the paper's cancellation partner

**What we did.** p* (section 3.4) is now computed for every saddle
(`ExtremumGraph.saddle_saturation`, exported as `saddles[i].saturation`) and
the viewer draws the paper's equation (3): the arcs of saddle s are shown when
p(s) <= p_hi and p*(s) >= p_lo, going to the saddle's own two peaks even when
those peaks are hidden. Links then run along the ridge through the peaks they
absorbed (checkbox "Keep the arcs of merged peaks", on by default).

**Three differences from the paper's Algorithm 1, on purpose.**

1. *Loop-saddle persistence.* The paper gives a loop saddle (both peaks
   already joined) the persistence |f(terminal) − f(s)|, measured against the
   current top of the merged group. We keep ours, min(f(a), f(b)) − f(s),
   measured against its own two peaks. Peak persistence and merge-saddle
   persistence agree exactly on all six viewer datasets either way (fuel has
   one tie in its 8-bit values). On Ackley the paper's version mixes true and
   diagonal links (3D: true 0.06–0.59, diagonal 0.50–0.61), while ours keeps
   them apart (true 0.06–0.11, diagonal 0.21–0.22), which is what lets p_hi
   cut the diagonal links.

2. *p\* is computed exactly, not with Algorithm 1's marking.* Algorithm 1
   marks the saddles on a path once, when a saddle is cancelled, but later
   cancellations reverse paths. With it, some drawn links had no ridge to the
   peak they join: on raw neghip, 14 of 366 hidden endpoints at t = 0.03, 39
   of 240 at 0.08 and 20 of 203 at 0.1. We use the paper's definition of p*
   ("the persistence of the last extremum that needed the arc") directly. At
   threshold t, a cancelled saddle is needed if it lies on the path from a
   drawn saddle's peak to the surviving peak of its tree. Going down in t, a
   needed saddle stays needed, so one sweep from high to low persistence
   (undoing the cancellations) gives p* for all saddles at the cost of
   Algorithm 1. Afterwards: 0 hidden endpoints without a ridge on any dataset
   or threshold we tried. This is checked in Python on the raw and simplified
   graphs, and in the browser on the fixtures (504 endpoints reached through
   ridges, 0 not).

3. *The peak a removed peak merges into.* Before this change, `simplify` merged
   a removed peak into the top of the group it joins (`peak_parent` from the
   merge-tree sweep, the "elder rule"). The paper cancels it into the peak
   across its saddle: REVERSE redirects its flow to the other terminal, which
   is its neighbour. The persistence values are the same, but the parent is
   often different: 765 of 900 raw peaks on neghip, 46 of 88 on fuel, 44 of 73
   on ext_10d. With the elder rule a removed peak could join a peak it does not
   touch, so the ridge (through its real neighbour) and the link disagreed.
   `peak_parent` is now the paper's partner. This changes regions and links of
   simplified graphs:

   | dataset | t | points that change region | links (old → new) |
   |---|---|---|---|
   | fuel | 0.02 | 84 % | 94 → 86 |
   | fuel | 0.1 | 89 % | 6 → 5 |
   | neghip | 0.02 | 25 % | 193 → 173 |
   | neghip | 0.1 | 26 % | 35 → 28 |
   | silicium | 0.1 | 30 % | 590 → 590 |
   | ext_10d | 0.05 | 18 % | 271 → 272 |
   | cyclic_3d / cyclic_10d | base | 0 % | 6 → 6 |

   Fuel's large share is its one huge background region, which now goes to the
   neighbouring peak instead of the global maximum. The unsimplified graph,
   all persistence values, the Ackley check and the 6-cycles are unchanged.

**Still open**
- [x] In the contour style the lobes follow the straight link from the
      surviving peak to the saddle, not the ridge; the ridge is only a thin line.
      Done: with p* on, a lobe whose saddle's own peak was merged away follows
      the ridge polyline (`ridgeRoute` in simplify.js, `route` in contours.js),
      with f running linearly in arc length and round joins at the bends.
- [ ] Most ridge arcs still have no lobe. On neghip at t = 0.1 only 1 of the
      28 links runs through a merged peak; the 39 ridge arcs are 19 merged
      saddles, 9 duplicate links (a second saddle between two peaks that
      already have a link, 8 of them through merged peaks) and 11 saddles with
      p >= t whose two peaks merged into the same peak (loops onto one peak,
      like the volcano rim). `simplify` drops the last two kinds, so they are
      drawn as thin lines with no saddle marker and no lobe. Fuel at 0.1:
      2 of 5 links routed; 19 + 3 + 32. Giving them lobes means making them
      nodes of the simplified graph, as in the paper.
- [ ] Two saddles between the same pair of surviving peaks: `simplify` keeps
      the highest, but equation (3) draws the arcs of both. The viewer follows
      equation (3), so such a pair shows two paths.

---

## 2026-11-10 — is our extremum graph the paper's? Ackley says: same peaks, extra links

**What we did.** The paper (Correa et al. 2011, Fig. 13a) states a checkable
result for Ackley's function: the 2D–5D versions have 3^d minima (a 3x3x…
lattice, global minimum at the centre) and the centre links to 2d of them. We
sampled -Ackley on [-1.5, 1.5]^d grids (301², 81³, 41⁴, 25⁵ points), so the
minima become maxima, ran `extremum_graph` on the grid graph and compared with
the lattice. Script: `tests/check_ackley.py`.

| d | peaks (expect 3^d) | at the lattice points | true lattice links found | extra links | centre degree (expect 2d) |
|---|--------------------|-----------------------|--------------------------|-------------|---------------------------|
| 2 | 9                  | yes                   | 12 of 12                 | 0           | 4                         |
| 3 | 27                 | yes                   | 54 of 54                 | 36          | 6                         |
| 4 | 81                 | yes                   | 216 of 216               | 328         | 32                        |
| 5 | 243                | yes                   | 810 of 810               | 808         | 130                       |

**Why it matters.** The peaks, their count and their positions are exactly
right, with no simplification needed, and no true link is missing. What differs
from the paper is the extra links, which join lattice points that are 2 or more
steps apart (diagonals). They exist because the axis-aligned grid graph makes
two diagonal basins touch, so `_passes` reports a pass between them. The paper
removes them with its second threshold, **p_hi** ("variation"): saddles with
persistence above p_hi are dropped. We have only the noise threshold (p_lo).
The extra links have a larger saddle persistence than the true ones, so p_hi
would work, but the gap closes with dimension:

| d | true links, saddle persistence | diagonal links, saddle persistence |
|---|--------------------------------|------------------------------------|
| 3 | 0.060 – 0.112                  | 0.214 – 0.222 (clean gap)          |
| 4 | 0.041 – 0.098                  | 0.087 – 0.193 and 0.195 – 0.197 (overlap at 0.087–0.098) |

We did not compare the real volumes (neghip, fuel) link by link, because the
paper gives no list of their peaks or links. Qualitatively, neghip at
threshold 0.1 gives 23 peaks and 35 saddles (28 since the parent change
above) with two mirror-symmetric
clusters, which matches the shape of Fig. 9; this is a visual match, not a test.

**Done since:** p_hi is implemented (`ExtremumGraph.simplify(threshold, variation=...)`,
the JS port, a slider and the paper's red curve in the viewer). The 3D result
above is what it does on the lattice: a p_hi between 0.12 and 0.21 keeps exactly
the 54 true links.

**Still open**
- [x] The paper's saturated persistence p* — done, see the entry above.
- [ ] Ackley 4D overlaps: p_hi alone will not separate every diagonal link
      (true 0.041–0.098, diagonal 0.087–0.193).
- [x] The kNN graphs of point clouds were not checked against a known lattice;
      Ackley was only run on grids. Done, see "Ackley on kNN graphs" above.

---

## 2026-11-10 — spine layout: fine for rings, many crossings for loopy graphs

**What we saw.** `spine_layout` (PCA start, then stress majorisation on
distances in the original space) on simplified graphs:

| field        | peaks | saddles | stress (PCA → final) | crossings |
|--------------|-------|---------|----------------------|-----------|
| cyclic_3d    | 6     | 6       | 0.0075 → 0.0045      | 0         |
| cyclic_10d   | 6     | 6       | 0.196 → 0.116        | 0         |
| fuel         | 6     | 6       | 0.021 → 0.009        | 0         |
| neghip       | 23    | 35      | 0.294 → 0.222        | 28        |
| silicium     | 114   | 590     | 0.257 → 0.224        | 4,908     |
| ext_10d      | 24    | 271     | 0.523 → 0.335        | 12,080    |

**Why it matters.** Refinement always helps, but it only trades distance
error. When most saddles close loops (silicium: 477 of 590, ext_10d: 248 of
271) the graph is not planar in any layout that keeps distances, and the
crossing count reaches thousands. This is the problem feature 5 (fewer
crossings, level of detail) has to solve. The two lattice-like and the
high-d graphs are the test cases.

**Still open**
- [ ] A crossing-aware term or a post-pass (e.g. move saddles to reduce crossings).
- [ ] Layout is only as good as straight-line distance in d dimensions; try
      graph (shortest-path) distances for the high-d sets.
- [ ] The stress is computed on the simplified graph; the layout should stay
      stable when the threshold slider changes.

---

## 2026-11-09 — building the kNN graph after dimensionality reduction does not help

**What we tried.** `pipeline/dim_reduction.py` reduces the points (PCA, random
projection, Isomap, t-SNE, ...) and builds the kNN graph in the embedding
(`reduced_knn_graph`), keeping the original field values. k=20, simplify
threshold 0.01 for `cyclic_10d` and 0.02 for the others.

| dataset    | method           | max degree | peaks after simplify | saddles |
|------------|------------------|------------|----------------------|---------|
| cyclic_10d | none             | 73    | **6** (correct 6-cycle) | 6     |
| cyclic_10d | PCA → 2          | 34    | 6                    | 8       |
| cyclic_10d | PCA → 3          | 39    | 6                    | 9       |
| cyclic_10d | random proj. → 3 | 35    | 2,265                | 17,329  |
| cyclic_10d | Isomap → 3       | 43    | 1,246                | 7,730   |
| cyclic_10d | t-SNE → 2        | 37    | 2,159                | 6,911   |
| ext_10d    | none             | 56    | 40                   | 738     |
| ext_10d    | PCA → 2 / t-SNE → 2 | 33 / 40 | 1,578 / 1,518     | 5,540 / 5,084 |
| ext_50d    | none             | 1,149 | 163                  | 12,251  |
| ext_50d    | PCA → 2 / t-SNE → 2 | 34 / 33 | 1,133 / 1,058     | 2,097 / 1,985 |
| smooth_20d | none             | 145   | 1,434                | 187,667 |
| smooth_20d | PCA → 2 / t-SNE → 2 | 33 / 34 | 1,831 / 1,866     | 6,456 / 6,509 |

**Why it matters.** Reduction removes the hubs (max degree drops to about 35
everywhere), but the field values come from the full space. Points that land
next to each other in the embedding can have very different values, so the
field looks like noise on the new graph: thousands of small peaks that
persistence simplification cannot remove. PCA only comes close on
`cyclic_10d`, because there the field changes mainly in the (x0, x1)
plane, and even there it adds 2 to 3 false saddles. The `ext_*` and
`smooth_20d` fields depend on every coordinate, so no 2D or 3D embedding
can keep them a clean function.

**Conclusion.** Use the reducers for *drawing* (the 2D scatterplot of
high-d data, the PCA start of the spine layout), not for building the
graph that the extremum graph is computed on.

**Still open**
- [ ] Hub fixes that keep the original space: cap the degree, or use
      mutual-proximity / local-scaling distances.
- [ ] Try PCA with enough components to keep, say, 95% of the variance
      (`n_components=0.95`) instead of 2 or 3.

---

## 2026-11-09 — extremum graph: false saddles around minima, and sparse high-d data

**What we saw.** The first version of `extremum_graph` found the right peaks
on `cyclic_2d` (k=20) but 13 saddles instead of 6. Six were the ring's true
saddles (value 0.953, radius 2.5, matching `truth.json`). Seven were at the
centre of the ring, where all six regions meet at the bottom of the bowl:
regions whose peaks are not neighbours touch there, so the pass between them
looks like a saddle. In 3D this linked all 15 pairs of peaks.

**What we did.** A saddle that only closes a loop (so it is not needed to
keep the peaks connected) is kept only if its *relief* is at least 0.5. Relief
is how far the valley below the saddle goes: real ring saddles join valleys
that run down to the field minimum (relief 1.0 on every ring dataset), the
centre passes sit inside one bowl (relief at most 0.17 in 2D, 0.15 in 3D and
0.26 in 10D for k from 10 to 80). A plain merge-tree sweep cannot be used for
this: it gives only 5 saddles for 6 peaks and never finds the loop-closing one.

**Result.** With `simplify(0.04)`, `cyclic_2d` gives exactly the 6-cycle for
k from 10 to 40, `cyclic_3d` for k from 12 to 40, and `cyclic_10d` (threshold
0.01) for k from 20 to 80.

**Still open**
- [ ] At k=8 (and k=10 in 3D) a few extra saddles remain between peaks that
      are not neighbours, mostly in sparse regions far from the data. The relief
      filter does not catch them.
- [ ] Only the highest saddle between two peaks is reported. Two true saddles
      between the same two peaks appear as one link.
- [ ] `ext_10d`, `ext_50d` and `ext_100d` cannot be checked against their
      designed maxima: samples barely reach the bumps (0 or 1 samples within
      two bump widths of a maximum in 10D, none in 50D and 100D; nearest sample
      to a centre is 2.5 to 3.8 away in 10D, 17 to 20 in 50D, about 30 in
      100D). The `cyclic_*` datasets are the high-d test. We have not yet run
      the hub check from the entry below on the extremum graph.

---

## 2026-11-08 — kNN graphs get lopsided ("hubs") in high dimensions

**What we saw.** In the undirected kNN graph (`knn_graph(points, k)`, union
rule), most points have about k neighbours. In high dimensions, though, a
few points become neighbours of hundreds of others, and many points are
nobody's neighbour. Measured with k = 10 on the generator datasets:

| dataset    | d   | N       | mean degree | max degree | points in no one's kNN | pieces with `mode="mutual"` |
|------------|-----|---------|-------------|------------|------------------------|-----------------------------|
| ext_2d_small | 2   | 5,000   | 11.4 | 18  | 0.0%  | 1      |
| ext_3d     | 3   | 20,000  | 11.6 | 20  | 0.0%  | 1      |
| ext_10d    | 10  | 50,000  | 12.8 | 32  | 0.1%  | 51     |
| cyclic_10d | 10  | 100,000 | 14.5 | 36  | 1.8%  | 2,907  |
| smooth_20d | 20  | 50,000  | 14.2 | 81  | 1.1%  | 657    |
| ext_50d    | 50  | 100,000 | 15.7 | **592** | 5.7%  | 7,095  |
| ext_100d   | 100 | 100,000 | 16.4 | **470** | 10.1% | 12,838 |

"Points in no one's kNN" means the point appears in no other point's list of
k nearest neighbours. Under the union rule such a point is still connected
through its own k neighbours.

This is a known effect of high dimensions, called *hubness*: points near the
centre of the data end up close to almost everything.

**Why it matters for the extremum graph.**
- A hub joins points that are far apart in the field. If a hub sits between
  two peaks, the two peaks can look directly connected (a fake saddle or
  edge), or a small peak can be absorbed into a bigger one.
- Peak and saddle detection compares each point with its neighbours. A hub
  with 500 neighbours is almost never a local maximum, and a point nobody
  links to has fewer chances to be compared.
- **Don't use `mode="mutual"` in high dimensions.** It splits the graph into
  thousands of pieces, and each piece would produce its own fake maximum.
  The default (`"union"`) is connected on every dataset.

**Still to check**
- [ ] When we compute the extremum graph, check whether wrong or extra
      edges pass through high-degree points. The `cyclic_*` datasets have a
      known answer (a 6-cycle), so they are the test case.
- [ ] How results change with k (try 5, 10, 20, 40) on `cyclic_10d` and
      `ext_50d`.
- [ ] Possible fixes if hubs cause trouble (none tried yet): cap each
      point's degree, scale distances to reduce hubness (e.g. mutual
      proximity or local scaling), or build the graph after PCA reduces the
      dimension (tried 2026-11-09: removes hubs but breaks the field, see above).

**Reproduce**

```python
import numpy as np
from pipeline.data_loader import load_dataset
from pipeline.neighbor_graph import knn_graph, knn_indices

ds = load_dataset("ext_100d")
g = knn_graph(ds.points, k=10)
print(g)                                   # degree min/mean/max
indeg = np.bincount(knn_indices(ds.points, 10).ravel(), minlength=ds.num_points)
print((indeg == 0).mean())                 # share of points in no one's kNN
```
