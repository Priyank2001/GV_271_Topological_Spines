# Web-Based Exploration of Extremum Graphs using Topological Spines

E0 271 Graphics and Visualization course project. Team: Yashi, Priyank.

## Objective

The extremum graph of a scalar field records its peaks and how neighbouring
peaks connect through saddles. Volume renderings hide parts of a field behind
other parts, and fields with more than three dimensions cannot be drawn
directly at all.

This project builds a **web-based interactive tool** that computes the
extremum graph and shows it as a 2D **topological spine** (Correa et al.,
2011). The spine is a flat map of the peaks, how they connect, and how much
data each one holds. The spine and the data are drawn in the browser with
**Three.js** (WebGL), for both low- and high-dimensional data.

It also addresses two problems the original paper leaves open:

- spines get hard to read when lines cross;
- the data view is not arranged to match the spine.

## Features (in priority order)

1. **Extremum graph computation.** For 3D grids, use TACHYON or TTK. For
   high-dimensional point samples, build a k-nearest-neighbour graph (any k)
   and find peaks and saddles on it. Export the graph and each point's
   region to JSON.
2. **2D spine layout.** Place peaks and saddles in 2D with PCA, then refine
   with a force-directed layout that roughly keeps their real distances.
3. **Spine rendering** in Three.js. Each part is sized by how much data its
   peak holds. Sliders hide small, noisy peaks, guided by a persistence plot.
4. **Data view** next to the spine: a 3D volume rendering for low-dimensional
   data and a 2D scatterplot for high-dimensional data. Clicking a peak
   highlights its region.
5. **Cleaner layout and level of detail.** Reduce line crossings, and collapse
   small peaks into their parent when the graph is large.
6. **Spine-anchored scatterplot.** Place each data point around the spine
   node of its peak, so both views share one arrangement. Brushing either
   view highlights the matching part of the other.

Future work: time-varying data, showing how the spine changes over time.

## Repository layout

| Folder       | Contents |
|--------------|----------|
| `generator/` | Synthetic scalar fields (point clouds, 2D to 100D) with known critical points, written to `generator/datasets/<name>/` as `points.npy`, `values.npy`, `truth.json`. Samples only, no neighbour graph. |
| `pipeline/`  | Python side: dataset loading (`data_loader.py`), neighbour graph (`neighbor_graph.py`), dimensionality reduction (`dim_reduction.py`), extremum graph (`extremum_graph.py`), CPU/GPU compute backends (`backends.py`), 2D spine layout (`spine_layout.py`), JSON export for the web viewer (`export_json.py`) |
| `anyeg/`     | A separate pipeline that computes extremum graphs the way AnyEG does: its original CUDA kernels (built by `anyeg/native/build.sh`) or a PyTorch port, plus `.vti`/`.vtp` I/O. Does not use `pipeline/`. See "AnyEG pipeline" below |
| `comparison/` | Runs AnyEG and our pipeline on the same fields and compares them (`python3 -m comparison`); report in `comparison/results/`. Also the registry of extraction variants the viewer can compare (`variants.py`) and their export (`viewer_export.py`) |
| `docs/`      | `extremum_graph_variants.md`: design of the compare mode and of running every method on distinct values |
| `benchmarks/` | `backend_benchmark.py`: our pipeline, NumPy vs torch |
| `third_party/` | AnyEG checkout (not committed; `anyeg/native/build.sh` clones it) |
| `notebooks/` | `extremum_graph_demo.ipynb`: worked example and plots of the extremum graph on the generator datasets. `klacansky_extremum_graphs.ipynb`: extremum graphs of six real volumes from the Open SciVis Datasets |
| `data/klacansky/` | Downloaded `.raw` volumes and the collection's `datasets.json`; fetched and read by `pipeline/volume_loader.py` |
| `tests/`     | `python3 -m pytest tests`. Also `check_ackley.py` (grid) and `check_ackley_knn.py` (kNN graphs, any k): the extremum graph of Ackley's function compared with its known 3^d lattice (paper, Fig. 13a); `check_volcano.py`: the paper's breached-volcano example (Fig. 5, datasets from `generator/make_volcano_dataset.py`); results in `observations.md` |
| `web/`       | Three.js front end: spine (paper-style contours), persistence slider and plot, data view: scatterplot with a dimensionality-reduction choice for point clouds, volume rendering for grids. See "Web viewer" below |

## Loading data and building the neighbour graph

```python
from pipeline.data_loader import load_dataset
from pipeline.neighbor_graph import knn_graph, grid_graph

ds = load_dataset("cyclic_3d")
ds.points, ds.values            # (N, d), (N,)
ds.maxima, ds.saddles           # ground truth, where known

g = knn_graph(ds.points, k=12)  # any k; undirected (union of kNN relations)
g.neighbors(0), g.edges         # CSR lookup, (E, 2) edge list

grid_graph((64, 64, 64))        # axis-aligned adjacency for regular grids
```

kNN uses a KD-tree up to 10 dimensions and GPU brute force above that.
`knn_graph`, `grid_graph` and `NeighborGraph.from_edges` take `backend=`
(see "GPU backend" below) for building the adjacency.

## Dimensionality reduction

```python
from pipeline.dim_reduction import make_reducer, reduced_knn_graph, available_reducers

Y = make_reducer("pca", n_components=2).fit_transform(ds.points)  # or "tsne", "isomap", ...
g, Y = reduced_knn_graph(ds.points, k=12, method="tsne", n_components=2)
make_reducer("pca>tsne", pca={"n_components": 50})                # chain of steps
```

Every method has the same `Reducer` interface and is chosen by name
(`available_reducers()`: none, pca, random_projection, kernel_pca, isomap,
tsne, umap). Any scikit-learn-style estimator can be passed in directly,
and `register_reducer` adds a new name. Slow methods take
`fit_sample_size` (fit on a subset, place the rest). Use reductions for
drawing high-d data; building the extremum graph on a reduced graph gives
many false peaks (see `observations.md`, 2026-11-09).

## Computing the extremum graph

```python
from pipeline.extremum_graph import extremum_graph

eg = extremum_graph(ds.values, g)   # works on any NeighborGraph, any dimension
eg.peaks, eg.saddles                # vertex indices, highest first
eg.saddle_peaks                     # (S, 2) the two peaks each saddle links
eg.peak_persistence                 # inf for the global maximum
eg.region, eg.region_size           # peak id of every point, points per peak

eg.simplify(0.04)                   # drop peaks with persistence < 0.04 (the paper's p_lo, "noise")
eg.simplify(0.04, variation=0.2)    # and cut links with persistence > 0.2 (p_hi, "variation")
eg.saddle_saturation                # saturated persistence p* of every saddle (paper, section 3.4)
eg.arc_mask(0.04, variation=0.2)    # saddles whose arcs are drawn: the paper's equation (3)
```

Peaks are points with no higher neighbour. Every point climbs to its highest
neighbour until it reaches a peak, which gives its region. Where two regions
touch, the highest pass is the saddle between them. Persistence comes from
merging regions from the highest saddle down. Simplification then replays the
paper's Algorithm 1: a removed peak's region joins the peak across the saddle
that cancels it, and the paths the flow takes through removed peaks give each
saddle its saturated persistence p*, so links can still be drawn along the ridge
through the peaks they absorbed. The method and its limits are
described in the docstring of `pipeline/extremum_graph.py`. Run time is about
a second for 100,000 points.

Choose the `simplify` threshold from the persistence of the peaks: real peaks
stand well above the noise (see `notebooks/extremum_graph_demo.ipynb`). The
right value depends on the data (0.04 for the 2D and 3D rings, about 0.01 for
the 10D ring).

## GPU backend

The bulk array work of `grid_graph`, `knn_graph` and `extremum_graph` runs on a
`ComputeBackend` (`pipeline/backends.py`). This covers sorting the values,
steepest ascent with pointer jumping, the highest pass between regions, and
building the CSR adjacency. The union-find sweeps over the saddles are short
and sequential, so they stay on the CPU.

```python
from pipeline.backends import get_backend

eg = extremum_graph(values, g)                    # default "auto": GPU from 100,000 vertices
eg = extremum_graph(values, g, backend="numpy")   # or "torch", "cuda:1", "torch:cpu"
g = grid_graph(vol.shape, backend=get_backend("torch"))
```

Every backend gives bit-identical results (`tests/test_backends.py`).
`"torch"` picks the GPU with the most free memory, because the lab GPUs are
shared. `"auto"` falls back to NumPy if the GPU runs out of memory. On an
RTX 3090, graph build plus extremum graph takes 0.70 s instead of 2.9 s for
`hydrogen_atom` (128³), and 5.7 s instead of 43 s for a 256³ field. Below
about 30,000 vertices the GPU is slower (`python3 benchmarks/backend_benchmark.py`).

## AnyEG pipeline and comparison

`anyeg/` computes extremum graphs as [AnyEG](https://github.com/BuptCIAGroup/AnyEG)
does (a CUDA + TTK filter), as a separate pipeline:

```bash
anyeg/native/build.sh                                  # once: clones AnyEG, builds its CUDA code
python3 -m anyeg third_party/AnyEG/data/scalar_field_v1.vti -o results/v1.vtp
python3 -m anyeg --volume fuel --kernels original -o results/fuel.vtp
python3 -m comparison                                  # -> comparison/results/anyeg_vs_ours.md
```

```python
from anyeg import AnyEGPipeline, read_vti, write_vtp, OriginalCudaKernels

field = read_vti("third_party/AnyEG/data/scalar_field_v1.vti")
graph = AnyEGPipeline().run(field)                 # PyTorch port on the GPU
graph = AnyEGPipeline(kernels=OriginalCudaKernels()).run(field)   # AnyEG's own CUDA code
graph.maxima, graph.saddles, graph.links(), graph.maximum_pairs()
write_vtp("v1.vtp", graph.to_polylines(field))     # same content as AnyEG's EG_*.vtp
```

The parts, each behind an interface so it can be swapped:

| part | file | what it does |
|---|---|---|
| `ScalarField` | `anyeg/field.py` | float32 values on a grid (VTK numbering) or at points |
| `read_vti`, `write_vtp`, ... | `anyeg/vtk_io.py` | VTK XML files without the VTK library |
| `NeighborhoodBuilder` | `anyeg/neighborhood.py` | TTK's 14-neighbour grid (AnyEG's), 6-neighbour grid, kNN |
| `ExtremumGraphKernels` | `anyeg/kernels/` | AnyEG's four GPU modules: `OriginalCudaKernels` (its CUDA code) or `TorchKernels` (port) |
| `ArcAssembler` | `anyeg/assembly.py` | port of AnyEG's CPU post-processing: arcs, persistence range, edge bundling |
| `AnyEGPipeline` | `anyeg/pipeline.py` | wires the three together |

AnyEG's `CMakeLists.txt` is a TTK module fragment and does not build on its
own. `build.sh` compiles `cudaHelper.cu` unchanged, except one include path
that points into the authors' home directory, together with a small driver
(`anyeg/native/anyeg_driver.cu`) in place of the TTK filter. The PyTorch port
gives exactly the original's output on all 27 test cases.

Main differences from our pipeline (details and numbers in `observations.md`,
2026-11-11):

- AnyEG does not break ties, so on 8-bit volumes every voxel of a flat region
  is a maximum (243,266 on `fuel`), even after its own preprocessing.
- With distinct values and the same 14-neighbour grid, maxima and basins are
  identical. AnyEG keeps only true saddles (upper link in two or more pieces)
  and has no persistence simplification. We link all touching basins and
  simplify afterwards.
- Our 6-neighbour `grid_graph` cuts diagonal ridges into extra maxima (6,184
  vs 938 on `hydrogen_atom`). AnyEG, TTK and TACHYON use 14 neighbours.

## 2D spine layout

```python
from pipeline.spine_layout import spine_layout

layout = spine_layout(eg, coords=ds.points)   # or coords=volume.coords, spacing=volume.spacing
layout.peak_xy, layout.saddle_xy              # (P, 2), (S, 2), fitted into [-1, 1]
layout.stress, layout.pca_stress              # 0 = distances kept exactly
layout.crossings                              # pairs of links that cross
```

Peaks and saddles are projected with PCA, then moved by stress majorisation
(SMACOF, weights 1/d²) so that on-screen distances match distances in the
original space. The result is never worse than the PCA start. It does
nothing about crossings; `crossings` only counts them. More than 3,000 nodes
is refused (simplify first).

## Exporting to JSON

```bash
python3 -m pipeline.export_json cyclic_3d -k 20 --simplify 0.04 --embed pca tsne -o out/cyclic_3d.json
python3 -m pipeline.export_json --volume fuel --simplify 0.1 -o out/fuel.json
```

```python
from pipeline.export_json import export_extremum_graph, points_data

export_extremum_graph(eg, "out/g.json", coords=ds.points, meta={"name": ds.name},
                      full_peak_persistence=raw.peak_persistence,
                      data=points_data(ds.values, embedding_2d))
```

One file per field: `meta`, `peaks`, `saddles` (with ids, vertex, value,
persistence, position in the original space), `spine` (2D layout, left out
with `--no-spine`), `region` (peak id of every
point), `persistence_curve` (for the persistence plot) and `data` (values and
one 2D embedding per chosen reducer for point clouds, `data.embeddings`; shape, spacing and values for
volumes). The global maximum's persistence is `null`. The full schema is in
the docstring of `pipeline/export_json.py`. `--no-region` and `--no-data`
shrink big files.

## Web viewer

```bash
python3 web/build_data.py              # writes web/data/*.json (once, or after pipeline changes)
python3 web/build_data.py --coords-only  # only the point clouds' coordinates (parallel coordinates)
python3 -m http.server -d web 8000     # then open http://localhost:8000/#fuel
```

`web/` is plain ES modules with Three.js (r160) vendored in `web/vendor/`, so
it needs no build step and no network. It must be served over HTTP (not opened
as a `file://` page).

- **Mode rail** (the toolbar on the left): *Explore* (spine + data view),
  *Compare* (two spines, see below; key `m`) and *Parallel* (spine + parallel
  coordinates; key `p`). The parallel mode is remembered in the browser;
  compare mode is in the link.
- **Parallel coordinates** (`src/parallel.js` data model, `src/parallelView.js`
  drawing): every point is a polyline across one axis per *original* dimension
  (x1..xd of a point cloud, x, y, z of a grid voxel), the field value, and a
  "peak" axis with one band per visible peak (highest at the top; inside a band
  a point sits by its value within that region). No dimensionality reduction
  is involved. Lines are coloured by peak region or value and drawn with
  WebGL (`LineSegments`), so 100k points x 12 axes and 262k voxels stay
  interactive; axes, labels and brushes are a 2D canvas on top. Drag along an
  axis to brush a range (brushes on several axes combine); the strip above the
  plot counts the points that pass and lists their peaks (click one to select
  it in the spine). Hovering or selecting a peak or saddle in the spine colours
  only its region's lines; clicking a line selects its peak. Click an axis name
  to flip it, drag it to move it. The "Axes" menu shows or hides any axis (at
  least two stay; "Only f and peak" hides the coordinates; an axis shown again
  returns next to its natural neighbours, and hiding one drops its brush).
  Point clouds need `<name>.coords.json` (the
  original coordinates, 16-bit per axis, written by `web/build_data.py` and
  loaded only when the mode is opened); grids derive x, y, z from the shape.
- **Spine** (`src/spineView.js`, `src/contours.js`): three drawing styles, chosen
  in the side panel. *Contours (NDDAV)* (default) draws the spine the way LLNL's
  [NDDAV](https://github.com/LLNL/NDDAV) does (`hdanalysis/core/TopoSpines.py`):
  above its saddles a peak's contours are circles centred on it; below a saddle
  the circle stretches along the link to a circle at the saddle whose radius is
  how much the region has grown since the saddle value (0 at the saddle's own
  value, the figure-eight). Isovalues are evenly spaced ("Even isovalues", 10 by
  default) plus every shown saddle value, an even level being skipped when it
  lies within about one spacing of a kept value. Every level is opaque with a
  dark outline, contour radius ~ (points)^(2/d) and the outer halo sets the size.
  The halo is as wide as the links are long. The "BrBG" colour map gives NDDAV's
  brown-to-teal colours. Choosing a style resets "Radius" and "Scale by" to that
  style's defaults. On volumes where most voxels are background (`neghip`, `fuel`)
  the halo dominates; the paper style below handles those better.
  *Contours (as in the paper)* follows the paper's section 4. Each
  link is the Minkowski sum of circles along the peak-saddle arc (drawn as the
  exact envelope); the circle at a point p has radius proportional to the number
  of points of the peak's whole region with value between the isovalue c and
  f(p) (section 4.2, f linear along the arc), so the contour through a saddle
  pinches to a figure-eight there. Isovalues are the values of the shown peaks
  and saddles plus the field minimum (the outer halo), no two closer than the
  "Min. isovalue gap" slider (section 4.4); every peak uses the same values.
  Low isovalues are drawn paler, standing in for the paper's transfer function.
  "Radius" switches between radius ~ volume (the paper) and radius ~ sqrt(volume)
  (area ~ volume), which keeps small regions visible when the outer halo holds
  most of the points. "Scale by" chooses which contour gets the full size: by
  default the widest contour at the lowest shown saddle (the outermost one that
  still shows how peaks connect), with the halo below it capped at 2x that
  radius; "outer halo" gives the halo the full size instead and caps nothing.
  The paper does not fix this scale. In `fuel`, 83 % of the voxels are
  background in one region, so scaling by the halo leaves the structure under
  a pixel wide. It needs `data.values` and `region` in the file
  (all exported files have them). *Discs and lines* is the plain version: discs
  (area ~ points in the region, colour = peak value), diamonds for saddles
  (orange if they close a loop), lines for links. Drag to pan, wheel to zoom,
  double-click to reset, click a node to select it. A "Contour size" slider
  scales all radii (useful when one huge region makes the rest thin).
- **Data view** (`src/dataView.js`), point clouds only: a scatterplot of the
  points. The **dimensionality reduction** menu (PCA, random projection, Isomap,
  t-SNE; `EMBEDS` in `web/build_data.py`) chooses where the points are drawn. It
  is for drawing only: the extremum graph is always built in the original
  space, because building it on reduced data gives false peaks
  (`observations.md`, 2026-11-09). Points are coloured by their peak's region
  or by value. Selecting a peak or saddle in the spine highlights its region;
  clicking a point selects its peak. Reductions are precomputed per file, so no
  server is needed.
- **Data view for grids** (`src/volumeView.js`): direct volume rendering. A
  fragment shader marches rays through a 3D texture of the field (the layout of
  three.js's `webgl2_materials_texture3d` example), with a second 3D texture
  holding each voxel's region colour and a weight. Controls: "Show above"
  (lowest value shown, fading in over 0.15), opacity, shading, colour by region
  (a distinct colour per peak, also used for the peak dots in the spine and the
  scatterplot) or by field value. Selecting a peak or saddle in the spine fades
  every other region, so the selected one shows inside the volume; clicking the
  volume selects the peak under the pointer (the ray is traced on the CPU with
  the same opacity function as the shader). Dots mark the visible peaks. Drag to
  rotate, wheel to zoom, double-click to reset. The slider and p_lo/p_hi update
  the region colours, so peaks that were simplified away merge into their parent.
  The drawing needs WebGL 2. It was checked in software WebGL (see below), not on
  a hardware GPU.
- **Two thresholds, as in the paper** (section 3.4): the *noise* threshold p_lo
  hides peaks (and their links) below it; the *variation* threshold p_hi cuts
  links whose saddle persistence is above it, so deep valley-like passes stop
  joining peaks, and the graph may fall into several pieces. p_hi never removes
  peaks. The plot shows the paper's two curves (Fig. 8b): blue = peaks left at
  each p_lo, red = links left at each p_hi. Click sets p_lo, shift+click sets
  p_hi. On the Ackley lattice in 3D a p_hi between 0.12 and 0.21 removes exactly
  the diagonal links that the paper's graph does not have (`observations.md`).
- **Saturated persistence p\*** (section 3.4, equation 3), the checkbox "Keep
  the arcs of merged peaks" (on by default): a simplified saddle keeps its two
  arcs while p* >= p_lo (and p <= p_hi), so a link to a peak that absorbed
  smaller ones runs along the ridge through them (small grey dots) instead of
  jumping straight to it, and a ring of peaks stays a ring after they merge
  (the volcano rim of Fig. 5b). Selecting a node highlights the whole ridge.
  Off: straight links between the surviving peaks. In the contour style a lobe
  whose saddle's own peak was merged away follows the same ridge. Ridges of
  saddles that `simplify` drops (a second link between the same two peaks, or
  a loop back onto one peak) are drawn as thin lines only.
- **Hover link and colour legend**: hovering a peak in the spine highlights its
  region in the volume or scatterplot (a saddle: both regions it joins) and dims
  the rest; leaving restores the selection. A legend in the data view shows the
  colours: *field value* uses viridis over the file's value range (as the spine);
  *peak region* gives each visible peak a distinct colour (its dot in the spine).
  For volumes it also shows the opacity: transparent below "Show above", fading
  in over the next 0.15 of value (`RAMP` in `volumeView.js`).
- **Panel toggles**: the "Data view" and "Controls" buttons in the header (keys
  `d` and `c`) hide or show the data/volume view (or the parallel coordinates)
  and the side panel; the spine takes the freed room. The choice is remembered
  in the browser.
- **Slider and persistence plot** (`src/persistencePlot.js`): the threshold
  hides peaks below it. This is done in the browser (`src/simplify.js`, a port
  of `ExtremumGraph.simplify`), and the layout in the file never changes, so
  nodes do not move when the slider does. Thresholds below a file's base
  threshold (`meta.simplify_threshold`) cannot be restored; the plot greys
  that part out.
- **Compare mode** ("Compare" in the mode rail, key `m`): two spines side by side, each
  with its own "Method" menu: Ours, AnyEG, and Ours on the 14-neighbour grid
  (volumes only). Every registered variant in `comparison/variants.py` is
  exported by `web/build_data.py` as `<name>.<variant>.json`. All variants of a
  field share one 2D frame, pan and zoom, and region colours, so a maximum found
  by both methods is drawn at the same spot and in the same colour.
  - Selecting a node selects the node at the same vertex in the other spine.
  - Peaks that the other method does not find at all get a magenta ring.
  - AnyEG has no simplification, so its full graph is always shown and the
    sliders act only on the other panel.
  - The comparison box (side panel) gives the counts shown now, the agreement
    of the full graphs, and "Why they differ": the design choices on which the
    two methods differ. Each variant declares these as `traits`.
  - The data view follows the left panel; "⇄ swap" exchanges the methods.
  - AnyEG runs on values made distinct (rank in (value, index) order). Its
    counts on the raw 8-bit values are listed in the box.
  - Links: `#fuel:ours:anyeg` (compare), `#fuel:anyeg` (one method).
  - Design and reasons: `docs/extremum_graph_variants.md`.
- **Not yet:** crossing reduction (5) and the spine-anchored scatterplot (6).

Tests:

```bash
node --import ./web/tests/register.mjs --test web/tests   # needs Node 20+
python3 web/tests/make_fixtures.py                        # regenerate Python-vs-JS simplify cases
```

Without Node, serve `web/` and open `tests/browser_checks.html` (logic
checks of the contour geometry, both data views and the parallel coordinates
with a fake renderer, plus the
Python-versus-JS comparison of `simplify` for p_lo and p_hi and of the p* arcs; every line should
say PASS). `tests/contours_preview.html?data=neghip&t=0.1&s=0.6`
draws the contour geometry with Canvas 2D, which shows the paper-style spine
even where WebGL is unavailable, and
`tests/volume_preview.html?data=fuel&t=0.1&sel=0` ray-marches the volume on the
CPU with the shader's algorithm (same camera, opacity, colours and weights).
The GLSL itself is not compiled by either; to see the real thing on a machine
without a GPU, run the viewer in headless Chromium with software WebGL (Playwright
plus its bundled Chromium; `pip install playwright`, `python3 -m playwright install
chromium`, then launch with `--use-gl=angle --use-angle=swiftshader
--enable-unsafe-swiftshader`). The shader and the click-to-select link were checked
that way on 2026-11-10 (no console errors).

The JS tests check `simplify.js` against what Python returns (4 graphs x 7
thresholds, and 6 p_lo/p_hi pairs each) and run the spine view against a fake renderer (instance
matrices, link buffers, picking, selection, pan and zoom). Nothing tests the
actual WebGL drawing; look at it in a browser.

## References

1. C. Correa, P. Lindstrom, P.-T. Bremer. *Topological Spines: A
   Structure-Preserving Visual Representation of Scalar Fields.* IEEE TVCG,
   2011. <https://doi.org/10.1109/TVCG.2011.244>
2. A. Ande, V. Subhash, V. Natarajan. *TACHYON: Efficient Shared Memory
   Parallel Computation of Extremum Graphs.* Computer Graphics Forum, 2023.
   <https://vgl.csa.iisc.ac.in/softwares.php?pid=013>
3. Topology ToolKit (TTK). <https://topology-tool-kit.github.io/>
4. AnyEG: fast extremum graph computation for large-scale arbitrary grids
   (BuptCIAGroup). <https://github.com/BuptCIAGroup/AnyEG>, commit 08068c2
