# Extremum-graph variants and the compare mode: design choices

This note explains how the viewer compares two ways of computing an extremum
graph (our pipeline and AnyEG today, more later), and why it is built this way.
The code is in `comparison/` (Python) and `web/src/compare.js`,
`comparePanel.js`, `comparisonBox.js` (viewer).

## The two decisions

1. **A method without simplification is shown as it is.** AnyEG computes no
   persistence, so the viewer draws its full graph and the persistence sliders
   do not touch it. We do not compute a persistence for it with our own merge
   sweep, because that would mix our method into AnyEG's result. Instead the
   comparison box says in a few lines *why* the two graphs differ (see
   "Traits" below).
2. **Every method runs on values made distinct.** This is the main subject of
   this note, next.

## Decision 2: distinct values

### The problem

Many of our volumes store 8-bit values, so neighbouring voxels often have
exactly the same value. A method has to decide what "higher" means between
two equal values:

* **Ours** breaks ties by vertex index: every vertex gets a rank in the order
  (value, index), and "higher" means "higher rank". All values become distinct.
* **AnyEG** breaks no ties. A vertex is a maximum when no neighbour is
  *strictly* higher, so every voxel of a flat region is a maximum, and no saddle
  joins two equal neighbours.

On the raw stored values AnyEG therefore reports 243,266 maxima on `fuel` (93 %
of the voxels), 124,320 on `neghip` and 36,320 on `silicium`. Most of them are
isolated, so they cannot be simplified away, and the spine layout accepts at
most 3,000 nodes. AnyEG's own fix, `preprocess_vti.py`, adds `1e-8 * rank` and
then stores the result as float32. Above about 0.1 that step is below float32
resolution, so ties survive: still 3,904 maxima on `neghip` and 156 on `fuel`,
against 175 and 65 with distinct values.

### What we do

Before AnyEG runs, every value is replaced by its rank in the (value, index)
order (`FieldSource.order`). The ranks go to AnyEG as float32, which is exact up
to 2^24 vertices (a check refuses larger fields).

**Only the order is changed, never the heights.** Peak values, the saddle
"value gap", contour sizes, colours and every number shown in the viewer use
the original values. The ranks only decide the comparisons inside the method.

### Why this is faithful to AnyEG

* Every decision AnyEG makes is a comparison: "is this neighbour strictly
  higher?", and "which neighbour has the largest rise f(n) − f(v)?". For a
  fixed v the largest rise is simply the highest neighbour. So any transform
  that keeps the order gives exactly the same decisions wherever the values
  were already distinct.
* Where values were equal, AnyEG had no rule. We supply the same rule as our
  pipeline (lower index = lower). This is what AnyEG's own preprocessing
  intends, done without the float32 precision loss.
* Ranks are whole numbers, so the float32 subtraction inside AnyEG is exact. On
  raw floats, two different rises could round to the same float32 number.

The PyTorch port of AnyEG is checked against AnyEG's own CUDA code on these
inputs (`python3 -m comparison`, `tests/test_anyeg.py`). It gives identical
results.

### What this buys for the comparison

Both methods now see the same strict order. When they also use the same
neighbourhood, they find exactly the same maxima and basins: AnyEG vs "Ours,
14-neighbour grid" has a maxima Jaccard of 1.0 on every volume. So whatever
differs in the compare mode comes from the algorithms (saddles, links,
simplification), not from tie handling. Tie handling is not hidden. It is
reported as a fact in the comparison box ("on the stored values it would report
243,266 maxima; 156 after its own preprocessing script"). Those two numbers are
computed only when the stored values actually have ties
(`AnyEGVariant.extract`, `notes`).

### Rule for future variants

`FieldSource` gives every variant both `values` (the real heights) and `order`
(distinct ranks):

* a method that **breaks ties itself** uses `values`;
* a method that **does not** uses `order` for its comparisons and `values` for
  everything it reports;
* either way it states its tie handling in its `ties` trait, so the comparison
  box shows it.

## How the pieces fit

```
FieldSource (comparison/sources.py)       one field, loaded once: values, order,
        |                                 coordinates, our neighbour graph
        v
ExtractionVariant.extract (variants.py)   one per method, registered by id
        |   -> VariantResult: the graph in OUR ExtremumGraph format,
        |      a CommonGraph (for agreement numbers), notes, extra saddle fields
        v
export_variants (viewer_export.py)        lays out each variant in one shared frame,
        |                                 writes <name>.<variant>.json, and the
        |                                 agreement of every pair for index.json
        v
viewer: compare.js (pure helpers), ComparePanel (second spine),
        ComparisonBox (the explanation), main.js (wiring)
```

### Choices, and why

* **Adapters into our format (`anyeg_to_extremum_graph`).** Every variant is
  converted into our `ExtremumGraph`. The JSON export, the spine layout and the
  whole viewer then work unchanged. Nothing of ours is added in the conversion:
  AnyEG's peaks get "no persistence" (`null`) and no parents, and its saddle
  number is AnyEG's own |f(saddle) − f(maximum of its basin)|.
* **A flag instead of fake numbers.** `meta.variant.simplifiable = false` tells
  the viewer to draw everything (`showAll` in `simplify.js`). It stops the
  viewer from merging peaks or dropping duplicate saddles, which our
  simplification would otherwise do.
* **Saddles that reach three or more maxima.** AnyEG links a saddle to every
  basin around it. Our format joins exactly two peaks per saddle, so such a
  saddle is stored as k − 1 entries at the same vertex, and `reaches` keeps the
  full list. Drawn, it is the same k lines from one point. The info panel lists
  all k maxima.
* **One frame for all methods (`PositionRegistry`, `anchored_layout`).** Our
  main file is laid out as before. Each further variant keeps the position of
  every vertex that is already placed and lays out only its new nodes (stress
  majorisation with the placed nodes held fixed). The first placement of a
  vertex wins, so a maximum or saddle found by several methods is at the same
  spot in all of them. The two spines also share pan and zoom, and a shared
  maximum gets the same colour on both sides.
* **Matching by vertex, not by id.** Peak ids are per file. "The same
  maximum" means "the same sample point", so selection links, rings and the
  counts in the box all compare vertex ids.
* **Rings mean "not found at all".** A peak is ringed when its vertex is not a
  maximum anywhere in the other method's *full* graph (`meta.all_maxima`). A
  peak that the other side merely hides at the current threshold is not ringed.
* **Traits explain the difference.** Each variant states its design choices
  in short sentences: neighbourhood, equal values, saddles, links, loop saddles
  and simplification. The box lists the traits on which the two shown variants
  differ. A new variant therefore explains itself without any viewer code.
* **Field data stored once.** Only the main file holds the values and
  embeddings; variant files borrow them (`meta.data_file`).
* **Agreement numbers come from the full graphs**, computed at build time
  (`comparisons` in `index.json`). The "shown now" numbers come from what is
  drawn at the current threshold.

### Limits

* The spine layout takes at most 3,000 nodes. AnyEG on the 10-D kNN graphs
  (`cyclic_10d`, `ext_10d`) produces 27,626 and 83,312 nodes: a kNN graph is not
  a triangulation, so its upper-link test accepts most candidates. Those
  variants are listed as "not available" with that reason.
* The data view (scatterplot or volume) follows the left panel. Use "swap" to
  see the other method's regions there.
* `index.json` and the variant files are rebuilt by `python3 web/build_data.py`.
  `--no-variants` rebuilds only our files and keeps the existing variants.

## Adding a variant

1. Write a subclass of `ExtractionVariant` in `comparison/variants.py`:
   * `traits(source)`: one sentence for each key of `TRAIT_LABELS`;
   * `extract(source)`: return a `VariantResult` (convert your result into an
     `ExtremumGraph`, the way `anyeg_to_extremum_graph` does);
   * set `simplifiable = False` if the method has no persistence;
   * override `applies_to` if the method only works on grids or only on point
     clouds.
2. `register(YourVariant())` at the bottom of the file.
3. `python3 web/build_data.py`, then open the viewer, press **Compare** (or
   `m`) and pick it in either menu.
