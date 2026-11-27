"""
Build the JSON files the web viewer loads (``web/data/*.json`` + ``index.json``).

Run from the repo root:

    python3 web/build_data.py               # all entries
    python3 web/build_data.py fuel neghip   # only these

Every file is the graph at a *base* threshold (low, so that the browser's
persistence slider has something to remove) with its 2D layout computed once
on that base graph. Raising the slider in the browser only hides nodes, so
the layout never moves.

Besides our own graph (``<name>.json``, which also holds the field data), every
registered extraction variant (``comparison/variants.py``: AnyEG, ours on the
14-neighbour grid, ...) is written as ``<name>.<variant>.json`` for the viewer's
compare mode, in the same 2D frame. ``--no-variants`` skips them (and keeps the
variants already listed in index.json).

Point clouds also get ``<name>.coords.json``: the original coordinates of every
point, for the parallel-coordinates mode (loaded only when that mode is opened).
``--coords-only`` writes just these files and leaves the graphs as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from comparison.sources import FieldSource  # noqa: E402
from comparison.variants import TRAIT_LABELS  # noqa: E402
from comparison.viewer_export import export_variants  # noqa: E402
from pipeline.export_json import build_document, coords_document, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"

# Reducers offered in the viewer for point clouds. They only place points in the
# scatterplot; the extremum graph is always built in the original space.
EMBEDS = ["pca", "random_projection", "isomap", "tsne"]

# name -> arguments of build_document, plus a label and a one-line description.
# No ``simplify`` floor: every file keeps the full graph, so the viewer's persistence
# slider can go all the way down to 0.
ENTRIES = {
    "communities_crime": dict(
        label="communities_crime (1,994 US communities, violent crime, 100D)", dataset="communities_crime", k=15,
        initial_threshold=0.25, embed=EMBEDS,
        text="UCI Communities and Crime: 1,994 US communities described by 100 census attributes (income, family, housing, immigration, ...). f = violent crimes per capita. Peaks: community profiles with high violent crime; saddles: how two such profiles connect. Clustering -> Known labels: Census region, state."),
    "pbmc3k_types": dict(
        label="pbmc3k_types (2,638 blood cells with cell types, 20 PCs)", dataset="pbmc3k_types", k=15,
        initial_threshold=0.04, embed=EMBEDS,
        text="Blood cells (10x PBMC 3k). f = density of cells. Peaks: cell types; saddles: how two types grade into each other; persistence: how distinct a type is. Clustering -> Known labels: reference cell types."),
    "alanine_dipeptide": dict(
        label="alanine_dipeptide (molecule shapes from simulation, 45D)", dataset="alanine_dipeptide", k=15,
        initial_threshold=0.08, embed=EMBEDS,
        text="Shapes of a small peptide from molecular dynamics. Peaks: stable shapes; saddles: transitions between them; persistence: how hard a shape is to leave. Scalar (f): density, or -free energy from the backbone angles (kT). Known labels: Ramachandran regions."),
    "bc_tumours": dict(
        label="bc_tumours (529 TCGA tumours, 1000 genes; switch the scalar f)", dataset="bc_tumours", k=15,
        initial_threshold=0.04, embed=EMBEDS,
        text="TCGA breast tumours on one fixed neighbourhood (1000 genes, k = 15). Scalar (f) switches what a "
             "peak means: density, deviation from normal, proliferation (G2M), estrogen response, hypoxia. "
             "Clustering -> Known labels: PAM50, ER, tissue source site, plate."),
    "bc_dsga": dict(
        label="bc_dsga (tumours, deviation from normal, 1000D)", dataset="bc_dsga", k=15,
        initial_threshold=0.04, embed=EMBEDS,
        text="BC-TCGA tumours. f = how far each tumour deviates from normal breast tissue (DSGA disease component, smoothed over neighbours, scaled to 0..1). Peaks: the most deviant tumour types."),
    "nki": dict(
        label="nki (breast cancer genes, NKI cohort, 1001D)", dataset="nki", k=15,
        initial_threshold=0.05, embed=EMBEDS,
        text="NKI breast cancer cohort (as in the thesis): 337 tumours, 1000 most variable genes. f = density of samples; Clustering -> Known labels: ER status, grade, metastasis, death."),
    "bc_tcga": dict(
        label="bc_tcga (breast cancer genes, 1003D)", dataset="bc_tcga", k=15,
        initial_threshold=0.05, embed=EMBEDS,
        text="TCGA breast cancer: 529 tumour and 61 normal samples, 1000 most variable genes. f = density of samples; Clustering -> Known labels shows tumour / normal."),
    "flower_2d": dict(
        label="flower_2d (point cloud, 2D)", dataset="flower_2d", k=20,
        initial_threshold=0.05, embed=EMBEDS,
        text="A centre peak with 10 petals; strong star plus a weak ring."),
    "flower_10d": dict(
        label="flower_10d (point cloud, 10D)", dataset="flower_10d", k=20,
        initial_threshold=0.05, embed=EMBEDS,
        text="The same flower hidden in 10 dimensions."),
    "demo_300d": dict(
        label="demo_300d (point cloud, 300D)", dataset="demo_300d", k=20,
        initial_threshold=0.05, embed=EMBEDS,
        text="5 peaks in a hidden 3-D space, embedded in 300 dimensions."),
    "cyclic_3d": dict(
        label="cyclic_3d (point cloud, 3D)", dataset="cyclic_3d", k=20,
        initial_threshold=0.04, embed=EMBEDS,
        text="Six peaks in a ring; the answer is a 6-cycle."),
    "cyclic_10d": dict(
        label="cyclic_10d (point cloud, 10D)", dataset="cyclic_10d", k=20,
        initial_threshold=0.01, embed=EMBEDS,
        text="The same ring hidden in 10 dimensions."),
    "ext_10d": dict(
        label="ext_10d (point cloud, 10D)", dataset="ext_10d", k=20,
        initial_threshold=0.05, embed=EMBEDS,
        text="Many bumps in 10D; most saddles close loops."),
    "pbmc3k": dict(
        label="pbmc3k (scRNA-seq, 50D PCA)", dataset="pbmc3k", k=15,
        initial_threshold=0.05, embed=EMBEDS,
        text="2,700 blood cells (10x PBMC 3k) in 50 PCA dimensions; the scalar is kNN density, so peaks are cell populations."),
    "loss_cnn_m4_e30": dict(
        label="loss_cnn_m4_e30 (CIFAR-10 loss landscape, 4 independent CNNs, 3D)", dataset="loss_cnn_m4_e30", k=15,
        initial_threshold=0.02, embed=["pca"],
        text="Training loss of 4 independently trained CNNs in the 3-D subspace through their weights (32,768 samples). "
             "Four deep minima, one per model, joined at about chance-level loss (2.2-2.4). Peaks are the MINIMA of the training loss (f = -log loss, scaled 0..1); saddles: the lowest passes between them (barriers); Clustering -> Known labels colours each sample by the trained model it is closest to, to compare with the peak regions. Scalar: switch to accuracy on the evaluation images. Data view: use Scatter (axes u1, u2 = subspace axes) or Parallel; a PCA of a uniformly sampled box shows little."),
    "loss_cnn_m4_e30_spawn2": dict(
        label="loss_cnn_m4_e30_spawn2 (CIFAR-10 loss landscape, 4 CNNs from one parent, 3D)", dataset="loss_cnn_m4_e30_spawn2",
        k=15, initial_threshold=0.01, embed=["pca"],
        text="4 CNNs trained on from one 2-epoch parent: four shallow minima joined by low saddles (loss 0.44 vs minima "
             "0.40-0.42). Peaks are the MINIMA of the training loss (f = -log loss, scaled 0..1); saddles: the lowest passes between them (barriers); Clustering -> Known labels colours each sample by the trained model it is closest to, to compare with the peak regions. Scalar: switch to accuracy on the evaluation images. Data view: use Scatter (axes u1, u2 = subspace axes) or Parallel; a PCA of a uniformly sampled box shows little."),
    "loss_cnn_m6_e30": dict(
        label="loss_cnn_m6_e30 (CIFAR-10 loss landscape, 6 independent CNNs, 5D)", dataset="loss_cnn_m6_e30", k=30,
        initial_threshold=0.02, embed=EMBEDS,
        text="6 independent CNNs, 5-D subspace (65,536 samples, k = 30). Six minima; a 7th weak peak (persistence 0.007, "
             "hidden at the start) is a sampling artefact: gradient descent from it ends at model 2. Peaks are the MINIMA of the training loss (f = -log loss, scaled 0..1); saddles: the lowest passes between them (barriers); Clustering -> Known labels colours each sample by the trained model it is closest to, to compare with the peak regions. Scalar: switch to accuracy on the evaluation images. Data view: use Scatter (axes u1, u2 = subspace axes) or Parallel; a PCA of a uniformly sampled box shows little."),
    "loss_cnn_m6_e30_spawn2": dict(
        label="loss_cnn_m6_e30_spawn2 (CIFAR-10 loss landscape, 6 CNNs from one parent, 5D)", dataset="loss_cnn_m6_e30_spawn2",
        k=30, initial_threshold=0.005, embed=EMBEDS,
        text="6 CNNs from one parent, 5-D subspace (65,536 samples, k = 30): six shallow minima, every pair linked "
             "(15 saddles). Peaks are the MINIMA of the training loss (f = -log loss, scaled 0..1); saddles: the lowest passes between them (barriers); Clustering -> Known labels colours each sample by the trained model it is closest to, to compare with the peak regions. Scalar: switch to accuracy on the evaluation images. Data view: use Scatter (axes u1, u2 = subspace axes) or Parallel; a PCA of a uniformly sampled box shows little."),
    "two_peaks_2d": dict(
        label="two_peaks_2d (2D, simplest case)", dataset="two_peaks_2d", k=15,
        initial_threshold=0.02, embed=["pca"],
        text="Two Gaussian peaks and one saddle between them: the spine is a single path, peak - saddle - peak."),
    "volcano_intact": dict(
        label="volcano_intact (paper Fig. 5, 2D)", dataset="volcano_intact", k=15,
        initial_threshold=0.02, embed=["pca"],
        text="Crater with 6 hills on its rim; every rim saddle is a ridge, so the spine is a ring."),
    "volcano_notched": dict(
        label="volcano_notched (paper Fig. 5, 2D)", dataset="volcano_notched", k=15,
        initial_threshold=0.02, embed=["pca"],
        text="One rim saddle (30°) notched but still closer to the hills than to the crater floor: "
             "the paper keeps the ring. A p_hi below 0.28 wrongly opens it."),
    "volcano_breached": dict(
        label="volcano_breached (paper Fig. 5, 2D)", dataset="volcano_breached", k=15,
        initial_threshold=0.02, embed=["pca"],
        text="The rim is breached at 30°, almost down to the crater floor: the paper cuts that saddle and "
             "the ring becomes a chain. Ours still draws a ring until p_hi is set below 0.51 (try 0.3)."),
    "fuel": dict(
        label="fuel (volume 64³)", volume="fuel",
        initial_threshold=0.1,
        text="A jet: one elongated structure with a few maxima."),
    "neghip": dict(
        label="neghip (volume 64³)", volume="neghip",
        initial_threshold=0.1,
        text="Electron density around a protein: mirror-symmetric blobs."),
    "bucky": dict(
        label="bucky (C60 electron density, 32³, paper)", volume="bucky",
        initial_threshold=0.2,
        text="Buckminster-fullerene: one peak per carbon atom (60) in a hollow cage; the paper's Bucky Ball."),
    "silicium": dict(
        label="silicium (volume 98x34x34)", volume="silicium",
        initial_threshold=0.1,
        text="One peak per atom on a lattice; mostly loops."),
}


def write_coords(name: str, spec: dict) -> str | None:
    """Write ``<name>.coords.json`` for a point-cloud entry; returns its file name.

    Checks that the generator still gives the field values stored in ``<name>.json``,
    so the coordinates line up with the regions in that file.
    """
    if "dataset" not in spec:
        return None                                   # grids: the browser derives x, y, z
    import numpy as np
    from pipeline.data_loader import load_dataset

    ds = load_dataset(spec["dataset"])
    main_file = OUT / f"{name}.json"
    if main_file.exists():
        stored = np.asarray(json.loads(main_file.read_text())["data"]["values"])
        if stored.shape != np.shape(ds.values) or np.abs(stored - np.asarray(ds.values)).max() > 1e-3:
            raise SystemExit(f"{name}: the dataset no longer matches {main_file.name}; rebuild it first")
    # optional, from truth.json: gene / feature names, and known labels of the points
    path = write_json(coords_document(ds.points, names=ds.truth.get("dimension_names"),
                                      classes=ds.truth.get("point_classes")),
                      OUT / f"{name}.coords.json")
    return path.name


def classes_name(spec: dict) -> str | None:
    """Name of the known labels of a point-cloud entry (truth.json ``point_classes``), or None."""
    if "dataset" not in spec:
        return None
    from pipeline.data_loader import load_dataset
    c = load_dataset(spec["dataset"]).truth.get("point_classes")
    if not c:
        return None
    sets = c if isinstance(c, list) else [c]
    return ", ".join(str(s.get("name", "class")) for s in sets)


def scalar_menu(spec: dict) -> dict | None:
    """The viewer's Scalar menu for an entry with extra functions (truth.json "scalars"):
    {"main": "<id>_s<smooth>", "choices": [{"id", "label"}]}, or None."""
    if "dataset" not in spec:
        return None
    from pipeline.data_loader import load_dataset
    sc = load_dataset(spec["dataset"]).truth.get("scalars")
    if not sc:
        return None
    choices = []
    for f in sc["functions"]:
        for s in sc.get("smooths", [0]):
            choices.append({"id": f"{f['id']}_s{s}",
                            "label": f["label"] + (f" (smoothed x{s})" if s else " (unsmoothed)")})
    return {"main": f"{sc['main']['id']}_s{sc['main']['smooth']}", "choices": choices}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help=f"entries to build (default: all of {list(ENTRIES)})")
    ap.add_argument("--no-variants", action="store_true",
                    help="only our own graph; keep the variant files listed in index.json")
    ap.add_argument("--coords-only", action="store_true",
                    help="only write <name>.coords.json for point clouds (parallel coordinates)")
    args = ap.parse_args()
    names = args.names or list(ENTRIES)
    unknown = [n for n in names if n not in ENTRIES]
    if unknown:
        ap.error(f"unknown entries {unknown}")

    index_path = OUT / "index.json"
    index = ({d["id"]: d for d in json.loads(index_path.read_text())["datasets"]}
             if index_path.exists() else {})
    for name in names:
        spec = dict(ENTRIES[name])
        label, text = spec.pop("label"), spec.pop("text")
        if args.coords_only:
            if name in index:
                index[name]["coords_file"] = write_coords(name, spec)
                index[name]["classes"] = classes_name(spec)
                print(f"{name:11s} coords: {index[name]['coords_file'] or '(grid, none needed)'}")
            continue
        t0 = time.time()
        doc = build_document(**spec)
        old = index.get(name, {})
        variants, comparisons = old.get("variants", []), old.get("comparisons", [])
        if not args.no_variants:
            source = (FieldSource.volume(spec["volume"]) if "volume" in spec
                      else FieldSource.dataset(spec["dataset"], spec["k"]))
            variants, comparisons = export_variants(
                source, doc, f"{name}.json", write=lambda d, f: write_json(d, OUT / f))
        path = write_json(doc, OUT / f"{name}.json")
        index[name] = {"file": path.name, "label": label, "text": text,
                       "peaks": len(doc["peaks"]), "saddles": len(doc["saddles"]),
                       "spine_stress": doc.get("spine", {}).get("stress"),
                       "crossings": doc.get("spine", {}).get("crossings"),
                       "variants": variants, "comparisons": comparisons,
                       "k": spec.get("k"),
                       "coords_file": write_coords(name, spec),
                       "classes": classes_name(spec),
                       "scalars": scalar_menu(spec)}
        print(f"{name:11s} {len(doc['peaks']):4d} peaks {len(doc['saddles']):5d} saddles "
              f"{path.stat().st_size / 1e6:6.2f} MB  {time.time() - t0:5.1f}s")

    order = [n for n in ENTRIES if n in index]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(
        {"trait_labels": TRAIT_LABELS, "datasets": [{**index[n], "id": n} for n in order]}, indent=1))


if __name__ == "__main__":
    main()
