"""
Build the JSON files the web viewer loads (``web/data/*.json`` + ``index.json``).

Run from the repo root:

    python3 web/build_data.py               # all entries
    python3 web/build_data.py fuel neghip   # only these

Every file is the graph at a *base* threshold (low, so that the browser's
persistence slider has something to remove) with its 2D layout computed once
on that base graph. Raising the slider in the browser only hides nodes, so
the layout never moves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.export_json import build_document, write_json  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"

# Reducers offered in the viewer for point clouds. They only place points in the
# scatterplot; the extremum graph is always built in the original space.
EMBEDS = ["pca", "random_projection", "isomap", "tsne"]

# name -> arguments of build_document, plus a label and a one-line description.
ENTRIES = {
    "cyclic_3d": dict(
        label="cyclic_3d (point cloud, 3D)", dataset="cyclic_3d", k=20,
        simplify=0.01, initial_threshold=0.04, embed=EMBEDS,
        text="Six peaks in a ring; the answer is a 6-cycle."),
    "cyclic_10d": dict(
        label="cyclic_10d (point cloud, 10D)", dataset="cyclic_10d", k=20,
        simplify=0.005, initial_threshold=0.01, embed=EMBEDS,
        text="The same ring hidden in 10 dimensions."),
    "ext_10d": dict(
        label="ext_10d (point cloud, 10D)", dataset="ext_10d", k=20,
        simplify=0.02, initial_threshold=0.05, embed=EMBEDS,
        text="Many bumps in 10D; most saddles close loops."),
    "volcano_intact": dict(
        label="volcano_intact (paper Fig. 5, 2D)", dataset="volcano_intact", k=15,
        simplify=0.005, initial_threshold=0.02, embed=["pca"],
        text="Crater with 6 hills on its rim; every rim saddle is a ridge, so the spine is a ring."),
    "volcano_notched": dict(
        label="volcano_notched (paper Fig. 5, 2D)", dataset="volcano_notched", k=15,
        simplify=0.005, initial_threshold=0.02, embed=["pca"],
        text="One rim saddle (30°) notched but still closer to the hills than to the crater floor: "
             "the paper keeps the ring. A p_hi below 0.28 wrongly opens it."),
    "volcano_breached": dict(
        label="volcano_breached (paper Fig. 5, 2D)", dataset="volcano_breached", k=15,
        simplify=0.005, initial_threshold=0.02, embed=["pca"],
        text="The rim is breached at 30°, almost down to the crater floor: the paper cuts that saddle and "
             "the ring becomes a chain. Ours still draws a ring until p_hi is set below 0.51 (try 0.3)."),
    "fuel": dict(
        label="fuel (volume 64³)", volume="fuel",
        simplify=0.02, initial_threshold=0.1,
        text="A jet: one elongated structure with a few maxima."),
    "neghip": dict(
        label="neghip (volume 64³)", volume="neghip",
        simplify=0.02, initial_threshold=0.1,
        text="Electron density around a protein: mirror-symmetric blobs."),
    "silicium": dict(
        label="silicium (volume 98x34x34)", volume="silicium",
        simplify=0.05, initial_threshold=0.1,
        text="One peak per atom on a lattice; mostly loops."),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", help=f"entries to build (default: all of {list(ENTRIES)})")
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
        t0 = time.time()
        doc = build_document(**spec)
        path = write_json(doc, OUT / f"{name}.json")
        index[name] = {"file": path.name, "label": label, "text": text,
                       "peaks": len(doc["peaks"]), "saddles": len(doc["saddles"]),
                       "spine_stress": doc.get("spine", {}).get("stress"),
                       "crossings": doc.get("spine", {}).get("crossings")}
        print(f"{name:11s} {len(doc['peaks']):4d} peaks {len(doc['saddles']):5d} saddles "
              f"{path.stat().st_size / 1e6:6.2f} MB  {time.time() - t0:5.1f}s")

    order = [n for n in ENTRIES if n in index]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(
        {"datasets": [{**index[n], "id": n} for n in order]}, indent=1))


if __name__ == "__main__":
    main()
