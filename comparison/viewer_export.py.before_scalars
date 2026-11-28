"""
Writing every extraction variant of a field for the web viewer's compare mode.

For one field the viewer gets:

* the **main file** (``<name>.json``, our pipeline, with the data section),
  built as before by ``pipeline.export_json.build_document``;
* one **variant file** per other variant (``<name>.<variant>.json``): the
  same schema, without the data section (the viewer borrows it from the main
  file), plus ``meta.variant``;
* in ``index.json``, per field: the variants (label, traits, notes) and the
  agreement of every pair of variants on their full graphs.

All files of a field share one 2D frame (``PositionRegistry``): a vertex that
is a node in several variants is drawn at the same place in each, so the two
spines can be compared by eye. The main file's layout is the reference; each
further variant keeps the positions of vertices already placed and lays out
only its new nodes (``pipeline.spine_layout.anchored_layout``).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.export_json import check_export, extremum_graph_to_dict
from pipeline.spine_layout import MAX_NODES, anchored_layout

from .metrics import compare
from .sources import FieldSource
from .variants import MAIN_VARIANT, ExtractionVariant, VariantResult, variants_for


class PositionRegistry:
    """Where each vertex is drawn, shared by all variants of one field."""

    def __init__(self, scale: float):
        self.scale = scale
        self.xy: Dict[int, List[float]] = {}

    @classmethod
    def from_document(cls, doc: dict) -> "PositionRegistry":
        reg = cls(doc["spine"]["scale"])
        reg.add_document(doc)
        return reg

    def add_document(self, doc: dict) -> None:
        """Remember every node of ``doc`` that is not placed yet (first placement wins)."""
        sp = doc["spine"]
        for nodes, xy in ((doc["peaks"], sp["peaks"]), (doc["saddles"], sp["saddles"])):
            for node, p in zip(nodes, xy):
                self.xy.setdefault(int(node["vertex"]), p)


def variant_document(variant: ExtractionVariant, result: VariantResult, source: FieldSource,
                     registry: PositionRegistry, main_doc: dict, main_file: str) -> dict:
    """The JSON document of one non-main variant (same schema as the main file)."""
    meta_main = main_doc["meta"]
    base = meta_main.get("simplify_threshold") if variant.simplifiable else None
    graph = result.graph.simplify(base) if base is not None else result.graph
    if result.saddle_extra and graph is not result.graph:
        raise ValueError("saddle_extra is only supported for variants that are not simplified")
    nodes = graph.num_peaks + graph.num_saddles
    if nodes > MAX_NODES:
        raise ValueError(f"{nodes} nodes: too many for the spine layout")

    layout = anchored_layout(graph, source.coords, registry.xy, registry.scale,
                             spacing=source.spacing)
    meta = {k: meta_main[k] for k in ("name", "kind", "dimensions", "grid_shape", "spacing", "k")
            if k in meta_main}
    meta.update(
        simplify_threshold=base,
        initial_threshold=meta_main.get("initial_threshold") if variant.simplifiable else None,
        value_range=meta_main["value_range"],
        data_file=main_file,
        variant=variant.describe(source),
        all_maxima=sorted(int(v) for v in result.graph.peaks),
        notes=result.notes,
    )
    doc = extremum_graph_to_dict(
        graph, coords=source.coords, meta=meta, include_region=True,
        full_peak_persistence=result.graph.peak_persistence if variant.simplifiable else None,
        spine=layout, saddle_extra=result.saddle_extra)
    check_export(doc)
    return doc


def export_variants(source: FieldSource, main_doc: dict, main_file: str,
                    write, log=print) -> Tuple[List[dict], List[dict]]:
    """Export every variant of ``source``; returns (variants, comparisons) for index.json.

    ``write(doc, filename)`` saves a document. The main document is given
    ``meta.variant`` / ``meta.all_maxima`` here and must be written by the caller
    afterwards.
    """
    registry = PositionRegistry.from_document(main_doc)
    entries, results = [], {}
    for variant in variants_for(source):
        try:
            result = variant.extract(source)
        except Exception as err:          # one failing variant must not stop the build
            log(f"  {variant.id}: skipped ({err})")
            entries.append({**variant.describe(source), "file": None, "error": str(err)})
            continue
        results[variant.id] = result
        if variant.id == MAIN_VARIANT:
            main_doc["meta"].update(variant=variant.describe(source), notes=result.notes,
                                    all_maxima=sorted(int(v) for v in result.graph.peaks))
            entries.append({**variant.describe(source), "file": main_file,
                            "peaks": len(main_doc["peaks"]), "saddles": len(main_doc["saddles"]),
                            "notes": result.notes})
            continue
        try:
            doc = variant_document(variant, result, source, registry, main_doc, main_file)
        except ValueError as err:
            log(f"  {variant.id}: not exported ({err})")
            entries.append({**variant.describe(source), "file": None, "error": str(err),
                            "notes": result.notes})
            continue
        registry.add_document(doc)
        filename = main_file.replace(".json", f".{variant.id}.json")
        write(doc, filename)
        entries.append({**variant.describe(source), "file": filename, "peaks": len(doc["peaks"]),
                        "saddles": len(doc["saddles"]), "notes": result.notes})
        log(f"  {variant.id}: {len(doc['peaks'])} peaks, {len(doc['saddles'])} saddle entries")

    comparisons = []
    for a, b in combinations(results, 2):
        agree = compare(results[a].common, results[b].common)
        comparisons.append({"a": a, "b": b, **{k: v for k, v in agree.as_dict().items()
                                                 if k not in ("a", "b")}})
    return entries, comparisons
