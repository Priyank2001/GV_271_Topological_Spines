"""
Command line: compute an AnyEG extremum graph and write it as a .vtp file.

    python3 -m anyeg third_party/AnyEG/data/scalar_field_v1.vti -o results/v1.vtp
    python3 -m anyeg --volume fuel -o results/fuel.vtp --kernels original
    python3 -m anyeg data.vti --array ImageFile --range 0 200 --bundle

``--volume`` reads a downloaded Open SciVis volume from data/klacansky/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import AnyEGPipeline, ArcAssembler, AssemblyOptions, ScalarField, make_kernels, read_vti, write_vtp

KLACANSKY = Path(__file__).resolve().parent.parent / "data" / "klacansky"


def load_klacansky(name: str) -> ScalarField:
    """A raw Open SciVis volume, values as stored (AnyEG reads them as float)."""
    meta = json.loads((KLACANSKY / "datasets.json").read_text())[name]
    nx, ny, nz = meta["size"]
    path = KLACANSKY / Path(meta["url"]).name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; download it with pipeline.volume_loader.download")
    values = np.fromfile(path, dtype=np.dtype(meta["type"])).reshape(nz, ny, nx)
    return ScalarField.from_grid(values, meta.get("spacing", (1, 1, 1)), name=name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("input", nargs="?", help=".vti file")
    src.add_argument("--volume", help="Open SciVis volume name (data/klacansky)")
    ap.add_argument("-o", "--output", help=".vtp file to write (default: results/EG_<name>.vtp)")
    ap.add_argument("--array", help="point array to use (default: the active scalars)")
    ap.add_argument("--kernels", default="torch", help="torch (default), torch:cpu or original")
    ap.add_argument("--range", type=float, nargs=2, default=(0.0, 20000.0), metavar=("LOW", "HIGH"),
                    help="keep arcs with LOW <= |f(saddle) - f(max)| < HIGH (AnyEG: 0 20000)")
    ap.add_argument("--bundle", action="store_true", help="AnyEG's edge bundling (off in AnyEG)")
    args = ap.parse_args()

    field = read_vti(args.input, args.array) if args.input else load_klacansky(args.volume)
    pipeline = AnyEGPipeline(kernels=make_kernels(args.kernels),
                             assembler=ArcAssembler(AssemblyOptions(tuple(args.range), args.bundle)))
    graph = pipeline.run(field)
    out = Path(args.output or f"results/EG_{field.name}.vtp")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_vtp(out, graph.to_polylines(field))
    print(field)
    print(graph)
    print("seconds:", {k: round(v, 4) for k, v in graph.seconds.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
