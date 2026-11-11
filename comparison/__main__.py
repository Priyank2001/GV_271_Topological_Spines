"""
Compare AnyEG's extremum graphs with our pipeline's on the same fields.

    anyeg/native/build.sh              # once: builds the original AnyEG kernels
    python3 -m comparison              # writes comparison/results/anyeg_vs_ours.{md,json}
    python3 -m comparison --volumes fuel --variants raw rank

Without the original build, only the PyTorch port of AnyEG is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anyeg.kernels import OriginalCudaKernels, TorchKernels

from .inputs import VARIANTS, anyeg_samples, klacansky_volumes
from .methods import AnyEGMethod, OursMethod
from .study import ComparisonStudy, write_report

DEFAULT_VOLUMES = ["nucleon", "marschner_lobb", "silicium", "fuel", "neghip", "hydrogen_atom"]

METHOD_NOTES = {
    "AnyEG original": "AnyEG's own CUDA `compute()` (cudaHelper.cu, unchanged) on TTK's "
                      "14-neighbour grid, plus the port of its arc assembly",
    "AnyEG torch": "`anyeg.TorchKernels`, the PyTorch port of the same four GPU modules (GPU)",
    "Ours 6-nbr CPU": "`pipeline.extremum_graph` on our 6-neighbour grid graph, NumPy backend",
    "Ours 6-nbr GPU": "the same with the torch backend (must give identical results)",
    "Ours 14-nbr": "our algorithm on AnyEG's 14-neighbour grid, torch backend: "
                   "isolates the algorithmic differences",
}

PAIRS = [
    ("AnyEG original", "AnyEG torch"),       # is the port exact?
    ("Ours 6-nbr CPU", "Ours 6-nbr GPU"),    # is our GPU backend exact?
    ("AnyEG torch", "Ours 14-nbr"),          # same neighbourhood: algorithm only
    ("AnyEG torch", "Ours 6-nbr GPU"),       # each method as designed
    ("Ours 14-nbr", "Ours 6-nbr GPU"),       # effect of the neighbourhood on ours
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volumes", nargs="*", default=DEFAULT_VOLUMES)
    ap.add_argument("--no-samples", action="store_true", help="skip AnyEG's sample .vti files")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    inputs = ([] if args.no_samples else anyeg_samples()) + klacansky_volumes(args.volumes)
    variants = [VARIANTS[v]() for v in args.variants]

    methods = []
    if OriginalCudaKernels.is_built():
        methods.append(AnyEGMethod("AnyEG original", OriginalCudaKernels()))
    else:
        print("anyeg/native/build/anyeg_driver not built: skipping the original AnyEG kernels")
    methods += [
        AnyEGMethod("AnyEG torch", TorchKernels()),
        OursMethod("Ours 6-nbr CPU", "axis-6", backend="numpy"),
        OursMethod("Ours 6-nbr GPU", "axis-6", backend="torch"),
        OursMethod("Ours 14-nbr", "freudenthal-14", backend="torch"),
    ]
    notes = {m.name: METHOD_NOTES[m.name] for m in methods}

    study = ComparisonStudy(inputs, variants, methods, PAIRS)
    results = study.run()
    md, js = write_report(results, Path(args.out), variants, notes)
    print("wrote", md, "and", js)


if __name__ == "__main__":
    main()
