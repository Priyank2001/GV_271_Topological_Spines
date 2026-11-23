#!/usr/bin/env python3
"""Build generator/datasets/bc_tumours (BC-TCGA tumours, several scalars; see comparison/bc_scalars.py).

    python3 scripts/make_bc_tumours.py [folder with BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt]
    python3 web/build_data.py bc_tumours
    python3 -m comparison.scalar_case_study      # the result tables (comparison/results/scalar_case_study.md)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from comparison import bc_scalars  # noqa: E402

src = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else bc_scalars.DEFAULT_SRC
data = bc_scalars.load(src)
bc_scalars.write_dataset(data, ROOT / "generator" / "datasets" / "bc_tumours")
print("wrote generator/datasets/bc_tumours; next: python3 web/build_data.py bc_tumours")
