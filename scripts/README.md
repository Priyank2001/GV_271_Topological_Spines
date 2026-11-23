# Dataset scripts (case studies)

Each script builds one dataset in `generator/datasets/` (ignored by git); then
`python3 web/build_data.py <name>` writes the viewer files. Run from the repo root.

| Viewer entry | Build | Input data |
|---|---|---|
| `bc_tcga` | `python3 scripts/add_bc_tcga.py [folder]` | BC-TCGA (Mendeley Data, `BC-TCGA-Tumor.txt`, `BC-TCGA-Normal.txt`), default `~/Downloads/v3cc2p38hb-1/BC-TCGA/BC-TCGA` |
| `nki` | `python3 scripts/add_nki.py` | Bioconductor `breastCancerNKI` (nki.rda) |
| `bc_dsga` | `python3 scripts/add_bc_dsga.py [folder]` | BC-TCGA |
| `bc_tumours` | `python3 scripts/make_bc_tumours.py [folder]` | BC-TCGA, plus `comparison/bc_data/` (PAM50, ER, survival; Hallmark gene sets) |
| `pbmc3k_types` | `python3 scripts/add_pbmc3k.py` | `data/pbmc3k_types/pbmc3k_counts.npz` (10x PBMC 3k counts + reference clusters) |
| `alanine_dipeptide` | `python3 scripts/add_alanine_dipeptide.py` | `data/alanine_dipeptide/alanine_dipeptide_md.npz`, made by `scripts/alanine_dipeptide_md.py` (OpenMM) |

The `add_*` scripts were written to patch an older tree; on this branch their code changes are already
in place, so they only build the dataset (and print "already has ..."). Results:
`comparison/results/scalar_case_study.md`, `pbmc3k_types.md`, `alanine_dipeptide.md`, `mapper_vs_spine.md`
(Mapper: `python3 scripts/mapper_compare.py`, needs `kmapper`).
