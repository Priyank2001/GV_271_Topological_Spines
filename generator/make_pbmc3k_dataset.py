#!/usr/bin/env python3
"""
PBMC 3k (10x Genomics scRNA-seq) as a point-cloud scalar field.

    python3 generator/make_pbmc3k_dataset.py            # downloads if needed
    python3 generator/make_pbmc3k_dataset.py --gene CD3E

A cell is a point; the scalar value is one number per cell. scRNA-seq has no
built-in scalar, so we offer two choices:

  density (default)  kNN density in the PCA space: 1 / mean distance to the 15
                     nearest cells, scaled to [0, 1]. Peaks are cell populations.
  --gene NAME        log-normalised expression of one gene (e.g. CD3E for T
                     cells, MS4A1 for B cells, LYZ for monocytes, NKG7 for NK).

Preprocessing is the standard Scanpy/Seurat recipe, done with numpy/scipy
(Scanpy need not be installed): drop cells with <200 genes or >5% mitochondrial
reads, drop genes seen in <3 cells, normalise to 1e4 counts per cell, log1p,
keep the 2000 most variable genes, scale, PCA to ``--pcs`` components.
The points are the PCA coordinates; the extremum graph is built on them.

Output: generator/datasets/pbmc3k/{points.npy, values.npy, truth.json}
(plus genes.json / cells.json with the names, not read by the pipeline).
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
URL = "https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
RAW_DIR = ROOT / "data" / "pbmc3k"
OUT_DIR = Path(__file__).parent / "datasets"
NAME = "pbmc3k"


def fetch() -> Path:
    mtx_dir = RAW_DIR / "raw" / "filtered_gene_bc_matrices" / "hg19"
    if (mtx_dir / "matrix.mtx").exists():
        return mtx_dir
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tgz = RAW_DIR / "pbmc3k_filtered_gene_bc_matrices.tar.gz"
    if not tgz.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(RAW_DIR / "raw")
    return mtx_dir


def load_counts(mtx_dir: Path):
    counts = scipy.io.mmread(mtx_dir / "matrix.mtx").tocsr().T.tocsr()     # cells x genes
    genes = [l.split("\t")[1] for l in (mtx_dir / "genes.tsv").read_text().splitlines()]
    cells = (mtx_dir / "barcodes.tsv").read_text().split()
    return counts.astype(np.float32), np.array(genes), np.array(cells)


def preprocess(counts, genes, cells, n_hvg: int, n_pcs: int):
    # QC filters
    n_genes = np.asarray((counts > 0).sum(1)).ravel()
    mito = np.array([g.startswith("MT-") for g in genes])
    mito_frac = np.asarray(counts[:, mito].sum(1)).ravel() / np.asarray(counts.sum(1)).ravel()
    keep_c = (n_genes >= 200) & (mito_frac < 0.05)
    counts, cells = counts[keep_c], cells[keep_c]
    keep_g = np.asarray((counts > 0).sum(0)).ravel() >= 3
    counts, genes = counts[:, keep_g], genes[keep_g]

    # normalise + log1p
    totals = np.asarray(counts.sum(1)).ravel()
    norm = sp.diags(1e4 / totals) @ counts
    norm.data = np.log1p(norm.data)
    norm = norm.tocsr()

    # highly variable genes: dispersion (var/mean) z-scored within 20 mean bins
    # (Seurat v1 / Scanpy "seurat" flavour)
    mean = np.asarray(norm.mean(0)).ravel()
    sq = np.asarray(norm.multiply(norm).mean(0)).ravel()
    var = (sq - mean ** 2) * norm.shape[0] / (norm.shape[0] - 1)
    disp = np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12))
    bins = np.digitize(mean, np.quantile(mean, np.linspace(0, 1, 21)[1:-1]))
    z = np.zeros_like(disp)
    for b in np.unique(bins):
        m = bins == b
        sd = disp[m].std()
        z[m] = (disp[m] - disp[m].mean()) / (sd if sd > 0 else 1.0)
    hvg = np.argsort(-z)[:n_hvg]

    dense = norm[:, hvg].toarray()
    dense = (dense - dense.mean(0)) / np.maximum(dense.std(0), 1e-8)
    dense = np.clip(dense, -10, 10)
    u, s, _ = np.linalg.svd(dense, full_matrices=False)
    pcs = (u[:, :n_pcs] * s[:n_pcs]).astype(np.float32)
    return pcs, norm, genes, cells


def knn_density(points: np.ndarray, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    return 1.0 / dist[:, 1:].mean(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gene", help="use this gene's expression as the scalar (default: kNN density)")
    ap.add_argument("--pcs", type=int, default=50, help="PCA dimensions (default 50)")
    ap.add_argument("--hvg", type=int, default=2000, help="highly variable genes (default 2000)")
    ap.add_argument("--name", default=NAME, help="output folder name")
    args = ap.parse_args()

    counts, genes, cells = load_counts(fetch())
    pcs, norm, genes, cells = preprocess(counts, genes, cells, args.hvg, args.pcs)

    if args.gene:
        idx = np.flatnonzero(genes == args.gene)
        if not idx.size:
            raise SystemExit(f"gene {args.gene!r} not found (after filtering)")
        raw = np.asarray(norm[:, idx[0]].todense()).ravel()
        mode, formula = "scrna_gene", f"log1p(CP10K) expression of {args.gene}"
    else:
        raw = knn_density(pcs)
        mode, formula = "scrna_density", "1 / mean distance to the 15 nearest cells in PCA space"
    lo, hi = float(raw.min()), float(raw.max())
    vals = ((raw - lo) / (hi - lo)).astype(np.float32)

    out = OUT_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", pcs)
    np.save(out / "values.npy", vals)
    truth = {
        "name": args.name, "dimensions": int(pcs.shape[1]), "num_points": int(pcs.shape[0]),
        "mode": mode, "has_extrema_labels": False,
        "source": URL, "formula": formula, "value_range_raw": [lo, hi],
        "preprocessing": f"QC, CP10K, log1p, {args.hvg} HVGs, scale, PCA {args.pcs}",
    }
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "cells.json").write_text(json.dumps(cells.tolist()))
    print(f"{args.name}: N={pcs.shape[0]}, d={pcs.shape[1]}, scalar = {formula}")


if __name__ == "__main__":
    main()
