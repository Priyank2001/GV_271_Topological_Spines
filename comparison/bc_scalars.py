"""
BC-TCGA tumours on one fixed neighbourhood, with several scalar functions.

The case study of "which question does each scalar answer": the 529 TCGA breast tumours of
BC-TCGA, one molecular space and one kNN graph, and several functions f on them. Switching f
changes what a peak means; the samples and their neighbours stay the same.

Fixed molecular space (documented, never changed between scalars)
  * points: the 529 tumours (normals are used only for the healthy reference);
  * dimensions: the 1000 genes with the highest variance across the tumours (log ratios as
    given), each standardised to mean 0 and standard deviation 1 across the tumours;
  * distance: Euclidean in those 1000 dimensions; kNN graph with k = 15.

Scalars (all computed from expression only; no label is used)
  * density: -log mean distance to the 15 nearest tumours, in the first 10 principal
    components of the 1000 genes (density in 1000 noisy dimensions is nearly flat);
  * deviation: deviation from normal tissue (the disease component of DSGA, Nicolau et al.
    2007/2011): healthy model = the principal components covering 90 % of the variance of the
    61 normals, each first "flattened" (replaced by its least-squares fit from the other
    normals); deviation = norm of the tumour minus its projection on the model, all genes;
  * g2m: proliferation = mean z-score (across tumours) of the MSigDB Hallmark G2M checkpoint
    genes present in the data;
  * er_early, hypoxia: Hallmark estrogen response (early) and hypoxia scores, same method,
    as context for the questions the first three raise.

Known labels (shown in the viewer, never used to build anything)
  * PAM50 subtype (TCGA PanCanAtlas, via cBioPortal brca_tcga_pan_can_atlas_2018);
  * ER status by IHC (cBioPortal brca_tcga);
  * tissue source site and plate (from the TCGA barcode), for checking batch effects.
Overall survival (same sources) is kept for a separate outcome analysis only.

Data files in ``comparison/bc_data/``: ``clinical.json`` (per sample: PAM50, ER, OS event,
OS months) and ``hallmark.json`` (the five gene sets, MSigDB Hallmark v6.2 symbols).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = Path.home() / "Downloads" / "v3cc2p38hb-1" / "BC-TCGA" / "BC-TCGA"
N_GENES, K, PCS, MODEL_VARIANCE = 1000, 15, 10, 0.90
SIGNATURES = {"g2m": "HALLMARK_G2M_CHECKPOINT", "e2f": "HALLMARK_E2F_TARGETS",
              "er_early": "HALLMARK_ESTROGEN_RESPONSE_EARLY", "er_late": "HALLMARK_ESTROGEN_RESPONSE_LATE",
              "hypoxia": "HALLMARK_HYPOXIA"}


def read_table(path):
    with open(path) as fh:
        samples = fh.readline().rstrip("\n").split("\t")[1:]
        genes, rows = [], []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            genes.append(parts[0])
            rows.append([float(x) if x not in ("", "NA", "NaN", "null") else np.nan for x in parts[1:]])
    return genes, samples, np.array(rows, dtype=np.float64)


def load(src: Path = DEFAULT_SRC) -> dict:
    """Everything the dataset and the case study need (see the module docstring)."""
    src = Path(src).expanduser()
    for name in ("BC-TCGA-Tumor.txt", "BC-TCGA-Normal.txt"):
        if not (src / name).exists():
            raise SystemExit(f"not found: {src / name}\npass the folder that has BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt")
    g_t, s_t, T = read_table(src / "BC-TCGA-Tumor.txt")
    g_n, _, N = read_table(src / "BC-TCGA-Normal.txt")
    if g_t != g_n:
        raise SystemExit("the tumour and normal files list different genes")
    keep = ~(np.isnan(T).any(axis=1) | np.isnan(N).any(axis=1))
    genes = np.array(g_t)[keep]
    T, N = T[keep], N[keep]                                  # genes x samples
    X = T.T                                                  # tumours x genes

    # fixed molecular space
    top = np.argsort(-X.var(axis=0), kind="stable")[:N_GENES]
    Z = (X[:, top] - X[:, top].mean(axis=0)) / X[:, top].std(axis=0)

    # density in the first principal components
    U, S, _ = np.linalg.svd(Z - Z.mean(axis=0), full_matrices=False)
    pc = U[:, :PCS] * S[:PCS]
    d = np.sqrt(((pc[:, None, :] - pc[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    density = -np.log(np.sort(d, axis=1)[:, :K].mean(axis=1))

    # deviation from normal (DSGA disease component)
    def model(normals):
        flat = np.empty_like(normals)
        for i in range(normals.shape[1]):
            others = np.delete(normals, i, axis=1)
            coef, *_ = np.linalg.lstsq(others, normals[:, i], rcond=None)
            flat[:, i] = others @ coef
        Ub, Sb, _ = np.linalg.svd(flat, full_matrices=False)
        r = int(np.searchsorted(np.cumsum(Sb ** 2) / (Sb ** 2).sum(), MODEL_VARIANCE) + 1)
        return Ub[:, :r]
    B = model(N)
    deviation = np.linalg.norm(T - B @ (B.T @ T), axis=0)
    held = []                                                # each normal against a model without it
    for i in range(N.shape[1]):
        Bi = model(np.delete(N, i, axis=1))
        held.append(np.linalg.norm(N[:, i] - Bi @ (Bi.T @ N[:, i])))

    # signature scores
    sets = json.loads((HERE / "bc_data" / "hallmark.json").read_text())
    gi = {g: i for i, g in enumerate(genes)}
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    scores, used = {}, {}
    for key, name in SIGNATURES.items():
        idx = [gi[g] for g in sets[name] if g in gi]
        scores[key] = Xz[:, idx].mean(axis=1)
        used[key] = (name, len(idx), len(sets[name]))

    # labels and outcome
    clin = json.loads((HERE / "bc_data" / "clinical.json").read_text())
    meta = []
    for s in s_t:
        pam50, er, os_event, os_months = clin.get(s, ["", "", None, None])
        meta.append({"id": s, "tss": s[5:7], "plate": s.split("-")[5], "pam50": pam50 or "",
                     "er": {"+": "ER+", "-": "ER-"}.get(er, ""), "os_event": os_event, "os_months": os_months})

    return {"samples": s_t, "genes": genes, "X": X, "Xz": Xz, "top": top, "Z": Z,
            "density": density, "deviation": deviation, "held_out_normals": np.array(held),
            "model_rank": int(B.shape[1]), "scores": scores, "signatures_used": used,
            "sets": sets, "meta": meta}


# ---- the viewer dataset -------------------------------------------------------------------
SCALARS = [  # id, label, summary (the main one first)
    ("density", "Density", "Peaks: locally concentrated molecular profiles (the main populated groups). "
     "A density peak does not mean more severe disease."),
    ("deviation", "Deviation from normal", "Peaks: locally high departure from the healthy-tissue model "
     "(DSGA disease component, all genes)."),
    ("g2m", "Proliferation (G2M)", "Peaks: locally high expression of the Hallmark G2M checkpoint genes."),
    ("er_early", "Estrogen response (early)", "Context score: Hallmark estrogen response, early genes."),
    ("hypoxia", "Hypoxia", "Context score: Hallmark hypoxia genes."),
]
SMOOTHS, KS, MAIN_SMOOTH = (0, 2), (10, 20, 30), 2


def write_dataset(data: dict, out: Path) -> dict:
    """Write generator/datasets/<name>/ for the viewer; returns truth.json."""
    from pipeline.neighbor_graph import knn_graph
    from comparison.variants import scalar_values

    out.mkdir(parents=True, exist_ok=True)
    (out / "scalars").mkdir(exist_ok=True)
    P = data["Z"].astype(np.float32)
    raw = {"density": data["density"], "deviation": data["deviation"], **data["scores"]}
    values = scalar_values(raw["density"], knn_graph(P, K), MAIN_SMOOTH)    # the main file's f
    np.save(out / "points.npy", P)
    np.save(out / "values.npy", values.astype(np.float64))
    for sid, *_ in SCALARS:
        np.save(out / "scalars" / f"{sid}.npy", np.asarray(raw[sid], dtype=np.float64))

    meta = data["meta"]

    def label_set(name, key, order):
        vals = [m[key] for m in meta]
        names = [x for x in order if x in vals] + sorted({v for v in vals if v and v not in order})
        return {"name": name, "names": names, "values": [names.index(v) if v in names else -1 for v in vals]}

    truth = {
        "name": out.name, "dimensions": int(P.shape[1]), "num_points": int(P.shape[0]),
        "description": ("BC-TCGA tumours on one fixed neighbourhood (1000 most variable genes, standardised; "
                        f"kNN k = {K}). f: density smoothed x{MAIN_SMOOTH}, scaled to 0..1; other functions in "
                        "scalars/ (see comparison/bc_scalars.py)."),
        "dimension_names": [str(g) for g in data["genes"][data["top"]]],
        "sample_ids": list(data["samples"]),
        "extrema": {"maxima": []},
        "point_classes": [
            label_set("PAM50 subtype", "pam50", ["LumA", "LumB", "Her2", "Basal", "Normal"]),
            label_set("ER status (IHC)", "er", ["ER+", "ER-"]),
            label_set("Tissue source site", "tss", []),
            label_set("Plate", "plate", []),
        ],
        "scalars": {
            "functions": [{"id": sid, "label": lab, "summary": summ, "file": f"scalars/{sid}.npy"}
                          for sid, lab, summ in SCALARS],
            "smooths": list(SMOOTHS), "ks": list(KS), "main": {"id": "density", "smooth": MAIN_SMOOTH},
        },
    }
    (out / "truth.json").write_text(json.dumps(truth))
    return truth
