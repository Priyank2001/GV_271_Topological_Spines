#!/usr/bin/env python3
"""
DomainNet as a point cloud: CLIP image embeddings of a few classes in all six domains.

    python3 scripts/make_domainnet.py                    # default 10 classes, <= 100 images per class x domain
    python3 scripts/make_domainnet.py --classes cat dog tiger lion --per-cell 150
    python3 web/build_data.py domainnet

DomainNet (Peng et al. 2019) has 345 object classes drawn in six domains: clipart, infograph,
painting, quickdraw, real, sketch. Its question for us: in a vision-language embedding, do the
density peaks of the extremum graph follow *what* is shown (the class) or *how* it is drawn (the
domain)?

Input (read-only, cached by the Ramen project; nothing is recomputed on the GPU):
  * ``--features``: zero-shot CLIP ViT-B/32 image features of all 596,006 images, fp16 (N, 512), in the
    stream order of the seed-0 NoAdapt run;
  * ``--meta``: that run's table (label, domain, CLIP's zero-shot prediction and confidence per row).
    It is a pandas >= 2 pickle, which the repo's Python 3.8 cannot read; the first run therefore needs
    an interpreter with pandas >= 2 (e.g. ``~/miniconda3/envs/RAMEN/bin/python``) and caches the needed
    columns in ``data/domainnet/meta.npz``. Later runs (any Python) use that cache.

How the point cloud is made
  * the chosen classes, every domain, at most ``--per-cell`` images per (class, domain), drawn at random
    with ``--seed`` (quickdraw and real have ~500 per class, clipart ~140: without the cap two domains
    would dominate the density);
  * points: the 512-D CLIP features, scaled to unit length (CLIP's cosine geometry: on the unit sphere
    Euclidean distance is a function of cosine similarity). The kNN graph is built in this full space;
  * f (main): density, -log of the mean distance to the ``K`` nearest images, scaled to 0..1;
  * second scalar: CLIP's zero-shot confidence (max softmax probability over all 345 class prompts).
  * known labels (never used to build f or the graph): class, domain, zero-shot right / wrong.

Output: generator/datasets/domainnet/ and comparison/results/domainnet.md.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAMEN = Path("/data/priyank1/Ramen/new_experiments/results")
FEATURES = RAMEN / "DomainNet_clip_vitbase32_mixed_nall_bs100_s0_NoAdapt_feats.npy"
META = RAMEN / "DomainNet_clip_vitbase32_mixed_nall_bs100_s0_NoAdapt.pkl"
CLASS_DIR = Path.home() / "data" / "domainbed" / "domain_net" / "clipart"   # folder names = class names
CACHE = ROOT / "data" / "domainnet" / "meta.npz"
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
# Pairs that should be near in meaning (cat-dog-tiger, car-bus) next to unrelated objects.
DEFAULT_CLASSES = ["airplane", "bicycle", "bus", "car", "cat", "dog", "tiger", "zebra", "guitar", "umbrella"]
K, TAU = 15, 0.04


# ---- data ---------------------------------------------------------------------------------------
def load_meta(meta: Path) -> dict:
    """label, domain index, zero-shot prediction and confidence per row (cached as npz)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    try:
        import pandas as pd
        df = pd.read_pickle(meta)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        sys.exit(f"cannot read {meta} with this Python ({e}).\nRun this script once with a pandas >= 2 "
                 f"interpreter, e.g. ~/miniconda3/envs/RAMEN/bin/python {' '.join(sys.argv)}; it caches {CACHE}.")
    assert (df.position.values == np.arange(len(df))).all()
    names = df.groupby("domain_idx", observed=True).domain.first()
    assert [str(names[d]) for d in range(len(names))] == DOMAINS, list(names)
    out = {"label": df.label.values.astype(np.int16), "domain": df.domain_idx.values.astype(np.int8),
           "pred": df.pred.values.astype(np.int16), "conf": df.conf.values.astype(np.float32),
           "sample_idx": df.sample_idx.values.astype(np.int32)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    print(f"cached {CACHE}")
    return out


def class_names() -> list:
    # ImageFolder's class order (sorted folder names) = the label ids of the Ramen runs
    names = sorted(e.name for e in CLASS_DIR.iterdir() if e.is_dir())
    assert len(names) == 345, len(names)
    return names


def choose_rows(meta: dict, classes: list, per_cell: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for c in classes:
        for d in range(len(DOMAINS)):
            idx = np.flatnonzero((meta["label"] == c) & (meta["domain"] == d))
            rows.append(np.sort(rng.choice(idx, min(per_cell, len(idx)), replace=False)))
    return np.concatenate(rows)


def density(P: np.ndarray, k: int) -> np.ndarray:
    import torch
    X = torch.as_tensor(P, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    f = -torch.log(d.topk(k, largest=False).values.mean(1)).cpu().numpy().astype(np.float64)
    return (f - f.min()) / (f.max() - f.min())


# ---- report -------------------------------------------------------------------------------------
def report(P, f, cls, dom, correct, cls_names, out_md: Path, name: str) -> None:
    from pipeline.extremum_graph import extremum_graph
    from pipeline.neighbor_graph import knn_graph

    spine = extremum_graph(f, knn_graph(P, K)).simplify(TAU)

    def mostly(labels, names, mask):
        (t, n), *rest = collections.Counter(labels[mask]).most_common()
        s = names[t]
        if rest and rest[0][1] >= 0.2 * mask.sum():
            s += f" + {names[rest[0][0]]}"
        return s, n / mask.sum()

    def label(mask):
        return f"{mostly(cls, cls_names, mask)[0]} / {mostly(dom, DOMAINS, mask)[0]}"

    def regions(k, tau):
        eg = extremum_graph(f, knn_graph(P, k)).simplify(tau)
        return [set(np.flatnonzero(eg.region == p)) for p in range(eg.num_peaks)]

    grid = [(k, tau) for k in (10, 15, 20, 30) for tau in (0.02, 0.04, 0.08)]
    others = [regions(k, tau) for k, tau in grid]

    # agreement of the peak regions with each labelling (adjusted Rand, NMI)
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
    reg = spine.region

    rows = []
    for p in np.argsort(-spine.region_size):
        m = reg == p
        c, cp = mostly(cls, cls_names, m)
        d, dp = mostly(dom, DOMAINS, m)
        mem = set(np.flatnonzero(m))
        found = np.mean([max(len(mem & r) / len(mem | r) for r in regs) >= 0.5 for regs in others])
        pers = spine.peak_persistence[p]
        rows.append((c, cp, d, dp, int(m.sum()), correct[m].mean(),
                     "highest" if not np.isfinite(pers) else f"{pers:.3f}", found))

    L = [f"# {name}: what the spine shows", "",
         f"{len(P)} DomainNet images ({len(cls_names)} classes x {len(DOMAINS)} domains); points: 512-D CLIP "
         f"ViT-B/32 image features (unit length); kNN graph k = {K}; f = density; persistence threshold {TAU}. "
         "Class and domain are known labels, not used to build anything.", "",
         f"**{spine.num_peaks} peaks, {len(spine.saddle_values)} saddles.** Agreement of the peak regions with",
         f"the labels: class ARI {ari(cls, reg):.2f} / NMI {nmi(cls, reg):.2f}; "
         f"domain ARI {ari(dom, reg):.2f} / NMI {nmi(dom, reg):.2f}; "
         f"class x domain NMI {nmi(cls * 10 + dom, reg):.2f}.", "",
         "| Mostly class | share | Mostly domain | share | Images | Zero-shot right | Persistence | Found in 12 settings |",
         "|---|---|---|---|---|---|---|---|"]
    for c, cp, d, dp, n, acc, pers, found in rows:
        L.append(f"| {c} | {cp:.0%} | {d} | {dp:.0%} | {n} | {acc:.0%} | {pers} | {found:.0%} |")
    # what the saddles join: same dominant class, same dominant domain, both or neither
    top_c = {p: collections.Counter(cls[reg == p]).most_common(1)[0][0] for p in range(spine.num_peaks)}
    top_d = {p: collections.Counter(dom[reg == p]).most_common(1)[0][0] for p in range(spine.num_peaks)}
    L += ["", "What the saddles join (by the dominant class and domain of the two peak regions):", "",
          "| Saddles | count | same class | same domain | both | neither |", "|---|---|---|---|---|---|"]
    for kind, sel in (("merge (spine tree)", spine.saddle_is_merge), ("loop", ~np.asarray(spine.saddle_is_merge))):
        pairs = [spine.saddle_peaks[s] for s in np.flatnonzero(sel)]
        sc = np.array([top_c[a] == top_c[b] for a, b in pairs], bool)
        sd = np.array([top_d[a] == top_d[b] for a, b in pairs], bool)
        L.append(f"| {kind} | {len(pairs)} | {sc.sum()} | {sd.sum()} | {(sc & sd).sum()} | {(~sc & ~sd).sum()} |")
    L += ["", "Highest 25 saddles (the regions that blend into each other most easily):", ""]
    for s in np.argsort(-spine.saddle_values)[:25]:
        a, b = spine.saddle_peaks[s]
        kind = "merge" if spine.saddle_is_merge[s] else "loop"
        L.append(f"* {label(reg == a)} -- {label(reg == b)}: {spine.saddle_values[s]:.2f} ({kind})")
    L += ["", "Stability: k in (10, 15, 20, 30) x threshold in (0.02, 0.04, 0.08); a peak is found when some peak "
          "there shares at least half of its images (Jaccard >= 0.5)."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---- main ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="DomainNet class folder names")
    ap.add_argument("--per-cell", type=int, default=100, help="max images per (class, domain) (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="domainnet", help="output folder in generator/datasets/")
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--meta", type=Path, default=META)
    args = ap.parse_args()

    names = class_names()
    unknown = [c for c in args.classes if c not in names]
    if unknown:
        sys.exit(f"unknown classes {unknown}")
    ids = [names.index(c) for c in args.classes]

    meta = load_meta(args.meta)
    rows = choose_rows(meta, ids, args.per_cell, args.seed)
    feats = np.load(args.features, mmap_mode="r")
    assert len(feats) == len(meta["label"]), (feats.shape, len(meta["label"]))
    P = np.asarray(feats[rows], dtype=np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)

    cls = np.array([ids.index(c) for c in meta["label"][rows]])
    dom = meta["domain"][rows].astype(int)
    correct = meta["pred"][rows] == meta["label"][rows]
    conf = meta["conf"][rows].astype(np.float64)
    f = density(P, K)

    out = ROOT / "generator" / "datasets" / args.name
    (out / "scalars").mkdir(parents=True, exist_ok=True)
    np.save(out / "points.npy", P)
    np.save(out / "values.npy", f)
    np.save(out / "scalars" / "density.npy", f)
    np.save(out / "scalars" / "confidence.npy", conf)
    pred_names = [names[p] for p in meta["pred"][rows]]
    truth = {
        "name": args.name, "dimensions": int(P.shape[1]), "num_points": int(len(P)),
        "description": (f"DomainNet, {len(ids)} classes x 6 domains, <= {args.per_cell} images per class x domain "
                        f"(seed {args.seed}). Points: zero-shot CLIP ViT-B/32 image features (512-D, unit length). "
                        f"f: density, -log mean distance to the {K} nearest images, scaled to 0..1."),
        "source": f"{args.features} (rows {', '.join(map(str, rows[:3]))}, ...)",
        "dimension_names": [f"clip{j + 1}" for j in range(P.shape[1])],
        "sample_ids": [f"{DOMAINS[d]}/{names[ids[c]]}/{int(meta['sample_idx'][r])}"
                       for c, d, r in zip(cls, dom, rows)],
        "extrema": {"maxima": []},
        "point_classes": [
            {"name": "Class", "names": list(args.classes), "values": cls.tolist()},
            {"name": "Domain", "names": DOMAINS, "values": dom.tolist()},
            {"name": "CLIP zero-shot", "names": ["right", "wrong"], "values": (~correct).astype(int).tolist()},
        ],
        "scalars": {
            "functions": [
                {"id": "density", "label": "density", "file": "scalars/density.npy",
                 "summary": f"-log mean distance to the {K} nearest images (512-D CLIP space)."},
                {"id": "confidence", "label": "CLIP zero-shot confidence", "file": "scalars/confidence.npy",
                 "summary": "Max softmax probability over all 345 DomainNet class prompts (logit scale 100)."},
            ],
            "smooths": [0], "ks": [10, 30], "main": {"id": "density", "smooth": 0},
        },
        "zero_shot_prediction": pred_names,
    }
    (out / "truth.json").write_text(json.dumps(truth))
    counts = np.bincount(dom, minlength=6)
    print(f"wrote {out}: {len(P)} images, 512-D; per domain " + ", ".join(f"{d} {n}" for d, n in zip(DOMAINS, counts))
          + f"; zero-shot accuracy {correct.mean():.1%}")

    report(P, f, cls, dom, correct, list(args.classes), ROOT / "comparison" / "results" / f"{args.name}.md", args.name)


if __name__ == "__main__":
    main()
