"""
Does our extremum graph open the ring of the paper's volcano (Fig. 5) when the rim is breached?

Datasets from generator/make_volcano_dataset.py: a crater with 6 hills on its
rim; one rim saddle is notched (depth b). The paper's rule (section 3.4): a
saddle closer in value to its hills than to the crater floor is ridge-like and
keeps the ring closed; otherwise it is valley-like and is cut, so the ring
opens into a chain. truth.json holds that expectation.

For each dataset this prints the spine our pipeline gives (kNN graph, noise
threshold 0.02), first with p_hi off, then for a few p_hi values, and says
whether it matches the paper.

Run from the repo root (needs the datasets: python3 generator/make_volcano_dataset.py):

    python3 tests/check_volcano.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data_loader import load_dataset  # noqa: E402
from pipeline.extremum_graph import extremum_graph  # noqa: E402
from pipeline.neighbor_graph import knn_graph  # noqa: E402

NAMES = ["volcano_intact", "volcano_notched", "volcano_breached"]
P_HI = [None, 0.25, 0.3, 0.45]


def shape(g) -> str:
    """'ring', 'chain' or 'other' for a simplified graph (peaks + their links)."""
    n, edges = g.num_peaks, {tuple(sorted(e)) for e in g.saddle_peaks.tolist()}
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    parts = len({find(i) for i in range(n)})
    deg = np.bincount(np.array(list(edges)).ravel(), minlength=n) if edges else np.zeros(n, int)
    if parts == 1 and len(edges) == n and (deg == 2).all():
        return "ring"
    if parts == 1 and len(edges) == n - 1 and deg.max() <= 2:
        return "chain"
    return f"other ({parts} parts, {len(edges)} links)"


def analyse(name: str, k: int = 15, threshold: float = 0.02) -> dict:
    ds = load_dataset(name)
    eg = extremum_graph(ds.values, knn_graph(ds.points, k)).simplify(threshold)
    sads = [c for c in ds.truth["critical_points"] if c["type"] == "saddle"]
    breach = min(sads, key=lambda s: abs(s["angle_deg"] - ds.truth["generator_metadata"]["breach_angle_deg"]))
    # our saddle nearest to the analytic breach saddle
    pos = ds.points[eg.saddles]
    ours = int(np.argmin(np.linalg.norm(pos - np.array(breach["center"]), axis=1)))
    others = np.delete(eg.saddle_persistence, ours)
    expect = "ring" if ds.truth["extremum_graph_is_cycle"] else "chain"
    rows = [(p, shape(eg.simplify(threshold, variation=p) if p is not None else eg)) for p in P_HI]
    return dict(name=name, b=ds.truth["generator_metadata"]["b"], peaks=eg.num_peaks, saddles=eg.num_saddles,
                breach=breach, ours_pers=float(eg.saddle_persistence[ours]),
                ours_relief=float(eg.saddle_relief[ours]), others=(float(others.min()), float(others.max())),
                expect=expect, rows=rows)


def main() -> None:
    for name in NAMES:
        r = analyse(name)
        br = r["breach"]
        print(f"{name} (b = {r['b']}): {r['peaks']} peaks, {r['saddles']} saddles")
        print(f"  breach saddle: f = {br['value']:.3f}, {br['to_peaks']:.3f} below its hills, "
              f"{br['to_floor']:.3f} above the crater floor -> paper: {br['paper_class']}, expect a {r['expect']}")
        print(f"  ours: breach persistence {r['ours_pers']:.3f} (other rim saddles "
              f"{r['others'][0]:.3f}-{r['others'][1]:.3f}), relief {r['ours_relief']:.2f}")
        for p, got in r["rows"]:
            ok = "matches" if got == r["expect"] else "WRONG"
            print(f"    p_hi {'off ' if p is None else f'{p:.2f}'}: {got:6s} {ok}")


if __name__ == "__main__":
    main()
