"""
Case study: three scalars on one fixed tumour neighbourhood (BC-TCGA, 529 tumours).

    python3 -m comparison.scalar_case_study [folder with BC-TCGA-Tumor.txt and BC-TCGA-Normal.txt]

Writes comparison/results/scalar_case_study.md (the result tables) and .json (everything).
Uses comparison/bc_scalars.py for the data (fixed space, scalars, labels); about 3 minutes.

Questions (one per scalar; the neighbourhood never changes)
  * density: what are the main populated molecular groups?
  * deviation from normal: do tumours with comparable deviation sit in molecularly
    different regions? (similar scalar magnitude, different molecular pattern)
  * proliferation (G2M): is high proliferation concentrated in one region or reached in several?
For every basin: size, scalar median and 5-95 % range, persistence, distinguishing genes,
PAM50 / ER composition (looked at after the genes), and stability over a predefined grid of
k, smoothing (including none) and persistence thresholds, by overlap of sample membership
(Jaccard of the best-matching basin), not by peak ids. Batch check: association of the basins
with tissue source site and plate (Cramer's V). Survival: a separate outcome analysis.
"""
import json, sys, itertools, collections
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.neighbor_graph import knn_graph
from pipeline.extremum_graph import extremum_graph
from comparison import bc_scalars
from comparison.variants import scalar_values

D = bc_scalars.load(Path(sys.argv[1]) if len(sys.argv) > 1 else bc_scalars.DEFAULT_SRC)
Z, X, genes, meta, Xz = D["Z"], D["X"], D["genes"], D["meta"], D["Xz"]
n = len(meta)
PROLIF_GENES = set(D["sets"]["HALLMARK_G2M_CHECKPOINT"]) | set(D["sets"]["HALLMARK_E2F_TARGETS"])
SCALARS = {
    'density': (D['density'], 'Density'),
    'deviation': (D['deviation'], 'Deviation from normal'),
    'g2m': (D['scores']['g2m'], 'Proliferation (G2M checkpoint)'),
}
CONTEXT = dict(D['scores']); CONTEXT['deviation'] = D['deviation']; CONTEXT['density'] = D['density']

KS, SMOOTHS, TAUS = (10, 15, 20, 30), (0, 1, 2), (0.02, 0.04, 0.08)
REF = dict(k=15, s=2, tau=0.04)
MIN_BASIN = 20
graphs = {k: knn_graph(Z.astype(np.float32), k) for k in KS}


def spine(raw, k, s, tau):
    g = graphs[k]; v = scalar_values(raw, g, s)
    eg = extremum_graph(v, g)
    return eg, eg.simplify(tau), v


def jac(a, b):
    return len(a & b) / len(a | b)


def cramers_v(labels_a, labels_b):
    ua, ub = sorted(set(labels_a)), sorted(set(labels_b))
    t = np.zeros((len(ua), len(ub)))
    for x, y in zip(labels_a, labels_b):
        t[ua.index(x), ub.index(y)] += 1
    t = t[t.sum(1) > 0][:, t.sum(0) > 0]
    chi2, p, *_ = chi2_contingency(t)
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))), float(p)


def effect(mask_a, mask_b, exclude=()):
    a, b = Xz[mask_a], Xz[mask_b]
    sd = np.sqrt((a.var(0) * (len(a) - 1) + b.var(0) * (len(b) - 1)) / (len(a) + len(b) - 2)) + 1e-9
    d = (a.mean(0) - b.mean(0)) / sd
    ok = np.array([x not in exclude for x in genes])
    order = np.argsort(-d); up = [(str(genes[i]), round(float(d[i]), 2)) for i in order if ok[i]][:6]
    order = np.argsort(d); dn = [(str(genes[i]), round(float(d[i]), 2)) for i in order if ok[i]][:6]
    return d, up, dn


def comp(mask, key):
    c = collections.Counter(meta[i][key] or 'n/a' for i in np.flatnonzero(mask))
    return dict(c.most_common())


out = {'settings': dict(ks=KS, smooths=SMOOTHS, taus=TAUS, ref=REF, min_basin=MIN_BASIN),
       'model_rank': D['model_rank'], 'held_out_normals_median': float(np.median(D['held_out_normals'])),
       'tumour_deviation_median': float(np.median(D['deviation'])),
       'held_out_normals_above_tumour_median': int((D['held_out_normals'] > np.median(D['deviation'])).sum()),
       'normals': int(len(D['held_out_normals'])),
       'sets': D['signatures_used'], 'scalars': {}}

pam = [m['pam50'] or 'n/a' for m in meta]; er = [m['er'] or 'n/a' for m in meta]
for sid, (raw, label) in SCALARS.items():
    R = {'label': label, 'peaks_by_setting': {}}
    # peak counts, raw graph at persistence 0 and after simplification
    for k in KS:
        for s in SMOOTHS:
            full, simp, _ = spine(raw, k, s, REF['tau'])
            R['peaks_by_setting'][f'k{k}_s{s}'] = [int(full.num_peaks), int(simp.num_peaks)]
    full, ref, v = spine(raw, REF['k'], REF['s'], REF['tau'])
    pers = ref.peak_persistence
    R['persistence_sorted'] = sorted([round(float(x), 4) for x in full.peak_persistence if np.isfinite(x)], reverse=True)[:12]
    basins = [p for p in range(ref.num_peaks) if (ref.region == p).sum() >= MIN_BASIN]
    # stability over the grid
    ref_sets = {p: set(np.flatnonzero(ref.region == p)) for p in basins}
    stab = {p: [] for p in basins}
    for k, s, tau in itertools.product(KS, SMOOTHS, TAUS):
        _, sp, _ = spine(raw, k, s, tau)
        regs = [set(np.flatnonzero(sp.region == q)) for q in range(sp.num_peaks)]
        for p in basins:
            stab[p].append(max(jac(ref_sets[p], r) for r in regs))
    rows = []
    for p in basins:
        m = ref.region == p
        ex = PROLIF_GENES if sid == 'g2m' else ()
        _, up, dn = effect(m, ~m, exclude=ex)
        js = np.array(stab[p])
        rows.append(dict(
            peak=int(p), n=int(m.sum()),
            persistence=None if not np.isfinite(pers[p]) else round(float(pers[p]), 3),
            peak_value=round(float(ref.peak_values[p]), 3),
            scalar_median=round(float(np.median(raw[m])), 3),
            scalar_range=[round(float(np.percentile(raw[m], 5)), 3), round(float(np.percentile(raw[m], 95)), 3)],
            context={c: round(float(np.median(CONTEXT[c][m])), 3) for c in CONTEXT},
            up=up, down=dn, pam50=comp(m, 'pam50'), er=comp(m, 'er'),
            stable_frac=round(float((js >= 0.5).mean()), 2), jaccard_median=round(float(np.median(js)), 2),
            stable_by_s={f's{sm}': round(float((np.array([j for j, (k, s_, t) in zip(js, itertools.product(KS, SMOOTHS, TAUS)) if s_ == sm]) >= 0.5).mean()), 2) for sm in SMOOTHS},
            stable_k15_only=round(float((np.array([j for j, (k, s, t) in zip(js, itertools.product(KS, SMOOTHS, TAUS)) if k == 15]) >= 0.5).mean()), 2)))
    rows.sort(key=lambda r: -r['n'])
    R['basins'] = rows
    lab = [str(x) for x in ref.region]
    R['assoc'] = {k: cramers_v(lab, vals) for k, vals in
                  [('pam50', pam), ('er', er), ('tss', [m['tss'] for m in meta]), ('plate', [m['plate'] for m in meta])]}
    R['region'] = ref.region.tolist()
    R['scalar_scaled_s'] = v.round(4).tolist()
    out['scalars'][sid] = R

# ---- deviation: pairs with overlapping deviation, different molecular pattern --------
dev = D['deviation']; R = out['scalars']['deviation']; reg = np.array(R['region'])
pairs = []
for a, b in itertools.combinations([r['peak'] for r in R['basins']], 2):
    ma, mb = reg == a, reg == b
    u = mannwhitneyu(dev[ma], dev[mb]); auc = u.statistic / (ma.sum() * mb.sum())
    d, up, dn = effect(ma, mb)
    pairs.append(dict(a=a, b=b, auc=round(float(auc), 3), p=float(u.pvalue),
                      dev_median=[round(float(np.median(dev[ma])), 1), round(float(np.median(dev[mb])), 1)],
                      genes_abs_d_gt1=int((np.abs(d) > 1).sum()), up_in_a=up[:5], up_in_b=[(g, -x) for g, x in dn[:5]],
                      pam50=[comp(ma, 'pam50'), comp(mb, 'pam50')]))
out['deviation_pairs'] = sorted(pairs, key=lambda q: abs(q['auc'] - 0.5))

# ---- proliferation: where do the most proliferative tumours sit? -------------------
g2m = D['scores']['g2m']; R = out['scalars']['g2m']; reg = np.array(R['region'])
hi = g2m >= np.quantile(g2m, 0.75)
out['prolif_top_quartile_by_basin'] = dict(collections.Counter(int(x) for x in reg[hi]).most_common())
out['prolif_top_quartile_by_density_basin'] = dict(collections.Counter(int(x) for x in np.array(out['scalars']['density']['region'])[hi]).most_common())
out['prolif_top_quartile_by_deviation_basin'] = dict(collections.Counter(int(x) for x in np.array(out['scalars']['deviation']['region'])[hi]).most_common())
out['prolif_top_quartile_pam50'] = comp(hi, 'pam50')
hi_pairs = []
for a, b in itertools.combinations([r['peak'] for r in R['basins']], 2):
    ma, mb = reg == a, reg == b
    hi_pairs.append(dict(a=a, b=b,
                         g2m_auc=round(float(mannwhitneyu(g2m[ma], g2m[mb]).statistic / (ma.sum() * mb.sum())), 3),
                         er_early_auc=round(float(mannwhitneyu(CONTEXT['er_early'][ma], CONTEXT['er_early'][mb]).statistic / (ma.sum() * mb.sum())), 3)))
out['prolif_pairs'] = hi_pairs
# is the high-proliferation set one connected region of the graph, or several?
g15 = graphs[REF['k']]
rows15 = np.repeat(np.arange(g15.num_nodes), g15.degree)
out['prolif_superlevel'] = {}
for q in (0.75, 0.90):
    hi_q = g2m >= np.quantile(g2m, q)
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in zip(rows15, g15.indices):
        if hi_q[a] and hi_q[b]:
            parent[find(a)] = find(b)
    comps = collections.Counter(find(i) for i in np.flatnonzero(hi_q))
    big = comps.most_common(1)[0][0]
    m_big = np.zeros(n, bool); m_big[[i for i in np.flatnonzero(hi_q) if find(i) == big]] = True
    groups = {}
    for lab in ('Basal', 'LumB', 'Her2'):
        mm = m_big & (np.array(pam) == lab)
        if mm.sum() >= 5:
            groups[lab] = dict(n=int(mm.sum()), g2m=round(float(np.median(g2m[mm])), 2),
                               er_early=round(float(np.median(CONTEXT['er_early'][mm])), 2))
    out['prolif_superlevel'][str(q)] = dict(n=int(hi_q.sum()), components=sorted(comps.values(), reverse=True)[:5],
                                            largest_pam50=comp(m_big, 'pam50'), groups=groups)
# the unsmoothed proliferation peaks (no persistence gap; for reading, not as findings)
_, sp0, _ = spine(g2m, REF['k'], 0, REF['tau'])
out['prolif_unsmoothed'] = sorted([dict(peak=int(p), n=int((sp0.region == p).sum()),
                                        persistence=None if not np.isfinite(sp0.peak_persistence[p]) else round(float(sp0.peak_persistence[p]), 3),
                                        g2m=round(float(np.median(g2m[sp0.region == p])), 2),
                                        er_early=round(float(np.median(CONTEXT['er_early'][sp0.region == p])), 2),
                                        pam50=comp(sp0.region == p, 'pam50'))
                                   for p in range(sp0.num_peaks) if (sp0.region == p).sum() >= MIN_BASIN], key=lambda r: -r['n'])

# ---- survival (separate outcome analysis) ------------------------------------------
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter
t = np.array([m['os_months'] if m['os_months'] is not None else np.nan for m in meta], dtype=float)
e = np.array([m['os_event'] or 0 for m in meta])
ok = ~np.isnan(t)
out['os'] = {'n': int(ok.sum()), 'events': int(e[ok].sum()), 'median_followup_months': round(float(np.median(t[ok])), 1)}
for sid in SCALARS:
    reg = np.array(out['scalars'][sid]['region']); big = [r['peak'] for r in out['scalars'][sid]['basins']]
    sel = ok & np.isin(reg, big)
    res = multivariate_logrank_test(t[sel], reg[sel], e[sel])
    per = {}
    for p in big:
        m = sel & (reg == p); km = KaplanMeierFitter().fit(t[m], e[m])
        per[p] = dict(n=int(m.sum()), events=int(e[m].sum()), os5=round(float(km.predict(60)), 2))
    out['scalars'][sid]['os'] = dict(p=float(res.p_value), per=per)
pm = np.array(pam); sel = ok & np.isin(pm, ['LumA', 'LumB', 'Basal', 'Her2'])
out['os']['pam50_p'] = float(multivariate_logrank_test(t[sel], pm[sel], e[sel]).p_value)



# ---- write the results -------------------------------------------------------------------
RES = ROOT / "comparison" / "results"
RES.mkdir(parents=True, exist_ok=True)
(RES / "scalar_case_study.json").write_text(json.dumps(
    out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def top(d, k=3):
    return ", ".join(f"{a} {b}" for a, b in list(d.items())[:k])


def pct(d, key):
    tot = sum(v for a, v in d.items() if a != "n/a")
    return f"{100 * d.get(key, 0) / tot:.0f} %" if tot else "n/a"


L = ["# Case study: three scalars on one fixed tumour neighbourhood", "",
     "Data: BC-TCGA, 529 breast tumours. Fixed molecular space: the 1000 genes with the highest variance "
     "across tumours, each standardised; Euclidean distance; kNN graph. Normals (61) are used only for the "
     "healthy reference of the deviation score. Generated by `python3 -m comparison.scalar_case_study`.", "",
     f"Reference setting (fixed before looking at results): k = {REF['k']}, f smoothed x{REF['s']} on the graph, "
     f"peak persistence threshold {REF['tau']} of the range of f; basins with at least {MIN_BASIN} tumours are reported. "
     f"Stability grid: k in {list(KS)} x smoothing in {list(SMOOTHS)} x threshold in {list(TAUS)} "
     f"({len(KS) * len(SMOOTHS) * len(TAUS)} settings). A basin counts as found in a setting when some basin there "
     "shares at least half of the samples (Jaccard >= 0.5).", "",
     f"Deviation check: healthy model of {out['model_rank']} components; deviation median "
     f"{out['tumour_deviation_median']:.0f} for tumours, {out['held_out_normals_median']:.0f} for held-out normals "
     f"({out['held_out_normals_above_tumour_median']} of {out['normals']} held-out normals above the tumour median).", ""]
for sid, S in out["scalars"].items():
    a = S["assoc"]
    L += [f"## {S['label']}", "",
          "| Basin | n | f median (5-95 %) | Persistence | Up (vs rest) | Down | PAM50 | ER+ | Found: all / unsmoothed / x2 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b in S["basins"]:
        pers = "highest" if b["persistence"] is None else f"{b['persistence']:.3f}"
        L.append(f"| {b['peak']} | {b['n']} | {b['scalar_median']:.2f} ({b['scalar_range'][0]:.2f} to {b['scalar_range'][1]:.2f}) "
                 f"| {pers} | {', '.join(g for g, _ in b['up'][:4])} | {', '.join(g for g, _ in b['down'][:4])} "
                 f"| {top(b['pam50'])} | {pct(b['er'], 'ER+')} "
                 f"| {b['stable_frac']:.2f} / {b['stable_by_s']['s0']:.2f} / {b['stable_by_s']['s2']:.2f} |")
    pk = S["peaks_by_setting"]
    L += ["", f"Peaks at persistence 0 / after the threshold, k = 15: unsmoothed {pk['k15_s0'][0]} / {pk['k15_s0'][1]}, "
          f"x1 {pk['k15_s1'][0]} / {pk['k15_s1'][1]}, x2 {pk['k15_s2'][0]} / {pk['k15_s2'][1]}.",
          f"Association of the basins (Cramer's V): PAM50 {a['pam50'][0]:.2f}, ER {a['er'][0]:.2f}, "
          f"tissue source site {a['tss'][0]:.2f} (p = {a['tss'][1]:.2g}), plate {a['plate'][0]:.2f} (p = {a['plate'][1]:.2g}).", ""]
    if sid == "g2m":
        L += ["Distinguishing genes for proliferation basins leave out the G2M and E2F gene sets "
              "(their difference follows from how f is built).", ""]

L += ["## Deviation: similar magnitude, different molecular pattern", "",
      "Pairs of deviation basins, most overlapping deviation first. AUC: probability that a tumour of the first "
      "basin deviates more than one of the second (0.5 = same distribution). Genes |d| > 1: genes whose mean differs "
      "by more than one pooled standard deviation between the two basins.", "",
      "| Basins | Deviation medians | AUC | Genes \\|d\\| > 1 | Up in first | Up in second | PAM50 first / second |",
      "|---|---|---|---|---|---|---|"]
for q in out["deviation_pairs"]:
    L.append(f"| {q['a']} vs {q['b']} | {q['dev_median'][0]:.0f} / {q['dev_median'][1]:.0f} | {q['auc']:.2f} | "
             f"{q['genes_abs_d_gt1']} | {', '.join(g for g, _ in q['up_in_a'][:3])} | {', '.join(g for g, _ in q['up_in_b'][:3])} | "
             f"{top(q['pam50'][0], 2)} / {top(q['pam50'][1], 2)} |")
L += ["", "## Proliferation: one region or several?", "",
      f"Top quartile of G2M score ({sum(out['prolif_top_quartile_by_basin'].values())} tumours) by basin: "
      f"proliferation spine {out['prolif_top_quartile_by_basin']}, density spine {out['prolif_top_quartile_by_density_basin']}, "
      f"deviation spine {out['prolif_top_quartile_by_deviation_basin']}; PAM50 {top(out['prolif_top_quartile_pam50'], 5)}.", ""]
for q, v in out["prolif_superlevel"].items():
    grp = "; ".join(f"{k} {g['n']} (G2M {g['g2m']:+.2f}, estrogen response {g['er_early']:+.2f})" for k, g in v["groups"].items())
    L.append(f"* G2M >= its {float(q) * 100:.0f}th percentile: {v['n']} tumours; connected pieces in the k = {REF['k']} graph "
             f"{v['components']}; inside the largest: {grp}.")
L += ["", f"Unsmoothed proliferation basins (k = {REF['k']}, threshold {REF['tau']}; no persistence gap, so read as a map, not as findings):", "",
      "| Basin | n | Persistence | G2M median | Estrogen response median | PAM50 |", "|---|---|---|---|---|---|"]
for r in out["prolif_unsmoothed"]:
    L.append(f"| {r['peak']} | {r['n']} | {'highest' if r['persistence'] is None else r['persistence']} | {r['g2m']:+.2f} | "
             f"{r['er_early']:+.2f} | {top(r['pam50'], 3)} |")
L += ["",
      "## Survival (separate outcome analysis)", "",
      f"Overall survival, {out['os']['n']} tumours, {out['os']['events']} deaths, median follow-up "
      f"{out['os']['median_followup_months']} months. Log-rank across PAM50 (LumA, LumB, Her2, Basal): "
      f"p = {out['os']['pam50_p']:.2g}."]
for sid, S in out["scalars"].items():
    per = ", ".join(f"basin {p}: {v['os5']:.2f} (n {v['n']})" for p, v in S["os"]["per"].items())
    L.append(f"* {S['label']} basins: log-rank p = {S['os']['p']:.2g}; 5-year OS {per}.")
L += ["", "Notes: PAM50 and ER are revealed after the genes, never used to build f or the graph. Subtype names in the "
      "text are read from these labels. Persistence is prominence in f, not biological importance; the graph's links "
      "are relations in the sampled scalar landscape, not tumour progression."]
(RES / "scalar_case_study.md").write_text("\n".join(L) + "\n")
print(f"wrote {RES / 'scalar_case_study.md'} and .json")
