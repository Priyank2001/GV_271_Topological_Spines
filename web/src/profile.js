// Peak profile: what the points of one peak's region look like, dimension by dimension.
//
// For every column of the field (the original coordinates x1..xd of a point cloud, or x, y, z
// of a grid voxel, and the value f; see ``baseColumns`` in parallel.js) it compares the points
// of the region with all points: a histogram of each, the region's median and 5-95 % range,
// and how far the region sits from the rest (``d``: difference of the means in units of the
// spread of all points, an effect size). Sorting by |d| puts first the dimensions that set
// this peak apart, which is what a user asks of a peak ("what characterises it?").
//
// No drawing here (main.js draws the panel), so it can be tested on its own.

import { baseColumns } from './parallel.js';

export const PROFILE_BINS = 24;

/**
 * The columns of ``doc`` with the statistics of all points: mean, standard deviation and a
 * histogram over the normalised range [0, 1]. ``coords``: a parsed coords file or null.
 */
export function profileColumns(doc, coords = null) {
  const { columns, note } = baseColumns(doc, coords);
  for (const c of columns) {
    const n = c.v.length;
    let s = 0, s2 = 0;
    const hist = new Float64Array(PROFILE_BINS);
    for (let i = 0; i < n; i++) {
      const x = c.v[i];
      s += x; s2 += x * x;
      hist[Math.min(PROFILE_BINS - 1, Math.max(0, Math.floor(x * PROFILE_BINS)))]++;
    }
    c.mean = n ? s / n : 0;
    c.std = n ? Math.sqrt(Math.max(0, s2 / n - c.mean * c.mean)) : 0;
    c.hist = hist;
  }
  return { columns, note };
}

/** Value ``q`` (0..1) of the sorted array ``a`` (linear between neighbours). */
function quantile(a, q) {
  if (!a.length) return NaN;
  const x = q * (a.length - 1), i = Math.floor(x), f = x - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}

/**
 * Profile of visible peak ``k`` (index into ``vis.peaks``): its region's points in every column.
 * Returns ``{ n, total, rows }``; each row: ``{ key, label, kind, lo, hi, hist, all, mean, median,
 * q05, q95, d, spread }`` with mean/median/q05/q95 in the column's own units, ``d`` the effect
 * size (region mean minus all-points mean, over the all-points standard deviation) and
 * ``spread`` the region's standard deviation over that of all points (< 1: concentrated).
 */
export function peakProfile(columns, doc, vis, k) {
  const region = doc.region, owner = vis.owner, n = doc.meta.num_points;
  const members = [];
  for (let i = 0; i < n; i++) if (owner[region[i]] === k) members.push(i);
  const rows = columns.map((c) => {
    const span = c.hi - c.lo;
    const vals = new Float32Array(members.length);
    const hist = new Float64Array(PROFILE_BINS);
    let s = 0, s2 = 0;
    members.forEach((i, j) => {
      const x = c.v[i];
      vals[j] = x; s += x; s2 += x * x;
      hist[Math.min(PROFILE_BINS - 1, Math.max(0, Math.floor(x * PROFILE_BINS)))]++;
    });
    vals.sort();
    const m = members.length ? s / members.length : NaN;
    const sd = members.length ? Math.sqrt(Math.max(0, s2 / members.length - m * m)) : NaN;
    const toUnits = (x) => c.lo + x * span;
    return {
      key: c.key, label: c.label, kind: c.kind, lo: c.lo, hi: c.hi, hist, all: c.hist,
      mean: toUnits(m), median: toUnits(quantile(vals, 0.5)),
      q05: toUnits(quantile(vals, 0.05)), q95: toUnits(quantile(vals, 0.95)),
      d: c.std > 1e-12 ? (m - c.mean) / c.std : 0,
      spread: c.std > 1e-12 ? sd / c.std : 1,
    };
  });
  return { n: members.length, total: n, rows };
}

/** The rows to show: the value first, then the coordinates by |d| (or in their own order). */
export function orderRows(rows, { sort = 'distinctive', limit = Infinity } = {}) {
  const value = rows.filter((r) => r.kind === 'value');
  let rest = rows.filter((r) => r.kind !== 'value');
  if (sort === 'distinctive') rest = [...rest].sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
  return [...value, ...rest.slice(0, limit)];
}
