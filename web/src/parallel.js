// Parallel coordinates: the data model, without any drawing (parallelView.js draws it).
//
// Every point of the field is one polyline across vertical axes. The axes are
//   - the original coordinates: x1..xd of a point cloud (``<name>.coords.json``,
//     written by web/build_data.py), or x, y, z of a grid voxel (from the shape);
//   - the field value f;
//   - "peak": the visible peak whose region holds the point (after simplification),
//     one band per peak, highest peak at the top. Inside its band a point sits by
//     its value relative to the rest of that region, so the lines still show order.
// Unlike the scatterplot there is no dimensionality reduction: every original
// dimension gets its own axis.
//
// A column holds its values normalised to [0, 1] (``v``) and the range ``lo``..``hi``
// that maps them back for labels. Brushes are ranges in that normalised space.

/** Decode one axis of a coords file: base64 of little-endian uint16, to [0, 1]. */
export function decodeAxis(b64, n) {
  const bin = atob(b64);
  if (bin.length !== 2 * n) throw new Error(`axis has ${bin.length / 2} values, expected ${n}`);
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = (bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8)) / 65535;
  return v;
}

/**
 * The fixed columns of ``doc``: coordinates (when known) and the field value.
 * ``coords`` is a parsed coords file (point clouds) or null. Returns
 * ``{columns, note}``; ``note`` explains a missing part.
 */
export function baseColumns(doc, coords = null) {
  const n = doc.meta.num_points;
  const values = doc.data?.values;
  const columns = [];
  let note = '';
  if (doc.meta.kind === 'grid') {
    const [nz, ny, nx] = doc.data?.shape ?? doc.meta.grid_shape;
    const [sx, sy, sz] = doc.data?.spacing ?? doc.meta.spacing ?? [1, 1, 1];
    const cols = [['x', nx, sx, (i) => i % nx], ['y', ny, sy, (i) => Math.floor(i / nx) % ny],
                  ['z', nz, sz, (i) => Math.floor(i / (nx * ny))]];
    for (const [key, size, step, of] of cols) {
      const v = new Float32Array(n), d = Math.max(size - 1, 1);
      for (let i = 0; i < n; i++) v[i] = of(i) / d;
      columns.push({ key, label: key, v, lo: 0, hi: (size - 1) * step, kind: 'coord' });
    }
  } else if (coords) {
    if (coords.num_points !== n) throw new Error(`coords file has ${coords.num_points} points, the graph ${n}`);
    for (const a of coords.axes) {
      columns.push({ key: `c:${a.name}`, label: a.name, v: decodeAxis(a.data, n), lo: a.min, hi: a.max, kind: 'coord' });
    }
  } else {
    note = 'No original coordinates for this file (run python3 web/build_data.py --coords-only); ' +
           'only the value and peak axes are shown.';
  }
  if (values) {
    let [lo, hi] = doc.meta.value_range ?? [Infinity, -Infinity];
    if (!doc.meta.value_range) for (const x of values) { if (x < lo) lo = x; if (x > hi) hi = x; }
    const span = hi - lo || 1, v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = (values[i] - lo) / span;
    columns.push({ key: 'value', label: 'f (value)', v, lo, hi, kind: 'value' });
  }
  return { columns, note };
}

/**
 * The "peak" column for the visible graph ``vis``: one band per visible peak,
 * ordered by peak value (highest at the top), points spread inside their band by
 * their value relative to their region. Also returns the bands for labelling.
 */
export function peakColumn(doc, vis) {
  const n = doc.meta.num_points, P = vis.peaks.length;
  const owner = vis.owner, region = doc.region, values = doc.data?.values;
  const order = vis.peaks.map((_, i) => i).sort((a, b) =>
    doc.peaks[vis.peaks[b]].value - doc.peaks[vis.peaks[a]].value || a - b);
  const rank = new Int32Array(P);
  order.forEach((p, r) => { rank[p] = r; });
  const lo = new Float64Array(P).fill(Infinity), hi = new Float64Array(P).fill(-Infinity);
  const of = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const p = owner[region[i]];
    of[i] = p;
    if (values) { const x = values[i]; if (x < lo[p]) lo[p] = x; if (x > hi[p]) hi[p] = x; }
  }
  const v = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = of[i], t = values && hi[p] > lo[p] ? (values[i] - lo[p]) / (hi[p] - lo[p]) : 0.5;
    v[i] = 1 - (rank[p] + 0.1 + 0.8 * (1 - t)) / P;       // band of rank r spans [1-(r+1)/P, 1-r/P]
  }
  const bands = order.map((p, r) => ({ index: p, id: vis.peaks[p], top: 1 - r / P, bottom: 1 - (r + 1) / P }));
  return { key: 'peak', label: 'peak', v, lo: 0, hi: 1, kind: 'peak', bands };
}

/** Where a column's value is drawn, in [0, 1] from the bottom, honouring a flipped axis. */
export const shown = (col, v) => (col.flipped ? 1 - v : v);

/**
 * Points that pass every brush (``Map`` key -> [lo, hi] in normalised value), as a
 * Uint8Array of 0/1, or null when there are no brushes.
 */
export function brushMask(columns, brushes, n) {
  const active = columns.filter((c) => brushes.has(c.key));
  if (!active.length) return null;
  const mask = new Uint8Array(n).fill(1);
  for (const c of active) {
    const [a, b] = brushes.get(c.key), v = c.v;
    for (let i = 0; i < n; i++) if (mask[i] && (v[i] < a || v[i] > b)) mask[i] = 0;
  }
  return mask;
}

/** AND of two optional masks (null = everything). */
export function andMask(a, b) {
  if (!a) return b;
  if (!b) return a;
  const m = new Uint8Array(a.length);
  for (let i = 0; i < a.length; i++) m[i] = a[i] & b[i];
  return m;
}

/**
 * Index buffer for ``LineSegments`` over the points with ``mask[i] === want``
 * (every point when mask is null): with A axes, point i owns vertices i*A..i*A+A-1
 * and draws A-1 segments between consecutive ones.
 */
export function lineIndex(mask, n, A, want = 1) {
  let count = 0;
  if (!mask) count = n;
  else for (let i = 0; i < n; i++) if (mask[i] === want) count++;
  const idx = new Uint32Array(count * 2 * (A - 1));
  let w = 0;
  for (let i = 0; i < n; i++) {
    if (mask && mask[i] !== want) continue;
    const base = i * A;
    for (let k = 0; k < A - 1; k++) { idx[w++] = base + k; idx[w++] = base + k + 1; }
  }
  return idx;
}

/**
 * The point whose polyline passes nearest to (x, y) in pixels, within ``radius``,
 * or -1. ``axisX`` are the axes' pixel x in drawing order, ``yOf(col, i)`` a point's
 * pixel y on a column. Only points with ``mask[i]`` are considered (all if null).
 */
export function pickLine(x, y, cols, axisX, yOf, n, mask = null, radius = 6) {
  const A = cols.length;
  if (A === 0) return -1;
  let k = 0;
  while (k < A - 2 && x > axisX[k + 1]) k++;
  let t = A === 1 ? 0 : (x - axisX[k]) / (axisX[k + 1] - axisX[k]);
  t = Math.min(1, Math.max(0, t));
  const c0 = cols[k], c1 = cols[Math.min(k + 1, A - 1)];
  let best = -1, bestD = radius;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const d = Math.abs(yOf(c0, i) * (1 - t) + yOf(c1, i) * t - y);
    if (d <= bestD) { bestD = d; best = i; }
  }
  return best;
}

/** Brushed points per visible peak, largest first: [{index, id, count}]. */
export function peakCounts(doc, vis, mask) {
  const counts = new Map();
  const owner = vis.owner, region = doc.region;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const p = owner[region[i]];
    counts.set(p, (counts.get(p) ?? 0) + 1);
  }
  return [...counts].map(([index, count]) => ({ index, id: vis.peaks[index], count }))
    .sort((a, b) => b.count - a.count || a.id - b.id);
}

/** Default line opacity: fewer points, more opaque. */
export function defaultOpacity(n) { return Math.min(0.8, Math.max(0.04, 40 / Math.sqrt(Math.max(n, 1)))); }
