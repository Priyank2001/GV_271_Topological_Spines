// Clustering of the points, to set beside the peak regions of the extremum graph.
//
// The thesis this tool is compared with (and NDDAV) offers ordinary clustering next to the
// topology; here it serves one question: do the peaks find the same groups as KMeans or DBSCAN?
// ``agreement`` answers it with the adjusted Rand index (1: the same grouping, 0: no better
// than chance) and a table of which cluster lands in which peak.
//
// Input: ``cols``, an array of columns, each a Float32Array with one value per point in [0, 1]
// (the normalised columns of the peak profile, see profile.js). Everything is deterministic: the
// random start of KMeans comes from a seeded generator, so the same settings give the same result.

/** Small seeded random generator (mulberry32). */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Squared distance between point i and the centre c (a Float64Array of length d). */
function dist2(cols, i, c) {
  let s = 0;
  for (let j = 0; j < cols.length; j++) { const t = cols[j][i] - c[j]; s += t * t; }
  return s;
}

/**
 * KMeans with k-means++ seeding. Returns ``{ labels: Int32Array, k, iterations }``.
 * ``maxIter`` rounds of Lloyd's updates at most; stops when no label changes.
 */
export function kmeans(cols, k, { maxIter = 60, seed = 1 } = {}) {
  const n = cols[0]?.length ?? 0, d = cols.length;
  k = Math.max(1, Math.min(k, n));
  const rand = rng(seed);
  const centres = [];
  const take = (i) => Float64Array.from(cols, (c) => c[i]);
  centres.push(take(Math.floor(rand() * n)));
  const best = new Float64Array(n).fill(Infinity);
  while (centres.length < k) {                         // k-means++: far points are likelier
    const c = centres[centres.length - 1];
    let total = 0;
    for (let i = 0; i < n; i++) { const t = dist2(cols, i, c); if (t < best[i]) best[i] = t; total += best[i]; }
    let r = rand() * total, pick = n - 1;
    for (let i = 0; i < n; i++) { r -= best[i]; if (r <= 0) { pick = i; break; } }
    centres.push(take(pick));
  }
  const labels = new Int32Array(n).fill(-1);
  let it = 0;
  for (; it < maxIter; it++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const t = dist2(cols, i, centres[c]); if (t < bd) { bd = t; bi = c; } }
      if (labels[i] !== bi) { labels[i] = bi; changed++; }
    }
    const sum = Array.from({ length: k }, () => new Float64Array(d)), cnt = new Float64Array(k);
    for (let i = 0; i < n; i++) { const c = labels[i]; cnt[c]++; for (let j = 0; j < d; j++) sum[c][j] += cols[j][i]; }
    for (let c = 0; c < k; c++) if (cnt[c]) for (let j = 0; j < d; j++) centres[c][j] = sum[c][j] / cnt[c];
    if (!changed) break;
  }
  return { labels, k, iterations: it + 1 };
}

/**
 * DBSCAN: points with at least ``minPts`` neighbours within ``eps`` (Euclidean, in the [0, 1]
 * units of the columns) are cores; clusters are cores linked through each other, plus the points
 * next to them; the rest is noise (label -1). Returns ``{ labels, k, noise }``.
 * Brute-force neighbour search: n^2 x d work, so callers limit it (see ``dbscanCost``).
 */
export function dbscan(cols, eps, minPts) {
  const n = cols[0]?.length ?? 0, d = cols.length, e2 = eps * eps;
  const neighbours = (i) => {
    const out = [];
    for (let m = 0; m < n; m++) {
      let s = 0;
      for (let j = 0; j < d && s <= e2; j++) { const t = cols[j][i] - cols[j][m]; s += t * t; }
      if (s <= e2) out.push(m);
    }
    return out;
  };
  const labels = new Int32Array(n).fill(-2);           // -2: not visited yet
  let k = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue;
    const nb = neighbours(i);
    if (nb.length < minPts) { labels[i] = -1; continue; }
    labels[i] = k;
    const queue = nb.filter((m) => m !== i);
    for (let q = 0; q < queue.length; q++) {
      const m = queue[q];
      if (labels[m] === -1) labels[m] = k;              // noise next to a core: a border point
      if (labels[m] !== -2) continue;
      labels[m] = k;
      const nb2 = neighbours(m);
      if (nb2.length >= minPts) for (const x of nb2) if (labels[x] === -2 || labels[x] === -1) queue.push(x);
    }
    k++;
  }
  let noise = 0;
  for (let i = 0; i < n; i++) if (labels[i] < 0) { labels[i] = -1; noise++; }
  return { labels, k, noise };
}

/** Work of a brute-force DBSCAN (distance terms), to refuse runs that would freeze the page. */
export function dbscanCost(n, d) { return n * n * d; }

/**
 * How well clusters ``a`` and peak regions ``b`` (two labelings of the same points) agree.
 * Noise (label -1 in ``a``) counts as a group of its own. Returns ``{ ari, table }``: the adjusted
 * Rand index and, per cluster, its size and the peak holding most of it with that share.
 */
export function agreement(a, b) {
  const n = a.length;
  const key = (x, y) => `${x}|${y}`;
  const cells = new Map(), ra = new Map(), rb = new Map();
  for (let i = 0; i < n; i++) {
    cells.set(key(a[i], b[i]), (cells.get(key(a[i], b[i])) ?? 0) + 1);
    ra.set(a[i], (ra.get(a[i]) ?? 0) + 1);
    rb.set(b[i], (rb.get(b[i]) ?? 0) + 1);
  }
  const c2 = (m) => (m * (m - 1)) / 2;
  let sIJ = 0, sA = 0, sB = 0;
  for (const v of cells.values()) sIJ += c2(v);
  for (const v of ra.values()) sA += c2(v);
  for (const v of rb.values()) sB += c2(v);
  const expected = (sA * sB) / (c2(n) || 1), max = (sA + sB) / 2;
  const ari = max === expected ? 1 : (sIJ - expected) / (max - expected);
  const table = [...ra.entries()].map(([cluster, size]) => {
    let bestPeak = null, bestCount = 0;
    for (const [k, v] of cells) {
      const [x, y] = k.split('|').map(Number);
      if (x === cluster && v > bestCount) { bestCount = v; bestPeak = y; }
    }
    return { cluster, size, peak: bestPeak, share: bestCount / size };
  }).sort((p, q) => (p.cluster < 0) - (q.cluster < 0) || q.size - p.size);
  return { ari, table };
}
