/**
 * Re-layout of the spine after simplification.
 *
 * The exported layout (``doc.spine``) is computed once, for the full graph. After the
 * persistence slider merges peaks away, the nodes that are left keep positions chosen
 * while many other nodes were crowding around them, so symmetric structures (the cross of
 * four peaks around the head of the fuel jet, Fig. 1 of Correa et al.) come out lopsided.
 *
 * ``relayout`` lays out only the given nodes again, with the same method as
 * pipeline/spine_layout.py: weighted stress, w_ij = 1 / d_ij^2, d_ij the distance between
 * the nodes in the data space (their ``position``). It minimises the stress with
 * localized stress majorization (Gansner, Koren and North 2004, eq. 9: each node moves to
 * the weighted average of where its distances to the others say it should be; every update
 * cannot increase the stress). It runs twice, from the exported positions and from a PCA of
 * the nodes' positions, and keeps the result with the lower stress. That result is rotated
 * (and mirrored if that fits better) and shifted onto the exported positions (Procrustes),
 * so the picture keeps its orientation and does not jump.
 *
 * Optional ``extra`` nodes (the merged peaks and saddles that the p* arcs pass through) are
 * placed afterwards, in a second pass that moves only them and keeps the main nodes fixed,
 * so they cannot pull the visible structure out of shape.
 */

export const MAX_RELAYOUT_NODES = 150;   // cost grows with the square of the node count; above this the
                                         // exported layout is used (it was made for a full, larger graph)

/**
 * @param doc    exported document (needs ``position`` on peaks and saddles, and ``spine``)
 * @param peaks  peak ids to lay out
 * @param saddles saddle ids to lay out
 * @param o      {maxIter = 300, tol = 1e-5, maxNodes = MAX_RELAYOUT_NODES,
 *                extraPeaks = [], extraSaddles = [] (placed second, main nodes fixed;
 *                skipped if main + extra exceeds 2 x maxNodes)}
 * @returns {peaks: Map(id -> [x, y]), saddles: Map(id -> [x, y]), stressBefore, stressAfter}
 *          in layout units, or null when the nodes cannot or need not be laid out again
 *          (too many, too few, or no positions in the file).
 */
export function relayout(doc, peaks, saddles, o = {}) {
  const maxIter = o.maxIter ?? 300, tol = o.tol ?? 1e-5, maxNodes = o.maxNodes ?? MAX_RELAYOUT_NODES;
  const sp = doc.spine;
  const nodes = [...peaks.map((id) => ['p', id]), ...saddles.map((id) => ['s', id])];
  const n = nodes.length;
  if (!sp || n < 3 || n > maxNodes) return null;

  const X = positions(doc, nodes);
  if (!X) return null;

  // target distances; equal positions are lifted off zero (as _weights in spine_layout.py)
  const D = new Float64Array(n * n);
  const pos = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let c = 0; c < X[i].length; c++) { const t = X[i][c] - X[j][c]; s += t * t; }
      D[i * n + j] = D[j * n + i] = Math.sqrt(s);
      if (s > 0) pos.push(Math.sqrt(s));
    }
  }
  if (!pos.length) return null;
  pos.sort((a, b) => a - b);
  const floor = 1e-3 * pos[pos.length >> 1];
  const W = new Float64Array(n * n);
  for (let k = 0; k < n * n; k++) {
    if (Math.floor(k / n) === k % n) continue;
    D[k] = Math.max(D[k], floor);
    W[k] = 1 / (D[k] * D[k]);
  }

  // start: the exported positions, in data units
  const scale = sp.scale > 0 ? sp.scale : 1;
  const base = nodes.map(([k, id]) => (k === 'p' ? sp.peaks : sp.saddles)[id]);
  const Y = base.map(([x, y]) => [x * scale, y * scale]);
  const stressBefore = stress(Y, D, W, n);
  const median = pos[pos.length >> 1];
  const fromBase = majorize(Y, D, W, n, maxIter, tol * median);
  const fromPca = majorize(pcaStart(X), D, W, n, maxIter, tol * median);
  const sB = stress(fromBase, D, W, n), sP = stress(fromPca, D, W, n);
  let out = sP < sB ? fromPca : fromBase;
  let stressAfter = Math.min(sB, sP);
  if (!(stressAfter <= stressBefore)) { out = base.map(([x, y]) => [x * scale, y * scale]); stressAfter = stressBefore; }

  // rotate (or mirror) and shift onto the exported positions, back to layout units
  const target = base.map(([x, y]) => [x * scale, y * scale]);
  const straight = procrustes(out, target);
  const mirrored = procrustes(out.map(([x, y]) => [x, -y]), target);
  const fit = misfit(mirrored, target) < misfit(straight, target) ? mirrored : straight;
  const res = { peaks: new Map(), saddles: new Map(), stressBefore, stressAfter };
  const put = ([k, id], [x, y]) => (k === 'p' ? res.peaks : res.saddles).set(id, [x / scale, y / scale]);
  nodes.forEach((node, i) => put(node, fit[i]));

  // second pass: the extra nodes, around the main ones held fixed
  const main = new Set(nodes.map(([k, id]) => k + id));
  const extra = [...(o.extraPeaks ?? []).map((id) => ['p', id]), ...(o.extraSaddles ?? []).map((id) => ['s', id])]
    .filter(([k, id]) => !main.has(k + id));
  if (!extra.length) return res;
  const all = [...nodes, ...extra];
  const XA = positions(doc, all);
  if (!XA || all.length > 2 * maxNodes) {
    // cannot place them: keep their exported positions, carried along by the same fit
    const moved = alignLike(base.map(([x, y]) => [x * scale, y * scale]), fit,
      extra.map(([k, id]) => (k === 'p' ? sp.peaks : sp.saddles)[id]).map(([x, y]) => [x * scale, y * scale]));
    extra.forEach((node, i) => put(node, moved[i]));
    return res;
  }
  const { D: DA, W: WA } = targets(XA);
  const nA = all.length;
  const startExtra = alignLike(base.map(([x, y]) => [x * scale, y * scale]), fit,
    extra.map(([k, id]) => (k === 'p' ? sp.peaks : sp.saddles)[id]).map(([x, y]) => [x * scale, y * scale]));
  const YA = [...fit.map(([x, y]) => [x, y]), ...startExtra];
  const movable = all.map((_, i) => i >= n);
  const medA = medianOf(DA, nA);
  const placed = majorize(YA, DA, WA, nA, maxIter, tol * medA, movable);
  extra.forEach((node, i) => put(node, placed[n + i]));
  return res;
}

/** Positions of the nodes in the data space (times the voxel spacing), or null if any is missing. */
function positions(doc, nodes) {
  const spacing = doc.meta?.spacing;
  const X = [];
  for (const [k, id] of nodes) {
    const p = (k === 'p' ? doc.peaks : doc.saddles)[id]?.position;
    if (!Array.isArray(p)) return null;
    X.push(Array.isArray(spacing) && spacing.length === p.length ? p.map((v, i) => v * spacing[i]) : p);
  }
  return X;
}

/** Target distances (zeros lifted to 1e-3 x median) and weights 1 / d^2, for the rows of X. */
function targets(X) {
  const n = X.length;
  const D = new Float64Array(n * n), W = new Float64Array(n * n);
  const pos = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    let s = 0;
    for (let c = 0; c < X[i].length; c++) { const t = X[i][c] - X[j][c]; s += t * t; }
    D[i * n + j] = D[j * n + i] = Math.sqrt(s);
    if (s > 0) pos.push(Math.sqrt(s));
  }
  pos.sort((a, b) => a - b);
  const floor = 1e-3 * (pos.length ? pos[pos.length >> 1] : 1);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i === j) continue;
    D[i * n + j] = Math.max(D[i * n + j], floor);
    W[i * n + j] = 1 / (D[i * n + j] ** 2);
  }
  return { D, W };
}

function medianOf(D, n) {
  const v = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) v.push(D[i * n + j]);
  v.sort((a, b) => a - b);
  return v.length ? v[v.length >> 1] : 1;
}

/**
 * Apply to ``pts`` the similarity (rotation, mirror, shift) that best carries ``from`` onto
 * ``to``, so nodes laid out with the old layout move along with the new one.
 */
function alignLike(from, to, pts) {
  const n = from.length;
  const mf = [0, 0], mt = [0, 0];
  for (let i = 0; i < n; i++) { mf[0] += from[i][0] / n; mf[1] += from[i][1] / n; mt[0] += to[i][0] / n; mt[1] += to[i][1] / n; }
  const best = [];
  for (const flip of [1, -1]) {
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      const ax = from[i][0] - mf[0], ay = flip * (from[i][1] - mf[1]), bx = to[i][0] - mt[0], by = to[i][1] - mt[1];
      sxx += ax * bx + ay * by; sxy += ax * by - ay * bx;
    }
    const th = Math.atan2(sxy, sxx), c = Math.cos(th), s = Math.sin(th);
    const map = ([x, y]) => { const ax = x - mf[0], ay = flip * (y - mf[1]); return [c * ax - s * ay + mt[0], s * ax + c * ay + mt[1]]; };
    best.push({ map, err: misfit(from.map(map), to) });
  }
  const { map } = best[0].err <= best[1].err ? best[0] : best[1];
  return pts.map(map);
}

/** Localized stress majorization from ``start`` (copied); only nodes with ``movable[i]`` move. */
function majorize(start, D, W, n, maxIter, stopMove, movable = null) {
  const Y = start.map(([x, y]) => [x, y]);
  for (let it = 0; it < maxIter; it++) {
    let moved = 0;
    for (let i = 0; i < n; i++) {
      if (movable && !movable[i]) continue;
      let sx = 0, sy = 0, sw = 0;
      const [xi, yi] = Y[i];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const w = W[i * n + j];
        const dx = xi - Y[j][0], dy = yi - Y[j][1];
        const len = Math.hypot(dx, dy) || 1e-12;
        const r = D[i * n + j] / len;
        sx += w * (Y[j][0] + r * dx);
        sy += w * (Y[j][1] + r * dy);
        sw += w;
      }
      const nx = sx / sw, ny = sy / sw;
      moved = Math.max(moved, Math.hypot(nx - xi, ny - yi));
      Y[i][0] = nx; Y[i][1] = ny;
    }
    if (moved < stopMove) break;
  }
  return Y;
}

/** First two principal coordinates of the rows of ``X`` (power iteration on the Gram matrix). */
function pcaStart(X) {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0);
  for (const r of X) for (let c = 0; c < d; c++) mean[c] += r[c] / n;
  const C = X.map((r) => r.map((v, c) => v - mean[c]));
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    let s = 0;
    for (let c = 0; c < d; c++) s += C[i][c] * C[j][c];
    G[i * n + j] = G[j * n + i] = s;
  }
  const out = Array.from({ length: n }, () => [0, 0]);
  const vecs = [];
  for (let k = 0; k < 2; k++) {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i * 12.9898 + k * 78.233 + 1));   // fixed start
    let lam = 0;
    for (let it = 0; it < 200; it++) {
      const w = new Array(n).fill(0);
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += G[i * n + j] * v[j]; w[i] = s; }
      for (const u of vecs) { const dot = w.reduce((a, x, i) => a + x * u[i], 0); for (let i = 0; i < n; i++) w[i] -= dot * u[i]; }
      const norm = Math.hypot(...w);
      if (!(norm > 0)) break;
      lam = norm;
      v = w.map((x) => x / norm);
    }
    vecs.push(v);
    for (let i = 0; i < n; i++) out[i][k] = v[i] * Math.sqrt(lam);
  }
  return out;
}

/** Sum of squared distances between matching points. */
function misfit(A, B) {
  let s = 0;
  for (let i = 0; i < A.length; i++) s += (A[i][0] - B[i][0]) ** 2 + (A[i][1] - B[i][1]) ** 2;
  return s;
}

/** Normalised weighted stress, sum w (|y_i - y_j| - d)^2 / sum w d^2. */
function stress(Y, D, W, n) {
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = W[i * n + j], d = D[i * n + j];
      const l = Math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1]);
      num += w * (l - d) ** 2; den += w * d * d;
    }
  }
  return den > 0 ? num / den : 0;
}

/** Rotate and translate ``A`` (no scaling, no reflection) to best match ``B``. */
function procrustes(A, B) {
  const n = A.length;
  const ma = [0, 0], mb = [0, 0];
  for (let i = 0; i < n; i++) { ma[0] += A[i][0] / n; ma[1] += A[i][1] / n; mb[0] += B[i][0] / n; mb[1] += B[i][1] / n; }
  let sxx = 0, sxy = 0;                       // cross terms of the centred sets
  for (let i = 0; i < n; i++) {
    const ax = A[i][0] - ma[0], ay = A[i][1] - ma[1], bx = B[i][0] - mb[0], by = B[i][1] - mb[1];
    sxx += ax * bx + ay * by;
    sxy += ax * by - ay * bx;
  }
  const th = Math.atan2(sxy, sxx), c = Math.cos(th), s = Math.sin(th);
  return A.map(([x, y]) => {
    const ax = x - ma[0], ay = y - ma[1];
    return [c * ax - s * ay + mb[0], s * ax + c * ay + mb[1]];
  });
}
