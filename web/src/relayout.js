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
 */

export const MAX_RELAYOUT_NODES = 150;   // cost grows with the square of the node count; above this the
                                         // exported layout is used (it was made for a full, larger graph)

/**
 * @param doc    exported document (needs ``position`` on peaks and saddles, and ``spine``)
 * @param peaks  peak ids to lay out
 * @param saddles saddle ids to lay out
 * @param o      {maxIter = 300, tol = 1e-5, maxNodes = MAX_RELAYOUT_NODES}
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

  const spacing = doc.meta?.spacing;
  const X = [];
  for (const [k, id] of nodes) {
    const p = (k === 'p' ? doc.peaks : doc.saddles)[id]?.position;
    if (!Array.isArray(p)) return null;
    X.push(Array.isArray(spacing) && spacing.length === p.length ? p.map((v, i) => v * spacing[i]) : p);
  }

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
  nodes.forEach(([k, id], i) => {
    (k === 'p' ? res.peaks : res.saddles).set(id, [fit[i][0] / scale, fit[i][1] / scale]);
  });
  return res;
}

/** Localized stress majorization from ``start`` (copied). */
function majorize(start, D, W, n, maxIter, stopMove) {
  const Y = start.map(([x, y]) => [x, y]);
  for (let it = 0; it < maxIter; it++) {
    let moved = 0;
    for (let i = 0; i < n; i++) {
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
