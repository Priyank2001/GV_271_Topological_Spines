// Paper-style topological spine geometry (Correa et al. 2011, section 4).
//
// Pure functions, no Three.js: they return flat vertex arrays that spineView.js
// uploads to one mesh. That also lets tests draw the result without a GPU.
//
// What is drawn, per visible peak u and isovalue c (lowest c first, so higher
// contours are painted over lower ones and the nesting is preserved):
//
//   * for each saddle s linked to u, the "canonical link" u -> s (section 4.1):
//     the Minkowski sum of circles centred on the arc, i.e. the region swept by
//     a circle whose radius changes along the arc. Its outline is the exact
//     envelope of those circles, plus the circles at both ends and at bends.
//   * a peak without links gets just its circle.
//
// Radius (section 4.2): the circle at a point p of the arc has radius
//
//     r_us(c, p) = rs * V(c, f(p))           V = |{x in D(u) : min(c,f(p)) < f(x) < max(c,f(p))}|
//
// where D(u) is u's whole region (descending manifold), and f(p) runs linearly
// from f(u) at the peak to f(s) at the saddle. So:
//
//   c = f(s)  ->  radius 0 at the saddle: the figure-eight "critical contour";
//   c < f(s)  ->  a fat shape that surrounds the whole link (contours below);
//   c > f(s)  ->  a teardrop that ends where f(p) = c (contours above).
//
// Each lobe uses its own D(u), so the two lobes of a link may have different
// widths at the saddle; the circle there joins them, as in the paper's Fig. 7.
// ``o.exponent`` = 0.5 gives radius ~ sqrt(V) (area ~ V) instead; 1 is the paper.
// ``o.maxRadius`` caps every radius (the caller uses it for the outer halo when
// the scale is set by the lowest saddle contour, see spineScale).
//
// Isovalues (section 4.4): the values of the visible critical points (peaks,
// saddles, and the field minimum, which gives the outermost contour), thinned
// so that no two are closer than a minimum gap (sharedLevels). All peaks use
// the same values, so the bands of neighbouring peaks meet.
//
// A lobe normally runs straight from u to s. With ``o.route`` it follows a
// polyline instead (the p* ridge through merged-away peaks, spineView.js);
// f(p) then runs linearly in arc length.

export const BINS = 64;

/** sRGB component in [0, 1] to linear light (what Three.js vertex colours expect). */
export function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/**
 * Cumulative value histogram of each visible peak's region.
 * Returns null when the document has no per-point values or regions (then the
 * caller falls back to plain discs).
 */
export function regionHistograms(doc, vis, bins = BINS) {
  const values = doc.data?.values, region = doc.region;
  if (!values || !region || values.length !== region.length) return null;
  const [lo, hi] = doc.meta.value_range ?? [0, 1];
  const span = hi - lo || 1;
  const nP = vis.peaks.length;
  const cum = new Float64Array(nP * (bins + 1));       // cum[i*(bins+1) + b] = points below bin b
  const count = new Float64Array(nP * bins);
  const first = new Int32Array(nP).fill(bins);         // lowest occupied bin, per peak
  const owner = vis.owner;
  for (let n = 0; n < values.length; n++) {
    let b = Math.floor(((values[n] - lo) / span) * bins);
    b = b < 0 ? 0 : b >= bins ? bins - 1 : b;
    const i = owner[region[n]];
    count[i * bins + b]++;
    if (b < first[i]) first[i] = b;
  }
  for (let i = 0; i < nP; i++) {
    for (let b = 0; b < bins; b++) cum[i * (bins + 1) + b + 1] = cum[i * (bins + 1) + b] + count[i * bins + b];
  }
  return { cum, bins, lo, span, first, nP };
}

/** Points of visible peak ``i`` with value in (a, b), a <= b (linear inside a bin). */
export function volumeBetween(h, i, a, b) {
  const at = (x) => {
    const t = Math.min(h.bins, Math.max(0, ((x - h.lo) / h.span) * h.bins));
    const k = Math.min(h.bins - 1, Math.floor(t));
    const base = i * (h.bins + 1);
    return h.cum[base + k] + (h.cum[base + k + 1] - h.cum[base + k]) * (t - k);
  };
  return Math.max(0, at(b) - at(a));
}

/**
 * Reference volume that gets radius rMax on screen, and the halo cap.
 *
 * mode 'saddle' (default): the largest count, over visible peaks, of region
 * points above the lowest visible saddle, i.e. the outermost contour that still
 * shows how peaks connect. Contours below it (down to the halo at the field
 * minimum) may be wider, up to ``capFactor`` x rMax (2: about the proportions of
 * the paper's Fig. 5b). Without saddles, or with
 * mode 'halo', the whole largest region is the reference (nothing is capped).
 */
export function spineScale(doc, vis, h, mode = 'saddle', capFactor = 2) {
  const whole = Math.max(1, ...vis.size);
  if (mode !== 'saddle' || !vis.saddles.length) return { ref: whole, capFactor: Infinity };
  const cRef = Math.min(...vis.saddles.map((id) => doc.saddles[id].value));
  let ref = 0;
  vis.peaks.forEach((id, i) => {
    const top = doc.peaks[id].value;
    if (top > cRef) ref = Math.max(ref, volumeBetween(h, i, cRef, top));
  });
  return ref >= 1 ? { ref, capFactor, cRef } : { ref: whole, capFactor: Infinity };
}

/**
 * Isovalues drawn by every peak, lowest first (section 4.4): the given critical
 * values plus the field minimum ``lo``, keeping a value only when it is at least
 * ``gap`` (a fraction of hi - lo) above the last one kept. Critical values are
 * taken highest first so that the saddles and peaks that matter win over the
 * floor. ``maxLevels`` is a safety cap; above it values are thinned evenly.
 */
export function sharedLevels(critical, lo, hi, { gap = 0.02, maxLevels = 40 } = {}) {
  const span = hi - lo || 1;
  const sep = Math.max(1e-9, gap * span);
  const vals = [...new Set(critical)].filter((c) => c > lo && c < hi).sort((a, b) => b - a);
  const kept = [];
  for (const c of vals) if (kept.every((k) => Math.abs(k - c) >= sep)) kept.push(c);
  const lv = kept.filter((c) => c - lo >= sep);
  lv.push(lo);
  lv.sort((a, b) => a - b);
  if (lv.length <= maxLevels) return lv;
  return Array.from({ length: maxLevels }, (_, k) => lv[Math.round((k * (lv.length - 1)) / (maxLevels - 1))]);
}

/**
 * Outline of the Minkowski sum of circles along a polyline: radius ``rad(t)``
 * at arc-length fraction t in [0, tEnd]. Calls ``tri`` for the swept band and
 * ``disc`` for the circles at the ends, bends, and wherever one circle swallows
 * the next (|dr/ds| >= 1, where the envelope does not exist).
 */
function sweep(pts, at, total, tEnd, samples, rad, grow, tri, disc) {
  const ts = [];
  for (let q = 0; q < pts.length - 1 && at[q] < tEnd; q++) {
    const t0 = at[q], t1 = Math.min(at[q + 1], tEnd);
    const n = Math.max(1, Math.ceil(samples * (t1 - t0)));
    for (let k = 0; k < n; k++) ts.push({ t: t0 + ((t1 - t0) * k) / n, q });
    if (t1 >= tEnd || q === pts.length - 2) ts.push({ t: t1, q });
  }
  const P = ts.map(({ t, q }) => {
    const [x0, y0] = pts[q], [x1, y1] = pts[q + 1];
    const w = at[q + 1] > at[q] ? (t - at[q]) / (at[q + 1] - at[q]) : 0;
    const len = Math.hypot(x1 - x0, y1 - y0) || 1;
    return { x: x0 + (x1 - x0) * w, y: y0 + (y1 - y0) * w, tx: (x1 - x0) / len, ty: (y1 - y0) / len,
             r: rad(t), t };
  });
  disc(P[0].x, P[0].y, P[0].r + grow);
  disc(P[P.length - 1].x, P[P.length - 1].y, P[P.length - 1].r + grow);
  for (let q = 1; q < pts.length - 1 && at[q] < tEnd; q++) disc(pts[q][0], pts[q][1], rad(at[q]) + grow);
  let prev = null;
  for (let k = 0; k < P.length; k++) {
    const p = P[k];
    const a = P[Math.max(0, k - 1)], b = P[Math.min(P.length - 1, k + 1)];
    const ds = (b.t - a.t) * total;
    let dr = ds > 0 ? (b.r - a.r) / ds : 0;             // dr/ds along the arc
    if (Math.abs(dr) >= 0.98) { disc(p.x, p.y, p.r + grow); dr = Math.sign(dr) * 0.98; }
    const nx = -p.ty, ny = p.tx, c = Math.sqrt(1 - dr * dr);
    const R = p.r + grow;
    // Envelope of the moving circle: the point of tangency is tilted back
    // (or forward) along the arc by asin(dr/ds).
    const cur = [[p.x + R * (-dr * p.tx + c * nx), p.y + R * (-dr * p.ty + c * ny)],
                 [p.x + R * (-dr * p.tx - c * nx), p.y + R * (-dr * p.ty - c * ny)]];
    if (prev) { tri(prev[0], prev[1], cur[0]); tri(prev[1], cur[1], cur[0]); }
    prev = cur;
  }
  return Math.max(...P.map((p) => p.r));
}

/**
 * Build the triangles of the whole spine drawing.
 *
 * @param doc        exported document (peaks, saddles, meta.value_range)
 * @param vis        result of simplify()
 * @param h          result of regionHistograms()
 * @param peakXY     layout of every document peak
 * @param saddleXY   layout of every document saddle
 * @param o          {rs, maxRadius (optional cap on every radius), exponent (1 = paper, 0.5 = sqrt), gap (min isovalue gap, fraction
 *                    of the value range), colorOf(t)->[r,g,b] sRGB, bg, edge: [r,g,b] sRGB,
 *                    edgeWidth, z0, zRange, maxLevels, samples,
 *                    route(i, sid) -> [[x, y], ...] from the peak to the saddle, or null
 *                    for a straight lobe (optional)}
 * @returns {{positions: Float32Array, colors: Float32Array, triangles: number,
 *            extent: Float32Array, levels: number[]}}  ``extent[i]``: outer radius around visible peak i
 */
export function buildContours(doc, vis, h, peakXY, saddleXY, o) {
  const [lo, hi] = doc.meta.value_range ?? [0, 1];
  const span = hi - lo || 1;
  const P = vis.peaks.length;
  const many = P > 80;
  const exponent = o.exponent ?? 1;
  const m = o.samples ?? (many ? 10 : 32);
  const segs = many ? 24 : 48;
  const edgeW = o.edgeWidth ?? 0.003;

  const inc = Array.from({ length: P }, () => []);      // saddles (document ids) per visible peak
  vis.saddles.forEach((id, j) => {
    const [a, b] = vis.saddleEnds[j];
    inc[a].push(id);
    if (b !== a) inc[b].push(id);
  });
  const levels = sharedLevels(
    [...vis.saddles.map((id) => doc.saddles[id].value), ...vis.peaks.map((id) => doc.peaks[id].value)],
    lo, hi, { gap: o.gap ?? 0.02, maxLevels: o.maxLevels ?? (many ? 12 : 40) });

  // The path of each lobe, from the peak to the saddle, with its arc length
  // (0 at the peak, 1 at the saddle) at every vertex. Computed once per lobe.
  const lobes = Array.from({ length: P }, (_, i) => inc[i].map((sid) => {
    const [ux, uy] = peakXY[vis.peaks[i]];
    const routed = o.route?.(i, sid);
    const pts = routed && routed.length >= 2 ? routed : [[ux, uy], saddleXY[sid]];
    const at = [0];
    for (let q = 1; q < pts.length; q++) {
      at.push(at[q - 1] + Math.hypot(pts[q][0] - pts[q - 1][0], pts[q][1] - pts[q - 1][1]));
    }
    const total = at[at.length - 1];
    return { sid, pts, at: at.map((a) => (total > 0 ? a / total : 0)), total };
  }));

  const pos = [], col = [];
  const extent = new Float32Array(P);
  const cap = o.maxRadius ?? Infinity;                  // monotone, so nesting is kept
  const size = (v) => Math.min(cap, o.rs * Math.max(0, v) ** exponent);
  const lin = (rgb) => rgb.map(srgbToLinear);

  for (const c of levels) {
    const t = (c - lo) / span;
    const tc = Math.min(1, Math.max(0, t));
    const base = o.colorOf(tc);
    // Low isovalues fade towards the background, like the paper's transfer
    // function (section 4.4): the large outer contours read as pale halos and
    // do not hide the structure drawn on top of them.
    const a = 0.22 + 0.68 * tc, ae = 0.3 + 0.7 * tc;
    const mixed = lin(base.map((v, k) => v * a + o.bg[k] * (1 - a)));
    const rim = lin(o.edge.map((v, k) => v * ae + o.bg[k] * (1 - ae)));
    const z = o.z0 + o.zRange * Math.min(1, Math.max(0, t));

    // All rims of this isovalue first, then all fills: shapes of different
    // peaks at one value merge into a single outline.
    for (const pass of [0, 1]) {
      const grow = pass === 0 ? edgeW : 0;
      const rgb = pass === 0 ? rim : mixed;
      const zz = pass === 0 ? z : z + o.zRange * 0.0004 + 1e-5;
      const tri = (a, b, d) => {
        pos.push(a[0], a[1], zz, b[0], b[1], zz, d[0], d[1], zz);
        for (let k = 0; k < 3; k++) col.push(rgb[0], rgb[1], rgb[2]);
      };
      const disc = (x, y, r) => {
        if (!(r > 0)) return;
        let prev = [x + r, y];
        for (let s = 1; s <= segs; s++) {
          const a = (2 * Math.PI * s) / segs;
          const next = [x + r * Math.cos(a), y + r * Math.sin(a)];
          tri([x, y], prev, next);
          prev = next;
        }
      };

      for (let i = 0; i < P; i++) {
        const top = doc.peaks[vis.peaks[i]].value;
        if (c >= top) continue;
        const [ux, uy] = peakXY[vis.peaks[i]];
        const r0 = size(volumeBetween(h, i, c, top));
        if (!(r0 > 0)) continue;
        let drawn = false;
        for (const { sid, pts, at, total } of lobes[i]) {
          const s = doc.saddles[sid];
          if (s.value >= top || total < 1e-9) continue;
          const tEnd = c > s.value ? Math.min(1, (top - c) / (top - s.value)) : 1;
          const rad = (tt) => {
            const f = top + (s.value - top) * tt;
            return size(volumeBetween(h, i, Math.min(c, f), Math.max(c, f)));
          };
          const r = sweep(pts, at, total, tEnd, m, rad, grow, tri, disc);
          if (pass === 1) extent[i] = Math.max(extent[i], r);
          drawn = true;
        }
        if (!drawn) disc(ux, uy, r0 + grow);
        if (pass === 1) extent[i] = Math.max(extent[i], r0);
      }
    }
  }
  return { positions: new Float32Array(pos), colors: new Float32Array(col),
           triangles: pos.length / 9, extent, levels };
}
