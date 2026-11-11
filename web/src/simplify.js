// Persistence simplification of an exported extremum graph, in the browser.
//
// A port of ExtremumGraph.simplify (pipeline/extremum_graph.py). It works on
// the exported document and returns *indices into that document*, so the
// layout stored in the file stays valid: nodes are only hidden, never moved.
//
// Rules (same as Python):
//  * a peak survives if its persistence is null (global maximum) or >= t;
//  * a removed peak joins its parent (repeatedly, if the parent is removed too);
//  * a saddle is dropped if both of its peaks end up as the same peak, if its
//    own persistence is below t, or if it is above the variation threshold
//    (the paper's p_hi; links only, never peaks); of several saddles that now
//    join the same pair of peaks only the highest (first in file order) is kept.
//
// Saturated persistence (the paper's p*, equation 3; ExtremumGraph.arc_mask):
// the arcs of saddle s are drawn when persistence(s) <= variation and
// saturation(s) >= threshold. They go to the saddle's own two peaks in the
// file, which may be hidden. This keeps the ridge through merged-away peaks
// visible (the rim of the paper's Fig. 5b). Files without ``saturation`` use
// the persistence, so only the surviving saddles have arcs.

/**
 * @param {{peaks: Array, saddles: Array}} doc exported document
 * @param {number} threshold persistence below which peaks are removed
 * @param {number} [variation] saddles with persistence above this are removed (default: none)
 * @returns {{
 *   peaks: number[],            // document ids of the surviving peaks, in file order
 *   saddles: number[],          // document ids of the surviving saddles
 *   saddleEnds: number[][],     // per surviving saddle: [higher, lower] as indices into `peaks`
 *   size: number[],             // per surviving peak: points in its (merged) region
 *   owner: Int32Array,          // per document peak: index into `peaks` of the peak it now belongs to
 *   arcs: number[],             // document ids of every saddle whose arcs are drawn (equation 3)
 *   ridges: number[],           // the ones among `arcs` that are not in `saddles`
 *   ghosts: number[],           // document ids of hidden peaks that arcs pass through
 * }}
 */
export function simplify(doc, threshold, variation = Infinity) {
  const P = doc.peaks.length;
  const survives = new Array(P);
  for (let i = 0; i < P; i++) {
    const p = doc.peaks[i].persistence;
    survives[i] = p === null || p >= threshold;
  }

  // Parents have smaller ids, so id order sees every parent resolved first.
  const root = new Int32Array(P);
  for (let i = 0; i < P; i++) {
    root[i] = survives[i] ? i : root[doc.peaks[i].parent];
  }

  const newId = new Int32Array(P);
  const kept = [];
  for (let i = 0; i < P; i++) {
    if (survives[i]) { newId[i] = kept.length; kept.push(i); }
  }
  const owner = new Int32Array(P);
  for (let i = 0; i < P; i++) owner[i] = newId[root[i]];

  const size = new Array(kept.length).fill(0);
  for (let i = 0; i < P; i++) size[owner[i]] += doc.peaks[i].size;

  const saddles = [];
  const saddleEnds = [];
  const seen = new Set();
  const n = kept.length;
  doc.saddles.forEach((s, id) => {
    const a = owner[s.peaks[0]];
    const b = owner[s.peaks[1]];
    if (a === b || s.persistence < threshold || s.persistence > variation) return;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    const key = lo * Math.max(n, 1) + hi;
    if (seen.has(key)) return;
    seen.add(key);
    saddles.push(id);
    saddleEnds.push([lo, hi]);
  });

  const shown = new Set(saddles);
  const arcs = [], ridges = [];
  const ghost = new Set();
  doc.saddles.forEach((s, id) => {
    const sat = s.saturation ?? s.persistence;
    if (s.persistence > variation || sat < threshold) return;
    arcs.push(id);
    if (!shown.has(id)) ridges.push(id);
    for (const p of s.peaks) if (!survives[p]) ghost.add(p);
  });
  const ghosts = [...ghost].sort((a, b) => a - b);

  return { peaks: kept, saddles, saddleEnds, size, owner, arcs, ridges, ghosts };
}

/**
 * Document saddle ids on the drawn path from document peak ``from`` to the
 * surviving peak it belongs to, through hidden peaks only (breadth first over
 * ``vis.arcs``). Empty if ``from`` survives, or if no such path exists.
 */
export function arcPath(doc, vis, from) {
  const target = vis.peaks[vis.owner[from]];
  if (from === target) return [];
  const { isKept, adj } = arcGraph(doc, vis);
  const via = new Map([[from, null]]);
  const queue = [from];
  while (queue.length) {
    const u = queue.shift();
    if (u === target) break;
    if (u !== from && isKept.has(u)) continue;          // do not pass through another surviving peak
    for (const [sid, w] of adj.get(u) ?? []) {
      if (!via.has(w)) { via.set(w, [sid, u]); queue.push(w); }
    }
  }
  if (!via.has(target)) return [];
  const path = [];
  for (let u = target; via.get(u); u = via.get(u)[1]) path.push(via.get(u)[0]);
  return path.reverse();
}

const arcGraphs = new WeakMap();                         // vis -> {isKept, adj}, built once per vis

function arcGraph(doc, vis) {
  let g = arcGraphs.get(vis);
  if (g) return g;
  const adj = new Map();
  for (const id of vis.arcs) {
    const [a, b] = doc.saddles[id].peaks;
    if (!adj.has(a)) adj.set(a, []);
    if (!adj.has(b)) adj.set(b, []);
    adj.get(a).push([id, b]);
    adj.get(b).push([id, a]);
  }
  g = { isKept: new Set(vis.peaks), adj };
  arcGraphs.set(vis, g);
  return g;
}

/**
 * Layout points of the drawn ridge from visible peak ``i`` to the visible saddle
 * ``sid`` that it is linked to: the peak, then the hidden peaks and the saddles
 * between them (``arcPath``), then ``sid``. Null when the link is direct (the
 * saddle's own peak survives) or no ridge is drawn.
 */
export function ridgeRoute(doc, vis, i, sid, peakXY, saddleXY) {
  const u = vis.peaks[i];
  const end = doc.saddles[sid].peaks.find((p) => vis.owner[p] === i);
  if (end === undefined || end === u) return null;
  const path = arcPath(doc, vis, end);
  if (!path.length) return null;
  const pts = [saddleXY[sid], peakXY[end]];
  let cur = end;
  for (const via of path) {
    const [a, b] = doc.saddles[via].peaks;
    cur = a === cur ? b : a;
    pts.push(saddleXY[via], peakXY[cur]);
  }
  return pts.reverse();                                   // from the peak to the saddle
}

/** Largest persistence of any saddle, for scaling the variation slider. */
export function maxSaddlePersistence(doc) {
  let m = 0;
  for (const s of doc.saddles) if (s.persistence > m) m = s.persistence;
  return m;
}

/** Number of saddles with persistence <= variation (the red curve of the paper's Fig. 8b). */
export function saddleCount(sorted, variation) {
  let lo = 0, hi = sorted.length;            // first index whose value is > variation
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= variation) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** Largest finite persistence among the peaks, for scaling the slider. */
export function maxPersistence(doc) {
  let m = 0;
  for (const p of doc.peaks) if (p.persistence !== null && p.persistence > m) m = p.persistence;
  const c = doc.persistence_curve;
  if (c && c.length && c[0] > m) m = c[0];
  return m;
}
