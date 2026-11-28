// Comparing the extremum graphs of two extraction variants (compare mode).
//
// Pure functions on exported documents, no DOM or WebGL, so they are easy to
// test (tests/browser_checks.html). Two documents of the same field are
// matched by *vertex*: a peak in one and a peak in the other are "the same
// maximum" when they sit at the same sample point. The build gives such
// vertices the same layout position in both files (comparison/viewer_export.py).

import { simplify, showAll } from './simplify.js';

/** False for methods without persistence (e.g. AnyEG): their full graph is always shown. */
export function isSimplifiable(doc) {
  return doc?.meta?.variant?.simplifiable !== false;
}

/** What to draw at noise threshold ``t`` and variation threshold ``v``. */
export function visibleGraph(doc, t, v = Infinity) {
  return isSimplifiable(doc) ? simplify(doc, t, v) : showAll(doc);
}

/** Vertices of every maximum the method found before any simplification. */
export function allMaxima(doc) {
  return new Set(doc.meta?.all_maxima ?? doc.peaks.map((p) => p.vertex));
}

/**
 * Compare what the two panels show now.
 * Returns counts of shown peaks, how many sit at the same vertex in both, and
 * the document ids of shown peaks whose vertex is *not a maximum at all* in the
 * other method (``missingA``: shown in A, absent from B; these get a ring).
 */
export function compareShown(docA, visA, docB, visB) {
  const shownA = visA.peaks.map((id) => docA.peaks[id].vertex);
  const shownB = visB.peaks.map((id) => docB.peaks[id].vertex);
  const setB = new Set(shownB);
  const allA = allMaxima(docA), allB = allMaxima(docB);
  return {
    shownA: shownA.length,
    shownB: shownB.length,
    shared: shownA.filter((v) => setB.has(v)).length,
    missingA: visA.peaks.filter((id) => !allB.has(docA.peaks[id].vertex)),
    missingB: visB.peaks.filter((id) => !allA.has(docB.peaks[id].vertex)),
  };
}

/** Visible index of the node of ``kind`` ('peak' | 'saddle') at ``vertex``, or -1. */
export function findVisible(doc, vis, kind, vertex) {
  const list = kind === 'peak' ? vis.peaks : vis.saddles;
  const nodes = kind === 'peak' ? doc.peaks : doc.saddles;
  return list.findIndex((id) => nodes[id].vertex === vertex);
}

/**
 * The design choices on which two variants differ: the short reason why their
 * graphs differ. ``labels`` gives the order and names of the traits.
 */
export function traitDifferences(variantA, variantB, labels) {
  const out = [];
  for (const [key, label] of Object.entries(labels ?? {})) {
    const a = variantA?.traits?.[key], b = variantB?.traits?.[key];
    if (a && b && a !== b) out.push({ key, label, a, b });
  }
  return out;
}

/** The precomputed agreement of two variants on their full graphs (index.json), either order. */
export function findAgreement(entry, idA, idB) {
  for (const c of entry?.comparisons ?? []) {
    if (c.a === idA && c.b === idB) return c;
    if (c.a === idB && c.b === idA) {
      return { ...c, a: idA, b: idB, maxima_only_a: c.maxima_only_b, maxima_only_b: c.maxima_only_a,
               pairs_only_a: c.pairs_only_b, pairs_only_b: c.pairs_only_a };
    }
  }
  return null;
}

/**
 * Colour ids for the peaks of ``docB`` so that a maximum shared with ``docA`` gets
 * the colour of its peak in ``docA`` (same vertex); the others get ids after
 * ``docA``'s, so they never borrow a colour of ``docA``.
 */
export function sharedColorIds(docA, docB) {
  const idOfVertex = new Map(docA.peaks.map((p, i) => [p.vertex, i]));
  let next = docA.peaks.length;
  return docB.peaks.map((p) => idOfVertex.get(p.vertex) ?? next++);
}
