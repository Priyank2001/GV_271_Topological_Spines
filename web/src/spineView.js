// Spine view: draws the 2D layout of an extremum graph with Three.js.
//
//  * peaks   - discs, area proportional to the number of points in the region,
//              coloured by the peak's field value;
//  * saddles - small diamonds (orange when the saddle closes a loop);
//  * links   - one line from each saddle to each of its two peaks.
//
// In "paper" style (``setStyle('paper')``, needs ``doc.data.values`` and ``doc.region``)
// the discs and plain links are replaced by the drawing of Correa et al.: nested
// contours around each peak and figure-eight lobes along the links (contours.js).
// The discs shrink to small dots; picking and rings use the outer contour.
// ``setStyle('nddav')`` (the default in the page) draws the same contours the way
// LLNL's NDDAV does: concentric rings above the saddles, evenly spaced isovalues,
// opaque fills with dark outlines, radius ~ count^(2/d) (contours.js).
//
// With ``setShowArcs(true)`` the links follow the paper's equation (3) instead:
// every saddle in ``vis.arcs`` (saturated persistence p* >= threshold) gets its
// two arcs to its own peaks in the file. Where such a peak has been merged away
// it is drawn as a small grey dot, so a link to a surviving peak runs along the
// ridge through the peaks it absorbed (the rim of the paper's Fig. 5b).
// In paper style the lobes then follow the same ridge (``ridgeRoute``).
//
// The layout comes from the exported file (``doc.spine``) and never changes.
// ``setVisible`` only chooses which nodes to draw (see simplify.js).
// Pan with the mouse, zoom with the wheel, double-click to reset.

import * as THREE from 'three';
import { regionHistograms, buildContours, spineScale, volumeBetween } from './contours.js';
import { arcPath, ridgeRoute } from './simplify.js';
import { relayout } from './relayout.js';

const Z_LINK = 0.0, Z_SADDLE = 0.1, Z_PEAK = 0.2, Z_RING = 0.3;
const Z_CONTOUR = -0.6, Z_CONTOUR_RANGE = 0.5;   // paper-style shapes sit below everything else
// Height mode: a node or contour stands at HEIGHT_BASE x scale x (value - lo) / (hi - lo) above
// the layout plane (the layout box is [-1, 1]), and the camera orbits it.
const HEIGHT_BASE = 0.8, HEIGHT_EPS = 0.004;
const NDDAV_HALO = 0.8;
const SPACING_FIT = 0.35; // reference contour radius <= this x median nearest-peak distance
                          // (so neighbouring blobs do not swallow each other, e.g. in high dimensions)   // 'nddav' style: outer halo radius / median link length (at contour size 1)
// Colour maps for field values, sampled at 11 even stops from matplotlib (sRGB).
// 'fig1' style: the banded colours of Correa et al. Fig. 1 (pale blue halo, blue, green,
// yellow, orange, magenta, white at the top), read off the figure.
export const FIG1_STOPS = [[0.80,0.84,0.93], [0.55,0.65,0.86], [0.45,0.66,0.38], [0.72,0.84,0.42],
  [0.97,0.93,0.55], [0.97,0.74,0.38], [0.90,0.50,0.40], [0.74,0.42,0.66], [0.93,0.80,0.92], [1, 1, 1]];
const FIG1_FIT = 1.4;
const FIG1_SPACING = 0.45;  // 'fig1': reference radius <= this x median nearest-peak distance      // 'fig1': saddle-level contour radius / median peak-saddle link
const FIG1_BG = '#ffffff';
export function fig1Color(t) {
  const S = FIG1_STOPS, x = Math.min(1, Math.max(0, t)) * (S.length - 1);
  const i = Math.min(S.length - 2, Math.floor(x)), f = x - i;
  return S[i].map((c, k) => c + (S[i + 1][k] - c) * f);
}

export const COLORMAPS = {
  viridis: [[0.267,0.005,0.329], [0.283,0.141,0.458], [0.254,0.265,0.530], [0.207,0.372,0.553], [0.164,0.471,0.558], [0.128,0.567,0.551], [0.135,0.659,0.518], [0.267,0.749,0.441], [0.478,0.821,0.318], [0.741,0.873,0.150], [0.993,0.906,0.144]],
  plasma: [[0.050,0.030,0.528], [0.255,0.014,0.615], [0.418,0.001,0.658], [0.563,0.052,0.642], [0.693,0.165,0.565], [0.798,0.280,0.470], [0.881,0.393,0.383], [0.949,0.518,0.296], [0.988,0.652,0.211], [0.989,0.810,0.145], [0.940,0.975,0.131]],
  inferno: [[0.001,0.000,0.014], [0.087,0.045,0.225], [0.258,0.039,0.406], [0.416,0.090,0.433], [0.578,0.148,0.404], [0.736,0.216,0.330], [0.865,0.317,0.226], [0.955,0.469,0.100], [0.988,0.645,0.040], [0.964,0.844,0.273], [0.988,0.998,0.645]],
  magma: [[0.001,0.000,0.014], [0.079,0.054,0.212], [0.232,0.060,0.438], [0.390,0.100,0.502], [0.550,0.161,0.506], [0.716,0.215,0.475], [0.869,0.288,0.409], [0.968,0.440,0.360], [0.995,0.624,0.427], [0.996,0.813,0.573], [0.987,0.991,0.750]],
  cividis: [[0.000,0.135,0.305], [0.032,0.201,0.441], [0.209,0.273,0.425], [0.310,0.340,0.425], [0.401,0.412,0.441], [0.489,0.485,0.471], [0.582,0.559,0.468], [0.684,0.639,0.444], [0.786,0.720,0.400], [0.897,0.811,0.321], [0.996,0.909,0.218]],
  turbo: [[0.190,0.072,0.232], [0.270,0.349,0.796], [0.244,0.609,0.997], [0.098,0.837,0.803], [0.276,0.971,0.517], [0.644,0.990,0.234], [0.883,0.866,0.217], [0.996,0.643,0.192], [0.941,0.356,0.070], [0.765,0.144,0.010], [0.480,0.016,0.011]],
  coolwarm: [[0.230,0.299,0.754], [0.348,0.466,0.888], [0.484,0.622,0.975], [0.619,0.744,0.999], [0.754,0.830,0.961], [0.867,0.864,0.863], [0.947,0.795,0.717], [0.968,0.674,0.557], [0.932,0.519,0.406], [0.839,0.322,0.265], [0.706,0.016,0.150]],
  RdYlBu: [[0.192,0.212,0.584], [0.269,0.454,0.703], [0.455,0.678,0.820], [0.666,0.848,0.912], [0.878,0.953,0.973], [1.000,0.998,0.745], [0.996,0.878,0.565], [0.991,0.677,0.378], [0.957,0.427,0.263], [0.839,0.185,0.153], [0.647,0.000,0.149]],
  Spectral: [[0.369,0.310,0.635], [0.199,0.529,0.739], [0.400,0.761,0.647], [0.665,0.865,0.643], [0.902,0.961,0.596], [1.000,0.998,0.745], [0.996,0.878,0.545], [0.991,0.677,0.378], [0.957,0.427,0.263], [0.831,0.238,0.309], [0.620,0.004,0.259]],
  BrBG: [[0.329,0.188,0.020], [0.545,0.315,0.039], [0.749,0.506,0.176], [0.872,0.756,0.484], [0.965,0.910,0.765], [0.957,0.960,0.960], [0.780,0.918,0.898], [0.496,0.800,0.753], [0.208,0.592,0.561], [0.004,0.397,0.365], [0.000,0.235,0.188]],
  grey: [[0.000,0.000,0.000], [0.098,0.098,0.098], [0.200,0.200,0.200], [0.298,0.298,0.298], [0.400,0.400,0.400], [0.502,0.502,0.502], [0.600,0.600,0.600], [0.702,0.702,0.702], [0.800,0.800,0.800], [0.902,0.902,0.902], [1.000,1.000,1.000]],
};

// The active value colour map. It is shared by every view that colours by field value
// (spine, scatterplot, volume), so one legend covers them all.
let cmapName = 'viridis', cmapReversed = false, cmapStops = COLORMAPS.viridis;

/** Choose the value colour map (a key of ``COLORMAPS``); ``reversed`` flips it. */
export function setColormap(name, reversed = false) {
  cmapName = COLORMAPS[name] ? name : 'viridis';
  cmapReversed = Boolean(reversed);
  cmapStops = cmapReversed ? [...COLORMAPS[cmapName]].reverse() : COLORMAPS[cmapName];
}

/** ``{name, reversed}`` of the active value colour map. */
export function getColormap() { return { name: cmapName, reversed: cmapReversed }; }

/** Colour of ``t`` in [0, 1] on the peak colour scale, as [r, g, b] in sRGB. */
export function peakColor(t) {
  const S = cmapStops;
  const x = Math.min(1, Math.max(0, t)) * (S.length - 1);
  const i = Math.min(S.length - 2, Math.floor(x));
  const f = x - i;
  return S[i].map((c, k) => c + (S[i + 1][k] - c) * f);
}

/**
 * A distinct colour for peak ``id`` (its id in the file, so it does not change with the
 * slider), as [r, g, b] in sRGB. Hues step by the golden angle so neighbours differ.
 * The data views colour a peak's region with it, and the paper-style spine its dot.
 */
export function regionColor(id) {
  const h = ((id * 137.508) % 360) / 360, s = 0.65, l = 0.5;
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** CSS gradient string matching ``peakColor``, for the legend. */
export function peakGradientCss() {
  const stops = cmapStops.map((c, i) =>
    `rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${(i / (cmapStops.length - 1)) * 100}%`);
  return `linear-gradient(to right, ${stops.join(', ')})`;
}

export class SpineView {
  /**
   * @param {HTMLElement} container element the canvas is added to
   * @param {{onHover?: Function, onSelect?: Function, onCamera?: Function}} handlers: onHover and
   *        onSelect get a node ``{kind: 'peak'|'saddle', index, docId}`` (index into the visible
   *        lists) or null; onCamera gets ``getCamera()`` after the user pans, zooms or resets
   * @param {{renderer?: object}} options ``renderer`` replaces the WebGL renderer
   *        (used by the tests, which have no GPU)
   */
  constructor(container, handlers = {}, options = {}) {
    this.container = container;
    this.handlers = handlers;
    this.colors = { bg: '#ffffff', link: '#8a8f98', saddle: '#6b7280', loop: '#d97706',
                    ring: '#111827', select: '#2563eb', mark: '#c026d3' };
    this.marked = new Set();   // document peak ids drawn with a mark ring (compare mode)
    this.colorIds = null;      // document peak id -> id whose regionColor it uses (compare mode); null: itself

    this.renderer = options.renderer ?? new THREE.WebGLRenderer({ antialias: true });
    if (!options.renderer) this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.touchAction = 'none';

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    this.camera.position.z = 5;
    this.center = new THREE.Vector2(0, 0);
    this.halfHeight = 1.25;
    this.homeHalfHeight = 1.25;

    this.style = 'simple';     // 'simple' (discs and lines), 'paper' or 'nddav' (nested contours)
    this.contourScale = 1;     // multiplies the radius of every contour
    this.radiusExponent = 1;   // contour radius ~ volume^k: 1 as in the paper, 0.5 = area ~ volume, 'nddav' = 2/d
    this.levelGap = 0.02;      // minimum isovalue gap, fraction of the value range (paper, section 4.4)
    this.levelCount = 10;      // evenly spaced isovalues of the 'nddav' style (plus the saddle values)
    this.scaleMode = 'saddle'; // which contour gets radius rMax: 'saddle' (lowest saddle; halo capped) or 'halo'
    this.showArcs = false;     // draw the arcs of equation (3), through merged-away peaks
    this.relayoutOn = true;
    this.floorOn = true;
    this.heightOn = false;     // value as height, the spine as a landscape (see setHeight)
    this.heightScale = 1;
    this.heightSmooth = true;  // true: small steps (smooth slopes, the default); false: one terrace per contour
    this.heightSolid = false;  // a solid mountain: walls between the levels, colours only at the contour values
    this.yaw = -Math.PI / 2;   // orbit camera: azimuth about the up (value) axis
    this.pitch = 0.95;         // and tilt from straight down (0) towards the horizon       // paper style: drop contours far below the lowest shown saddle    // lay the visible nodes out again after simplification (relayout.js)
    this.relayoutCache = new Map();
    this.contourMesh = null;
    this.bands = null;         // the drawn bands (see #bandAt), set with the contours
    this.bandMesh = null;      // the highlighted band
    this.bandHover = null;
    this.doc = null;
    this.vis = null;           // result of simplify(), or null
    this.hover = null;
    this.selected = null;
    this.dirty = true;

    this.#buildFixedObjects();
    this.#bindEvents();
    this.resize();
    this.#loop();
  }

  // ---- public API -------------------------------------------------------
  /** Show the layout of ``doc`` (an exported document with a ``spine``). */
  setGraph(doc) {
    this.doc = doc;
    this.hover = this.selected = null;
    this.marked = new Set();                   // ids of the previous document mean nothing here
    this.colorIds = null;
    const sp = doc.spine;
    if (!sp) throw new Error('this file has no spine layout');

    this.peakXY = this.basePeakXY = sp.peaks;          // exported layout of the full graph
    this.saddleXY = this.baseSaddleXY = sp.saddles;
    this.relayoutCache = new Map();
    const values = doc.peaks.map((p) => p.value);
    this.valueRange = [Math.min(...values), Math.max(...values)];

    const nP = doc.peaks.length, nS = doc.saddles.length;
    this.#allocate(nP, nS);
    this.fit();
  }

  /** Draw only the nodes in ``vis`` (the result of ``simplify(doc, t)``). */
  setVisible(vis) {
    this.vis = vis;
    const doc = this.doc;
    this.#applyRelayout(vis);
    const nP = vis.peaks.length, nS = vis.saddles.length;

    const maxSize = Math.max(...vis.size, 1);
    this.rMax = Math.min(0.14, Math.max(0.035, 0.42 / Math.sqrt(Math.max(nP, 1))));
    this.rMin = this.rMax * 0.22;
    this.radius = vis.size.map((s) => this.rMin + (this.rMax - this.rMin) * Math.sqrt(s / maxSize));
    const atHome = this.center.x === 0 && this.center.y === 0 && this.halfHeight === this.homeHalfHeight;
    this.pad = 0;
    const paper = this.#buildPaperStyle(vis);
    if (atHome) this.fit();               // reframe for the new shapes unless the user has panned or zoomed
    const arcs = this.#arcsActive();
    const fig1 = paper && this.style === 'fig1';
    this.linkLines.visible = !paper || arcs || fig1;  // over the contours, the arcs mark the ridges; Fig. 1 draws every link
    this.paperActive = paper;

    const [lo, hi] = this.valueRange;
    const m = new THREE.Matrix4(), col = new THREE.Color();
    for (let i = 0; i < nP; i++) {
      const id = vis.peaks[i];
      const [x, y] = this.peakXY[id];
      const r = this.radius[i];
      const dot = fig1 ? SADDLE_R * 0.9 : paper ? Math.max(SADDLE_R * 0.9, Math.min(r * 0.14, 0.024)) : r;   // peak marker
      m.makeScale(dot, dot, 1).setPosition(x, y, this.#zOf(doc.peaks[id].value, Z_PEAK - HEIGHT_EPS) + HEIGHT_EPS);
      this.peakMesh.setMatrixAt(i, m);
      const [R, G, B] = peakColor(hi > lo ? (doc.peaks[id].value - lo) / (hi - lo) : 0.5);
      const [rr, rg, rb] = regionColor(this.colorIds?.[id] ?? id);   // paper style: the dot shows the region's colour
      this.peakMesh.setColorAt(i, fig1 ? col.setRGB(0.22, 0.24, 0.3, THREE.SRGBColorSpace)
                                : paper ? col.setRGB(rr, rg, rb, THREE.SRGBColorSpace)
                                        : col.setRGB(R, G, B, THREE.SRGBColorSpace));
    }
    this.peakMesh.count = nP;
    this.peakMesh.instanceMatrix.needsUpdate = true;
    if (this.peakMesh.instanceColor) this.peakMesh.instanceColor.needsUpdate = true;

    const loopCol = new THREE.Color(this.colors.loop), mergeCol = new THREE.Color(this.colors.saddle);
    for (let j = 0; j < nS; j++) {
      const id = vis.saddles[j];
      const [x, y] = this.saddleXY[id];
      const sr = fig1 ? SADDLE_R * 0.45 : SADDLE_R;      // Fig. 1 marks no saddles: just a speck where the lobes meet
      m.makeScale(sr, sr, 1).setPosition(x, y, this.#zOf(doc.saddles[id].value, Z_SADDLE - HEIGHT_EPS) + HEIGHT_EPS);
      this.saddleMesh.setMatrixAt(j, m);
      this.saddleMesh.setColorAt(j, doc.saddles[id].is_merge ? mergeCol : loopCol);
    }
    this.saddleMesh.count = nS;
    this.saddleMesh.instanceMatrix.needsUpdate = true;
    if (this.saddleMesh.instanceColor) this.saddleMesh.instanceColor.needsUpdate = true;

    const pos = this.linkGeom.attributes.position.array;
    const put = (k, [px, py], [sx, sy], pv, sv) => {
      const o = k * 6;
      pos[o] = px; pos[o + 1] = py; pos[o + 2] = this.#zOf(pv, Z_LINK - HEIGHT_EPS) + HEIGHT_EPS;
      pos[o + 3] = sx; pos[o + 4] = sy; pos[o + 5] = this.#zOf(sv, Z_LINK - HEIGHT_EPS) + HEIGHT_EPS;
    };
    let segments = 0;
    if (arcs) {
      for (const id of vis.arcs) {
        for (const p of doc.saddles[id].peaks) {
          put(segments++, this.peakXY[p], this.saddleXY[id], doc.peaks[p].value, doc.saddles[id].value);
        }
      }
    } else {
      for (let j = 0; j < nS; j++) {
        for (let e = 0; e < 2; e++) {
          const pid = vis.peaks[vis.saddleEnds[j][e]], sid = vis.saddles[j];
          put(segments++, this.peakXY[pid], this.saddleXY[sid], doc.peaks[pid].value, doc.saddles[sid].value);
        }
      }
    }
    this.linkGeom.setDrawRange(0, segments * 2);
    this.linkGeom.attributes.position.needsUpdate = true;

    const ghosts = arcs ? vis.ghosts : [];
    ghosts.forEach((id, g) => {
      const [x, y] = this.peakXY[id];
      m.makeScale(GHOST_R, GHOST_R, 1).setPosition(x, y, this.#zOf(doc.peaks[id].value, Z_SADDLE - HEIGHT_EPS) + HEIGHT_EPS);
      this.ghostMesh.setMatrixAt(g, m);
    });
    this.ghostMesh.count = ghosts.length;
    this.ghostMesh.instanceMatrix.needsUpdate = true;
    this.#updateMarks();

    // Selections refer to visible indices, which just changed: keep the same
    // peak/saddle if it is still drawn, otherwise drop the selection.
    this.selected = this.#reindex(this.selected);
    this.hover = null;
    this.#updateOverlay();
    this.dirty = true;
  }

  /** Select a node from outside (e.g. ``{kind: 'peak', index: 0}``) or clear with null. */
  select(node) {
    if (node && node.docId === undefined && this.vis) {
      const list = node.kind === 'peak' ? this.vis.peaks : this.vis.saddles;
      node = { ...node, docId: list[node.index] };
    }
    this.selected = node;
    this.#updateOverlay();
    this.handlers.onSelect?.(node);
    this.dirty = true;
  }

  // ---- contour bands under the pointer --------------------------------------
  /**
   * The band (one contour level around one peak) under the pointer: the highest level whose
   * fill contains the point (height mode: where the pointer's ray meets that level's height).
   * Returns ``{ c, i, docId }`` or null.
   */
  #bandAt(ev) {
    const B = this.bands;
    if (!B || !B.shapes.length || !this.vis) return null;
    const el = this.renderer.domElement, rect = el.getBoundingClientRect();
    const nx = (ev.clientX - rect.left) / rect.width * 2 - 1, ny = 1 - (ev.clientY - rect.top) / rect.height * 2;
    let ray = null, w = null;
    if (this.heightOn) {
      this.#placeCamera();
      const rc = new THREE.Raycaster();
      rc.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
      ray = rc.ray;
    } else {
      const aspect = this.#aspect();
      w = [this.center.x + nx * this.halfHeight * aspect, this.center.y + ny * this.halfHeight];
    }
    for (const k of B.order) {
      const sh = B.shapes[k];
      let x, y;
      if (ray) {
        const dz = ray.direction.z;
        if (Math.abs(dz) < 1e-9) continue;
        const t = (sh.z - ray.origin.z) / dz;
        x = ray.origin.x + ray.direction.x * t; y = ray.origin.y + ray.direction.y * t;
      } else {
        [x, y] = w;
      }
      const [x0, y0, x1, y1] = sh.box;
      if (x < x0 || x > x1 || y < y0 || y > y1) continue;
      if (insideTriangles(B.pos, sh.start, sh.end, x, y)) {
        return { c: this.#bandBase(sh.c), i: sh.i, docId: this.vis.peaks[sh.i] };   // the whole band, not a sub-step
      }
    }
    return null;
  }

  /** The contour value whose band holds ``c``: the highest contour value at or below it. */
  #bandBase(c) {
    const base = this.bands?.base ?? [];
    let b = -Infinity;
    for (const v of base) if (v <= c + 1e-9 && v > b) b = v;
    return Number.isFinite(b) ? b : c;
  }

  /** Visible peaks joined to peak ``i`` at isovalue ``c`` (through shown saddles at or above it). */
  #bandComponent(c, i) {
    const vis = this.vis, n = vis.peaks.length;
    const parent = Array.from({ length: n }, (_, k) => k);
    const find = (a) => { while (parent[a] !== a) a = parent[a] = parent[parent[a]]; return a; };
    const tol = this.bands?.tol ?? 0;
    vis.saddles.forEach((sid, j) => {
      if (this.doc.saddles[sid].value < c - tol) return;
      const [a, b] = vis.saddleEnds[j];
      parent[find(a)] = find(b);
    });
    const root = find(i), comp = new Set();
    for (let k = 0; k < n; k++) if (find(k) === root && this.doc.peaks[vis.peaks[k]].value > c) comp.add(k);
    comp.add(i);
    return comp;
  }

  /** Highlight band ``{c, i}`` (and the peaks it joins at that level), or clear with null. */
  #setBandHighlight(band) {
    if (this.bandMesh) {
      this.scene.remove(this.bandMesh);
      this.bandMesh.geometry.dispose();
      this.bandMesh = null;
    }
    this.bandHover = band;
    this.dirty = true;
    const B = this.bands;
    if (!band || !B) return;
    const comp = this.#bandComponent(band.c, band.i);
    band.peaks = [...comp].map((k) => this.vis.peaks[k]);
    const tint = hexToRgb(this.colors.select).map(srgbToLinearApprox);
    const pos = [], col = [];
    // the band runs from its contour value up to the next one (with smooth slopes: all the sub-steps)
    const next = B.base.find((v) => v > band.c + 1e-9) ?? Infinity;
    for (const sh of B.shapes) {
      if (sh.c < band.c - 1e-9 || sh.c >= next - 1e-9 || !comp.has(sh.i)) continue;
      for (let k = sh.start * 9; k < sh.end * 9; k += 3) {
        pos.push(B.pos[k], B.pos[k + 1], B.pos[k + 2] + B.lift);
        for (let q = 0; q < 3; q++) col.push(B.col[k + q] * 0.3 + tint[q] * 0.7);
      }
    }
    if (!pos.length) return;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(col), 3));
    this.bandMesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    this.bandMesh.frustumCulled = false;
    this.scene.add(this.bandMesh);
  }

  /**
   * Highlight, on behalf of another view, the band that holds value ``band.c`` around peak
   * ``band.docId`` here (the highest of this view's levels at or below it, as levels can differ,
   * e.g. a smooth landscape has more). Null clears it. Does not call the handlers.
   */
  showLinkedBand(band) {
    if (!band || !this.bands || !this.vis) { this.#setBandHighlight(null); return; }
    const i = this.vis.peaks.indexOf(band.docId);
    if (i < 0) { this.#setBandHighlight(null); return; }
    let best = -Infinity;
    for (const sh of this.bands.shapes) if (sh.i === i && sh.c <= band.c + 1e-9 && sh.c > best) best = sh.c;
    this.#setBandHighlight(Number.isFinite(best) ? { c: this.#bandBase(best), i, docId: band.docId } : null);
  }

  /**
   * Show ``node`` as hovered (ring and its links highlighted) on behalf of another view, e.g.
   * the landscape next to the flat spine; null clears it. Does not call the handlers.
   */
  showLinkedHover(node) {
    if (node && this.vis) {
      const list = node.kind === 'peak' ? this.vis.peaks : this.vis.saddles;
      if (!(node.index >= 0 && node.index < list.length)) node = null;
      else node = { kind: node.kind, index: node.index, docId: list[node.index] };
    }
    this.hover = node ?? null;
    this.#updateOverlay();
    this.dirty = true;
  }

  /**
   * Draw a ring around these peaks (document ids), e.g. the maxima that the other
   * method of a comparison does not have. Hidden peaks are skipped.
   */
  setMarked(docIds) {
    this.marked = new Set(docIds ?? []);
    this.#updateMarks();
    this.dirty = true;
  }

  /**
   * Colour peak ``i`` like peak ``ids[i]`` (compare mode: the same maximum gets the
   * same colour in both spines although the two files number their peaks differently).
   */
  setColorIds(ids) {
    this.colorIds = ids ?? null;
    if (this.vis) this.setVisible(this.vis);
  }

  #updateMarks() {
    if (!this.markMesh || !this.vis) return;
    const m = new THREE.Matrix4();
    let n = 0;
    this.vis.peaks.forEach((id, i) => {
      if (!this.marked.has(id)) return;
      const [x, y] = this.peakXY[id];
      const r = Math.max(this.radius[i] * 1.3, SADDLE_R * 2.5);
      m.makeScale(r, r, 1).setPosition(x, y, this.#zOf(this.doc.peaks[id].value, Z_RING - 0.01 - 2 * HEIGHT_EPS) + 2 * HEIGHT_EPS);
      this.markMesh.setMatrixAt(n++, m);
    });
    this.markMesh.count = n;
    this.markMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * 'simple' (discs and lines), 'paper' (nested contours, as in the paper) or
   * 'nddav' (nested contours drawn the way NDDAV draws them).
   */
  setStyle(style) {
    if (!['simple', 'paper', 'nddav', 'fig1'].includes(style)) throw new Error(`unknown spine style ${style}`);
    this.style = style;
    this.renderer.setClearColor(new THREE.Color(style === 'fig1' ? FIG1_BG : this.colors.bg), 1);
    if (this.vis) this.setVisible(this.vis);
  }

  /** Scale every contour radius (1 = the largest outer contour has radius rMax). */
  setContourScale(k) {
    this.contourScale = k;
    if (this.vis && this.style !== 'simple') this.setVisible(this.vis);
  }

  /**
   * Contour radius ~ volume^k: 1 is the paper's rule, 0.5 makes the area
   * proportional instead, 'nddav' uses k = 2/d (d = dimension of the data), as NDDAV.
   */
  setRadiusExponent(k) {
    if (k !== 'nddav' && !(Number(k) > 0)) throw new Error(`bad radius exponent ${k}`);
    this.radiusExponent = k === 'nddav' ? k : Number(k);
    if (this.vis && this.style !== 'simple') this.setVisible(this.vis);
  }

  /** The exponent actually used: 2/d for 'nddav' (d from the file, 2 when unknown). */
  get effectiveExponent() {
    if (this.radiusExponent !== 'nddav') return this.radiusExponent;
    const d = Number(this.doc?.meta?.dimensions);
    return 2 / (d >= 1 ? d : 2);
  }

  /** Number of evenly spaced isovalues of the 'nddav' style (the saddle values are always added). */
  setLevelCount(n) {
    this.levelCount = Math.max(2, Math.round(n));
    if (this.vis && this.style === 'nddav') this.setVisible(this.vis);
  }

  /**
   * Which contour sets the scale: 'saddle' gives the outermost contour at the
   * lowest shown saddle radius rMax and caps the halo below it at 2 x rMax;
   * 'halo' gives the outer halo (field minimum) radius rMax, nothing capped.
   */
  setScaleMode(mode) {
    if (mode !== 'saddle' && mode !== 'halo') throw new Error(`unknown scale mode ${mode}`);
    this.scaleMode = mode;
    if (this.vis && this.style !== 'simple') this.setVisible(this.vis);
  }

  /** Minimum gap between drawn isovalues, as a fraction of the value range. */
  setLevelGap(g) {
    this.levelGap = g;
    if (this.vis && this.style !== 'simple') this.setVisible(this.vis);
  }

  /** True while the paper-style drawing is on screen (it falls back to discs without per-point data). */
  get paperStyleActive() { return Boolean(this.paperActive); }

  /** Draw the arcs of equation (3) (saturated persistence) instead of straight links. */
  setShowArcs(on) {
    this.showArcs = Boolean(on);
    if (this.vis) this.setVisible(this.vis);
  }

  #arcsActive() { return this.showArcs && Array.isArray(this.vis?.arcs); }

  /** Lay the visible nodes out again after simplification (on by default). */
  /**
   * Height mode: every contour, peak, saddle and link stands at a height given by its value, so
   * the spine reads as a landscape (peaks as summits, saddles as passes); drag to orbit,
   * shift-drag to pan. ``scale`` stretches the heights; ``smooth`` adds levels between the
   * contours so the slopes rise in small steps instead of a few terraces; ``solid`` builds one
   * solid, shaded mountain (walls between the levels) coloured only at the contour values.
   */
  setHeight({ on = this.heightOn, scale = this.heightScale, smooth = this.heightSmooth, solid = this.heightSolid } = {}) {
    const turnedOn = on && !this.heightOn;
    const next = [Boolean(on), Math.max(0.05, Number(scale) || 1), Boolean(smooth), Boolean(solid)];
    const prev = [this.heightOn, this.heightScale, this.heightSmooth, this.heightSolid];
    [this.heightOn, this.heightScale, this.heightSmooth, this.heightSolid] = next;
    if (turnedOn) { this.yaw = -Math.PI / 2; this.pitch = 0.95; }
    // rebuild only if it shows: a flat view's sub-steps or scale do not change its drawing
    const changed = next[0] !== prev[0] || (next[0] && next.some((v, k) => v !== prev[k]));
    if (changed && this.vis) this.setVisible(this.vis);
    this.dirty = true;
  }

  /** Height of value ``v`` in height mode, else ``flat`` (the 2-D drawing order). */
  #zOf(v, flat) {
    if (!this.heightOn) return flat;
    const [lo, hi] = this.doc?.meta?.value_range ?? this.valueRange ?? [0, 1];   // as the contours
    const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
    return HEIGHT_BASE * this.heightScale * Math.min(1, Math.max(0, t));
  }

  /** Position of a visible node ([x, y, z]); z is its height in height mode. */
  #nodeXYZ(node, flatZ = 0) {
    const peak = node.kind === 'peak';
    const id = peak ? this.vis.peaks[node.index] : this.vis.saddles[node.index];
    const [x, y] = peak ? this.peakXY[id] : this.saddleXY[id];
    const v = peak ? this.doc.peaks[id].value : this.doc.saddles[id].value;
    return [x, y, this.#zOf(v, flatZ)];
  }

  /** Paper style: on, no contour far below the lowest shown saddle (see ``#contourFloor``). */
  setFloor(on) {
    this.floorOn = Boolean(on);
    if (this.vis) this.setVisible(this.vis);
  }

  setRelayout(on) {
    this.relayoutOn = Boolean(on);
    if (this.vis) this.setVisible(this.vis);
  }

  /**
   * Positions for ``vis``: the exported layout, with the drawn nodes (and, when the p*
   * arcs are on, the merged peaks and saddles they pass through) laid out again by
   * ``relayout``. Falls back to the exported layout when that is not possible.
   */
  #applyRelayout(vis) {
    this.peakXY = this.basePeakXY;
    this.saddleXY = this.baseSaddleXY;
    this.relayoutInfo = null;
    if (!this.relayoutOn || !this.basePeakXY) return;
    const arcs = this.showArcs && Array.isArray(vis.arcs);
    const peaks = arcs ? [...new Set([...vis.peaks, ...(vis.ghosts ?? [])])].sort((a, b) => a - b) : vis.peaks;
    const saddles = arcs ? [...new Set([...vis.saddles, ...vis.arcs])].sort((a, b) => a - b) : vis.saddles;
    const key = peaks.join(',') + '|' + saddles.join(',');
    if (!this.relayoutCache.has(key)) this.relayoutCache.set(key, relayout(this.doc, peaks, saddles));
    const r = this.relayoutCache.get(key);
    if (!r) return;
    const P = this.basePeakXY.slice(), S = this.baseSaddleXY.slice();
    r.peaks.forEach((xy, id) => { P[id] = xy; });
    r.saddles.forEach((xy, id) => { S[id] = xy; });
    this.peakXY = P;
    this.saddleXY = S;
    this.relayoutInfo = r;
  }

  /** Recolour with the active value colour map (after ``setColormap``). */
  refreshColors() {
    if (this.vis) this.setVisible(this.vis);     // rebuilds the contour colours and the disc colours
    this.dirty = true;
  }

  setTheme(colors) {
    Object.assign(this.colors, colors);
    this.renderer.setClearColor(new THREE.Color(this.style === 'fig1' ? FIG1_BG : this.colors.bg), 1);
    this.linkMat.color.set(this.colors.link);
    this.ghostMat.color.set(this.colors.link);
    this.ringMat.color.set(this.colors.ring);
    this.selectMat.color.set(this.colors.select);
    this.highlightMat.color.set(this.colors.select);
    this.markMat.color.set(this.colors.mark);
    if (this.vis) this.setVisible(this.vis);     // saddle colours are per instance
    this.dirty = true;
  }

  fit() {
    const aspect = this.#aspect();
    const reach = 1 + Math.max(0.25, (this.pad ?? 0) + 0.05);   // layout box is [-1, 1], plus the shapes
    this.center.set(0, 0);
    this.halfHeight = this.homeHalfHeight = Math.max(reach, reach / aspect);
    this.dirty = true;
    this.#cameraChanged();
  }

  /** The visible part of the layout: centre and half height (plus the reset zoom). */
  getCamera() {
    return { x: this.center.x, y: this.center.y, halfHeight: this.halfHeight, homeHalfHeight: this.homeHalfHeight };
  }

  /** Show the same part of the layout as another view (compare mode); does not call onCamera. */
  setCamera(cam) {
    this.center.set(cam.x, cam.y);
    this.halfHeight = cam.halfHeight;
    this.homeHalfHeight = cam.homeHalfHeight ?? this.homeHalfHeight;
    this.dirty = true;
  }

  #cameraChanged() { this.handlers.onCamera?.(this.getCamera()); }

  resize() {
    const w = Math.max(this.container.clientWidth, 10);
    const h = Math.max(this.container.clientHeight, 10);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.dirty = true;
  }

  dispose() {
    this.stopped = true;
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  /** Name of the node's peak ids etc. is up to the caller; this gives screen position. */
  nodeScreenPosition(node) {
    const el = this.renderer.domElement;
    if (this.heightOn) {
      this.#placeCamera();
      const p = new THREE.Vector3(...this.#nodeXYZ(node)).project(this.camera);
      return { x: (p.x * 0.5 + 0.5) * el.clientWidth, y: (0.5 - p.y * 0.5) * el.clientHeight };
    }
    const [x, y] = node.kind === 'peak'
      ? this.peakXY[this.vis.peaks[node.index]] : this.saddleXY[this.vis.saddles[node.index]];
    const aspect = this.#aspect();
    return {
      x: ((x - this.center.x) / (this.halfHeight * aspect) * 0.5 + 0.5) * el.clientWidth,
      y: (0.5 - (y - this.center.y) / this.halfHeight * 0.5) * el.clientHeight,
    };
  }

  /**
   * Build (or clear) the contour mesh. Returns true when the paper style is
   * being drawn; also widens ``this.radius`` to the outer contour so that
   * picking and the selection ring cover the whole shape.
   */
  #buildPaperStyle(vis) {
    if (this.contourMesh) {
      this.scene.remove(this.contourMesh);
      this.contourMesh.geometry.dispose();
      this.contourMesh = null;
    }
    this.bands = null;                     // the bands under the pointer are found again on the new shapes
    this.#setBandHighlight(null);
    if (this.style === 'simple') return false;
    const h = regionHistograms(this.doc, vis);
    if (!h) return false;
    const sc = spineScale(this.doc, vis, h, this.scaleMode);
    this.scaleInfo = sc;
    const nddav = this.style === 'nddav', fig1 = this.style === 'fig1';
    const rBase = nddav ? this.#nddavRadius(vis) : this.rMax;
    // 'fig1': radius ~ volume^(1/k), the radius of a ball holding that many points, with k = min(d, 3)
    // (in high dimensions 1/d flattens every size to the same; 3 keeps the rings nested). Sized so the
    // contours fill about half of each link, as in the paper's Fig. 1. Above 3-D also never more than
    // FIG1_SPACING x the median nearest-peak distance, so neighbouring circles do not swallow each other.
    const d = Number(this.doc?.meta?.dimensions) >= 1 ? Number(this.doc.meta.dimensions) : 2;
    const rRef = this.contourScale * (fig1
      ? Math.min(FIG1_FIT * this.#medianLink(vis), d > 3 ? FIG1_SPACING * this.#peakSpacing(vis) : Infinity)
      : Math.min(rBase, SPACING_FIT * this.#peakSpacing(vis)));
    const exponent = fig1 ? 1 / Math.min(d, 3) : this.effectiveExponent;
    if (fig1) this.#fig1Positions(vis, h, rRef / sc.ref ** exponent, exponent);
    const g = buildContours(this.doc, vis, h, this.peakXY, this.saddleXY, {
      rs: rRef / sc.ref ** exponent,
      maxRadius: rRef * sc.capFactor,
      knee: nddav ? rRef : undefined,          // NDDAV style: squeeze the halo softly, keep the taper
      exponent, gap: this.levelGap,
      floor: nddav ? undefined : this.#contourFloor(vis),
      shape: nddav ? 'nddav' : fig1 ? 'fig1' : 'paper', levelCount: this.levelCount,
      edgeWidth: nddav ? 0.004 : fig1 ? 0.0018 : 0.003,
      halo: fig1 ? 0.32 * rRef : 0,
      colorOf: fig1 ? fig1Color : peakColor, bg: hexToRgb(fig1 ? FIG1_BG : this.colors.bg),
      edge: fig1 ? [0.42, 0.44, 0.5] : hexToRgb(this.colors.ring),
      fade: fig1 ? false : undefined,
      z0: this.heightOn ? 0 : Z_CONTOUR,
      zRange: this.heightOn ? HEIGHT_BASE * this.heightScale : Z_CONTOUR_RANGE,
      subdivide: !this.heightOn ? 0 : this.heightSolid ? 6 : this.heightSmooth ? 4 : 0,
      solid: this.heightOn && this.heightSolid,
      route: this.#arcsActive()            // lobes follow the p* ridge through merged peaks
        ? (i, sid) => ridgeRoute(this.doc, vis, i, sid, this.peakXY, this.saddleXY) : null,
    });
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(g.positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(g.colors, 3));
    this.contourMesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({
      vertexColors: true, side: THREE.DoubleSide }));
    this.contourMesh.frustumCulled = false;
    this.scene.add(this.contourMesh);
    const zRange = this.heightOn ? HEIGHT_BASE * this.heightScale : Z_CONTOUR_RANGE;
    this.bands = {
      shapes: g.shapes, pos: g.positions, col: g.colors,
      order: g.shapes.map((_, k) => k).sort((a, b) => g.shapes[b].c - g.shapes[a].c),   // highest first
      lift: zRange * 0.0008 + 2e-5,                        // the highlight sits just above its band
      base: [...(g.baseLevels ?? g.levels)].sort((a, b) => a - b),   // contour values (smooth slopes add sub-steps)
      tol: fig1 ? Math.max(this.levelGap, 0.02) * ((this.doc.meta.value_range?.[1] ?? 1) - (this.doc.meta.value_range?.[0] ?? 0)) : 0,
    };
    this.radius = this.radius.map((r, i) => Math.max(this.rMin, Math.min(g.extent[i], this.rMax * 2.5)));
    this.pad = Math.max(0, ...g.extent);
    return true;
  }

  /**
   * Lowest isovalue of the paper style: one level gap below the lowest shown
   * saddle, so every link still closes into one outline, but the empty space
   * far below it (most of a volume) does not count. Undefined when it is off.
   */
  #contourFloor(vis) {
    if (!this.floorOn || !vis.saddles.length) return undefined;
    const [lo, hi] = this.doc.meta.value_range ?? [0, 1];
    const low = Math.min(...vis.saddles.map((id) => this.doc.saddles[id].value));
    return Math.max(lo, low - Math.max(this.levelGap, 0.02) * (hi - lo));
  }

  /**
   * Reference radius of the 'nddav' style. NDDAV sizes its contours in layout
   * units, so the outer halo is about as wide as the links are long; here it is
   * NDDAV_HALO x the median peak-saddle link (never below the disc size rMax,
   * at most 0.6 of the [-1, 1] layout box).
   */
  /**
   * Median, over the visible peaks, of the layout distance from each peak to its
   * nearest other visible peak. Infinity with fewer than two peaks (no cap).
   */
  #peakSpacing(vis) {
    const pts = vis.peaks.map((id) => this.peakXY[id]);
    if (pts.length < 2) return Infinity;
    const nn = pts.map(([x, y], i) => {
      let best = Infinity;
      pts.forEach(([u, v], j) => { if (j !== i) best = Math.min(best, Math.hypot(x - u, y - v)); });
      return best;
    }).filter((d) => d > 0).sort((a, b) => a - b);
    return nn.length ? nn[nn.length >> 1] : Infinity;
  }

  /**
   * 'fig1' style: where the drawing puts the saddles and some leaf peaks, to look like the paper's
   * Fig. 1. A saddle sits on the straight line between its two peaks, where their circles at the
   * saddle's value meet. An arm (a single-link peak much lower than the peak it hangs from) gets its
   * saddle beyond the big peak's circle, so the big contour spikes out to it and the arm's teardrop
   * meets the spike (a figure-eight); the arm is moved out to make room. Only drawing copies change.
   */
  #fig1Positions(vis, h, rs, k) {
    const doc = this.doc;
    const P = this.peakXY.slice(), S = this.saddleXY.slice();
    const deg = new Array(vis.peaks.length).fill(0);
    vis.saddleEnds.forEach(([a, b]) => { if (a !== b) { deg[a]++; deg[b]++; } });
    const top = (i) => doc.peaks[vis.peaks[i]].value;
    const rad = (i, c) => rs * volumeBetween(h, i, c, top(i)) ** k;   // circle of peak i at value c
    const floorC = this.#contourFloor(vis) ?? (doc.meta.value_range?.[0] ?? 0);
    // big peaks first, so an arm is placed after the circle it hangs from
    const order = vis.saddles.map((_, j) => j).sort((x, y) => {
      const [a, b] = vis.saddleEnds[y], [c, d] = vis.saddleEnds[x];
      return Math.max(vis.size[a], vis.size[b]) - Math.max(vis.size[c], vis.size[d]);
    });
    for (const j of order) {
      const [a0, b0] = vis.saddleEnds[j];
      if (a0 === b0) continue;
      const sv = doc.saddles[vis.saddles[j]].value;
      const ra0 = rad(a0, sv), rb0 = rad(b0, sv);
      const [A, B] = ra0 >= rb0 ? [a0, b0] : [b0, a0];          // A: the larger at the saddle's value
      const rA = Math.max(ra0, rb0), rB = Math.min(ra0, rb0);
      const pa = P[vis.peaks[A]], pb = P[vis.peaks[B]];
      let dx = pb[0] - pa[0], dy = pb[1] - pa[1];
      let dist = Math.hypot(dx, dy);
      if (dist < 1e-9) { dx = 1; dy = 0; dist = 1e-9; }
      const ux = dx / dist, uy = dy / dist;
      let t;                                                    // distance of the saddle from A
      // An arm: a single-link peak that rises only a little above its saddle compared with the peak
      // it hangs from (fuel's four side peaks; not the jet's tip, which is as high as its neighbour).
      const arm = deg[B] === 1 && top(B) - sv < 0.5 * (top(A) - sv);
      if (arm) {
        // A's contour at the saddle's value runs out as a spike to the saddle (0.7 x its circle beyond
        // the circle), and the arm's teardrop comes back from the other side: the figure-eight.
        t = rA * 1.7;
        const tear = 3.2 * Math.max(rad(B, floorC), 0.25 * rA); // length of the arm's teardrop
        if (dist < t + tear) {
          dist = t + tear;
          P[vis.peaks[B]] = [pa[0] + ux * dist, pa[1] + uy * dist];
        }
      } else {
        // two peaks of a chain: where their circles at the saddle's value would meet, pushed apart
        // if they overlap so the lens pinches visibly
        const need = 1.15 * (rA + rB);
        if (deg[B] === 1 && dist < need) {
          dist = need;
          P[vis.peaks[B]] = [pa[0] + ux * dist, pa[1] + uy * dist];
        }
        t = dist * (rA / Math.max(1e-12, rA + rB));
      }
      S[vis.saddles[j]] = [pa[0] + ux * t, pa[1] + uy * t];
    }
    this.peakXY = P;
    this.saddleXY = S;
  }

  /** Median layout length of the shown peak-saddle links (rMax * 2 without links). */
  #medianLink(vis) {
    const len = [];
    vis.saddles.forEach((id, j) => {
      const [sx, sy] = this.saddleXY[id];
      for (const e of new Set(vis.saddleEnds[j])) {
        const [px, py] = this.peakXY[vis.peaks[e]];
        len.push(Math.hypot(px - sx, py - sy));
      }
    });
    if (!len.length) return this.rMax * 2;
    len.sort((a, b) => a - b);
    return len[len.length >> 1];
  }

  #nddavRadius(vis) {
    const len = [];
    vis.saddles.forEach((id, j) => {
      const [sx, sy] = this.saddleXY[id];
      for (const e of new Set(vis.saddleEnds[j])) {
        const [px, py] = this.peakXY[vis.peaks[e]];
        len.push(Math.hypot(px - sx, py - sy));
      }
    });
    if (!len.length) return this.rMax * 2;
    len.sort((a, b) => a - b);
    return Math.min(0.6, Math.max(this.rMax, NDDAV_HALO * len[len.length >> 1]));
  }

  // ---- scene objects ----------------------------------------------------
  #buildFixedObjects() {
    this.linkMat = new THREE.LineBasicMaterial({ color: this.colors.link, transparent: true, opacity: 0.55 });
    this.highlightMat = new THREE.LineBasicMaterial({ color: this.colors.select });
    this.ghostMat = new THREE.MeshBasicMaterial({ color: this.colors.link });
    this.ringMat = new THREE.MeshBasicMaterial({ color: this.colors.ring });
    this.selectMat = new THREE.MeshBasicMaterial({ color: this.colors.select });
    this.markMat = new THREE.MeshBasicMaterial({ color: this.colors.mark });
    this.peakGeom = new THREE.CircleGeometry(1, 48);
    this.saddleGeom = new THREE.CircleGeometry(1, 4);
    this.saddleGeom.rotateZ(Math.PI / 4);                // axis-aligned diamond
    this.ringGeom = new THREE.RingGeometry(1.0, 1.18, 48);

    this.hoverRing = new THREE.Mesh(this.ringGeom, this.ringMat);
    this.selectRing = new THREE.Mesh(this.ringGeom, this.selectMat);
    for (const r of [this.hoverRing, this.selectRing]) {
      r.visible = false; r.position.z = Z_RING; r.renderOrder = 10; this.scene.add(r);
    }
    this.renderer.setClearColor(new THREE.Color(this.colors.bg), 1);
  }

  #allocate(nP, nS) {
    // peakGeom / saddleGeom are shared by every allocation, so only the
    // per-graph line buffers and instance materials are released here.
    for (const o of [this.peakMesh, this.saddleMesh, this.linkLines, this.highlightLines]) {
      if (!o) continue;
      this.scene.remove(o);
      if (o.isInstancedMesh) { o.material.dispose(); o.dispose(); } else o.geometry.dispose();
    }
    if (this.ghostMesh) { this.scene.remove(this.ghostMesh); this.ghostMesh.dispose(); }   // shared material
    if (this.markMesh) { this.scene.remove(this.markMesh); this.markMesh.dispose(); }      // shared material
    this.markMesh = new THREE.InstancedMesh(this.ringGeom, this.markMat, Math.max(nP, 1));
    this.markMesh.count = 0;
    this.markMesh.frustumCulled = false;
    this.markMesh.renderOrder = 9;
    this.ghostMesh = new THREE.InstancedMesh(this.peakGeom, this.ghostMat, Math.max(nP, 1));
    this.ghostMesh.count = 0;
    this.ghostMesh.frustumCulled = false;
    this.peakMesh = new THREE.InstancedMesh(this.peakGeom, new THREE.MeshBasicMaterial(), Math.max(nP, 1));
    this.saddleMesh = new THREE.InstancedMesh(this.saddleGeom, new THREE.MeshBasicMaterial(), Math.max(nS, 1));
    this.peakMesh.count = this.saddleMesh.count = 0;
    this.peakMesh.frustumCulled = this.saddleMesh.frustumCulled = false;

    this.linkGeom = new THREE.BufferGeometry();
    this.linkGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(nS, 1) * 12), 3));
    this.linkGeom.setDrawRange(0, 0);
    this.linkLines = new THREE.LineSegments(this.linkGeom, this.linkMat);
    this.linkLines.frustumCulled = false;

    this.highlightGeom = new THREE.BufferGeometry();
    this.highlightGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(Math.max(nS, 1) * 12), 3));
    this.highlightGeom.setDrawRange(0, 0);
    this.highlightLines = new THREE.LineSegments(this.highlightGeom, this.highlightMat);
    this.highlightLines.frustumCulled = false;
    this.highlightLines.position.z = 0.05;

    this.scene.add(this.linkLines, this.highlightLines, this.ghostMesh, this.saddleMesh, this.peakMesh,
                   this.markMesh);
  }

  /** Move a selection to the new visible indices after the slider changed. */
  #reindex(node) {
    if (!node || !this.vis) return null;
    // The previous visible lists are gone, so selections are remembered by document id.
    const id = node.docId;
    const list = node.kind === 'peak' ? this.vis.peaks : this.vis.saddles;
    const index = list.indexOf(id);
    return index < 0 ? null : { ...node, index };
  }

  #updateOverlay() {
    const place = (ring, node) => {
      if (!node || !this.vis) { ring.visible = false; return; }
      const peak = node.kind === 'peak';
      const [x, y, z] = this.#nodeXYZ(node, Z_RING - 3 * HEIGHT_EPS);
      const r = (peak ? this.radius[node.index] : SADDLE_R) ;
      ring.position.set(x, y, z + 3 * HEIGHT_EPS);
      ring.scale.set(r, r, 1);
      ring.visible = true;
    };
    place(this.hoverRing, this.hover);
    place(this.selectRing, this.selected);

    // Links of the selected node in the accent colour.
    const pos = this.highlightGeom?.attributes.position.array;
    let n = 0;
    const focus = this.selected ?? this.hover;     // a hovered node's links light up too
    if (focus && this.vis && pos) {
      const { vis } = this;
      const arcs = this.#arcsActive();
      const along = new Set();         // arcs mode: saddles whose two arcs are highlighted
      for (let j = 0; j < vis.saddles.length; j++) {
        const ends = vis.saddleEnds[j];
        const touches = focus.kind === 'peak'
          ? ends.includes(focus.index) : focus.index === j;
        if (!touches) continue;
        const sid = vis.saddles[j];
        if (arcs) {
          // follow the drawn arcs: saddle -> its own peaks -> ridge -> surviving peaks
          along.add(sid);
          for (const p of this.doc.saddles[sid].peaks) for (const r of arcPath(this.doc, vis, p)) along.add(r);
          continue;
        }
        const [sx, sy] = this.saddleXY[sid];
        const sz = this.#zOf(this.doc.saddles[sid].value, -2 * HEIGHT_EPS) + 2 * HEIGHT_EPS;
        for (const e of ends) {
          const [px, py] = this.peakXY[vis.peaks[e]];
          const pz = this.#zOf(this.doc.peaks[vis.peaks[e]].value, -2 * HEIGHT_EPS) + 2 * HEIGHT_EPS;
          pos.set([px, py, pz, sx, sy, sz], n * 6);
          n++;
        }
      }
      for (const sid of along) {
        const [sx, sy] = this.saddleXY[sid];
        const sz = this.#zOf(this.doc.saddles[sid].value, -2 * HEIGHT_EPS) + 2 * HEIGHT_EPS;
        for (const p of this.doc.saddles[sid].peaks) {
          const [px, py] = this.peakXY[p];
          const pz = this.#zOf(this.doc.peaks[p].value, -2 * HEIGHT_EPS) + 2 * HEIGHT_EPS;
          pos.set([px, py, pz, sx, sy, sz], n * 6);
          n++;
        }
      }
    }
    if (this.highlightGeom) {
      this.highlightGeom.setDrawRange(0, n * 2);
      this.highlightGeom.attributes.position.needsUpdate = true;
    }
  }

  // ---- interaction ------------------------------------------------------
  #aspect() {
    const el = this.renderer.domElement;
    return Math.max(el.clientWidth, 1) / Math.max(el.clientHeight, 1);
  }

  #toWorld(ev) {
    if (this.heightOn) return { ev };                // height mode picks on screen (#pickScreen)
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    const nx = (ev.clientX - rect.left) / rect.width * 2 - 1;
    const ny = 1 - (ev.clientY - rect.top) / rect.height * 2;
    const aspect = this.#aspect();
    return new THREE.Vector2(this.center.x + nx * this.halfHeight * aspect,
                             this.center.y + ny * this.halfHeight);
  }

  /** Height mode: nearest node to the pointer on screen (peaks first), else null. */
  #pickScreen(ev) {
    if (!this.vis) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    const near = (kind, n, slack) => {
      let best = null, bestD = slack;
      for (let i = 0; i < n; i++) {
        const node = { kind, index: i };
        const q = this.nodeScreenPosition(node);
        const d = Math.hypot(q.x - mx, q.y - my);
        if (d < bestD) {
          bestD = d;
          best = { kind, index: i, docId: kind === 'peak' ? this.vis.peaks[i] : this.vis.saddles[i] };
        }
      }
      return best;
    };
    return near('peak', this.vis.peaks.length, 10) ?? near('saddle', this.vis.saddles.length, 8);
  }

  /** Nearest peak under the pointer, else nearest saddle, else null. */
  #pick(world) {
    if (world?.ev) return this.#pickScreen(world.ev);
    if (!this.vis) return null;
    const px = (2 * this.halfHeight) / Math.max(this.renderer.domElement.clientHeight, 1);
    const slack = 3 * px;
    let best = null, bestD = Infinity;
    this.vis.peaks.forEach((id, i) => {
      const [x, y] = this.peakXY[id];
      const d = Math.hypot(world.x - x, world.y - y) - this.radius[i];
      if (d < slack && d < bestD) { bestD = d; best = { kind: 'peak', index: i, docId: id }; }
    });
    if (best) return best;
    this.vis.saddles.forEach((id, j) => {
      const [x, y] = this.saddleXY[id];
      const d = Math.hypot(world.x - x, world.y - y) - SADDLE_R;
      if (d < slack * 1.5 && d < bestD) { bestD = d; best = { kind: 'saddle', index: j, docId: id }; }
    });
    return best;
  }

  #bindEvents() {
    const el = this.renderer.domElement;
    let drag = null;

    el.addEventListener('pointerdown', (ev) => {
      el.setPointerCapture(ev.pointerId);
      drag = { x: ev.clientX, y: ev.clientY, moved: false, center: this.center.clone(),
               yaw: this.yaw, pitch: this.pitch, orbit: this.heightOn && !ev.shiftKey };
    });
    el.addEventListener('pointermove', (ev) => {
      if (drag) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.hypot(dx, dy) > 4) drag.moved = true;
        if (drag.moved && drag.orbit) {          // height mode: drag orbits the landscape
          this.yaw = drag.yaw - dx * 0.008;
          this.pitch = Math.min(1.45, Math.max(0, drag.pitch - dy * 0.008));
          this.dirty = true;
        } else if (drag.moved && this.heightOn) {   // shift-drag pans along the screen
          const px = (2 * this.halfHeight) / el.clientHeight;
          const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0);
          const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1);
          this.center.set(drag.center.x - (right.x * dx - up.x * dy) * px,
                          drag.center.y - (right.y * dx - up.y * dy) * px);
          this.dirty = true;
        } else if (drag.moved) {
          const px = (2 * this.halfHeight) / el.clientHeight;
          this.center.set(drag.center.x - dx * px, drag.center.y + dy * px);
          this.dirty = true;
          this.#cameraChanged();
        }
        return;
      }
      const hit = this.#pick(this.#toWorld(ev));
      const same = (a, b) => (!a && !b) || (a && b && a.kind === b.kind && a.index === b.index);
      if (!same(hit, this.hover)) {
        this.hover = hit;
        this.#updateOverlay();
        el.style.cursor = hit ? 'pointer' : 'grab';
        this.handlers.onHover?.(hit, hit ? this.nodeScreenPosition(hit) : null);
        this.dirty = true;
      }
      // no node under the pointer: the contour band there (after the node, so its label wins)
      const band = hit ? null : this.#bandAt(ev);
      const sameBand = (a, b) => (!a && !b) || (a && b && a.c === b.c && a.i === b.i);
      if (!sameBand(band, this.bandHover)) {
        this.#setBandHighlight(band);
        const rect = el.getBoundingClientRect();
        this.handlers.onBandHover?.(band ? { c: band.c, docId: band.docId, peaks: band.peaks } : null,
          band ? { x: ev.clientX - rect.left, y: ev.clientY - rect.top } : null);
      }
    });
    el.addEventListener('pointerup', (ev) => {
      if (drag && !drag.moved) this.select(this.#pick(this.#toWorld(ev)));
      drag = null;
    });
    el.addEventListener('pointerleave', () => {
      if (this.bandHover && !drag) { this.#setBandHighlight(null); this.handlers.onBandHover?.(null, null); }
      if (this.hover && !drag) {
        this.hover = null; this.#updateOverlay(); this.handlers.onHover?.(null, null); this.dirty = true;
      }
    });
    el.addEventListener('dblclick', () => this.fit());
    el.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const factor = Math.exp(ev.deltaY * 0.0015);
      if (this.heightOn) {                                   // height mode: zoom about the centre
        this.halfHeight = Math.min(this.homeHalfHeight * 4, Math.max(0.04, this.halfHeight * factor));
        this.dirty = true;
        this.#cameraChanged();
        return;
      }
      const before = this.#toWorld(ev);
      this.halfHeight = Math.min(this.homeHalfHeight * 4, Math.max(0.04, this.halfHeight * factor));
      const after = this.#toWorld(ev);                       // keep the point under the cursor fixed
      this.center.x += before.x - after.x;
      this.center.y += before.y - after.y;
      this.dirty = true;
      this.#cameraChanged();
    }, { passive: false });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    el.style.cursor = 'grab';
  }

  // ---- rendering --------------------------------------------------------
  /** Flat: straight down on the layout. Height mode: orbiting the landscape (yaw, pitch). */
  #placeCamera() {
    const aspect = this.#aspect();
    const c = this.camera;
    c.left = -this.halfHeight * aspect; c.right = this.halfHeight * aspect;
    c.top = this.halfHeight; c.bottom = -this.halfHeight;
    if (this.heightOn) {
      const H = HEIGHT_BASE * this.heightScale;
      const target = new THREE.Vector3(this.center.x, this.center.y, H / 2);
      const sp = Math.sin(this.pitch), cp = Math.cos(this.pitch);
      // at pitch 0 look straight down with +y up on screen; tilting keeps the near side at the bottom
      const dir = new THREE.Vector3(sp * Math.cos(this.yaw), sp * Math.sin(this.yaw), cp);
      c.position.copy(target).addScaledVector(dir, 20);
      c.up.set(-Math.cos(this.yaw), -Math.sin(this.yaw), 0).multiplyScalar(cp).add(new THREE.Vector3(0, 0, sp));
      if (this.pitch < 1e-3) c.up.set(-Math.cos(this.yaw), -Math.sin(this.yaw), 0);
      c.near = 0.1; c.far = 60;
      c.lookAt(target);
    } else {
      c.position.set(this.center.x, this.center.y, 5);
      c.up.set(0, 1, 0);
      c.near = -10; c.far = 10;
      c.lookAt(this.center.x, this.center.y, 0);
    }
    c.updateProjectionMatrix();
    c.updateMatrixWorld();
  }

  #loop() {
    if (this.stopped) return;
    if (this.dirty) {
      this.#placeCamera();
      this.renderer.render(this.scene, this.camera);
      this.dirty = false;
    }
    requestAnimationFrame(() => this.#loop());
  }
}

const SADDLE_R = 0.016;
const GHOST_R = 0.007;        // merged-away peak on a drawn arc

/** '#rgb' or '#rrggbb' to [r, g, b] in [0, 1] (sRGB); white if it cannot be read. */
export function hexToRgb(hex) {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(hex).trim());
  if (!m) return [1, 1, 1];
  const t = m[1].length === 3 ? [...m[1]].map((c) => c + c).join('') : m[1];
  return [0, 2, 4].map((o) => parseInt(t.slice(o, o + 2), 16) / 255);
}

/** Point (x, y) inside any triangle start..end of a flat [x, y, z] x 3 array? */
function insideTriangles(pos, start, end, x, y) {
  for (let k = start * 9; k < end * 9; k += 9) {
    const ax = pos[k], ay = pos[k + 1], bx = pos[k + 3], by = pos[k + 4], cx = pos[k + 6], cy = pos[k + 7];
    const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by);
    const d2 = (x - cx) * (by - cy) - (bx - cx) * (y - cy);
    const d3 = (x - ax) * (cy - ay) - (cx - ax) * (y - ay);
    const neg = d1 < 0 || d2 < 0 || d3 < 0, pos3 = d1 > 0 || d2 > 0 || d3 > 0;
    if (!(neg && pos3)) return true;
  }
  return false;
}

/** sRGB component to linear (the contour colours are stored linear). */
function srgbToLinearApprox(c) { return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
