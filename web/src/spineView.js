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
import { regionHistograms, buildContours, spineScale } from './contours.js';
import { arcPath, ridgeRoute } from './simplify.js';

const Z_LINK = 0.0, Z_SADDLE = 0.1, Z_PEAK = 0.2, Z_RING = 0.3;
const Z_CONTOUR = -0.6, Z_CONTOUR_RANGE = 0.5;   // paper-style shapes sit below everything else
// Colour maps for field values, sampled at 11 even stops from matplotlib (sRGB).
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
   * @param {{onHover?: Function, onSelect?: Function}} handlers called with a node
   *        ``{kind: 'peak'|'saddle', index, docId}`` (index into the visible lists) or null
   * @param {{renderer?: object}} options ``renderer`` replaces the WebGL renderer
   *        (used by the tests, which have no GPU)
   */
  constructor(container, handlers = {}, options = {}) {
    this.container = container;
    this.handlers = handlers;
    this.colors = { bg: '#ffffff', link: '#8a8f98', saddle: '#6b7280', loop: '#d97706',
                    ring: '#111827', select: '#2563eb' };

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

    this.style = 'simple';     // 'simple' (discs and lines) or 'paper' (nested contours)
    this.contourScale = 1;     // multiplies the radius of every contour
    this.radiusExponent = 1;   // contour radius ~ volume^k: 1 as in the paper, 0.5 = area ~ volume
    this.levelGap = 0.02;      // minimum isovalue gap, fraction of the value range (paper, section 4.4)
    this.scaleMode = 'saddle'; // which contour gets radius rMax: 'saddle' (lowest saddle; halo capped) or 'halo'
    this.showArcs = false;     // draw the arcs of equation (3), through merged-away peaks
    this.contourMesh = null;
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
    const sp = doc.spine;
    if (!sp) throw new Error('this file has no spine layout');

    this.peakXY = sp.peaks;
    this.saddleXY = sp.saddles;
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
    this.linkLines.visible = !paper || arcs;         // over the contours, the arcs mark the ridges
    this.paperActive = paper;

    const [lo, hi] = this.valueRange;
    const m = new THREE.Matrix4(), col = new THREE.Color();
    for (let i = 0; i < nP; i++) {
      const id = vis.peaks[i];
      const [x, y] = this.peakXY[id];
      const r = this.radius[i];
      const dot = paper ? Math.max(SADDLE_R * 0.9, Math.min(r * 0.14, 0.024)) : r;   // peak marker
      m.makeScale(dot, dot, 1).setPosition(x, y, Z_PEAK);
      this.peakMesh.setMatrixAt(i, m);
      const [R, G, B] = peakColor(hi > lo ? (doc.peaks[id].value - lo) / (hi - lo) : 0.5);
      const [rr, rg, rb] = regionColor(id);              // paper style: the dot shows the region's colour
      this.peakMesh.setColorAt(i, paper ? col.setRGB(rr, rg, rb, THREE.SRGBColorSpace)
                                        : col.setRGB(R, G, B, THREE.SRGBColorSpace));
    }
    this.peakMesh.count = nP;
    this.peakMesh.instanceMatrix.needsUpdate = true;
    if (this.peakMesh.instanceColor) this.peakMesh.instanceColor.needsUpdate = true;

    const loopCol = new THREE.Color(this.colors.loop), mergeCol = new THREE.Color(this.colors.saddle);
    for (let j = 0; j < nS; j++) {
      const id = vis.saddles[j];
      const [x, y] = this.saddleXY[id];
      m.makeScale(SADDLE_R, SADDLE_R, 1).setPosition(x, y, Z_SADDLE);
      this.saddleMesh.setMatrixAt(j, m);
      this.saddleMesh.setColorAt(j, doc.saddles[id].is_merge ? mergeCol : loopCol);
    }
    this.saddleMesh.count = nS;
    this.saddleMesh.instanceMatrix.needsUpdate = true;
    if (this.saddleMesh.instanceColor) this.saddleMesh.instanceColor.needsUpdate = true;

    const pos = this.linkGeom.attributes.position.array;
    const put = (k, [px, py], [sx, sy]) => {
      const o = k * 6;
      pos[o] = px; pos[o + 1] = py; pos[o + 2] = Z_LINK;
      pos[o + 3] = sx; pos[o + 4] = sy; pos[o + 5] = Z_LINK;
    };
    let segments = 0;
    if (arcs) {
      for (const id of vis.arcs) {
        for (const p of doc.saddles[id].peaks) put(segments++, this.peakXY[p], this.saddleXY[id]);
      }
    } else {
      for (let j = 0; j < nS; j++) {
        for (let e = 0; e < 2; e++) {
          put(segments++, this.peakXY[vis.peaks[vis.saddleEnds[j][e]]], this.saddleXY[vis.saddles[j]]);
        }
      }
    }
    this.linkGeom.setDrawRange(0, segments * 2);
    this.linkGeom.attributes.position.needsUpdate = true;

    const ghosts = arcs ? vis.ghosts : [];
    ghosts.forEach((id, g) => {
      const [x, y] = this.peakXY[id];
      m.makeScale(GHOST_R, GHOST_R, 1).setPosition(x, y, Z_SADDLE);
      this.ghostMesh.setMatrixAt(g, m);
    });
    this.ghostMesh.count = ghosts.length;
    this.ghostMesh.instanceMatrix.needsUpdate = true;

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

  /** 'simple' (discs and lines) or 'paper' (nested contours, as in the paper). */
  setStyle(style) {
    if (style !== 'simple' && style !== 'paper') throw new Error(`unknown spine style ${style}`);
    this.style = style;
    if (this.vis) this.setVisible(this.vis);
  }

  /** Scale every contour radius (1 = the largest outer contour has radius rMax). */
  setContourScale(k) {
    this.contourScale = k;
    if (this.vis && this.style === 'paper') this.setVisible(this.vis);
  }

  /** Contour radius ~ volume^k: 1 is the paper's rule, 0.5 makes the area proportional instead. */
  setRadiusExponent(k) {
    this.radiusExponent = k;
    if (this.vis && this.style === 'paper') this.setVisible(this.vis);
  }

  /**
   * Which contour sets the scale: 'saddle' gives the outermost contour at the
   * lowest shown saddle radius rMax and caps the halo below it at 2 x rMax;
   * 'halo' gives the outer halo (field minimum) radius rMax, nothing capped.
   */
  setScaleMode(mode) {
    if (mode !== 'saddle' && mode !== 'halo') throw new Error(`unknown scale mode ${mode}`);
    this.scaleMode = mode;
    if (this.vis && this.style === 'paper') this.setVisible(this.vis);
  }

  /** Minimum gap between drawn isovalues, as a fraction of the value range. */
  setLevelGap(g) {
    this.levelGap = g;
    if (this.vis && this.style === 'paper') this.setVisible(this.vis);
  }

  /** True while the paper-style drawing is on screen (it falls back to discs without per-point data). */
  get paperStyleActive() { return Boolean(this.paperActive); }

  /** Draw the arcs of equation (3) (saturated persistence) instead of straight links. */
  setShowArcs(on) {
    this.showArcs = Boolean(on);
    if (this.vis) this.setVisible(this.vis);
  }

  #arcsActive() { return this.showArcs && Array.isArray(this.vis?.arcs); }

  /** Recolour with the active value colour map (after ``setColormap``). */
  refreshColors() {
    if (this.vis) this.setVisible(this.vis);     // rebuilds the contour colours and the disc colours
    this.dirty = true;
  }

  setTheme(colors) {
    Object.assign(this.colors, colors);
    this.renderer.setClearColor(new THREE.Color(this.colors.bg), 1);
    this.linkMat.color.set(this.colors.link);
    this.ghostMat.color.set(this.colors.link);
    this.ringMat.color.set(this.colors.ring);
    this.selectMat.color.set(this.colors.select);
    this.highlightMat.color.set(this.colors.select);
    if (this.vis) this.setVisible(this.vis);     // saddle colours are per instance
    this.dirty = true;
  }

  fit() {
    const aspect = this.#aspect();
    const reach = 1 + Math.max(0.25, (this.pad ?? 0) + 0.05);   // layout box is [-1, 1], plus the shapes
    this.center.set(0, 0);
    this.halfHeight = this.homeHalfHeight = Math.max(reach, reach / aspect);
    this.dirty = true;
  }

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
    const [x, y] = node.kind === 'peak'
      ? this.peakXY[this.vis.peaks[node.index]] : this.saddleXY[this.vis.saddles[node.index]];
    const el = this.renderer.domElement;
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
    if (this.style !== 'paper') return false;
    const h = regionHistograms(this.doc, vis);
    if (!h) return false;
    const sc = spineScale(this.doc, vis, h, this.scaleMode);
    this.scaleInfo = sc;
    const rRef = this.contourScale * this.rMax;
    const g = buildContours(this.doc, vis, h, this.peakXY, this.saddleXY, {
      rs: rRef / sc.ref ** this.radiusExponent,
      maxRadius: rRef * sc.capFactor,
      exponent: this.radiusExponent, gap: this.levelGap,
      colorOf: peakColor, bg: hexToRgb(this.colors.bg), edge: hexToRgb(this.colors.ring),
      z0: Z_CONTOUR, zRange: Z_CONTOUR_RANGE,
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
    this.radius = this.radius.map((r, i) => Math.max(this.rMin, Math.min(g.extent[i], this.rMax * 2.5)));
    this.pad = Math.max(0, ...g.extent);
    return true;
  }

  // ---- scene objects ----------------------------------------------------
  #buildFixedObjects() {
    this.linkMat = new THREE.LineBasicMaterial({ color: this.colors.link, transparent: true, opacity: 0.55 });
    this.highlightMat = new THREE.LineBasicMaterial({ color: this.colors.select });
    this.ghostMat = new THREE.MeshBasicMaterial({ color: this.colors.link });
    this.ringMat = new THREE.MeshBasicMaterial({ color: this.colors.ring });
    this.selectMat = new THREE.MeshBasicMaterial({ color: this.colors.select });
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

    this.scene.add(this.linkLines, this.highlightLines, this.ghostMesh, this.saddleMesh, this.peakMesh);
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
      const [x, y] = peak ? this.peakXY[this.vis.peaks[node.index]]
                          : this.saddleXY[this.vis.saddles[node.index]];
      const r = (peak ? this.radius[node.index] : SADDLE_R) ;
      ring.position.set(x, y, Z_RING);
      ring.scale.set(r, r, 1);
      ring.visible = true;
    };
    place(this.hoverRing, this.hover);
    place(this.selectRing, this.selected);

    // Links of the selected node in the accent colour.
    const pos = this.highlightGeom?.attributes.position.array;
    let n = 0;
    if (this.selected && this.vis && pos) {
      const { vis } = this;
      const arcs = this.#arcsActive();
      const along = new Set();         // arcs mode: saddles whose two arcs are highlighted
      for (let j = 0; j < vis.saddles.length; j++) {
        const ends = vis.saddleEnds[j];
        const touches = this.selected.kind === 'peak'
          ? ends.includes(this.selected.index) : this.selected.index === j;
        if (!touches) continue;
        const sid = vis.saddles[j];
        if (arcs) {
          // follow the drawn arcs: saddle -> its own peaks -> ridge -> surviving peaks
          along.add(sid);
          for (const p of this.doc.saddles[sid].peaks) for (const r of arcPath(this.doc, vis, p)) along.add(r);
          continue;
        }
        const [sx, sy] = this.saddleXY[sid];
        for (const e of ends) {
          const [px, py] = this.peakXY[vis.peaks[e]];
          pos.set([px, py, 0, sx, sy, 0], n * 6);
          n++;
        }
      }
      for (const sid of along) {
        const [sx, sy] = this.saddleXY[sid];
        for (const p of this.doc.saddles[sid].peaks) {
          const [px, py] = this.peakXY[p];
          pos.set([px, py, 0, sx, sy, 0], n * 6);
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
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    const nx = (ev.clientX - rect.left) / rect.width * 2 - 1;
    const ny = 1 - (ev.clientY - rect.top) / rect.height * 2;
    const aspect = this.#aspect();
    return new THREE.Vector2(this.center.x + nx * this.halfHeight * aspect,
                             this.center.y + ny * this.halfHeight);
  }

  /** Nearest peak under the pointer, else nearest saddle, else null. */
  #pick(world) {
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
      drag = { x: ev.clientX, y: ev.clientY, moved: false, center: this.center.clone() };
    });
    el.addEventListener('pointermove', (ev) => {
      if (drag) {
        const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
        if (Math.hypot(dx, dy) > 4) drag.moved = true;
        if (drag.moved) {
          const px = (2 * this.halfHeight) / el.clientHeight;
          this.center.set(drag.center.x - dx * px, drag.center.y + dy * px);
          this.dirty = true;
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
    });
    el.addEventListener('pointerup', (ev) => {
      if (drag && !drag.moved) this.select(this.#pick(this.#toWorld(ev)));
      drag = null;
    });
    el.addEventListener('pointerleave', () => {
      if (this.hover && !drag) {
        this.hover = null; this.#updateOverlay(); this.handlers.onHover?.(null, null); this.dirty = true;
      }
    });
    el.addEventListener('dblclick', () => this.fit());
    el.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const before = this.#toWorld(ev);
      const factor = Math.exp(ev.deltaY * 0.0015);
      this.halfHeight = Math.min(this.homeHalfHeight * 4, Math.max(0.04, this.halfHeight * factor));
      const after = this.#toWorld(ev);                       // keep the point under the cursor fixed
      this.center.x += before.x - after.x;
      this.center.y += before.y - after.y;
      this.dirty = true;
    }, { passive: false });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    el.style.cursor = 'grab';
  }

  // ---- rendering --------------------------------------------------------
  #loop() {
    if (this.stopped) return;
    if (this.dirty) {
      const aspect = this.#aspect();
      const c = this.camera;
      c.left = -this.halfHeight * aspect; c.right = this.halfHeight * aspect;
      c.top = this.halfHeight; c.bottom = -this.halfHeight;
      c.position.set(this.center.x, this.center.y, 5);
      c.updateProjectionMatrix();
      this.renderer.render(this.scene, c);
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
