// Data view for point clouds: a 2D scatterplot of the points, drawn with Three.js.
//
// Where each point sits comes from a dimensionality reduction chosen by the user
// (``doc.data.embeddings[name]``, flat [x0, y0, x1, y1, ...] in [-1, 1]). The
// reduction is for drawing only: the extremum graph was built in the original
// space (see observations.md, 2026-11-09).
//
// Colour: a distinct colour per peak region (``regionColor``, the colour of the peak's
// dot in the spine), or the field value. Peaks selected in the spine are shown in
// full colour and drawn on top; all other points fade into the background.
// Clicking a point reports its index so the spine can select that point's peak.
// Pan with the mouse, zoom with the wheel, double-click to reset.
//
// Draw 'density' (``setDraw``): instead of one dot per point, a grid of small cells over the
// plot, each as bright as the (log of the) number of points in it and coloured as the average of
// its points' colours (so by region or value, selected peaks in full colour). Where thousands of
// dots would cover each other, this shows where the points actually pile up.

import * as THREE from 'three';
import { peakColor, regionColor } from './spineView.js';

const FADE = 0.82;          // how far unselected points fade towards the background
const EXTENT = 1.08;        // the density grid covers [-EXTENT, EXTENT]^2 (embeddings are in [-1, 1])

export class DataView {
  /**
   * @param {HTMLElement} container
   * @param {{onPick?: (pointIndex: number|null) => void}} handlers
   * @param {{renderer?: object}} options ``renderer`` replaces WebGL (tests have no GPU)
   */
  constructor(container, handlers = {}, options = {}) {
    this.container = container;
    this.handlers = handlers;
    this.bg = [1, 1, 1];
    this.renderer = options.renderer ?? new THREE.WebGLRenderer({ antialias: true });
    if (!options.renderer) this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.touchAction = 'none';

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
    this.camera.position.z = 5;
    this.center = new THREE.Vector2(0, 0);
    this.halfHeight = this.homeHalfHeight = 1.1;

    this.doc = null;
    this.vis = null;
    this.xy = null;             // Float32Array, 2 per point
    this.points = null;
    this.colorBy = 'region';
    this.draw = 'points';       // or 'density'
    this.density = null;        // the density plane (mesh with a texture)
    this.selectedPeaks = new Set();   // visible-peak indices
    this.brushable = Boolean(options.brushable);   // shift-drag draws a box brush (Scatter mode)
    this.brush = null;                // [x0, y0, x1, y1] in plot units, or null
    this.brushMask = null;            // Uint8Array: the points inside the brush
    this.dirty = true;
    this.#bindEvents();
    this.resize();
    this.#loop();
  }

  /** Names of the embeddings in ``doc`` (empty for volumes). */
  static embeddingNames(doc) {
    const e = doc.data?.embeddings;
    return e ? Object.keys(e) : [];
  }

  /** Load the points of ``doc``; returns the embedding names that can be chosen. */
  setGraph(doc) {
    this.doc = doc;
    this.vis = null;
    this.selectedPeaks = new Set();
    const names = DataView.embeddingNames(doc);
    this.#dispose();
    if (!names.length) return names;

    const n = doc.meta.num_points;
    this.n = n;
    this.geometry = new THREE.BufferGeometry();
    this.xy = new Float32Array(n * 2);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.material = new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, vertexColors: true });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.setEmbedding(doc.data.embedding_default ?? names[0]);
    this.fit();
    return names;
  }

  /**
   * Load the points of ``doc`` at given positions instead of an embedding (the Scatter mode:
   * two chosen dimensions). ``xy``: flat [x0, y0, x1, y1, ...] in [-1, 1].
   */
  setPoints(doc, xy) {
    this.doc = doc;
    this.vis = null;
    this.custom = null;                                   // a dimension of the previous file
    this.clusters = null;
    if (this.colorBy === 'cluster') this.colorBy = 'region';
    this.brush = this.brushMask = null;
    if (this.colorBy === 'custom') this.colorBy = 'region';
    this.selectedPeaks = new Set();
    this.#dispose();
    const n = doc.meta.num_points;
    this.n = n;
    this.geometry = new THREE.BufferGeometry();
    this.xy = new Float32Array(n * 2);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.material = new THREE.PointsMaterial({ size: 2.2, sizeAttenuation: false, vertexColors: true });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);
    this.setXY(xy);
    this.fit();
  }

  /** Move the points to new positions (flat [x, y, ...] in [-1, 1]). A brush is cleared. */
  setXY(xy) {
    if (!this.points || xy.length !== this.n * 2) throw new Error('positions of the wrong length');
    this.name = null;
    this.xy.set(xy);
    if (this.brush) this.clearBrush();
    this.#updatePositions();
    this.#updateDensity();
    this.dirty = true;
  }

  /** Remove the box brush (and tell the handler). */
  clearBrush() {
    if (!this.brush) return;
    this.brush = this.brushMask = null;
    this.#drawBrushBox();
    this.#updateColors();
    this.handlers.onBrush?.(null);
  }

  /** The brush box as an HTML rectangle over the canvas (it follows pan and zoom). */
  #drawBrushBox() {
    if (!this.brushBox) {
      this.brushBox = document.createElement('div');
      Object.assign(this.brushBox.style, { position: 'absolute', pointerEvents: 'none', border: '1.5px solid var(--accent)',
        background: 'color-mix(in srgb, var(--accent) 12%, transparent)', borderRadius: '2px', display: 'none' });
      this.container.style.position ||= 'relative';
      this.container.appendChild(this.brushBox);
    }
    if (!this.brush) { this.brushBox.style.display = 'none'; return; }
    const el = this.renderer.domElement, aspect = this.#aspect();
    const sx = (x) => ((x - this.center.x) / (this.halfHeight * aspect) + 1) / 2 * el.clientWidth;
    const sy = (y) => (1 - (y - this.center.y) / this.halfHeight) / 2 * el.clientHeight;
    const [x0, y0, x1, y1] = this.brush;
    Object.assign(this.brushBox.style, { display: 'block', left: `${sx(x0)}px`, top: `${sy(y1)}px`,
      width: `${sx(x1) - sx(x0)}px`, height: `${sy(y0) - sy(y1)}px` });
  }

  /** Points inside the brush box. */
  #applyBrush() {
    const [x0, y0, x1, y1] = this.brush;
    const mask = new Uint8Array(this.n);
    let count = 0;
    for (let i = 0; i < this.n; i++) {
      const x = this.xy[2 * i], y = this.xy[2 * i + 1];
      if (x >= x0 && x <= x1 && y >= y0 && y <= y1) { mask[i] = 1; count++; }
    }
    this.brushMask = mask;
    this.#drawBrushBox();
    this.#updateColors();
    this.handlers.onBrush?.({ mask, count, total: this.n });
  }

  /** Cluster labels (Int32Array, -1 = noise) for colouring by 'cluster'; null clears them. */
  setClusters(labels) {
    this.clusters = labels && labels.length === this.n ? labels : null;
    if (this.colorBy === 'cluster') this.#updateColors();
  }

  /** Values in [0, 1] that colour the points when colouring by 'custom' (a chosen dimension). */
  setCustomColors(v) {
    this.custom = v;
    if (this.colorBy === 'custom') this.#updateColors();
  }

  /** Move every point to its position in the named embedding. */
  setEmbedding(name) {
    const flat = this.doc?.data?.embeddings?.[name];
    if (!flat || !this.points) throw new Error(`no embedding called ${name}`);
    if (flat.length !== this.n * 2) throw new Error(`embedding ${name} has the wrong length`);
    this.name = name;
    this.xy.set(flat);
    this.#updatePositions();
    this.#updateDensity();
    this.dirty = true;
  }

  /** Colour by 'region' (peak colour), 'value' (field value), 'custom' (see setCustomColors) or 'cluster'. */
  setColorBy(mode) {
    this.colorBy = mode === 'value' || mode === 'custom' || mode === 'cluster' ? mode : 'region';
    this.#updateColors();
  }

  /** Draw the points as dots ('points') or as a density grid ('density'). */
  setDraw(mode) {
    this.draw = mode === 'density' ? 'density' : 'points';
    if (this.points) this.points.visible = this.draw === 'points';
    this.#updateDensity();
    this.dirty = true;
  }

  /** Recolour with the active value colour map. */
  refreshColors() { this.#updateColors(); }

  /** Show the result of ``simplify(doc, t)``: points take the colour of their (merged) peak. */
  setVisible(vis) {
    this.vis = vis;
    this.selectedPeaks = new Set([...this.selectedPeaks].filter((i) => i < vis.peaks.length));
    this.#updateColors();
  }

  /** Highlight these visible peaks (an iterable of indices into ``vis.peaks``); empty clears. */
  setSelectedPeaks(indices) {
    this.selectedPeaks = new Set(indices);
    this.#updateColors();
  }

  setTheme({ bg }) {
    this.bg = bg;
    this.renderer.setClearColor(new THREE.Color().setRGB(bg[0], bg[1], bg[2], THREE.SRGBColorSpace), 1);
    this.#updateColors();
  }

  fit() {
    this.center.set(0, 0);
    this.halfHeight = this.homeHalfHeight = Math.max(1.1, 1.1 / this.#aspect());
    this.dirty = true;
  }

  resize() {
    const w = Math.max(this.container.clientWidth, 10), h = Math.max(this.container.clientHeight, 10);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.dirty = true;
  }

  /** Index of the point nearest to client position (x, y) within ``radiusPx``, or -1. */
  pickPoint(clientX, clientY, radiusPx = 8) {
    if (!this.xy) return -1;
    const el = this.renderer.domElement, rect = el.getBoundingClientRect();
    const world = this.#toWorld(clientX - rect.left, clientY - rect.top, rect);
    const pxWorld = (2 * this.halfHeight) / Math.max(rect.height, 1);
    let best = -1, bestD = (radiusPx * pxWorld) ** 2;
    for (let i = 0; i < this.n; i++) {
      const dx = this.xy[2 * i] - world.x, dy = this.xy[2 * i + 1] - world.y;
      const d = dx * dx + dy * dy;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  }

  dispose() {
    this.stopped = true;
    this.#dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---- internals --------------------------------------------------------
  #dispose() {
    if (this.points) { this.scene.remove(this.points); this.geometry.dispose(); this.material.dispose(); }
    this.points = this.geometry = this.material = null;
    if (this.density) {
      this.scene.remove(this.density);
      this.density.geometry.dispose(); this.density.material.map.dispose(); this.density.material.dispose();
      this.density = null;
    }
  }

  #updatePositions() {
    const pos = this.geometry.attributes.position.array;
    this.#writeZ(pos);
    this.geometry.attributes.position.needsUpdate = true;
  }

  #writeZ(pos) {
    const sel = this.#selectedMask();
    for (let i = 0; i < this.n; i++) {
      pos[3 * i] = this.xy[2 * i];
      pos[3 * i + 1] = this.xy[2 * i + 1];
      pos[3 * i + 2] = sel && sel[i] ? 0.1 : 0;          // selected points are drawn on top
    }
  }

  /** Uint8Array flagging the points whose visible peak is selected, or null if nothing is selected. */
  #selectedMask() {
    if (!this.vis || !this.selectedPeaks.size) return null;
    const { region } = this.doc, owner = this.vis.owner;
    const mask = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) mask[i] = this.selectedPeaks.has(owner[region[i]]) ? 1 : 0;
    return mask;
  }

  #updateColors() {
    if (!this.points || !this.doc) return;
    const doc = this.doc, col = this.geometry.attributes.color.array;
    const [lo, hi] = doc.meta.value_range ?? [0, 1];
    const span = hi - lo || 1;
    const sel = this.#selectedMask();
    const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const bgLin = this.bg.map(lin);

    // One colour per visible peak (or per value bin), converted once.
    let table = null;
    if (this.colorBy === 'region' && this.vis) {
      table = this.vis.peaks.map((id) => regionColor(id).map(lin));
    }
    const values = doc.data.values, region = doc.region, owner = this.vis?.owner;
    for (let i = 0; i < this.n; i++) {
      let rgb;
      if (table) rgb = table[owner[region[i]]];
      else if (this.colorBy === 'custom' && this.custom && this.custom.length === this.n) rgb = peakColor(this.custom[i]).map(lin);
      else if (this.colorBy === 'cluster' && this.clusters) rgb = clusterColor(this.clusters[i]).map(lin);
      else rgb = peakColor((values[i] - lo) / span).map(lin);
      const k = (sel && !sel[i]) || (this.brushMask && !this.brushMask[i]) ? FADE : 0;
      col[3 * i] = rgb[0] * (1 - k) + bgLin[0] * k;
      col[3 * i + 1] = rgb[1] * (1 - k) + bgLin[1] * k;
      col[3 * i + 2] = rgb[2] * (1 - k) + bgLin[2] * k;
    }
    this.geometry.attributes.color.needsUpdate = true;
    this.#writeZ(this.geometry.attributes.position.array);
    this.geometry.attributes.position.needsUpdate = true;
    this.#updateDensity();
    this.dirty = true;
  }

  /**
   * Rebuild the density grid: per cell the number of points, smoothed with a small Gaussian.
   * Brightness is log(1 + n) / log(1 + max). Colour by region: the colour of the region with the
   * most points in the cell (its own colour, faded if a selection leaves it out), so overlaps
   * never invent a colour; by value: the average of the points' colours.
   */
  #updateDensity() {
    if (!this.points || this.draw !== 'density' || !this.vis) {
      if (this.density) this.density.visible = false;
      if (this.points) this.points.visible = this.draw === 'points' || !this.vis;
      return;
    }
    const n = this.n, G = Math.max(96, Math.min(256, Math.round(Math.sqrt(n) * 2)));
    const scale = G / (2 * EXTENT);
    const cellOf = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const gx = Math.floor((this.xy[2 * i] + EXTENT) * scale), gy = Math.floor((this.xy[2 * i + 1] + EXTENT) * scale);
      cellOf[i] = gx < 0 || gy < 0 || gx >= G || gy >= G ? -1 : gy * G + gx;
    }
    // separable Gaussian, sigma 1.4 cells
    const K = [-4, -3, -2, -1, 0, 1, 2, 3, 4].map((t) => Math.exp(-(t * t) / (2 * 1.4 * 1.4)));
    const ks = K.reduce((x, y) => x + y, 0);
    for (let t = 0; t < K.length; t++) K[t] /= ks;
    const R = (K.length - 1) / 2;
    const blur = (a) => {
      const tmp = new Float32Array(a.length);
      for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
        let v = 0;
        for (let t = -R; t <= R; t++) { const xx = x + t; if (xx >= 0 && xx < G) v += K[t + R] * a[y * G + xx]; }
        tmp[y * G + x] = v;
      }
      for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
        let v = 0;
        for (let t = -R; t <= R; t++) { const yy = y + t; if (yy >= 0 && yy < G) v += K[t + R] * tmp[yy * G + x]; }
        a[y * G + x] = v;
      }
    };
    const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
    const bg = this.bg.map(lin);
    const cnt = new Float32Array(G * G);
    for (let i = 0; i < n; i++) if (cellOf[i] >= 0) cnt[cellOf[i]]++;
    blur(cnt);
    const rgb = new Float32Array(G * G * 3);           // the colour of each cell (linear)
    const owner = this.vis.owner, region = this.doc.region;
    const P = this.vis.peaks.length;
    const sel = this.selectedPeaks;
    if (this.colorBy === 'region' && P <= 64) {
      // the region with the most (smoothed) points wins the cell
      const best = new Float32Array(G * G), per = new Float32Array(G * G);
      for (let k = 0; k < P; k++) {
        per.fill(0);
        let any = false;
        for (let i = 0; i < n; i++) if (cellOf[i] >= 0 && owner[region[i]] === k) { per[cellOf[i]]++; any = true; }
        if (!any) continue;
        blur(per);
        let c = regionColor(this.vis.peaks[k]).map(lin);
        if (sel.size && !sel.has(k)) c = c.map((v, q) => v * (1 - FADE) + bg[q] * FADE);
        for (let m = 0; m < G * G; m++) {
          if (per[m] > best[m]) { best[m] = per[m]; rgb[3 * m] = c[0]; rgb[3 * m + 1] = c[1]; rgb[3 * m + 2] = c[2]; }
        }
      }
    } else {
      // the average of the points' colours (already faded for a selection)
      const col = this.geometry.attributes.color.array;
      const sum = [new Float32Array(G * G), new Float32Array(G * G), new Float32Array(G * G)];
      for (let i = 0; i < n; i++) {
        const m = cellOf[i];
        if (m < 0) continue;
        sum[0][m] += col[3 * i]; sum[1][m] += col[3 * i + 1]; sum[2][m] += col[3 * i + 2];
      }
      sum.forEach(blur);
      for (let m = 0; m < G * G; m++) if (cnt[m] > 1e-6) for (let q = 0; q < 3; q++) rgb[3 * m + q] = sum[q][m] / cnt[m];
    }
    let cmax = 0;
    for (let m = 0; m < G * G; m++) if (cnt[m] > cmax) cmax = cnt[m];
    const lmax = Math.log1p(cmax) || 1;
    const data = new Uint8Array(G * G * 4);
    for (let m = 0; m < G * G; m++) {
      const a = cnt[m] > 0.03 ? Math.min(1, 0.06 + Math.log1p(cnt[m]) / lmax) : 0;   // sparse cells stay faint
      for (let q = 0; q < 3; q++) data[4 * m + q] = Math.round(255 * toSrgb(Math.min(1, bg[q] * (1 - a) + rgb[3 * m + q] * a)));
      data[4 * m + 3] = 255;
    }
    if (!this.density) {
      const tex = new THREE.DataTexture(data, G, G, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
      this.density = new THREE.Mesh(new THREE.PlaneGeometry(2 * EXTENT, 2 * EXTENT),
                                    new THREE.MeshBasicMaterial({ map: tex }));
      this.density.position.z = -0.5;
      this.scene.add(this.density);
    } else if (this.density.material.map.image.width !== G) {
      this.density.material.map.dispose();
      const tex = new THREE.DataTexture(data, G, G, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
      this.density.material.map = tex;
    } else {
      this.density.material.map.image.data.set(data);
    }
    this.density.material.map.needsUpdate = true;
    this.density.visible = true;
    this.points.visible = false;
  }

  #aspect() {
    const el = this.renderer.domElement;
    return Math.max(el.clientWidth, 1) / Math.max(el.clientHeight, 1);
  }

  #toWorld(px, py, rect) {
    const nx = (px / rect.width) * 2 - 1, ny = 1 - (py / rect.height) * 2;
    return new THREE.Vector2(this.center.x + nx * this.halfHeight * this.#aspect(),
                             this.center.y + ny * this.halfHeight);
  }

  #bindEvents() {
    const el = this.renderer.domElement;
    let drag = null;
    el.addEventListener('pointerdown', (ev) => {
      el.setPointerCapture?.(ev.pointerId);
      const rect = el.getBoundingClientRect();
      drag = { x: ev.clientX, y: ev.clientY, moved: false, center: this.center.clone(),
               brush: this.brushable && ev.shiftKey, start: this.#toWorld(ev.clientX - rect.left, ev.clientY - rect.top, rect) };
    });
    el.addEventListener('pointermove', (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.moved && drag.brush) {                       // shift-drag: draw the box brush
        const rect = el.getBoundingClientRect();
        const w = this.#toWorld(ev.clientX - rect.left, ev.clientY - rect.top, rect);
        this.brush = [Math.min(drag.start.x, w.x), Math.min(drag.start.y, w.y), Math.max(drag.start.x, w.x), Math.max(drag.start.y, w.y)];
        this.#drawBrushBox();
        return;
      }
      if (drag.moved) {
        const px = (2 * this.halfHeight) / el.clientHeight;
        this.center.set(drag.center.x - dx * px, drag.center.y + dy * px);
        this.dirty = true;
      }
    });
    el.addEventListener('pointerup', (ev) => {
      if (drag && drag.brush && drag.moved && this.brush) { this.#applyBrush(); drag = null; return; }
      if (drag && !drag.moved) {
        const i = this.pickPoint(ev.clientX, ev.clientY);
        this.handlers.onPick?.(i < 0 ? null : i);
      }
      drag = null;
    });
    el.addEventListener('dblclick', () => this.fit());
    el.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const rect = el.getBoundingClientRect();
      const before = this.#toWorld(ev.clientX - rect.left, ev.clientY - rect.top, rect);
      this.halfHeight = Math.min(this.homeHalfHeight * 4, Math.max(0.02, this.halfHeight * Math.exp(ev.deltaY * 0.0015)));
      const after = this.#toWorld(ev.clientX - rect.left, ev.clientY - rect.top, rect);
      this.center.x += before.x - after.x;
      this.center.y += before.y - after.y;
      this.dirty = true;
    }, { passive: false });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
  }

  #loop() {
    if (this.stopped) return;
    if (this.dirty) {
      const aspect = this.#aspect(), c = this.camera;
      c.left = -this.halfHeight * aspect; c.right = this.halfHeight * aspect;
      c.top = this.halfHeight; c.bottom = -this.halfHeight;
      c.position.set(this.center.x, this.center.y, 5);
      c.updateProjectionMatrix();
      this.renderer.render(this.scene, c);
      if (this.brush) this.#drawBrushBox();
      this.dirty = false;
    }
    requestAnimationFrame(() => this.#loop());
  }
}

/** Colour of cluster ``c`` ([r, g, b] sRGB); noise (-1) is grey. Distinct from the peak colours' order. */
export function clusterColor(c) {
  if (c < 0) return [0.55, 0.56, 0.6];
  return CLUSTER_COLORS[c % CLUSTER_COLORS.length];
}
// 10 well-separated colours (Tableau 10), then they repeat
const CLUSTER_COLORS = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#b07aa1', '#edc948', '#76b7b2', '#ff9da7', '#9c755f', '#bab0ac']
  .map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16) / 255));
