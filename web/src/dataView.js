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

import * as THREE from 'three';
import { peakColor, regionColor } from './spineView.js';

const FADE = 0.82;          // how far unselected points fade towards the background

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
    this.selectedPeaks = new Set();   // visible-peak indices
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

  /** Move every point to its position in the named embedding. */
  setEmbedding(name) {
    const flat = this.doc?.data?.embeddings?.[name];
    if (!flat || !this.points) throw new Error(`no embedding called ${name}`);
    if (flat.length !== this.n * 2) throw new Error(`embedding ${name} has the wrong length`);
    this.name = name;
    this.xy.set(flat);
    this.#updatePositions();
    this.dirty = true;
  }

  /** Colour by 'region' (peak colour) or 'value' (field value). */
  setColorBy(mode) {
    this.colorBy = mode === 'value' ? 'value' : 'region';
    this.#updateColors();
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
      else rgb = peakColor((values[i] - lo) / span).map(lin);
      const k = sel && !sel[i] ? FADE : 0;
      col[3 * i] = rgb[0] * (1 - k) + bgLin[0] * k;
      col[3 * i + 1] = rgb[1] * (1 - k) + bgLin[1] * k;
      col[3 * i + 2] = rgb[2] * (1 - k) + bgLin[2] * k;
    }
    this.geometry.attributes.color.needsUpdate = true;
    this.#writeZ(this.geometry.attributes.position.array);
    this.geometry.attributes.position.needsUpdate = true;
    this.dirty = true;
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
      drag = { x: ev.clientX, y: ev.clientY, moved: false, center: this.center.clone() };
    });
    el.addEventListener('pointermove', (ev) => {
      if (!drag) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.moved) {
        const px = (2 * this.halfHeight) / el.clientHeight;
        this.center.set(drag.center.x - dx * px, drag.center.y + dy * px);
        this.dirty = true;
      }
    });
    el.addEventListener('pointerup', (ev) => {
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
      this.dirty = false;
    }
    requestAnimationFrame(() => this.#loop());
  }
}
