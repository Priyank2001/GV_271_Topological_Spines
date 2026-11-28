// Parallel-coordinates view: every point of the field is a polyline across one
// vertical axis per original coordinate, plus the field value and the point's peak
// (the data model is in parallel.js).
//
// Drawing: the polylines are Three.js LineSegments (one vertex per point and axis,
// so 100k points x 12 axes stay interactive); axes, labels, brushes and the hovered
// line are drawn on a 2D canvas laid over them. The lines in focus (passing every
// brush, and in the region of the peak hovered or selected in the spine) are drawn
// in colour; the others stay as a faint grey context.
//
// Interaction:
//   drag along an axis      brush a range (several axes: a point must pass all)
//   click an axis           clear its brush
//   click an axis name      flip the axis; drag the name to move the axis
//   click a line            report the point (the page selects its peak)
//   double-click            clear every brush
// Any axis can be hidden (setAxisShown, the page's "Axes" menu); at least two stay.

import * as THREE from 'three';
import { peakColor, regionColor } from './spineView.js';
import { baseColumns, peakColumn, shown, brushMask, andMask, lineIndex, pickLine, peakCounts,
         defaultOpacity } from './parallel.js';

const M = { left: 46, right: 46, top: 48, bottom: 30 };   // plot margins in CSS pixels
const AXIS_HIT = 12;                                        // pixels either side of an axis that brush
const CONTEXT = 0.22;                                       // context opacity, relative to the lines in focus

const lin = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fmt = (x) => (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(2)
  : Number(x.toPrecision(3)).toString());

export class ParallelView {
  /**
   * @param {HTMLElement} container
   * @param {{onPick?: (i: number|null) => void, onBrush?: (summary: object|null) => void,
   *          onAxes?: (axes: {key: string, label: string, shown: boolean}[]) => void}} handlers
   * @param {{renderer?: object, overlay?: object}} options stand-ins for WebGL and the 2D canvas (tests)
   */
  constructor(container, handlers = {}, options = {}) {
    this.container = container;
    this.handlers = handlers;
    this.renderer = options.renderer ?? new THREE.WebGLRenderer({ antialias: true });
    if (!options.renderer) this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.overlay = options.overlay ?? document.createElement('canvas');
    this.tip = options.overlay ? null : document.createElement('div');
    for (const el of [this.renderer.domElement, this.overlay]) {
      Object.assign(el.style, { position: 'absolute', left: '0', top: '0', width: '100%', height: '100%', display: 'block' });
      container.appendChild(el);
    }
    this.overlay.style.touchAction = 'none';
    if (this.tip) { this.tip.className = 'pc-tip'; this.tip.hidden = true; container.appendChild(this.tip); }

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(0, 1, 1, 0, -10, 10);
    this.camera.position.z = 5;
    this.focusMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthTest: false, depthWrite: false });
    this.ctxMat = new THREE.LineBasicMaterial({ color: 0x9aa3b0, transparent: true, depthTest: false, depthWrite: false });
    this.theme = { bg: [1, 1, 1], text: '#1f2430', muted: '#6b7280', accent: '#2563eb', context: '#9aa3b0' };

    this.doc = null; this.vis = null; this.coords = null;
    this.base = []; this.peakCol = null; this.order = []; this.note = '';
    this.brushes = new Map();          // column key -> [lo, hi] in normalised value
    this.selectedPeaks = new Set();
    this.colorBy = 'region';
    this.hiddenAxes = new Set();       // keys of the axes the user turned off
    this.opacity = 0.3;
    this.active = false;
    this.hovered = -1;
    this.focus = null;                 // Uint8Array or null (everything in focus)
    this.need = new Set(['columns']);
    this.size = { w: 10, h: 10 };
    this.#bindEvents();
  }

  // ---- inputs from the page ------------------------------------------------
  /** Only an active view rebuilds and draws; changes made meanwhile are applied on activation. */
  setActive(on) {
    this.active = Boolean(on);
    if (this.active) { this.resize(); this.#schedule(); }
  }

  /** A new document: columns, brushes and axis order start afresh. Coordinates come via setCoords. */
  setGraph(doc) {
    if (doc.meta.name !== this.doc?.meta.name) this.hiddenAxes.clear();   // another method keeps the choice
    this.doc = doc; this.vis = null; this.coords = null;
    this.brushes.clear(); this.selectedPeaks = new Set(); this.order = []; this.hovered = -1;
    this.opacity = defaultOpacity(doc.meta.num_points);
    this.#want('columns');
    return this.opacity;
  }

  /** The parsed ``<name>.coords.json`` of a point cloud (null: none). */
  setCoords(coords) { this.coords = coords; this.order = []; this.brushes.clear(); this.#want('columns'); }

  /** The visible graph: owners (and so the peak axis and region colours) change with the sliders. */
  setVisible(vis) {
    this.vis = vis;
    this.selectedPeaks = new Set([...this.selectedPeaks].filter((i) => i < vis.peaks.length));
    this.#want('peak');
  }

  /** Highlight the regions of these visible peaks (indices into ``vis.peaks``); empty clears. */
  setSelectedPeaks(indices) {
    const next = new Set(indices);
    if (next.size === this.selectedPeaks.size && [...next].every((i) => this.selectedPeaks.has(i))) return;
    this.selectedPeaks = next;
    this.#want('focus');
  }

  setColorBy(mode) { this.colorBy = mode === 'value' ? 'value' : 'region'; this.#want('colors'); }
  refreshColors() { this.#want('colors'); }
  setOpacity(a) { this.opacity = a; this.#applyOpacity(); this.dirty = true; this.#schedule(); }

  /** Every axis the data offers, in its natural order, and whether it is drawn. */
  axes() {
    return [...this.base, this.peakCol ?? { key: 'peak', label: 'peak' }]
      .map((c) => ({ key: c.key, label: c.label, shown: !this.hiddenAxes.has(c.key) }));
  }

  /**
   * Show or hide one axis. A hidden axis loses its brush; shown again, it returns
   * next to its natural neighbours. Refused (returns false) if fewer than two would remain.
   */
  setAxisShown(key, on) {
    const known = this.axes();
    if (!known.some((a) => a.key === key)) return false;
    if (!on && known.filter((a) => a.shown && a.key !== key).length < 2) return false;
    if (on) this.hiddenAxes.delete(key); else this.hiddenAxes.add(key);
    this.#syncOrder();
    this.#want('positions');
    this.handlers.onAxes?.(this.axes());
    return true;
  }

  /** Show every axis again. */
  showAllAxes() {
    this.hiddenAxes.clear();
    this.#syncOrder();
    this.#want('positions');
    this.handlers.onAxes?.(this.axes());
  }

  clearBrushes() { this.brushes.clear(); this.#want('focus'); }

  setTheme({ bg, text, muted, accent, context }) {
    const bgCss = `rgb(${bg.map((c) => Math.round(c * 255)).join(',')})`;
    Object.assign(this.theme, { bg, bgCss, text, muted, accent, context });
    this.renderer.setClearColor(new THREE.Color().setRGB(bg[0], bg[1], bg[2], THREE.SRGBColorSpace), 1);
    this.ctxMat.color.setStyle(context);
    this.#want('overlay');
  }

  resize() {
    const w = Math.max(this.container.clientWidth, 10), h = Math.max(this.container.clientHeight, 10);
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    this.overlay.width = Math.round(w * dpr); this.overlay.height = Math.round(h * dpr);
    this.dpr = dpr;
    this.dirty = true;
    this.#want('overlay');
  }

  /** Columns in drawing order. */
  columns() {
    return this.order.map((k) => (k === 'peak' ? this.peakCol : this.base.find((c) => c.key === k))).filter(Boolean);
  }

  /** Apply pending changes now (normally done once per animation frame). */
  flush() {
    this.scheduled = false;
    if (!this.active || !this.doc || !this.vis || !this.doc.region) return;
    const need = this.need;
    if (!need.size && !this.dirty) return;
    // Anything but 'overlay' changes the lines, so WebGL must draw again.
    if ([...need].some((x) => x !== 'overlay')) this.dirty = true;
    if (need.has('columns')) {
      try {
        ({ columns: this.base, note: this.note } = baseColumns(this.doc, this.coords));
      } catch (err) {
        console.error(err);
        ({ columns: this.base } = baseColumns(this.doc, null));
        this.note = `The coordinates could not be read (${err.message}).`;
      }
      this.base.forEach((c) => { c.flipped = false; });
      this.order = [];
      this.#syncOrder();
      need.add('peak');
      this.handlers.onAxes?.(this.axes());
    }
    if (need.has('peak')) {
      const flipped = this.peakCol?.flipped ?? false;
      this.peakCol = peakColumn(this.doc, this.vis);
      this.peakCol.flipped = flipped;
      need.add('positions'); need.add('colors'); need.add('focus');
    }
    if (need.has('positions')) { this.#writePositions(); need.add('colors'); need.add('focus'); }
    if (need.has('colors')) this.#writeColors();
    if (need.has('focus')) this.#writeFocus();
    need.clear();
    this.#drawOverlay();
    this.#render();
  }

  /** Pixel x of the k-th axis (in drawing order) and pixel y of point i on a column. */
  axisX(k, A = this.order.length) {
    const { w } = this.size;
    return A < 2 ? w / 2 : M.left + (k * (w - M.left - M.right)) / (A - 1);
  }
  yPx(col, i) { const { h } = this.size; return h - M.bottom - shown(col, col.v[i]) * (h - M.top - M.bottom); }

  dispose() {
    this.stopped = true;
    this.#disposeBuffers();
    this.resizeObserver?.disconnect();
    this.renderer.dispose();
    this.renderer.domElement.remove(); this.overlay.remove(); this.tip?.remove();
  }

  // ---- building the buffers --------------------------------------------------
  #want(what) { this.need.add(what); this.#schedule(); }

  #schedule() {
    if (this.scheduled || !this.active) return;
    this.scheduled = true;
    requestAnimationFrame(() => this.flush());
  }

  /**
   * The drawn axes: the user's order of the shown ones; an axis shown again goes
   * right after the last drawn axis that precedes it in the natural order
   * (coordinates, value, peak). Brushes on hidden axes are dropped.
   */
  #syncOrder() {
    const all = [...this.base.map((c) => c.key), 'peak'];
    const want = all.filter((k) => !this.hiddenAxes.has(k));
    const order = this.order.filter((k) => want.includes(k));
    for (const k of want) {
      if (order.includes(k)) continue;
      const rank = all.indexOf(k);
      let at = 0;
      order.forEach((o, j) => { if (all.indexOf(o) < rank) at = j + 1; });
      order.splice(at, 0, k);
    }
    this.order = order;
    for (const k of [...this.brushes.keys()]) if (!order.includes(k)) this.brushes.delete(k);
  }

  #disposeBuffers() {
    if (this.focusLines) { this.scene.remove(this.focusLines, this.ctxLines); this.focusGeom.dispose(); this.ctxGeom.dispose(); }
    this.focusLines = this.ctxLines = this.focusGeom = this.ctxGeom = null;
    this.bufN = this.bufA = 0;
  }

  #writePositions() {
    const cols = this.columns(), A = cols.length, n = this.doc.meta.num_points;
    if (this.bufN !== n || this.bufA !== A) {
      this.#disposeBuffers();
      this.posAttr = new THREE.BufferAttribute(new Float32Array(n * A * 3), 3);
      this.colAttr = new THREE.BufferAttribute(new Float32Array(n * A * 3), 3);
      this.focusGeom = new THREE.BufferGeometry();
      this.ctxGeom = new THREE.BufferGeometry();
      for (const g of [this.focusGeom, this.ctxGeom]) { g.setAttribute('position', this.posAttr); g.setAttribute('color', this.colAttr); }
      this.ctxLines = new THREE.LineSegments(this.ctxGeom, this.ctxMat);
      this.focusLines = new THREE.LineSegments(this.focusGeom, this.focusMat);
      this.ctxLines.renderOrder = 0; this.focusLines.renderOrder = 1;
      this.ctxLines.frustumCulled = this.focusLines.frustumCulled = false;
      this.scene.add(this.ctxLines, this.focusLines);
      this.bufN = n; this.bufA = A;
    }
    const pos = this.posAttr.array;
    for (let k = 0; k < A; k++) {
      const c = cols[k], v = c.v, flip = c.flipped;
      for (let i = 0, j = k * 3; i < n; i++, j += A * 3) {
        pos[j] = k; pos[j + 1] = flip ? 1 - v[i] : v[i]; pos[j + 2] = 0;
      }
    }
    this.posAttr.needsUpdate = true;
  }

  #writeColors() {
    if (!this.colAttr) return;
    const doc = this.doc, vis = this.vis, n = doc.meta.num_points, A = this.bufA, col = this.colAttr.array;
    const valueCol = this.base.find((c) => c.key === 'value');
    const byValue = this.colorBy === 'value' && valueCol;
    const table = byValue ? null : vis.peaks.map((id) => regionColor(id).map(lin));
    const ramp = byValue ? Array.from({ length: 256 }, (_, k) => peakColor(k / 255).map(lin)) : null;
    const owner = vis.owner, region = doc.region;
    for (let i = 0; i < n; i++) {
      const rgb = byValue ? ramp[Math.max(0, Math.min(255, Math.round(valueCol.v[i] * 255)))] : table[owner[region[i]]];
      for (let k = 0, j = i * A * 3; k < A; k++, j += 3) { col[j] = rgb[0]; col[j + 1] = rgb[1]; col[j + 2] = rgb[2]; }
    }
    this.colAttr.needsUpdate = true;
  }

  /** Split the lines into focus (in colour) and context (faint grey), and report the brush. */
  #writeFocus() {
    if (!this.focusGeom) return;
    const doc = this.doc, vis = this.vis, n = doc.meta.num_points, A = this.bufA;
    const cols = this.columns();
    let sel = null;
    if (this.selectedPeaks.size) {
      sel = new Uint8Array(n);
      const owner = vis.owner, region = doc.region;
      for (let i = 0; i < n; i++) sel[i] = this.selectedPeaks.has(owner[region[i]]) ? 1 : 0;
    }
    const brushed = brushMask(cols, this.brushes, n);
    this.focus = andMask(brushed, sel);
    this.focusGeom.setIndex(new THREE.BufferAttribute(lineIndex(this.focus, n, A, 1), 1));
    this.ctxGeom.setIndex(new THREE.BufferAttribute(this.focus ? lineIndex(this.focus, n, A, 0) : new Uint32Array(0), 1));
    this.#applyOpacity();
    if (!brushed) { this.handlers.onBrush?.(null); return; }
    let count = 0;
    for (let i = 0; i < n; i++) count += brushed[i];
    this.handlers.onBrush?.({ count, total: n, peaks: peakCounts(doc, vis, brushed) });
  }

  #applyOpacity() {
    this.focusMat.opacity = this.opacity;
    this.ctxMat.opacity = Math.max(0.008, this.opacity * CONTEXT);
  }

  // ---- drawing -----------------------------------------------------------------
  #render() {
    if (!this.dirty || !this.active) return;
    const { w, h } = this.size, A = Math.max(this.bufA || 0, 1);
    const sx = A < 2 ? (w - M.left - M.right) : (w - M.left - M.right) / (A - 1);
    const x0 = A < 2 ? w / 2 : M.left;
    const plotH = h - M.top - M.bottom, yBottom = h - M.bottom;
    const c = this.camera;
    c.left = -x0 / sx; c.right = (w - x0) / sx;
    c.top = yBottom / plotH; c.bottom = -(h - yBottom) / plotH;
    c.updateProjectionMatrix();
    this.renderer.render(this.scene, c);
    this.dirty = false;
  }

  #drawOverlay() {
    const ctx = this.overlay.getContext?.('2d');
    if (!ctx) return;
    const { w, h } = this.size, t = this.theme, dpr = this.dpr ?? 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cols = this.columns(), A = cols.length;
    const top = M.top, bottom = h - M.bottom;
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'center';

    // the hovered point's polyline, under the axes
    if (this.hovered >= 0 && this.hovered < this.doc.meta.num_points) {
      ctx.strokeStyle = t.accent; ctx.lineWidth = 2.5; ctx.beginPath();
      cols.forEach((c, k) => { const x = this.axisX(k, A), y = this.yPx(c, this.hovered); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.fillStyle = t.accent;
      cols.forEach((c, k) => { ctx.beginPath(); ctx.arc(this.axisX(k, A), this.yPx(c, this.hovered), 3, 0, 2 * Math.PI); ctx.fill(); });
    }

    cols.forEach((c, k) => {
      const dragging = this.drag?.type === 'axis' && this.drag.moved && this.drag.k === k;
      const x = dragging ? this.drag.x : this.axisX(k, A);
      ctx.globalAlpha = dragging ? 0.6 : 1;
      ctx.strokeStyle = t.text; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
      // name (click: flip, drag: move) and the range at both ends
      ctx.fillStyle = t.text; ctx.font = '600 12px system-ui, sans-serif';
      this.#text(ctx, c.label + (c.flipped ? ' ↕' : ''), x, top - 26);
      ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = t.muted;
      if (c.kind === 'peak') {
        this.#drawPeakBands(ctx, c, x, top, bottom);
      } else {
        const [a, b] = c.flipped ? [c.hi, c.lo] : [c.lo, c.hi];
        this.#text(ctx, fmt(b), x, top - 9);
        this.#text(ctx, fmt(a), x, bottom + 15);
      }
      // brush
      const br = this.brushes.get(c.key);
      if (br) {
        const ya = this.#yOfValue(c, br[0]), yb = this.#yOfValue(c, br[1]);
        const y0 = Math.min(ya, yb), y1 = Math.max(ya, yb);
        ctx.fillStyle = t.accent; ctx.globalAlpha = 0.2; ctx.fillRect(x - 8, y0, 16, y1 - y0);
        ctx.globalAlpha = 1; ctx.strokeStyle = t.accent; ctx.lineWidth = 1.5; ctx.strokeRect(x - 8, y0, 16, y1 - y0);
        if (c.kind !== 'peak') {
          ctx.fillStyle = t.accent; ctx.textAlign = 'left';
          const v = (u) => fmt(c.lo + u * (c.hi - c.lo));
          this.#text(ctx, v(c.flipped ? br[0] : br[1]), x + 11, y0 + 4);
          this.#text(ctx, v(c.flipped ? br[1] : br[0]), x + 11, y1 + 4);
          ctx.textAlign = 'center';
        }
      }
      ctx.globalAlpha = 1;
    });
    if (this.drag?.type === 'axis' && this.drag.moved) {          // where the axis will land
      const k = this.#slotAt(this.drag.x);
      ctx.strokeStyle = t.accent; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(this.axisX(k, A), top - 18); ctx.lineTo(this.axisX(k, A), bottom); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  #drawPeakBands(ctx, c, x, top, bottom) {
    const H = bottom - top, bands = c.bands, t = this.theme;
    const bandPx = H / Math.max(bands.length, 1);
    const every = Math.max(1, Math.ceil(13 / bandPx));      // label every band that has room
    ctx.textAlign = 'right';
    bands.forEach((b, r) => {
      const y0 = this.#yOfValue(c, b.top), y1 = this.#yOfValue(c, b.bottom);
      ctx.strokeStyle = t.muted; ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(x - 4, y1); ctx.lineTo(x + 4, y1); ctx.stroke();
      ctx.globalAlpha = 1;
      if (r % every === 0) {
        const sel = this.selectedPeaks.has(b.index);
        ctx.fillStyle = sel ? t.accent : t.muted;
        ctx.font = `${sel ? 600 : 400} 11px system-ui, sans-serif`;
        this.#text(ctx, String(b.id), x - 7, (y0 + y1) / 2 + 4);
      }
    });
    ctx.textAlign = 'center'; ctx.font = '11px system-ui, sans-serif'; ctx.fillStyle = t.muted;
    this.#text(ctx, c.flipped ? 'low' : 'high', x, top - 9);
    this.#text(ctx, c.flipped ? 'high' : 'low', x, bottom + 15);
  }

  /** Text with a halo in the background colour, so it stays readable over the lines. */
  #text(ctx, str, x, y) {
    ctx.save();
    ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.strokeStyle = this.theme.bgCss ?? '#fff';
    ctx.strokeText(str, x, y);
    ctx.restore();
    ctx.fillText(str, x, y);
  }

  #yOfValue(c, v) { const { h } = this.size; return h - M.bottom - shown(c, v) * (h - M.top - M.bottom); }

  /** Normalised value of a column under pixel y (clamped to the axis). */
  #valueAt(c, y) {
    const { h } = this.size;
    const s = Math.min(1, Math.max(0, (h - M.bottom - y) / (h - M.top - M.bottom)));
    return c.flipped ? 1 - s : s;
  }

  /** Axis slot nearest to pixel x. */
  #slotAt(x) {
    const A = this.order.length;
    let best = 0;
    for (let k = 1; k < A; k++) if (Math.abs(this.axisX(k) - x) < Math.abs(this.axisX(best) - x)) best = k;
    return best;
  }

  /** What is under the pointer: {type: 'label'|'axis'|'plot', k}. */
  #hit(x, y) {
    const A = this.order.length;
    if (!A) return { type: 'plot', k: -1 };
    const k = this.#slotAt(x), dx = Math.abs(this.axisX(k) - x);
    const spacing = A > 1 ? this.axisX(1) - this.axisX(0) : this.size.w;
    if (y < M.top - 16 && dx < Math.min(40, spacing / 2)) return { type: 'label', k };
    if (dx <= AXIS_HIT && y >= M.top - 16 && y <= this.size.h - M.bottom + 6) return { type: 'axis', k };
    return { type: 'plot', k };
  }

  #pick(x, y) {
    const cols = this.columns(), n = this.doc?.meta.num_points ?? 0;
    if (!cols.length || !this.vis) return -1;
    const ax = cols.map((_, k) => this.axisX(k, cols.length));
    const yOf = (c, i) => this.yPx(c, i);
    let i = this.focus ? pickLine(x, y, cols, ax, yOf, n, this.focus) : -1;
    if (i < 0) i = pickLine(x, y, cols, ax, yOf, n, null);
    return i;
  }

  #showTip(i, x, y) {
    if (!this.tip) return;
    if (i < 0) { this.tip.hidden = true; return; }
    const p = this.vis.owner[this.doc.region[i]];
    const val = this.doc.data?.values?.[i];
    this.tip.textContent = `point ${i.toLocaleString('en-US')} · peak ${this.vis.peaks[p]}` +
      (val !== undefined ? ` · f ${fmt(val)}` : '');
    this.tip.hidden = false;
    const right = x > this.size.w - 220;
    this.tip.style.left = `${right ? x - 12 : x + 12}px`;
    this.tip.style.top = `${y}px`;
    this.tip.style.transform = right ? 'translate(-100%, -50%)' : 'translate(0, -50%)';
  }

  #bindEvents() {
    const el = this.overlay;
    const local = (ev) => { const r = el.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
    let hoverPending = null;

    el.addEventListener('pointerdown', (ev) => {
      if (!this.order.length) return;
      el.setPointerCapture?.(ev.pointerId);
      const [x, y] = local(ev), hit = this.#hit(x, y);
      this.drag = { type: hit.type, k: hit.k, x, y, x0: x, y0: y, moved: false };
      this.hovered = -1; this.#showTip(-1);
    });
    el.addEventListener('pointermove', (ev) => {
      const [x, y] = local(ev);
      const d = this.drag;
      if (!d) {
        const hit = this.#hit(x, y);
        el.style.cursor = hit.type === 'label' ? 'grab' : hit.type === 'axis' ? 'ns-resize' : 'default';
        if (!hoverPending) {
          hoverPending = requestAnimationFrame(() => {
            hoverPending = null;
            if (this.drag || !this.active) return;
            const i = hit.type === 'plot' ? this.#pick(x, y) : -1;
            if (i !== this.hovered) { this.hovered = i; this.#drawOverlay(); }
            this.#showTip(i, x, y);
          });
        }
        return;
      }
      if (Math.hypot(x - d.x0, y - d.y0) > 3) d.moved = true;
      if (!d.moved) return;
      if (d.type === 'axis') {
        const c = this.columns()[d.k];
        const a = this.#valueAt(c, d.y0), b = this.#valueAt(c, y);
        this.brushes.set(c.key, [Math.min(a, b), Math.max(a, b)]);
        this.#want('focus');
      } else if (d.type === 'label') {
        d.x = x; el.style.cursor = 'grabbing';
        this.#drawOverlay();
      }
    });
    el.addEventListener('pointerup', (ev) => {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      const [x, y] = local(ev);
      const cols = this.columns();
      if (d.type === 'axis' && !d.moved) {                      // click on an axis: clear its brush
        if (this.brushes.delete(cols[d.k].key)) this.#want('focus');
      } else if (d.type === 'label') {
        if (!d.moved) {                                          // click on a name: flip the axis
          const c = cols[d.k];
          c.flipped = !c.flipped;
          this.#want('positions');
        } else {                                                 // drop: move the axis there
          const to = this.#slotAt(d.x), [key] = this.order.splice(d.k, 1);
          this.order.splice(to, 0, key);
          this.#want('positions');
        }
        el.style.cursor = 'grab';
      } else if (d.type === 'plot' && !d.moved) {
        const i = this.#pick(x, y);
        this.handlers.onPick?.(i < 0 ? null : i);
      }
      this.#want('overlay');
    });
    el.addEventListener('pointerleave', () => {
      if (this.drag) return;
      this.hovered = -1; this.#showTip(-1); this.#drawOverlay();
    });
    el.addEventListener('dblclick', (ev) => {
      const [x, y] = local(ev);
      if (this.#hit(x, y).type === 'label') return;
      if (this.brushes.size) { this.brushes.clear(); this.#want('focus'); }
    });
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => { if (this.active) { this.resize(); this.#schedule(); } });
      this.resizeObserver.observe(this.container);
    }
  }
}
