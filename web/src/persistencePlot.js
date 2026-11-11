// Persistence plot: how many peaks are left at each threshold.
//
// A real peak stands high above the noise, so the curve has a flat stretch
// (a stable peak count) after the noise peaks have dropped out. Choosing a
// threshold on that plateau is the usual guide. Click or drag on the plot to
// set the threshold.
//
// The paper's second curve (red, Fig. 8b) is drawn on the same x axis: how many
// links (saddles) have persistence <= p_hi, i.e. survive the variation threshold.
// Click sets the noise threshold p_lo; shift+click sets p_hi.
//
// x axis: threshold on a square-root scale (fine control near 0, the same
//         mapping as the slider); y axis: counts on a log scale (peaks on the
//         left, links on the right, each from 1 to its own maximum).

import { saddleCount } from './simplify.js';

/** Slider position s in [0, 1] -> persistence threshold. */
export function sliderToThreshold(s, tmax) { return tmax * s * s; }

/** Persistence threshold -> slider position in [0, 1]. */
export function thresholdToSlider(t, tmax) {
  return tmax > 0 ? Math.sqrt(Math.min(Math.max(t, 0), tmax) / tmax) : 0;
}

/**
 * Number of peaks with persistence >= t, counting the global maximum.
 * @param {number[]} curve persistence of every peak but the global maximum, descending
 */
export function peakCount(curve, t) {
  let lo = 0, hi = curve.length;           // first index whose value is < t
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (curve[mid] >= t) lo = mid + 1; else hi = mid;
  }
  return 1 + lo;
}

function fmt(t) {
  if (t === 0) return '0';
  if (t >= 0.1) return t.toFixed(2);
  return t.toPrecision(1);
}

export class PersistencePlot {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {(threshold: number) => void} onChange called while the user clicks or drags
   * @param {(variation: number) => void} [onChangeVariation] same, with shift held
   */
  constructor(canvas, onChange, onChangeVariation) {
    this.canvas = canvas;
    this.onChange = onChange;
    this.onChangeVariation = onChangeVariation;
    this.saddles = [];            // saddle persistences, ascending
    this.variation = Infinity;
    this.curve = [0];
    this.tmax = 1;
    this.minThreshold = 0;
    this.threshold = 0;
    this.colors = { text: '#444', axis: '#bbb', line: '#2563eb', fill: 'rgba(37,99,235,0.12)',
                    marker: '#d97706', locked: 'rgba(120,120,120,0.18)', variation: '#dc2626' };
    this.pad = { l: 34, r: 34, t: 8, b: 24 };

    let dragging = false;
    const set = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const x = (ev.clientX - rect.left - this.pad.l) / (rect.width - this.pad.l - this.pad.r);
      const s = Math.min(1, Math.max(thresholdToSlider(this.minThreshold, this.tmax), x));
      const value = sliderToThreshold(s, this.tmax);
      if (ev.shiftKey && this.onChangeVariation) this.onChangeVariation(value); else this.onChange(value);
    };
    canvas.addEventListener('pointerdown', (ev) => { dragging = true; canvas.setPointerCapture(ev.pointerId); set(ev); });
    canvas.addEventListener('pointermove', (ev) => { if (dragging) set(ev); });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  /** @param {{curve: number[], tmax: number, minThreshold: number, saddles?: number[]}} d
   *  ``saddles``: persistence of every saddle of the file, ascending. */
  setData({ curve, tmax, minThreshold, saddles = [] }) {
    this.curve = curve; this.tmax = tmax; this.minThreshold = minThreshold; this.saddles = saddles;
    this.draw();
  }

  /** Current variation threshold p_hi (Infinity = no links cut). */
  setVariation(v) { this.variation = v; this.draw(); }

  setThreshold(t) { this.threshold = t; this.draw(); }

  setTheme(colors) { Object.assign(this.colors, colors); this.draw(); }

  draw() {
    const c = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);

    const { l, r, t, b } = this.pad;
    const pw = w - l - r, ph = h - t - b;
    const maxCount = this.curve.length + 1;
    const X = (s) => l + s * pw;
    const Y = (n) => t + ph * (1 - (maxCount > 1 ? Math.log(n) / Math.log(maxCount) : 0));

    g.font = '11px system-ui, sans-serif';
    g.fillStyle = this.colors.text; g.strokeStyle = this.colors.axis; g.lineWidth = 1;

    // y ticks: powers of ten and the maximum
    g.textAlign = 'right'; g.textBaseline = 'middle';
    const ticks = new Set([1, maxCount]);
    for (let p = 10; p < maxCount; p *= 10) {
      if (Math.abs(Y(p) - Y(maxCount)) >= 14) ticks.add(p);     // skip a label that would overlap the max
    }
    for (const n of ticks) {
      g.beginPath(); g.moveTo(l, Math.round(Y(n)) + 0.5); g.lineTo(l + pw, Math.round(Y(n)) + 0.5);
      g.globalAlpha = 0.35; g.stroke(); g.globalAlpha = 1;
      g.fillText(String(n), l - 4, Y(n));
    }
    // x ticks
    g.textBaseline = 'top';
    for (const s of [0, 0.25, 0.5, 0.75, 1]) {
      g.textAlign = s === 1 ? 'right' : s === 0 ? 'left' : 'center';   // keep end labels inside the canvas
      g.fillText(fmt(sliderToThreshold(s, this.tmax)), X(s), t + ph + 5);
    }

    // thresholds below the file's base graph cannot be shown
    const s0 = thresholdToSlider(this.minThreshold, this.tmax);
    if (s0 > 0) { g.fillStyle = this.colors.locked; g.fillRect(l, t, s0 * pw, ph); }

    // the step curve
    const N = Math.max(2, Math.floor(pw));
    g.beginPath();
    for (let i = 0; i <= N; i++) {
      const s = i / N;
      const y = Y(peakCount(this.curve, sliderToThreshold(s, this.tmax)));
      if (i === 0) g.moveTo(X(s), y); else g.lineTo(X(s), y);
    }
    g.strokeStyle = this.colors.line; g.lineWidth = 1.6; g.stroke();
    g.lineTo(X(1), t + ph); g.lineTo(X(0), t + ph); g.closePath();
    g.fillStyle = this.colors.fill; g.fill();

    // links left at each variation threshold (red), with its own scale on the right
    const nLinks = this.saddles.length;
    if (nLinks > 0) {
      const YL = (n) => t + ph * (1 - (nLinks > 1 ? Math.log(Math.max(n, 1)) / Math.log(nLinks) : 1));
      g.beginPath();
      for (let i = 0; i <= N; i++) {
        const sx = i / N;
        const y = YL(saddleCount(this.saddles, sliderToThreshold(sx, this.tmax)));
        if (i === 0) g.moveTo(X(sx), y); else g.lineTo(X(sx), y);
      }
      g.strokeStyle = this.colors.variation; g.lineWidth = 1.4; g.stroke();
      g.fillStyle = this.colors.variation; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText(String(nLinks), l + pw + 4, YL(nLinks));
      g.fillText('1', l + pw + 4, YL(1));
      if (Number.isFinite(this.variation)) {
        const sv = thresholdToSlider(this.variation, this.tmax);
        g.setLineDash([4, 3]); g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(X(sv), t); g.lineTo(X(sv), t + ph); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.arc(X(sv), YL(saddleCount(this.saddles, this.variation)), 3.5, 0, Math.PI * 2); g.fill();
      }
    }

    // the current threshold
    const s = thresholdToSlider(this.threshold, this.tmax);
    g.strokeStyle = this.colors.marker; g.lineWidth = 2;
    g.beginPath(); g.moveTo(X(s), t); g.lineTo(X(s), t + ph); g.stroke();
    g.fillStyle = this.colors.marker;
    g.beginPath();
    g.arc(X(s), Y(peakCount(this.curve, this.threshold)), 3.5, 0, Math.PI * 2);
    g.fill();
  }
}
