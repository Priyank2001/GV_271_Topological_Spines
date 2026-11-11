// Wires the page together: dataset list, spine view, slider, persistence plot, info panel.

import { simplify, maxPersistence, maxSaddlePersistence } from './simplify.js';
import { SpineView, peakGradientCss, hexToRgb, regionColor, setColormap, getColormap } from './spineView.js';
import { DataView } from './dataView.js';
import { VolumeView, RAMP, bandColor } from './volumeView.js';
import { PersistencePlot, sliderToThreshold, thresholdToSlider } from './persistencePlot.js';

const $ = (id) => document.getElementById(id);
const el = {
  dataset: $('dataset'), status: $('status'), spine: $('spine'), tooltip: $('tooltip'),
  blurb: $('blurb'), slider: $('threshold'), thresholdOut: $('thresholdOut'), plot: $('plot'),
  counts: $('counts'), info: $('info'), valueBar: $('valueBar'), vLo: $('vLo'), vHi: $('vHi'),
  layoutInfo: $('layoutInfo'),
  data: $('data'), dataCanvas: $('dataCanvas'), dataBar: $('dataBar'), dataNote: $('dataNote'),
  variation: $('variation'), variationOut: $('variationOut'),
  dataHint: $('dataHint'), volumeCanvas: $('volumeCanvas'), embeddingRow: $('embeddingRow'),
  volumeControls: $('volumeControls'), volThresh: $('volThresh'), volThreshOut: $('volThreshOut'),
  volDensity: $('volDensity'), volShade: $('volShade'), colorByIso: $('colorByIso'), isoControls: $('isoControls'),
  volIsoLevels: $('volIsoLevels'), volIsoSurf: $('volIsoSurf'), embedding: $('embedding'), colorBy: $('colorBy'),
  style: $('style'), scale: $('scale'), scaleOut: $('scaleOut'), scaleRow: $('paperRows'),
  radiusLaw: $('radiusLaw'), gap: $('gap'), gapOut: $('gapOut'), scaleMode: $('scaleMode'),
  colormap: $('colormap'), cmapReverse: $('cmapReverse'), cmapPreview: $('cmapPreview'),
  styleNote: $('styleNote'), discLegend: $('discLegend'), arcs: $('arcs'),
  toggleData: $('toggleData'), toggleSide: $('toggleSide'),
  colorLegend: $('colorLegend'), clTitle: $('clTitle'), clBar: $('clBar'), clFade: $('clFade'),
  clTick: $('clTick'), clRange: $('clRange'), clLo: $('clLo'), clHi: $('clHi'),
  clSwatches: $('clSwatches'), clNote: $('clNote'), clSub: $('clSub'),
};

// ---- panel toggles ------------------------------------------------------
// The data view and the controls can be hidden; the spine then takes the room.
// Each view watches its own size (ResizeObserver), so nothing else is needed.
// The choice is remembered per browser.
const PANELS = [
  { button: el.toggleData, cls: 'hide-data', key: 'd', store: 'gv271.hideData' },
  { button: el.toggleSide, cls: 'hide-side', key: 'c', store: 'gv271.hideSide' },
];
function setPanel(panel, shown) {
  document.body.classList.toggle(panel.cls, !shown);
  panel.button.setAttribute('aria-pressed', String(shown));
  try { localStorage.setItem(panel.store, shown ? '0' : '1'); } catch { /* storage blocked */ }
}
for (const panel of PANELS) {
  let hidden = false;
  try { hidden = localStorage.getItem(panel.store) === '1'; } catch { /* storage blocked */ }
  setPanel(panel, !hidden);
  panel.button.addEventListener('click', () => setPanel(panel, document.body.classList.contains(panel.cls)));
}
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, select, textarea')) return;
  const panel = PANELS.find((x) => x.key === e.key.toLowerCase());
  if (panel) setPanel(panel, document.body.classList.contains(panel.cls));
});

// variation = the paper's p_hi; Infinity means no link is cut.
const state = { doc: null, index: [], tmax: 1, base: 0, threshold: 0, variation: Infinity, vis: null };

const fmt = (x, digits = 4) => (x === null ? '–' : Number(x.toPrecision(digits)).toString());
const pct = (a, b) => `${((100 * a) / b).toFixed(a / b < 0.01 ? 2 : 1)} %`;
const count = (n) => n.toLocaleString('en-US');

function cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function themeColors() {
  return {
    bg: cssColor('--bg'), link: cssColor('--link'), saddle: cssColor('--saddle'),
    loop: cssColor('--loop'), ring: cssColor('--ring'), select: cssColor('--accent'),
  };
}

// ---- views --------------------------------------------------------------
/** Stands in for the spine view when WebGL cannot start, so the rest of the page still works. */
class UnavailableView {
  constructor(container, reason) {
    this.selected = null;
    const box = document.createElement('div');
    box.className = 'no-webgl';
    box.innerHTML = '<p><strong>The spine cannot be drawn here.</strong> ' +
      'This browser could not create a WebGL context' +
      (reason ? ` (${String(reason).split('\n')[0]})` : '') +
      '. Try another browser or enable hardware acceleration. ' +
      'The controls on the right still work.</p>';
    container.appendChild(box);
  }
  setGraph() {}
  setVisible() {}
  select() {}
  setTheme() {}
}

/** Stands in for the data view when WebGL cannot start. */
/** Stands in for the volume view when WebGL cannot start. */
class UnavailableVolume {
  constructor(container) {
    const box = document.createElement('div');
    box.className = 'no-webgl';
    box.innerHTML = '<p><strong>The volume cannot be drawn here</strong> (no WebGL).</p>';
    container.appendChild(box);
  }
  setGraph(doc) { return VolumeView.supports(doc); }          // the controls still work
  setVisible() {} setSelectedPeaks() {} setTheme() {} setColorBy() {} setThreshold() {} setDensity() {}
  setShading() {} getThreshold() { return 0.3; } resize() {}
}

class UnavailableData {
  constructor(container) {
    const box = document.createElement('div');
    box.className = 'no-webgl';
    box.innerHTML = '<p><strong>The data view cannot be drawn here</strong> (no WebGL).</p>';
    container.appendChild(box);
  }
  setGraph(doc) { return DataView.embeddingNames(doc); }       // the menu still works
  setVisible() {} setSelectedPeaks() {} setTheme() {} setEmbedding() {} setColorBy() {}
}

function createView() {
  try {
    return new SpineView(el.spine, { onHover, onSelect });
  } catch (err) {
    console.error(err);
    setStatus('WebGL unavailable', true);
    return new UnavailableView(el.spine, err.message);
  }
}
const view = createView();
let dataView;
try {
  dataView = new DataView(el.dataCanvas, { onPick: onPickPoint });
} catch (err) {
  console.error(err);
  dataView = new UnavailableData(el.dataCanvas);
}
let volumeView;
try {
  volumeView = new VolumeView(el.volumeCanvas, { onPick: onPickPoint });
} catch (err) {
  console.error(err);
  volumeView = new UnavailableVolume(el.volumeCanvas);
}
view.setStyle?.('paper');
view.setShowArcs?.(true);
const plot = new PersistencePlot(el.plot, (t) => setThreshold(t), (v) => setVariation(v));

function applyTheme() {
  const c = themeColors();
  view.setTheme(c);
  dataView.setTheme({ bg: hexToRgb(cssColor('--bg')) });
  volumeView.setTheme({ bg: hexToRgb(cssColor('--bg')), marker: hexToRgb(cssColor('--ring')),
                        select: hexToRgb(cssColor('--accent')) });
  plot.setTheme({ text: cssColor('--muted'), axis: cssColor('--muted'), line: cssColor('--accent'),
                  fill: cssColor('--plot-fill'), marker: cssColor('--loop'), locked: cssColor('--locked'),
                  variation: cssColor('--variation') });
}
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
applyTheme();
setColormap(el.colormap.value, el.cmapReverse.checked);   // the browser may restore the form on reload
el.valueBar.style.background = el.cmapPreview.style.background = peakGradientCss();

// ---- colour map ---------------------------------------------------------
// One value colour map for the spine (discs / contours), the scatterplot and the volume.
function applyColormap() {
  setColormap(el.colormap.value, el.cmapReverse.checked);
  el.valueBar.style.background = el.cmapPreview.style.background = peakGradientCss();
  view.refreshColors?.();
  dataView.refreshColors?.();
  volumeView.refreshColors?.();
  syncedKey = ''; syncDataSelection();             // redraw the data-view legend
}
el.colormap.addEventListener('change', applyColormap);
el.cmapReverse.addEventListener('change', applyColormap);

// ---- threshold ----------------------------------------------------------
function setThreshold(t) { setThresholds(t, state.variation); }

/** p_hi: links with persistence above it are cut. At the end of the slider (or above every link) it is off. */
function setVariation(v) { setThresholds(state.threshold, v); }

function setThresholds(t, v) {
  const { doc } = state;
  if (!doc) return;
  t = Math.min(state.tmax, Math.max(state.base, t));
  v = v >= state.tmax * 0.999 ? Infinity : Math.max(v, t);       // p_hi is never below p_lo
  state.threshold = t;
  state.variation = v;
  state.vis = simplify(doc, t, v);
  view.setVisible(state.vis);
  dataView.setVisible(state.vis);
  volumeView.setVisible(state.vis);
  lastHover = null;                    // hover indices refer to the previous visible lists
  syncDataSelection();

  el.slider.value = String(thresholdToSlider(t, state.tmax));
  el.thresholdOut.textContent = fmt(t, 3);
  el.variation.min = String(thresholdToSlider(t, state.tmax));
  el.variation.value = String(Number.isFinite(v) ? thresholdToSlider(v, state.tmax) : 1);
  el.variationOut.textContent = Number.isFinite(v) ? fmt(v, 3) : 'off';
  plot.setThreshold(t);
  plot.setVariation(v);
  const ridges = el.arcs.checked && state.vis.ridges.length
    ? `; ${count(state.vis.ridges.length)} more arcs kept by p* through ${count(state.vis.ghosts.length)} merged peaks` : '';
  el.counts.textContent = `${count(state.vis.peaks.length)} of ${count(doc.peaks.length)} peaks, ` +
    `${count(state.vis.saddles.length)} of ${count(doc.saddles.length)} saddles shown${ridges}`;
  lastHover = null;                    // hover indices refer to the previous visible lists
  renderInfo(view.selected);
}

el.variation.addEventListener('input', () => {
  setVariation(sliderToThreshold(Number(el.variation.value), state.tmax));
});

el.slider.addEventListener('input', () => {
  setThreshold(sliderToThreshold(Number(el.slider.value), state.tmax));
});

// ---- hover / selection --------------------------------------------------
let lastHover = null;

function onHover(node, pos) {
  lastHover = node;
  syncDataSelection();                 // hovering previews the region in the data view
  if (node && pos) {
    el.tooltip.hidden = false;
    el.tooltip.style.left = `${pos.x}px`;
    el.tooltip.style.top = `${pos.y}px`;
    el.tooltip.textContent = node.kind === 'peak'
      ? `Peak ${node.docId} · ${pct(state.vis.size[node.index], state.doc.meta.num_points)} of the points`
      : `Saddle ${node.docId}`;
  } else {
    el.tooltip.hidden = true;
  }
  renderInfo(view.selected ?? node);
}

function onSelect() { renderInfo(view.selected ?? lastHover); syncDataSelection(); }

// ---- linking the spine and the data view -----------------------------------
/**
 * Highlight in the data view the region(s) of the node under the pointer in the
 * spine, or else of the selected one: a peak's region, or both regions a saddle
 * joins. Updates are batched to one per frame, and skipped when nothing changed,
 * because the volume view rewrites a per-voxel texture for each change.
 */
let syncPending = false, syncedKey = '';
function syncDataSelection() {
  if (syncPending) return;
  syncPending = true;
  requestAnimationFrame(() => {
    syncPending = false;
    const node = lastHover ?? view.selected, vis = state.vis;
    const picked = !node || !vis ? [] : node.kind === 'peak' ? [node.index] : vis.saddleEnds[node.index];
    const key = `${state.loads}|${vis ? vis.peaks.length : 0}|${state.threshold}|${state.variation}|${picked.join(',')}`;
    if (key === syncedKey) return;
    syncedKey = key;
    dataView.setSelectedPeaks(picked);
    volumeView.setSelectedPeaks(picked);
    updateLegend(picked);
  });
}

// ---- colour legend of the data view ---------------------------------------
/**
 * What the colours of the scatterplot / volume mean. 'value': the chosen
 * colour map over the file's value range (the same as the spine's peak colours);
 * 'region': one colour per visible peak (``regionColor``, as the dots in the
 * contour-style spine). For volumes it also shows the opacity transfer function:
 * transparent below "Show above", fading in over the next RAMP of value.
 */
function updateLegend(picked = []) {
  const doc = state.doc, vis = state.vis;
  const volume = !el.volumeCanvas.hidden, cloud = !el.dataCanvas.hidden && !volume;
  if (!doc || !vis || !(volume || cloud) || el.dataNote.hidden === false) { el.colorLegend.hidden = true; return; }
  el.colorLegend.hidden = false;
  const [lo, hi] = doc.meta.value_range ?? [0, 1];
  const span = hi - lo || 1;
  const byValue = el.colorBy.value === 'value';
  const byIso = volume && el.colorBy.value === 'iso';
  const thr = volume ? Number(el.volThresh.value) : null;            // volumes hold values in [0, 1]
  const fadePct = (x) => `${(100 * Math.min(1, Math.max(0, (x - lo) / span))).toFixed(1)}%`;

  const cmapLabel = `${getColormap().name}${getColormap().reversed ? ', reversed' : ''}`;
  const levels = byIso ? volumeView.getIsoLevels?.() ?? [] : [];
  el.clTitle.textContent = byIso ? `Colour: iso-value bands (${levels.length + 1} bands, ${cmapLabel})`
    : byValue ? `Colour: field value (${cmapLabel})` : 'Colour: peak region';
  el.clBar.hidden = el.clRange.hidden = !byValue && !volume;
  el.clLo.textContent = fmt(lo, 3);
  el.clHi.textContent = fmt(hi, 3);
  if (byIso) {                                                       // hard stops at the iso values
    const n = levels.length + 1, edges = [lo, ...levels, hi];
    const rgb = (k) => `rgb(${bandColor(k, n).map((c) => Math.round(c * 255)).join(',')})`;
    const pct = (x) => `${(100 * (x - lo) / span).toFixed(2)}%`;
    el.clBar.style.background = `linear-gradient(to right, ${Array.from({ length: n },
      (_, k) => `${rgb(k)} ${pct(edges[k])}, ${rgb(k)} ${pct(edges[k + 1])}`).join(', ')})`;
  } else if (byValue) {
    el.clBar.style.background = peakGradientCss();
  } else {                                                           // grey ramp: opacity only
    el.clBar.style.background = 'linear-gradient(to right, var(--muted), var(--text))';
  }
  if (volume) {                                                      // shade the transparent part
    const a = fadePct(thr), b = fadePct(thr + RAMP);
    el.clFade.style.background =
      `linear-gradient(to right, var(--panel) 0, var(--panel) ${a}, transparent ${b})`;
    el.clTick.hidden = false;
    el.clTick.style.left = `calc(${a} - 1px)`;
  } else {
    el.clFade.style.background = 'none';
    el.clTick.hidden = true;
  }

  el.clSwatches.replaceChildren();
  if (!byValue && !byIso) {
    const show = new Set(picked);
    const order = vis.peaks.map((_, i) => i).sort((a, b) => (show.has(b) - show.has(a)) || vis.size[b] - vis.size[a]);
    const MAX = 8;
    for (const i of order.slice(0, MAX)) {
      const item = document.createElement('span');
      if (show.has(i)) item.className = 'on';
      const sw = document.createElement('i');
      sw.style.background = `rgb(${regionColor(vis.peaks[i]).map((c) => Math.round(c * 255)).join(',')})`;
      item.append(sw, `peak ${vis.peaks[i]}`);
      el.clSwatches.append(item);
    }
    if (order.length > MAX) el.clSwatches.append(`+${order.length - MAX} more`);
  }
  el.clSub.textContent = byIso
    ? (el.volIsoLevels.value === 'spine' ? 'bands between the critical values of the visible peaks and saddles' : 'even bands of value')
    : !byValue ? 'as the peak dots in the spine' + (volume ? '; bar: opacity by value' : '') : '';
  const parts = [];
  if (byIso) parts.push(levels.length ? `Iso values: ${levels.map((c) => fmt(c, 3)).join(', ')}.` : 'No iso values inside the range: one band.');
  if (volume && el.volIsoSurf.value !== 'off' && byIso) parts.push('Iso surfaces get extra opacity.');
  if (volume) parts.push(`Transparent below ${fmt(thr, 2)} (tick, "Show above"), fades in to ${fmt(Math.min(hi, thr + RAMP), 2)}.`);
  if (picked.length) parts.push('Hovered/selected peak in bold, others dimmed.');
  el.clNote.textContent = parts.join(' ');
}

/** A point (or voxel) was clicked in the data view: select the peak whose region holds it. */
function onPickPoint(i) {
  if (i === null || !state.vis) { view.select(null); return; }
  view.select({ kind: 'peak', index: state.vis.owner[state.doc.region[i]] });
}

el.embedding.addEventListener('change', () => dataView.setEmbedding(el.embedding.value));
el.colorBy.addEventListener('change', () => {
  dataView.setColorBy(el.colorBy.value);
  volumeView.setColorBy(el.colorBy.value);
  syncIsoControls();
  syncedKey = ''; syncDataSelection();             // redraw the legend
});
el.volThresh.addEventListener('input', () => {
  volumeView.setThreshold(Number(el.volThresh.value));
  syncedKey = ''; syncDataSelection();             // move the legend's tick
  el.volThreshOut.textContent = Number(el.volThresh.value).toFixed(2);
});
/** The iso controls belong to the 'iso-value bands' colour of the volume; surfaces go off with them. */
function syncIsoControls() {
  const on = !el.volumeCanvas.hidden && el.colorBy.value === 'iso';
  el.isoControls.hidden = !on;
  if (!on && el.volIsoSurf.value !== 'off') { el.volIsoSurf.value = 'off'; volumeView.setIsoSurfaces?.('off'); }
}
el.volIsoLevels.addEventListener('change', () => {
  volumeView.setIsoLevels?.(el.volIsoLevels.value);
  syncedKey = ''; syncDataSelection();             // redraw the legend
});
el.volIsoSurf.addEventListener('change', () => {
  volumeView.setIsoSurfaces?.(el.volIsoSurf.value);
  syncedKey = ''; syncDataSelection();
});
el.volDensity.addEventListener('input', () => volumeView.setDensity(Number(el.volDensity.value)));
el.volShade.addEventListener('change', () => volumeView.setShading(el.volShade.checked));

// ---- spine drawing style -----------------------------------------------
function applyStyle() {
  view.setStyle?.(el.style.value);
  const paper = el.style.value === 'paper';
  el.scaleRow.hidden = !paper;
  const active = Boolean(state.doc?.data?.values && state.doc?.region);   // what the paper style needs
  el.styleNote.textContent = !paper ? '' : active
    ? 'Rings are isovalues (the values of the shown peaks and saddles, and the minimum), nested as in ' +
      'Correa et al., section 4. A link is swept by circles whose radius is the number of points of the ' +
      'peak\'s region between the contour and that point of the link, so the contour through a saddle ' +
      'pinches to a figure-eight there. Colour = isovalue.'
    : 'This file has no per-point values, so discs are drawn instead.';
  el.discLegend.textContent = paper
    ? `contour ${el.radiusLaw.value === '1' ? 'radius' : 'area'} is proportional to the points of the peak's region it encloses` +
      (el.scaleMode.value === 'saddle' ? ' (the pale outer halo is capped at 2× the widest saddle-level contour)' : '') +
      '; the dot is the maximum'
    : 'disc area is proportional to the points in the peak\'s region';
}
el.style.addEventListener('change', applyStyle);
el.arcs.addEventListener('change', () => {
  view.setShowArcs?.(el.arcs.checked);
  setThresholds(state.threshold, state.variation);          // refresh the counts
});
el.scale.addEventListener('input', () => {
  const k = Number(el.scale.value);
  el.scaleOut.textContent = `×${k.toFixed(2)}`;
  view.setContourScale?.(k);
});
el.scaleOut.textContent = '×1.00';
el.radiusLaw.addEventListener('change', () => { view.setRadiusExponent?.(Number(el.radiusLaw.value)); applyStyle(); });
el.scaleMode.addEventListener('change', () => { view.setScaleMode?.(el.scaleMode.value); applyStyle(); });
el.gap.addEventListener('input', () => {
  const g = Number(el.gap.value);
  el.gapOut.textContent = `${(100 * g).toFixed(1)} %`;
  view.setLevelGap?.(g);
});
el.gapOut.textContent = '2.0 %';

function row(label, value) { return `<tr><th>${label}</th><td>${value}</td></tr>`; }

function renderInfo(node) {
  const { doc, vis } = state;
  if (!node || !vis) { el.info.innerHTML = '<p class="muted">Hover or click a node.</p>'; return; }
  const chosen = view.selected && view.selected.kind === node.kind && view.selected.index === node.index;
  const tag = chosen ? ' <span class="muted">(selected)</span>' : '';

  if (node.kind === 'peak') {
    const p = doc.peaks[node.docId];
    const merged = Array.from(vis.owner).filter((o) => o === node.index).length - 1;
    el.info.innerHTML = `<div class="title">Peak ${node.docId}${tag}</div><table>` +
      row('value', fmt(p.value)) +
      row('persistence', p.persistence === null ? 'global maximum' : fmt(p.persistence)) +
      row('region', `${count(vis.size[node.index])} points (${pct(vis.size[node.index], doc.meta.num_points)})`) +
      (merged > 0 ? row('absorbed', `${merged} smaller peak${merged > 1 ? 's' : ''}`) : '') +
      (p.position ? row('position', p.position.map((x) => fmt(x, 3)).slice(0, 6).join(', ') +
                                    (p.position.length > 6 ? ', …' : '')) : '') +
      '</table>';
  } else {
    const s = doc.saddles[node.docId];
    const [a, b] = vis.saddleEnds[node.index].map((i) => vis.peaks[i]);
    el.info.innerHTML = `<div class="title">Saddle ${node.docId}${tag}</div><table>` +
      row('value', fmt(s.value)) +
      row('joins', `peak ${a} and peak ${b}`) +
      row('persistence', fmt(s.persistence)) +
      (s.saturation !== undefined ? row('saturated p*', fmt(s.saturation)) : '') +
      row('type', s.is_merge ? 'merges two peaks' : 'closes a loop') +
      row('relief', fmt(s.relief, 3)) +
      row('support', `${count(s.support)} graph edges`) +
      '</table>';
  }
}

// ---- loading ------------------------------------------------------------
function setStatus(text, error = false) {
  el.status.textContent = text;
  el.status.classList.toggle('error', error);
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return res.json();
}

/**
 * Show the data view that fits the file: a scatterplot with a dimensionality-reduction
 * menu for point clouds, a volume rendering for grids.
 */
function showDataPanel(doc, names, volume) {
  const cloud = names.length > 0;
  const any = cloud || volume;
  el.dataBar.hidden = !any;
  el.dataHint.hidden = !any;
  el.dataNote.hidden = any;
  el.dataCanvas.hidden = !cloud;
  el.volumeCanvas.hidden = !volume;
  el.embeddingRow.hidden = !cloud;
  el.volumeControls.hidden = !volume;
  el.colorByIso.hidden = !volume;                  // bands are drawn by the volume view only
  el.isoControls.hidden = true;                    // shown again by syncIsoControls for volumes

  if (!any) {
    el.dataNote.textContent = doc.meta.kind === 'grid'
      ? 'This file has no per-voxel data, so the volume cannot be drawn. Re-export it without --no-data / --no-region.'
      : 'This file has no 2D embedding of the points. Re-export it with --embed pca tsne …';
    return;
  }
  el.colorBy.value = 'region';
  if (volume) {
    el.dataHint.textContent = 'drag to rotate · wheel to zoom · double-click to reset · click the volume to select its peak ' +
      '· dots are the visible peaks';
    const t = volumeView.getThreshold();
    el.volThresh.value = String(t);
    el.volThreshOut.textContent = t.toFixed(2);
    el.volDensity.value = '1'; el.volShade.checked = true;
    volumeView.setColorBy('region');
    volumeView.setIsoLevels?.(el.volIsoLevels.value);
    syncIsoControls();
    volumeView.resize?.();
    return;
  }
  el.dataHint.textContent = 'click a point to select its peak · select a peak in the spine to highlight its region';
  el.embedding.replaceChildren(...names.map((n) => {
    const o = document.createElement('option');
    o.value = n; o.textContent = EMBEDDING_LABELS[n] ?? n;
    return o;
  }));
  el.embedding.value = doc.data.embedding_default ?? names[0];
  dataView.setColorBy?.('region');
  dataView.resize?.();
}

const EMBEDDING_LABELS = {
  pca: 'PCA', random_projection: 'Random projection', isomap: 'Isomap', tsne: 't-SNE',
  umap: 'UMAP', kernel_pca: 'Kernel PCA',
};

async function loadDataset(id) {
  const entry = state.index.find((d) => d.id === id);
  if (!entry) return;
  setStatus(`loading ${id}…`);
  try {
    const doc = await getJSON(`data/${entry.file}`);
    state.doc = doc;
    state.tmax = Math.max(maxPersistence(doc), maxSaddlePersistence(doc)) || 1;
    state.variation = Infinity;
    state.base = doc.meta.simplify_threshold ?? 0;
    const initial = doc.meta.initial_threshold ?? state.base;

    view.setGraph(doc);
    plot.setData({ curve: doc.persistence_curve ?? [], tmax: state.tmax, minThreshold: state.base,
                  saddles: doc.saddles.map((x) => x.persistence).sort((a, b) => a - b) });
    el.slider.min = String(thresholdToSlider(state.base, state.tmax));
    el.blurb.textContent = entry.text ?? '';
    el.vLo.textContent = fmt(Math.min(...doc.peaks.map((p) => p.value)), 3);
    el.vHi.textContent = fmt(Math.max(...doc.peaks.map((p) => p.value)), 3);
    const sp = doc.spine;
    el.layoutInfo.textContent =
      `PCA + stress majorisation, computed once on the ${doc.peaks.length} peaks and ` +
      `${doc.saddles.length} saddles in this file. Stress ${fmt(sp.stress, 3)} ` +
      `(PCA alone ${fmt(sp.pca_stress, 3)}); ${count(sp.crossings)} crossing link pairs at full detail.`;
    lastHover = null;
    view.select(null);
    const names = dataView.setGraph(doc);
    const volume = volumeView.setGraph(doc);
    showDataPanel(doc, names, volume);
    state.loads = (state.loads ?? 0) + 1;
    syncedKey = '';
    setThreshold(initial);
    applyStyle();
    setStatus(`${count(doc.meta.num_points)} points · ${doc.meta.dimensions}D`);
    history.replaceState(null, '', `#${id}`);
  } catch (err) {
    console.error(err);
    setStatus(String(err.message ?? err), true);
  }
}

async function init() {
  try {
    state.index = (await getJSON('data/index.json')).datasets;
  } catch (err) {
    setStatus(location.protocol === 'file:'
      ? 'open this page through a web server: python3 -m http.server -d web'
      : String(err.message), true);
    return;
  }
  for (const d of state.index) {
    const o = document.createElement('option');
    o.value = d.id; o.textContent = d.label;
    el.dataset.appendChild(o);
  }
  const wanted = location.hash.slice(1);
  const first = state.index.some((d) => d.id === wanted) ? wanted : state.index[0].id;
  el.dataset.value = first;
  el.dataset.addEventListener('change', () => loadDataset(el.dataset.value));
  await loadDataset(first);
}

window.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') view.select(null); });
init();
