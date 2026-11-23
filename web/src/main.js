// Wires the page together: dataset list, spine view, slider, persistence plot, info panel,
// and the modes chosen in the rail on the left: explore (spine + data view), compare
// (a second spine with another extraction method, see compare.js) and parallel
// coordinates (spine + every point as a line across its original dimensions).

import { maxPersistence, maxSaddlePersistence, maxArcHeight } from './simplify.js';
import { visibleGraph, isSimplifiable, compareShown, traitDifferences, findAgreement, findVisible,
         allMaxima, sharedColorIds } from './compare.js';
import { ComparePanel } from './comparePanel.js';
import { ComparisonBox } from './comparisonBox.js';
import { SpineView, peakGradientCss, hexToRgb, regionColor, setColormap, getColormap } from './spineView.js';
import { DataView, clusterColor } from './dataView.js';
import { VolumeView, RAMP, bandColor } from './volumeView.js';
import { PersistencePlot, sliderToThreshold, thresholdToSlider } from './persistencePlot.js';
import { ParallelView } from './parallelView.js';
import { profileColumns, peakProfile, orderRows, PROFILE_BINS } from './profile.js';
import { kmeans, dbscan, dbscanCost, agreement } from './cluster.js';

const $ = (id) => document.getElementById(id);
const el = {
  dataset: $('dataset'), status: $('status'), spine: $('spine'), tooltip: $('tooltip'),
  blurb: $('blurb'), slider: $('threshold'), thresholdOut: $('thresholdOut'), plot: $('plot'),
  counts: $('counts'), info: $('info'), valueBar: $('valueBar'), vLo: $('vLo'), vHi: $('vHi'),
  layoutInfo: $('layoutInfo'),
  data: $('data'), dataCanvas: $('dataCanvas'), dataBar: $('dataBar'), dataNote: $('dataNote'),
  variation: $('variation'), variationOut: $('variationOut'),
  arcRow: $('arcRow'), arcLow: $('arcLow'), arcOut: $('arcOut'),
  dataHint: $('dataHint'), volumeCanvas: $('volumeCanvas'), embeddingRow: $('embeddingRow'),
  drawRow: $('drawRow'), drawMode: $('drawMode'),
  scatter: $('scatter'), scX: $('scX'), scY: $('scY'), scColor: $('scColor'), scColorDims: $('scColorDims'),
  scDraw: $('scDraw'), scPeakDims: $('scPeakDims'), scInfo: $('scInfo'), scCanvas: $('scCanvas'), scNote: $('scNote'),
  scBrush: $('scBrush'), scClear: $('scClear'),
  clMethod: $('clMethod'), clK: $('clK'), clEps: $('clEps'), clMin: $('clMin'), clSpace: $('clSpace'),
  clKRow: $('clKRow'), clEpsRow: $('clEpsRow'), clMinRow: $('clMinRow'), clRun: $('clRun'), clClear: $('clClear'),
  clResult: $('clResult'), colorByCluster: $('colorByCluster'), scColorCluster: $('scColorCluster'),
  volumeControls: $('volumeControls'), volThresh: $('volThresh'), volThreshOut: $('volThreshOut'),
  volDensity: $('volDensity'), volShade: $('volShade'), colorByIso: $('colorByIso'), isoControls: $('isoControls'),
  volIsoLevels: $('volIsoLevels'), volIsoSurf: $('volIsoSurf'), volIsoCmap: $('volIsoCmap'), volIsoRev: $('volIsoRev'), embedding: $('embedding'), colorBy: $('colorBy'),
  style: $('style'), scale: $('scale'), scaleOut: $('scaleOut'), scaleRow: $('paperRows'),
  radiusLaw: $('radiusLaw'), gap: $('gap'), gapOut: $('gapOut'), scaleMode: $('scaleMode'),
  gapRow: $('gapRow'), levels: $('levels'), levelsOut: $('levelsOut'), levelsRow: $('levelsRow'),
  colormap: $('colormap'), cmapReverse: $('cmapReverse'), cmapPreview: $('cmapPreview'),
  styleNote: $('styleNote'), discLegend: $('discLegend'), arcs: $('arcs'), relayout: $('relayout'), floor: $('floor'), height: $('height'), heightRows: $('heightRows'),
  panelH: $('panelH'), spineH: $('spineH'), tooltipH: $('tooltipH'), tipBox: $('tipBox'),
  profile: $('profile'), profileShow: $('profileShow'), profileOrder: $('profileOrder'),
  heightScale: $('heightScale'), heightScaleOut: $('heightScaleOut'), heightShape: $('heightShape'),
  toggleData: $('toggleData'), toggleSide: $('toggleSide'),
  colorLegend: $('colorLegend'), clTitle: $('clTitle'), clBar: $('clBar'), clFade: $('clFade'),
  clTick: $('clTick'), clRange: $('clRange'), clLo: $('clLo'), clHi: $('clHi'),
  clSwatches: $('clSwatches'), clNote: $('clNote'), clSub: $('clSub'),
  toggleCompare: $('toggleCompare'), toggleSpine: $('toggleSpine'), variantA: $('variantA'), variantB: $('variantB'), barA: $('barA'),
  swapVariants: $('swapVariants'), panelB: $('panelB'), spineB: $('spineB'), compareBox: $('compareBox'),
  sliderNote: $('sliderNote'), markLegend: $('markLegend'),
  modeButtons: [...document.querySelectorAll('#modeRail .mode')],
  parallel: $('parallel'), pcCanvas: $('pcCanvas'), pcColorBy: $('pcColorBy'), pcOpacity: $('pcOpacity'),
  pcOpacityOut: $('pcOpacityOut'), pcClear: $('pcClear'), pcBrush: $('pcBrush'),
  pcAxes: $('pcAxes'), pcAxesCount: $('pcAxesCount'), pcAxesList: $('pcAxesList'), pcAxesAll: $('pcAxesAll'),
  pcAxesCoords: $('pcAxesCoords'),
  pcNote: $('pcNote'),
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
// Compare mode: variantA is drawn in the main panel (and drives the data view), variantB in panelB.
// sliderDoc: the shown document whose persistence the sliders walk (null when no shown method simplifies).
// mode: 'explore' | 'compare' | 'parallel' (the rail on the left); compare mirrors state.compare.
const state = { doc: null, index: [], tmax: 1, base: 0, threshold: 0, variation: Infinity, arcLow: 0, arcMax: 1, vis: null,
                entry: null, variantA: null, variantB: null, compare: false, sliderDoc: null, traitLabels: {},
                mode: 'explore' };

const fmt = (x, digits = 4) => (x === null || x === undefined ? '–' : Number(x.toPrecision(digits)).toString());
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (a, b) => `${((100 * a) / b).toFixed(a / b < 0.01 ? 2 : 1)} %`;
const count = (n) => n.toLocaleString('en-US');
// Brushes that filter the spine: the points each one keeps (Uint8Array) or null (see applyBrushFocus)
const brushes = { parallel: null, scatter: null };
// Clustering result (see runClustering near the end): labels per point of ``doc``, or null
let clusterRes = null;
const FOCUS_SHARE = 0.2;    // a peak stays in colour if at least this share of its region is brushed
// Scatter mode state (see loadScatter near the end): its view and the document it shows
let scatterView = null;
const sc = { doc: null };
// peak profile state (see renderProfile near the end): declared early, the info panel renders during loading
const prof = { doc: null, coords: null, cols: null, note: '', loading: null, vis: null, cache: new Map(), last: null };

function cssColor(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function themeColors() {
  return {
    bg: cssColor('--bg'), link: cssColor('--link'), saddle: cssColor('--saddle'),
    loop: cssColor('--loop'), ring: cssColor('--ring'), select: cssColor('--accent'),
    mark: cssColor('--mark'),
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
    return new SpineView(el.spine, { onHover, onSelect, onCamera: (c) => linkCamera('A', c),
      onBandHover: (b, pos) => { if (state.mode === 'height') heightView?.showLinkedBand?.(b); showBandTip(el.tooltip, b, pos); } });
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
// The parallel-coordinates view is created when its mode is first opened (see setMode).
let pcView = null;
let pcDoc = null;              // the document the parallel view shows (it lags while coordinates load)
view.setStyle?.(el.style.value);                        // 'simple': the page opens on the extremum graph itself
view.setRadiusExponent?.(el.radiusLaw.value === 'nddav' ? 'nddav' : Number(el.radiusLaw.value));
view.setScaleMode?.(el.scaleMode.value);
view.setLevelGap?.(Number(el.gap.value));
view.setLevelCount?.(Number(el.levels.value));
view.setShowArcs?.(true);
const plot = new PersistencePlot(el.plot, (t) => setThreshold(t), (v) => setVariation(v));

// ---- compare mode: the second spine ---------------------------------------
const panelB = new ComparePanel(el.spineB, {
  onHover: (node) => (node ? renderInfo(node, 'B') : renderInfo(view.selected ?? lastHover)),
  onSelect: () => linkSelection('B'),
  onCamera: (c) => linkCamera('B', c),
});
const comparisonBox = new ComparisonBox(el.compareBox);

// ---- height mode: the same spine as a landscape, next to the flat one -------
// Created the first time the mode is opened. It shows the main panel's graph (same visible
// peaks and saddles, same indices), so hover and selection carry over one to one.
let heightView = null;
let heightLinking = false;             // true while one of the two selects on behalf of the other

/** Every spine view on screen (the second exists once compare mode, the third once height mode was opened). */
function spineViews() { return [view, panelB.view, heightView].filter(Boolean); }

function ensureHeightView() {
  if (heightView) return heightView;
  try {
    heightView = new SpineView(el.spineH, { onHover: onHoverH, onSelect: onSelectH,
      onBandHover: (b, pos) => { view.showLinkedBand?.(b); showBandTip(el.tooltipH, b, pos); } });
  } catch (err) {
    console.error(err);
    setStatus('WebGL unavailable for the landscape', true);
    return null;
  }
  applyViewSettings(heightView);          // (it has no graph yet, so nothing is built twice)
  return heightView;
}

/** Bring the landscape up to date with the main panel (graph, visible part, selection). */
function syncHeightView() {
  const hv = ensureHeightView();
  if (!hv || !state.doc) return;
  hv.setHeight?.(heightFor(hv));             // stand up first, so the landscape is built once
  if (hv.doc !== state.doc) hv.setGraph(state.doc);
  if (state.vis) hv.setVisible(state.vis);
  heightLinking = true;
  try { hv.select(view.selected ? { kind: view.selected.kind, index: view.selected.index } : null); }
  finally { heightLinking = false; }
  applyHeight();
}

/** Label of a hovered contour band: its value and the peaks it goes round. */
function showBandTip(tip, band, pos) {
  if (!band || !pos) { if (tip.dataset.kind === 'band') tip.hidden = true; return; }   // a node label stays
  tip.dataset.kind = 'band';
  const more = band.peaks && band.peaks.length > 1 ? `, joined with ${band.peaks.length - 1} more peak${band.peaks.length > 2 ? 's' : ''}` : '';
  tip.hidden = false;
  tip.style.left = `${pos.x}px`;
  tip.style.top = `${pos.y}px`;
  tip.textContent = `Contour ≥ ${fmt(band.c, 3)} around peak ${band.docId}${more}`;
}

/** Text of the small hover label of a node. */
function nodeLabel(node) {
  return node.kind === 'peak'
    ? `Peak ${node.docId} · ${pct(state.vis.size[node.index], state.doc.meta.num_points)} of the points`
    : `Saddle ${node.docId}`;
}

function onHoverH(node, pos) {
  view.showLinkedHover?.(node);                 // the same node lights up in the flat spine
  lastHover = node;
  syncDataSelection();                          // and its region in the data view
  if (node && pos) {
    el.tooltipH.hidden = false;
    el.tooltipH.dataset.kind = 'node';
    el.tooltipH.style.left = `${pos.x}px`;
    el.tooltipH.style.top = `${pos.y}px`;
    el.tooltipH.textContent = nodeLabel(node);
  } else {
    el.tooltipH.hidden = true;
  }
  renderInfo(view.selected ?? node);
}

function onSelectH() {
  if (heightLinking) return;
  heightLinking = true;
  try {
    const n = heightView.selected;
    view.select(n ? { kind: n.kind, index: n.index } : null);
  } finally {
    heightLinking = false;
  }
}

/** Give a newly created spine view the current drawing settings. */
function applyViewSettings(v) {
  v.setTheme(themeColors());
  v.setStyle?.(el.style.value);
  v.setShowArcs?.(el.arcs.checked);
  v.setRelayout?.(el.relayout?.checked ?? true);
  v.setFloor?.(el.floor?.checked ?? true);
  v.setHeight?.(heightFor(v));
  v.setContourScale?.(Number(el.scale.value));
  v.setRadiusExponent?.(radiusLaw());
  v.setScaleMode?.(el.scaleMode.value);
  v.setLevelGap?.(Number(el.gap.value));
  v.setLevelCount?.(Number(el.levels.value));
}

function applyTheme() {
  const c = themeColors();
  for (const v of spineViews()) v.setTheme(c);
  dataView.setTheme({ bg: hexToRgb(cssColor('--bg')) });
  scatterView?.setTheme({ bg: hexToRgb(cssColor('--bg')) });
  pcView?.setTheme(pcTheme());
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
  for (const v of spineViews()) v.refreshColors?.();
  dataView.refreshColors?.();
  scatterView?.refreshColors?.();
  volumeView.refreshColors?.();
  pcView?.refreshColors();
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
  if (state.sliderDoc) {
    t = Math.min(state.tmax, Math.max(state.base, t));
    v = v >= state.tmax * 0.999 ? Infinity : Math.max(v, t);       // p_hi is never below p_lo
  } else {
    t = 0; v = Infinity;                                             // no shown method simplifies
  }
  state.threshold = t;
  state.variation = v;
  state.vis = visibleGraph(doc, t, v, state.arcLow);                 // AnyEG: its own arc filter instead
  view.setVisible(state.vis);
  if (heightView && state.mode === 'height') {
    if (heightView.doc !== doc) heightView.setGraph(doc);
    heightView.setVisible(state.vis);
  }
  dataView.setVisible(state.vis);
  if (scatterView && sc.doc === state.doc) scatterView.setVisible(state.vis);
  applyBrushFocus();                               // the same brush, the peaks of the new threshold
  if (clusterRes?.doc === state.doc) renderClusters();   // agreement with the peaks of this threshold
  volumeView.setVisible(state.vis);
  if (pcReady()) pcView.setVisible(state.vis);
  if (state.compare && panelB.doc) panelB.setThresholds(t, v, state.arcLow);
  lastHover = null;                    // hover indices refer to the previous visible lists
  syncDataSelection();

  el.slider.value = String(thresholdToSlider(t, state.tmax));
  el.thresholdOut.textContent = state.sliderDoc ? fmt(t, 3) : '–';
  el.variation.min = String(thresholdToSlider(t, state.tmax));
  el.variation.value = String(Number.isFinite(v) ? thresholdToSlider(v, state.tmax) : 1);
  el.variationOut.textContent = !state.sliderDoc ? '–' : Number.isFinite(v) ? fmt(v, 3) : 'off';
  plot.setThreshold(t);
  plot.setVariation(v);
  el.counts.textContent = isSimplifiable(doc)
    ? `${count(state.vis.peaks.length)} of ${count(doc.peaks.length)} peaks · ` +
      `${count(state.vis.saddles.length)} of ${count(doc.saddles.length)} saddles shown`
    : `All ${count(doc.peaks.length)} peaks · ${count(state.vis.saddles.length)} of ${count(doc.saddles.length)} links shown` +
      (state.arcLow > 0 ? ' (arc filter)' : '');
  el.counts.title = el.arcs.checked && state.vis.ridges.length
    ? `${count(state.vis.ridges.length)} more arcs kept by p* through ${count(state.vis.ghosts.length)} merged peaks` : '';
  renderInfo(view.selected);
  updateComparison();
}

/**
 * Point the sliders and the persistence plot at the first shown document that can
 * be simplified (left, else right). A method without simplification (AnyEG)
 * always shows its full graph, whatever the sliders say.
 */
function configureSlider() {
  const shown = [state.doc, state.compare ? panelB.doc : null].filter(Boolean);
  const sdoc = shown.find(isSimplifiable) ?? null;
  state.sliderDoc = sdoc;
  el.slider.disabled = el.variation.disabled = !sdoc;
  const fixed = shown.filter((d) => !isSimplifiable(d)).map(variantLabel);
  el.sliderNote.hidden = fixed.length === 0;
  el.sliderNote.textContent = !fixed.length ? ''
    : sdoc ? `Peak persistence simplification acts on ${variantLabel(sdoc)} only. ${fixed.join(' and ')} ` +
             'has no peak hierarchy; it uses its own arc height-difference filter (below).'
           : `${fixed.join(' and ')} has no peak hierarchy: use its arc height-difference filter (below).`;
  // AnyEG's own filter: its own control, over the arc heights of the shown method without simplification
  const adoc = shown.find((d) => !isSimplifiable(d)) ?? null;
  el.arcRow.hidden = !adoc;
  if (adoc) {
    state.arcMax = maxArcHeight(adoc) || 1;
    $('arcLabel').textContent = `Arc height-difference filter (${variantLabel(adoc)})`;
    state.arcLow = Math.min(state.arcLow ?? 0, state.arcMax);
    el.arcLow.value = String(state.arcLow / state.arcMax);
    el.arcOut.textContent = state.arcLow > 0 ? fmt(state.arcLow, 3) : 'off';
  } else state.arcLow = 0;
  if (!sdoc) {
    state.tmax = 1; state.base = 0;
    plot.setData({ curve: [], tmax: 1, minThreshold: 0, saddles: [] });
    return;
  }
  state.tmax = Math.max(maxPersistence(sdoc), maxSaddlePersistence(sdoc)) || 1;
  state.base = sdoc.meta.simplify_threshold ?? 0;
  plot.setData({ curve: sdoc.persistence_curve ?? [], tmax: state.tmax, minThreshold: state.base,
                 saddles: sdoc.saddles.map((x) => x.persistence).sort((a, b) => a - b) });
  el.slider.min = String(thresholdToSlider(state.base, state.tmax));
}

/** Threshold to open a document at: its file's initial value, or 0 without a slider. */
function initialThreshold() {
  return state.sliderDoc ? (state.sliderDoc.meta.initial_threshold ?? state.base) : 0;
}

el.arcLow.addEventListener('input', () => {
  state.arcLow = Number(el.arcLow.value) * (state.arcMax ?? 1);
  el.arcOut.textContent = state.arcLow > 0 ? fmt(state.arcLow, 3) : 'off';
  // only the panel(s) showing a method with arc filtering change; the other keeps its graph and view
  if (state.compare && panelB.doc && !isSimplifiable(panelB.doc)) {
    panelB.setThresholds(state.threshold, state.variation, state.arcLow);
  }
  if (state.doc && !isSimplifiable(state.doc)) setThresholds(state.threshold, state.variation);
  else updateComparison();
});

el.variation.addEventListener('input', () => {
  setVariation(sliderToThreshold(Number(el.variation.value), state.tmax));
});

el.slider.addEventListener('input', () => {
  setThreshold(sliderToThreshold(Number(el.slider.value), state.tmax));
});

// ---- hover / selection --------------------------------------------------
let lastHover = null;

function onHover(node, pos) {
  if (state.mode === 'height') heightView?.showLinkedHover?.(node);   // light it up in the landscape too
  lastHover = node;
  syncDataSelection();                 // hovering previews the region in the data view
  if (node && pos) {
    el.tooltip.hidden = false;
    el.tooltip.dataset.kind = 'node';
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

function onSelect() {
  renderInfo(view.selected ?? lastHover); syncDataSelection(); linkSelection('A');
  if (heightView && state.mode === 'height' && !heightLinking) {
    heightLinking = true;
    try { heightView.select(view.selected ? { kind: view.selected.kind, index: view.selected.index } : null); }
    finally { heightLinking = false; }
  }
}

// ---- compare mode: linked selection, linked camera, the comparison box ------
let linking = false;                   // true while one panel selects on behalf of the other
let cameraLinking = false;

/** Both spines share one frame (comparison/viewer_export.py), so they also share pan and zoom. */
function linkCamera(from, camera) {
  if (!state.compare || cameraLinking) return;
  const other = from === 'A' ? panelB.view : view;
  if (!other?.setCamera) return;
  cameraLinking = true;
  try { other.setCamera(camera); } finally { cameraLinking = false; }
}

/** Vertex of a node of the main panel. */
function vertexOfA(node) { return (node.kind === 'peak' ? state.doc.peaks : state.doc.saddles)[node.docId].vertex; }

/** Selecting a node in one panel selects the node at the same vertex in the other (or clears it). */
function linkSelection(from) {
  if (linking) return;
  if (!state.compare || !panelB.doc || !panelB.vis) {
    if (from === 'B') renderInfo(panelB.view?.selected ?? null, 'B');
    return;
  }
  linking = true;
  try {
    if (from === 'A') {
      const n = view.selected;
      panelB.selectVertex(n?.kind ?? 'peak', n ? vertexOfA(n) : null);
    } else {
      const n = panelB.view.selected;
      const index = n ? findVisible(state.doc, state.vis, n.kind, panelB.vertexOf(n)) : -1;
      view.select(index < 0 ? null : { kind: n.kind, index });
      if (n && index < 0) renderInfo(n, 'B');           // nothing on the left: describe the right node
    }
  } finally {
    linking = false;
  }
  updateComparison();
}

/** Rings on the maxima the other method lacks, and the text of the comparison box. */
function updateComparison() {
  const on = state.compare && panelB.doc && panelB.vis && state.vis;
  el.markLegend.hidden = !on;
  if (!on) {
    comparisonBox.hide();
    view.setMarked?.([]);
    panelB.view?.setMarked?.([]);
    return;
  }
  const shown = compareShown(state.doc, state.vis, panelB.doc, panelB.vis);
  view.setMarked?.(shown.missingA);
  panelB.view.setMarked?.(shown.missingB);
  const infoA = variantInfo(state.variantA), infoB = variantInfo(state.variantB);
  const selA = view.selected, selB = panelB.view.selected;
  let selection = '';
  const one = selA && !selB ? [selA, 'left', vertexOfA(selA), variantLabel(panelB.doc)]
    : selB && !selA ? [selB, 'right', panelB.vertexOf(selB), variantLabel(state.doc)] : null;
  if (one) {
    const [n, side, vertex, otherLabel] = one;
    selection = `The selected ${n.kind} (${side}, vertex ${count(vertex)}) is not drawn by ${esc(otherLabel)}` +
      (n.kind === 'peak' && !allMaxima(side === 'left' ? panelB.doc : state.doc).has(vertex)
        ? ', which does not find a maximum there.' : ' at the current setting.');
  }
  comparisonBox.render({
    infoA, infoB, shown, selection,
    agreement: findAgreement(state.entry, state.variantA, state.variantB),
    diffs: traitDifferences(infoA, infoB, state.traitLabels),
  });
}

// ---- linking the spine and the data view -----------------------------------
/** Visible peaks whose regions the data views highlight: the hovered node's, else the selected one's. */
function pickedPeaks() {
  const node = lastHover ?? view.selected, vis = state.vis;
  return !node || !vis ? [] : node.kind === 'peak' ? [node.index] : vis.saddleEnds[node.index];
}

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
    const vis = state.vis, picked = pickedPeaks();
    const key = `${state.loads}|${vis ? vis.peaks.length : 0}|${state.threshold}|${state.variation}|${picked.join(',')}`;
    if (key === syncedKey) return;
    syncedKey = key;
    dataView.setSelectedPeaks(picked);
    volumeView.setSelectedPeaks(picked);
    if (pcReady()) pcView.setSelectedPeaks(picked);
    if (scatterView && sc.doc === state.doc) scatterView.setSelectedPeaks(picked);
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
  const byCluster = cloud && el.colorBy.value === 'cluster' && clusterRes?.doc === doc;
  const byIso = volume && el.colorBy.value === 'iso';
  const thr = volume ? Number(el.volThresh.value) : null;            // volumes hold values in [0, 1]
  const fadePct = (x) => `${(100 * Math.min(1, Math.max(0, (x - lo) / span))).toFixed(1)}%`;

  const cmapLabel = `${getColormap().name}${getColormap().reversed ? ', reversed' : ''}`;
  const levels = byIso ? volumeView.getIsoLevels?.() ?? [] : [];
  const bandLabel = bandMapName() === 'fig1' ? 'the paper\'s Fig. 1 colours' : el.volIsoCmap.value === 'spine' ? cmapLabel
    : `${el.volIsoCmap.value}${el.volIsoRev.checked ? ', reversed' : ''}`;
  el.clTitle.textContent = (byIso ? `Colour: iso-value bands (${levels.length + 1} bands, ${bandLabel})`
    : byValue ? `Colour: field value (${cmapLabel})`
    : byCluster ? (clusterRes.names ? `Colour: ${clusterRes.label}` : `Colour: cluster (${clusterRes.label})`) : 'Colour: peak region') +
    (cloud && el.drawMode?.value === 'density' ? ' · density: brighter = more points (log scale)' : '');
  el.clBar.hidden = el.clRange.hidden = !byValue && !volume;
  el.clLo.textContent = fmt(lo, 3);
  el.clHi.textContent = fmt(hi, 3);
  if (byIso) {                                                       // hard stops at the iso values
    const n = levels.length + 1, edges = [lo, ...levels, hi];
    const rgb = (k) => `rgb(${bandColor(k, n, bandMapName(), el.volIsoRev.checked, levels, [lo, hi]).map((c) => Math.round(c * 255)).join(',')})`;
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
  if (byCluster) {
    for (const r of clusterRes.table.slice(0, 8)) {
      const item = document.createElement('span');
      const sw = document.createElement('i');
      sw.style.background = `rgb(${clusterColor(r.cluster).map((c) => Math.round(c * 255)).join(',')})`;
      item.append(sw, r.cluster < 0 ? (clusterRes.names ? 'unknown' : 'noise') : clusterRes.names?.[r.cluster] ?? `cluster ${r.cluster + 1}`);
      el.clSwatches.append(item);
    }
  } else if (!byValue && !byIso) {
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
  if (byIso && el.volIsoSurf.value === 'add') parts.push('Iso surfaces (above "Show above") drawn as shells over the volume.');
  if (byIso && el.volIsoSurf.value === 'only') parts.push('Only the iso surfaces above "Show above" are drawn, as shells.');
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
el.drawMode?.addEventListener('change', () => { dataView.setDraw?.(el.drawMode.value); updateLegend(pickedPeaks()); });
el.colorBy.addEventListener('change', () => {
  dataView.setColorBy(el.colorBy.value);
  volumeView.setColorBy(el.colorBy.value === 'cluster' ? 'region' : el.colorBy.value);
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
/** Band colour map in effect: "as the spine" follows the spine, so the Fig. 1 style gives the paper's palette. */
function bandMapName() {
  return el.volIsoCmap.value === 'spine' && el.style.value === 'fig1' ? 'fig1' : el.volIsoCmap.value;
}
function applyBandColormap() {
  volumeView.setBandColormap?.(bandMapName(), el.volIsoRev.checked);
  syncedKey = ''; syncDataSelection();             // redraw the legend
}
el.volIsoCmap.addEventListener('change', applyBandColormap);
el.volIsoRev.addEventListener('change', applyBandColormap);
el.volIsoSurf.addEventListener('change', () => {
  volumeView.setIsoSurfaces?.(el.volIsoSurf.value);
  syncedKey = ''; syncDataSelection();
});
el.volDensity.addEventListener('input', () => volumeView.setDensity(Number(el.volDensity.value)));
el.volShade.addEventListener('change', () => volumeView.setShading(el.volShade.checked));

// ---- spine drawing style -----------------------------------------------
/** Value of the radius law control: a number, or 'nddav' (exponent 2/d). */
function radiusLaw() { return el.radiusLaw.value === 'nddav' ? 'nddav' : Number(el.radiusLaw.value); }

// Size settings each contour style starts from when it is chosen (they can be changed afterwards).
const STYLE_DEFAULTS = { nddav: { radiusLaw: 'nddav', scaleMode: 'saddle' }, paper: { radiusLaw: '1', scaleMode: 'saddle' } };

function applyStyle() {
  for (const v of spineViews()) v.setStyle?.(el.style.value);
  const style = el.style.value, contours = style !== 'simple';
  el.scaleRow.hidden = !contours;
  el.gapRow.hidden = style !== 'paper' && style !== 'fig1';
  el.levelsRow.hidden = style !== 'nddav';
  const active = Boolean(state.doc?.data?.values && state.doc?.region);   // what the contour styles need
  el.styleNote.textContent = !contours
    ? 'The extremum graph: a disc per maximum, a diamond per saddle and a line for every link a saddle makes. ' +
      'Press Spine (key s) to draw the topological spine, the nested contours, around it.'
    : !active
    ? 'This file has no per-point values, so discs are drawn instead.'
    : style === 'fig1'
      ? 'Drawn to look like Correa et al., Fig. 1: the critical values as banded contours (blue, green, yellow, ' +
        'orange, magenta, white), circles around each peak above its saddles, pinching to a point at each saddle ' +
        '(the figure-eight), radius ~ (points of the region)^(1/d). Saddles are placed where the circles meet; ' +
        'a low side peak (arm) gets its saddle beyond the big circle, which spikes out to meet the arm\'s ' +
        'teardrop. Node positions here are drawn, not measured. Radius and colour map do not apply; ' +
        'contour size does (×1 is the intended size).'
    : style === 'nddav'
      ? 'Drawn as LLNL\'s NDDAV draws topological spines. Rings are isovalues: evenly spaced values plus the ' +
        'value of every shown saddle. Above its saddles a peak\'s contours are circles around it; below a saddle ' +
        'the circle stretches along the link to a circle at the saddle, which is 0 at the saddle\'s own value ' +
        '(the figure-eight). Colour = isovalue.'
      : 'Rings are isovalues (the values of the shown peaks and saddles, and the minimum), nested as in ' +
        'Correa et al., section 4. A link is swept by circles whose radius is the number of points of the ' +
        'peak\'s region between the contour and that point of the link, so the contour through a saddle ' +
        'pinches to a figure-eight there. Colour = isovalue.';
  const law = el.radiusLaw.value;
  el.discLegend.textContent = contours ? 'contour size: points in the peak\'s region; dot: the maximum'
    : 'disc size: points in the peak\'s region';
  el.discLegend.title = contours
    ? (law === 'nddav' ? 'Contour radius grows as (points of the peak\'s region it encloses)^(2/d).'
      : `Contour ${law === '1' ? 'radius' : 'area'} is proportional to the points of the peak's region it encloses.`) +
      (el.scaleMode.value === 'saddle' ? ' The pale outer halo is capped at 2x the widest saddle-level contour.' : '')
    : 'Disc area is proportional to the points in the peak\'s region.';
}
// The page opens on the extremum graph (discs and lines); "Spine" draws the contours of the
// topological spine around it, "Spine" again goes back. The contour style comes back as it was.
let contourStyle = 'nddav';
function syncSpineButton() {
  el.toggleSpine.setAttribute('aria-pressed', String(el.style.value !== 'simple'));
}
function setSpineShown(on) {
  el.style.value = on ? contourStyle : 'simple';
  el.style.dispatchEvent(new Event('change'));
}
el.toggleSpine.addEventListener('click', () => setSpineShown(el.style.value === 'simple'));
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, select, textarea')) return;
  if (e.key.toLowerCase() === 's') setSpineShown(el.style.value === 'simple');
});
el.style.addEventListener('change', () => {
  if (el.style.value !== 'simple') contourStyle = el.style.value;
  syncSpineButton();
  const preset = STYLE_DEFAULTS[el.style.value];
  if (preset) {
    el.radiusLaw.value = preset.radiusLaw;
    el.scaleMode.value = preset.scaleMode;
    for (const v of spineViews()) { v.setRadiusExponent?.(radiusLaw()); v.setScaleMode?.(el.scaleMode.value); }
  }
  applyStyle();
  applyBandColormap();          // "as the spine colour map": the 3D bands follow the spine's style
});
/** Height mode settings from the page (see SpineView.setHeight). */
function heightSettings() {
  return { on: Boolean(el.height?.checked), scale: Number(el.heightScale?.value ?? 1),
           smooth: el.heightShape?.value === 'smooth', solid: el.heightShape?.value === 'solid' };
}
/** Height mode: the landscape always stands up and the main spine stays flat; else the checkbox. */
function heightFor(v) {
  const base = heightSettings();
  if (v && v === heightView) return { ...base, on: true };
  if (state.mode === 'height') return { ...base, on: false };
  return base;
}
function applyHeight() {
  const mode = state.mode === 'height';
  if (el.heightRows) el.heightRows.hidden = !(el.height?.checked || mode);
  if (el.height) {
    el.height.disabled = mode;
    el.height.closest('label').title = mode ? 'In height mode the left spine stays flat and the right one is the landscape' : '';
  }
  if (el.heightScaleOut) el.heightScaleOut.textContent = `×${Number(el.heightScale.value).toFixed(2)}`;
  for (const v of spineViews()) v.setHeight?.(heightFor(v));
}
el.height?.addEventListener('change', applyHeight);
el.heightScale?.addEventListener('input', applyHeight);
el.heightShape?.addEventListener('change', applyHeight);
el.floor?.addEventListener('change', () => {
  for (const v of spineViews()) v.setFloor?.(el.floor.checked);
});
el.relayout?.addEventListener('change', () => {
  for (const v of spineViews()) v.setRelayout?.(el.relayout.checked);
});
el.arcs.addEventListener('change', () => {
  for (const v of spineViews()) v.setShowArcs?.(el.arcs.checked);
  setThresholds(state.threshold, state.variation);          // refresh the counts
});
el.scale.addEventListener('input', () => {
  const k = Number(el.scale.value);
  el.scaleOut.textContent = `×${k.toFixed(2)}`;
  for (const v of spineViews()) v.setContourScale?.(k);
});
el.scaleOut.textContent = '×1.00';
el.radiusLaw.addEventListener('change', () => {
  for (const v of spineViews()) v.setRadiusExponent?.(radiusLaw());
  applyStyle();
});
el.scaleMode.addEventListener('change', () => {
  for (const v of spineViews()) v.setScaleMode?.(el.scaleMode.value);
  applyStyle();
});
el.gap.addEventListener('input', () => {
  const g = Number(el.gap.value);
  el.gapOut.textContent = `${(100 * g).toFixed(1)} %`;
  for (const v of spineViews()) v.setLevelGap?.(g);
});
el.gapOut.textContent = '2.0 %';
el.levels.addEventListener('input', () => {
  el.levelsOut.textContent = el.levels.value;
  for (const v of spineViews()) v.setLevelCount?.(Number(el.levels.value));
});
el.levelsOut.textContent = el.levels.value;

function row(label, value) { return `<tr><th>${label}</th><td>${value}</td></tr>`; }

/**
 * Describe a node in the side panel. ``side`` 'B' means a node of the second spine
 * (compare mode). Methods without simplification have no persistence to show.
 */
function renderInfo(node, side = 'A') {
  renderProfile(side === 'B' ? null : node);
  const B = side === 'B';
  const doc = B ? panelB.doc : state.doc, vis = B ? panelB.vis : state.vis;
  const shownIn = B ? panelB.view : view;
  if (!node || !vis) { el.info.innerHTML = '<p class="muted">Hover or click a node.</p>'; return; }
  const chosen = shownIn?.selected && shownIn.selected.kind === node.kind && shownIn.selected.index === node.index;
  const tag = (chosen ? ' <span class="muted">(selected)</span>' : '') +
    (state.compare ? ` <span class="muted">· ${esc(variantLabel(doc))}, ${B ? 'right' : 'left'}</span>` : '');
  const other = state.compare && panelB.doc ? (B ? state.doc : panelB.doc) : null;
  const simplifies = isSimplifiable(doc);

  if (node.kind === 'peak') {
    const p = doc.peaks[node.docId];
    const merged = Array.from(vis.owner).filter((o) => o === node.index).length - 1;
    el.info.innerHTML = `<div class="title">Peak ${node.docId}${tag}</div><table>` +
      row('value', fmt(p.value)) +
      row('persistence', !simplifies ? 'not computed (no simplification in this method)'
        : p.persistence === null ? 'global maximum' : fmt(p.persistence)) +
      row('region', `${count(vis.size[node.index])} points (${pct(vis.size[node.index], doc.meta.num_points)})`) +
      (merged > 0 ? row('absorbed', `${merged} smaller peak${merged > 1 ? 's' : ''}`) : '') +
      row('vertex', count(p.vertex)) +
      (other ? row(`in ${esc(variantLabel(other))}`,
                   allMaxima(other).has(p.vertex) ? 'also a maximum' : '<b>not a maximum</b> (ringed)') : '') +
      (p.position ? row('position', p.position.map((x) => fmt(x, 3)).slice(0, 6).join(', ') +
                                    (p.position.length > 6 ? ', …' : '')) : '') +
      '</table>';
  } else {
    const s = doc.saddles[node.docId];
    const [a, b] = vis.saddleEnds[node.index].map((i) => vis.peaks[i]);
    const joins = s.reaches && s.reaches.length > 2
      ? `peaks ${s.reaches.join(', ')} (${s.reaches.length} maxima)` : `peak ${a} and peak ${b}`;
    el.info.innerHTML = `<div class="title">Saddle ${node.docId}${tag}</div><table>` +
      row('value', fmt(s.value)) +
      row('joins', joins) +
      (simplifies
        ? row('persistence', fmt(s.persistence)) +
          (s.saturation !== undefined ? row('saturated p*', fmt(s.saturation)) : '') +
          row('type', s.is_merge ? 'merges two peaks' : 'closes a loop') +
          row('relief', fmt(s.relief, 3)) +
          row('support', `${count(s.support)} graph edges`)
        : row('below its maximum', `${fmt(s.persistence)} (value gap to the maximum of its own basin)`)) +
      row('vertex', count(s.vertex)) +
      (other ? row(`in ${esc(variantLabel(other))}`,
                   other.saddles.some((x) => x.vertex === s.vertex) ? 'also a saddle' : 'not a saddle') : '') +
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
  if (el.drawRow) el.drawRow.hidden = !cloud;
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
    volumeView.setBandColormap?.(bandMapName(), el.volIsoRev.checked);   // the browser may restore the form
    applyHeight();
    dataView.setDraw?.(el.drawMode?.value ?? 'points');
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

// ---- extraction variants ---------------------------------------------------
// index.json lists, per dataset, the extraction methods ("variants") that were
// exported: the main file (ours, with the field data) and one file per other
// method (same schema, no data section). See comparison/viewer_export.py.
const docCache = new Map();                   // file -> promise of the document (current dataset only)

function cachedJSON(file) {
  if (!docCache.has(file)) docCache.set(file, getJSON(`data/${file}`));
  return docCache.get(file);
}

/** Variants of a dataset; an index.json from before variants existed has only the main file. */
function availableVariants(entry) {
  return entry?.variants?.length ? entry.variants
    : [{ id: 'ours', label: 'Ours', file: entry?.file, simplifiable: true, traits: {} }];
}
const usableVariants = (entry) => availableVariants(entry).filter((v) => v.file);
function variantInfo(id) { return availableVariants(state.entry).find((v) => v.id === id) ?? null; }
function variantLabel(doc) { return doc?.meta?.variant?.label ?? 'Ours'; }

/** The document of one variant. The field data (values, embeddings) is stored once, in the main file. */
async function variantDoc(entry, id) {
  const main = await cachedJSON(entry.file);
  const v = variantInfo(id);
  if (!v?.file || v.file === entry.file) return main;
  const doc = await cachedJSON(v.file);
  // a variant with another function on the same points carries its own values; the rest
  // of the data section (embeddings) is shared with the main file
  if (main.data) doc.data = doc.data ? { ...main.data, ...doc.data } : main.data;
  return doc;
}

// ---- our pipeline's family of files: one per (scalar function, k). Ids: 'ours' (the main file),
// 'ours_k<k>' (the main function at another k) and, for datasets with several scalar functions
// (index.json "scalars"), 'ours_f_<function>_s<smoothing>[_k<k>]'. The Method menus show one
// "Ours"; the Scalar menu and the k slider of each panel pick the file.
const K_ID = /^ours_k(\d+)$/;
const F_ID = /^ours_f_(\w+?_s\d+)(?:_k(\d+))?$/;
const isOurs = (id) => id === 'ours' || K_ID.test(id) || F_ID.test(id);
const methodOf = (id) => (isOurs(id) ? 'ours' : id);
const mainScalar = (entry = state.entry) => entry?.scalars?.main ?? null;
/** {f, k} of an id of our family (f: scalar menu id, null for datasets with one function). */
function parseOurs(id, entry = state.entry) {
  if (id === 'ours') return { f: mainScalar(entry), k: entry?.k };
  let m = K_ID.exec(id);
  if (m) return { f: mainScalar(entry), k: Number(m[1]) };
  m = F_ID.exec(id);
  return m ? { f: m[1], k: m[2] ? Number(m[2]) : entry?.k } : null;
}
function oursId(f, k, entry = state.entry) {
  if (!f || f === mainScalar(entry)) return k === entry?.k ? 'ours' : `ours_k${k}`;
  return k === entry?.k ? `ours_f_${f}` : `ours_f_${f}_k${k}`;
}
const hasFile = (id, entry = state.entry) => usableVariants(entry).some((v) => v.id === id);
/** The k values with a file for function ``f``, ascending: [{k, id}]. Empty for volumes and old index files. */
function kChoices(entry = state.entry, f = mainScalar(entry)) {
  if (!entry?.k) return [];
  const list = usableVariants(entry).map((v) => ({ id: v.id, p: parseOurs(v.id, entry) }))
    .filter((x) => x.p && x.p.f === f).map((x) => ({ k: x.p.k, id: x.id }));
  return list.length > 1 ? list.sort((a, b) => a.k - b.k) : [];
}
function kOfVariant(id) { return parseOurs(id)?.k ?? state.entry?.k; }
const kState = { A: null, B: null };            // chosen k per panel (null: the dataset's own)
const fState = { A: null, B: null };            // chosen scalar function per panel (null: the main one)
const panelId = (side) => (side === 'A' ? state.variantA : state.variantB);
/** The scalar function shown (or, for another method, chosen) in panel ``side``. */
function fOf(side) {
  const id = panelId(side);
  return methodOf(id) === 'ours' ? parseOurs(id)?.f ?? null : (fState[side] ?? mainScalar());
}
/** The variant id to load for a Method-menu choice in panel ``side``. */
function resolveMethod(value, side) {
  if (value !== 'ours') return value;
  const f = fState[side] ?? mainScalar(), k = kState[side] ?? state.entry?.k;
  for (const id of [oursId(f, k), oursId(f, state.entry?.k)]) if (hasFile(id)) return id;
  return 'ours';
}
function syncKControls() {
  for (const side of ['A', 'B']) {
    const row = $(`kRow${side}`), input = $(`k${side}`), out = $(`kOut${side}`);
    if (!row) continue;
    const id = panelId(side);
    const choices = kChoices(state.entry, fOf(side));
    const show = choices.length > 1 && (side === 'A' || state.compare);
    row.hidden = !show;
    if (!show) continue;
    const ours = methodOf(id) === 'ours';
    const k = ours ? kOfVariant(id) : (kState[side] ?? state.entry.k);
    input.max = String(choices.length - 1);
    input.value = String(Math.max(0, choices.findIndex((c) => c.k === k)));
    input.disabled = !ours;
    out.textContent = ours ? String(k) : `${k} (Ours only)`;
  }
  const lab = $('kLabelA');
  if (lab) lab.textContent = state.compare ? 'Neighbours (k), left' : 'Neighbours (k)';
  syncFControls();
}
/** The Scalar menus: shown for datasets with several functions, when the panel shows Ours. */
function syncFControls() {
  const menu = state.entry?.scalars;
  for (const side of ['A', 'B']) {
    const row = $(`fRow${side}`), select = $(`f${side}`);
    if (!row) continue;
    const ours = methodOf(panelId(side)) === 'ours';
    const show = !!menu && ours && (side === 'A' || state.compare);
    row.hidden = !show;
    if (!show) continue;
    const choices = menu.choices.filter((c) => hasFile(oursId(c.id, state.entry.k)));
    if (select.dataset.entry !== state.entry.id) {
      select.replaceChildren(...choices.map((c) => {
        const o = document.createElement('option');
        o.value = c.id;
        o.textContent = c.label;
        return o;
      }));
      select.dataset.entry = state.entry.id;
    }
    select.value = fOf(side) ?? menu.main;
  }
  const lab = $('fLabelA');
  if (lab) lab.textContent = state.compare ? 'Scalar (f), left' : 'Scalar (f)';
}
for (const side of ['A', 'B']) {
  const input = $(`k${side}`);
  if (!input) continue;
  input.addEventListener('input', () => { $(`kOut${side}`).textContent = String(kChoices(state.entry, fOf(side))[Number(input.value)]?.k ?? ''); });
  input.addEventListener('change', async () => {
    const c = kChoices(state.entry, fOf(side))[Number(input.value)];
    if (!c) return;
    kState[side] = c.k;
    if (side === 'A') { state.variantA = c.id; await showVariantA(); }
    else { state.variantB = c.id; await showVariantB(); }
  });
  const fsel = $(`f${side}`);
  fsel?.addEventListener('change', async () => {
    fState[side] = fsel.value;
    const id = resolveMethod('ours', side);
    if (side === 'A') { state.variantA = id; await showVariantA(); }
    else { state.variantB = id; await showVariantB(); }
  });
}

function fillVariantMenus(entry) {
  for (const select of [el.variantA, el.variantB]) {
    select.replaceChildren(...availableVariants(entry).filter((v) => v.id === 'ours' || !isOurs(v.id)).map((v) => {
      const o = document.createElement('option');
      o.value = v.id;
      o.textContent = v.file ? v.label : `${v.label} (not available)`;
      o.disabled = !v.file;
      o.title = v.file ? (v.summary ?? '') : (v.error ?? '');
      return o;
    }));
  }
  const n = usableVariants(entry).length;
  el.barA.hidden = n < 2;
  el.toggleCompare.disabled = n < 2;
  el.toggleCompare.title = n < 2 ? 'This dataset has only one extraction method'
    : 'Compare two extraction methods side by side (key: m)';
}

function updateHash() {
  const e = state.entry;
  if (!e) return;
  const parts = [e.id];
  if (state.compare) parts.push(state.variantA, state.variantB);
  else if (state.variantA !== usableVariants(e)[0]?.id) parts.push(state.variantA);
  history.replaceState(null, '', `#${parts.join(':')}`);
}

function layoutText(doc) {
  const sp = doc.spine;
  const frame = doc.meta.data_file
    ? ' Drawn in the frame of the main file: a node at the same vertex sits at the same place in every method.'
    : '';
  return `PCA + stress majorisation, computed once on the ${doc.peaks.length} peaks and ` +
    `${doc.saddles.length} saddles in this file. Stress ${fmt(sp.stress, 3)} ` +
    `(PCA alone ${fmt(sp.pca_stress, 3)}); ${count(sp.crossings)} crossing link pairs at full detail.${frame}`;
}

async function loadDataset(id, wantA = null, wantB = null) {
  const entry = state.index.find((d) => d.id === id);
  if (!entry) return;
  if (state.entry?.id !== id) { docCache.clear(); panelB.doc = panelB.vis = null; }
  state.entry = entry;
  fillVariantMenus(entry);
  const ids = usableVariants(entry).map((v) => v.id);
  state.variantA = ids.includes(wantA) ? wantA : ids[0];
  state.variantB = ids.includes(wantB) && wantB !== state.variantA ? wantB
    : ids.find((v) => v !== state.variantA) ?? state.variantA;
  el.variantA.value = methodOf(state.variantA);
  el.variantB.value = methodOf(state.variantB);
  kState.A = isOurs(state.variantA) && state.variantA !== 'ours' ? kOfVariant(state.variantA) : null;
  kState.B = isOurs(state.variantB) && state.variantB !== 'ours' ? kOfVariant(state.variantB) : null;
  fState.A = isOurs(state.variantA) ? parseOurs(state.variantA)?.f ?? null : null;
  fState.B = isOurs(state.variantB) ? parseOurs(state.variantB)?.f ?? null : null;
  if (state.compare && ids.length < 2) await setCompare(false);
  await showVariantA();
  if (state.compare) await showVariantB();
}

/** Load variantA into the main panel and the data view. */
async function showVariantA() {
  const entry = state.entry;
  setStatus(`loading ${entry.id}…`);
  try {
    const doc = await variantDoc(entry, state.variantA);
    state.doc = doc;
    state.variation = Infinity;
    view.setGraph(doc);
    configureSlider();
    el.blurb.textContent = entry.text ?? '';
    el.vLo.textContent = fmt(Math.min(...doc.peaks.map((p) => p.value)), 3);
    el.vHi.textContent = fmt(Math.max(...doc.peaks.map((p) => p.value)), 3);
    el.layoutInfo.textContent = layoutText(doc);
    lastHover = null;
    view.select(null);
    const names = dataView.setGraph(doc);
    if (clusterRes && clusterRes.doc !== doc) clearClusters();
    syncLabelOption();
    const volume = volumeView.setGraph(doc);
    showDataPanel(doc, names, volume);
    state.loads = (state.loads ?? 0) + 1;
    syncedKey = '';
    setThreshold(initialThreshold());
    if (state.mode === 'parallel') loadParallel();
    if (state.mode === 'scatter') loadScatter();
    if (state.compare && panelB.doc) panelB.view.setColorIds?.(sharedColorIds(doc, panelB.doc));
    applyStyle();
    setStatus(`${count(doc.meta.num_points)} points · ${doc.meta.dimensions}D`);
    syncKControls();
    updateHash();
  } catch (err) {
    console.error(err);
    setStatus(String(err.message ?? err), true);
  }
}

/** Load variantB into the second panel (compare mode only). */
async function showVariantB() {
  if (!state.compare) return;
  try {
    panelB.ensureView(applyViewSettings);
  } catch (err) {
    console.error(err);
    setStatus('WebGL unavailable for the second spine', true);
    return;
  }
  try {
    const doc = await variantDoc(state.entry, state.variantB);
    const hadSlider = Boolean(state.sliderDoc);
    panelB.setDocument(doc);
    configureSlider();
    setThresholds(!hadSlider && state.sliderDoc ? initialThreshold() : state.threshold, state.variation);
    if (view.getCamera) panelB.view.setCamera?.(view.getCamera());   // start from the left panel's view
    panelB.view.setColorIds?.(sharedColorIds(state.doc, doc));       // a shared maximum: same colour
    syncKControls();
    updateHash();
  } catch (err) {
    console.error(err);
    setStatus(String(err.message ?? err), true);
  }
}

async function setCompare(on) {
  on = Boolean(on) && usableVariants(state.entry).length >= 2;
  state.compare = on;
  if (on) state.mode = 'compare';
  else if (state.mode === 'compare') state.mode = 'explore';
  syncModeUi();
  document.body.classList.toggle('compare', on);
  el.panelB.hidden = !on;                         // shown before the view is created, so it gets a size
  el.swapVariants.hidden = !on;
  el.toggleCompare.setAttribute('aria-pressed', String(on));
  syncKControls();
  if (on) {
    await showVariantB();
    return;
  }
  const hadSlider = Boolean(state.sliderDoc);
  configureSlider();
  setThresholds(!hadSlider && state.sliderDoc ? initialThreshold() : state.threshold, state.variation);
  updateHash();
}

el.toggleCompare.addEventListener('click', () => setCompare(!state.compare));
el.variantA.addEventListener('change', async () => { state.variantA = resolveMethod(el.variantA.value, 'A'); await showVariantA(); });
el.variantB.addEventListener('change', async () => { state.variantB = resolveMethod(el.variantB.value, 'B'); await showVariantB(); });
el.swapVariants.addEventListener('click', async () => {
  [state.variantA, state.variantB] = [state.variantB, state.variantA];
  [kState.A, kState.B] = [kState.B, kState.A];
  [fState.A, fState.B] = [fState.B, fState.A];
  el.variantA.value = methodOf(state.variantA);
  el.variantB.value = methodOf(state.variantB);
  await showVariantA();
  await showVariantB();
});
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, select, textarea')) return;
  if (e.key.toLowerCase() === 'm' && !el.toggleCompare.disabled) setCompare(!state.compare);
});

async function init() {
  let index;
  try {
    index = await getJSON('data/index.json');
  } catch (err) {
    setStatus(location.protocol === 'file:'
      ? 'open this page through a web server: python3 -m http.server -d web'
      : String(err.message), true);
    return;
  }
  state.index = index.datasets;
  state.traitLabels = index.trait_labels ?? {};
  for (const d of state.index) {
    const o = document.createElement('option');
    o.value = d.id; o.textContent = d.label;
    el.dataset.appendChild(o);
  }
  // #name, #name:variant, or #name:left:right (compare mode)
  const [wanted, a, b] = decodeURIComponent(location.hash.slice(1)).split(':');
  const first = state.index.some((d) => d.id === wanted) ? wanted : state.index[0].id;
  el.dataset.value = first;
  el.dataset.addEventListener('change', () => loadDataset(el.dataset.value, state.variantA, state.variantB));
  await loadDataset(first, a, b);
  if (b) await setCompare(true);
  else if (storedMode() === 'parallel') await setMode('parallel');
  else if (storedMode() === 'height') await setMode('height');
  else if (storedMode() === 'scatter') await setMode('scatter');
  window.addEventListener('hashchange', () => openHash(location.hash));   // a typed or followed link
}

/** Open ``#name``, ``#name:variant`` or ``#name:left:right`` (compare mode). */
async function openHash(hash) {
  const [id, a, b] = decodeURIComponent(hash.slice(1)).split(':');
  if (!state.index.some((d) => d.id === id)) return;
  el.dataset.value = id;
  if (!b && state.compare) await setCompare(false);
  await loadDataset(id, a, b);
  if (b && !state.compare) await setCompare(true);
}

// ---- modes: the rail on the left ----------------------------------------------
// Explore: spine + data view. Compare: two spines (setCompare). Parallel: spine +
// parallel coordinates in place of the data view. The parallel mode is remembered
// in the browser; compare mode lives in the link (#name:left:right).
const MODE_STORE = 'gv271.mode';
function storedMode() { try { return localStorage.getItem(MODE_STORE); } catch { return null; } }

function syncModeUi() {
  for (const b of el.modeButtons) b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode));
  document.body.classList.toggle('mode-parallel', state.mode === 'parallel');
  document.body.classList.toggle('mode-height', state.mode === 'height');
  document.body.classList.toggle('mode-scatter', state.mode === 'scatter');
  el.scatter.hidden = state.mode !== 'scatter';
  el.panelH.hidden = state.mode !== 'height';
  el.parallel.hidden = state.mode !== 'parallel';
  el.data.hidden = state.mode === 'parallel' || state.mode === 'scatter';
  el.toggleData.title = state.mode === 'parallel'
    ? 'Show or hide the parallel coordinates (key: d)' : 'Show or hide the data view (key: d)';
  if (state.mode !== 'parallel') pcView?.setActive(false);
}

async function setMode(mode) {
  if (mode === 'compare') { await setCompare(true); return; }      // setCompare sets the mode
  if (state.compare) await setCompare(false);
  state.mode = mode;
  try { localStorage.setItem(MODE_STORE, mode); } catch { /* storage blocked */ }
  syncModeUi();
  if (mode === 'parallel') await loadParallel();
  if (mode === 'scatter') await loadScatter();
  applyBrushFocus();                               // the spine follows the brush of the view on screen
  if (mode === 'height') syncHeightView();
  else applyHeight();                            // the main spine follows the checkbox again
  syncedKey = ''; syncDataSelection();
}

for (const b of el.modeButtons) b.addEventListener('click', () => setMode(b.dataset.mode));
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, select, textarea')) return;
  if (e.key.toLowerCase() === 'p') setMode(state.mode === 'parallel' ? 'explore' : 'parallel');
  if (e.key.toLowerCase() === 'h') setMode(state.mode === 'height' ? 'explore' : 'height');
});

// ---- parallel coordinates --------------------------------------------------------
// The view (and its WebGL context) is created the first time the mode is opened.
// Point clouds need their original coordinates (<name>.coords.json, loaded then);
// grids derive x, y, z from the shape.
/** True when the parallel view holds the current document, so updates for it can be passed on. */
function pcReady() { return Boolean(pcView && pcDoc && pcDoc === state.doc); }

function pcTheme() {
  return { bg: hexToRgb(cssColor('--bg')), text: cssColor('--text'), muted: cssColor('--muted'),
           accent: cssColor('--accent'), context: cssColor('--link') };
}

/** Stands in for the parallel view when WebGL cannot start. */
class UnavailableParallel {
  constructor(container) {
    const box = document.createElement('div');
    box.className = 'no-webgl';
    box.innerHTML = '<p><strong>The parallel coordinates cannot be drawn here</strong> (no WebGL).</p>';
    container.appendChild(box);
  }
  setGraph() { return 0.3; } setCoords() {} setVisible() {} setSelectedPeaks() {} setColorBy() {}
  refreshColors() {} setOpacity() {} setAxisShown() { return false; } showAllAxes() {} clearBrushes() {}
  setTheme() {} setActive() {}
}

function ensureParallelView() {
  if (pcView) return;
  try {
    pcView = new ParallelView(el.pcCanvas, { onPick: onPickPoint, onBrush: renderBrush, onAxes: renderAxesMenu });
  } catch (err) {
    console.error(err);
    pcView = new UnavailableParallel(el.pcCanvas);
  }
  pcView.setTheme(pcTheme());
}

/** Give the parallel view the current document (and its coordinates), then show it. */
async function loadParallel() {
  ensureParallelView();
  const doc = state.doc, entry = state.entry;
  if (!doc) return;
  if (pcDoc !== doc) {
    let coords = null, note = '';
    if (!doc.region) note = 'This file has no per-point regions, so the lines cannot be drawn.';
    else if (doc.meta.kind !== 'grid') {
      if (!entry.coords_file) {
        note = 'No original coordinates for this file (run python3 web/build_data.py --coords-only); ' +
               'only the value and peak axes are shown.';
      } else {
        try {
          setStatus('loading coordinates…');
          coords = await cachedJSON(entry.coords_file);
        } catch (err) {
          console.error(err);
          note = `The coordinates could not be loaded (${err.message}); only the value and peak axes are shown.`;
        }
        setStatus(`${count(doc.meta.num_points)} points · ${doc.meta.dimensions}D`);
      }
    }
    if (state.doc !== doc || state.mode !== 'parallel') return;    // the user moved on meanwhile
    const opacity = pcView.setGraph(doc);
    if (coords) pcView.setCoords(coords);
    pcView.setColorBy(el.pcColorBy.value);
    pcView.setVisible(state.vis);
    pcView.setSelectedPeaks(pickedPeaks());
    pcDoc = doc;
    setPcOpacity(opacity);
    renderBrush(null);
    el.pcNote.textContent = note;
    el.pcNote.hidden = !note;
  }
  if (state.mode === 'parallel') pcView.setActive(true);
}

function setPcOpacity(a) {
  el.pcOpacity.value = String(a);
  el.pcOpacityOut.textContent = Number(el.pcOpacity.value).toFixed(2);
  pcView?.setOpacity(Number(el.pcOpacity.value));
}

/**
 * The spine follows the brush of the view on screen (Parallel or Scatter): a peak whose region has
 * at least FOCUS_SHARE of its points inside the brush stays in colour, the others fade.
 */
function applyBrushFocus() {
  const mask = state.mode === 'parallel' ? brushes.parallel : state.mode === 'scatter' ? brushes.scatter : null;
  const vis = state.vis, doc = state.doc;
  if (!mask || !vis || !doc || mask.length !== doc.meta.num_points) { view.setFocus?.(null); return null; }
  const inside = new Float64Array(vis.peaks.length), total = new Float64Array(vis.peaks.length);
  const owner = vis.owner, region = doc.region;
  for (let i = 0; i < mask.length; i++) { const k = owner[region[i]]; total[k]++; if (mask[i]) inside[k]++; }
  const keep = [], shares = [];
  for (let k = 0; k < vis.peaks.length; k++) {
    const sh = total[k] ? inside[k] / total[k] : 0;
    if (sh >= FOCUS_SHARE) keep.push(k);
    if (inside[k]) shares.push({ k, id: vis.peaks[k], share: sh, count: inside[k] });
  }
  view.setFocus?.(keep);
  return shares.sort((a, b) => b.share - a.share);
}

/** The brush summary above the plot: how many points pass, and in which peaks (click one to select it). */
function renderBrush(summary) {
  brushes.parallel = summary?.mask ?? null;
  applyBrushFocus();
  el.pcClear.disabled = !summary;
  if (!summary) {
    el.pcBrush.innerHTML = '<span>No brush: drag along an axis to keep the points in a range; ' +
      'brushes on several axes combine (a point must pass all).</span>';
    return;
  }
  const { count: n, total, peaks } = summary;
  const parts = [`<span><b>${count(n)}</b> of ${count(total)} points (${pct(n, total)}) pass</span>`];
  if (n) {
    const sel = new Set(view.selected?.kind === 'peak' ? [view.selected.index] : []);
    parts.push('<span>in peak</span>');
    const MAX = 6;
    for (const p of peaks.slice(0, MAX)) {
      const rgb = regionColor(p.id).map((c) => Math.round(c * 255)).join(',');
      parts.push(`<button type="button" data-peak="${p.index}" class="${sel.has(p.index) ? 'on' : ''}" ` +
        `title="Select peak ${p.id} in the spine"><i style="background:rgb(${rgb})"></i>${p.id} · ${pct(p.count, n)}</button>`);
    }
    if (peaks.length > MAX) parts.push(`<span>+${peaks.length - MAX} more</span>`);
  }
  parts.push(`<span class="muted">· peaks with under ${Math.round(100 * FOCUS_SHARE)}% of their region brushed fade in the spine</span>`);
  el.pcBrush.innerHTML = parts.join(' ');
}
el.pcBrush.addEventListener('click', (e) => {
  const b = e.target.closest('button[data-peak]');
  if (!b) return;
  const index = Number(b.dataset.peak);
  const on = view.selected?.kind === 'peak' && view.selected.index === index;
  view.select(on ? null : { kind: 'peak', index });
  for (const x of el.pcBrush.querySelectorAll('button[data-peak]')) x.classList.toggle('on', !on && x === b);
});
el.pcColorBy.addEventListener('change', () => pcView?.setColorBy(el.pcColorBy.value));
el.pcOpacity.addEventListener('input', () => setPcOpacity(Number(el.pcOpacity.value)));

// The "Axes" menu: one checkbox per axis the data offers (rebuilt when the view reports its axes).
function renderAxesMenu(axes) {
  const shown = axes.filter((a) => a.shown).length;
  el.pcAxesCount.textContent = `${shown}/${axes.length}`;
  el.pcAxesList.replaceChildren(...axes.map((a) => {
    const label = document.createElement('label');
    const box = document.createElement('input');
    box.type = 'checkbox'; box.checked = a.shown; box.dataset.key = a.key;
    box.disabled = a.shown && shown <= 2;                  // the last two axes stay
    label.classList.toggle('locked', box.disabled);
    if (box.disabled) label.title = 'At least two axes are drawn';
    label.append(box, a.label);
    return label;
  }));
  el.pcAxesCoords.disabled = !axes.some((a) => a.key.startsWith('c:') || ['x', 'y', 'z'].includes(a.key));
}
el.pcAxesList.addEventListener('change', (e) => {
  const box = e.target.closest('input[data-key]');
  if (box && !pcView?.setAxisShown(box.dataset.key, box.checked)) box.checked = !box.checked;
});
el.pcAxesAll.addEventListener('click', () => pcView?.showAllAxes());
el.pcAxesCoords.addEventListener('click', () => {
  if (!pcView) return;
  pcView.showAllAxes();
  for (const a of pcView.axes()) if (a.key !== 'value' && a.key !== 'peak') pcView.setAxisShown(a.key, false);
});
// Close the menu on a click elsewhere or on Escape.
document.addEventListener('pointerdown', (e) => { if (el.pcAxes.open && !el.pcAxes.contains(e.target)) el.pcAxes.open = false; });
el.pcAxes.addEventListener('keydown', (e) => { if (e.key === 'Escape') { el.pcAxes.open = false; el.pcAxes.querySelector('summary').focus(); } });
el.pcClear.addEventListener('click', () => pcView?.clearBrushes());

window.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') view.select(null); });
// ?debug exposes the page's objects for scripted checks (tests, Playwright).
if (new URLSearchParams(location.search).has('debug')) {
  window.gv = { state, view, panelB, setCompare, setMode, get pcView() { return pcView; } };
}
init();

// ---- descriptions of the controls as hover tips ----------------------------
// The long captions under the options stay in the page (some are rewritten as settings change)
// but are hidden; hovering the option (marked with a small i) shows the caption in a box.
// Collapsible Controls sections (<details class="sec" data-sec=...>): remember which are open.
for (const d of document.querySelectorAll('details[data-sec]')) {
  const key = `gv.sec.${d.dataset.sec}`;
  try { const v = localStorage.getItem(key); if (v !== null) d.open = v === '1'; } catch { /* no storage */ }
  d.addEventListener('toggle', () => { try { localStorage.setItem(key, d.open ? '1' : '0'); } catch { /* no storage */ } });
}

const TIPS = [['fNote', 'fA'], ['kNote', 'kA'], ['arcNote', 'arcLow'], ['clusterNote', 'clMethod'], ['profileNote', 'profileShow'], ['plotNote', 'plot'], ['heightNote', 'height'], ['styleNote', 'style'],
              ['arcsNote', 'arcs'], ['floorNote', 'floor'], ['relayoutNote', 'relayout']];
function initTips() {
  const box = el.tipBox;
  if (!box) return;
  const hide = () => { box.hidden = true; };
  for (const [noteId, ctrlId] of TIPS) {
    const note = $(noteId), ctrl = $(ctrlId);
    if (!note || !ctrl) continue;
    const anchor = ctrl.tagName === 'CANVAS' ? ctrl : (ctrl.closest('label') ?? ctrl);
    note.classList.add('tip-src');
    anchor.classList.add('has-tip');
    if (anchor !== ctrl || ctrl.tagName !== 'CANVAS') {
      const badge = document.createElement('i');
      badge.className = 'tip-badge';
      badge.textContent = 'i';
      badge.setAttribute('aria-hidden', 'true');
      const head = anchor.classList.contains('slider') ? anchor.querySelector('span') : null;   // a slider: next to its name
      if (head) head.firstElementChild ? head.insertBefore(badge, head.querySelector('output')) : head.appendChild(badge);
      else if (ctrl.tagName === 'SELECT') anchor.insertBefore(badge, ctrl); else anchor.appendChild(badge);
    }
    anchor.setAttribute('aria-describedby', noteId);
    const show = () => {
      if (!note.textContent.trim()) return;
      box.innerHTML = note.innerHTML;
      box.hidden = false;
      const r = anchor.getBoundingClientRect();
      const w = box.offsetWidth, h = box.offsetHeight;
      let left = r.left - w - 12;                       // to the left of the side panel
      if (left < 8) left = Math.min(window.innerWidth - w - 8, r.left);
      const top = Math.max(8, Math.min(window.innerHeight - h - 8, r.top));
      box.style.left = `${left}px`;
      box.style.top = `${left === r.left ? r.bottom + 6 : top}px`;
    };
    anchor.addEventListener('mouseenter', show);
    anchor.addEventListener('mouseleave', hide);
    anchor.addEventListener('focusin', show);
    anchor.addEventListener('focusout', hide);
  }
}
initTips();

// ---- peak profile: what sets a peak's region apart ---------------------------
// For the hovered or selected peak of the main panel: every dimension (original coordinates of
// a point cloud, x, y, z of a volume, and the value) as a small histogram of the region against
// all points, sorted by the effect size |d| (see profile.js).

/** The columns of the current document, loading a point cloud's coordinates the first time. */
function profileData() {
  const doc = state.doc, entry = state.entry;
  if (!doc?.region || !doc.data?.values) return null;
  if (prof.doc !== doc) {
    Object.assign(prof, { doc, coords: null, cols: null, note: '', loading: null, vis: null, cache: new Map() });
  }
  if (prof.cols) return prof;
  const cloud = doc.meta.kind !== 'grid';
  if (cloud && entry?.coords_file && !prof.coords) {
    if (!prof.loading) {
      prof.loading = cachedJSON(entry.coords_file).then((c) => {
        if (prof.doc !== doc) return;
        prof.coords = c;
        renderProfile(prof.last);
      }).catch((err) => {
        console.error(err);
        if (prof.doc !== doc) return;
        prof.coords = false;                       // build without coordinates
        renderProfile(prof.last);
      });
    }
    return 'loading';
  }
  try {
    ({ columns: prof.cols, note: prof.note } = profileColumns(doc, prof.coords || null));
  } catch (err) {
    console.error(err);
    ({ columns: prof.cols, note: prof.note } = profileColumns(doc, null));
  }
  return prof;
}

/** A number of a profile row, with values that are zero up to rounding shown as 0. */
function fz(x, r) { return Math.abs(x) < 1e-9 * Math.max(1e-30, Math.abs(r.hi - r.lo)) ? '0' : fmt(x, 3); }

/** A row's two histograms (grey: all points, blue: this region) and a tick at the region's median. */
function profileSvg(r) {
  const W = 160, H = 26, bw = W / PROFILE_BINS;
  const amax = Math.max(...r.all) || 1, rmax = Math.max(...r.hist) || 1;
  let area = `M0 ${H}`;
  r.all.forEach((c, i) => { const y = H - (c / amax) * (H - 2); area += ` L${(i * bw).toFixed(1)} ${y.toFixed(1)} L${((i + 1) * bw).toFixed(1)} ${y.toFixed(1)}`; });
  area += ` L${W} ${H} Z`;
  const bars = Array.from(r.hist, (c, i) => {
    if (!c) return '';
    const h = (c / rmax) * (H - 2);
    return `<rect x="${(i * bw + 0.6).toFixed(1)}" y="${(H - h).toFixed(1)}" width="${(bw - 1.2).toFixed(1)}" height="${h.toFixed(1)}"/>`;
  }).join('');
  const span = r.hi - r.lo || 1;
  const mx = Math.min(W, Math.max(0, ((r.median - r.lo) / span) * W));
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">` +
    `<path d="${area}" fill="var(--muted)" opacity="0.28"/>` +
    `<g fill="var(--accent)" opacity="0.85">${bars}</g>` +
    `<line x1="${mx.toFixed(1)}" x2="${mx.toFixed(1)}" y1="0" y2="${H}" stroke="var(--text)" stroke-width="1.2"/></svg>`;
}

function renderProfile(node) {
  if (!el.profile) return;
  const vis = state.vis, doc = state.doc;
  // the peak: the one hovered or clicked; a hovered saddle keeps the selected peak's profile
  const sel = view.selected?.kind === 'peak' ? view.selected : null;
  const peak = node?.kind === 'peak' ? node : sel;
  prof.last = peak;
  if (!peak || !vis || !doc) {
    el.profile.innerHTML = '<p class="muted">Select a peak to see what sets its region apart.</p>';
    return;
  }
  const data = profileData();
  if (data === null) { el.profile.innerHTML = '<p class="muted">This file has no per-point regions to profile.</p>'; return; }
  if (data === 'loading') { el.profile.innerHTML = '<p class="muted">Loading the coordinates…</p>'; return; }
  if (prof.vis !== vis) { prof.vis = vis; prof.cache = new Map(); }
  let p = prof.cache.get(peak.index);
  if (!p) { p = peakProfile(prof.cols, doc, vis, peak.index); prof.cache.set(peak.index, p); }

  const show = el.profileShow?.value ?? '8';
  const rows = orderRows(p.rows, { sort: el.profileOrder?.checked ? 'order' : 'distinctive',
                                   limit: show === 'all' ? Infinity : Number(show) });
  const dims = p.rows.filter((r) => r.kind !== 'value').length;
  const pk = doc.peaks[peak.docId];
  // the peaks it is linked to, through which saddles (highest pass first)
  const links = [];
  vis.saddleEnds.forEach(([a, b], j) => {
    if (a !== peak.index && b !== peak.index) return;
    const o = a === peak.index ? b : a;
    if (o === peak.index) return;
    links.push([vis.peaks[o], doc.saddles[vis.saddles[j]].value]);
  });
  links.sort((x, y) => y[1] - x[1]);
  const linkText = links.length
    ? links.slice(0, 5).map(([id, v]) => `peak ${id} at ${fmt(v, 3)}`).join(', ') + (links.length > 5 ? `, +${links.length - 5} more` : '')
    : 'none';
  const absorbed = Array.from(vis.owner).filter((o) => o === peak.index).length - 1;
  const sum = `<p class="psum"><b>Peak ${peak.docId}</b> · value <b>${fmt(pk.value, 3)}</b>` +
    (isSimplifiable(doc) && pk.persistence !== null ? ` · persistence <b>${fmt(pk.persistence, 3)}</b>` : '') +
    `<br>${count(p.n)} points (${pct(p.n, p.total)})` + (absorbed > 0 ? ` · absorbed ${absorbed} smaller peak${absorbed > 1 ? 's' : ''}` : '') +
    `<br>linked to: ${linkText}` +
    (dims > rows.length - 1 ? `<br>showing ${rows.length - 1} of ${dims} dimensions` : '') +
    (prof.note ? `<br>${esc(prof.note)}` : '') + '</p>';
  const body = rows.map((r) => {
    const strong = Math.abs(r.d) >= 0.8;
    const narrow = r.spread < 0.5 ? ' <span class="pnarrow">narrow</span>' : '';
    return `<div class="prow"><span class="pname" title="${esc(r.label)}">${esc(r.label)}</span>${profileSvg(r)}` +
      `<span class="pstat"><span>median ${fz(r.median, r)} <span class="prange">(${fz(r.q05, r)} to ${fz(r.q95, r)})</span></span>` +
      `<span><span class="pd${strong ? ' strong' : ''}">d ${r.d >= 0 ? '+' : ''}${r.d.toFixed(2)}</span>${narrow}</span></span></div>`;
  }).join('');
  el.profile.innerHTML = sum + body;
}
el.profileShow?.addEventListener('change', () => renderProfile(prof.last));
el.profileOrder?.addEventListener('change', () => renderProfile(prof.last));

// ---- Scatter mode: two chosen dimensions against each other ------------------
// Like the scatterplot in NDDAV (and the thesis built on it): any two original dimensions of the
// data (x, y, z for a volume) or the value on the axes, coloured by peak region, value or a third
// dimension; drawn as dots or as a density. Linked to the spine like the data view. The columns
// are the peak profile's (same coordinates file, loaded once).
function ensureScatterView() {
  if (scatterView) return scatterView;
  try {
    scatterView = new DataView(el.scCanvas, { onPick: onPickPoint, onBrush: onScatterBrush }, { brushable: true });
    scatterView.setTheme({ bg: hexToRgb(cssColor('--bg')) });
  } catch (err) {
    console.error(err);
    setStatus('WebGL unavailable for the scatter plot', true);
    return null;
  }
  return scatterView;
}

const scColumn = (key) => prof.cols?.find((c) => c.key === key) ?? null;

async function loadScatter() {
  const doc = state.doc;
  if (!doc || !ensureScatterView()) return;
  let d = profileData();
  if (d === 'loading') {
    el.scNote.textContent = 'Loading the coordinates…';
    el.scNote.hidden = false;
    try { await prof.loading; } catch { /* handled in profileData */ }
    if (state.doc !== doc || state.mode !== 'scatter') return;
    d = profileData();
  }
  if (!d || d === 'loading' || !prof.cols?.length) {
    el.scNote.textContent = 'This file has no per-point data to plot.';
    el.scNote.hidden = false;
    return;
  }
  el.scNote.hidden = !prof.note;
  el.scNote.textContent = prof.note ?? '';
  if (sc.doc !== doc) {
    const coords = prof.cols.filter((c) => c.kind !== 'value'), value = prof.cols.find((c) => c.kind === 'value');
    const opts = [...coords, ...(value ? [value] : [])]
      .map((c) => `<option value="${esc(c.key)}">${esc(c.label)}</option>`).join('');
    el.scX.innerHTML = opts;
    el.scY.innerHTML = opts;
    el.scColorDims.innerHTML = coords.map((c) => `<option value="dim:${esc(c.key)}">${esc(c.label)}</option>`).join('');
    el.scX.value = (coords[0] ?? value).key;
    el.scY.value = (coords[1] ?? value ?? coords[0]).key;
    if (el.scColor.value.startsWith('dim:') || el.scColor.value === 'cluster') el.scColor.value = 'region';
    sc.doc = doc;
    scatterView.setPoints(doc, scatterXY());
    onScatterBrush(null);
  }
  scatterView.setVisible(state.vis);
  scatterView.setSelectedPeaks(pickedPeaks());
  applyScatter();
}

/** Positions of the points for the chosen X and Y columns, in [-1, 1]. */
function scatterXY() {
  const n = state.doc.meta.num_points, xy = new Float32Array(2 * n);
  const cx = scColumn(el.scX.value), cy = scColumn(el.scY.value);
  // a volume: voxels below "Show above" (mostly empty space) are left out, as in the 3D view
  const grid = state.doc.meta.kind === 'grid', cut = grid ? Number(el.volThresh.value) : -Infinity;
  const values = state.doc.data.values;
  for (let i = 0; i < n; i++) {
    if (values[i] < cut) { xy[2 * i] = xy[2 * i + 1] = 9; continue; }     // off the plot
    xy[2 * i] = cx ? 2 * cx.v[i] - 1 : 0;
    xy[2 * i + 1] = cy ? 2 * cy.v[i] - 1 : 0;
  }
  return xy;
}

/** Axes, colour and drawing of the scatter plot from the controls; the line above the plot. */
function applyScatter(moved = false) {
  if (!scatterView || sc.doc !== state.doc) return;
  if (moved) scatterView.setXY(scatterXY());
  const mode = el.scColor.value;
  if (mode === 'cluster') {
    scatterView.setClusters(clusterRes?.doc === state.doc ? clusterRes.labels : null);
    scatterView.setColorBy('cluster');
  } else if (mode.startsWith('dim:')) {
    const c = scColumn(mode.slice(4));
    scatterView.setCustomColors(c ? c.v : null);
    scatterView.setColorBy('custom');
  } else {
    scatterView.setColorBy(mode);
  }
  scatterView.setDraw(el.scDraw.value);
  const cx = scColumn(el.scX.value), cy = scColumn(el.scY.value);
  const range = (c) => c ? `<b>${esc(c.label)}</b> ${fmt(c.lo, 3)} to ${fmt(c.hi, 3)}` : '';
  const colourOf = mode === 'cluster' ? (clusterRes?.names ? clusterRes.label : `cluster (${clusterRes?.label ?? 'not run'})`) : mode === 'region' ? 'peak region'
    : mode === 'value' ? 'field value' : `<b>${esc(scColumn(mode.slice(4))?.label ?? '')}</b>`;
  const bar = mode === 'region' || mode === 'cluster' ? '' : ` <span class="sc-bar" style="background:${peakGradientCss()}"></span> low to high`;
  el.scInfo.innerHTML = `x: ${range(cx)} · y: ${range(cy)} · colour: ${colourOf}${bar}` +
    (el.scDraw.value === 'density' ? ' · density: brighter = more points' : '') +
    (state.doc.meta.kind === 'grid' ? ` · voxels below ${fmt(Number(el.volThresh.value), 3)} ("Show above") left out` : '');
}

/** A box brush in the scatter plot: the spine fades the peaks it leaves out; the line above says how many. */
function onScatterBrush(b) {
  brushes.scatter = b?.mask ?? null;
  const shares = applyBrushFocus();
  el.scClear.disabled = !b;
  if (!b) { el.scBrush.innerHTML = '<span>No brush: shift-drag a box to keep the points inside it; the spine fades the peaks it leaves out.</span>'; return; }
  const list = (shares ?? []).slice(0, 6).map((x) => `peak ${x.id} <b>${Math.round(100 * x.share)}%</b>`).join(', ');
  el.scBrush.innerHTML = `<span>Brush: <b>${count(b.count)}</b> of ${count(b.total)} points · ` +
    `share of each peak's region inside: ${list || 'none'}${(shares?.length ?? 0) > 6 ? ', …' : ''} · ` +
    `peaks under ${Math.round(100 * FOCUS_SHARE)}% fade in the spine</span>`;
}
el.scClear?.addEventListener('click', () => scatterView?.clearBrush());
el.scX?.addEventListener('change', () => applyScatter(true));
el.volThresh.addEventListener('input', () => { if (state.mode === 'scatter') applyScatter(true); });
el.scY?.addEventListener('change', () => applyScatter(true));
el.scColor?.addEventListener('change', () => applyScatter());
el.scDraw?.addEventListener('change', () => applyScatter());
el.scPeakDims?.addEventListener('click', () => {
  const sel = view.selected?.kind === 'peak' ? view.selected : null;
  if (!sel || !prof.cols) {
    el.scNote.textContent = 'Select a peak first (in the spine, or click a point): its two most distinctive dimensions go on the axes.';
    el.scNote.hidden = false;
    return;
  }
  if (prof.vis !== state.vis) { prof.vis = state.vis; prof.cache = new Map(); }
  let p = prof.cache.get(sel.index);
  if (!p) { p = peakProfile(prof.cols, state.doc, state.vis, sel.index); prof.cache.set(sel.index, p); }
  const top = orderRows(p.rows).filter((r) => r.kind !== 'value');
  if (top.length < 2) return;
  el.scX.value = top[0].key;
  el.scY.value = top[1].key;
  el.scNote.textContent = `Peak ${sel.docId}: ${top[0].label} (d ${top[0].d.toFixed(2)}) and ${top[1].label} (d ${top[1].d.toFixed(2)}) on the axes.`;
  el.scNote.hidden = false;
  applyScatter(true);
});

// ---- Clustering: KMeans / DBSCAN beside the peak regions ----------------------
// Point clouds only. Runs on the profile's normalised columns (all dimensions, [0, 1] each) or on
// the 2D positions of the data view. The result colours the data view and the scatter plot
// ('cluster') and is scored against the peak regions of the current threshold.
function syncClusterForm() {
  const db = el.clMethod.value === 'dbscan', lab = el.clMethod.value.startsWith('labels');
  el.clKRow.hidden = db || lab; el.clEpsRow.hidden = !db; el.clMinRow.hidden = !db;
  el.clSpace.closest('label').hidden = lab;
}

/** Known label sets of the points (coords file ``classes``: one set or a list of sets, e.g.
 *  tumour / normal, ER status, grade); [] when there are none. A value of -1 means unknown. */
function knownClasses() {
  const c = prof.doc === state.doc ? prof.coords?.classes : null;
  const sets = Array.isArray(c) ? c : c ? [c] : [];
  return sets.filter((s) => s.values?.length === state.doc?.meta.num_points);
}
const labelsIndex = (value) => (value.startsWith('labels') ? Number(value.split(':')[1] ?? 0) || 0 : -1);

/** Offer "Known labels" only when the dataset has them (loads the coordinates to find out). */
async function syncLabelOption() {
  const opt = $('clLabelsOpt');
  if (!opt) return;
  const doc = state.doc;
  const reset = () => { if (el.clMethod.value.startsWith('labels')) { el.clMethod.value = 'kmeans'; syncClusterForm(); } };
  for (const o of el.clMethod.querySelectorAll('option.labels-extra')) o.remove();
  opt.value = 'labels:0'; opt.disabled = true; opt.textContent = 'Known labels (from the data)';
  if (!doc || doc.meta.kind === 'grid' || !state.entry?.classes) { reset(); return; }
  let d = profileData();
  if (d === 'loading') { try { await prof.loading; } catch { /* none */ } d = profileData(); }
  if (state.doc !== doc) return;
  const sets = knownClasses();
  if (!sets.length) { reset(); return; }
  const text = (c) => `Known labels: ${c.name} (${c.names.join(' / ')})`;
  opt.disabled = false; opt.textContent = text(sets[0]);
  let after = opt;
  sets.slice(1).forEach((c, i) => {
    const o = document.createElement('option');
    o.className = 'labels-extra'; o.value = `labels:${i + 1}`; o.textContent = text(c);
    after.after(o); after = o;
  });
}
el.clMethod?.addEventListener('change', syncClusterForm);

function clearClusters() {
  clusterRes = null;
  el.clClear.disabled = true;
  el.colorByCluster.disabled = el.scColorCluster.disabled = true;
  if (el.colorBy.value === 'cluster') { el.colorBy.value = 'region'; dataView.setColorBy('region'); }
  if (el.scColor.value === 'cluster') { el.scColor.value = 'region'; applyScatter(); }
  dataView.setClusters?.(null);
  scatterView?.setClusters?.(null);
  el.clResult.innerHTML = '<p class="muted">Not run yet (point clouds only).</p>';
  syncedKey = ''; syncDataSelection();
}

/** The agreement with the peak regions of the current threshold, and the table. */
function renderClusters() {
  const r = clusterRes, doc = state.doc, vis = state.vis;
  if (!r || r.doc !== doc || !vis) return;
  const peaksOf = Int32Array.from(doc.region, (g) => vis.peaks[vis.owner[g]]);
  const { ari, table } = agreement(r.labels, peaksOf);
  r.table = table;
  const k = table.filter((t) => t.cluster >= 0).length, noise = table.find((t) => t.cluster < 0);
  const rows = table.slice(0, 12).map((t) => {
    const sw = `rgb(${clusterColor(t.cluster).map((c) => Math.round(c * 255)).join(',')})`;
    const name = t.cluster < 0 ? (r.names ? 'unknown' : 'noise') : r.names?.[t.cluster] ? esc(r.names[t.cluster]) : `cluster ${t.cluster + 1}`;
    return `<div class="clrow"><i style="background:${sw}"></i>${name} · ${count(t.size)} points` +
      `<span class="muted">mostly peak ${t.peak} (${Math.round(100 * t.share)}%)</span></div>`;
  }).join('');
  el.clResult.innerHTML = `<p class="psum"><b>${esc(r.label)}</b>: ${k} cluster${k === 1 ? '' : 's'}` +
    (noise ? `, ${count(noise.size)} ${r.names ? 'unknown' : 'noise points'}` : '') + ` · ${vis.peaks.length} peaks shown<br>` +
    `agreement with the peak regions: <b>${ari.toFixed(2)}</b> (adjusted Rand index)</p>` + rows +
    (table.length > 12 ? `<p class="muted">+${table.length - 12} more</p>` : '');
}

async function runClustering() {
  const doc = state.doc;
  if (!doc || doc.meta.kind === 'grid') {
    el.clResult.innerHTML = '<p class="muted">Clustering is for point clouds (a volume\'s voxels are a grid).</p>';
    return;
  }
  let d = profileData();
  if (d === 'loading') { el.clResult.innerHTML = '<p class="muted">Loading the coordinates…</p>'; try { await prof.loading; } catch { /* below */ } d = profileData(); }
  if (!d || d === 'loading' || !prof.cols) { el.clResult.innerHTML = '<p class="muted">No coordinates to cluster.</p>'; return; }
  const n = doc.meta.num_points;
  if (el.clMethod.value.startsWith('labels')) {
    const c = knownClasses()[labelsIndex(el.clMethod.value)];
    if (!c) { el.clResult.innerHTML = '<p class="muted">This dataset has no known labels.</p>'; return; }
    clusterRes = { doc, labels: Int32Array.from(c.values), label: `Known labels: ${c.name}`, names: c.names, table: [] };
    showClusters(doc);
    return;
  }
  let cols, where;
  if (el.clSpace.value === 'view' && dataView.xy) {
    const xs = new Float32Array(n), ys = new Float32Array(n);
    for (let i = 0; i < n; i++) { xs[i] = (dataView.xy[2 * i] + 1) / 2; ys[i] = (dataView.xy[2 * i + 1] + 1) / 2; }
    cols = [xs, ys]; where = 'the 2D view';
  } else {
    cols = prof.cols.filter((c) => c.kind !== 'value').map((c) => c.v);
    where = `${cols.length} dimension${cols.length === 1 ? '' : 's'}`;
  }
  const method = el.clMethod.value;
  if (method === 'dbscan' && dbscanCost(n, cols.length) > 3e9) {
    el.clResult.innerHTML = `<p class="muted">DBSCAN on ${where} of ${count(n)} points would take too long here; ` +
      'choose "On: the 2D view", or KMeans.</p>';
    return;
  }
  el.clRun.disabled = true;
  el.clResult.innerHTML = '<p class="muted">Running…</p>';
  await new Promise((r) => setTimeout(r, 30));            // let the panel show it
  try {
    let res, label;
    if (method === 'dbscan') {
      const eps = Math.max(1e-4, Number(el.clEps.value) || 0.05), minPts = Math.max(2, Math.round(Number(el.clMin.value) || 8));
      res = dbscan(cols, eps, minPts);
      label = `DBSCAN, radius ${eps}, ${minPts} neighbours, on ${where}`;
    } else {
      const k = Math.max(2, Math.min(40, Math.round(Number(el.clK.value) || 5)));
      res = kmeans(cols, k, { seed: 1 });
      label = `KMeans, k = ${k}, on ${where}`;
    }
    if (state.doc !== doc) return;
    clusterRes = { doc, labels: res.labels, label, table: [] };
    showClusters(doc);
  } catch (err) {
    console.error(err);
    el.clResult.innerHTML = `<p class="muted">Clustering failed: ${esc(err.message ?? String(err))}</p>`;
  } finally {
    el.clRun.disabled = false;
  }
}

/** Colour the views by ``clusterRes`` and show its agreement with the peaks. */
function showClusters(doc) {
  {
    const labels = clusterRes.labels;
    el.clClear.disabled = false;
    el.colorByCluster.disabled = el.scColorCluster.disabled = false;
    el.colorByCluster.textContent = el.scColorCluster.textContent = clusterRes.names ? 'known labels' : 'cluster';
    dataView.setClusters?.(labels);
    if (scatterView && sc.doc === doc) scatterView.setClusters(labels);
    renderClusters();
    if (el.colorBy.value !== 'cluster' && el.scColor.value !== 'cluster') {
      el.colorBy.value = 'cluster';                         // show them right away
      dataView.setColorBy('cluster');
    }
    if (state.mode === 'scatter') { el.scColor.value = 'cluster'; applyScatter(); }
    syncedKey = ''; syncDataSelection();
  }
}
el.clRun?.addEventListener('click', runClustering);
el.clClear?.addEventListener('click', clearClusters);
syncClusterForm();
