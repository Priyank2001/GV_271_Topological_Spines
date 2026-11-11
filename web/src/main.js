// Wires the page together: dataset list, spine view, slider, persistence plot, info panel,
// and the modes chosen in the rail on the left: explore (spine + data view), compare
// (a second spine with another extraction method, see compare.js) and parallel
// coordinates (spine + every point as a line across its original dimensions).

import { maxPersistence, maxSaddlePersistence } from './simplify.js';
import { visibleGraph, isSimplifiable, compareShown, traitDifferences, findAgreement, findVisible,
         allMaxima, sharedColorIds } from './compare.js';
import { ComparePanel } from './comparePanel.js';
import { ComparisonBox } from './comparisonBox.js';
import { SpineView, peakGradientCss, hexToRgb, regionColor, setColormap, getColormap } from './spineView.js';
import { DataView } from './dataView.js';
import { VolumeView, RAMP, bandColor } from './volumeView.js';
import { PersistencePlot, sliderToThreshold, thresholdToSlider } from './persistencePlot.js';
import { ParallelView } from './parallelView.js';

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
  volIsoLevels: $('volIsoLevels'), volIsoSurf: $('volIsoSurf'), volIsoCmap: $('volIsoCmap'), volIsoRev: $('volIsoRev'), embedding: $('embedding'), colorBy: $('colorBy'),
  style: $('style'), scale: $('scale'), scaleOut: $('scaleOut'), scaleRow: $('paperRows'),
  radiusLaw: $('radiusLaw'), gap: $('gap'), gapOut: $('gapOut'), scaleMode: $('scaleMode'),
  gapRow: $('gapRow'), levels: $('levels'), levelsOut: $('levelsOut'), levelsRow: $('levelsRow'),
  colormap: $('colormap'), cmapReverse: $('cmapReverse'), cmapPreview: $('cmapPreview'),
  styleNote: $('styleNote'), discLegend: $('discLegend'), arcs: $('arcs'),
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
const state = { doc: null, index: [], tmax: 1, base: 0, threshold: 0, variation: Infinity, vis: null,
                entry: null, variantA: null, variantB: null, compare: false, sliderDoc: null, traitLabels: {},
                mode: 'explore' };

const fmt = (x, digits = 4) => (x === null || x === undefined ? '–' : Number(x.toPrecision(digits)).toString());
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (a, b) => `${((100 * a) / b).toFixed(a / b < 0.01 ? 2 : 1)} %`;
const count = (n) => n.toLocaleString('en-US');

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
    return new SpineView(el.spine, { onHover, onSelect, onCamera: (c) => linkCamera('A', c) });
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

/** Every spine view on screen (the second one exists once compare mode has been opened). */
function spineViews() { return panelB.view ? [view, panelB.view] : [view]; }

/** Give a newly created spine view the current drawing settings. */
function applyViewSettings(v) {
  v.setTheme(themeColors());
  v.setStyle?.(el.style.value);
  v.setShowArcs?.(el.arcs.checked);
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
  state.vis = visibleGraph(doc, t, v);                               // the full graph if doc cannot simplify
  view.setVisible(state.vis);
  dataView.setVisible(state.vis);
  volumeView.setVisible(state.vis);
  if (pcReady()) pcView.setVisible(state.vis);
  if (state.compare && panelB.doc) panelB.setThresholds(t, v);
  lastHover = null;                    // hover indices refer to the previous visible lists
  syncDataSelection();

  el.slider.value = String(thresholdToSlider(t, state.tmax));
  el.thresholdOut.textContent = state.sliderDoc ? fmt(t, 3) : '–';
  el.variation.min = String(thresholdToSlider(t, state.tmax));
  el.variation.value = String(Number.isFinite(v) ? thresholdToSlider(v, state.tmax) : 1);
  el.variationOut.textContent = !state.sliderDoc ? '–' : Number.isFinite(v) ? fmt(v, 3) : 'off';
  plot.setThreshold(t);
  plot.setVariation(v);
  const ridges = el.arcs.checked && state.vis.ridges.length
    ? `; ${count(state.vis.ridges.length)} more arcs kept by p* through ${count(state.vis.ghosts.length)} merged peaks` : '';
  el.counts.textContent = isSimplifiable(doc)
    ? `${count(state.vis.peaks.length)} of ${count(doc.peaks.length)} peaks, ` +
      `${count(state.vis.saddles.length)} of ${count(doc.saddles.length)} saddles shown${ridges}`
    : `All ${count(doc.peaks.length)} peaks and ${count(doc.saddles.length)} saddle links shown ` +
      `(${variantLabel(doc)} has no simplification).`;
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
    : sdoc ? `The sliders act on ${variantLabel(sdoc)} only. ${fixed.join(' and ')} has no simplification, ` +
             'so its full graph is always shown.'
           : `${fixed.join(' and ')} has no simplification: its full graph is shown and the sliders are off.`;
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

function onSelect() { renderInfo(view.selected ?? lastHover); syncDataSelection(); linkSelection('A'); }

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
  const bandLabel = el.volIsoCmap.value === 'spine' ? cmapLabel
    : `${el.volIsoCmap.value}${el.volIsoRev.checked ? ', reversed' : ''}`;
  el.clTitle.textContent = byIso ? `Colour: iso-value bands (${levels.length + 1} bands, ${bandLabel})`
    : byValue ? `Colour: field value (${cmapLabel})` : 'Colour: peak region';
  el.clBar.hidden = el.clRange.hidden = !byValue && !volume;
  el.clLo.textContent = fmt(lo, 3);
  el.clHi.textContent = fmt(hi, 3);
  if (byIso) {                                                       // hard stops at the iso values
    const n = levels.length + 1, edges = [lo, ...levels, hi];
    const rgb = (k) => `rgb(${bandColor(k, n, el.volIsoCmap.value, el.volIsoRev.checked).map((c) => Math.round(c * 255)).join(',')})`;
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
function applyBandColormap() {
  volumeView.setBandColormap?.(el.volIsoCmap.value, el.volIsoRev.checked);
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
  el.gapRow.hidden = style !== 'paper';
  el.levelsRow.hidden = style !== 'nddav';
  const active = Boolean(state.doc?.data?.values && state.doc?.region);   // what the contour styles need
  el.styleNote.textContent = !contours
    ? 'The extremum graph: a disc per maximum, a diamond per saddle and a line for every link a saddle makes. ' +
      'Press Spine (key s) to draw the topological spine, the nested contours, around it.'
    : !active
    ? 'This file has no per-point values, so discs are drawn instead.'
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
  el.discLegend.textContent = contours
    ? (law === 'nddav' ? 'contour radius grows as (points of the peak\'s region it encloses)^(2/d)'
      : `contour ${law === '1' ? 'radius' : 'area'} is proportional to the points of the peak's region it encloses`) +
      (el.scaleMode.value === 'saddle' ? ' (the pale outer halo is capped at 2× the widest saddle-level contour)' : '') +
      '; the dot is the maximum'
    : 'disc area is proportional to the points in the peak\'s region';
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
    volumeView.setBandColormap?.(el.volIsoCmap.value, el.volIsoRev.checked);   // the browser may restore the form
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
  if (!doc.data && main.data) doc.data = main.data;
  return doc;
}

function fillVariantMenus(entry) {
  for (const select of [el.variantA, el.variantB]) {
    select.replaceChildren(...availableVariants(entry).map((v) => {
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
  el.variantA.value = state.variantA;
  el.variantB.value = state.variantB;
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
    const volume = volumeView.setGraph(doc);
    showDataPanel(doc, names, volume);
    state.loads = (state.loads ?? 0) + 1;
    syncedKey = '';
    setThreshold(initialThreshold());
    if (state.mode === 'parallel') loadParallel();
    if (state.compare && panelB.doc) panelB.view.setColorIds?.(sharedColorIds(doc, panelB.doc));
    applyStyle();
    setStatus(`${count(doc.meta.num_points)} points · ${doc.meta.dimensions}D`);
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
el.variantA.addEventListener('change', async () => { state.variantA = el.variantA.value; await showVariantA(); });
el.variantB.addEventListener('change', async () => { state.variantB = el.variantB.value; await showVariantB(); });
el.swapVariants.addEventListener('click', async () => {
  [state.variantA, state.variantB] = [state.variantB, state.variantA];
  el.variantA.value = state.variantA;
  el.variantB.value = state.variantB;
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
  el.parallel.hidden = state.mode !== 'parallel';
  el.data.hidden = state.mode === 'parallel';
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
  syncedKey = ''; syncDataSelection();
}

for (const b of el.modeButtons) b.addEventListener('click', () => setMode(b.dataset.mode));
window.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, select, textarea')) return;
  if (e.key.toLowerCase() === 'p') setMode(state.mode === 'parallel' ? 'explore' : 'parallel');
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

/** The brush summary above the plot: how many points pass, and in which peaks (click one to select it). */
function renderBrush(summary) {
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
