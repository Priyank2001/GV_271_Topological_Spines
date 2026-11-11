// Tests the scene logic of SpineView without a GPU: a fake renderer records
// what would be drawn, and fake pointer events drive hover, click, pan and zoom.
// Run from the repo root:  node --import ./web/tests/register.mjs --test web/tests
import test from 'node:test';
import assert from 'node:assert/strict';

// Minimal browser stand-ins, installed before the view module is loaded.
globalThis.window = { devicePixelRatio: 1 };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = () => 0;      // no render loop: tests call render by hand

const { SpineView, peakColor } = await import('../src/spineView.js');
const { simplify } = await import('../src/simplify.js');

const W = 800, H = 600;

function fakeCanvas() {
  const listeners = {};
  return {
    style: {}, clientWidth: W, clientHeight: H, listeners,
    addEventListener(type, fn) { (listeners[type] ??= []).push(fn); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: W, height: H }),
    setPointerCapture() {}, remove() {},
    fire(type, ev = {}) { for (const fn of listeners[type] ?? []) fn({ pointerId: 1, preventDefault() {}, ...ev }); },
  };
}

function setup() {
  const canvas = fakeCanvas();
  const renderer = { domElement: canvas, setPixelRatio() {}, setSize() {}, setClearColor() {},
                     render() {}, dispose() {} };
  const container = { clientWidth: W, clientHeight: H, appendChild() {} };
  const events = { hover: [], select: [] };
  const view = new SpineView(container,
    { onHover: (n) => events.hover.push(n), onSelect: (n) => events.select.push(n) }, { renderer });
  return { view, canvas, events };
}

// Three peaks (0 is the global maximum) and two saddles, hand-placed.
function toyDoc() {
  const peaks = [
    { id: 0, vertex: 0, value: 1.0, persistence: null, parent: -1, size: 600 },
    { id: 1, vertex: 1, value: 0.8, persistence: 0.5, parent: 0, size: 300 },
    { id: 2, vertex: 2, value: 0.4, persistence: 0.02, parent: 1, size: 100 },
  ];
  const saddles = [
    { id: 0, vertex: 3, value: 0.3, peaks: [0, 1], persistence: 0.5, is_merge: true, support: 4, relief: 1 },
    { id: 1, vertex: 4, value: 0.38, peaks: [1, 2], persistence: 0.02, is_merge: true, support: 2, relief: 1 },
  ];
  return {
    version: 1, meta: { num_points: 1000, simplify_threshold: 0 }, peaks, saddles,
    spine: { peaks: [[-0.8, 0], [0.2, 0.5], [0.8, -0.6]], saddles: [[-0.3, 0.25], [0.5, -0.05]] },
  };
}

// World position -> pixel position for the default camera (center 0, halfHeight 1.25).
function toPixel(view, [x, y]) {
  const halfW = view.halfHeight * (W / H);
  return { clientX: ((x - view.center.x) / halfW * 0.5 + 0.5) * W,
           clientY: (0.5 - (y - view.center.y) / view.halfHeight * 0.5) * H };
}

test('peak colour scale runs from dark purple to yellow', () => {
  const [r0, g0, b0] = peakColor(0), [r1, g1, b1] = peakColor(1);
  assert.ok(b0 > r0 && r1 > b1 && g1 > g0);
  assert.deepEqual(peakColor(-5), peakColor(0));
  assert.deepEqual(peakColor(9), peakColor(1));
});

test('instances sit at the layout positions and radius follows sqrt(region size)', () => {
  const { view } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));

  assert.equal(view.peakMesh.count, 3);
  assert.equal(view.saddleMesh.count, 2);
  const e = view.peakMesh.instanceMatrix.array;           // column-major 4x4 per instance
  doc.spine.peaks.forEach(([x, y], i) => {
    assert.ok(Math.abs(e[i * 16 + 12] - x) < 1e-6 && Math.abs(e[i * 16 + 13] - y) < 1e-6);
    assert.ok(Math.abs(e[i * 16] - view.radius[i]) < 1e-6);     // x scale = radius
  });
  assert.ok(Math.abs(view.radius[0] - view.rMax) < 1e-12);       // biggest region gets rMax
  const share = (view.radius[1] - view.rMin) / (view.rMax - view.rMin);
  assert.ok(Math.abs(share - Math.sqrt(300 / 600)) < 1e-9);
  assert.ok(view.peakMesh.instanceColor, 'peaks are coloured per instance');
});

test('each visible saddle has two links to its peaks', () => {
  const { view } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));
  const pos = view.linkGeom.attributes.position.array;
  assert.equal(view.linkGeom.drawRange.count, 2 * 4);            // 2 saddles x 2 links x 2 vertices
  // saddle 0 -> peak 0, then peak 1
  assert.deepEqual(Array.from(pos.slice(0, 6)), [-0.8, 0, 0, -0.3, 0.25, 0].map(Math.fround));
  assert.deepEqual(Array.from(pos.slice(6, 12)).map((v) => +v.toFixed(4)), [0.2, 0.5, 0, -0.3, 0.25, 0]);
});

test('raising the threshold hides nodes and drops links without moving the rest', () => {
  const { view } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));
  view.setVisible(simplify(doc, 0.1));                            // peak 2 and saddle 1 go
  assert.equal(view.peakMesh.count, 2);
  assert.equal(view.saddleMesh.count, 1);
  assert.equal(view.linkGeom.drawRange.count, 4);
  const e = view.peakMesh.instanceMatrix.array;
  assert.ok(Math.abs(e[16 + 12] - 0.2) < 1e-6);                   // peak 1 did not move
  assert.deepEqual(simplify(doc, 0.1).size, [600, 400]);          // peak 2's region joined peak 1
  const want = view.rMin + (view.rMax - view.rMin) * Math.sqrt(400 / 600);
  assert.ok(Math.abs(view.radius[1] - want) < 1e-9);
});

test('hover and click pick the node under the pointer, peaks before saddles', () => {
  const { view, canvas, events } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));

  canvas.fire('pointermove', toPixel(view, doc.spine.peaks[1]));
  assert.deepEqual(events.hover.at(-1), { kind: 'peak', index: 1, docId: 1 });
  assert.equal(view.hoverRing.visible, true);

  canvas.fire('pointermove', toPixel(view, doc.spine.saddles[0]));
  assert.deepEqual(events.hover.at(-1), { kind: 'saddle', index: 0, docId: 0 });

  canvas.fire('pointermove', toPixel(view, [0.0, -1.0]));         // empty space
  assert.equal(events.hover.at(-1), null);

  const px = toPixel(view, doc.spine.peaks[2]);
  canvas.fire('pointerdown', px);
  canvas.fire('pointerup', px);
  assert.deepEqual(view.selected, { kind: 'peak', index: 2, docId: 2 });
  assert.equal(view.selectRing.visible, true);
  // the two links of peak 2's only saddle are highlighted
  assert.equal(view.highlightGeom.drawRange.count, 2 * 2);

  const empty = toPixel(view, [0.0, -1.0]);
  canvas.fire('pointerdown', empty);
  canvas.fire('pointerup', empty);
  assert.equal(view.selected, null);
  assert.equal(view.selectRing.visible, false);
});

test('a selection survives the slider only while its node is still drawn', () => {
  const { view } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));

  view.select({ kind: 'peak', index: 1 });
  view.setVisible(simplify(doc, 0.1));                            // peak 2 removed; peak 1 stays at index 1
  assert.deepEqual(view.selected, { kind: 'peak', index: 1, docId: 1 });

  view.setVisible(simplify(doc, 0));
  view.select({ kind: 'peak', index: 2 });
  view.setVisible(simplify(doc, 0.1));
  assert.equal(view.selected, null);
  assert.equal(view.selectRing.visible, false);
});

test('dragging pans, a drag is not a click, and the wheel zooms about the cursor', () => {
  const { view, canvas, events } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));

  const start = toPixel(view, [0, 0]);
  canvas.fire('pointerdown', start);
  canvas.fire('pointermove', { clientX: start.clientX + 100, clientY: start.clientY });
  canvas.fire('pointerup', { clientX: start.clientX + 100, clientY: start.clientY });
  assert.ok(view.center.x < 0, 'content followed the pointer to the right');
  assert.equal(events.select.length, 0, 'a drag must not select');

  view.fit();
  const cursor = { clientX: 600, clientY: 200 };
  const world = (v) => {
    const a = v.halfHeight * (W / H);
    return [v.center.x + ((cursor.clientX / W) * 2 - 1) * a, v.center.y + (1 - (cursor.clientY / H) * 2) * v.halfHeight];
  };
  const before = world(view), h0 = view.halfHeight;
  canvas.fire('wheel', { ...cursor, deltaY: -400 });
  assert.ok(view.halfHeight < h0, 'zoomed in');
  const after = world(view);
  assert.ok(Math.abs(before[0] - after[0]) < 1e-9 && Math.abs(before[1] - after[1]) < 1e-9,
            'the point under the cursor stays put');
  canvas.fire('wheel', { ...cursor, deltaY: 1e6 });
  assert.ok(view.halfHeight <= view.homeHalfHeight * 4 + 1e-9, 'zoom out is bounded');
  canvas.fire('wheel', { ...cursor, deltaY: -1e6 });
  assert.ok(view.halfHeight >= 0.04 - 1e-12, 'zoom in is bounded');
});

test('switching to another graph resets the view and selection', () => {
  const { view } = setup();
  const doc = toyDoc();
  view.setGraph(doc);
  view.setVisible(simplify(doc, 0));
  view.select({ kind: 'peak', index: 0 });
  view.setGraph(doc);
  assert.equal(view.selected, null);
  view.setVisible(simplify(doc, 0));
  assert.equal(view.peakMesh.count, 3);
});

test('a document without a spine is refused', () => {
  const { view } = setup();
  const doc = toyDoc();
  delete doc.spine;
  assert.throws(() => view.setGraph(doc), /no spine/);
});
