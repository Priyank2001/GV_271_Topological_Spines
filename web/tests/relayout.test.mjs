// Re-layout after simplifying (src/relayout.js, SpineView.setRelayout).
// Run from the repo root:  node --import ./web/tests/register.mjs --test web/tests
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

globalThis.window = { devicePixelRatio: 1 };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = () => 0;

const { SpineView } = await import('../src/spineView.js');
const { simplify } = await import('../src/simplify.js');
const { relayout } = await import('../src/relayout.js');

const fuelPath = fileURLToPath(new URL('../data/fuel.json', import.meta.url));
const fuel = fs.existsSync(fuelPath) ? JSON.parse(fs.readFileSync(fuelPath)) : null;

function view() {
  const canvas = { style: {}, clientWidth: 800, clientHeight: 600, addEventListener() {},
                   getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
                   setPointerCapture() {}, remove() {} };
  const renderer = { domElement: canvas, setPixelRatio() {}, setSize() {}, setClearColor() {}, render() {}, dispose() {} };
  return new SpineView({ clientWidth: 800, clientHeight: 600, appendChild() {} }, {}, { renderer });
}

// Four points on a square around one centre, joined to it: the answer is a symmetric cross.
function crossDoc() {
  const P = [[0, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]];
  const S = [[0.5, 0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]];
  return {
    meta: {},
    peaks: P.map((p, i) => ({ id: i, value: i ? 0.5 : 1, persistence: i ? 0.4 : null, parent: i ? 0 : -1, size: 10, position: p })),
    saddles: S.map((p, i) => ({ id: i, value: 0.2, peaks: [0, i + 1], persistence: 0.3, saturation: 0.3, is_merge: true, position: p })),
    // a lopsided exported layout, as if made for a bigger graph
    spine: { scale: 1, peaks: [[0, 0], [1.2, 0.3], [0.9, -1.4], [-0.3, 1.1], [-1.6, -0.2]],
             saddles: [[0.6, 0.15], [0.45, -0.7], [-0.15, 0.55], [-0.8, -0.1]] },
  };
}

test('relayout fits the nodes to their own distances (toy cross)', () => {
  const doc = crossDoc();
  const r = relayout(doc, [0, 1, 2, 3, 4], [0, 1, 2, 3]);
  assert.ok(r.stressAfter < r.stressBefore * 0.1, `stress ${r.stressBefore} -> ${r.stressAfter}`);
  const c = r.peaks.get(0);
  const arms = [1, 2, 3, 4].map((p) => Math.hypot(r.peaks.get(p)[0] - c[0], r.peaks.get(p)[1] - c[1]));
  for (const a of arms) assert.ok(Math.abs(a - Math.SQRT2) < 0.05, `arm length ${a}`);
});

test('extra nodes are placed without moving the main ones', () => {
  const doc = crossDoc();
  const main = relayout(doc, [0, 1, 2, 3], [0, 1, 2]);
  const both = relayout(doc, [0, 1, 2, 3], [0, 1, 2], { extraPeaks: [4], extraSaddles: [3] });
  for (const id of [0, 1, 2, 3]) assert.deepEqual(both.peaks.get(id), main.peaks.get(id));
  const [x, y] = both.peaks.get(4), c = both.peaks.get(0);
  assert.ok(Math.abs(Math.hypot(x - c[0], y - c[1]) - Math.SQRT2) < 0.15, 'extra peak near its true distance');
});

test('relayout returns null when it cannot or need not run', () => {
  const doc = crossDoc();
  assert.equal(relayout(doc, [0, 1], []), null);                      // fewer than 3 nodes
  assert.equal(relayout(doc, [0, 1, 2, 3, 4], [0, 1, 2, 3], { maxNodes: 5 }), null);
  const noPos = crossDoc(); delete noPos.peaks[2].position;
  assert.equal(relayout(noPos, [0, 1, 2, 3, 4], [0, 1, 2, 3]), null);
});

test('SpineView swaps in the new layout and restores the exported one when off', () => {
  const doc = crossDoc();
  const v = view();
  v.setGraph(doc);
  v.setVisible(simplify(doc, 0));
  assert.notDeepEqual(v.peakXY[2], doc.spine.peaks[2]);
  v.setRelayout(false);
  assert.deepEqual(v.peakXY[2], doc.spine.peaks[2]);
  assert.equal(v.peakXY, doc.spine.peaks);
});

test('fuel at 0.08: the four head peaks form a symmetric cross around the jet tip', { skip: !fuel && 'no web/data/fuel.json' }, () => {
  for (const style of ['simple', 'paper', 'nddav']) {
    for (const arcs of [false, true]) {
      const v = view();
      v.setGraph(fuel);
      v.setStyle?.(style);
      v.setShowArcs(arcs);
      const vis = simplify(fuel, 0.08);
      v.setVisible(vis);
      assert.ok(v.relayoutInfo, `${style}/${arcs}: relayout ran`);
      assert.ok(v.relayoutInfo.stressAfter < v.relayoutInfo.stressBefore, `${style}/${arcs}: stress went down`);
      // Peaks by voxel position (x, y, z), not by id: ids change when the grid or saddles do.
      const at = (x, y, z) => fuel.peaks.findIndex((p) => p.position.join() === [x, y, z].join());
      const tip = v.peakXY[at(47, 31, 34)];
      const ang = [at(50, 40, 40), at(50, 23, 40), at(50, 40, 23), at(50, 23, 23)].map((p) => Math.atan2(v.peakXY[p][1] - tip[1], v.peakXY[p][0] - tip[0]) * 180 / Math.PI).sort((a, b) => a - b);
      // two arms on each side of the jet, mirror images within 15 degrees
      assert.ok(ang[0] < 0 && ang[1] < 0 && ang[2] > 0 && ang[3] > 0, `arms ${ang}`);
      assert.ok(Math.abs(ang[0] + ang[3]) < 15 && Math.abs(ang[1] + ang[2]) < 15, `arms ${ang}`);
    }
  }
});
