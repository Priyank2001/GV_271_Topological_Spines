// The 'fig1' spine style draws at the layout positions: it never moves a peak, keeps a saddle of many
// peaks where the layout put it, and draws an ordinary saddle on the line between its two peaks.
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

const load = (name) => {
  const p = fileURLToPath(new URL(`../data/${name}.json`, import.meta.url));
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p)) : null;
};

function drawn(doc, style, threshold) {
  const canvas = { style: {}, clientWidth: 800, clientHeight: 600, addEventListener() {},
                   getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
                   setPointerCapture() {}, remove() {} };
  const renderer = { domElement: canvas, setPixelRatio() {}, setSize() {}, setClearColor() {}, render() {}, dispose() {} };
  const view = new SpineView({ clientWidth: 800, clientHeight: 600, appendChild() {} }, {}, { renderer });
  view.setGraph(doc);
  view.setStyle(style);
  const vis = simplify(doc, threshold);
  view.setVisible(vis);
  return { view, vis };
}

const star = load('star_saddle_2d');
test('fig1: the 10 peaks of star_saddle_2d stay put around one saddle', { skip: !star && 'no star_saddle_2d.json' }, () => {
  const a = drawn(star, 'fig1', 0.02);
  assert.equal(a.vis.peaks.length, 10);
  const layout = (id) => a.view.relayoutInfo?.peaks.get(id) ?? a.view.basePeakXY[id];
  for (const id of a.vis.peaks) assert.deepEqual(a.view.peakXY[id], layout(id));          // no peak moved
  const hub = a.view.saddleXY[a.vis.saddles[0]];
  for (const id of a.vis.saddles) assert.deepEqual(a.view.saddleXY[id], hub);              // one saddle
});

const volcano = load('volcano_intact');
test('fig1: an ordinary saddle is drawn on the line between its two peaks', { skip: !volcano && 'no volcano_intact.json' }, () => {
  const { view, vis } = drawn(volcano, 'fig1', 0.02);
  assert.ok(vis.saddles.length >= 6);
  vis.saddles.forEach((sid, j) => {
    const [a, b] = vis.saddleEnds[j].map((i) => view.peakXY[vis.peaks[i]]);
    const s = view.saddleXY[sid];
    const cross = (b[0] - a[0]) * (s[1] - a[1]) - (b[1] - a[1]) * (s[0] - a[0]);
    const t = ((s[0] - a[0]) * (b[0] - a[0]) + (s[1] - a[1]) * (b[1] - a[1])) / ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);
    assert.ok(Math.abs(cross) < 1e-9 && t > 0 && t < 1, `saddle ${sid} off the segment (t = ${t})`);
  });
});

test('every style draws the rows of one saddle (same vertex) at one point', { skip: !star && 'no star_saddle_2d.json' }, () => {
  for (const style of ['simple', 'nddav', 'paper', 'fig1']) {
    const { view, vis } = drawn(star, style, 0.02);
    const pts = new Set(vis.saddles.map((id) => view.saddleXY[id].join(',')));
    assert.equal(pts.size, 1, `${style}: ${pts.size} saddle positions`);
  }
});
