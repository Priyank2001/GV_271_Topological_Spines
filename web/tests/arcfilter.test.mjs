// AnyEG's arc height-difference filter (src/simplify.js arcFiltered).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { arcFiltered, arcHeights, maxArcHeight } from '../src/simplify.js';

// three maxima (values 1.0, 0.9, 0.5); saddle entries 0-1 at 0.8, 0-2 at 0.45, and 0-1 again at 0.85
const doc = {
  peaks: [{ value: 1.0, size: 10 }, { value: 0.9, size: 8 }, { value: 0.5, size: 3 }],
  saddles: [{ value: 0.8, peaks: [0, 1] }, { value: 0.45, peaks: [0, 2] }, { value: 0.85, peaks: [0, 1] }],
};

test('arc heights are |f(saddle) - f(peak)| for both peaks of an entry', () => {
  assert.deepEqual(arcHeights(doc, 0).map((x) => +x.toFixed(3)), [0.2, 0.1]);
  assert.deepEqual(arcHeights(doc, 1).map((x) => +x.toFixed(3)), [0.55, 0.05]);
  assert.equal(+maxArcHeight(doc).toFixed(3), 0.1);
});

test('off at 0: every link shown', () => {
  const v = arcFiltered(doc, 0);
  assert.deepEqual(v.saddles, [0, 1, 2]);
  assert.deepEqual(v.peaks, [0, 1, 2]);
});

test('a link stays only if both of its arcs reach the bound; maxima and regions never change', () => {
  const v = arcFiltered(doc, 0.06);
  assert.deepEqual(v.saddles, [0]);                 // 0.05 arc to peak 2 and 0.05 arc of entry 2 fall below
  assert.deepEqual(v.peaks, [0, 1, 2]);             // no maximum is removed
  assert.deepEqual([...v.owner], [0, 1, 2]);        // every peak keeps its own region
  assert.deepEqual(v.saddleEnds, [[0, 1]]);
});
