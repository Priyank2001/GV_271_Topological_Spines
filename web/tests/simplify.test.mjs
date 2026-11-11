// Run from the repo root:  node --test web/tests
// Checks src/simplify.js against what Python's ExtremumGraph.simplify returns
// (fixtures/simplify_cases.json, made by make_fixtures.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { simplify, maxPersistence, arcPath } from '../src/simplify.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/simplify_cases.json', import.meta.url)));

for (const c of fixture.cases) {
  fixture.thresholds.forEach((t, i) => {
    test(`${c.name}: threshold ${t} matches Python`, () => {
      const want = c.expected[i];
      const got = simplify(c.doc, t);
      assert.deepEqual(got.peaks.map((i) => c.doc.peaks[i].vertex), want.peaks);
      assert.deepEqual(got.saddles.map((i) => c.doc.saddles[i].vertex), want.saddles);
      assert.deepEqual(got.saddleEnds, want.saddleEnds);
      assert.deepEqual(got.size, want.size);
      assert.deepEqual(got.arcs.map((i) => c.doc.saddles[i].vertex), want.arcs);
    });
  });
}

for (const c of fixture.cases) {
  for (const v of c.variation) {
    test(`${c.name}: p_lo ${v.threshold}, p_hi ${v.variation} matches Python`, () => {
      const got = simplify(c.doc, v.threshold, v.variation);
      assert.deepEqual(got.peaks.map((i) => c.doc.peaks[i].vertex), v.peaks);
      assert.deepEqual(got.saddles.map((i) => c.doc.saddles[i].vertex), v.saddles);
      assert.deepEqual(got.saddleEnds, v.saddleEnds);
      assert.deepEqual(got.size, v.size);
      assert.deepEqual(got.arcs.map((i) => c.doc.saddles[i].vertex), v.arcs);
    });
  }
}

test('every document peak is owned by exactly one surviving peak', () => {
  const c = fixture.cases[0];
  const got = simplify(c.doc, 0.2);
  assert.equal(got.owner.length, c.doc.peaks.length);
  assert.ok(got.owner.every((o) => o >= 0 && o < got.peaks.length));
  assert.equal(got.size.reduce((a, b) => a + b, 0), c.doc.meta.num_points);
});

test('maxPersistence is the largest finite value, also from the curve', () => {
  assert.equal(maxPersistence({ peaks: [{ persistence: null }, { persistence: 0.3 }] }), 0.3);
  assert.equal(
    maxPersistence({ peaks: [{ persistence: null }], persistence_curve: [0.9, 0.1] }), 0.9);
});

test('arcs lead from every visible link, through hidden peaks, to the peaks it joins', () => {
  for (const c of fixture.cases) {
    for (const t of fixture.thresholds) {
      const vis = simplify(c.doc, t);
      vis.saddles.forEach((sid, j) => {
        c.doc.saddles[sid].peaks.forEach((p) => {
          const hidden = vis.peaks[vis.owner[p]] !== p;
          assert.ok(!hidden || arcPath(c.doc, vis, p).length > 0, `${c.name} t=${t} saddle ${sid}`);
        });
      });
    }
  }
});
