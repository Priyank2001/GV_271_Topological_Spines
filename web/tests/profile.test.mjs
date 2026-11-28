// Peak profile statistics (src/profile.js).
import test from 'node:test';
import assert from 'node:assert/strict';
import { profileColumns, peakProfile, orderRows, PROFILE_BINS } from '../src/profile.js';

// Six points in 2-D: peak 0 owns the first three (x1 high, x2 like everyone), peak 1 the rest.
function fixture() {
  const enc = (xs, lo, hi) => {                       // the coords file's uint16-base64 encoding
    let s = '';
    for (const x of xs) { const u = Math.round(((x - lo) / (hi - lo)) * 65535); s += String.fromCharCode(u & 255, u >> 8); }
    return btoa(s);
  };
  const x1 = [9, 10, 11, 0, 1, 2], x2 = [0, 5, 10, 0, 5, 10];
  const doc = {
    meta: { num_points: 6, kind: 'points', value_range: [0, 1] },
    data: { values: Float32Array.from([1, 0.9, 0.8, 0.5, 0.4, 0.3]) },
    region: [0, 0, 0, 1, 1, 1],
  };
  const coords = { num_points: 6, axes: [
    { name: 'x1', min: 0, max: 11, data: enc(x1, 0, 11) },
    { name: 'x2', min: 0, max: 10, data: enc(x2, 0, 10) },
  ] };
  const vis = { peaks: [0, 1], owner: Int32Array.from([0, 1]) };
  return { doc, coords, vis };
}

test('a peak region: its size, medians and ranges in the original units', () => {
  const { doc, coords, vis } = fixture();
  const { columns } = profileColumns(doc, coords);
  const p = peakProfile(columns, doc, vis, 0);
  assert.equal(p.n, 3);
  assert.equal(p.total, 6);
  const x1 = p.rows.find((r) => r.label === 'x1');
  assert.ok(Math.abs(x1.median - 10) < 0.01, `median ${x1.median}`);
  assert.ok(x1.q05 > 8.9 && x1.q95 < 11.01);
  assert.equal(x1.hist.length, PROFILE_BINS);
  assert.equal(x1.hist.reduce((a, b) => a + b, 0), 3);
});

test('the dimension that sets a peak apart comes first', () => {
  const { doc, coords, vis } = fixture();
  const { columns } = profileColumns(doc, coords);
  const p = peakProfile(columns, doc, vis, 0);
  const x1 = p.rows.find((r) => r.label === 'x1'), x2 = p.rows.find((r) => r.label === 'x2');
  assert.ok(x1.d > 0.9, `x1 effect size ${x1.d}`);          // peak 0 sits high on x1
  assert.ok(Math.abs(x2.d) < 0.01, `x2 effect size ${x2.d}`); // and like everyone on x2
  const shown = orderRows(p.rows);
  assert.equal(shown[0].kind, 'value');                       // the value always first
  assert.equal(shown[1].label, 'x1');
  assert.equal(orderRows(p.rows, { limit: 1 }).length, 2);    // value + 1 coordinate
  assert.deepEqual(orderRows(p.rows, { sort: 'order' }).map((r) => r.label).slice(1), ['x1', 'x2']);
});
