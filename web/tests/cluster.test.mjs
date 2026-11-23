// Clustering and agreement (src/cluster.js).
import test from 'node:test';
import assert from 'node:assert/strict';
import { kmeans, dbscan, agreement } from '../src/cluster.js';

// Three tight blobs in 2-D (30 points each) plus 3 far-off stray points.
function blobs() {
  const xs = [], ys = [], truth = [];
  const centres = [[0.2, 0.2], [0.8, 0.25], [0.5, 0.8]];
  centres.forEach(([cx, cy], c) => {
    for (let i = 0; i < 30; i++) {
      const a = (i / 30) * 2 * Math.PI, r = 0.02 + 0.01 * (i % 3);
      xs.push(cx + r * Math.cos(a)); ys.push(cy + r * Math.sin(a)); truth.push(c);
    }
  });
  for (const [x, y] of [[0.02, 0.98], [0.98, 0.98], [0.5, 0.45]]) { xs.push(x); ys.push(y); truth.push(3); }
  return { cols: [Float32Array.from(xs), Float32Array.from(ys)], truth: Int32Array.from(truth) };
}

test('KMeans finds three blobs', () => {
  const { cols, truth } = blobs();
  const { labels, k } = kmeans(cols, 3, { seed: 7 });
  assert.equal(k, 3);
  const blobOnly = (arr) => arr.slice(0, 90);
  const { ari } = agreement(blobOnly(labels), blobOnly(truth));
  assert.ok(ari > 0.99, `ARI ${ari}`);
});

test('DBSCAN finds the blobs and calls the strays noise', () => {
  const { cols, truth } = blobs();
  const { labels, k, noise } = dbscan(cols, 0.05, 4);
  assert.equal(k, 3);
  assert.equal(noise, 3);
  assert.deepEqual([...labels.slice(90)], [-1, -1, -1]);
  const { ari } = agreement(labels.slice(0, 90), truth.slice(0, 90));
  assert.ok(ari > 0.99, `ARI ${ari}`);
});

test('agreement: identical groupings score 1, the table names the main peak of each cluster', () => {
  const a = Int32Array.from([0, 0, 1, 1, 2, 2]), b = Int32Array.from([5, 5, 7, 7, 9, 9]);
  const { ari, table } = agreement(a, b);
  assert.ok(Math.abs(ari - 1) < 1e-12);
  assert.deepEqual(table.map((r) => [r.cluster, r.peak, r.share]), [[0, 5, 1], [1, 7, 1], [2, 9, 1]]);
  const mixed = agreement(Int32Array.from([0, 1, 0, 1, 0, 1]), b);   // splits every peak: worse than chance
  assert.ok(mixed.ari < 0.01, `ARI ${mixed.ari}`);
});
