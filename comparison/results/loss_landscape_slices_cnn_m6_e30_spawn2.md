# Loss landscape, 2-D slices vs the full 5-D subspace (cnn_m6_e30_spawn2)

Full view: our extremum graph on the knn (65536 Sobol samples, k = 30); τ = 0.005 (f = −log L scaled to the full view's range). Slices: 65×65 Freudenthal grids. A slice minimum is *true* when it lies within 0.1 (box scaled to the unit cube) of the minimum of the full basin it falls in, *displaced* otherwise.

## Full view: 6 minima, 15 saddles (10 close loops)

Minimum losses: 0.406, 0.407, 0.411, 0.413, 0.415, 0.420.

Method-free checks: the 6 models lie in **6 distinct basins**; full barrier ≤ straight-line barrier for 12 of 15 pairs (**violations:** [0, 4] 0.4277 > 0.4277, [1, 2] 0.4350 > 0.4349, [3, 5] 0.4374 > 0.4369).

Barriers between full minima (loss):

| | #0 | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|---|
| #0 | – | 0.422 | 0.437 | 0.428 | 0.432 | 0.435 |
| #1 | 0.422 | – | 0.437 | 0.428 | 0.432 | 0.435 |
| #2 | 0.437 | 0.437 | – | 0.437 | 0.437 | 0.437 |
| #3 | 0.428 | 0.428 | 0.437 | – | 0.432 | 0.435 |
| #4 | 0.432 | 0.432 | 0.437 | 0.432 | – | 0.435 |
| #5 | 0.435 | 0.435 | 0.437 | 0.435 | 0.435 | – |

## Slices

| slice | models in it | minima shown | true | displaced | split basins | full minima missed | loops | barrier excess (median / max) |
|---|---|---|---|---|---|---|---|---|
| models012 | 0, 1, 2 | 3 | 3 | 0 | 0 | 3 | 1 | -0.2% / -0.2% |
| models013 | 0, 1, 3 | 3 | 3 | 0 | 0 | 3 | 1 | +1.1% / +1.1% |
| models014 | 0, 1, 4 | 3 | 3 | 0 | 0 | 3 | 1 | +1.0% / +1.0% |
| models015 | 0, 1, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +0.8% / +0.8% |
| models023 | 0, 2, 3 | 3 | 3 | 0 | 0 | 3 | 1 | +0.3% / +0.3% |
| models024 | 0, 2, 4 | 3 | 3 | 0 | 0 | 3 | 1 | -0.1% / +0.6% |
| models025 | 0, 2, 5 | 3 | 3 | 0 | 0 | 3 | 1 | -0.0% / +0.1% |
| models034 | 0, 3, 4 | 3 | 3 | 0 | 0 | 3 | 1 | +1.5% / +1.5% |
| models035 | 0, 3, 5 | 3 | 3 | 0 | 0 | 3 | 1 | -0.3% / +0.0% |
| models045 | 0, 4, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +0.4% / +0.4% |
| models123 | 1, 2, 3 | 3 | 3 | 0 | 0 | 3 | 1 | +0.4% / +0.4% |
| models124 | 1, 2, 4 | 3 | 3 | 0 | 0 | 3 | 1 | -0.0% / +1.0% |
| models125 | 1, 2, 5 | 3 | 3 | 0 | 0 | 3 | 1 | -0.1% / -0.1% |
| models134 | 1, 3, 4 | 3 | 3 | 0 | 0 | 3 | 1 | +1.1% / +1.3% |
| models135 | 1, 3, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +1.2% / +1.8% |
| models145 | 1, 4, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +1.5% / +2.4% |
| models234 | 2, 3, 4 | 3 | 3 | 0 | 0 | 3 | 1 | +0.3% / +0.8% |
| models235 | 2, 3, 5 | 3 | 3 | 0 | 0 | 3 | 1 | -0.2% / +0.0% |
| models245 | 2, 4, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +0.1% / +1.0% |
| models345 | 3, 4, 5 | 3 | 3 | 0 | 0 | 3 | 1 | +1.1% / +2.5% |
| li_model0 | 0 | 1 | – | – | – | – | 0 | – / – |
| li_model1 | 1 | 1 | – | – | – | – | 0 | – / – |
| li_model2 | 2 | 1 | – | – | – | – | 0 | – / – |
| li_model3 | 3 | 1 | – | – | – | – | 0 | – / – |
| li_model4 | 4 | 1 | – | – | – | – | 0 | – / – |
| li_model5 | 5 | 1 | – | – | – | – | 0 | – / – |

## Summary

* Planes through three models (**20**): each shows on average 3.00 of the 6 full minima (at most 3); 0 displaced minima in total; 0 split basins.
* Barriers seen in those planes vs the full graph's own estimate: median +0.3%, max +2.5% (over 60 slice pairs). The best plane for each pair of full minima: 0-1 -0.3%, 0-2 -0.2%, 0-3 -0.1%, 0-4 -0.2%, 0-5 -0.2%, 1-2 -0.3%, 1-3 -0.4%, 1-4 -0.0%, 1-5 -0.2%, 2-3 +0.3%, 2-4 -0.3%, 2-5 +0.4%, 3-4 +0.1%, 3-5 -0.0%, 4-5 -0.1%; 0 of 15 pairs never appear together in any plane. The loss measured densely along the graph's own path is 0.0% (median) above the graph's estimate, so the method-free comparison is the path table below.
* Full graph: 15 saddles with 10 independent loops; the planes show 20 loop saddles in total.
* Li et al. random planes around each model (±1 filter-normalised units, 6): minima shown 1, 1, 1, 1, 1, 1; loss 0.41..48.4, 0.42..33.6, 0.40..38.1, 0.41..32.7, 0.41..37.9, 0.41..35.8.

## Full view: robustness to k

| k | minima | models in distinct basins | weakest kept persistence |
|---|---|---|---|
| 15 | 6 | 6 of 6 | 0.011 |
| 30 | 6 | 6 of 6 | 0.011 |
| 50 | 6 | 6 of 6 | 0.010 |

## Barriers between models: slices vs a real path in the full subspace

*Path* = the full graph's minimax path between the two basins, with the loss evaluated densely along it: a real path, so the true barrier is at most its maximum. *Best slice* = the lowest barrier any plane containing both models shows. A positive last column means every plane overstates the true barrier by at least that much, whatever method computed the full view.

| models | straight line | full graph estimate | path (upper bound) | best slice | best slice vs path |
|---|---|---|---|---|---|
| 0–1 | 0.4392 | 0.4350 | 0.4350 | 0.4341 | -0.2% |
| 0–2 | 0.4248 | 0.4219 | 0.4220 | 0.4207 | -0.3% |
| 0–3 | 0.4441 | 0.4374 | 0.4374 | 0.4360 | -0.3% |
| 0–4 | 0.4277 | 0.4277 | 0.4279 | 0.4259 | -0.5% |
| 0–5 | 0.4334 | 0.4315 | 0.4315 | 0.4315 | -0.0% |
| 1–2 | 0.4349 | 0.4350 | 0.4350 | 0.4341 | -0.2% |
| 1–3 | 0.4426 | 0.4374 | 0.4374 | 0.4392 | +0.4% |
| 1–4 | 0.4411 | 0.4350 | 0.4350 | 0.4349 | -0.0% |
| 1–5 | 0.4444 | 0.4350 | 0.4350 | 0.4347 | -0.1% |
| 2–3 | 0.4397 | 0.4374 | 0.4374 | 0.4365 | -0.2% |
| 2–4 | 0.4321 | 0.4277 | 0.4279 | 0.4273 | -0.1% |
| 2–5 | 0.4317 | 0.4315 | 0.4315 | 0.4307 | -0.2% |
| 3–4 | 0.4466 | 0.4374 | 0.4374 | 0.4385 | +0.3% |
| 3–5 | 0.4369 | 0.4374 | 0.4374 | 0.4360 | -0.3% |
| 4–5 | 0.4424 | 0.4315 | 0.4315 | 0.4319 | +0.1% |

Best slice vs path over 15 model pairs: median -0.2%, min -0.5%, max +0.4%; the path is lower in 3 of 15.
