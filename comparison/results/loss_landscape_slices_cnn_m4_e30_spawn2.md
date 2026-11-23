# Loss landscape, 2-D slices vs the full 3-D subspace (cnn_m4_e30_spawn2)

Full view: our extremum graph on the grid (49^3); τ = 0.005 (f = −log L scaled to the full view's range). Slices: 65×65 Freudenthal grids. A slice minimum is *true* when it lies within 0.1 (box scaled to the unit cube) of the minimum of the full basin it falls in, *displaced* otherwise.

## Full view: 4 minima, 6 saddles (3 close loops)

Minimum losses: 0.404, 0.411, 0.413, 0.421.

Method-free checks: the 4 models lie in **4 distinct basins**; full barrier ≤ straight-line barrier for 6 of 6 pairs.

Barriers between full minima (loss):

| | #0 | #1 | #2 | #3 |
|---|---|---|---|---|
| #0 | – | 0.440 | 0.441 | 0.443 |
| #1 | 0.440 | – | 0.441 | 0.443 |
| #2 | 0.441 | 0.441 | – | 0.443 |
| #3 | 0.443 | 0.443 | 0.443 | – |

## Slices

| slice | models in it | minima shown | true | displaced | split basins | full minima missed | loops | barrier excess (median / max) |
|---|---|---|---|---|---|---|---|---|
| models012 | 0, 1, 2 | 3 | 3 | 0 | 0 | 1 | 1 | +0.9% / +0.9% |
| models013 | 0, 1, 3 | 3 | 3 | 0 | 0 | 1 | 1 | +0.2% / +0.2% |
| models023 | 0, 2, 3 | 3 | 3 | 0 | 0 | 1 | 1 | -0.0% / +0.4% |
| models123 | 1, 2, 3 | 3 | 3 | 0 | 0 | 1 | 1 | +0.7% / +0.7% |
| li_model0 | 0 | 1 | – | – | – | – | 0 | – / – |
| li_model1 | 1 | 1 | – | – | – | – | 0 | – / – |
| li_model2 | 2 | 1 | – | – | – | – | 0 | – / – |
| li_model3 | 3 | 1 | – | – | – | – | 0 | – / – |

## Summary

* Planes through three models (**4**): each shows on average 3.00 of the 4 full minima (at most 3); 0 displaced minima in total; 0 split basins.
* Barriers seen in those planes vs the full graph's own estimate: median +0.2%, max +0.9% (over 12 slice pairs). The best plane for each pair of full minima: 0-1 +0.1%, 0-2 +0.2%, 0-3 -0.0%, 1-2 +0.0%, 1-3 +0.7%, 2-3 -0.0%; 0 of 6 pairs never appear together in any plane. The loss measured densely along the graph's own path is 0.0% (median) above the graph's estimate, so the method-free comparison is the path table below.
* Full graph: 6 saddles with 3 independent loops; the planes show 4 loop saddles in total.
* Li et al. random planes around each model (±1 filter-normalised units, 4): minima shown 1, 1, 1, 1; loss 0.41..42.1, 0.41..45.8, 0.42..38.5, 0.41..38.3.

## Barriers between models: slices vs a real path in the full subspace

*Path* = the full graph's minimax path between the two basins, with the loss evaluated densely along it: a real path, so the true barrier is at most its maximum. *Best slice* = the lowest barrier any plane containing both models shows. A positive last column means every plane overstates the true barrier by at least that much, whatever method computed the full view.

| models | straight line | full graph estimate | path (upper bound) | best slice | best slice vs path |
|---|---|---|---|---|---|
| 0–1 | 0.4411 | 0.4397 | 0.4397 | 0.4399 | +0.1% |
| 0–2 | 0.4485 | 0.4434 | 0.4435 | 0.4434 | -0.0% |
| 0–3 | 0.4441 | 0.4409 | 0.4411 | 0.4419 | +0.2% |
| 1–2 | 0.4495 | 0.4434 | 0.4435 | 0.4467 | +0.7% |
| 1–3 | 0.4428 | 0.4409 | 0.4411 | 0.4410 | -0.0% |
| 2–3 | 0.4673 | 0.4434 | 0.4435 | 0.4434 | -0.0% |

Best slice vs path over 6 model pairs: median +0.0%, min -0.0%, max +0.7%; the path is lower in 3 of 6.
