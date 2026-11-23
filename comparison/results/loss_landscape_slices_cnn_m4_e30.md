# Loss landscape, 2-D slices vs the full 3-D subspace (cnn_m4_e30)

Full view: our extremum graph on the grid (49^3); τ = 0.005 (f = −log L scaled to the full view's range). Slices: 65×65 Freudenthal grids. A slice minimum is *true* when it lies within 0.1 (box scaled to the unit cube) of the minimum of the full basin it falls in, *displaced* otherwise.

## Full view: 4 minima, 6 saddles (3 close loops)

Minimum losses: 0.409, 0.422, 0.423, 0.427.

Method-free checks: the 4 models lie in **4 distinct basins**; full barrier ≤ straight-line barrier for 6 of 6 pairs.

Barriers between full minima (loss):

| | #0 | #1 | #2 | #3 |
|---|---|---|---|---|
| #0 | – | 2.358 | 2.334 | 2.358 |
| #1 | 2.358 | – | 2.358 | 2.221 |
| #2 | 2.334 | 2.358 | – | 2.358 |
| #3 | 2.358 | 2.221 | 2.358 | – |

## Slices

| slice | models in it | minima shown | true | displaced | split basins | full minima missed | loops | barrier excess (median / max) |
|---|---|---|---|---|---|---|---|---|
| models012 | 0, 1, 2 | 3 | 3 | 0 | 0 | 1 | 1 | +7.0% / +7.0% |
| models013 | 0, 1, 3 | 3 | 3 | 0 | 0 | 1 | 1 | +6.9% / +8.1% |
| models023 | 0, 2, 3 | 3 | 3 | 0 | 0 | 1 | 1 | +3.7% / +3.7% |
| models123 | 1, 2, 3 | 3 | 3 | 0 | 0 | 1 | 1 | +5.2% / +5.2% |
| li_model0 | 0 | 1 | – | – | – | – | 0 | – / – |
| li_model1 | 1 | 1 | – | – | – | – | 0 | – / – |
| li_model2 | 2 | 1 | – | – | – | – | 0 | – / – |
| li_model3 | 3 | 1 | – | – | – | – | 0 | – / – |

## Summary

* Planes through three models (**4**): each shows on average 3.00 of the 4 full minima (at most 3); 0 displaced minima in total; 0 split basins.
* Barriers seen in those planes vs the full graph's own estimate: median +5.2%, max +8.1% (over 12 slice pairs). The best plane for each pair of full minima: 0-1 +3.7%, 0-2 +1.9%, 0-3 +5.2%, 1-2 +3.7%, 1-3 +1.2%, 2-3 +5.2%; 0 of 6 pairs never appear together in any plane. The loss measured densely along the graph's own path is 0.3% (median) above the graph's estimate, so the method-free comparison is the path table below.
* Full graph: 6 saddles with 3 independent loops; the planes show 4 loop saddles in total.
* Li et al. random planes around each model (±1 filter-normalised units, 4): minima shown 1, 1, 1, 1; loss 0.42..46.5, 0.43..31.7, 0.42..37.7, 0.41..33.7.

## Barriers between models: slices vs a real path in the full subspace

*Path* = the full graph's minimax path between the two basins, with the loss evaluated densely along it: a real path, so the true barrier is at most its maximum. *Best slice* = the lowest barrier any plane containing both models shows. A positive last column means every plane overstates the true barrier by at least that much, whatever method computed the full view.

| models | straight line | full graph estimate | path (upper bound) | best slice | best slice vs path |
|---|---|---|---|---|---|
| 0–1 | 2.5453 | 2.2209 | 2.2209 | 2.2474 | +1.2% |
| 0–2 | 2.5694 | 2.3577 | 2.3658 | 2.4442 | +3.3% |
| 0–3 | 2.6512 | 2.3577 | 2.3658 | 2.4442 | +3.3% |
| 1–2 | 2.6013 | 2.3577 | 2.3658 | 2.4793 | +4.8% |
| 1–3 | 2.5342 | 2.3577 | 2.3658 | 2.4793 | +4.8% |
| 2–3 | 2.4962 | 2.3337 | 2.3340 | 2.3774 | +1.9% |

Best slice vs path over 6 model pairs: median +3.3%, min +1.2%, max +4.8%; the path is lower in 6 of 6.
