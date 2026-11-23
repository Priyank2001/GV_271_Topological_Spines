# Loss landscape, 2-D slices vs the full 5-D subspace (cnn_m6_e30)

Full view: our extremum graph on the knn (65536 Sobol samples, k = 30); τ = 0.005 (f = −log L scaled to the full view's range). Slices: 65×65 Freudenthal grids. A slice minimum is *true* when it lies within 0.1 (box scaled to the unit cube) of the minimum of the full basin it falls in, *displaced* otherwise.

## Full view: 7 minima, 21 saddles (15 close loops)

Minimum losses: 0.433, 0.457, 0.469, 0.470, 0.484, 0.485, 0.525.

Method-free checks: the 6 models lie in **6 distinct basins**; full barrier ≤ straight-line barrier for 15 of 15 pairs.

Barriers between full minima (loss):

| | #0 | #1 | #2 | #3 | #4 | #5 | #6 |
|---|---|---|---|---|---|---|---|
| #0 | – | 1.926 | 2.103 | 2.079 | 2.093 | 2.166 | 2.103 |
| #1 | 1.926 | – | 2.103 | 2.079 | 2.093 | 2.166 | 2.103 |
| #2 | 2.103 | 2.103 | – | 2.103 | 2.103 | 2.166 | 0.552 |
| #3 | 2.079 | 2.079 | 2.103 | – | 2.093 | 2.166 | 2.103 |
| #4 | 2.093 | 2.093 | 2.103 | 2.093 | – | 2.166 | 2.103 |
| #5 | 2.166 | 2.166 | 2.166 | 2.166 | 2.166 | – | 2.166 |
| #6 | 2.103 | 2.103 | 0.552 | 2.103 | 2.103 | 2.166 | – |

## Slices

| slice | models in it | minima shown | true | displaced | split basins | full minima missed | loops | barrier excess (median / max) |
|---|---|---|---|---|---|---|---|---|
| models012 | 0, 1, 2 | 3 | 3 | 0 | 0 | 4 | 1 | +15.2% / +16.5% |
| models013 | 0, 1, 3 | 3 | 3 | 0 | 0 | 4 | 1 | +8.8% / +11.5% |
| models014 | 0, 1, 4 | 3 | 3 | 0 | 0 | 4 | 1 | +16.6% / +16.6% |
| models015 | 0, 1, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +11.9% / +12.6% |
| models023 | 0, 2, 3 | 3 | 3 | 0 | 0 | 4 | 1 | +7.9% / +9.9% |
| models024 | 0, 2, 4 | 3 | 3 | 0 | 0 | 4 | 1 | +12.5% / +13.8% |
| models025 | 0, 2, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +7.8% / +7.8% |
| models034 | 0, 3, 4 | 3 | 3 | 0 | 0 | 4 | 1 | +17.5% / +22.4% |
| models035 | 0, 3, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +7.2% / +9.2% |
| models045 | 0, 4, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +12.8% / +13.5% |
| models123 | 1, 2, 3 | 3 | 3 | 0 | 0 | 4 | 0 | +12.8% / +16.2% |
| models124 | 1, 2, 4 | 3 | 3 | 0 | 0 | 4 | 0 | +8.4% / +9.5% |
| models125 | 1, 2, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +10.8% / +10.8% |
| models134 | 1, 3, 4 | 3 | 3 | 0 | 0 | 4 | 1 | +11.5% / +11.5% |
| models135 | 1, 3, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +4.7% / +7.5% |
| models145 | 1, 4, 5 | 3 | 3 | 0 | 0 | 4 | 1 | +3.5% / +8.8% |
| models234 | 2, 3, 4 | 3 | 3 | 0 | 0 | 4 | 1 | +9.1% / +12.4% |
| models235 | 2, 3, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +7.3% / +8.9% |
| models245 | 2, 4, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +11.2% / +11.2% |
| models345 | 3, 4, 5 | 3 | 3 | 0 | 0 | 4 | 0 | +9.4% / +9.4% |
| li_model0 | 0 | 1 | – | – | – | – | 0 | – / – |
| li_model1 | 1 | 1 | – | – | – | – | 0 | – / – |
| li_model2 | 2 | 1 | – | – | – | – | 0 | – / – |
| li_model3 | 3 | 1 | – | – | – | – | 0 | – / – |
| li_model4 | 4 | 1 | – | – | – | – | 0 | – / – |
| li_model5 | 5 | 1 | – | – | – | – | 0 | – / – |

## Summary

* Planes through three models (**20**): each shows on average 3.00 of the 7 full minima (at most 3); 0 displaced minima in total; 0 split basins.
* Barriers seen in those planes vs the full graph's own estimate: median +9.3%, max +22.4% (over 60 slice pairs). The best plane for each pair of full minima: 0-1 +5.8%, 0-2 +8.4%, 0-3 +13.5%, 0-4 +3.5%, 0-5 +9.1%, 1-2 +8.4%, 1-3 +11.5%, 1-4 +3.5%, 1-5 +4.7%, 2-3 +7.8%, 2-4 +7.8%, 2-5 +5.0%, 3-4 +4.5%, 3-5 +7.2%, 4-5 +4.7%; 6 of 21 pairs never appear together in any plane. The loss measured densely along the graph's own path is 5.6% (median) above the graph's estimate, so the method-free comparison is the path table below.
* Full graph: 21 saddles with 15 independent loops; the planes show 9 loop saddles in total.
* Li et al. random planes around each model (±1 filter-normalised units, 6): minima shown 1, 1, 1, 1, 1, 1; loss 0.42..28.2, 0.41..29.3, 0.41..25.6, 0.43..38.2, 0.41..41.4, 0.42..37.2.

## Minima of the full view that no model sits in

Checked by gradient descent in the subspace (finite differences, no topology method) from the minimum's sample.

| minimum | loss at the sample | loss after descent | moved (unit box) | nearest model (distance) | stays in its own basin |
|---|---|---|---|---|---|
| #6 | 0.5253 | 0.4119 | 0.111 | 2 (0.015) | no |

## Full view: robustness to k

| k | minima | models in distinct basins | weakest kept persistence |
|---|---|---|---|
| 15 | 12 | 6 of 6 | 0.005 |
| 30 | 7 | 6 of 6 | 0.007 |
| 50 | 7 | 6 of 6 | 0.007 |

## Barriers between models: slices vs a real path in the full subspace

*Path* = the full graph's minimax path between the two basins, with the loss evaluated densely along it: a real path, so the true barrier is at most its maximum. *Best slice* = the lowest barrier any plane containing both models shows. A positive last column means every plane overstates the true barrier by at least that much, whatever method computed the full view.

| models | straight line | full graph estimate | path (upper bound) | best slice | best slice vs path |
|---|---|---|---|---|---|
| 0–1 | 2.8338 | 2.0792 | 2.1706 | 2.3186 | +6.8% |
| 0–2 | 2.4545 | 2.1025 | 2.2175 | 2.2656 | +2.2% |
| 0–3 | 2.5969 | 2.1663 | 2.2895 | 2.3214 | +1.4% |
| 0–4 | 2.6514 | 2.0792 | 2.1706 | 2.3602 | +8.7% |
| 0–5 | 2.4340 | 2.0931 | 2.2103 | 2.1874 | -1.0% |
| 1–2 | 2.6363 | 2.1025 | 2.2175 | 2.2788 | +2.8% |
| 1–3 | 2.4956 | 2.1663 | 2.2895 | 2.2671 | -1.0% |
| 1–4 | 2.3636 | 1.9257 | 2.0439 | 2.0368 | -0.3% |
| 1–5 | 2.3438 | 2.0931 | 2.2103 | 2.1669 | -2.0% |
| 2–3 | 2.5085 | 2.1663 | 2.2895 | 2.2741 | -0.7% |
| 2–4 | 2.6577 | 2.1025 | 2.2175 | 2.2788 | +2.8% |
| 2–5 | 2.4156 | 2.1025 | 2.2175 | 2.2656 | +2.2% |
| 3–4 | 2.6496 | 2.1663 | 2.2895 | 2.3634 | +3.2% |
| 3–5 | 2.4042 | 2.1663 | 2.2895 | 2.2671 | -1.0% |
| 4–5 | 2.3915 | 2.0931 | 2.2103 | 2.1669 | -2.0% |

Best slice vs path over 15 model pairs: median +1.4%, min -2.0%, max +8.7%; the path is lower in 8 of 15.
