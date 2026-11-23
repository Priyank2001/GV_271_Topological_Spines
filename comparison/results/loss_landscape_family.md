# Loss landscape topology vs model quality (step 4)

Two settings. **A. Single model** (the papers' setting): a box of filter-normalised random directions around one model. **B. Several solutions** of one configuration (4 seeds): the 3-D subspace through them, as in steps 1–3, where basins, saddles and barriers between solutions exist.

## A. Single-model landscapes

23 SmallCNNs on CIFAR-10 (two seeds per setting). Landscape per model: training objective at θ + Σ aᵢdᵢ, a ∈ [−1, 1]^4, four filter-normalised random directions; 16,384 Sobol samples, our extremum graph on the kNN graph (k = 30); the 2-D column is the slice a₃ = a₄ = 0 on a 65² PL grid. Persistence in log-loss units (0.03 = a 3% loss ratio). Minima of the loss are the peaks of f = −log L.

Diverged during training (no landscape): w32_lr0.15_e30_s1.

## Spearman correlation with test accuracy (all models)

| measure | 4-D landscape | 2-D slice | simpler for better models would mean |
|---|---|---|---|
| minima@0.01 | constant | constant | negative |
| minima@0.03 | constant | constant | negative |
| minima@0.1 | constant | constant | negative |
| saddles@0.03 | constant | constant | negative |
| total_pers | +0.00 | -0.44 | negative |
| basin_frac | constant | constant | positive |
| escape | constant | constant | positive (deeper own basin) |
| smd | constant | constant | per Landscaper: larger = smoother |
| loss_mean (no topology) | +0.54 | +0.54 | negative (flatter) |

With 23 models, |ρ| ≥ 0.45 is roughly the 5% significance level (two-sided, no correction for testing several measures).

## Factor: epochs (mean over seeds)

| epochs | test acc | minima@0.01 (4-D / 2-D) | minima@0.03 | minima@0.1 | saddles@0.03 | total pers. | own basin share | escape | SMD | mean loss | minima@0.03 for k = 15/30/50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.551 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 2.9 | 1.0 / 1.0 / 1.0 |
| 2 | 0.632 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 3.7 | 1.0 / 1.0 / 1.0 |
| 4 | 0.694 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 6.0 | 1.0 / 1.0 / 1.0 |
| 8 | 0.754 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 8.8 | 1.0 / 1.0 / 1.0 |
| 16 | 0.803 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 12.4 | 1.5 / 1.0 / 1.0 |
| 30 | 0.855 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 17.4 | 1.5 / 1.0 / 1.0 |

## Factor: width (mean over seeds)

| width | test acc | minima@0.01 (4-D / 2-D) | minima@0.03 | minima@0.1 | saddles@0.03 | total pers. | own basin share | escape | SMD | mean loss | minima@0.03 for k = 15/30/50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.730 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 37.9 | 2.0 / 1.0 / 1.0 |
| 16 | 0.808 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 24.0 | 1.0 / 1.0 / 1.0 |
| 32 | 0.855 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 17.4 | 1.5 / 1.0 / 1.0 |
| 64 | 0.884 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 11.1 | 1.0 / 1.0 / 1.0 |

## Factor: learning rate (mean over seeds)

| learning rate | test acc | minima@0.01 (4-D / 2-D) | minima@0.03 | minima@0.1 | saddles@0.03 | total pers. | own basin share | escape | SMD | mean loss | minima@0.03 for k = 15/30/50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.005 | 0.791 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 11.4 | 2.0 / 1.0 / 1.0 |
| 0.02 | 0.848 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 14.9 | 1.5 / 1.0 / 1.0 |
| 0.05 | 0.855 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 17.4 | 1.5 / 1.0 / 1.0 |
| 0.15 | 0.792 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 / 0.0 | 0.00 / 0.00 | 1.00 / 1.00 | ∞ / ∞ | nan / nan | 16.1 | 1.0 / 1.0 / 1.0 |

## B. Subspace through four solutions of each configuration

For each (width, lr, epochs) with all four seeds: the 3-D affine subspace through the four trained models, 8,192 Sobol samples, our extremum graph on the kNN graph (k = 15). Barrier = highest loss on the path minus the mean loss of its two models (the usual definition), averaged over the 6 pairs: along the straight segment (linear mode connectivity), and along the full graph's lowest path, evaluated densely (a real path, so an upper bound on the true barrier). Persistence in log-loss units.

| configuration | test acc | models in distinct basins | graph minima@0.01 | real minima (descent) | loops@0.01 | model loss | barrier, straight | barrier, lowest path | top of lowest path (loss) | lowest / straight |
|---|---|---|---|---|---|---|---|---|---|---|
| w32_lr0.05_e1 | 0.542 | 4 of 4 | 6 | 4 | 9 | 1.300 | 1.136 | 0.588 | 1.889 | 0.52 |
| w32_lr0.05_e2 | 0.632 | 4 of 4 | 5 | 4 | 5 | 1.074 | 1.501 | 1.055 | 2.129 | 0.70 |
| w32_lr0.05_e4 | 0.689 | 4 of 4 | 5 | 4 | 5 | 0.956 | 1.762 | 1.189 | 2.145 | 0.67 |
| w32_lr0.05_e8 | 0.754 | 4 of 4 | 4 | 4 | 3 | 0.813 | 1.855 | 1.372 | 2.185 | 0.74 |
| w32_lr0.05_e16 | 0.806 | 4 of 4 | 4 | 4 | 3 | 0.632 | 1.998 | 1.678 | 2.310 | 0.84 |
| w32_lr0.05_e30 | 0.856 | 4 of 4 | 4 | 4 | 3 | 0.418 | 2.189 | 1.887 | 2.305 | 0.86 |
| w8_lr0.05_e30 | 0.729 | 4 of 4 | 4 | 4 | 3 | 0.793 | 1.702 | 1.564 | 2.357 | 0.92 |
| w16_lr0.05_e30 | 0.808 | 4 of 4 | 4 | 4 | 3 | 0.557 | 1.898 | 1.731 | 2.288 | 0.91 |
| w32_lr0.005_e30 | 0.791 | 4 of 4 | 4 | 4 | 3 | 0.563 | 1.640 | 1.468 | 2.031 | 0.90 |
| w32_lr0.02_e30 | 0.845 | 4 of 4 | 4 | 4 | 3 | 0.406 | 1.928 | 1.681 | 2.087 | 0.87 |
| w64_lr0.05_e30 | 0.882 | 4 of 4 | 4 | 4 | 3 | 0.344 | 2.185 | 1.796 | 2.141 | 0.82 |

Spearman correlation with test accuracy over the 11 configurations (|ρ| ≥ 0.62 ≈ 5% level):

* minima@0.01: ρ = -0.78
* verified_minima: constant (no correlation to compute)
* loops@0.01: ρ = -0.78
* minimum_loss_mean: ρ = -0.97
* barrier_straight_mean: ρ = +0.90
* barrier_low_mean: ρ = +0.95
* path_top_mean: ρ = +0.25
* low_vs_straight: ρ = +0.49

