# alanine_dipeptide: what the spine shows

5000 shapes from 30 ns of simulation; kNN graph (k = 15) on 45 heavy-atom distances. Persistence threshold: 0.08 of the range for density (the result is the same from 0.06 to 0.10), 1 kT for free energy (smaller barriers are below thermal noise). Regions are the Ramachandran labels (not used to build anything).

Time spent in each region: beta 14%, PPII 42%, alpha-R 42%, other 2%.

## f = density of shapes

| Peak = mostly | Shapes | Largest region | Persistence | Median phi, psi |
|---|---|---|---|---|
| alpha-R | 2185 | 97% | highest | -78, -13 |
| PPII | 2148 | 97% | 0.480 | -72, 151 |
| beta | 667 | 98% | 0.193 | -145, 154 |

Saddles (highest first: the easiest transitions):

* PPII -- beta: at 0.73 (merge)
* alpha-R -- PPII: at 0.51 (merge)
* alpha-R -- beta: at 0.34 (loop)

## f = -free energy from phi/psi

| Peak = mostly | Shapes | Largest region | Persistence (kT) | Median phi, psi |
|---|---|---|---|---|
| alpha-R | 2184 | 97% | 3.4 | -78, -13 |
| PPII | 2167 | 97% | highest | -72, 150 |
| beta | 649 | 99% | 1.0 | -146, 155 |

Saddles (highest first: the easiest transitions):

* PPII -- beta: at -5.6 kT (merge)
* PPII -- alpha-R: at -6.9 kT (merge)
* alpha-R -- beta: at -7.4 kT (loop)

In the viewer (f scaled to 0..1), 1 kT is a persistence of 0.194.

Not visited: alpha-L (phi > 0) is reached only on much longer time scales, so it has no peak here.
