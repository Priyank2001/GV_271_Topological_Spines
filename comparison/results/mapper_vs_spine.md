# Mapper vs extremum graph on bc_dsga (lens / f = deviation from normal)

Spine (k = 15, smoothed twice, peaks above 4% of the range): 7 peaks.

| Spine peak | Tumours | Persistence (share of range) | Genes most raised | Genes most lowered |
|---|---|---|---|---|
| 0 | 112 | global maximum | CCND1, THSD4, COX6C, FAM77C, CST9L | EHF, GSTP1, FUT3 |
| 1 | 119 | 0.167 | FAM79B, SERPINA11, PGR, FMO5, RGS22 | C1orf106, RARRES1, CCL13 |
| 2 | 97 | 0.160 | FOXC1, ROPN1, ROPN1B, SRrp35, VGLL1 | FOXA1, MLPH, SPDEF |
| 3 | 85 | 0.078 | AMZ1, ANXA9, LRFN2, OSR2, HS6ST3 | GPRIN2, SLC5A1, SLPI |
| 4 | 57 | 0.077 | CLCA2, AKR1B10, GLYATL2, MUCL1, ATP13A4 | BCL2, COX6C, RLN2 |
| 5 | 23 | 0.061 | CLCA2, PNMT, POF1B, ATP8A2, NR2F1 | TFCP2L1, DCDC2, FAM79B |
| 6 | 36 | 0.054 | FOXN4, ROPN1L, CCNO, RNF183, TTC29 | TM4SF1, TRIM2, ZIC1 |

## Spine stability (k in 10, 15, 20, 30; smoothing 2 or 3 rounds: 8 settings)

| Spine peak | Median best overlap (Jaccard) | Lowest | Settings with overlap >= 0.5 |
|---|---|---|---|
| 0 | 0.52 | 0.31 | 4/8 |
| 1 | 0.68 | 0.54 | 8/8 |
| 2 | 0.90 | 0.66 | 8/8 |
| 3 | 0.45 | 0.30 | 2/8 |
| 4 | 0.52 | 0.26 | 4/8 |
| 5 | 0.32 | 0.07 | 2/8 |
| 6 | 0.17 | 0.08 | 1/8 |

## Mapper (KeplerMapper) over 24 settings

| Lens | Intervals | Overlap | eps | Nodes | Pieces | Branch tips | Spine peaks the tips fall in | Tip purity |
|---|---|---|---|---|---|---|---|---|
| raw deviation | 10 | 0.3 | q0.5 | 13 | 1 | 5 | [0, 1, 2, 4] | 0.81 |
| raw deviation | 10 | 0.3 | q0.7 | 12 | 1 | 4 | [2, 4, 6] | 0.78 |
| raw deviation | 10 | 0.5 | q0.5 | 15 | 1 | 8 | [1, 2, 3, 4] | 0.71 |
| raw deviation | 10 | 0.5 | q0.7 | 14 | 1 | 6 | [0, 1, 2, 4, 6] | 0.77 |
| raw deviation | 15 | 0.3 | q0.5 | 19 | 3 | 7 | [0, 1, 2, 3] | 0.74 |
| raw deviation | 15 | 0.3 | q0.7 | 20 | 4 | 8 | [0, 1, 2, 4, 5, 6] | 0.78 |
| raw deviation | 15 | 0.5 | q0.5 | 23 | 2 | 10 | [1, 2, 3, 4, 6] | 0.79 |
| raw deviation | 15 | 0.5 | q0.7 | 18 | 1 | 5 | [1, 2, 4] | 0.71 |
| raw deviation | 20 | 0.3 | q0.5 | 21 | 7 | 9 | [0, 1, 2, 3, 4, 6] | 0.68 |
| raw deviation | 20 | 0.3 | q0.7 | 23 | 4 | 8 | [0, 1, 2, 4, 5, 6] | 0.71 |
| raw deviation | 20 | 0.5 | q0.5 | 23 | 1 | 6 | [1, 2, 3, 4] | 0.80 |
| raw deviation | 20 | 0.5 | q0.7 | 27 | 1 | 8 | [2, 3, 4, 6] | 0.85 |
| smoothed deviation | 10 | 0.3 | q0.5 | 15 | 2 | 6 | [0, 1, 2, 4] | 0.87 |
| smoothed deviation | 10 | 0.3 | q0.7 | 14 | 2 | 4 | [0, 2, 4] | 0.82 |
| smoothed deviation | 10 | 0.5 | q0.5 | 19 | 1 | 7 | [0, 1, 2, 3, 4] | 0.90 |
| smoothed deviation | 10 | 0.5 | q0.7 | 13 | 1 | 3 | [0, 1, 2] | 0.83 |
| smoothed deviation | 15 | 0.3 | q0.5 | 24 | 7 | 9 | [0, 1, 2, 3, 4] | 0.87 |
| smoothed deviation | 15 | 0.3 | q0.7 | 18 | 2 | 3 | [1, 2] | 0.80 |
| smoothed deviation | 15 | 0.5 | q0.5 | 25 | 3 | 8 | [0, 1, 2, 4] | 0.88 |
| smoothed deviation | 15 | 0.5 | q0.7 | 20 | 1 | 4 | [0, 1, 2, 4] | 0.84 |
| smoothed deviation | 20 | 0.3 | q0.5 | 28 | 8 | 9 | [0, 1, 2, 3, 4] | 0.81 |
| smoothed deviation | 20 | 0.3 | q0.7 | 23 | 2 | 3 | [1, 2, 4] | 0.80 |
| smoothed deviation | 20 | 0.5 | q0.5 | 33 | 4 | 9 | [0, 1, 2, 3, 4] | 0.92 |
| smoothed deviation | 20 | 0.5 | q0.7 | 25 | 2 | 3 | [1, 2, 4] | 0.76 |

## Each spine peak: how many of the 24 Mapper settings have a branch tip in it

| Spine peak | Tumours | Settings |
|---|---|---|
| 0 | 112 | 15/24 |
| 1 | 119 | 21/24 |
| 2 | 97 | 24/24 |
| 3 | 85 | 10/24 |
| 4 | 57 | 21/24 |
| 5 | 23 | 2/24 |
| 6 | 36 | 7/24 |
