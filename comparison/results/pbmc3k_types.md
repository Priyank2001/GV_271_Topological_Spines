# pbmc3k_types: what the spine shows

2638 blood cells; kNN graph (k = 15) on 20 principal components; f = density; persistence threshold 0.04. Cell types are the reference labels (not used to build anything).

| Peak = mostly | Cells | Largest type | Persistence | Top genes (higher than the rest) | Found in 12 settings |
|---|---|---|---|---|---|
| CD4 T + CD8 T | 1473 | 77% | highest | CD3D, IL32, CD3E, LDHB, LTB | 100% |
| CD14 monocyte | 487 | 94% | 0.388 | S100A9, LYZ, S100A8, CST3, TYROBP | 100% |
| B | 342 | 98% | 0.246 | HLA-DRA, CD74, CD79A, HLA-DPB1, CD79B | 92% |
| FCGR3A monocyte | 171 | 96% | 0.056 | LST1, AIF1, FCER1G, CST3, TYROBP | 67% |
| NK | 140 | 98% | 0.069 | GNLY, NKG7, GZMB, PRF1, FGFBP2 | 67% |
| Dendritic | 25 | 96% | 0.115 | CST3, HLA-DRA, HLA-DPB1, LYZ, HLA-DRB1 | 83% |

Saddles (highest first: the two types that grade into each other most easily):

* CD14 monocyte -- FCGR3A monocyte: saddle at 0.81 (merge)
* CD4 T + CD8 T -- NK: saddle at 0.65 (merge)
* CD4 T + CD8 T -- B: saddle at 0.64 (merge)
* CD14 monocyte -- Dendritic: saddle at 0.60 (merge)
* FCGR3A monocyte -- Dendritic: saddle at 0.51 (loop)
* CD4 T + CD8 T -- CD14 monocyte: saddle at 0.50 (merge)
* B -- Dendritic: saddle at 0.49 (loop)
* CD4 T + CD8 T -- Dendritic: saddle at 0.48 (loop)
* CD4 T + CD8 T -- FCGR3A monocyte: saddle at 0.16 (loop)

Stability: settings k in (10, 15, 20, 30) x threshold in (0.02, 0.04, 0.08); a peak is found when some peak there shares at least half of its cells (Jaccard >= 0.5).
