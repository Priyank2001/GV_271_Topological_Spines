# domainnet: what the spine shows

5763 DomainNet images (10 classes x 6 domains); points: 512-D CLIP ViT-B/32 image features (unit length); kNN graph k = 15; f = density; persistence threshold 0.04. Class and domain are known labels, not used to build anything.

**27 peaks, 188 saddles.** Agreement of the peak regions with
the labels: class ARI 0.42 / NMI 0.62; domain ARI 0.10 / NMI 0.25; class x domain NMI 0.66.

| Mostly class | share | Mostly domain | share | Images | Zero-shot right | Persistence | Found in 12 settings |
|---|---|---|---|---|---|---|---|
| bicycle | 52% | infograph | 60% | 753 | 66% | 0.169 | 83% |
| guitar | 96% | painting + real | 23% | 461 | 90% | 0.108 | 92% |
| cat | 79% | quickdraw + sketch | 32% | 451 | 59% | 0.172 | 100% |
| bus | 98% | clipart + real | 26% | 368 | 52% | 0.152 | 100% |
| zebra | 93% | clipart + painting | 28% | 355 | 91% | 0.330 | 100% |
| dog + zebra | 50% | quickdraw + sketch | 64% | 301 | 33% | 0.068 | 67% |
| airplane | 91% | real + infograph | 30% | 291 | 82% | 0.091 | 92% |
| bus + car | 61% | quickdraw | 67% | 288 | 14% | 0.114 | 50% |
| umbrella | 95% | clipart + sketch | 32% | 283 | 87% | 0.062 | 100% |
| bicycle | 44% | quickdraw | 95% | 223 | 15% | 0.043 | 33% |
| car | 93% | clipart + real | 50% | 195 | 78% | 0.093 | 75% |
| dog | 95% | real + clipart | 48% | 193 | 86% | 0.096 | 83% |
| car | 96% | sketch + real | 55% | 180 | 88% | 0.252 | 100% |
| airplane | 84% | quickdraw | 67% | 160 | 28% | 0.078 | 67% |
| tiger | 97% | sketch + painting | 49% | 159 | 94% | 0.061 | 67% |
| tiger | 86% | clipart | 63% | 157 | 87% | 0.076 | 58% |
| tiger | 97% | real + infograph | 60% | 151 | 93% | 0.140 | 83% |
| airplane | 99% | painting | 73% | 111 | 47% | 0.112 | 100% |
| umbrella | 94% | quickdraw | 84% | 106 | 48% | 0.102 | 58% |
| guitar | 93% | quickdraw + sketch | 57% | 102 | 35% | 0.099 | 75% |
| bicycle | 74% | painting | 97% | 89 | 79% | 0.122 | 92% |
| zebra | 97% | real | 76% | 87 | 92% | 0.080 | 83% |
| cat | 90% | real | 94% | 79 | 87% | 0.041 | 67% |
| umbrella | 99% | real | 75% | 75 | 92% | 0.084 | 83% |
| dog | 100% | painting | 98% | 60 | 97% | 0.058 | 50% |
| umbrella | 88% | painting | 96% | 57 | 60% | 0.062 | 58% |
| tiger | 21% | clipart + infograph | 36% | 28 | 0% | highest | 75% |

What the saddles join (by the dominant class and domain of the two peak regions):

| Saddles | count | same class | same domain | both | neither |
|---|---|---|---|---|---|
| merge (spine tree) | 26 | 13 | 9 | 0 | 4 |
| loop | 162 | 9 | 37 | 0 | 116 |

Highest 25 saddles (the regions that blend into each other most easily):

* bus + car / quickdraw -- dog + zebra / quickdraw + sketch: 0.85 (merge)
* dog + zebra / quickdraw + sketch -- bicycle / quickdraw: 0.85 (merge)
* bus + car / quickdraw -- bicycle / quickdraw: 0.85 (loop)
* cat / quickdraw + sketch -- bicycle / quickdraw: 0.85 (merge)
* bus + car / quickdraw -- airplane / quickdraw: 0.85 (merge)
* airplane / quickdraw -- bicycle / quickdraw: 0.85 (loop)
* bus + car / quickdraw -- umbrella / quickdraw: 0.84 (merge)
* cat / quickdraw + sketch -- bus + car / quickdraw: 0.83 (loop)
* cat / quickdraw + sketch -- airplane / quickdraw: 0.83 (loop)
* cat / quickdraw + sketch -- dog + zebra / quickdraw + sketch: 0.83 (loop)
* umbrella / quickdraw -- bicycle / quickdraw: 0.82 (loop)
* tiger / clipart + infograph -- umbrella / quickdraw: 0.81 (merge)
* tiger / clipart + infograph -- airplane / quickdraw: 0.81 (loop)
* tiger / clipart + infograph -- bicycle / quickdraw: 0.81 (loop)
* umbrella / quickdraw -- umbrella / clipart + sketch: 0.81 (merge)
* airplane / quickdraw -- dog + zebra / quickdraw + sketch: 0.81 (loop)
* airplane / quickdraw -- guitar / quickdraw + sketch: 0.80 (merge)
* umbrella / quickdraw -- airplane / quickdraw: 0.80 (loop)
* guitar / quickdraw + sketch -- bicycle / quickdraw: 0.79 (loop)
* umbrella / quickdraw -- dog + zebra / quickdraw + sketch: 0.78 (loop)
* bus + car / quickdraw -- guitar / quickdraw + sketch: 0.75 (loop)
* umbrella / quickdraw -- guitar / quickdraw + sketch: 0.74 (loop)
* cat / quickdraw + sketch -- guitar / quickdraw + sketch: 0.72 (loop)
* dog + zebra / quickdraw + sketch -- guitar / quickdraw + sketch: 0.72 (loop)
* tiger / clipart + infograph -- dog + zebra / quickdraw + sketch: 0.71 (loop)

Stability: k in (10, 15, 20, 30) x threshold in (0.02, 0.04, 0.08); a peak is found when some peak there shares at least half of its images (Jaccard >= 0.5).
