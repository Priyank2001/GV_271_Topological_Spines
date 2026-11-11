"""
How similar are two extremum graphs of the same field?

``compare(a, b)`` returns an ``Agreement`` with, for maxima, saddles and
linked pairs of maxima, the counts in each graph and the Jaccard index
(|A and B| / |A or B|, 1.0 = identical sets). ``region_agreement`` is the
fraction of vertices that ascend to the same maximum in both. Everything is
exact vertex identity; the methods run on the same grid, so no tolerance
is needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .common import CommonGraph


@dataclass
class Agreement:
    a: str
    b: str
    maxima_jaccard: float
    maxima_only_a: int
    maxima_only_b: int
    region_agreement: float
    saddle_jaccard: float
    pair_jaccard: float
    pairs_only_a: int
    pairs_only_b: int

    @property
    def identical(self) -> bool:
        return (self.maxima_jaccard == 1.0 and self.region_agreement == 1.0
                and self.saddle_jaccard == 1.0 and self.pair_jaccard == 1.0)

    def as_dict(self) -> dict:
        return asdict(self)


def jaccard(a: Iterable, b: Iterable) -> float:
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def compare(a: CommonGraph, b: CommonGraph) -> Agreement:
    if a.region.shape != b.region.shape:
        raise ValueError("the two graphs are not on the same vertices")
    ma, mb = set(a.maxima.tolist()), set(b.maxima.tolist())
    return Agreement(
        a=a.method, b=b.method,
        maxima_jaccard=jaccard(ma, mb),
        maxima_only_a=len(ma - mb),
        maxima_only_b=len(mb - ma),
        region_agreement=float(np.mean(a.region == b.region)),
        saddle_jaccard=jaccard(a.saddles.tolist(), b.saddles.tolist()),
        pair_jaccard=jaccard(a.maximum_pairs, b.maximum_pairs),
        pairs_only_a=len(a.maximum_pairs - b.maximum_pairs),
        pairs_only_b=len(b.maximum_pairs - a.maximum_pairs),
    )
