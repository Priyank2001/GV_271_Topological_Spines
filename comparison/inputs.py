"""
Inputs of the comparison: fields, and how their values are prepared.

Fields
------
* AnyEG's own samples, ``third_party/AnyEG/data/scalar_field_v{1,2,3}.vti``
  (10^3 grids with 3 designed maxima, values already made distinct).
* Open SciVis volumes from ``data/klacansky`` (8-bit, so full of equal values).

Value variants
--------------
AnyEG never breaks ties, so the result on 8-bit data depends on how the values
are prepared. Each ``ValueVariant`` turns the stored values into the float32
values both methods receive:

* ``raw``     the stored values, as AnyEG reads them.
* ``anyeg-preprocess``  AnyEG's own recipe (``preprocess/preprocess_vti.py``):
              ``v + 1e-8 * rank`` in float64, then stored as float32. Above
              about 0.1 the 1e-8 steps are smaller than float32 resolution,
              so most ties survive the cast.
* ``rank``    each value replaced by its rank in the (value, index) order:
              all values distinct, in exactly the order our pipeline uses.
              Exact in float32 up to 2^24 vertices.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

from anyeg import ScalarField, read_vti

ROOT = Path(__file__).resolve().parent.parent
ANYEG_DATA = ROOT / "third_party" / "AnyEG" / "data"
KLACANSKY = ROOT / "data" / "klacansky"


@dataclass
class FieldInput:
    name: str
    source: str
    load: Callable[[], ScalarField]


def anyeg_samples(root: Path = ANYEG_DATA) -> List[FieldInput]:
    paths = sorted(root.glob("scalar_field_v*.vti"))
    return [FieldInput(p.stem, "AnyEG repo sample", lambda p=p: read_vti(p)) for p in paths]


def klacansky_volumes(names: Sequence[str], root: Path = KLACANSKY) -> List[FieldInput]:
    index = json.loads((root / "datasets.json").read_text())
    out = []
    for name in names:
        meta = index[name]
        path = root / Path(meta["url"]).name
        if not path.exists():
            continue

        def load(meta=meta, path=path, name=name):
            nx, ny, nz = meta["size"]
            values = np.fromfile(path, dtype=np.dtype(meta["type"])).reshape(nz, ny, nx)
            return ScalarField.from_grid(values, meta.get("spacing", (1, 1, 1)), name=name)

        out.append(FieldInput(name, f"Open SciVis, {meta['type']}", load))
    return out


# ---------------------------------------------------------------------------
class ValueVariant(ABC):
    name: str = "abstract"
    description: str = ""

    @abstractmethod
    def apply(self, values: np.ndarray) -> np.ndarray:
        """float32 values both methods receive."""

    def field(self, field: ScalarField) -> ScalarField:
        return ScalarField(self.apply(field.values), name=field.name, array_name=field.array_name,
                           dims=field.dims, spacing=field.spacing, origin=field.origin,
                           positions=field.positions)


class RawValues(ValueVariant):
    name = "raw"
    description = "values as stored"

    def apply(self, values):
        return np.asarray(values, dtype=np.float32)


class AnyEGPreprocess(ValueVariant):
    name = "anyeg-preprocess"
    description = "v + 1e-8*rank (float64) cast to float32, as AnyEG's preprocess_vti.py"

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def apply(self, values):
        v = np.asarray(values, dtype=np.float64)
        rank = np.empty(v.shape[0], dtype=np.float64)
        rank[np.argsort(v, kind="stable")] = np.arange(v.shape[0])
        return (v + self.epsilon * rank).astype(np.float32)


class RankValues(ValueVariant):
    name = "rank"
    description = "rank in (value, index) order: all distinct, our pipeline's order"

    def apply(self, values):
        v = np.asarray(values)
        if v.shape[0] > 2 ** 24:
            raise ValueError("ranks above 2^24 are not exact in float32")
        rank = np.empty(v.shape[0], dtype=np.float32)
        rank[np.argsort(v, kind="stable")] = np.arange(v.shape[0], dtype=np.float32)
        return rank


VARIANTS = {cls.name: cls for cls in (RawValues, AnyEGPreprocess, RankValues)}
