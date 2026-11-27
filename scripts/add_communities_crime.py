#!/usr/bin/env python3
"""
add_communities_crime.py

Builds the UCI "Communities and Crime" dataset (Redmond, 2009; UCI id 183) as the point
cloud "communities_crime" in generator/datasets/. Run from the repo root:

    python3 scripts/add_communities_crime.py [path to communities.data or the .zip]
    python3 web/build_data.py communities_crime

Without an argument it uses data/communities_crime/, downloading the zip there if needed.

The point cloud
  * points: the 1994 US communities. Dimensions: the 122 predictive attributes (1990 census,
    1990 LEMAS police survey) minus the 22 police attributes that are missing for 1675
    communities, so 100 dimensions. They are kept in the 0..1 scaling UCI provides
    (equal-interval binning, values beyond 3 SD clipped). The one missing value
    (OtherPerCap) is filled with the column mean.
  * f: ViolentCrimesPerPop (violent crimes per 100k people, 1995 FBI UCR, scaled 0..1 by UCI),
    the dataset's target; not part of the points. Peaks: socio-economic profiles with high
    violent crime; saddles: how two such profiles connect.
  * known labels (Clustering -> Known labels; never used to build f or the graph):
    US Census region and state, from the state FIPS code.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
if not (ROOT / "web" / "build_data.py").exists():
    sys.exit("run this from the repo root (the folder that has web/build_data.py)")

URL = "http://archive.ics.uci.edu/static/public/183/communities+and+crime.zip"
RAW_DIR = ROOT / "data" / "communities_crime"
MISSING_LIMIT = 0.10          # drop attributes missing in more than 10 % of communities

# State FIPS code -> (postal code, Census region)
STATES = {
    1: ("AL", "South"), 2: ("AK", "West"), 4: ("AZ", "West"), 5: ("AR", "South"),
    6: ("CA", "West"), 8: ("CO", "West"), 9: ("CT", "Northeast"), 10: ("DE", "South"),
    11: ("DC", "South"), 12: ("FL", "South"), 13: ("GA", "South"), 15: ("HI", "West"),
    16: ("ID", "West"), 17: ("IL", "Midwest"), 18: ("IN", "Midwest"), 19: ("IA", "Midwest"),
    20: ("KS", "Midwest"), 21: ("KY", "South"), 22: ("LA", "South"), 23: ("ME", "Northeast"),
    24: ("MD", "South"), 25: ("MA", "Northeast"), 26: ("MI", "Midwest"), 27: ("MN", "Midwest"),
    28: ("MS", "South"), 29: ("MO", "Midwest"), 30: ("MT", "West"), 31: ("NE", "Midwest"),
    32: ("NV", "West"), 33: ("NH", "Northeast"), 34: ("NJ", "Northeast"), 35: ("NM", "West"),
    36: ("NY", "Northeast"), 37: ("NC", "South"), 38: ("ND", "Midwest"), 39: ("OH", "Midwest"),
    40: ("OK", "South"), 41: ("OR", "West"), 42: ("PA", "Northeast"), 44: ("RI", "Northeast"),
    45: ("SC", "South"), 46: ("SD", "Midwest"), 47: ("TN", "South"), 48: ("TX", "South"),
    49: ("UT", "West"), 50: ("VT", "Northeast"), 51: ("VA", "South"), 53: ("WA", "West"),
    54: ("WV", "South"), 55: ("WI", "Midwest"), 56: ("WY", "West"),
}
REGIONS = ["Northeast", "Midwest", "South", "West"]


# ---- 1. read communities.data and the attribute names --------------------------------
def read_text(arg: Path | None) -> tuple[str, str]:
    """(communities.data, communities.names) as text."""
    if arg is not None and arg.suffix != ".zip":
        return arg.read_text(), arg.with_name("communities.names").read_text()
    archive = arg if arg is not None else RAW_DIR / "communities_crime.zip"
    if not archive.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as r:
            archive.write_bytes(r.read())
    with zipfile.ZipFile(archive) as z:
        return z.read("communities.data").decode(), z.read("communities.names").decode()


arg = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
data_txt, names_txt = read_text(arg)
columns = [line.split()[1] for line in names_txt.splitlines() if line.startswith("@attribute")]
rows = [line.split(",") for line in data_txt.splitlines() if line.strip()]
if len(columns) != 128 or any(len(r) != len(columns) for r in rows):
    sys.exit(f"unexpected layout: {len(columns)} attributes, row lengths {sorted({len(r) for r in rows})}")
col = {name: i for i, name in enumerate(columns)}
print(f"{len(rows)} communities, {len(columns)} attributes")

NON_PREDICTIVE = ["state", "county", "community", "communityname", "fold"]
TARGET = "ViolentCrimesPerPop"


def numeric(name: str) -> np.ndarray:
    return np.array([np.nan if r[col[name]] == "?" else float(r[col[name]]) for r in rows])


# ---- 2. dimensions: the predictive attributes that are (almost) complete ------------
predictive = [c for c in columns if c not in NON_PREDICTIVE and c != TARGET]
X = np.stack([numeric(c) for c in predictive], axis=1)
keep = np.isnan(X).mean(axis=0) <= MISSING_LIMIT
dropped = [c for c, k in zip(predictive, keep) if not k]
X, names = X[:, keep], [c for c, k in zip(predictive, keep) if k]
filled = int(np.isnan(X).sum())
X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
print(f"{len(names)} dimensions; dropped {len(dropped)} police attributes "
      f"(missing for most communities); filled {filled} missing value(s) with the column mean")

# ---- 3. f: violent crime rate ---------------------------------------------------------
f = numeric(TARGET)
if np.isnan(f).any():
    sys.exit(f"{TARGET} has missing values")

# ---- 4. known labels -------------------------------------------------------------------
state_code = numeric("state").astype(int)
unknown = sorted(set(state_code) - set(STATES))
if unknown:
    sys.exit(f"unknown state FIPS codes: {unknown}")
present = sorted({STATES[s][0] for s in state_code})
classes = [
    {"name": "Census region", "names": REGIONS,
     "values": [REGIONS.index(STATES[s][1]) for s in state_code]},
    {"name": "state", "names": present,
     "values": [present.index(STATES[s][0]) for s in state_code]},
]
sample_ids = [f"{r[col['communityname']]}, {STATES[s][0]}" for r, s in zip(rows, state_code)]

# ---- 5. dataset folder ---------------------------------------------------------------------
out = ROOT / "generator" / "datasets" / "communities_crime"
out.mkdir(parents=True, exist_ok=True)
np.save(out / "points.npy", X.astype(np.float32))
np.save(out / "values.npy", f.astype(np.float32))
truth = {
    "name": "communities_crime",
    "mode": "real",
    "dimensions": int(X.shape[1]),
    "num_points": int(X.shape[0]),
    "description": (f"UCI Communities and Crime: {X.shape[0]} US communities. Dimensions: "
                    f"{X.shape[1]} socio-economic attributes from the 1990 census (UCI's 0..1 "
                    f"scaling; {len(dropped)} mostly-missing police attributes dropped). "
                    f"f: {TARGET} (violent crimes per 100k people, scaled 0..1)."),
    "source": URL,
    "scalar": TARGET,
    "dimension_names": names,
    "dropped_attributes": dropped,
    "sample_ids": sample_ids,
    "fold": [int(r[col["fold"]]) for r in rows],
    "point_classes": classes,
    "extrema": {"maxima": []},
}
(out / "truth.json").write_text(json.dumps(truth))
print(f"wrote {out}: {X.shape[0]} points, {X.shape[1]} dimensions, "
      f"f in [{f.min():.2f}, {f.max():.2f}] ({len(np.unique(f))} distinct values)")
print("\nnext:  python3 web/build_data.py communities_crime")
