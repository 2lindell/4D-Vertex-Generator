"""A dense survey of the toxitic / oxitic domain for small regions the grid missed (survey_<sub>.json).

Halton points (off any lattice) in the whole domain and in a thin zone along each of its edges (two betas, or a beta
and b1 - b4, below 0.04), and near each corner (three below 0.04); each classified and compared, by arrangement key,
with the map's region types (snub_map_<sub>.json). A polytope not among them is new, given with its margin.

    CONTIC_SUB=tox|ox|oxb python survey.py [n per zone]
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
CONTIC = os.path.join(os.path.dirname(HERE), "contic_atlas")
sys.path.insert(0, CONTIC)
import group as G  # noqa: E402
from margin import margin  # noqa: E402

W = 0.04


def quantities(X):
    """The domain's edge coordinates: the four betas and b1 - b4."""
    return np.column_stack([X, X[:, 0] - X[:, 3]])


def zone_points(small, n, seed):
    X = qmc.Halton(4, seed=seed).random(n * 400)
    X = -np.log(1 - X * 0.999999)
    X = X / X.sum(1, keepdims=True)
    X = X[X[:, 0] > X[:, 3]]
    if small:
        Q = quantities(X)
        X = X[(Q[:, list(small)] < W).all(1)]
    return X[:n]


def job(b):
    try:
        return G.label(np.asarray(b))
    except Exception:
        return "none", None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    M = json.load(open(os.path.join(CONTIC, G.fname("snub_map"))))
    regions = [t for t in M["types"] if t["grid_regions"]]
    names = ["b1", "b2", "b3", "b4", "b1-b4"]
    zones = {"whole domain": ()}
    for pr in itertools.combinations(range(5), 2):
        if pr != (0, 3):                         # (b1 and b4 small make b1 - b4 small: the b1 = b4 = 0 edge is (0, 3))
            zones[f"{names[pr[0]]}, {names[pr[1]]} small"] = pr
    zones["b1, b4 small"] = (0, 3)
    for tr in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (1, 2, 4)]:
        zones[" ".join(names[i] for i in tr) + " small"] = tr
    out = {}
    with Pool(4) as pool:
        known = {k for _, k in pool.map(job, [tuple(t["example"]) for t in regions])}
        newall = {}
        for zi, (zn, small) in enumerate(zones.items()):
            P = zone_points(small, n, zi + 3)
            R = pool.map(job, list(map(tuple, P)), chunksize=8)
            types = {}
            for b, (sg, k) in zip(P, R):
                if k is not None:
                    types.setdefault(k, {"signature": sg, "points": []})["points"].append(b.tolist())
            new = {k: t for k, t in types.items() if k not in known}
            out[zn] = {"points": len(P), "types": {k: len(t["points"]) for k, t in types.items()},
                       "new": {k: {"count": len(t["points"]), "signature": t["signature"],
                                   "example": t["points"][len(t["points"]) // 2]} for k, t in new.items()}}
            for k, t in new.items():
                newall.setdefault(k, out[zn]["new"][k])
            print(f"{zn:22} {len(P):4} points  {len(types)} polytopes, {len(new)} not among the map's regions", flush=True)
    for k, t in newall.items():
        t["margin"] = float(margin(np.asarray(t["example"])))
        print(f"  NEW {k} {t['count']:4} pts margin {t['margin']:.1e} e.g. {np.round(t['example'], 5).tolist()}  "
              f"{t['signature'][:100]}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "group": G.SUB, "known_regions": sorted(known),
               "zones": out, "new": newall}, open(os.path.join(HERE, f"survey_{G.SUB}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
