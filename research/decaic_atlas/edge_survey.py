"""A dense survey of the domain's edge zones, where the small snub regions live (edge_survey_<sub>.json).

The kidecaic map found its small regions (1-72 grid points at N = 40) hugging two edges of the domain: b1 and b4 small,
or b2 and b3 small. The kicontic / iocontic grids (N = 32) are coarser, so those zones are surveyed here with many more
points (a Halton sequence in each zone, off any lattice), each classified and compared (by incidences) with the
region types of the map; any polytope not among them is new, with its margin.

    CONTIC_SUB=ki python edge_survey.py [n per zone]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

import group as G
from margin import margin

ZONES = {"b1, b4 small": [(0.003, 0.15), (0.0, 1.0), (0.0, 1.0), (0.002, 0.08)],
         "b2, b3 small": [(0.0, 1.0), (0.002, 0.08), (0.003, 0.15), (0.0, 1.0)],
         "whole domain": None}


def points(zone, n, seed):
    if zone is None:
        X = qmc.Halton(4, seed=seed).random(n * 3)
        X = -np.log(1 - X * 0.999999)                       # (uniform on the simplex: normalised exponentials)
        X = X / X.sum(1, keepdims=True)
        return X[X[:, 0] > X[:, 3]][:n]
    lo = np.array([a for a, _ in zone]); hi = np.array([b for _, b in zone])
    X = qmc.Halton(4, seed=seed).random(n * 4)
    X = lo + X * (hi - lo)
    free = [i for i in range(4) if hi[i] - lo[i] == 1.0]
    small = [i for i in range(4) if i not in free]
    # (the two free betas share what the small ones leave: split by the free fraction)
    t = X[:, free[0]]
    rest = 1 - X[:, small].sum(1)
    X[:, free[0]] = t * rest
    X[:, free[1]] = (1 - t) * rest
    X = X[(X.min(1) > 0) & (X[:, 0] > X[:, 3])]
    return X[:n]


def job(b):
    sg, k = G.label(b)
    return sg, k


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    W = json.load(open(G.fname("wall_shapes")))["shapes"]
    reps = [(w["signature"], w["seed_beta"]) for w in W if w["kind"] == "region"]
    out = {}
    with Pool(4) as pool:
        for zi, (zname, zone) in enumerate(ZONES.items()):
            P = points(zone, n, zi)
            R = pool.map(job, list(map(tuple, P)), chunksize=8)
            types = {}
            for b, (sg, k) in zip(P, R):
                if k is None:
                    continue
                types.setdefault(k, {"signature": sg, "points": []})["points"].append(b.tolist())
            found = []
            for k, t in types.items():
                b = t["points"][len(t["points"]) // 2]
                known = next((i for i, (sg, rb) in enumerate(reps) if sg.split(":")[0] == t["signature"].split(":")[0]
                              and G.same_polytope(b, rb)), None)
                found.append({"known_region": known, "count": len(t["points"]), "signature": t["signature"],
                              "example": b, "margin": float(margin(b)) if known is None else None})
            out[zname] = {"points": len(P), "types": found}
            print(f"{zname}: {len(P)} points, {len(types)} polytopes, "
                  f"{sum(1 for f in found if f['known_region'] is None)} not among the map's regions", flush=True)
            for f in found:
                if f["known_region"] is None:
                    print(f"   NEW {f['count']:4} pts margin {f['margin']:.1e}  {f['signature'][:110]}  e.g. {np.round(f['example'], 5).tolist()}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "zones": out}, open(G.fname("edge_survey"), "w"), indent=1)


if __name__ == "__main__":
    main()
