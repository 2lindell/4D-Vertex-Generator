"""Classify seeds at golden-field barycentric points of the half-cell (plus every uniform point)."""
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from cellframe import PHI, in_half, seed_from_beta
from classify import classify, signature


def golden_values(amax, bmax):
    return sorted({(a, b) for a in range(amax + 1) for b in range(bmax + 1)})

def grid(amax, bmax):
    vals = golden_values(amax, bmax); seen = {}; pts = []
    for combo in itertools.product(vals, repeat=4):
        beta = np.array([a + b * PHI for a, b in combo])
        if beta.sum() == 0 or not in_half(beta): continue
        key = tuple(np.round(beta / beta.sum(), 10))
        if key in seen: continue
        seen[key] = combo; pts.append({"beta": (beta / beta.sum()).tolist(), "golden": [list(c) for c in combo]})
    return pts

def work(pt):
    try:
        info = classify(seed_from_beta(pt["beta"]))
        pt["sig"] = signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    except Exception as exc:
        pt["sig"] = f"ERR {type(exc).__name__}"
    return pt

if __name__ == "__main__":
    amax, bmax, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    pts = grid(amax, bmax)
    for u in json.load(open("uniform_in_cell.json")):                  # make sure every uniform point is included
        key = tuple(np.round(np.array(u["beta"]) / sum(u["beta"]), 10))
        if not any(tuple(np.round(p["beta"], 10)) == key for p in pts):
            pts.append({"beta": u["beta"], "golden": None})
    print(len(pts), "golden points", flush=True)
    if len(sys.argv) > 4 and sys.argv[4] == "count": sys.exit()
    res, t0 = [], time.time()
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(work, pts, chunksize=4)):
            res.append(r)
            if (i + 1) % 250 == 0:
                print(f"{i + 1}/{len(pts)} {time.time() - t0:.0f}s", flush=True); json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w")); print("done", flush=True)
