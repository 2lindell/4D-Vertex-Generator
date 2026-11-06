"""Targeted search for ixitic shapes the map may have missed (surface_survey_<sheet>.json).

On one sheet (HYIC_SUB = ix0..ix9): every wall surface the map recorded, the planes b1 = b4 and b1 = b3 + b4 (where
mastethi lies), and the four mirror faces are sampled densely (Halton points of the chamber put exactly on the surface by
Newton's method, or on the face by setting a beta to 0), each classified at the wall tolerance. A polytope whose key is
not among the merged ixitic shapes (ixitic_shapes.json) is new; any polytope is compared with the accepted list's
unmatched names by the edge valences at a vertex (awosbitdy, sastithi, sisbethi, gisbethi).

    HYIC_SUB=ix3 python surface_survey.py [n per surface]
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import group as G  # noqa: E402
from snub_map import Surface  # noqa: E402

TARGETS = {"awosbitdy": {3: 10, 4: 4, 8: 2, 12: 1}, "sastithi": {3: 5, 4: 6},
           "sisbethi": {3: 4, 4: 2, 5: 2, 6: 2, 8: 1}, "gisbethi": {3: 10, 4: 4, 5: 2}}


def profile(sg):
    V = int(sg.split(":")[0])
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += 2 * int(n) // V
    return dict(e)


def job(item):
    b = np.asarray(item, float)
    try:
        sg, k = G.label(b, 1e-7)
    except Exception:
        return "none", None
    return sg, k


def points(n, seed, face=None):
    X = qmc.Halton(4, seed=seed).random(n)
    X = -np.log(1 - X * 0.999999)
    if face is not None:
        X[:, face] = 0
    return X / X.sum(1, keepdims=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    M = json.load(open(os.path.join(HERE, G.fname("snub_map"))))
    texts = list(dict.fromkeys(M["surfaces"] + ["b1 - b4", "b1 - b3 - b4"]))
    known = {s["key"] for s in json.load(open(os.path.join(HERE, "ixitic_shapes.json")))["shapes"]}
    samples = []                                        # (source, beta)
    for si, t in enumerate(texts):
        S = Surface(t)
        for x in points(n * 3, 11 + si):
            y = S.project(x)
            if y.min() > 1e-6 and S.dist(y) < 1e-10:
                samples.append((t, y / y.sum()))
            if sum(1 for s in samples if s[0] == t) >= n:
                break
    for f in range(4):
        for x in points(n, 101 + f, face=f):
            samples.append((f"face b{f + 1} = 0", x))
    with Pool(4) as pool:
        R = pool.map(job, [tuple(b) for _, b in samples], chunksize=8)
    found, hits = {}, []
    for (src, b), (sg, k) in zip(samples, R):
        if k is None:
            continue
        p = profile(sg)
        for nm, tp in TARGETS.items():
            if p == tp:
                hits.append({"target": nm, "key": k, "at": b.tolist(), "on": src, "signature": sg})
        if k not in known:
            found.setdefault(k, {"signature": sg, "at": b.tolist(), "on": src, "count": 0, "profile": p})["count"] += 1
    out = {"_about": __doc__.strip().split("\n\n")[0], "sheet": G.SUB, "samples": len(samples), "surfaces": len(texts),
           "new": found, "target_hits": hits}
    json.dump(out, open(os.path.join(HERE, f"surface_survey_{G.SUB}.json"), "w"), indent=1)
    print(f"{G.SUB}: {len(samples)} samples on {len(texts)} surfaces and 4 faces; {len(found)} new polytopes; "
          f"{len(hits)} target hits {sorted({h['target'] for h in hits})}", flush=True)
    for k, f in found.items():
        print(f"  NEW {k} x{f['count']} on {f['on'][:50]}  edges/vertex {dict(sorted(f['profile'].items()))}  "
              f"{f['signature'].split('|')[0][:90]}", flush=True)


if __name__ == "__main__":
    main()
