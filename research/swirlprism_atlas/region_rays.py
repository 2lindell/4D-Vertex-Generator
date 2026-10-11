"""The boundaries of a region shape by rays: from a sample of the shape, random directions (in the hyperplane
sum(beta) = 1) are walked outward until the shape changes, the crossing is bisected, and the shape on the boundary
(loose classifier) and the one past it are recorded. A few of the farthest interior points found become new starting
points, so a region that is not star-shaped from its sample is still covered. Faster than region_map.py for thin
regions.

    python region_rays.py X71 out.json b1,b2,b3,b4 [rays]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from fexact import _lab_loose

U = np.linalg.svd(np.ones((1, 4)))[2][1:]


def lab(x):
    x = np.asarray(x, float)
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab_loose(x / x.sum())


def _ray(args):
    tid, x0, seed = args
    x0 = np.asarray(x0, float)
    d = U.T @ np.random.default_rng(seed).normal(size=3)
    d /= np.linalg.norm(d)
    lo, r = 0.0, 2e-4
    while r < 2.0:
        l = lab(x0 + r * d)
        if l != tid:
            break
        lo, r = r, r * 1.6
    else:
        return None
    hi, beyond = r, l
    for _ in range(26):
        m = (lo + hi) / 2
        l = lab(x0 + m * d)
        if l == tid:
            lo = m
        else:
            hi, beyond = m, l
    x = x0 + (lo + hi) / 2 * d
    return (x0 + lo * d).tolist(), x.tolist(), lab(x), beyond


def main(tid, out, start, rays=64):
    starts, seen, res = [np.asarray(start, float) / sum(start)], [], []
    with Pool(4) as pool:
        for gen in range(2):
            jobs = [(tid, s.tolist(), 1000 * gen + 97 * i + j) for i, s in enumerate(starts) for j in range(rays // len(starts))]
            got = [g for g in pool.map(_ray, jobs) if g]
            res += got
            inner = np.array([g[0] for g in got])
            far = inner[np.argsort(-np.linalg.norm(inner - starts[0], axis=1))]
            starts = [starts[0]] + [p for p in far[:3]]
            print(f"{tid} gen {gen}: {len(got)} crossings", flush=True)
    on = Counter(r[2] for r in res if r[2] != tid)
    beyond = Counter(r[3] for r in res)
    json.dump({"tid": tid, "on": on.most_common(), "beyond": beyond.most_common(),
               "points": [{"beta": r[1], "on": r[2], "beyond": r[3]} for r in res]}, open(out, "w"), indent=1)
    print(f"{tid} on the boundary: {[(k[:40], v) for k, v in on.most_common()]}", flush=True)
    print(f"{tid} past it: {[(k[:40], v) for k, v in beyond.most_common()]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [float(v) for v in sys.argv[3].split(",")], *(int(a) for a in sys.argv[4:]))
