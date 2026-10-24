"""The regions on the two sides of each meshed curved wall (xsurfaces.json): points of the mesh are stepped off the
surface along its normal, both ways, and classified. Written to surface_sides.json as {wall: {region: count}}.

    python surface_sides.py [X10 X33 ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from fexact import _lab
from xsurface import surface, project, grad


def _job(args):
    tid, x = args
    sf = surface(tid)
    x = project(sf, np.asarray(x, float))
    n = grad(sf, x)
    n /= np.linalg.norm(n)
    out = []
    for s in (2e-5, -2e-5):
        y = x + s * n
        out.append(_lab(y / y.sum()) if y.min() > 0 and y[2] >= y[3] else "OUT")
    return tid, out


def main(tids=None):
    X = json.load(open("xsurfaces.json"))
    jobs = []
    for tid, sf in X.items():
        if tids and tid not in tids:
            continue
        T = sf["tris"]
        for t in T[:: max(1, len(T) // 24)]:
            jobs.append((tid, np.mean(t, axis=0).tolist()))
    res = {}
    with Pool(4) as pool:
        for tid, labs in pool.imap_unordered(_job, jobs, chunksize=2):
            res.setdefault(tid, Counter()).update(l for l in labs if l not in ("OUT", "ERR", tid))
    try:
        old = json.load(open("surface_sides.json"))
    except FileNotFoundError:
        old = {}
    old.update({t: dict(c.most_common()) for t, c in res.items()})
    json.dump(old, open("surface_sides.json", "w"), indent=1)
    for t, c in res.items():
        print(t, c.most_common(6), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
