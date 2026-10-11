"""The shapes living on the curved edges of the meshed walls (xsurfaces.json): for each wall and each neighbour past its
edge, a few edge points are refined by bisecting across the edge while staying on the wall's surface, and the point on
the edge is classified (loose classifier). A known line shape there is the seam between the two; an unnumbered
signature is a new transitional shape.

    python edge_shapes.py out.json [X10 X33 ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from fexact import _lab, _lab_loose
from xsurface import surface, project, grad

U = np.linalg.svd(np.ones((1, 4)))[2][1:]


def lab(x):
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab(x)


def _job(args):
    tid, beyond, x = args
    sf = surface(tid)
    x = np.asarray(x)
    rng = np.random.default_rng(int(abs(x[0]) * 1e9) % 2 ** 32)
    n = grad(sf, x)
    n /= np.linalg.norm(n)
    for r in (1e-3, 3e-3):
        for _ in range(10):
            t = U.T @ rng.normal(size=3)
            t -= (t @ n) * n
            t /= np.linalg.norm(t)
            a, b = project(sf, x - r * t), project(sf, x + r * t)
            la, lb = lab(a), lab(b)
            if tid in (la, lb) and la != lb and "OUT" not in (la, lb):
                if lb == tid:
                    a, b, la, lb = b, a, lb, la
                for _ in range(32):
                    m = project(sf, (a + b) / 2)
                    if lab(m) == tid:
                        a = m
                    else:
                        b = m
                m = project(sf, (a + b) / 2)
                return tid, beyond, lb, _lab_loose(m), m.tolist()
    return tid, beyond, None, None, None


def main(out, tids=None):
    X = json.load(open("xsurfaces.json"))
    jobs = []
    for tid, sf in X.items():
        if tids and tid not in tids:
            continue
        groups = {}
        for line in sf["outline"]:
            for q in line:
                groups.setdefault(q["beyond"], []).append(q["beta"])
        for beyond, P in groups.items():
            if beyond.startswith(("face:", "new:")) or beyond in ("ERR", "OUT") or len(P) < 10:
                continue
            for x in P[len(P) // 6:: max(1, len(P) // 3)][:3]:
                jobs.append((tid, beyond, x))
    print(f"{len(jobs)} edge points", flush=True)
    res = {}
    with Pool(4) as pool:
        for tid, beyond, other, on, m in pool.imap_unordered(_job, jobs):
            res.setdefault(f"{tid}|{beyond}", []).append({"other": other, "on": on, "beta": m})
            print(f"{tid} | {beyond}: across {other}, on the edge {(on or '-')[:60]}", flush=True)
    summary = {k: Counter(r["on"] for r in v if r["on"]).most_common() for k, v in res.items()}
    try:                                         # (merge with an earlier run: walls checked now replace their entries)
        old = json.load(open(out))
        done = {k.split("|")[0] for k in res}
        summary = {**{k: v for k, v in old["summary"].items() if k.split("|")[0] not in done}, **summary}
        res = {**{k: v for k, v in old["points"].items() if k.split("|")[0] not in done}, **res}
    except FileNotFoundError:
        pass
    json.dump({"summary": summary, "points": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or None)
