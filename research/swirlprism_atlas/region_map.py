"""Map a region shape and the walls around it: a 3D grid (in the hyperplane sum(beta) = 1) flooded outward from the
shape's samples until every grid point of the shape has its neighbours classified; then each grid edge from the shape to
a neighbour is bisected for a boundary point, which is classified too (the wall shape there), and the boundary points
toward each neighbour are fitted by a plane (golden normal when it snaps) or reported as curved.

    python region_map.py X45 h out.json beta1 beta2 ...      (betas as comma-separated golden-free floats)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from fexact import _lab_loose

U = np.linalg.svd(np.ones((1, 4)))[2][1:]
MAX_EDGES = 160
NB = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def lab(x):
    x = np.asarray(x, float)
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab_loose(x / x.sum())


_C, _H = None, None


def _init(c, h):
    global _C, _H
    _C, _H = np.asarray(c), h


def at(k):
    return _C + _H * (U.T @ np.asarray(k, float))


def _job(k):
    return k, lab(at(k))


def _edge(args):
    tid, kin, kout = args
    a, b = at(kin), at(kout)
    lo, hi, beyond = 0.0, 1.0, None
    for _ in range(24):
        m = (lo + hi) / 2
        l = lab(a + m * (b - a))
        if l == tid:
            lo = m
        else:
            hi, beyond = m, l
    x = a + (lo + hi) / 2 * (b - a)
    return x.tolist(), lab(x), beyond or lab(b)


def main(tid, h, out, seeds, cap=7000):
    seeds = [np.array(s, float) / sum(s) for s in seeds]
    c = np.mean(seeds, axis=0)
    known = {}
    start = {tuple(int(round(v)) for v in (U @ (s - c)) / h) for s in seeds}
    with Pool(4, initializer=_init, initargs=(c, h)) as pool:
        _init(c, h)
        front = sorted(start)
        while front and len(known) < cap:
            for k, l in pool.imap_unordered(_job, front, chunksize=4):
                known[k] = l
            front = sorted({tuple(int(v) for v in np.add(k, d)) for k, l in known.items() if l == tid for d in NB} - set(known))
            print(f"{tid}: {sum(l == tid for l in known.values())} of {len(known)} grid points in the shape, "
                  f"{len(front)} to classify", flush=True)
        edges = [(tid, k, tuple(int(v) for v in np.add(k, d))) for k, l in known.items() if l == tid for d in NB
                 if known.get(tuple(int(v) for v in np.add(k, d)), tid) not in (tid, "OUT")]
        rng = np.random.default_rng(0)
        if len(edges) > MAX_EDGES:
            edges = [edges[i] for i in rng.choice(len(edges), MAX_EDGES, replace=False)]
        print(f"{tid}: bisecting {len(edges)} boundary edges", flush=True)
        res = pool.map(_edge, edges, chunksize=2)
    by = defaultdict(list)
    for x, on, beyond in res:
        by[beyond].append((x, on))
    walls = {}
    for beyond, pts in by.items():
        P = np.array([x for x, _ in pts])
        n = np.linalg.svd(P)[2][-1] if len(P) >= 4 else None
        resid = float(np.abs(P @ n).max()) if n is not None else None
        walls[beyond] = {"count": len(pts), "on": Counter(o for _, o in pts).most_common(4), "normal": None if n is None else (n / n[np.argmax(np.abs(n))]).tolist(),
                         "plane_residual": resid, "points": P.tolist()}
        print(f"  beyond {beyond[:40]}: {len(pts)} points, on {walls[beyond]['on']}, plane residual {resid}", flush=True)
    json.dump({"tid": tid, "h": h, "centre": c.tolist(), "grid": [[[int(v) for v in k], l] for k, l in known.items()],
               "walls": walls},
              open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], [json.loads(a) for a in sys.argv[4:]])
