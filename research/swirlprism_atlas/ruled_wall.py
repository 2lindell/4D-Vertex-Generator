"""Mesh a curved wall on a cone exactly by its rulings: the wall is a sector of the cone between two rulings through
the apex, out to a far edge. Each ruling between the two (found on the cone's section by the plane through the two
ruling ends) is walked out from the apex and its far end bisected (strict classifier); the strips between neighbouring
rulings are triangulated, with a fan at the apex, so the wall reaches its apex. Replaces the wall's entry in
xsurfaces.json (marching tetrahedra cannot resolve a sector close to a cone's apex, where it narrows to a point).

    python ruled_wall.py X82 [out.json]      (out: a file to merge into xsurfaces.json later)
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from fexact import _lab, _lab_loose
from xsurface import surface, F, grad

# wall: (apex, first ruling's other point, last ruling's other point, what lies past each ruling)
WALLS = {"X82": ((0, 1, 0, 0), (1, -2, 1, 1), (0, -1, 1, 1), "X13", "face:D2")}
K, J = 64, 36


def lab(x):
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab(x / x.sum())


def section(tid, s):
    """The point of the cone's section by the plane beta2 = 0 (it misses the apex) between the two ruling ends, at
    chord parameter s: the chord point moved onto the cone along the gradient's part in that plane."""
    A, W0, W1, _, _ = WALLS[tid]
    sf = surface(tid)
    A, W0, W1 = (np.asarray(v, float) for v in (A, W0, W1))
    W0, W1 = (A + (W - A) / -(W - A)[1] for W in (W0, W1))
    x = (1 - s) * W0 + s * W1
    d = grad(sf, x)
    d[1] = 0
    d[[0, 2, 3]] -= d[[0, 2, 3]].mean()
    for _ in range(60):
        f = F(sf, x)
        g = grad(sf, x) @ d
        if abs(f) < 1e-15 or abs(g) < 1e-15:
            break
        x = x - f / g * d
    return x


def _ruling(args):
    tid, k = args
    A = np.asarray(WALLS[tid][0], float)
    w = section(tid, min(max(k / K, 1e-4), 1 - 1e-4))   # (the end rulings are the bounding lines: walk just inside)
    d = w - A                                  # walk out along A + t d
    t, step = 0.0, 0.02
    while lab(A + (t + step) * d) == tid:
        t += step
        if t > 3:
            break
    lo, hi = t, t + step
    beyond = lab(A + hi * d)
    for _ in range(30):
        m = (lo + hi) / 2
        l = lab(A + m * d)
        if l == tid:
            lo = m
        else:
            hi, beyond = m, l
    if beyond == "ERR":                       # (the strict classifier fails right at the edge: look just past it)
        x = A + (lo + 1e-6) * d
        beyond = _lab_loose(x / x.sum())
    return k, (A + lo * (section(tid, k / K) - A)).tolist(), beyond, lab(A + 0.5 * lo * d)


def main(tid, out="xsurfaces.json"):
    A = np.asarray(WALLS[tid][0], float)
    with Pool(4) as pool:
        res = sorted(pool.map(_ruling, [(tid, k) for k in range(K + 1)]))
    bad = [(k, mid) for k, _, _, mid in res if mid != tid]
    print(f"{tid}: {K + 1} rulings; midpoints off the wall: {bad}", flush=True)
    ends = [np.asarray(e, float) for _, e, _, _ in res]
    # points along each ruling (j = 0 is the apex), triangulated strip by strip
    R = [[A + (j / J) * (e - A) for j in range(J + 1)] for e in ends]
    tris = []
    for k in range(K):
        a, b = R[k], R[k + 1]
        tris.append([a[0], a[1], b[1]])
        for j in range(1, J):
            tris.append([a[j], a[j + 1], b[j + 1]])
            tris.append([a[j], b[j + 1], b[j]])
    first, last = WALLS[tid][3], WALLS[tid][4]
    outline = ([{"beta": p.tolist(), "beyond": first} for p in R[0]]
               + [{"beta": e.tolist(), "beyond": bey} for (_, _, bey, _), e in zip(res, ends)]
               + [{"beta": p.tolist(), "beyond": last} for p in R[K][::-1]])
    from collections import Counter
    beyond = Counter(o["beyond"] for o in outline)
    X = json.load(open(out)) if os.path.exists(out) else {}
    if tid not in X:
        X[tid] = dict(json.load(open("xsurfaces.json"))[tid])
    sf = X[tid]
    sf.update({"tris": [[p.tolist() for p in t] for t in tris], "outline": [outline + [outline[0]]],
               "beyond": beyond.most_common(), "ruled": True})
    json.dump(X, open(out, "w"))
    print(f"{tid}: {len(tris)} triangles; far edge: {Counter(b for _, _, b, _ in res).most_common()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
