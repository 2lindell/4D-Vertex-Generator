"""Map every shape inside one plane of the cell: classify a triangular grid, then bisect each grid
edge whose ends differ so the boundaries between regions are located exactly.

    python planemap.py <name> <corner1> <corner2> <corner3> <N> <out.json>
corners are barycentric weights such as 0,1,0,0 (numbers or expressions in phi, e.g. 1+phi).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from probe import _one

PHI = (1 + 5 ** 0.5) / 2
STEPS = 16


def grid(corners: np.ndarray, n: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    idx, pts = [], []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            idx.append((i, j))
            pts.append((i * corners[0] + j * corners[1] + k * corners[2]) / n)
    return idx, np.array(pts)


def _bisect(args):
    a, b, la, lb = args
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = 0.0, 1.0
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        lab = _one(a + mid * (b - a))[0]
        if lab == la:
            lo = mid
        elif lab == lb:
            hi = mid
        else:                       # a third region in between: stop here and report it
            return {"pt": (a + mid * (b - a)).tolist(), "between": [la, lab], "third": True}
    return {"pt": (a + (lo + hi) / 2 * (b - a)).tolist(), "between": sorted([la, lb]), "third": False}


def main(name, corners, n, out):
    corners = np.array([c / c.sum() for c in corners])
    idx, pts = grid(corners, n)
    with Pool(4) as pool:
        res = pool.map(_one, [p.tolist() for p in pts], chunksize=4)
        labels = [r[0] for r in res]
        where = {ij: m for m, ij in enumerate(idx)}
        edges = []
        for (i, j), m in where.items():
            for di, dj in ((1, 0), (0, 1), (-1, 1)):
                o = where.get((i + di, j + dj))
                if o is not None and labels[m] != labels[o]:
                    edges.append((pts[m].tolist(), pts[o].tolist(), labels[m], labels[o]))
        bounds = pool.map(_bisect, edges, chunksize=2)
    sigs = {r[0]: r[1] for r in res}
    json.dump({"name": name, "corners": corners.tolist(), "n": n, "points": pts.tolist(), "labels": labels,
               "sigs": sigs, "boundary": bounds}, open(out, "w"))
    from collections import Counter
    print(name, Counter(labels).most_common(), f"{len(bounds)} boundary points")


if __name__ == "__main__":
    name, c1, c2, c3, n, out = sys.argv[1:7]
    parse = lambda t: np.array([eval(x, {"phi": PHI}) for x in t.split(",")], float)  # noqa: E731
    main(name, [parse(c1), parse(c2), parse(c3)], int(n), out)
