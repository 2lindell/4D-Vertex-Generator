"""Does each name stand for one polytope? (shape_survey.json)

The classifier names a shape by its counts (the sizes of its cell, face and edge orbits and its edge valences); two
different polytopes can share them. Here every shape is sampled across its whole extent and each point is classified
with its key too (classify_fast.py: how the cells, faces and edges fit together at a vertex, which every vertex is
like); a name whose points fall into more than one key covers that many different polytopes.

Points sampled:
  * a grid filling the half-cell (b >= 0, b3 >= b4), set slightly off every plane: the regions;
  * on each wall's exact surfaces as placed in the half-cell (exact_defs.json), the grid points near each surface put
    on it: the walls;
  * along each exact line piece (exact_lines.json): the lines;
  * the exact points (exact_points.json).
The walls, lines and points are classified as on a wall (the hull tolerance relaxed, as lab_edge).

    python shape_survey.py [N]          (N: the grid's divisions, default 28)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from core import library, shape_key
from exact_math import project

OUT = "shape_survey.json"


def in_half(x, tol=1e-9):
    return (x > -tol).all() and x[2] >= x[3] - tol


def grid(N):
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            for k in range(N + 1 - i - j):
                l_ = N - i - j - k
                if k >= l_:
                    pts.append(np.array([i, j, k, l_], float) / N)
    return np.array(pts)


def job(item):
    b, tol, src = item
    name, key = shape_key(b, hull_tol=tol)
    return name, key, [round(float(v), 10) for v in b], src


def main(N=28):
    LIB = {s.name: s for s in sum(library(), [])}
    work = json.load(open("working_data.json"))
    G = grid(N)
    # (off every plane: a small shift with irrational parts, summing to 0)
    shift = 1e-4 * np.array([0.6180339887, -0.2360679775, -0.1458980338, -0.2360679775])
    items = [(x + shift, 1e-9, "grid") for x in G if in_half(x + shift, 0)]
    near = 0.04
    for w, names in work["exact"]["wall_surfaces"].items():
        for n in names:
            S = LIB.get(n)
            if S is None:
                continue
            got = []
            for x in grid(N // 2 + 6):
                v = float(np.ravel(S.f(x[None]))[0])
                if abs(v) > near * 4:
                    continue
                y = project([S], x)
                if np.linalg.norm(y - x) < near and in_half(y, 1e-12) and y.min() > 1e-6 and y[2] - y[3] > 1e-6:
                    got.append(y)
            for y in got:
                items.append((y, 1e-7, f"on {w}"))
    for L, v in work["exact"]["lines"].items():
        for P in v["points"]:
            P = np.asarray(P, float)
            if len(P) < 3:
                continue
            for i in np.linspace(1, len(P) - 2, min(12, len(P) - 2)).astype(int):
                items.append((P[i], 1e-7, f"on {L}"))
    for p in work["exact"]["points"]:
        items.append((np.asarray(p["float"], float), 1e-7, f"at {p['name']}"))
    print(f"{len(items)} points to classify", flush=True)
    with Pool() as pool:
        recs = pool.map(job, items, chunksize=20)
    # (each name, the keys its points fall into)
    by = defaultdict(lambda: defaultdict(list))
    for name, key, b, src in recs:
        if name != "ERR" and key:
            by[name][key].append((b, src))
    out = {"_about": __doc__.strip().split("\n\n")[0], "grid": N, "names": {}}
    for name, keys in sorted(by.items()):
        out["names"][name] = {k: {"count": len(v), "sources": sorted({s for _, s in v}),
                                  "points": [b for b, _ in v]} for k, v in keys.items()}
    json.dump(out, open(OUT, "w"))
    split = {n: {k: v["count"] for k, v in ks.items()} for n, ks in out["names"].items() if len(ks) > 1}
    print(f"{len(out['names'])} names met; more than one polytope under: {split}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 28)
