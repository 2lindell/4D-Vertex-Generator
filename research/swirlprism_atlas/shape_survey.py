"""Does each name stand for one polytope? (shape_survey.json)

The classifier names a shape by its counts (the sizes of its cell, face and edge orbits and its edge valences); two
different polytopes can share them. Here each shape is sampled from the pieces already found, and each point is
classified with its key too (classify_fast.py: how the cells, faces and edges fit together at a vertex, which every
vertex is like); a name whose points fall into more than one key covers that many different polytopes.

The points, all built from the exact data (nothing random):
  * the lines: points along every exact line piece (exact_lines.json);
  * the walls: along each of a wall's edge lines lying on one of its exact surfaces (exact_defs.json), steps into the
    surface, square to the line, both ways (2e-4, 1e-3, 5e-3, 2e-2), each put back on the surface; kept where it is
    the wall;
  * the regions: from each wall point, steps off the surface along its normal, both ways (2e-4, 1e-3): every region
    is bounded by walls, so each is reached from each of its walls;
  * the exact points (exact_points.json).
Walls, lines and points are classified as on a wall (the hull tolerance relaxed, as lab_edge).

    python shape_survey.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from core import library, shape_key

OUT = "shape_survey.json"
ALONG = 10                          # (points along each line piece used for the walls)
INTO = (2e-4, 1e-3, 5e-3, 2e-2)
OFF = (2e-4, 1e-3)
CELL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])
U = np.linalg.svd(np.ones((1, 4)))[2][1:]          # (the betas' sum-zero directions)
LIB = {s.name: s for s in sum(library(), [])}


def inside(b):
    return (b @ CELL.T > 1e-9).all()


def normal(S, b):
    g = np.asarray(S.grad(b), float)
    g = g - g.mean()
    return g / (np.linalg.norm(g) + 1e-300)


def onto(S, b, steps=30):
    """b put on the surface S (sum 1 kept), by least-change Newton steps."""
    b = np.asarray(b, float).copy()
    for _ in range(steps):
        f = float(np.ravel(S.f(b[None]))[0])
        g = np.asarray(S.grad(b), float)
        g = g - g.mean()
        if abs(f) < 1e-15 or not g.any():
            break
        b = b - f * g / (g @ g)
    return b / b.sum()


def dist(S, b):
    return abs(float(np.ravel(S.f(b[None]))[0])) / (np.linalg.norm(np.asarray(S.grad(b)) - np.mean(S.grad(b))) + 1e-300)


def wall_points(work, defs):
    """(point, wall) pairs: from each edge line of each wall, steps into each of its surfaces the line lies on."""
    XL = work["exact"]["lines"]
    out = []
    for w, wd in work["walls"].items():
        for L in wd["edges"]:
            if L not in XL:
                continue
            for n in defs["walls"].get(w, {}).get("surface", []):
                S = LIB[n]
                for P in XL[L]["points"]:
                    P = np.asarray(P, float)
                    if len(P) < 3:
                        continue
                    idx = np.unique(np.linspace(1, len(P) - 2, ALONG).astype(int))
                    if max(dist(S, P[i]) for i in idx) > 1e-7:
                        continue                      # (this piece of the line is not on this surface)
                    for i in idx:
                        x, t = P[i], P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
                        t = t - t.mean()
                        nS = normal(S, x)
                        # (square to the line, within the surface: in the sum-zero 3-space, across both)
                        a, c = U @ t, U @ nS
                        d = np.cross(a, c) @ U
                        if np.linalg.norm(d) < 1e-12:
                            continue
                        d = d / np.linalg.norm(d)
                        for s in (1, -1):
                            for e in INTO:
                                y = onto(S, x + s * e * d)
                                if inside(y):
                                    out.append((y, w, n))
    return out


def job(item):
    b, tol = item
    return shape_key(b, tol)


def main():
    work = json.load(open("working_data.json"))
    defs = json.load(open("exact_defs.json"))
    W = wall_points(work, defs)
    print(f"{len(W)} wall points to try", flush=True)
    recs = []
    with Pool() as pool:
        WL = pool.map(job, [(y, 1e-7) for y, _, _ in W], chunksize=16)
        # (the region points: off each wall point that is a wall, both ways along the surface's normal)
        R = []
        for (y, w, n), (name, key) in zip(W, WL):
            recs.append((name, key, y, f"wall {w}"))
            if name in work["walls"]:
                m = normal(LIB[n], y)
                for s in (1, -1):
                    for e in OFF:
                        z = y + s * e * m
                        if inside(z):
                            R.append((z, f"off {name}"))
        print(f"{len(R)} region points", flush=True)
        RL = pool.map(job, [(z, 1e-9) for z, _ in R], chunksize=16)
        recs += [(name, key, z, src) for (z, src), (name, key) in zip(R, RL)]
        Lp = []
        for L, v in work["exact"]["lines"].items():
            for P in v["points"]:
                P = np.asarray(P, float)
                for i in range(1, len(P) - 1, max(1, (len(P) - 2) // 30)):
                    Lp.append((P[i], f"line {L}"))
        Lp += [(np.asarray(p["float"], float), f"point {p['name']}") for p in work["exact"]["points"]]
        print(f"{len(Lp)} line and point points", flush=True)
        LL = pool.map(job, [(b, 1e-7) for b, _ in Lp], chunksize=16)
        recs += [(name, key, b, src) for (b, src), (name, key) in zip(Lp, LL)]
    by = defaultdict(lambda: defaultdict(list))
    for name, key, b, src in recs:
        if name != "ERR" and key:
            by[name][key].append((np.round(b, 10).tolist(), src))
    out = {"_about": __doc__.strip().split("\n\n")[0], "names": {}}
    for name, keys in sorted(by.items()):
        out["names"][name] = {k: {"count": len(v), "sources": sorted({s for _, s in v}), "points": [b for b, _ in v]}
                              for k, v in keys.items()}
    json.dump(out, open(OUT, "w"))
    met = set(out["names"])
    shapes = set(work["shapes"])
    print(f"{len(met)} names met; shapes never met: {sorted(shapes - met)}", flush=True)
    print(f"names met that are not in the working data: {sorted(n[:60] for n in met - shapes)}", flush=True)
    for n, ks in out["names"].items():
        if len(ks) > 1:
            print(f"  {n[:40]}: " + ", ".join(f"{k} x{v['count']}" for k, v in sorted(ks.items(), key=lambda kv: -kv[1]['count'])), flush=True)


if __name__ == "__main__":
    main()
