"""The curved walls X10 and X33: their exact surfaces, their extent in the half tetrahedron, and meshes to draw.

X10 lies on the quadric  b1 b4 - b2 b3 + b3^2 - b4^2 = 0  and X33 on a golden quartic (xsurface_x33.json, fitted to 60
X33 points found by coincidence and snapped to golden coefficients; residual 5e-16). Each surface is parametrised:
  X10: the second intersection with the quadric of each line through the sample s0 = (phi, 2, phi, 1), over a grid of
       line directions (theta, psi) in the hyperplane sum(beta) = 1;
  X33: a square grid in the tangent plane at its sample (1, phi, 1+2phi, 1+phi), each point pushed onto the surface by
       Newton steps along the gradient.
The grid points are classified (fexact._lab), growing the grid outward from the shape until it is surrounded. The grid
triangles are then marched: corners in the shape are kept, and every edge from a corner in the shape to one outside is
bisected in the parameter plane for the boundary point (and the shape beyond it, or, at a face or mirror of the half
tetrahedron, the shape on the face there). The result, xsurfaces.json, holds per shape the triangles (as betas), the
boundary polylines with the shape beyond each boundary point, and the equation.

    python xsurface.py X10 [cache.json] [grid.json]
    python xsurface.py X33 [cache.json] [grid.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from fexact import _lab, _lab_loose

PHI = (1 + 5 ** 0.5) / 2
STEPS = 7          # bisection steps along a grid edge (1/128 of the grid spacing)


# ---------------------------------------------------------------------------------------------------------- surfaces
def _x10():
    mons = [(i, j) for i in range(4) for j in range(i, 4)]
    c = np.zeros(10)
    c[3], c[5], c[7], c[9] = 1, -1, 1, -1          # b1b4 - b2b3 + b3^2 - b4^2
    Q = np.zeros((4, 4))
    for v, (i, j) in zip(c, mons):
        Q[i, j] += v / (1 if i == j else 2)
        Q[j, i] = Q[i, j]
    s0 = np.array([PHI, 2, PHI, 1.0])
    s0 /= s0.sum()
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    n1, n2 = 60, 120
    ths = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, n1)

    def param(x):                                      # x = (theta, psi), continuous
        th, ps = x
        u = U.T @ np.array([np.cos(th) * np.cos(ps), np.cos(th) * np.sin(ps), np.sin(th)])
        qu = u @ Q @ u
        if abs(qu) < 1e-14:
            return None
        p = s0 + (-2 * (s0 @ Q @ u) / qu) * u
        return p

    def at(key):                                       # grid key (i, j): i a row of theta, j any integer (psi wraps)
        return np.array([ths[key[0]], 2 * np.pi * key[1] / n2])

    def norm_key(key):
        return (key[0], key[1] % n2)

    def neighbours(key):
        i, j = key
        return [norm_key((i + di, j + dj)) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if n1 // 2 - 1 <= i + di < n1]

    def cells():                                       # each line through s0 once: theta >= 0 (and one row below)
        for i in range(n1 // 2 - 1, n1 - 1):
            for j in range(n2):
                yield (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)

    return {"param": param, "at": at, "norm_key": norm_key, "neighbours": neighbours, "cells": cells,
            "equation": "β1β4 − β2β3 + β3² − β4² = 0", "kind": "a quadric"}


def _x33():
    S = json.load(open("xsurface_x33.json"))
    coeffs, mons = np.array(S["coeffs"]), [tuple(m) for m in S["mons"]]
    s0 = np.array(S["points"][0], float)
    s0 /= s0.sum()

    def F(b):
        return float(sum(c * np.prod(np.asarray(b)[list(m)]) for c, m in zip(coeffs, mons) if c))

    def grad(b):
        g = np.array([(F(b + e * 1e-7) - F(b - e * 1e-7)) / 2e-7 for e in np.eye(4)])
        return g - g.mean()

    n = grad(s0)
    n /= np.linalg.norm(n)
    B = np.linalg.svd(np.vstack([np.ones(4), n]))[2][2:]
    h = 0.01

    def param(x):
        p = s0 + x[0] * B[0] + x[1] * B[1]
        for _ in range(30):                            # Newton along the gradient
            g = grad(p)
            gg = g @ g
            if gg < 1e-30:
                return None
            step = F(p) / gg
            p = p - step * g
            if abs(step) < 1e-16:
                break
        return p if abs(F(p)) < 1e-13 else None

    def at(key):
        return np.array([key[0] * h, key[1] * h])

    def neighbours(key):
        i, j = key
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

    def cells_from(known):
        def cells():
            for (i, j) in list(known):
                yield (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
        return cells

    return {"param": param, "at": at, "norm_key": lambda k: k, "neighbours": neighbours, "cells_from": cells_from,
            "equation": S["text"], "kind": "a golden quartic"}


SURF = {"X10": _x10, "X33": _x33}


# ------------------------------------------------------------------------------------------------------------ labels
def in_half(p):
    return p is not None and p.min() > -1e-12 and p[2] >= p[3] - 1e-12 and abs(p.sum() - 1) < 1e-9


_S = None


def _init(tid):
    global _S
    _S = SURF[tid]()


def _label_job(key):
    p = _S["param"](_S["at"](key))
    if not in_half(p):
        return key, None, "OUT"
    return key, p.tolist(), _lab(p)


def _edge_job(args):
    """Bisect the grid edge from a corner in the shape to one outside: the boundary point (on the surface) and the
    shape beyond it (at a face or mirror of the half tetrahedron: the shape on the face there)."""
    tid, x_in, x_out, lab_out = args
    x_in, x_out = np.array(x_in), np.array(x_out)
    P = _S["param"]
    if lab_out == "OUT":                               # the cell face: bisect the cheap in-cell test
        for _ in range(40):
            m = (x_in + x_out) / 2
            if in_half(P(m)):
                x_in = m
            else:
                x_out = m
        p = P(x_in)
        return p.tolist(), "face:" + _lab_loose(p)
    beyond = lab_out
    for _ in range(STEPS):
        m = (x_in + x_out) / 2
        pm = P(m)
        lab = _lab(pm) if in_half(pm) else "OUT"
        if lab == tid:
            x_in = m
        else:
            x_out, beyond = m, lab
            if lab == "OUT":                           # (the face comes first along this edge)
                return _edge_job((tid, x_in.tolist(), x_out.tolist(), "OUT"))
    return P((x_in + x_out) / 2).tolist(), beyond


# -------------------------------------------------------------------------------------------------------------- main
def main(tid, cache_path, grid_path=None, procs=4):
    _init(tid)
    S = _S
    known = {}
    if os.path.exists(cache_path):
        known = {tuple(json.loads(k)): tuple(v) for k, v in json.load(open(cache_path)).items()}
    elif grid_path:                                     # the first grid (xsurface's predecessor scripts)
        g = json.load(open(grid_path))
        if tid == "X10":
            n2 = g["n2"]
            for k, r in enumerate(g["res"]):
                known[(k // n2, k % n2)] = (r[0], r[1]) if r else (None, "OUT")
        else:
            m = g["m"]
            for k, r in enumerate(g["res"]):
                known[(k // m - m // 2, k % m - m // 2)] = (r[0], r[1]) if r else (None, "OUT")

    def save():
        json.dump({json.dumps(list(k)): list(v) for k, v in known.items()}, open(cache_path, "w"))

    with Pool(procs, initializer=_init, initargs=(tid,)) as pool:
        # grow the grid until every grid point of the shape has its neighbours classified
        while True:
            front = sorted({S["norm_key"](nb) for k, (p, lab) in known.items() if lab == tid
                            for nb in S["neighbours"](k)} - set(known))
            print(f"{tid}: {sum(v[1] == tid for v in known.values())} grid points in the shape, {len(front)} to classify",
                  flush=True)
            if not front:
                break
            for key, p, lab in pool.imap_unordered(_label_job, front, chunksize=4):
                known[key] = (p, lab)
            save()

        cells = S["cells"] if "cells" in S else S["cells_from"](known)
        inside = lambda k: known.get(S["norm_key"](k), (None, "OUT"))[1] == tid
        tri_list = []
        for q in cells():
            for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                if any(inside(k) for k in t) and all(S["norm_key"](k) in known for k in t):
                    tri_list.append(t)
        edges = {}
        for t in tri_list:
            for a in range(3):
                u, v = t[a], t[(a + 1) % 3]
                if inside(u) != inside(v):
                    ui, vo = (u, v) if inside(u) else (v, u)
                    key = (S["norm_key"](ui), S["norm_key"](vo))
                    edges.setdefault(key, (S["at"](ui).tolist(), S["at"](vo).tolist(),
                                           known[S["norm_key"](vo)][1]))
        print(f"{tid}: bisecting {len(edges)} boundary edges", flush=True)
        keys = list(edges)
        res = pool.map(_edge_job, [(tid, *edges[k]) for k in keys], chunksize=2)
        crossing = dict(zip(keys, res))

    def beta(k):
        return np.asarray(known[S["norm_key"](k)][0], float)

    def cross(u, v):
        ui, vo = (u, v) if inside(u) else (v, u)
        return (S["norm_key"](ui), S["norm_key"](vo))

    tris, segs = [], []
    for t in tri_list:
        flags = [inside(k) for k in t]
        if all(flags):
            tris.append([beta(k).tolist() for k in t])
            continue
        poly = []                                      # the triangle clipped to the shape (marching triangles)
        ends = []
        for a in range(3):
            u, v = t[a], t[(a + 1) % 3]
            if inside(u):
                poly.append(beta(u).tolist())
            if inside(u) != inside(v):
                e = cross(u, v)
                poly.append(crossing[e][0])
                ends.append(e)
        for k in range(1, len(poly) - 1):
            tris.append([poly[0], poly[k], poly[k + 1]])
        if len(ends) == 2:
            segs.append(tuple(ends))

    # chain the boundary segments into polylines
    adj = {}
    for a, b in segs:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, lines = set(), []
    for start in adj:
        if start in seen:
            continue
        # walk to one end first (an open line), then along
        cur, prev = start, None
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if len(adj[cur]) < 2 or not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
        line, prev = [cur], None
        seen.add(cur)
        while True:
            nxt = [n for n in adj[line[-1]] if n != prev and n not in seen]
            if not nxt:
                if line[0] in adj[line[-1]] and len(line) > 2:
                    line.append(line[0])                # closed
                break
            prev = line[-1]
            line.append(nxt[0])
            seen.add(nxt[0])
        lines.append([{"beta": crossing[e][0], "beyond": crossing[e][1]} for e in line])

    out = json.load(open("xsurfaces.json")) if os.path.exists("xsurfaces.json") else {}
    out[tid] = {"equation": S["equation"], "kind": S["kind"], "tris": tris, "outline": lines,
                "beyond": Counter(c[1] for c in crossing.values()).most_common()}
    json.dump(out, open("xsurfaces.json", "w"))
    print(f"{tid}: {len(tris)} triangles, {len(lines)} boundary lines; beyond: {out[tid]['beyond']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f"xsurface_{sys.argv[1]}_cache.json",
         sys.argv[3] if len(sys.argv) > 3 else None)
