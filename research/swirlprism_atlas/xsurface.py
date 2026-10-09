"""The curved walls X10, X33, X48 and X51: their exact surfaces, their extent in the half tetrahedron, and meshes to draw.

X10 lies on the quadric  b1 b4 - b2 b3 + b3^2 - b4^2 = 0, X48 on the ruled quadric
phi b1 (b2 - b3) + b4 (b2 - b1) = 0, X51 on the ruled quadric
b2 (b2 - b3) + phi b4 (b2 - b1) = 0 (which holds its four traced lines), and X33 on a golden quartic (xsurface_x33.json, fitted to 60
X33 points found by coincidence and snapped to golden coefficients; residual 5e-16).

The surface is meshed directly from its equation, by marching tetrahedra over a uniform grid in the hyperplane
sum(beta) = 1 (each mesh vertex then moved onto the surface exactly by Newton steps), so the mesh is even and its
triangles share their edges. Only the grid cells around the shape are meshed: starting from cells holding known points
of the shape, the cells next to any cell with a mesh vertex in the shape are added until the shape is surrounded. The
mesh vertices are classified (fexact._lab_loose), and the triangles are marched: corners in the shape are kept, and
each mesh edge from a corner in the shape to one outside is bisected (along the edge, pushed onto the surface) for the
boundary point and the shape beyond it (at a face or mirror of the half tetrahedron: the shape on the face there).
The result, xsurfaces.json, holds per shape the triangles (as betas), the boundary polylines with the shape beyond each
boundary point, and the equation.

    python xsurface.py X10|X33|X48|X51 [h] [cache.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import permutations
from multiprocessing import Pool

import numpy as np

from fexact import _lab_loose

PHI = (1 + 5 ** 0.5) / 2
STEPS = 6            # bisection steps along a boundary mesh edge
U = np.linalg.svd(np.ones((1, 4)))[2][1:]            # sum-zero directions: beta = C0 + U^T y
C0 = np.full(4, 0.25)


# ---------------------------------------------------------------------------------------------------------- surfaces
def surface(tid):
    if tid == "X10":
        mons = [(0, 3), (1, 2), (2, 2), (3, 3)]
        coeffs = [1.0, -1.0, 1.0, -1.0]
        eq, kind = "β1β4 − β2β3 + β3² − β4² = 0", "a quadric"
    elif tid == "X48":                                   # (holds X48's four traced lines: it is ruled by them)
        mons = [(0, 1), (0, 2), (1, 3), (0, 3)]
        coeffs = [PHI, -PHI, 1.0, -1.0]
        eq, kind = "φ·β1(β2 − β3) + β4(β2 − β1) = 0", "a ruled quadric"
    elif tid == "X51":                                   # (holds X51's four traced lines: it is ruled by them)
        mons = [(1, 1), (1, 2), (1, 3), (0, 3)]
        coeffs = [1.0, -1.0, PHI, -PHI]
        eq, kind = "β2(β2 − β3) + φ·β4(β2 − β1) = 0", "a ruled quadric"
    else:
        S = json.load(open("xsurface_x33.json"))
        mons = [tuple(m) for m, c in zip(S["mons"], S["coeffs"]) if c]
        coeffs = [c for c in S["coeffs"] if c]
        eq, kind = S["text"], "a golden quartic"
    return {"mons": mons, "coeffs": np.array(coeffs, float), "equation": eq, "kind": kind}


def F(sf, B):
    """The polynomial at betas B (n, 4)."""
    B = np.atleast_2d(B)
    out = np.zeros(len(B))
    for c, m in zip(sf["coeffs"], sf["mons"]):
        out += c * np.prod(B[:, list(m)], axis=1)
    return out


def grad(sf, b):
    g = np.zeros(4)
    for c, m in zip(sf["coeffs"], sf["mons"]):
        for k in range(len(m)):
            rest = m[:k] + m[k + 1:]
            g[m[k]] += c * np.prod(b[list(rest)])
    return g - g.mean()                                  # (stay in sum(beta) = 1)


def project(sf, b):
    b = np.array(b, float)
    for _ in range(40):
        g = grad(sf, b)
        gg = g @ g
        if gg < 1e-30:
            break
        step = F(sf, b)[0] / gg
        b = b - step * g
        if abs(step) < 1e-17:
            break
    return b


def in_half(b):
    return b.min() > -1e-12 and b[2] >= b[3] - 1e-12


# ------------------------------------------------------------------------------------------------------------- jobs
_SF = None


def _init(tid):
    global _SF
    _SF = surface(tid)


def _label(b):
    b = np.asarray(b, float)
    return _lab_loose(b / b.sum()) if in_half(b) else "OUT"


def _edge_job(args):
    """Bisect the mesh edge from a vertex in the shape to one outside: the boundary point (on the surface) and the
    shape beyond it, or, at a face or mirror of the half tetrahedron, the shape on the face there."""
    tid, a, b, lab_out = args
    a, b = np.array(a), np.array(b)
    lo, hi, beyond = 0.0, 1.0, lab_out
    if lab_out == "OUT":
        for _ in range(40):
            m = (lo + hi) / 2
            if in_half(project(_SF, a + m * (b - a))):
                lo = m
            else:
                hi = m
        x = project(_SF, a + lo * (b - a))
        return x.tolist(), "face:" + _lab(np.clip(x, 0, None))
    for _ in range(STEPS):
        m = (lo + hi) / 2
        x = project(_SF, a + m * (b - a))
        lab = _label(x)
        if lab == tid:
            lo = m
        elif lab == "OUT":
            return _edge_job((tid, (a + lo * (b - a)).tolist(), x.tolist(), "OUT"))
        else:
            hi, beyond = m, lab
    return project(_SF, a + (lo + hi) / 2 * (b - a)).tolist(), beyond


# ------------------------------------------------------------------------------------------------------------- mesh
CUBE = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
# the cube cut into six tetrahedra along its main diagonal (the same cut in every cube, so faces match)
TETS = []
for perm in permutations(range(3)):
    path = [np.zeros(3, int)]
    for ax in perm:
        nxt = path[-1].copy()
        nxt[ax] = 1
        path.append(nxt)
    TETS.append([int(np.flatnonzero((CUBE == p).all(axis=1))[0]) for p in path])


def main(tid, h=0.012, cache_path=None, procs=4):
    sf = surface(tid)
    cache_path = cache_path or f"xsurface_{tid}_labels.json"
    labels = {}
    if os.path.exists(cache_path):
        labels = {tuple(json.loads(k)): v for k, v in json.load(open(cache_path)).items()}

    def node_beta(n):
        return C0 + U.T @ (h * np.asarray(n, float))

    Fn = {}

    def fval(n):
        if n not in Fn:
            Fn[n] = F(sf, node_beta(n))[0]
        return Fn[n]

    verts = {}                                           # (node, node) -> beta on the surface

    def vert(n1, n2):
        key = (n1, n2) if n1 < n2 else (n2, n1)
        if key not in verts:
            f1, f2 = fval(key[0]), fval(key[1])
            t = f1 / (f1 - f2)
            verts[key] = project(sf, node_beta(key[0]) + t * (node_beta(key[1]) - node_beta(key[0])))
        return key

    def cell_tris(c):
        """Marching tetrahedra in cube c: triangles as triples of vertex keys."""
        nodes = [tuple(int(v) for v in np.add(c, d)) for d in CUBE]
        vals = [fval(n) for n in nodes]
        out = []
        for tet in TETS:
            ns = [nodes[i] for i in tet]
            fs = [vals[i] for i in tet]
            pos = [i for i in range(4) if fs[i] > 0]
            neg = [i for i in range(4) if fs[i] <= 0]
            if not pos or not neg:
                continue
            if len(pos) == 1 or len(neg) == 1:
                one, rest = (pos, neg) if len(pos) == 1 else (neg, pos)
                out.append(tuple(vert(ns[one[0]], ns[r]) for r in rest))
            else:
                a, b = pos
                c_, d = neg
                q = [vert(ns[a], ns[c_]), vert(ns[a], ns[d]), vert(ns[b], ns[d]), vert(ns[b], ns[c_])]
                out += [(q[0], q[1], q[2]), (q[0], q[2], q[3])]
        return out

    # seed cells: those holding the shape's known points (the samples of the first mesh, if any)
    seeds = []
    if os.path.exists("xsurfaces.json"):
        old = json.load(open("xsurfaces.json")).get(tid, {})
        seeds = [b for t in old.get("tris", [])[::7] for b in t[:1]]
    if not seeds and tid in ("X48", "X51"):            # points along its lines
        from surface_from_lines import lines_of
        lines = lines_of(tid)
        seeds = [(np.array(a) / sum(a)) * (1 - t) + (np.array(b) / sum(b)) * t for a, b in lines for t in (0.2, 0.5, 0.8)]
    if not seeds:
        seeds = [json.load(open("xsurface_x33.json"))["points"][0]] if tid == "X33" else [[PHI, 2, PHI, 1]]
    active = {tuple(int(v) for v in np.floor(U @ (np.asarray(b, float) / np.sum(b) - C0) / h)) for b in seeds}
    done, tris_by_cell = set(), {}
    with Pool(procs, initializer=_init, initargs=(tid,)) as pool:
        _init(tid)
        while True:
            todo = active - done
            if not todo:
                break
            new_v = set()
            for c in todo:
                tris_by_cell[c] = cell_tris(c)
                for t in tris_by_cell[c]:
                    new_v.update(k for k in t if k not in labels)
            done |= todo
            new_v = sorted(new_v)
            for k, lab in zip(new_v, pool.map(_label, [verts[k] for k in new_v], chunksize=4)):
                labels[k] = lab
            json.dump({json.dumps([list(a), list(b)]): v for (a, b), v in labels.items()}, open(cache_path, "w"))
            # grow: every cell next to a cell with a vertex in the shape (and crossed by the surface)
            grow = set()
            for c in todo:
                if any(labels[k] == tid for t in tris_by_cell[c] for k in t):
                    grow.update(tuple(int(v) for v in np.add(c, d)) for d in
                                [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
            active |= grow
            n_in = sum(1 for v in labels.values() if v == tid)
            print(f"{tid}: {len(done)} cells meshed, {len(labels)} vertices classified ({n_in} in the shape), "
                  f"{len(active - done)} cells to mesh", flush=True)

        tri_list = [t for c in done for t in tris_by_cell[c]]
        inside = lambda k: labels.get(k) == tid
        edges = {}
        for t in tri_list:
            if not any(inside(k) for k in t):
                continue
            for a in range(3):
                u, v = t[a], t[(a + 1) % 3]
                if inside(u) != inside(v):
                    ui, vo = (u, v) if inside(u) else (v, u)
                    edges.setdefault((ui, vo), (verts[ui].tolist(), verts[vo].tolist(), labels[vo]))
        print(f"{tid}: bisecting {len(edges)} boundary edges", flush=True)
        keys = list(edges)
        crossing = dict(zip(keys, pool.map(_edge_job, [(tid, *edges[k]) for k in keys], chunksize=2)))

    def cross(u, v):
        return (u, v) if inside(u) else (v, u)

    tris, segs = [], []
    for t in tri_list:
        flags = [inside(k) for k in t]
        if not any(flags):
            continue
        if all(flags):
            tris.append([verts[k].tolist() for k in t])
            continue
        poly, ends = [], []
        for a in range(3):
            u, v = t[a], t[(a + 1) % 3]
            if inside(u):
                poly.append(verts[u].tolist())
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
    for start in sorted(adj, key=lambda e: len(adj[e])):      # (open lines start at their ends)
        if start in seen:
            continue
        line = [start]
        seen.add(start)
        while True:
            nxt = [n for n in adj[line[-1]] if n not in seen]
            if not nxt:
                if len(line) > 2 and line[0] in adj[line[-1]]:
                    line.append(line[0])                    # closed
                break
            line.append(nxt[0])
            seen.add(nxt[0])
        lines.append([{"beta": crossing[e][0], "beyond": crossing[e][1]} for e in line])

    out = json.load(open("xsurfaces.json")) if os.path.exists("xsurfaces.json") else {}
    out[tid] = {"equation": sf["equation"], "kind": sf["kind"], "h": h, "tris": tris, "outline": lines,
                "beyond": Counter(c[1] for c in crossing.values()).most_common()}
    json.dump(out, open("xsurfaces.json", "w"))
    print(f"{tid}: {len(tris)} triangles, {len(lines)} boundary lines; beyond: {out[tid]['beyond']}", flush=True)


def _lab(b):
    return _lab_loose(np.asarray(b) / np.sum(b))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.012, sys.argv[3] if len(sys.argv) > 3 else None)
