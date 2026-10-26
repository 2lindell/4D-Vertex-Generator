"""Each wall's piece, built from its equations alone (pieces.json): the part of the wall's surface, in the half-cell,
on the same side of each of its edge surfaces (wall_equations.json) as a point known to be on the wall (its example
seed, optimal_seeds.json, carried onto the wall's own surface by the group). Meshed by marching tetrahedra over a grid
in the 3-space sum(b) = 1; each mesh vertex is on the surface (Newton steps), and where the piece ends the boundary
point is solved exactly onto the wall's surface and the bounding surface (an edge's surface, or a face of the
half-cell or its mirror, which only the views cut along). Each boundary point is labelled with what bounds it there.

The piece is checked: points of it should classify as the wall.

    python pieces.py [X14,X41,...] [h]
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

import optimal_seeds as osd
from probe import _one
from t2_pieces import extended
from xsurface import CUBE, TETS, U, C0

FOLD = np.eye(4)[[1, 0, 3, 2]]
HALF = {"face b1=0": np.eye(4)[0], "face b2=0": np.eye(4)[1], "face b3=0": np.eye(4)[2], "face b4=0": np.eye(4)[3],
        "mirror b3=b4": np.array([0, 0, 1, -1.0])}


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def grad(s, x):
    if isinstance(s, osd.Plane):
        g = s.n.astype(float)
    else:
        e = 1e-7
        g = np.array([(s.f((x + e * u)[None]).ravel()[0] - s.f((x - e * u)[None]).ravel()[0]) / (2 * e) for u in np.eye(4)])
    return g - g.mean()


def fval(s, x):
    return float(np.atleast_1d(s.f(np.atleast_2d(x)).ravel())[0])


def project(surfs, x, steps=30):
    """x moved onto all the surfaces (sum(b) = 1 kept), by Newton steps (least change)."""
    x = np.asarray(x, float).copy()
    for _ in range(steps):
        F = np.array([fval(s, x) for s in surfs])
        if np.abs(F).max() < 1e-14:
            break
        J = np.array([grad(s, x) for s in surfs])
        x = x - J.T @ np.linalg.lstsq(J @ J.T, F, rcond=None)[0]
    return x


class Piece:
    def __init__(self, w, S, bounds, ref):
        self.w, self.S, self.bounds = w, S, bounds            # bounds: [(name, surface, sign)]
        self.ref = ref

    def inside(self, x):
        """None if inside, else what it is beyond (the first condition that fails)."""
        if (x < -1e-12).any():
            k = int(np.argmin(x))
            return f"face b{k + 1}=0"
        if x[2] < x[3] - 1e-12:
            return "mirror b3=b4"
        for name, s, sg in self.bounds:
            if np.sign(fval(s, x)) == -sg:
                return name
        return None

    def bound_surface(self, name):
        if name in HALF:
            return osd.Plane(HALF[name], name)
        return next(s for n, s, _ in self.bounds if n == name)


def build(w, S, bounds, ref, h=0.008, max_cells=60000):
    P = Piece(w, S, bounds, ref)
    node = lambda n: C0 + U.T @ (h * np.asarray(n, float))
    Fn = {}

    def F(n):
        if n not in Fn:
            Fn[n] = fval(S, node(n))
        return Fn[n]
    verts, lab = {}, {}

    def vert(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in verts:
            fa, fb = F(k[0]), F(k[1])
            t = fa / (fa - fb)
            x = project([S], node(k[0]) + t * (node(k[1]) - node(k[0])))
            verts[k] = nrm(x)
            lab[k] = P.inside(verts[k])
        return k

    def cell_tris(c):
        ns = [tuple(int(v) for v in np.add(c, d)) for d in CUBE]
        vals = [F(n) for n in ns]
        out = []
        for tet in TETS:
            q = [ns[i] for i in tet]
            fs = [vals[i] for i in tet]
            pos = [i for i in range(4) if fs[i] > 0]
            neg = [i for i in range(4) if fs[i] <= 0]
            if not pos or not neg:
                continue
            if len(pos) == 1 or len(neg) == 1:
                one, rest = (pos, neg) if len(pos) == 1 else (neg, pos)
                out.append(tuple(vert(q[one[0]], q[r]) for r in rest))
            else:
                a, b_ = pos
                c_, d = neg
                k = [vert(q[a], q[c_]), vert(q[a], q[d]), vert(q[b_], q[d]), vert(q[b_], q[c_])]
                out += [(k[0], k[1], k[2]), (k[0], k[2], k[3])]
        return out
    start = tuple(int(v) for v in np.floor(U @ (ref - C0) / h))
    todo, done, tris = {start}, set(), {}
    nb = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    while todo and len(done) < max_cells:
        c = todo.pop()
        done.add(c)
        tris[c] = cell_tris(c)
        if any(lab[k] is None for t in tris[c] for k in t):
            todo |= {tuple(np.add(c, d)) for d in nb} - done
    # (triangles: inside corners kept; a mesh edge from inside to outside cut at the exact boundary point)
    cross = {}

    def boundary(ki, ko):
        key = (ki, ko)
        if key not in cross:
            a, b = verts[ki], verts[ko]
            lo, hi = 0.0, 1.0
            for _ in range(40):
                m = (lo + hi) / 2
                x = nrm(project([S], a + m * (b - a)))
                if P.inside(x) is None:
                    lo = m
                else:
                    hi = m
            x = nrm(project([S], a + hi * (b - a)))
            name = P.inside(x) or lab[ko]
            x = nrm(project([S, P.bound_surface(name)], a + lo * (b - a)))
            cross[key] = (x, name)
        return cross[key]
    out_tris, segs = [], []
    for c in done:
        for t in tris[c]:
            ins = [lab[k] is None for k in t]
            if not any(ins):
                continue
            if all(ins):
                out_tris.append([verts[k] for k in t])
                continue
            poly, ends = [], []
            for a in range(3):
                u, v = t[a], t[(a + 1) % 3]
                if lab[u] is None:
                    poly.append(verts[u])
                if (lab[u] is None) != (lab[v] is None):
                    x, name = boundary(u, v) if lab[u] is None else boundary(v, u)
                    poly.append(x)
                    ends.append((x, name))
            for k in range(1, len(poly) - 1):
                out_tris.append([poly[0], poly[k], poly[k + 1]])
            if len(ends) == 2:
                segs.append(ends)
    return out_tris, segs, len(done) >= max_cells


def main(only=None, h=0.008):
    work = json.load(open("working_data.json"))
    eq = json.load(open("wall_equations.json"))["walls"]
    ex = json.load(open("exact_defs.json"))["walls"]
    seeds = json.load(open("optimal_seeds.json"))
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    G = extended()
    GG = np.concatenate([G, G @ FOLD])
    out = json.load(open("pieces.json")) if only else {}
    for w in work["walls"]:
        if only and w not in only:
            continue
        own = ex[w]["surface"]
        if len(own) != 1 or own[0].startswith("plane"):
            print(f"{w}: flat (planes {len(own)}): later")
            continue
        S = LIB[own[0]]
        # (the reference point: the wall's seed carried by the group onto its own surface, in the half-cell)
        ref = None
        b0 = nrm(seeds[w]["beta"]) if w in seeds else None
        if b0 is not None:
            for g in GG:
                y = g @ b0
                if abs(y.sum()) < 1e-12:
                    continue
                y = y / y.sum()
                if (y > -1e-12).all() and y[2] >= y[3] - 1e-12 and abs(fval(S, y)) < 1e-10:
                    ref = y
                    break
        if ref is None:
            print(f"{w}: no point of the wall on its own surface (seed {'missing' if b0 is None else 'not carried onto it'})")
            continue
        bounds = []
        for e, r in eq[w]["edges"].items():
            if r.get("surface") in LIB:
                s = LIB[r["surface"]]
                v = fval(s, ref)
                bounds.append((e, s, int(np.sign(v))))
        tris, segs, capped = build(w, S, bounds, ref, h)
        T = np.array(tris)
        lab_count = Counter(n for sg in segs for _, n in sg)
        # (check: the piece's points classify as the wall)
        rng = np.random.default_rng(0)
        pts = T.mean(1) if len(T) else np.zeros((0, 4))
        pts = np.array([nrm(project([S], p)) for p in pts[rng.choice(len(pts), min(60, len(pts)), replace=False)]]) if len(pts) else pts
        cls = Counter(_one(p)[0][:8] for p in pts)
        share = cls.get(w, 0) / max(1, len(pts))
        out[w] = {"surface": own[0], "ref": ref.round(12).tolist(), "h": h,
                  "bounds": [[e, s.name, sg] for e, s, sg in bounds],
                  "tris": T.round(10).tolist(),
                  "boundary": [[[x.round(10).tolist(), n] for x, n in sg] for sg in segs],
                  "check": {"share_classified_as_wall": round(share, 3), "classified": dict(cls.most_common(5))},
                  "bounded_by": dict(lab_count), "capped": capped}
        print(f"{w}: {len(T)} triangles; bounded by {dict(lab_count)}; points classified {dict(cls.most_common(4))}"
              f"{' (capped)' if capped else ''}", flush=True)
    json.dump(out, open("pieces.json", "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.008)
