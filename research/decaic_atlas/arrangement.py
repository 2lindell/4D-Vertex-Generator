"""The decaic domain as an exact arrangement of planes (arrangement.json).

Every form of forms.py factors into linear forms in the betas (checked: sympy factors each into linear factors), so a
polychoron can change only on planes. Factors whose coefficients all have one sign vanish nowhere in the open chamber;
the others are the candidate planes. The domain beta >= 0, beta_1 >= beta_4 (beta summing to 1) is cut by them into
faces -- cells, polygons, segments, points -- each a set of seeds where every candidate keeps one sign (+, -, 0).

The candidates are found by a fixpoint: the arrangement of the planes known so far is built exactly (vertices are
rational, from three planes and beta summing to 1), each of its faces is given a point inside it, the forms of that
point's polychoron (forms.region_conditions) are computed, and any plane not yet known is added; until no face gives a
new plane. Then each face's polychoron is the same over the face, and the face's point stands for it.

Faces are found exactly, from the vertices: a segment between consecutive vertices on the line of two planes; a polygon
on a plane from the centroid of three of its vertices not in a line (every polygon has three such, and the centroid of
three of any polygon's vertices lies inside it); a cell on each side of each polygon (the polygon's signs with its zero
replaced by + or -).

    python arrangement.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import sympy as sp

import forms as F
import group as G

OUT = G.fname("arrangement")
BOUNDARY = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, -1)]   # (beta_k >= 0, beta_1 >= beta_4)


class NonLinear(ValueError):
    pass


def linear(text):
    p = sp.Poly(sp.sympify(text), *F.B)
    if p.total_degree() != 1:
        raise NonLinear(text)
    c = [int(p.coeff_monomial(v)) for v in F.B]
    g = np.gcd.reduce([abs(x) for x in c if x])
    c = [int(x // g) for x in c]
    first = next(x for x in c if x)
    return tuple(x if first > 0 else -x for x in c)


def candidates(beta):
    """The candidate planes of the seed's polychoron: mixed-sign linear factors of its forms. On a mirror (some
    beta_k = 0) vertices coincide and their labels are not unique, so a form means something only on that mirror: it
    is restricted there (those beta_k set to 0) before factoring, and its factors are planes through the mirror's
    span extended by 0 in the missing betas. A factor that is not linear is reported (NONLINEAR) and kept out."""
    zero = [v for v, b in zip(F.B, beta) if abs(b) < 1e-12]
    out = set()
    for co in F.region_conditions(beta):
        f = sp.expand(F.poly(co).subs({v: 0 for v in zero}))
        if f == 0:
            continue
        for fac, _ in sp.factor_list(f)[1]:
            if not fac.free_symbols:
                continue
            try:
                c = linear(str(fac))
            except NonLinear:
                print(f"  NONLINEAR at {np.round(beta, 6).tolist()}: {fac}", flush=True)
                continue
            if any(x > 0 for x in c) and any(x < 0 for x in c):
                out.add(c)
    return out


def job_candidates(beta):
    try:
        return candidates(beta)
    except Exception as e:                      # (a seed whose hull fails: reported, no planes)
        print(f"  forms failed at {beta}: {e}", flush=True)
        return set()


def dot(c, x):
    return sum(a * b for a, b in zip(c, x))


def solve3(p, q, r):
    M = sp.Matrix([list(p), list(q), list(r), [1, 1, 1, 1]])
    if M.det() == 0:
        return None
    x = M.solve(sp.Matrix([0, 0, 0, 1]))
    return tuple(Fraction(int(sp.fraction(v)[0]), int(sp.fraction(v)[1])) for v in x)


def sgn(v):
    return (v > 0) - (v < 0)


class Arrangement:
    def __init__(self, planes):
        self.inner = sorted(planes)
        self.planes = [tuple(b) for b in BOUNDARY] + [p for p in self.inner if p not in BOUNDARY]

    def signs(self, x):
        return tuple(sgn(dot(c, x)) for c in self.planes)

    def inside(self, s):
        return all(v >= 0 for v in s[:5])

    def build(self):
        P = self.planes
        n = len(P)
        verts = {}
        for i, j, k in itertools.combinations(range(n), 3):
            x = solve3(P[i], P[j], P[k])
            if x is None or not all(v >= 0 for v in x[:4]) or x[0] < x[3]:
                continue
            verts.setdefault(x, None)
        V = list(verts)
        S = [self.signs(x) for x in V]
        self.vertices = [(x, s) for x, s in zip(V, S)]
        faces = {}                                     # (sign vector) -> point
        for x, s in self.vertices:
            faces.setdefault(s, x)
        # (segments: on the line of two planes, between consecutive vertices)
        for i, j in itertools.combinations(range(n), 2):
            on = [(x, s) for x, s in self.vertices if s[i] == 0 and s[j] == 0]
            if len(on) < 2:
                continue
            a = np.array([[float(v) for v in x] for x, _ in on])
            d = a[np.argmax(np.linalg.norm(a - a[0], axis=1))] - a[0]
            order = np.argsort(a @ d)
            for u, w in zip(order, order[1:]):
                m = tuple((p + q) / 2 for p, q in zip(on[u][0], on[w][0]))
                s = self.signs(m)
                if s != on[u][1]:
                    faces.setdefault(s, m)
        # (polygons: on each plane, the centroid of three of its vertices not in a line)
        for i in range(n):
            on = [x for x, s in self.vertices if s[i] == 0]
            for a, b, c in itertools.combinations(on, 3):
                A = np.array([[float(v) for v in x] for x in (a, b, c)])
                if np.linalg.matrix_rank(A[1:] - A[0], 1e-12) < 2:
                    continue
                m = tuple((p + q + r) / 3 for p, q, r in zip(a, b, c))
                s = self.signs(m)
                if sum(v == 0 for v in s) == 1:
                    faces.setdefault(s, m)
        # (cells: each side of each polygon; the cell's point is the centroid of its vertices)
        cells = set()
        for s in list(faces):
            if sum(v == 0 for v in s) == 1:
                i = s.index(0)
                for e in (1, -1):
                    t = s[:i] + (e,) + s[i + 1:]
                    if self.inside(t):
                        cells.add(t)
        for t in cells:
            vs = [x for x, s in self.vertices if all(a == 0 or a == b for a, b in zip(s, t))]
            m = tuple(sum(c) / len(vs) for c in zip(*vs))
            if self.signs(m) != t:
                raise RuntimeError("cell centroid not inside its cell")
            faces[t] = m
        self.faces = {s: m for s, m in faces.items() if self.inside(s)}
        return self

    def dim(self, s):
        """The dimension of the face (3 minus the rank of the planes it lies on)."""
        Z = [self.planes[i] for i, v in enumerate(s) if v == 0]
        return 3 - (np.linalg.matrix_rank(np.array(Z, float)) if Z else 0)


def fixpoint(start, pool, dims=(3, 2, 1, 0)):
    planes = set(start)
    done = set()
    rnd = 0
    while True:
        rnd += 1
        A = Arrangement(planes).build()
        todo = [(s, m) for s, m in A.faces.items() if A.dim(s) in dims and m not in done]
        t0 = time.time()
        res = pool.map(job_candidates, [tuple(float(v) for v in m) for _, m in todo], chunksize=1)
        done |= {m for _, m in todo}
        new = set().union(*res) - planes if res else set()
        print(f"round {rnd}: {len(planes)} planes, {len(A.vertices)} vertices, {len(A.faces)} faces; "
              f"{len(todo)} new faces' forms in {time.time() - t0:.0f}s; {len(new)} new planes", flush=True)
        if not new:
            return planes, A
        planes |= new


def main():
    t0 = time.time()
    dims = tuple(int(c) for c in sys.argv[1]) if len(sys.argv) > 1 else (3, 2, 1, 0)
    seeds = [(0.4, 0.3, 0.2, 0.1), (0.3, 0.1, 0.4, 0.2), (0.4, 0.1, 0.1, 0.4), (0.25, 0.25, 0.25, 0.25),
             (0.5, 0.2, 0.2, 0.1), (0.35, 0.05, 0.4, 0.2), (0.2, 0.4, 0.3, 0.1), (0.3, 0.3, 0.1, 0.3)]
    with Pool(4) as pool:
        start = set().union(*pool.map(job_candidates, seeds))
        planes, A = fixpoint(start, pool, dims)
    out = {"_about": __doc__.strip().split("\n\n")[0],
           "boundary": [list(b) for b in BOUNDARY], "planes": [list(p) for p in sorted(planes)],
           "faces": [{"signs": list(s), "dim": int(A.dim(s)), "point": [str(v) for v in m]} for s, m in A.faces.items()],
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(planes)} planes; {len(A.faces)} faces; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
