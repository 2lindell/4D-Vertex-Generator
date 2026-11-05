"""The tessic domain (the B4 chamber) as an exact arrangement of planes over Q(sqrt2) (arrangement.json).

As decaic_atlas/arrangement.py: every form factors into linear forms in beta (here with coefficients in Q(sqrt2)), so a
polychoron can change only on planes. Fixpoint: build the exact arrangement of the planes known so far (vertices in
Q(sqrt2), from three planes and beta summing to 1), give each face a point inside it, add the mixed-sign linear factors
of that point's forms, until no face gives a new plane. Faces: segments between consecutive vertices on a line of two
planes; polygons from centroids of three vertices on a plane not in a line; cells on each side of each polygon.

    python arrangement.py
"""
from __future__ import annotations

import itertools
import json
import time
from multiprocessing import Pool

import numpy as np

import forms as F
import group as G
from qsqrt2 import K

OUT = G.fname("arrangement")
ONE, ZERO = K(1), K(0)
# (the chamber's mirrors; and the fold b1 = b4 where the group has the swap -- not for B4)
BOUNDARY = [tuple(K(v) for v in b) for b in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
            + ([(1, 0, 0, -1)] if getattr(G, "FOLD", True) else [])]
NB = len(BOUNDARY)


def job_candidates(beta):
    logs = []
    try:
        return F.candidates(beta, log=logs.append), logs
    except Exception as e:
        return set(), [f"  forms failed at {beta}: {e!r}"]


def dot(c, x):
    s = ZERO
    for a, b in zip(c, x):
        s = s + a * b
    return s


def det(M):
    if len(M) == 1:
        return M[0][0]
    s = ZERO
    for j in range(len(M)):
        if M[0][j] == ZERO:
            continue
        sub = [row[:j] + row[j + 1:] for row in M[1:]]
        t = M[0][j] * det(sub)
        s = s + t if j % 2 == 0 else s - t
    return s


def solve3(p, q, r):
    M = [list(p), list(q), list(r), [ONE] * 4]
    d = det(M)
    if d == ZERO:
        return None
    rhs = [ZERO, ZERO, ZERO, ONE]
    out = []
    for j in range(4):
        Mj = [row[:j] + [rhs[i]] + row[j + 1:] for i, row in enumerate(M)]
        out.append(det(Mj) / d)
    return tuple(out)


def key(c):
    return tuple((x.a, x.b) for x in c)


class Arrangement:
    def __init__(self, planes):
        inner = sorted(set(planes) - set(BOUNDARY), key=lambda c: [float(x) for x in c])
        self.planes = list(BOUNDARY) + inner

    def signs(self, x):
        return tuple(dot(c, x).sign() for c in self.planes)

    def inside(self, s):
        return all(v >= 0 for v in s[:NB])

    def build(self):
        P = self.planes
        n = len(P)
        verts = {}
        for i, j, k in itertools.combinations(range(n), 3):
            x = solve3(P[i], P[j], P[k])
            if x is None or not all(v.sign() >= 0 for v in x) or (G.FOLD and (x[0] - x[3]).sign() < 0):
                continue
            verts.setdefault(x, None)
        self.vertices = [(x, self.signs(x)) for x in verts]
        faces = {}
        for x, s in self.vertices:
            faces.setdefault(s, x)
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
        for i in range(n):
            on = [x for x, s in self.vertices if s[i] == 0]
            fl = {x: np.array([float(v) for v in x]) for x in on}
            for a, b, c in itertools.combinations(on, 3):
                A = np.array([fl[a], fl[b], fl[c]])
                if np.linalg.matrix_rank(A[1:] - A[0], 1e-12) < 2:
                    continue
                m = tuple((p + q + r) / 3 for p, q, r in zip(a, b, c))
                s = self.signs(m)
                if sum(v == 0 for v in s) == 1:
                    faces.setdefault(s, m)
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
            m = tuple(sum(c, ZERO) / len(vs) for c in zip(*vs))
            if self.signs(m) != t:
                raise RuntimeError("cell centroid not inside its cell")
            faces[t] = m
        self.faces = {s: m for s, m in faces.items() if self.inside(s)}
        return self

    def dim(self, s):
        Z = [[float(v) for v in self.planes[i]] for i, v in enumerate(s) if v == 0]
        return int(3 - (np.linalg.matrix_rank(np.array(Z), 1e-9) if Z else 0))


def fixpoint(start, pool):
    planes = set(start)
    done = set()
    rnd = 0
    while True:
        rnd += 1
        A = Arrangement(planes).build()
        todo = [(s, m) for s, m in A.faces.items() if m not in done]
        t0 = time.time()
        res = pool.map(job_candidates, [tuple(float(v) for v in m) for _, m in todo], chunksize=1)
        done |= {m for _, m in todo}
        new = set()
        for c, logs in res:
            new |= c
            for l in logs:
                print(l, flush=True)
        new -= planes
        print(f"round {rnd}: {len(planes)} planes, {len(A.vertices)} vertices, {len(A.faces)} faces; "
              f"{len(todo)} new faces' forms in {time.time() - t0:.0f}s; {len(new)} new planes", flush=True)
        if not new:
            return planes, A
        planes |= new


def ktext(x):
    return [str(x.a), str(x.b)]


def main():
    t0 = time.time()
    seeds = [(0.4, 0.3, 0.2, 0.1), (0.3, 0.1, 0.4, 0.2), (0.4, 0.1, 0.1, 0.4), (0.25, 0.25, 0.25, 0.25),
             (0.5, 0.2, 0.2, 0.1), (0.35, 0.05, 0.4, 0.2), (0.2, 0.4, 0.3, 0.1), (0.3, 0.3, 0.1, 0.3)]
    with Pool(4) as pool:
        start = set()
        for c, logs in pool.map(job_candidates, seeds):
            start |= c
        planes, A = fixpoint(start, pool)
    out = {"_about": __doc__.strip().split("\n\n")[0], "numbers": "each number is [a, b] meaning a + b sqrt2",
           "boundary": [[ktext(v) for v in b] for b in BOUNDARY],
           "planes": [[ktext(v) for v in p] for p in A.planes[NB:]],
           "faces": [{"signs": list(s), "dim": A.dim(s), "point": [ktext(v) for v in m],
                      "point_float": [float(v) for v in m]} for s, m in A.faces.items()],
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(planes)} planes; {len(A.faces)} faces; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
