"""A curved wall's mesh (xsurfaces.json) filled out to one of its exact edges, without remeshing it.

Marching over a grid leaves a wall's mesh ending a little short of an edge that is a straight line of the wall (its
boundary vertices lie up to a grid step off it). The strip between is filled: the mesh's boundary vertices along the
edge, in order, each joined to its own foot on the line, two triangles per boundary edge; the line's ends are already
mesh vertices. The feet lie on the line, so on the wall exactly.

    python fill_edge.py X7 "0.25,0.25,0.25,0.25" "1,1,1,0"      (the edge E2, from C1 to C5)
    python fill_edge.py --corner X7 corner P Q                  (out to an exact corner of two edges, from P and Q)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def nrm(x):
    x = np.asarray(x, float)
    return x / x.sum(axis=-1, keepdims=True)


def fill(tid, A, B, reach=0.008):
    data = json.load(open("xsurfaces.json"))
    T = np.array(data[tid]["tris"], float)
    T = nrm(T)
    A, B = nrm(A), nrm(B)
    key = lambda p: tuple(np.round(p, 9))
    edges = defaultdict(int)
    for t in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((key(t[a]), key(t[b])))] += 1
    boundary = [tuple(e) for e, n in edges.items() if n == 1 and len(e) == 2]
    d = B - A
    L2 = d @ d
    par = lambda p: float((np.asarray(p) - A) @ d / L2)
    dist = lambda p: float(np.linalg.norm(np.asarray(p) - A - par(p) * d))
    # (the boundary edges along the line: both ends near it, between its ends)
    near = [(p, q) for p, q in boundary if dist(p) < reach and dist(q) < reach and -1e-9 <= par(p) <= 1 + 1e-9
            and -1e-9 <= par(q) <= 1 + 1e-9]
    added = []
    for p, q in near:
        p, q = np.array(p), np.array(q)
        fp, fq = A + par(p) * d, A + par(q) * d
        for tri in ((p, fp, fq), (p, fq, q)):
            a = np.linalg.norm(np.cross((tri[1] - tri[0])[:3], (tri[2] - tri[0])[:3]))
            if a > 1e-14:
                added.append([list(map(float, x)) for x in tri])
    gap = max((dist(p) for e in near for p in e), default=0.0)
    covered = sum(abs(par(q) - par(p)) for p, q in near)
    data[tid]["tris"] = data[tid]["tris"] + added
    data[tid].setdefault("filled_edges", []).append({"from": A.tolist(), "to": B.tolist(), "triangles": len(added)})
    json.dump(data, open("xsurfaces.json", "w"))
    print(f"{tid}: {len(near)} boundary edges along the line (covering {covered:.0%} of it), up to {gap:.4f} off it; "
          f"{len(added)} triangles added")


def fill_corner(tid, corner, P, Q, tol=1e-7):
    """The mesh carried out to an exact corner where two of its straight edges meet (lines from P and from Q to the
    corner), when its boundary stops short of it: a fan from the corner to the mesh's boundary between its last vertex
    on each line."""
    data = json.load(open("xsurfaces.json"))
    T = nrm(np.array(data[tid]["tris"], float))
    K = nrm(corner)
    P, Q = nrm(P), nrm(Q)
    key = lambda p: tuple(np.round(p, 9))
    pos, edges = {}, defaultdict(int)
    for t in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pos[key(t[a])] = t[a]
            edges[frozenset((key(t[a]), key(t[b])))] += 1
    nbr = defaultdict(list)
    for e, n in edges.items():
        if n == 1 and len(e) == 2:
            a, b = tuple(e)
            nbr[a].append(b)
            nbr[b].append(a)

    def on_line(p, A):
        d = K - A
        t = (p - A) @ d / (d @ d)
        return np.linalg.norm(p - A - t * d) < tol, t
    # (each line's boundary vertex nearest the corner)
    ends = []
    for A in (P, Q):
        best = max(((on_line(pos[k], A)[1], k) for k in nbr if on_line(pos[k], A)[0]), default=None)
        ends.append(best)
    if None in ends:
        print(f"{tid}: the boundary does not run along both lines")
        return
    (ta, a), (tb, b) = ends
    if min(np.linalg.norm(pos[a] - K), np.linalg.norm(pos[b] - K)) < 1e-9 and a == b:
        print(f"{tid}: the mesh already reaches the corner")
        return
    # (the boundary path from a to b, the short way)
    prev, todo = {a: None}, [a]
    while todo and b not in prev:
        nxt = []
        for k in todo:
            for m in nbr[k]:
                if m not in prev:
                    prev[m] = k
                    nxt.append(m)
        todo = nxt
    path, k = [], b
    while k is not None:
        path.append(k)
        k = prev[k]
    added = [[list(map(float, K)), list(map(float, pos[path[i]])), list(map(float, pos[path[i + 1]]))]
             for i in range(len(path) - 1)]
    data[tid]["tris"] = data[tid]["tris"] + added
    data[tid].setdefault("filled_corners", []).append({"corner": K.tolist(), "triangles": len(added)})
    for cp in data[tid].get("corner_points", []):          # (the corner the mesh stopped short of, now exact)
        if np.linalg.norm(nrm(cp[0]) - K) < 0.02 and np.linalg.norm(nrm(cp[0]) - K) > 1e-12:
            cp[0] = K.tolist()
    json.dump(data, open("xsurfaces.json", "w"))
    print(f"{tid}: boundary stopped {np.linalg.norm(pos[a] - K):.4f} and {np.linalg.norm(pos[b] - K):.4f} short of the "
          f"corner along the two lines; {len(added)} triangles fan out to it")


if __name__ == "__main__":
    if sys.argv[1] == "--corner":
        f = lambda s: [float(x) for x in s.split(",")]
        fill_corner(sys.argv[2], f(sys.argv[3]), f(sys.argv[4]), f(sys.argv[5]))
    else:
        fill(sys.argv[1], [float(x) for x in sys.argv[2].split(",")], [float(x) for x in sys.argv[3].split(",")])
