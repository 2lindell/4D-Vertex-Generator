"""Describe a decaic-family polychoron in the words of the 'Pennic and Decaic Isogonals' page: cells by orbit (grouped
by kind), vertices and the dualcell's face count (= edges at a vertex), edges by valence (trigonic = three faces meet)
with their orbit sizes, faces by shape (equits, isots, scalene triangles, squares, rectangles, trapezoids, ...).

    DECAIC_SUB=ki|io (or CONTIC_SUB=full|ki|io|ico|icp|icpb|tox|ox|oxb) python describe.py '<json list of [id, beta, hull_tol]>'   -> JSON on stdout
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
# (the atlas: the contic family when CONTIC_SUB is set, else the decaic family (DECAIC_SUB))
CONTIC = "CONTIC_SUB" in os.environ
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "contic_atlas" if CONTIC else "decaic_atlas"))
import group as G  # noqa: E402

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402

VAL = {3: "trigonic", 4: "tetragonic", 5: "pentagonic", 6: "hexagonic", 7: "heptagonic", 8: "octagonic", 9: "enneagonic",
       10: "decagonic", 11: "hendecagonic", 12: "dodecagonic"}


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def face_kind(P):
    n = len(P)
    L = [np.linalg.norm(P[(i + 1) % n] - P[i]) for i in range(n)]
    ang = []
    for i in range(n):
        u, v = P[i - 1] - P[i], P[(i + 1) % n] - P[i]
        ang.append(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
    eqL = all(close(l, L[0]) for l in L)
    eqA = all(close(a, ang[0]) for a in ang)
    if n == 3:
        k = sum(close(L[i], L[j]) for i, j in ((0, 1), (1, 2), (0, 2)))
        return "equits" if eqL else ("isots" if k == 1 else "scalene triangles")
    if n == 4:
        right = all(close(a, np.pi / 2) for a in ang)
        if right:
            return "squares" if eqL else "rectangles"
        par = lambda i: abs(abs((P[(i + 1) % 4] - P[i]) @ (P[(i + 3) % 4 + 0 - 0] - P[(i + 2) % 4])) /
                            (np.linalg.norm(P[(i + 1) % 4] - P[i]) * np.linalg.norm(P[(i + 3) % 4] - P[(i + 2) % 4])) - 1) < 1e-6
        if eqL:
            return "rhombi"
        if par(0) or par(1):
            return "trapezoids"
        return "irregular quadrilaterals"
    name = {5: "pentagons", 6: "hexagons", 8: "octagons", 10: "decagons", 12: "dodecagons"}.get(n, f"{n}-gons")
    if eqL and eqA:
        return name
    if eqA and all(close(L[i], L[i % 2]) for i in range(n)):
        return {6: "ditrigons", 8: "ditetragons", 10: "dipentagons", 12: "dihexagons"}.get(n, "di-" + name)
    return "irregular " + name


def cell_kind(nv, nf, Pfaces, P):
    """The cell's kind in the page's words; P: its vertices (index -> point) for the shape tests."""
    sizes = Counter(len(f) for f in Pfaces)
    pts = np.array(list(P.values()))
    c = pts.mean(0)
    central = all(np.min(np.linalg.norm(pts - (2 * c - p), axis=1)) < 1e-7 for p in pts)
    if (nv, nf) == (4, 4):
        L = {frozenset((a, b)): np.linalg.norm(P[a] - P[b]) for a in P for b in P if a < b}
        ks = list(P)
        opp = [(frozenset((ks[0], ks[1])), frozenset((ks[2], ks[3]))), (frozenset((ks[0], ks[2])), frozenset((ks[1], ks[3]))),
               (frozenset((ks[0], ks[3])), frozenset((ks[1], ks[2])))]
        if all(close(l, list(L.values())[0]) for l in L.values()):
            return "regular tets"
        if all(close(L[a], L[b]) for a, b in opp):
            pairs = [L[a] for a, _ in opp]
            return "tetragonal disphenoids" if sum(close(pairs[i], pairs[j]) for i, j in ((0, 1), (1, 2), (0, 2))) == 1 \
                else "disphenoids"
        return "irregular tets"
    if (nv, nf) == (6, 8):
        E = [np.linalg.norm(P[f[i]] - P[f[i - 1]]) for f in Pfaces for i in range(3)]
        if all(close(e, E[0]) for e in E):
            return "octs"
        return "traps" if central else "gyro-traps"
    if (nv, nf) == (6, 5):
        tri = [f for f in Pfaces if len(f) == 3]
        if len(tri) == 2:
            # (parallel triangles: a prism (trip); else a wedge)
            U = np.linalg.svd(pts - c)[2][:3]
            def normal(f):
                a, b, d = (U @ (P[v] - c) for v in f)
                n = np.cross(b - a, d - a)
                return n / np.linalg.norm(n)
            if abs(abs(normal(tri[0]) @ normal(tri[1])) - 1) < 1e-6:
                return "trips"
            return "wedges"
        return "6-vertex 5-faced cells"
    if (nv, nf) == (5, 6) and sizes.get(3) == 6:
        return "triangular bipyramids"
    if (nv, nf) == (5, 5):
        return "square pyramids" if sizes.get(4) == 1 else "5-vertex 5-faced cells"
    if (nv, nf) == (12, 20):
        return "snits"
    if (nv, nf) == (12, 8) and sizes.get(4) == 6:
        return "ditrips / hips"
    if (nv, nf) == (12, 8):
        if sizes.get(6) == 4 and sizes.get(3) == 4:
            hexes = [face_kind(np.array([P[v] for v in f])) for f in Pfaces if len(f) == 6]
            return "tuts" if all(h in ("hexagons", "ditrigons") for h in hexes) else "gyrotuts"
        return "12-vertex 8-faced cells"
    if (nv, nf) == (12, 14):
        return "ratets"
    if (nv, nf) == (24, 14):
        return "tics" if sizes.get(8) else "gratets / toes"
    if (nv, nf) == (24, 26):
        return "sircoes"
    if (nv, nf) == (48, 26):
        return "gircoes"
    if (nv, nf) == (8, 10) and sizes.get(3) == 8 and sizes.get(4) == 2:
        return "squaps / squigyps"
    if (nv, nf) == (24, 38) and sizes.get(3) == 32:
        return "snics"
    if (nv, nf) == (6, 7) and sizes.get(3) == 6 and sizes.get(4) == 1:
        return "gyrowedges"
    if (nv, nf) == (9, 8) and sizes.get(3) == 4 and sizes.get(4) == 3 and sizes.get(6) == 1:
        return "tricus"
    if (nv, nf) == (8, 6):
        return "cubes"
    desc = " + ".join(f"{n} {k}-gons" for k, n in sorted(sizes.items()))
    return f"{nf}-faced cells ({nv} vertices: {desc})"


def plus(counts):
    """[120, 120, 60] -> '60+120+120'."""
    return "+".join(str(c) for c in sorted(counts))


def describe(beta, tol):
    if CONTIC:
        V5, V, idx, faces, cells = G.polytope(np.asarray(beta, float), tol)
    else:
        V5, V, faces, cells = G.polytope(np.asarray(beta, float), tol)
    tree = cKDTree(V5)
    perms = [tree.query(V5 @ g.T)[1] for g in G.GENS]
    edges = [tuple(e) for e in hull_edges(faces)]
    face_sets = [frozenset(f) for f in faces]
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    edge_sets = [frozenset(e) for e in edges]
    val = Counter()
    for c in cells:
        for e in {frozenset(e) for f in c for e in zip(faces[f], faces[f][1:] + faces[f][:1])}:
            val[e] += 1
    cc, fc, ec = G._classes(cell_sets, perms), G._classes(face_sets, perms), G._classes(edge_sets, perms)
    cells_by = defaultdict(list)
    for cl in cc:
        c = cells[cl[0]]
        nv = len(cell_sets[cl[0]])
        vs = sorted(cell_sets[cl[0]])
        cells_by[cell_kind(nv, len(c), [faces[f] for f in c], {v: V[v] for v in vs})].append(len(cl))
    faces_by = defaultdict(list)
    for cl in fc:
        faces_by[face_kind(V[faces[cl[0]]])].append(len(cl))
    edges_by = defaultdict(list)
    for cl in ec:
        edges_by[val[edge_sets[cl[0]]]].append(len(cl))
    return {"vertices": len(V), "dualcell_faces": 2 * len(edges) // len(V), "n_cells": len(cells),
            "cells": {k: sorted(v) for k, v in cells_by.items()}, "faces": {k: sorted(v) for k, v in faces_by.items()},
            "edges": {VAL.get(k, f"{k}-gonic"): sorted(v) for k, v in sorted(edges_by.items())}, "key": G.label(np.asarray(beta), tol)[1]}


if __name__ == "__main__":
    out = {}
    for ident, beta, tol in json.loads(sys.argv[1]):
        out[ident] = describe(beta, tol)
    print(json.dumps(out))
