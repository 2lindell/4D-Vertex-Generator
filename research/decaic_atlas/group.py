"""The decaic group [[3,3,3]] (the 5-cell's symmetries with the swap of the 5-cell and its dual: S5 x {+-1}, order 240),
its fundamental domain, exact vertices, and the classifier.

Coordinates: R^5 with coordinates summing to 0, where S5 permutes the coordinates and -1 is the central inversion (the
swap: -x read backwards). A seed is x(beta) = sum_k beta_k w_k with the fundamental weights w_k = (1,..,1,0,..,0) - k/5
(1,..,1), so that beta_k = x_k - x_{k+1}: the Weyl chamber is beta >= 0. Its vertices are every signed rearrangement
+-sigma(x): 240 for a generic seed. The swap sends the chamber to itself with the weights reversed (beta_1..beta_4 ->
beta_4..beta_1), so half the chamber is a fundamental domain:

    beta >= 0, beta_1 >= beta_4      (on beta_1 = beta_4 the swap still pairs (b, c) with (c, b)).

The faces beta_k = 0 are mirrors of the group (seeds there have fewer vertices: the Wythoffian shapes); beta_1 = beta_4
is the fold. With 5 x the weights the coordinates are integers in beta: every form below has integer coefficients.
"""
from __future__ import annotations

import hashlib
import itertools
import os

import numpy as np
from scipy.spatial import cKDTree

from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

NAME = "a4_extended"
W5 = np.array([[5 * (i < k) - k for i in range(5)] for k in range(1, 5)], float)        # (5 w_k: integer rows)
# (an orthonormal basis of the sum-zero hyperplane: the 4D frame vertices are drawn and hulled in)
BASIS = np.linalg.svd(np.ones((1, 5)))[2][1:]
PERMS = np.array(list(itertools.permutations(range(5))))


def _parity(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


# (the group: DECAIC_SUB = full (decaic [[3,3,3]], order 240), ki (kidecaic [[3,3,3]]+: A5 x {+-1}, the rotations,
# order 120), io (iodecaic [[3,3,3]+]: even sigma and -sigma for odd sigma, order 120), pen (pennic [3,3,3] = S5,
# 120) or kipen (kipennic [3,3,3]+ = A5, 60). Its elements are (sign, perm):
# x -> sign * x[perm].)
SUB = os.environ.get("DECAIC_SUB", "full")
TAG = {"full": "", "ki": "_ki", "io": "_io", "pen": "_pen", "kipen": "_kipen"}[SUB]
_PAR = np.array([_parity(p) for p in PERMS])
if SUB == "full":
    ELEM = [(s, p) for s in (1, -1) for p in PERMS]
    GENS = [np.eye(5)[[1, 0, 2, 3, 4]], np.eye(5)[[1, 2, 3, 4, 0]], -np.eye(5)]           # ((0 1), (0 1 2 3 4), -1)
elif SUB == "ki":
    ELEM = [(s, p) for s in (1, -1) for p, e in zip(PERMS, _PAR) if e == 1]
    GENS = [np.eye(5)[[1, 2, 0, 3, 4]], np.eye(5)[[1, 2, 3, 4, 0]], -np.eye(5)]           # ((0 1 2), (0 1 2 3 4), -1)
elif SUB == "pen":                          # (pennic [3,3,3] = S5: the permutations, order 120)
    ELEM = [(1, p) for p in PERMS]
    GENS = [np.eye(5)[[1, 0, 2, 3, 4]], np.eye(5)[[1, 2, 3, 4, 0]]]
elif SUB == "kipen":                        # (kipennic [3,3,3]+ = A5: the even permutations, order 60)
    ELEM = [(1, p) for p, e in zip(PERMS, _PAR) if e == 1]
    GENS = [np.eye(5)[[1, 2, 0, 3, 4]], np.eye(5)[[1, 2, 3, 4, 0]]]
else:
    ELEM = [(e, p) for p, e in zip(PERMS, _PAR)]
    GENS = [np.eye(5)[[1, 2, 0, 3, 4]], np.eye(5)[[1, 2, 3, 4, 0]], -np.eye(5)[[1, 0, 2, 3, 4]]]  # (.., -(0 1))
ORDER = len(ELEM)
FOLD = SUB not in ("pen", "kipen")              # (the decaic swap halves the chamber; pennic groups have no swap)
_SGN = np.array([s for s, _ in ELEM], float)
_PRM = np.array([p for _, p in ELEM])


def fname(stem, ext="json"):
    """A data file of this group (decaic files unsuffixed; kidecaic _ki, iodecaic _io)."""
    return f"{stem}{TAG}.{ext}"


def x5(beta):
    """The seed in R^5 (5 x the weights, so integer coefficients)."""
    return np.asarray(beta, float) @ W5


def vertices5(beta, tol=1e-9):
    """The distinct points +-sigma(x): the orbit of the seed."""
    x = x5(beta)
    V = _SGN[:, None] * x[_PRM]
    keep = []
    tree = None
    out = []
    for v in V:
        if not any(np.abs(v - u).max() < tol * max(1.0, np.abs(x).max()) for u in out):
            out.append(v)
    return np.array(out)


def in_domain(B, eps=0.0):
    """The fundamental domain: the chamber beta >= 0, halved by the fold b1 >= b4 where the group has the swap
    (decaic and its subgroups); the whole chamber for pennic and kipennic (no swap)."""
    B = np.atleast_2d(B)
    if SUB in ("pen", "kipen"):
        return B.min(1) >= -eps
    return (B.min(1) >= -eps) & (B[:, 0] >= B[:, 3] - eps)


def _classes(items, perms):
    index = {it: i for i, it in enumerate(items)}
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in perms:
        for i, it in enumerate(items):
            j = index.get(frozenset(int(p[v]) for v in it))
            if j is None:
                raise ValueError("hull not symmetric")
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    groups = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def polytope(beta, hull_tol=1e-9):
    """(vertices in 4D, faces, cells) of the seed's polychoron."""
    V5 = vertices5(beta)
    V = V5 @ BASIS.T
    V = V / np.linalg.norm(V[0])
    faces, cells = compute_convex_hull(V, tol=hull_tol)
    return V5, V, faces, cells


def classify(beta, hull_tol=1e-9):
    """Counts of the seed's polychoron by symmetry class (as the atlases do), and its arrangement key."""
    try:
        V5, V, faces, cells = polytope(beta, hull_tol)
    except Exception:
        return None
    if len(V) < 5:
        return None
    tree = cKDTree(V5)
    perms = []
    for g in GENS:
        d, idx = tree.query(V5 @ g.T)
        if d.max() > 1e-6 * max(1.0, np.abs(V5).max()):
            return None
        perms.append(idx)
    edges = hull_edges(faces)
    face_sets = [frozenset(f) for f in faces]
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    edge_sets = [frozenset(e) for e in edges]
    valence = {e: 0 for e in edge_sets}
    for c in cells:
        for e in {frozenset(e) for f in c for e in zip(faces[f], faces[f][1:] + faces[f][:1])}:
            if e in valence:
                valence[e] += 1
    try:
        cc, fc, ec = _classes(cell_sets, perms), _classes(face_sets, perms), _classes(edge_sets, perms)
    except ValueError:
        return None
    describe = lambda classes, label: sorted((len(cl), label(cl[0])) for cl in classes)
    return {"vertices": len(V),
            "cells": describe(cc, lambda i: (len(cell_sets[i]), len(cells[i]))),
            "faces": describe(fc, lambda i: len(face_sets[i])),
            "edges": describe(ec, lambda i: valence[edge_sets[i]]),
            "key": _arrangement_key(faces, cells, cell_sets, face_sets, edge_sets, valence, cc, fc, ec)}


def signature(info):
    if info is None:
        return "none"
    cells = "+".join(f"{n}x[{v}v{f}f]" for n, (v, f) in info["cells"])
    faces = "+".join(f"{n}x{s}g" for n, s in info["faces"])
    edges = "+".join(f"{n}x{k}" for n, k in info["edges"])
    return f"{info['vertices']}: cells {cells} | faces {faces} | edges {edges}"


def label(beta, hull_tol=1e-9):
    """(signature, arrangement key) of the seed's polychoron."""
    info = classify(beta, hull_tol)
    return (signature(info), info["key"]) if info else ("none", None)


def _arrangement_key(faces, cells, cell_sets, face_sets, edge_sets, valence, cell_classes, face_classes, edge_classes):
    """How the cells, faces and edges fit together at vertex 0 (every vertex is like it): each labelled by its size and
    its class's size, joined where one lies on another, refined over its neighbours and hashed."""
    size = {}
    for kind, classes in (("c", cell_classes), ("f", face_classes), ("e", edge_classes)):
        for cl in classes:
            for i in cl:
                size[(kind, i)] = len(cl)
    lab, nbr = {}, {}
    eix = {e: i for i, e in enumerate(edge_sets)}
    for i, f in enumerate(faces):
        if 0 not in f:
            continue
        lab[("f", i)] = f"f{len(f)}:{size[('f', i)]}"
        k = f.index(0)
        for w in (f[k - 1], f[(k + 1) % len(f)]):
            e = ("e", eix[frozenset((0, w))])
            lab[e] = f"e{valence[edge_sets[e[1]]]}:{size[e]}"
            nbr.setdefault(("f", i), []).append(e)
            nbr.setdefault(e, []).append(("f", i))
    for j, c in enumerate(cells):
        at = [f for f in c if 0 in faces[f]]
        if not at:
            continue
        lab[("c", j)] = f"c{len(cell_sets[j])}.{len(c)}:{size[('c', j)]}"
        for f in at:
            nbr.setdefault(("c", j), []).append(("f", f))
            nbr.setdefault(("f", f), []).append(("c", j))
    for _ in range(4):
        lab = {n: hashlib.sha1((lab[n] + "|" + ",".join(sorted(lab[m] for m in nbr.get(n, [])))).encode()).hexdigest()[:16]
               for n in lab}
    return hashlib.sha1(",".join(sorted(lab.values())).encode()).hexdigest()[:12]


def same_polytope(b1, b2):
    """Are the two seeds' polychora the same, by their incidences (vertices and cells, any relabelling)?"""
    import networkx as nx

    def inc(beta):
        _, V, faces, cells = polytope(beta)
        G = nx.Graph()
        G.add_nodes_from((("v", i) for i in range(len(V))), k="v")
        for i, c in enumerate(cells):
            G.add_node(("c", i), k="c")
            G.add_edges_from((("v", v), ("c", i)) for v in {v for f in c for v in faces[f]})
        return G
    return nx.is_isomorphic(inc(b1), inc(b2), node_match=lambda a, b: a["k"] == b["k"])
