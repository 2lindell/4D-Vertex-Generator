"""The tessic group [4,3,3] (the tesseract's symmetries: the signed permutations of four coordinates, order 384), its
subgroups, its fundamental domain, exact vertices, and the classifier (the contic atlas's, on B4).

Coordinates: R^4, simple roots a1 = e1 - e2, a2 = e2 - e3, a3 = e3 - e4 (long), a4 = e4 (short). A seed is given by
beta_k = its distance to mirror k, so x(beta) = X beta with X the inverse of the unit roots: (1,0,0,0) is the
hexadecachoron's vertex e1, (0,0,0,1) the tesseract's (1,1,1,1)/2. X has entries in Q(sqrt2) (the roots have two
lengths). B4 has no diagram symmetry: the fundamental domain is the whole chamber beta >= 0 (no fold).

TESSIC_SUB chooses the group (every element is a signed permutation matrix):
    full    tessic        [4,3,3]           every signed permutation                   384
    ki      kitessic      [4,3,3]+          determinant +1                             192
    pyr     tessipyritic  [4,(3,3)+]        even permutations, any signs               192
    demi    demitessic    [3^{1,1,1}] = D4  an even number of minus signs              192
    kidemi  kidemitessic  D4+               D4 with determinant +1                      96
Seeds differing by an element of B4 give congruent polytopes (B4 normalizes each), so every group uses the B4 chamber.
"""
from __future__ import annotations

import hashlib
import itertools
import os
from collections import deque

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

NAME = "b4"
FOLD = False                                              # (no swap: the domain is the whole chamber)
S2 = np.sqrt(2.0)
ROOTS = np.array([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 1.0]])
UNIT = ROOTS / np.linalg.norm(ROOTS, axis=1)[:, None]
X = np.linalg.inv(UNIT)                                   # (x = X beta: x . unit root k = beta_k)
REFL = [np.eye(4) - 2 * np.outer(r, r) / (r @ r) for r in ROOTS]
SQ2 = sp.sqrt(2)


def _elements(gens):
    out = {tuple(np.eye(4).round(8).ravel()): np.eye(4)}
    q = deque([np.eye(4)])
    while q:
        g = q.popleft()
        for s in gens:
            h = s @ g
            k = tuple(h.round(8).ravel())
            if k not in out:
                out[k] = h
                q.append(h)
    return np.array(list(out.values()))


ALL = _elements(REFL)
assert len(ALL) == 384


def _parity(g):
    p = list(np.abs(g).argmax(1))
    s = 1
    for i in range(4):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


_DET = np.array([round(np.linalg.det(g)) for g in ALL])
_PAR = np.array([_parity(g) for g in ALL])
_MINUS = np.array([int((g.sum(1) < 0).sum()) for g in ALL])
SUB = os.environ.get("TESSIC_SUB", "full")
TAG = {"full": "", "ki": "_ki", "pyr": "_pyr", "demi": "_demi", "kidemi": "_kidemi"}[SUB]
_KEEP = {"full": np.ones(len(ALL), bool), "ki": _DET > 0, "pyr": _PAR > 0, "demi": _MINUS % 2 == 0,
         "kidemi": (_MINUS % 2 == 0) & (_DET > 0)}[SUB]
ELEMENTS = ALL[_KEEP]
ORDER = len(ELEMENTS)


def _generators(E):
    """A few elements that generate E (greedily: each adds to the group so far)."""
    keys = {tuple(np.round(g, 6).ravel()) for g in E}
    gens, have = [], {tuple(np.round(np.eye(4), 6).ravel())}
    for g in E:
        if tuple(np.round(g, 6).ravel()) in have:
            continue
        gens.append(g)
        have = {tuple(np.round(h, 6).ravel()) for h in _elements(gens)}
        if len(have) == len(keys):
            break
    assert have == keys
    return gens


GENS = _generators(ELEMENTS)


def fname(stem, ext="json"):
    return f"{stem}{TAG}.{ext}"


def exact(M):
    """A float matrix with entries a + b sqrt2 (a, b dyadic, denominators up to 16) as a sympy matrix."""
    out = sp.zeros(*M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            for b16 in range(-64, 65):
                a = v - b16 / 16 * S2
                if abs(a * 16 - round(a * 16)) < 1e-9:
                    out[i, j] = sp.Rational(round(a * 16), 16) + sp.Rational(b16, 16) * SQ2
                    break
            else:
                raise ValueError(f"not in Q(sqrt2)/16: {v}")
    return out


def x4(beta):
    return X @ np.asarray(beta, float)


def vertices(beta, tol=1e-9):
    x = x4(beta)
    P = ELEMENTS @ x
    tree = cKDTree(P)
    seen = np.full(len(P), -1)
    keep = []
    for i in range(len(P)):
        if seen[i] >= 0:
            continue
        for j in tree.query_ball_point(P[i], tol * max(1.0, np.linalg.norm(x))):
            seen[j] = i
        keep.append(i)
    return P[keep], np.array(keep)


def in_domain(B, eps=0.0):
    """The chamber beta >= 0 (B4 has no fold)."""
    B = np.atleast_2d(B)
    return B.min(1) >= -eps


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
    V, idx = vertices(beta)
    Vn = V / np.linalg.norm(V[0])
    faces, cells = compute_convex_hull(Vn, tol=hull_tol)
    return V, Vn, idx, faces, cells


def perms_of(V):
    tree = cKDTree(V)
    out = []
    for g in GENS:
        d, idx = tree.query(V @ g.T)
        if d.max() > 1e-6 * max(1.0, np.abs(V).max()):
            return None
        out.append(idx)
    return out


def classify(beta, hull_tol=1e-9):
    try:
        V, Vn, idx, faces, cells = polytope(beta, hull_tol)
    except Exception:
        return None
    if len(V) < 5:
        return None
    perms = perms_of(V)
    if perms is None:
        return None
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
            "key": _arrangement_key(faces, cells, cell_sets, face_sets, edge_sets, valence, cc, fc, ec),
            "_hull": (V, idx, faces, cells, perms, cc, fc)}


def signature(info):
    if info is None:
        return "none"
    cells = "+".join(f"{n}x[{v}v{f}f]" for n, (v, f) in info["cells"])
    faces = "+".join(f"{n}x{s}g" for n, s in info["faces"])
    edges = "+".join(f"{n}x{k}" for n, k in info["edges"])
    return f"{info['vertices']}: cells {cells} | faces {faces} | edges {edges}"


def label(beta, hull_tol=1e-9):
    info = classify(beta, hull_tol)
    return (signature(info), info["key"]) if info else ("none", None)


def _arrangement_key(faces, cells, cell_sets, face_sets, edge_sets, valence, cell_classes, face_classes, edge_classes):
    """How the cells, faces and edges fit together at vertex 0, refined over neighbours and hashed (as decaic)."""
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
    import networkx as nx

    def inc(beta):
        _, V, _, faces, cells = polytope(beta)
        G = nx.Graph()
        G.add_nodes_from((("v", i) for i in range(len(V))), k="v")
        for i, c in enumerate(cells):
            G.add_node(("c", i), k="c")
            G.add_edges_from((("v", v), ("c", i)) for v in {v for f in c for v in faces[f]})
        return G
    return nx.is_isomorphic(inc(b1), inc(b2), node_match=lambda a, b: a["k"] == b["k"])
