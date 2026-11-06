"""The hyic family ([5,3,3] = H4, order 14400, and its subgroups): roots, the chamber's seed map, elements, the
classifier (the contic atlas's: arrangement keys, signatures, incidence isomorphism), exact forms over Q(sqrt5).

Simple roots from the generator page's Coxeter construction (library._coxeter_roots, [5,3,3]: mirror 1 meets mirror 2
at pi/5, 2 meets 3 and 3 meets 4 at pi/3). A seed is beta_k = its distance to mirror k, x(beta) = X beta. There is no
diagram symmetry, so the whole chamber is the domain (no fold).

HYIC_SUB = full ([5,3,3], 14400), ki (kihyic [5,3,3]+, the rotations, 7200), ix (ixitic 5[3+,4,3+] = +-[I x T], 1440:
found inside H4 through the icosian (quaternion) form of H4, where it is the pairs (l, r) with l in the binary
icosahedral group and r in its binary tetrahedral subgroup, carried to this basis by the map of simple roots).

Exactness: in root coordinates (y = UNIT x) every element of H4 has entries in Z[(1+sqrt5)/2], and a vertex is
(UNIT g X) beta; a 5-vertex determinant there is the true one times det(UNIT), so its zero set is the same.
"""
from __future__ import annotations

import hashlib
import os
from collections import deque

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

from four_d_vertex_generator.library import _coxeter_roots
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = ((0, 1, 5), (1, 2, 3), (2, 3, 3))
ROOTS = np.asarray(_coxeter_roots(EDGES), float)
UNIT = ROOTS / np.linalg.norm(ROOTS, axis=1)[:, None]
X = np.linalg.inv(UNIT)
REFL = [np.eye(4) - 2 * np.outer(u, u) for u in UNIT]
FOLD = False
SQ5 = sp.sqrt(5)
S5 = 5 ** 0.5
SUB = os.environ.get("HYIC_SUB", "full")
TAG = {"full": "", "ki": "_ki", "ix": "_ix"}[SUB]
NAME = {"full": "hyic", "ki": "kihyic", "ix": "ixitic"}[SUB]


def _closure(gens):
    out = {tuple(np.round(np.eye(4), 7).ravel()): np.eye(4)}
    todo = deque([np.eye(4)])
    while todo:
        g = todo.popleft()
        for s in gens:
            h = s @ g
            k = tuple(np.round(h, 7).ravel())
            if k not in out:
                out[k] = h
                todo.append(h)
    return np.array(list(out.values()))


def _ixitic_generators():
    """Ixitic in this basis: the generator page's ixitic (icosian form) carried over by the simple-root map."""
    from four_d_vertex_generator.library import _reflection_chamber_roots, named_symmetry
    Rq = np.asarray(_reflection_chamber_roots("h4_icosian"), float)
    Rq = Rq / np.linalg.norm(Rq, axis=1)[:, None]
    Gq = Rq @ Rq.T
    G0 = UNIT @ UNIT.T
    if np.abs(Gq - G0).max() > 1e-9:                       # (the icosian roots listed from the other end of the diagram)
        Rq = Rq[::-1]
    assert np.abs(Rq @ Rq.T - G0).max() < 1e-9
    C = UNIT.T @ np.linalg.inv(Rq.T)                       # (C r_q = r: the orthogonal map of the icosian chamber to ours)
    return [C @ g @ C.T for g in named_symmetry("ixitic").generators]


if SUB == "full":
    GENS = list(REFL)
elif SUB == "ki":
    GENS = [REFL[i] @ REFL[j] for i in range(4) for j in range(i + 1, 4)]
else:
    GENS = _ixitic_generators()
_cache = os.path.join(HERE, f".elements_{SUB}.npy")
if os.path.exists(_cache):
    ELEMENTS = np.load(_cache)
else:
    ELEMENTS = _closure(GENS)
    np.save(_cache, ELEMENTS)
ORDER = len(ELEMENTS)
assert ORDER == {"full": 14400, "ki": 7200, "ix": 1440}[SUB], ORDER
if SUB == "ix":                                            # (ixitic lies in H4: every element is an H4 element)
    _H4 = np.load(os.path.join(HERE, ".elements_full.npy")) if os.path.exists(os.path.join(HERE, ".elements_full.npy")) \
        else _closure(REFL)
    assert np.abs(cKDTree(_H4.reshape(len(_H4), -1)).query(ELEMENTS.reshape(len(ELEMENTS), -1))[0]).max() < 1e-7


def fname(stem, ext="json"):
    return f"{stem}{TAG}.{ext}"


def exact(M):
    """A float matrix with entries (a + b sqrt5) / 2 (a, b integers up to 64) as a sympy matrix."""
    out = sp.zeros(*M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            for b in range(-64, 65):
                a = 2 * v - b * S5
                if abs(a - round(a)) < 1e-8:
                    out[i, j] = (sp.Integer(round(a)) + sp.Integer(b) * SQ5) / 2
                    break
            else:
                raise ValueError(f"not in Z[phi]/2: {v}")
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
