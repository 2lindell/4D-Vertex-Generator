"""The contic group [[3,4,3]] (the 24-cell's symmetries with the swap of the 24-cell and its dual: F4 extended by its
diagram automorphism, order 2304), its fundamental domain, exact vertices, and the classifier.

Coordinates: the standard F4 frame, simple roots a1 = e2 - e3, a2 = e3 - e4 (long), a3 = e4, a4 = (e1 - e2 - e3 - e4)/2
(short). A seed is given by beta_k = its distance to mirror k, so x(beta) = X beta with X the inverse of the unit roots.
The F4 reflections are rational; the diagram automorphism phi (an isometry sending mirror k to mirror 5 - k, long roots
to short) has entries in Q(sqrt 2). Every group element is A + sqrt2 B with A, B dyadic rationals (exact in floats).
phi reverses beta, so half the chamber is a fundamental domain:

    beta >= 0, beta_1 >= beta_4.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

NAME = "f4_extended"
FOLD = True                                               # (the swap halves the chamber: b1 >= b4)
ORDER = 2304
S2 = np.sqrt(2.0)
ROOTS = np.array([[0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [0.5, -0.5, -0.5, -0.5]])
UNIT = ROOTS / np.linalg.norm(ROOTS, axis=1)[:, None]
X = np.linalg.inv(UNIT)                                   # (x = X beta: x . unit root k = beta_k)
REFL = [np.eye(4) - 2 * np.outer(r, r) / (r @ r) for r in ROOTS]
PHI = UNIT[::-1].T @ np.linalg.inv(UNIT.T)               # (phi: unit root k -> unit root 5 - k)
GENS = REFL + [PHI]

# (exact forms: sympy versions)
SQ2 = sp.sqrt(2)
ROOTS_S = sp.Matrix([[0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 1], [sp.Rational(1, 2), -sp.Rational(1, 2), -sp.Rational(1, 2), -sp.Rational(1, 2)]])
UNIT_S = sp.Matrix([[v / sp.sqrt(sum(w ** 2 for w in ROOTS_S.row(i))) for v in ROOTS_S.row(i)] for i in range(4)])
X_S = sp.simplify(UNIT_S.inv())


def _elements():
    out = {tuple(np.eye(4).round(8).ravel()): np.eye(4)}
    q = deque([np.eye(4)])
    while q:
        g = q.popleft()
        for s in GENS:
            h = s @ g
            k = tuple(h.round(8).ravel())
            if k not in out:
                out[k] = h
                q.append(h)
    return np.array(list(out.values()))


ELEMENTS = _elements()
assert len(ELEMENTS) == ORDER, len(ELEMENTS)

# (the subgroup: CONTIC_SUB = full ([[3,4,3]], 2304), ki (kicontic [[3,4,3]]+: the rotations, 1152) or io (iocontic
# [[3,4,3]+]: F4's rotations, and F4's reflections times the swap, 1152). An element is in F4 (no swap) when it sends
# the long root e1 + e2 to a long root (integer coordinates).)
import os  # noqa: E402

SUB = os.environ.get("CONTIC_SUB", "full")
TAG = {"full": "", "ki": "_ki", "io": "_io", "tox": "_tox", "ox": "_ox", "oxb": "_oxb", "ico": "_ico", "icp": "_icp", "icpb": "_icpb"}[SUB]
_IN_F4 = np.array([np.abs(g @ np.array([1, 1, 0, 0.0]) - np.round(g @ np.array([1, 1, 0, 0.0]))).max() < 1e-9 for g in ELEMENTS])
_DET = np.array([np.sign(np.linalg.det(g)) for g in ELEMENTS])
if SUB == "ki":
    _KEEP = _DET > 0
    GENS = [REFL[i] @ REFL[j] for i in range(4) for j in range(i + 1, 4)] + [PHI]
elif SUB == "io":
    _KEEP = (_IN_F4 & (_DET > 0)) | (~_IN_F4 & (_DET < 0))
    GENS = [REFL[i] @ REFL[j] for i in range(4) for j in range(i + 1, 4)] + [REFL[0] @ PHI]
elif SUB == "ico":
    # (icoic [3,4,3] = F4 itself (1152): the swap phi normalises it, so the fold still halves the chamber)
    _KEEP = _IN_F4
    GENS = list(REFL)
elif SUB in ("icp", "icpb"):
    # (icopyritic [3,4,3+] = <r1, r2, r3 r4> (576: the long mirrors kept, the short ones as rotations); icpb is its image
    #  under the swap, [3+,4,3] = <r1 r2, r3, r4>. The swap does not normalise icopyritic, so the chamber's two halves
    #  (b1 >= b4 and b1 <= b4) are two sheets: icopyritic orbits of the half b1 >= b4 (icp) and of its swap image
    #  (icpb on b1 >= b4), joined along the fold.)
    GENS = [REFL[0], REFL[1], REFL[2] @ REFL[3]] if SUB == "icp" else [REFL[0] @ REFL[1], REFL[2], REFL[3]]
    _sub = {tuple(np.eye(4).round(8).ravel())}
    _q = [np.eye(4)]
    while _q:
        _g = _q.pop()
        for _s in GENS:
            _h = _s @ _g
            _k = tuple(_h.round(8).ravel())
            if _k not in _sub:
                _sub.add(_k)
                _q.append(_h)
    _KEEP = np.array([tuple(g.round(8).ravel()) in _sub for g in ELEMENTS])
elif SUB in ("tox", "ox", "oxb"):
    # (toxitic [3+,4,3+] = <r1 r2, r3 r4> (288: an even number of long and of short reflections); oxitic [[3+,4,3+]]
    #  adds the swap phi (576). Oxitic's mirror image (oxb: phi r1 r4 in place of phi) is another subgroup: the F4
    #  reflections do not normalise oxitic (its normaliser is kicontic), so a seed x in the chamber and its mirror image
    #  r x give oxitic polychora that are not congruent; the oxitic shapes are the oxitic orbits of the chamber's seeds
    #  and the oxb orbits of them (mirror images of the oxitic orbits of the reflected seeds).)
    GENS = [REFL[0] @ REFL[1], REFL[2] @ REFL[3]] + {"tox": [], "ox": [PHI], "oxb": [PHI @ REFL[0] @ REFL[3]]}[SUB]
    _sub = {tuple(np.eye(4).round(8).ravel())}
    _q = [np.eye(4)]
    while _q:
        _g = _q.pop()
        for _s in GENS:
            _h = _s @ _g
            _k = tuple(_h.round(8).ravel())
            if _k not in _sub:
                _sub.add(_k)
                _q.append(_h)
    _KEEP = np.array([tuple(g.round(8).ravel()) in _sub for g in ELEMENTS])
else:
    _KEEP = np.ones(len(ELEMENTS), bool)
ELEMENTS = ELEMENTS[_KEEP]
ORDER = len(ELEMENTS)


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
    B = np.atleast_2d(B)
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
