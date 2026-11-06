"""The hyic group [5,3,3] (H4, order 14400): roots, the chamber's seed map, elements and polytopes.

Simple roots from the generator page's Coxeter construction (library._coxeter_roots with the [5,3,3] diagram: mirror 1
meets mirror 2 at pi/5, mirror 2 meets 3 and 3 meets 4 at pi/3). A seed is given by beta_k = its distance to mirror k,
so x(beta) = X beta with X the inverse of the unit roots. No diagram symmetry: the whole chamber is the domain.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from four_d_vertex_generator.library import _coxeter_roots
from four_d_vertex_generator.off import compute_convex_hull

EDGES = ((0, 1, 5), (1, 2, 3), (2, 3, 3))
ROOTS = np.asarray(_coxeter_roots(EDGES), float)
UNIT = ROOTS / np.linalg.norm(ROOTS, axis=1)[:, None]
X = np.linalg.inv(UNIT)
REFL = [np.eye(4) - 2 * np.outer(u, u) for u in UNIT]
SUB = os.environ.get("HYIC_SUB", "full")
FOLD = False


def _elements(gens):
    out = {tuple(np.round(np.eye(4), 7).ravel()): np.eye(4)}
    todo = [np.eye(4)]
    while todo:
        g = todo.pop()
        for s in gens:
            h = s @ g
            k = tuple(np.round(h, 7).ravel())
            if k not in out:
                out[k] = h
                todo.append(h)
    return np.array(list(out.values()))


GENS = list(REFL) if SUB == "full" else [REFL[i] @ REFL[j] for i in range(4) for j in range(i + 1, 4)]
_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".elements_{SUB}.npy")
if os.path.exists(_cache):
    ELEMENTS = np.load(_cache)
else:
    ELEMENTS = _elements(GENS)
    np.save(_cache, ELEMENTS)
ORDER = len(ELEMENTS)


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


def polytope(beta, hull_tol=1e-9):
    V, idx = vertices(beta)
    Vn = V / np.linalg.norm(V[0])
    faces, cells = compute_convex_hull(Vn, tol=hull_tol)
    return V, Vn, idx, faces, cells
