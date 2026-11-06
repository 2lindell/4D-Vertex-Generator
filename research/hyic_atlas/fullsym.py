"""The full symmetry order of an orbit polytope's vertex set (every isometry of R^4 keeping it, not only those in a
given group): the set is vertex-transitive, so |Sym| = |V| x |Stab(v0)|, and the stabiliser is found by sending v0
and three neighbours (spanning R^4 with it) to v0 and every triple of vertices with the same distances and inner
products, keeping the maps that are orthogonal and keep the whole set."""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree


def full_symmetry(V, tol=1e-7):
    V = np.asarray(V, float)
    V = V / np.linalg.norm(V[0])
    t = cKDTree(V)
    v0 = V[0]
    d = np.linalg.norm(V - v0, axis=1)
    order = np.argsort(d)[1:]
    base = [0]
    for i in order:
        if np.linalg.matrix_rank(V[base + [i]], 1e-6) == len(base) + 1:
            base.append(i)
        if len(base) == 4:
            break
    A = V[base]                                            # (rows v0, n1, n2, n3)
    G = A @ A.T
    cands = [[j for j in range(len(V)) if abs(V[j] @ v0 - G[0, k]) < tol and abs(np.linalg.norm(V[j]) - 1) < tol]
             for k in (1, 2, 3)]
    stab, improper = 0, 0
    Ainv = np.linalg.inv(A)
    for m1 in cands[0]:
        for m2 in cands[1]:
            if abs(V[m1] @ V[m2] - G[1, 2]) > tol:
                continue
            for m3 in cands[2]:
                if abs(V[m1] @ V[m3] - G[1, 3]) > tol or abs(V[m2] @ V[m3] - G[2, 3]) > tol:
                    continue
                B = np.vstack([v0, V[m1], V[m2], V[m3]])
                M = (Ainv @ B).T                           # (M A_k = B_k)
                if np.abs(M @ M.T - np.eye(4)).max() > 1e-6:
                    continue
                if t.query(V @ M.T)[0].max() < 1e-6:
                    stab += 1
                    improper += np.linalg.det(M) < 0
    return len(V) * stab, len(V) * improper
