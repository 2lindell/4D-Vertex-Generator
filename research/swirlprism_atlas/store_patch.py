"""Store an exactly built patch (corners given as golden betas) in fexact_patches.json, with its edge planes and its
check against the traced grid; edge shapes are labelled afterwards (fexact.py split_straight_edges / label_edges).

    from store_patch import store
    store("X20", "n(0,1,-1,0)", [[1, phi**2, phi**2, 0], ...], built="...")
"""
from __future__ import annotations

import json

import numpy as np

from fexact import NORMALS, check_patch, load_walls


def store(target, wall, corners, built, key=None, curved=None):
    load_walls()
    C = np.array(corners, float)
    C = C / C.sum(axis=1, keepdims=True)
    n = np.asarray(NORMALS[wall], float)
    edges = []
    for k in range(len(C)):
        a, b = C[k], C[(k + 1) % len(C)]
        m = np.linalg.svd(np.vstack([a, b, n]))[2][-1]
        edges.append([(m / m[np.argmax(np.abs(m))]).tolist(), "", ""])
    allp = json.load(open("fexact_patches.json"))
    allp[key or f"{target} {wall}"] = {"target": target, "wall": wall, "normal": NORMALS[wall], "corners": C.tolist(),
                                       "edges": edges, "curved": curved or [],
                                       "check": list(check_patch(target, wall, C.tolist(), margin=0.01 / 3)),
                                       "built": built}
    json.dump(allp, open("fexact_patches.json", "w"), indent=1)
    return allp[key or f"{target} {wall}"]["check"]


def images(corners, tol=1e-9):
    """The pieces of a patch's images under the 2400-element group that land in the displayed half: each image inside
    the cell is cut by the splitting mirror and the part beyond it folded back. [(wall key or None, corners), ...]."""
    from cellframe import T, TINV
    from normalizer import extended_group
    load_walls()
    C = np.array(corners, float)
    C = C / C.sum(axis=1, keepdims=True)
    M = np.einsum("ij,gjk,kl->gil", TINV, np.array(extended_group(), float), np.array(T, float).T)

    def clip(poly, sign):
        out, n = [], len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            da, db = sign * (a[2] - a[3]), sign * (b[2] - b[3])
            if da >= -tol:
                out.append(a)
            if (da > tol and db < -tol) or (da < -tol and db > tol):
                out.append(a + (b - a) * da / (da - db))
        return out
    pieces = []
    for g in M:
        I = C @ g.T
        if np.any(I.sum(axis=1) <= 0):
            continue
        I = I / I.sum(axis=1, keepdims=True)
        if I.min() < -tol:
            continue
        for sign in (1, -1):
            P = clip(list(I), sign)
            if len(P) < 3:
                continue
            P = np.array(P)
            if sign < 0:
                P = P[:, [1, 0, 3, 2]]
            if np.linalg.svd(P - P.mean(axis=0), compute_uv=False)[1] < 1e-12:      # (a sliver with no area)
                continue
            n = np.linalg.svd(P)[2][-1]
            wall = next((w for w, v in NORMALS.items()
                         if abs(abs(np.asarray(v, float) @ n) - np.linalg.norm(v)) < 1e-7), None)
            key = (wall, tuple(np.round(P.mean(axis=0), 7)))
            if any(k == key for k, _ in pieces):
                continue
            pieces.append((key, P))
    return [(k[0], P.tolist()) for k, P in pieces]
