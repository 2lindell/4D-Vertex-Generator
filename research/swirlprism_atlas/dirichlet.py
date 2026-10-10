"""Dirichlet (Voronoi) fundamental domains for the swirlprism group and its 2400-element extension.

The domain with centre p is every point of the 3-sphere closer to p than to any image g.p. It is a
convex spherical polyhedron that tiles the sphere |group| times; each face is the bisector of p and
some g.p, and is glued to the face of g^-1 by g. A half-turn's axis is equidistant from p and h.p, so
when that bisector is a face, the axis lies in it and the face folds onto itself across the axis.
Polyhedra are computed in the gnomonic chart at p (great spheres become planes).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


def dirichlet(p: np.ndarray, elements: np.ndarray) -> dict:
    p = np.asarray(p, float) / np.linalg.norm(p)
    basis = np.linalg.svd(p[None])[2][1:]                    # orthonormal basis of p's tangent space
    rows, gens = [], []
    for k, g in enumerate(elements):
        q = g @ p
        if np.allclose(q, p, atol=1e-12):
            continue
        # closer to p than to q:  (p + y).(p - q) >= 0  ->  -(E(p - q)).u - (1 - p.q) <= 0
        rows.append(np.append(-(basis @ (p - q)), -(1 - p @ q)))
        gens.append(k)
    rows = np.array(rows)
    hs = HalfspaceIntersection(rows, np.zeros(3))
    pts = hs.intersections
    keep = []
    for u in pts:                                            # merge duplicate corners
        if not any(np.linalg.norm(u - w) < 1e-9 for w in keep):
            keep.append(u)
    pts = np.array(keep)
    faces = []
    for r, k in zip(rows, gens):
        on = np.flatnonzero(np.abs(pts @ r[:3] + r[3]) < 1e-9)
        if len(on) >= 3:
            faces.append({"element": int(k), "corners": on.tolist()})
    sphere = []
    for u in pts:
        x = p + basis.T @ u
        sphere.append(x / np.linalg.norm(x))
    return {"centre": p, "corners_chart": pts, "corners": np.array(sphere), "faces": faces,
            "volume_chart": ConvexHull(pts).volume}


def display_faces(domain: dict, elements: np.ndarray, n_g: int) -> list[dict]:
    """Faces as ordered corner polygons (barycentric, cell frame) with how each one is glued.

    kind: "q-axis" (folded onto itself across the axis of a coset half-turn, a light purple axis),
    "ring-axis" (folded across the axis of a half-turn of G, a cross ring), or "pair-<n>" (glued to
    the other face with the same n by an element and its inverse). n_g is |G|: elements[k] with
    k >= n_g are in the coset G.Q.
    """
    from cellframe import TINV
    chart = domain["corners_chart"]
    corners = domain["corners"]
    out, pairs = [], {}
    for f in domain["faces"]:
        idx = f["corners"]
        pts = chart[idx]
        mid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mid)
        ang = np.arctan2((pts - mid) @ vt[1], (pts - mid) @ vt[0])
        order = [idx[i] for i in np.argsort(ang)]
        g = elements[f["element"]]
        coset = f["element"] >= n_g
        if np.allclose(g @ g, np.eye(4), atol=1e-9):
            kind = "q-axis" if coset else "ring-axis"
            _, _, axis_vt = np.linalg.svd(g - np.eye(4))
            axis = axis_vt[2:]
            ends = [corners[i] for i in order if np.linalg.norm(corners[i] - (corners[i] @ axis.T) @ axis) < 1e-8]
        else:
            ginv = np.linalg.inv(g)
            key = min(f["element"], next(k for k, h in enumerate(elements) if np.allclose(h, ginv, atol=1e-9)))
            pairs.setdefault(key, len(pairs) + 1)
            kind, ends = f"pair-{pairs[key]}", []
        bary = [TINV @ corners[i] for i in order]
        out.append({"kind": kind, "coset": bool(coset), "corners": [b / b.sum() for b in bary],
                    "axis": [(TINV @ e) / (TINV @ e).sum() for e in ends]})
    return out
