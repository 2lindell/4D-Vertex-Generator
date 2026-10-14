"""Index-3 tetrahedral swirl group Sigma_1/3[T x C3] (order 24, n = 1): fundamental domain and classifier.

Every seed is equivalent to one in the Voronoi cell of an anchor on a fixed circle of an order-3 element (the points
closer to the anchor than to any other image of it). The cell is drawn in the gnomonic chart at the anchor,
x = (a + u) / |a + u| with u in the 3D tangent space, where great spheres are flat planes. The anchor's stabiliser
(order 3) rotates the chart about its z axis, so one third of the cell (azimuth 0 to 120 degrees) is a fundamental
domain.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

from four_d_vertex_generator.generation import generate_vertices_from_seed, group_elements
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

import os

GROUPS = {
    "tet": "h4_tubical_tetrahedral_cyclic_index3_1",
    "oct": "h4_tubical_octahedral_dihedral_index6_6n_1",
}
KEY = os.environ.get("SWIRL_GROUP", "tet")
NAME = GROUPS[KEY]
ACTION = named_symmetry(NAME)
ELEMENTS = np.stack(group_elements(ACTION))


def _anchor_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Anchor a on a fixed circle of the rotation with the smallest angle (largest cyclic stabiliser, order m), and
    the chart frame (rows e1, e2, e3 span a-perp; the stabiliser rotates about e3)."""
    best, best_trace = None, 4.0
    for g in ELEMENTS:
        tr = np.trace(g)
        if abs(np.linalg.matrix_rank(g - np.eye(4), tol=1e-8) - 2) < 1e-9 and tr < 3.99 and (best is None or tr > best_trace):
            best, best_trace = g, tr
    assert best is not None
    m = int(round(2 * np.pi / np.arccos((best_trace - 2) / 2)))
    u_, sv, vt = np.linalg.svd(best - np.eye(4))
    fixed = vt[2:].T
    a = fixed[:, 0]
    e3 = fixed[:, 1]
    rest = np.linalg.svd(np.eye(4) - fixed @ fixed.T)[0][:, :2]
    frame = np.stack([rest[:, 0], rest[:, 1], e3])
    if np.linalg.det(np.vstack([a, frame])) < 0:
        frame[1] = -frame[1]
    return a, frame, best, m


A, FRAME, STAB, M = _anchor_frame()


def seed_from_chart(u: np.ndarray) -> np.ndarray:
    """Unit seed for chart coordinates u (n x 3, or 3)."""
    u = np.asarray(u, dtype=float)
    p = A + u @ FRAME
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def chart_from_seed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x / (x @ A)[..., None] - A) @ FRAME.T


def stabiliser_in_chart() -> np.ndarray:
    return FRAME @ STAB @ FRAME.T


def cell_halfspaces() -> np.ndarray:
    """Rows [n, d] with n . u + d <= 0 inside: x . (a - g a) >= 0 for the images g a != a, with x ~ a + u."""
    rows = []
    seen = []
    for g in ELEMENTS:
        ga = g @ A
        if np.linalg.norm(ga - A) < 1e-9 or any(np.linalg.norm(ga - s) < 1e-9 for s in seen):
            continue
        seen.append(ga)
        w = A - ga                       # (a + u) . w >= 0, with u = chart coordinate
        rows.append(np.concatenate([-(FRAME @ w), [-(A @ w)]]))
    return np.array(rows)


def voronoi_cell_vertices() -> np.ndarray:
    H = cell_halfspaces()
    hs = HalfspaceIntersection(H, np.zeros(3))
    return hs.intersections


def wedge_halfspaces() -> np.ndarray:
    """Azimuth 0 to 360/M degrees about e3."""
    if M == 1:
        return np.zeros((0, 4))
    rows = [[0.0, -1.0, 0.0, 0.0]]
    if M >= 3:
        t = 2 * np.pi / M
        rows.append([-np.sin(t), np.cos(t), 0.0, 0.0])
    return np.array(rows)


def domain_halfspaces() -> np.ndarray:
    return np.vstack([cell_halfspaces(), wedge_halfspaces()])


def _interior_point() -> np.ndarray:
    """Chart point strictly inside the domain: the centroid of random wedge samples."""
    if not hasattr(_interior_point, "p"):
        rng = np.random.default_rng(0)
        t = 2 * np.pi / M
        best, best_m = None, -1
        H = domain_halfspaces()
        for _ in range(4000):
            r, a, z = rng.uniform(0.01, 0.3), rng.uniform(0.2, 0.8) * t, rng.uniform(-0.3, 0.3)
            p = np.array([r * np.cos(a), r * np.sin(a), z])
            mgn = np.min(-(H[:, :3] @ p + H[:, 3]) / np.linalg.norm(H[:, :3], axis=1))
            if mgn > best_m:
                best, best_m = p, mgn
        _interior_point.p = best
    return _interior_point.p


def domain_vertices() -> np.ndarray:
    hs = HalfspaceIntersection(domain_halfspaces(), _interior_point())
    return hs.intersections


def in_domain(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    H = domain_halfspaces()
    return np.all(points @ H[:, :3].T + H[:, 3] <= eps, axis=1)


# ---------------------------------------------------------------- classification
_TREE_CACHE: dict = {}


def _perms(vertices: np.ndarray) -> list[np.ndarray]:
    tree = cKDTree(vertices)
    perms = []
    for g in ACTION.generators:
        d, idx = tree.query(vertices @ g.T)
        assert d.max() < 1e-6
        perms.append(idx)
    return perms


def _classes(items: list[frozenset], perms: list[np.ndarray]) -> list[list[int]]:
    index = {item: i for i, item in enumerate(items)}
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in perms:
        for i, item in enumerate(items):
            j = index.get(frozenset(int(perm[v]) for v in item))
            if j is None:
                raise ValueError("hull not symmetric")
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def classify(seed: np.ndarray, hull_tol: float = 1e-9) -> dict | None:
    """Combinatorial type of the convex hull of the seed's orbit; None if it is not a 4D polytope."""
    vertices = generate_vertices_from_seed(np.asarray(seed, dtype=float), ACTION, tol=1e-9)
    if len(vertices) < 5:
        return None
    try:
        faces, cells = compute_convex_hull(vertices, tol=hull_tol)
    except ValueError:
        return None
    edges = hull_edges(faces)
    perms = _perms(vertices)
    face_sets = [frozenset(f) for f in faces]
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    edge_sets = [frozenset(e) for e in edges]
    valence = {e: 0 for e in edge_sets}
    for c in cells:
        for e in {frozenset(e) for f in c for e in zip(faces[f], faces[f][1:] + faces[f][:1])}:
            valence[e] += 1

    def describe(classes, sets, label):
        out = sorted((len(cl), label(sets[cl[0]], cl[0])) for cl in classes)
        return out

    try:
        cell_classes = _classes(cell_sets, perms)
        face_classes = _classes(face_sets, perms)
        edge_classes = _classes(edge_sets, perms)
    except ValueError:       # on a wall the hull merges inconsistently; treated as "no type"
        return None
    return {
        "vertices": len(vertices),
        "cells": describe(cell_classes, cell_sets, lambda s, i: (len(s), len(cells[i]))),
        "faces": describe(face_classes, face_sets, lambda s, i: len(s)),
        "edges": describe(edge_classes, edge_sets, lambda s, i: valence[s]),
    }


def signature(info: dict | None) -> str:
    if info is None:
        return "none"
    cells = "+".join(f"{n}x[{v}v{f}f]" for n, (v, f) in info["cells"])
    faces = "+".join(f"{n}x{s}g" for n, s in info["faces"])
    edges = "+".join(f"{n}x{k}" for n, k in info["edges"])
    return f"{info['vertices']}: cells {cells} | faces {faces} | edges {edges}"
