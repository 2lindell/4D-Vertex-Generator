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


def _cell16() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """A tetrahedral cell of a 16-cell the group maps to itself, and the chart at its centre.

    The group's orbits of 8 points that are the vertices of a 16-cell (each such point lies on the fixed planes of two
    rotations); of the 16-cells, the first found. The group moves its 16 cells into one another, and a cell is fixed
    only by an order-3 rotation about one of its vertices V (through the centre of the opposite face): one third of
    the cell is a fundamental domain. Returns (anchor = the cell's centre, chart frame with e3 along V, the rotation,
    3, the cell's vertices [V, W1, W2, W3] with W1, W2, W3 at azimuths 0, 120 and 240 degrees about e3)."""
    import itertools
    planes = [np.linalg.svd(g - np.eye(4))[2][2:] for g in ELEMENTS if np.linalg.matrix_rank(g - np.eye(4), tol=1e-8) == 2]
    for P, Q in itertools.combinations(planes, 2):
        _, sv, vt = np.linalg.svd(np.vstack([np.eye(4) - P.T @ P, np.eye(4) - Q.T @ Q]))
        for k in range(4):
            if sv[k] > 1e-8:
                continue
            O = []
            for q in ELEMENTS @ vt[k]:
                if not any(np.linalg.norm(q - u) < 1e-7 for u in O):
                    O.append(q)
            O = np.array(O)
            if len(O) != 8 or not set(np.round(O @ O.T, 5).ravel()) <= {1.0, -1.0, 0.0}:
                continue
            B = []
            for u in O:
                if not any(abs(abs(u @ b) - 1) < 1e-6 for b in B):
                    B.append(u)
            B = np.array(B)                            # (one of each opposite pair: a cell's vertices)
            c = B.sum(0) / 2
            stab = [g for g in ELEMENTS if np.linalg.norm(g @ c - c) < 1e-9 and np.trace(g) < 3.99]
            g = stab[0]
            v = [i for i in range(4) if np.linalg.norm(g @ B[i] - B[i]) < 1e-9][0]
            W = [i for i in range(4) if i != v]
            w1 = W[0]
            w2 = [i for i in W if np.linalg.norm(g @ B[w1] - B[i]) < 1e-9][0]
            w3 = [i for i in W if i not in (w1, w2)][0]
            cell = B[[v, w1, w2, w3]]
            a = c / np.linalg.norm(c)
            to_chart = lambda x: x / (x @ a) - a
            e3 = to_chart(cell[0])
            e3 /= np.linalg.norm(e3)
            e1 = to_chart(cell[1]) - (to_chart(cell[1]) @ e3) * e3
            e1 /= np.linalg.norm(e1)
            # (e2 completes a right-handed frame of a-perp, oriented so W2 is at +120 degrees)
            rest = np.linalg.svd(np.vstack([a, e1, e3]))[2][3]
            e2 = rest if to_chart(cell[2]) @ rest > 0 else -rest
            return a, np.stack([e1, e2, e3]), g, 3, cell
    raise ValueError("no 16-cell is fixed by the group")


# (the octahedral group's domain is a third of a 16-cell's tetrahedron, as the swirlprism atlas uses half of a 600-cell's;
# the tetrahedral group keeps the Voronoi wedge: its 24 rotations do not divide the 16 cells evenly)
DOMAIN = os.environ.get("SWIRL_DOMAIN", "cell16" if KEY == "oct" else "voronoi")
if DOMAIN == "cell16":
    A, FRAME, STAB, M, CELL = _cell16()
else:
    A, FRAME, STAB, M = _anchor_frame()
    CELL = None


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
    if DOMAIN == "cell16":                   # (the 16-cell's tetrahedron: its four faces in the chart)
        P = chart_from_seed(CELL)
        rows = []
        for i in range(4):
            o = [j for j in range(4) if j != i]
            n = np.cross(P[o[1]] - P[o[0]], P[o[2]] - P[o[0]])
            if n @ (P[i] - P[o[0]]) > 0:
                n = -n
            rows.append(np.concatenate([n, [-(n @ P[o[0]])]]))
        return np.array(rows)
    return _voronoi_halfspaces()


def beta_from_seed(x: np.ndarray) -> np.ndarray:
    """Barycentric weights in the 16-cell's tetrahedron [V, W1, W2, W3] (cell16 domain only)."""
    b = np.linalg.solve(CELL.T, np.asarray(x, float).T).T
    return b / b.sum(axis=-1, keepdims=True)


def domain_faces() -> list[tuple[np.ndarray, str, str]]:
    """(halfspace row, name, kind) for every face that may bound the domain; kind "cell" for a face of the cell (or a
    bisector of the anchor and an image), "cut" for a cut through the rotation axis."""
    out = []
    if DOMAIN == "cell16":
        names = ["β_V = 0 (the face opposite V)", "β₁ = 0", "β₂ = 0", "β₃ = 0"]
        out += [(r, f"face {n} of the 16-cell's tetrahedron", "cell") for r, n in zip(cell_halfspaces(), names)]
        cuts = ["β₂ = β₃ (through V, W₁ and the face centre)", "β₁ = β₃ (through V, W₂ and the face centre)"]
    else:
        seen = []
        for g in ELEMENTS:
            ga = g @ A
            if np.linalg.norm(ga - A) < 1e-9 or any(np.linalg.norm(ga - s_) < 1e-9 for s_ in seen):
                continue
            seen.append(ga)
            w = A - ga
            ang = np.degrees(np.arccos(np.clip(A @ ga, -1, 1)))
            out.append((np.concatenate([-(FRAME @ w), [-(A @ w)]]), f"bisector of the anchor and an image {ang:.1f}° away", "cell"))
        cuts = [f"cut through the rotation axis at 0°", f"cut through the rotation axis at {round(360 / M)}°"]
    out += [(r, n, "cut") for r, n in zip(wedge_halfspaces(), cuts)]
    return out


def _voronoi_halfspaces() -> np.ndarray:
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
