"""Independent consistency checks for a 4D convex hull.

Faces are vertex cycles and cells are lists of face indices.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def check_hull(
    verts: np.ndarray,
    faces: list[list[int]],
    cells: list[list[int]],
    generators: list[np.ndarray] | None = None,
    margin: float = 1e-9,
) -> None:
    """Assert that faces and cells really are the boundary of the convex hull of ``verts``.

    Checked independently of how the hull was computed:
    - every cell's vertices lie in one hyperplane with every other vertex strictly inside it,
      and no two cells share a hyperplane;
    - every face is a flat, convex polygon listed in cyclic order, lies in exactly two cells,
      and is exactly the set of vertices those two cells share;
    - each cell is a closed polyhedron and V - E + F - C = 0;
    - every vertex of the hull (an extreme point) is used;
    - with ``generators``, each generator maps faces onto faces and cells onto cells.
    """
    verts = np.asarray(verts, float)
    radius = float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1)))
    tol = margin * radius
    cell_vertices = [sorted({v for f in cell for v in faces[f]}) for cell in cells]

    planes = []
    for vs in cell_vertices:
        pts = verts[vs]
        mid = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - mid)
        assert s[2] > tol, "a cell is flat (lower-dimensional)"
        normal = vt[3]
        offset = float(normal @ mid)
        dist = verts @ normal - offset
        if dist.max() > -dist.min():
            normal, offset, dist = -normal, -offset, -dist
        # flatness is judged at ``margin``; lying "on" the cell only within float noise, since a
        # vertex any farther inside is resolved by the hull however close it is
        on = set(np.flatnonzero(np.abs(dist) <= tol * 1e-3).tolist()) | set(vs)
        # other points on the hyperplane are allowed only as duplicates of the cell's vertices
        extra = sorted(on - set(vs))
        if extra:
            gaps = np.min(np.linalg.norm(verts[extra][:, None] - verts[vs][None], axis=2), axis=1)
            assert gaps.max() <= tol, "a cell's hyperplane holds vertices the cell does not list"
        assert np.abs(dist[vs]).max() <= tol, "a cell's vertices are not in one hyperplane"
        assert dist.max() <= tol, "a vertex lies outside a cell's hyperplane"
        planes.append(np.append(normal, offset))
    planes_arr = np.array(planes)
    for i in range(len(planes_arr)):
        if i + 1 < len(planes_arr):
            diff = np.abs(planes_arr[i + 1:] - planes_arr[i]).max(axis=1)
            assert not np.any(diff < 1e-9), "two cells share a hyperplane"

    face_cells: dict[int, list[int]] = defaultdict(list)
    for c, cell in enumerate(cells):
        for f in cell:
            face_cells[f].append(c)
    for f, face in enumerate(faces):
        assert len(face) >= 3 and len(set(face)) == len(face), "a face repeats a vertex or has < 3"
        assert len(face_cells[f]) == 2, "a face is not in exactly two cells"
        a, b = face_cells[f]
        common = set(cell_vertices[a]) & set(cell_vertices[b])
        assert set(face) == common, "a face is not the two cells' common part"
        pts = verts[face]
        mid = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - mid)
        assert s[1] > 0 and s[2] <= tol * max(1, len(face)), "a face is not flat"
        q = (pts - mid) @ vt[:2].T
        d = np.roll(q, -1, axis=0) - q                 # edge vectors around the polygon
        e = np.roll(d, -1, axis=0)
        cross = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]  # turn at each vertex
        convex = all(c > 0 for c in cross) or all(c < 0 for c in cross)
        assert convex, "a face is not convex, or its vertices are out of order"

    edges: set[tuple[int, int]] = set()
    for c, cell in enumerate(cells):
        uses: dict[tuple[int, int], int] = defaultdict(int)
        for f in cell:
            face = faces[f]
            for u, w in zip(face, face[1:] + face[:1]):
                uses[(min(u, w), max(u, w))] += 1
        assert set(uses.values()) == {2}, "a cell is not closed"
        assert len(cell_vertices[c]) - len(uses) + len(cell) == 2, "a cell is not a polyhedron"
        edges.update(uses)
    used = {v for face in faces for v in face}
    assert len(used) - len(edges) + len(faces) - len(cells) == 0, "Euler characteristic is not 0"

    from scipy.spatial import ConvexHull
    extreme = set(ConvexHull(verts).vertices.tolist())
    assert extreme <= used, "an extreme point of the set is missing from the hull"

    if generators is not None:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        face_keys = {frozenset(f) for f in faces}
        cell_keys = {frozenset(vs) for vs in cell_vertices}
        for g in generators:
            d, perm = tree.query(verts @ np.asarray(g).T)
            assert d.max() < 1e-7 * max(1.0, radius), "a generator moves a vertex off the set"
            mapped_faces = {frozenset(int(perm[v]) for v in f) for f in faces}
            mapped_cells = {frozenset(int(perm[v]) for v in vs) for vs in cell_vertices}
            assert mapped_faces == face_keys, "faces are not symmetric"
            assert mapped_cells == cell_keys, "cells are not symmetric"
