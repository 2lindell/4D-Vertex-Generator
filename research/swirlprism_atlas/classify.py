"""Classify convex hulls of h4_swirlprism orbits by symmetry-class element counts."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

ACTION = named_symmetry("h4_swirlprism")


def _perms(vertices: np.ndarray) -> list[np.ndarray]:
    tree = cKDTree(vertices)
    perms = []
    for g in ACTION.generators:
        d, idx = tree.query(vertices @ g.T)
        assert d.max() < 1e-5
        perms.append(idx)
    return perms


def _class_sizes(items: list[frozenset], perms: list[np.ndarray]) -> list[int]:
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
                raise ValueError("hull not symmetric (merge tolerance issue)")
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
    sizes: dict[int, int] = {}
    for i in range(len(items)):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return sorted(sizes.values())


def classify(seed: np.ndarray, hull_tol: float = 1e-5) -> dict:
    vertices = generate_vertices_from_seed(seed, ACTION, tol=1e-6)
    faces, cells = compute_convex_hull(vertices, tol=hull_tol)
    edges = hull_edges(faces)
    perms = _perms(vertices)
    face_sets = [frozenset(f) for f in faces]
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    edge_sets = [frozenset(e) for e in edges]
    # edge valence = number of cells containing the edge
    valence: dict[frozenset, int] = {e: 0 for e in edge_sets}
    for c in cells:
        cell_edges = {frozenset(e) for f in c for e in zip(faces[f], faces[f][1:] + faces[f][:1])}
        for e in cell_edges:
            valence[e] += 1
    edge_classes_by_valence: dict[int, int] = {}
    for e, k in valence.items():
        edge_classes_by_valence[k] = edge_classes_by_valence.get(k, 0) + 1
    return {
        "vertices": len(vertices),
        "cells": _class_sizes(cell_sets, perms),
        "faces": _class_sizes(face_sets, perms),
        "edges": _class_sizes(edge_sets, perms),
        "valence": dict(sorted(edge_classes_by_valence.items())),
    }


def signature(info: dict) -> str:
    def j(xs: list[int]) -> str:
        return "+".join(map(str, xs))

    return f"{info['vertices']}: cells {j(info['cells'])} | faces {j(info['faces'])} | edges {j(info['edges'])}"
