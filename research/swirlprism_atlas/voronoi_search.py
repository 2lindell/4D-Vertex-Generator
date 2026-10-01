"""Voronoi (Dirichlet) domains of the swirlprism group G whose edges follow symmetry axes.

An edge lies on an axis when both of its ends lie in the axis's plane (the fixed great circle of a
rotation). Axes: fixed circles of G's half-turns (cross rings), of G's other rotations with a fixed
circle (the main ring), and of the half-turns in the coset G.Q (the light purple 2400-symmetry axes).
"""
from __future__ import annotations

import numpy as np
from cellframe import seed_from_beta
from dirichlet import dirichlet
from normalizer import halfturn

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

G_ELEMENTS = np.stack(group_elements(named_symmetry("h4_swirlprism")))


def _axis_planes() -> list[tuple[str, np.ndarray]]:
    planes: list[tuple[str, np.ndarray]] = []
    for kind, elements in (("G", G_ELEMENTS), ("coset", G_ELEMENTS @ halfturn())):
        for g in elements:
            _, s, vt = np.linalg.svd(g - np.eye(4))
            if np.sum(s < 1e-9) != 2:
                continue
            F = vt[2:]
            if not any(np.allclose(np.abs(np.linalg.det(F @ P.T)), 1, atol=1e-9) for _, P in planes):
                order = next(k for k in range(1, 61) if np.allclose(np.linalg.matrix_power(g, k), np.eye(4), atol=1e-7))
                label = "purple axis" if kind == "coset" else ("cross ring" if order == 2 else "main ring")
                planes.append((label, F))
    return planes


AXES = _axis_planes()


def edges_of(domain):
    """Domain edges as pairs of corner indices (from the face polygons, in the chart)."""
    chart = domain["corners_chart"]
    out = set()
    for f in domain["faces"]:
        idx = f["corners"]
        pts = chart[idx]
        mid = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mid)
        order = [idx[i] for i in np.argsort(np.arctan2((pts - mid) @ vt[1], (pts - mid) @ vt[0]))]
        for a, b in zip(order, order[1:] + order[:1]):
            out.add((min(a, b), max(a, b)))
    return sorted(out)


def edge_report(beta_p):
    p = seed_from_beta(np.asarray(beta_p, float))
    if any(np.allclose(g @ p, p, atol=1e-9) for g in G_ELEMENTS[1:]):
        return None
    D = dirichlet(p, G_ELEMENTS)
    X = D["corners"]
    total, on, kinds = 0.0, 0.0, {}
    for a, b in edges_of(D):
        length = float(np.arccos(np.clip(X[a] @ X[b], -1, 1)))
        total += length
        for label, F in AXES:
            if all(np.linalg.norm(X[i] - (X[i] @ F.T) @ F) < 1e-8 for i in (a, b)):
                on += length
                kinds[label] = kinds.get(label, 0) + 1
                break
    return {"share": on / total, "edges": len(edges_of(D)), "on_axes": kinds,
            "corners": len(X), "faces": len(D["faces"])}
