"""Groups containing the swirlprism group G, and the lines on which G's orbits gain their symmetry.

G = (left 2I) x (right 2D10), where 2D10 = <zeta, j> with zeta = cos 36 + u sin 36 (u = zeta's axis). The
groups G_k = (left 2I) x (right 2D_{10k}) replace zeta by cos(36/k) + u sin(36/k); G_2 is the 2400-element
group of the light purple axes (G and Q). A seed fixed by an element h of G_k but not of G has a G-orbit of
1200 points that G_k maps to itself exactly when its stabilizer in G_k has order k.
"""
from __future__ import annotations

import numpy as np

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import (
    _ICOSIAN_ORDER_TEN,
    _quaternion_left_matrix,
    _quaternion_right_matrix,
)
from four_d_vertex_generator.symmetry import SymmetryAction

_LEFT = (
    (-0.809016994374947, -0.309016994374947, 0.0, 0.5),
    (0.809016994374947, -0.309016994374947, 0.0, -0.5),
)


def supergroup(k: int) -> np.ndarray:
    w = np.array(_ICOSIAN_ORDER_TEN)
    u = w[1:] / np.linalg.norm(w[1:])
    angle = np.arccos(w[0]) / k
    zeta_k = (np.cos(angle), *(np.sin(angle) * u))
    gens = [_quaternion_left_matrix(q) for q in _LEFT]
    gens += [_quaternion_right_matrix(zeta_k), _quaternion_right_matrix((0.0, 0.0, 1.0, 0.0))]
    return np.stack(group_elements(SymmetryAction(generators=gens), 100_000))


def girdle_segments(k: int = 3, order: int = 3, tol: float = 1e-10):
    """Segments of the displayed half-cell fixed by the order-`order` rotations of G_k that are not in G.

    For k = 3, order = 3 these are Bowers' 20 ghost girdles with "30/3-gyrogonic" symmetry: seeds on them
    give 1200-vertex polytopes with 3600 symmetries. Same clipping and folding as normalizer.coset_axes.
    """
    from cellframe import TINV

    from four_d_vertex_generator.library import named_symmetry
    G = np.stack(group_elements(named_symmetry("h4_swirlprism")))
    known = {tuple(np.round(g, 7).ravel()) for g in G}
    out = []

    def add(a, b):
        a, b = a / a.sum(), b / b.sum()
        if np.linalg.norm(a - b) < 1e-9:
            return
        for p, q in out:
            if (np.allclose(p, a, atol=1e-9) and np.allclose(q, b, atol=1e-9)) or (
                np.allclose(p, b, atol=1e-9) and np.allclose(q, a, atol=1e-9)
            ):
                return
        out.append((a, b))

    for g in supergroup(k):
        if tuple(np.round(g, 7).ravel()) in known or np.allclose(g, np.eye(4)):
            continue
        if not np.allclose(np.linalg.matrix_power(g, order), np.eye(4), atol=1e-7):
            continue
        _, s, vt = np.linalg.svd(g - np.eye(4))
        if np.sum(s < 1e-9) != 2:
            continue
        A = vt[2:] @ TINV.T
        angles = []
        for k_ in range(4):
            if np.hypot(A[0, k_], A[1, k_]) > tol:
                t = np.arctan2(-A[0, k_], A[1, k_])
                angles += [t % (2 * np.pi), (t + np.pi) % (2 * np.pi)]
        fold = A[:, 2] - A[:, 3]
        if np.hypot(*fold) > tol:
            t = np.arctan2(-fold[0], fold[1])
            angles += [t % (2 * np.pi), (t + np.pi) % (2 * np.pi)]
        angles = sorted(set(np.round(angles, 14)))
        for i, t0 in enumerate(angles):
            t1 = angles[(i + 1) % len(angles)] + (2 * np.pi if i == len(angles) - 1 else 0)
            mid = np.cos((t0 + t1) / 2) * A[0] + np.sin((t0 + t1) / 2) * A[1]
            if np.any(mid < -tol) or mid.sum() <= 0:
                continue
            a = np.clip(np.cos(t0) * A[0] + np.sin(t0) * A[1], 0, None)
            b = np.clip(np.cos(t1) * A[0] + np.sin(t1) * A[1], 0, None)
            if mid[2] < mid[3]:
                a, b = a[[1, 0, 3, 2]], b[[1, 0, 3, 2]]
            add(a, b)
    return out


def girdle_families():
    """Returns f(a, b): the size of the G-family (150 or 30) of the 2400-axis circle through a segment."""
    from cellframe import T

    from four_d_vertex_generator.library import named_symmetry
    G = np.stack(group_elements(named_symmetry("h4_swirlprism")))
    known = {tuple(np.round(g, 7).ravel()) for g in G}
    planes = []
    for g in supergroup(2):
        if tuple(np.round(g, 7).ravel()) in known or not np.allclose(g @ g, np.eye(4), atol=1e-9):
            continue
        _, s, vt = np.linalg.svd(g - np.eye(4))
        if np.sum(s < 1e-9) == 2 and not any(
            abs(abs(np.linalg.det(vt[2:] @ P.T)) - 1) < 1e-9 for P in planes
        ):
            planes.append(vt[2:])
    size = {}
    for i in range(len(planes)):
        if i in size:
            continue
        orbit = set()
        for g in G:
            Q = planes[i] @ g.T
            orbit.add(next(j for j, R in enumerate(planes) if abs(abs(np.linalg.det(Q @ R.T)) - 1) < 1e-9))
        for j in orbit:
            size[j] = len(orbit)

    def family(a, b):
        for beta in (np.asarray((a + b) / 2), np.asarray((a + b) / 2)[[1, 0, 3, 2]]):
            x = beta @ T
            x = x / np.linalg.norm(x)
            for i, F in enumerate(planes):
                if np.linalg.norm(x - (x @ F.T) @ F) < 1e-7:
                    return size[i]
        return None

    return family
