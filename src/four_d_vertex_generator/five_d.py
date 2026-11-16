"""5D coordinates for the pentachoric (A4) symmetries: the polychoron as one orthant facet of a B5 polyteron.

The A4 groups (pennic, kipennic, decaic, kidecaic, iodecaic) act on the hyperplane x1 + ... + x5 = const of 5D space:
the reflections of [3,3,3] are the transpositions of the coordinates, and the decaic swap is the central inversion of
that hyperplane about its centre point (c, c, c, c, c). The generator works in A4's Coxeter root basis in 4D; this module
maps that frame isometrically onto the hyperplane, sending the four Coxeter roots to (e_i - e_{i+1}) / sqrt(2), so that
the group's reflections become coordinate swaps. Edge lengths and angles are kept.

The 5D points are then moved along (1, 1, 1, 1, 1) so that the smallest coordinate is 0. With every coordinate >= 0 the
polychoron is the facet, in the all-positive orthant (normal (1, 1, 1, 1, 1)), of the B5 polyteron whose vertices are all
the sign changes and permutations of these points: its 32 such orthant facets are the polychoron's 32 copies.
"""
from __future__ import annotations

import numpy as np

A4_EDGES = ((0, 1, 3), (1, 2, 3), (2, 3, 3))
# (the generator's names for the A4 symmetries, and the aliases that resolve to them)
PENTACHORIC_SYMMETRIES = frozenset({
    "a4", "a4+", "a4_extended", "a4_chiral_extended", "a4_rotoreflection", "iodecaic", "a4_basic", "a4_chiral",
    "a4_extended_chiral", "pennic", "kipennic", "decaic", "kidecaic",
})


def _frame() -> np.ndarray:
    """The 5x4 isometry from the generator's A4 frame onto the hyperplane sum = 0 of R^5."""
    from .library import _coxeter_roots

    roots = _coxeter_roots(A4_EDGES)                    # (rows: the four unit roots)
    target = np.zeros((4, 5))
    for i in range(4):
        target[i, i], target[i, i + 1] = 1.0, -1.0
    target /= np.sqrt(2.0)
    return target.T @ np.linalg.inv(roots.T)


def has_5d_coordinates(symmetry: str) -> bool:
    """Is this one of the A4 symmetries, whose polychora have 5D orthant-facet coordinates?"""
    return symmetry in PENTACHORIC_SYMMETRIES


def to_5d(vertices: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    """The vertices (in the A4 frame) as 5D points with smallest coordinate 0: one orthant facet of a B5 polyteron.

    Every point has the same coordinate sum, and coordinates within tol of an integer multiple of the smallest nonzero
    one are left as computed (no rounding is applied beyond clearing values within tol of 0).
    """
    V = np.asarray(vertices, float)
    P = V @ _frame().T
    P -= P.min()
    P[np.abs(P) < tol] = 0.0
    return P


def to_5d_text(points: np.ndarray, *, digits: int = 12) -> str:
    """The 5D points as plain text, one per line (x1 x2 x3 x4 x5), after a comment header."""
    lines = ["# 5D coordinates: one orthant facet (x1..x5 >= 0, x1+...+x5 = "
             f"{points.sum(1).mean():.{digits}g}) of a B5 polyteron; {len(points)} vertices"]
    lines += [" ".join(f"{x:.{digits}g}" for x in p) for p in points]
    return "\n".join(lines) + "\n"
