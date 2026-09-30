"""Exact T2 regions in the mirrors (from planemap.py + conicfit.py) and their copies under Q."""
from __future__ import annotations

import numpy as np
from cellframe import PHI, T

F2 = PHI * PHI


def conic_a(x: float) -> float:
    """On beta1 = beta3 = a (sum 1): the T2/X17 boundary a^2 + phi^2 x^2 - a x - a y - phi x y = 0, y = 1 - 2a - x."""
    # substituting y gives 3a^2 + (2 phi x - 1) a + (phi^2 + phi) x^2 - phi x = 0; take the root through (1/3, 0, 1/3, 1/3)
    b, c = 2 * PHI * x - 1, (F2 + PHI) * x * x - PHI * x
    return (-b + np.sqrt(b * b - 12 * c)) / 6


def patch_a(n: int = 24) -> np.ndarray:
    """In the mirror beta1 = beta3: spidrox -> C4 along T1, the conic to the cell centre, the E2 line
    to the face centre, and the face beta4 = 0 back to spidrox."""
    pts = [np.array([0.5, 0, 0.5, 0]), np.array([1, 0, 1, 1]) / 3]
    for x in np.linspace(0, 0.25, n + 1)[1:]:
        a = conic_a(x)
        pts.append(np.array([a, x, a, 1 - 2 * a - x]))
    pts.append(np.array([1, 1, 1, 0]) / 3)
    return np.array(pts)


def patch_b() -> np.ndarray:
    """In the mirror beta2 = beta3: spidrox (0,1,1,0) -> C4 (0,1,1,1) along T1, the line
    phi^2 beta1 = beta2 - beta4 to (1, phi^2, phi^2, 0), and the face beta4 = 0 (a C2a cross ring) back."""
    pts = np.array([[0, 1, 1, 0], [0, 1, 1, 1], [1, F2, F2, 0]], float)
    return pts / pts.sum(axis=1, keepdims=True)


def to_x(beta_poly: np.ndarray) -> np.ndarray:
    return beta_poly @ T


PATCHES = {"A": patch_a, "B": patch_b}
DESCRIPTIONS = {  # name -> (mirror, boundary description)
    "A": ("β1 = β3", "corners spidrox (1,0,1,0), C4 (1,0,1,1), cell centre C1, face centre C5; sides: T1 line, "
                     "conic β1² + φ²β2² − β1β2 − β1β4 − φβ2β4 = 0, E2 line, face β4 = 0"),
    "B": ("β2 = β3", "triangle spidrox (0,1,1,0), C4 (0,1,1,1), (1, φ², φ², 0); sides: T1 line, "
                     "line φ²β1 = β2 − β4, face β4 = 0 (C2a cross ring)"),
}


def copies(name: str) -> list[np.ndarray]:
    from normalizer import G, displayed_pieces, halfturn

    from four_d_vertex_generator.generation import group_elements
    E = np.stack(group_elements(G))
    X = to_x(PATCHES[name]()) @ halfturn().T
    return displayed_pieces(X, E)


if __name__ == "__main__":
    import json

    from catalog import match
    from cell_atlas import to_upper
    S = json.load(open("golden_22.json"))
    B = [to_upper(s["beta"]) for s in S if match(s["sig"]) == "T2"]
    B = [b / b.sum() for b in B]
    for name in PATCHES:
        pieces = copies(name)
        print(name, "copies:", len(pieces), [len(p) for p in pieces])

        def inside(b, P):
            # b in the planar polygon P: coplanar, and inside or on it by winding angle (P need not be convex)
            c = P.mean(axis=0)
            _, s, vt = np.linalg.svd(np.vstack([P - c, b - c]))
            if s[2] > 1e-9:
                return False
            q = (P - c) @ vt[:2].T - (b - c) @ vt[:2].T
            if np.min(np.linalg.norm(q, axis=1)) < 1e-9:
                return True
            for i in range(len(q)):          # on an edge
                u, w = q[i], q[(i + 1) % len(q)]
                if abs(u[0] * w[1] - u[1] * w[0]) < 1e-9 * np.linalg.norm(w - u) and u @ w <= 0:
                    return True
            ang = np.arctan2(q[:, 1], q[:, 0])
            turn = np.sum((np.roll(ang, -1) - ang + np.pi) % (2 * np.pi) - np.pi)
            return abs(turn) > np.pi

        hits = sum(any(inside(b, P) for P in pieces) for b in B)
        own = sum(inside(b, PATCHES[name]()) for b in B)
        print(f"  T2 samples inside the patch: {own}, inside its copies: {hits}")
