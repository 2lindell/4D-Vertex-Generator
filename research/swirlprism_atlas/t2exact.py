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


def patch_c() -> np.ndarray:
    """In the mirror beta1 = beta2: spidrox (1,1,0,0) -> face centre C5 (1,1,1,0) along the face beta4 = 0,
    the line beta1 = beta3 + phi^2 beta4 to (2+phi, 2+phi, 1, 1) (the C5 end of the E2 copy), and the
    cross ring beta3 = beta4 back to spidrox."""
    pts = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1 + F2, 1 + F2, 1, 1]], float)
    return pts / pts.sum(axis=1, keepdims=True)


def curve_d(n: int = 32) -> np.ndarray:
    """In the mirror beta2 = beta3: the conic phi b1^2 - b1 b2 - b1 b4 / phi + b2^2 - b4^2 = 0 from
    C4 (0,1,1,1) to the cell centre C1, a T2 curve between X21 and X27 (not the edge of a patch)."""
    pts = []
    for t in np.linspace(0, 0.25, n + 1):
        # b1 = t, b2 = b3 = (1 - t - u) / 2, b4 = u; the conic is quadratic in u
        def f(u, t=t):
            b2 = (1 - t - u) / 2
            return PHI * t * t - t * b2 - t * u / PHI + b2 * b2 - u * u
        if t == 0:
            pts.append([0.0, 1 / 3, 1 / 3, 1 / 3])   # C4 exactly: the conic gives b4 = b2 there
            continue
        lo, hi = (1 - t) / 3 - 0.2, 1 - t
        lo = max(lo, 0.0)
        us = np.linspace(lo, hi, 2001)
        fs = np.array([f(u) for u in us])
        k = int(np.flatnonzero(np.sign(fs[:-1]) != np.sign(fs[1:]))[0])
        a, b = us[k], us[min(k + 1, len(us) - 1)]
        for _ in range(80):
            m = (a + b) / 2
            if np.sign(f(m)) == np.sign(f(a)):
                a = m
            else:
                b = m
        u = (a + b) / 2
        pts.append([t, (1 - t - u) / 2, (1 - t - u) / 2, u])
    return np.array(pts)


def curve_copies(curve: np.ndarray, jump: float = 0.05) -> list[np.ndarray]:
    """Q-images of a sampled curve in the displayed half-cell, chained into polylines."""
    from normalizer import qcopies
    chains: list[list[np.ndarray]] = []
    for p in curve:
        for q in qcopies(p):
            best = min(chains, key=lambda c: np.linalg.norm(c[-1] - q), default=None)
            if best is not None and np.linalg.norm(best[-1] - q) < jump:
                best.append(q)
            else:
                chains.append([q])
    return [np.array(c) for c in chains if len(c) >= 2]


def to_x(beta_poly: np.ndarray) -> np.ndarray:
    return beta_poly @ T


PATCHES = {"A": patch_a, "B": patch_b, "C": patch_c}
CURVES = {"D": curve_d}
CURVE_DESCRIPTIONS = {
    "D": ("β2 = β3", "conic φβ1² − β1β2 − β1β4/φ + β2² − β4² = 0 from C4 (0,1,1,1) to the cell centre C1, "
                     "between X21 and X27"),
}
DESCRIPTIONS = {  # name -> (mirror, boundary description)
    "A": ("β1 = β3", "corners spidrox (1,0,1,0), C4 (1,0,1,1), cell centre C1, face centre C5; sides: T1 line, "
                     "conic β1² + φ²β2² − β1β2 − β1β4 − φβ2β4 = 0, E2 line, face β4 = 0"),
    "B": ("β2 = β3", "triangle spidrox (0,1,1,0), C4 (0,1,1,1), (1, φ², φ², 0); sides: T1 line, "
                     "line φ²β1 = β2 − β4, face β4 = 0 (C2a cross ring)"),
    "C": ("β1 = β2", "triangle spidrox (1,1,0,0), face centre C5 (1,1,1,0), (2+φ, 2+φ, 1, 1); sides: face β4 = 0, "
                     "line β1 = β3 + φ²β4, cross ring β3 = β4"),
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
