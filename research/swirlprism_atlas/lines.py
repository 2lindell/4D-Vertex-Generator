"""Map the ring results (cross rings and main rings) into the gnomonic dodecahedron."""
from __future__ import annotations

import json

import numpy as np
from survey import BASIS, C1, FULL, A, in_cell

from four_d_vertex_generator.generation import group_elements

PHI = (1 + 5**0.5) / 2
AT = lambda x: float(np.degrees(np.arctan(x)))  # noqa: E731
H = AT(PHI) / 2

# Cross ring 1 types by angle (period 90°), from the ring sweep.
POINTS = [  # (angle, id)
    (0.0, "p600"), (AT(PHI**-3), "p120cell"), (AT(PHI**-2), "prss"), (AT(0.5), "psdr"),
    (H, "icosafold"), (AT(1 / PHI), "psdr"), (AT(2 * PHI**-2), "prss"), (45.0, "p120cell"),
    (AT(PHI), "p600"), (AT(PHI**2), "bigyro"), (45 + H, "icosafold"), (AT(2 * PHI**2), "bigyro"),
]
RANGES = [  # (start, end, id)
    (0.0, AT(PHI**-3), "r600e"), (AT(PHI**-3), AT(PHI**-2), "r360"), (AT(PHI**-2), AT(0.5), "rC"),
    (AT(0.5), AT(1 / PHI), "rCmid"), (AT(1 / PHI), AT(2 * PHI**-2), "rC"),
    (AT(2 * PHI**-2), 45.0, "r360"), (45.0, AT(PHI), "r600e"), (AT(PHI), AT(PHI**2), "r600d"),
    (AT(PHI**2), AT(2 * PHI**2), "rD"), (AT(2 * PHI**2), 90.0, "r600d"),
]

ELEMENTS = group_elements(FULL)
STACK = np.stack(ELEMENTS)


def ring_parameter(p: np.ndarray) -> float:
    """Angle along cross ring 1 (mod 90) of a point on some half-turn circle."""
    X = STACK @ p  # images of p under every element
    resid = X - np.outer(X @ A, A) - np.outer(X @ C1, C1)  # on cross ring 1: in span(A, C1)
    i = int(np.argmin(np.linalg.norm(resid, axis=1)))
    if np.linalg.norm(resid[i]) > 1e-7:
        raise ValueError("point is not on a half-turn circle")
    return float(np.degrees(np.arctan2(X[i] @ C1, X[i] @ A))) % 90.0


def range_id(t: float) -> str:
    for a, b, rid in RANGES:
        if a - 1e-9 <= t <= b + 1e-9:
            return rid
    return "?"


def fixed_circles(order: int) -> list[np.ndarray]:
    """Planes (2x4 orthonormal) fixed pointwise by elements of the given order."""
    planes: list[np.ndarray] = []
    for e in ELEMENTS:
        if np.allclose(e, np.eye(4)):
            continue
        k = next(n for n in range(1, 61) if np.allclose(np.linalg.matrix_power(e, n), np.eye(4), atol=1e-7))
        if k != order:
            continue
        w, v = np.linalg.eigh((e + e.T) / 2)
        F = v[:, np.isclose(w, 1.0, atol=1e-7)].T
        if len(F) != 2:
            continue
        if not any(abs(abs(np.linalg.det(F @ P.T)) - 1) < 1e-6 for P in planes):
            planes.append(F)
    return planes


def clip(plane: np.ndarray, samples: int = 3000) -> list[tuple[np.ndarray, np.ndarray]]:
    """Points of the circle inside the cell, as (seed, q) pairs."""
    out = []
    for s in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        p = np.cos(s) * plane[0] + np.sin(s) * plane[1]
        if p @ A <= 0:
            continue
        q = BASIS @ (p / (p @ A))
        if in_cell(q, slack=1e-9):
            out.append((p, q))
    return out


if __name__ == "__main__":
    data = {"cross": [], "main": []}
    for F in fixed_circles(2):
        pts = clip(F)
        if len(pts) < 2:
            continue
        seg = []
        for p, q in pts:
            t = ring_parameter(p)
            seg.append({"q": [round(float(c), 6) for c in q], "t": round(t, 5), "id": range_id(t)})
        data["cross"].append(seg)
    for F in fixed_circles(5):
        pts = clip(F)
        if len(pts) >= 2:
            data["main"].append([[round(float(c), 6) for c in q] for _, q in pts])
    # special points of cross ring 1, mapped to every image inside the cell
    specials = []
    for t, pid in POINTS:
        p = np.cos(np.radians(t)) * A + np.sin(np.radians(t)) * C1
        seen = []
        for g in ELEMENTS:
            x = g @ p
            if x @ A <= 0:
                continue
            q = BASIS @ (x / (x @ A))
            if in_cell(q, slack=1e-9) and not any(np.linalg.norm(q - s) < 1e-6 for s in seen):
                seen.append(q)
        for q in seen:
            specials.append({"q": [round(float(c), 6) for c in q], "id": pid, "t": round(t, 5)})
    data["points"] = specials
    json.dump(data, open("lines.json", "w"))
    print(f"cross-ring segments in cell: {len(data['cross'])}, main-ring segments: {len(data['main'])}, special points: {len(specials)}")
    from collections import Counter
    print(Counter(s["id"] for s in specials))
