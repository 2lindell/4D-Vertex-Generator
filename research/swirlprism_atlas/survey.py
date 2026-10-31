"""Survey the seed space of h4_swirlprism (H3●I2(10), order 1200).

Every seed is equivalent to one in the anchor's Voronoi dodecahedron (points
closer to the anchor than to any other 600-cell vertex). We use the gnomonic
projection at the anchor, q = p / (p . a) - a, in which great circles are
straight lines and the Voronoi cell is a flat-faced dodecahedron. The anchor's
stabilizer (D5, order 10) acts linearly on q, so only one tenth needs sampling:
z >= 0 and 0 <= azimuth < 72 degrees, with azimuth 0 along cross ring 1.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from classify import classify, signature

from four_d_vertex_generator.generation import generate_vertices_from_seed, group_elements
from four_d_vertex_generator.library import (
    _h4_swirlprism_ring_basis,
    h4_swirlprism_anchor_seed,
    named_symmetry,
)

FULL = named_symmetry("h4_swirlprism")
A = h4_swirlprism_anchor_seed()
C1, C2, MAIN = _h4_swirlprism_ring_basis()
E1 = C1
E2 = C2 - (C2 @ C1) * C1
E2 /= np.linalg.norm(E2)
E3 = MAIN
BASIS = np.array([E1, E2, E3])  # gnomonic x, y, z

_V = generate_vertices_from_seed(A, FULL, tol=1e-6)
_d = np.linalg.norm(_V - A, axis=1)
NEIGHBOURS = _V[np.isclose(_d, np.sort(_d)[1], atol=1e-9)]


def seed_from_q(q: np.ndarray) -> np.ndarray:
    p = A + np.asarray(q) @ BASIS
    return p / np.linalg.norm(p)


def q_from_seed(p: np.ndarray) -> np.ndarray:
    return BASIS @ (p / (p @ A)) - 0.0


def in_cell(q: np.ndarray, slack: float = 1e-12) -> bool:
    p = A + np.asarray(q) @ BASIS
    return bool(np.all(p @ A >= p @ NEIGHBOURS.T - slack))


def stabilizer_q_maps() -> list[np.ndarray]:
    """The 10 anchor-fixing symmetries as 3x3 maps on gnomonic coordinates."""
    maps = []
    for e in group_elements(FULL):
        if np.allclose(e @ A, A, atol=1e-8):
            maps.append(BASIS @ e @ BASIS.T)
    return maps


def work(q: tuple[float, float, float]) -> dict:
    try:
        info = classify(seed_from_q(np.array(q)))
        sig = signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    except Exception as exc:  # near a boundary the hull can come out asymmetric
        sig = f"ERR {type(exc).__name__}"
    return {"q": list(q), "sig": sig}


def grid(spacing: float) -> list[tuple[float, float, float]]:
    r_out = 0.45
    pts = []
    ticks = np.arange(-r_out, r_out + 1e-9, spacing)
    for x in ticks:
        for y in ticks:
            az = np.degrees(np.arctan2(y, x)) % 360
            if not (0 <= az < 72) and not (abs(x) < 1e-12 and abs(y) < 1e-12):
                continue
            for z in np.arange(0.0, r_out + 1e-9, spacing):
                q = (round(float(x), 6), round(float(y), 6), round(float(z), 6))
                if in_cell(np.array(q)):
                    pts.append(q)
    return pts


if __name__ == "__main__":
    spacing = float(sys.argv[1])
    out = sys.argv[2]
    pts = grid(spacing)
    print(f"{len(pts)} grid points at spacing {spacing}", flush=True)
    t0 = time.time()
    results = []
    with Pool(4) as pool:
        for i, r in enumerate(pool.imap_unordered(work, pts, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(pts)}  {time.time() - t0:.0f}s", flush=True)
                json.dump(results, open(out, "w"))
    json.dump(results, open(out, "w"))
    print("done", flush=True)
