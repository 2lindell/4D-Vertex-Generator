"""A region's view straight from the database's faces (region_pieces.json): every wall face with the region on one
side and something else on the other. With --check, compared wall by wall with the region's hand-checked colouring
(region_walls.json): the area each has that the other lacks.

    python region_view.py F2 --check
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

FLIP = np.array([1, 1, -1.0])


def view(rid, path="region_pieces.json"):
    """{wall: triangles (n, 3, 3) in cell xyz} of the faces the region borders."""
    fp = json.load(open(path))
    out = defaultdict(list)
    for wid, faces in fp.items():
        for f in faces:
            a, b = f["sides"]
            if f["mirror"] or (a == rid) == (b == rid):
                continue
            out[wid] += f["tris"]
    return {w: np.array(T).reshape(-1, 3, 3) for w, T in out.items()}


def area(T):
    return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)


def sample(T, step=4e-4, rng=np.random.default_rng(0)):
    """Points spread over the triangles, about one per step² of area."""
    a = area(T)
    n = np.maximum(1, np.ceil(a / step ** 2)).astype(int)
    idx = np.repeat(np.arange(len(T)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = T[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0]), idx


def uncovered(A, B, tol=1.2e-3, step=4e-4):
    """Area of A further than tol from B (both triangles), and the points where."""
    if len(A) == 0:
        return 0.0, np.zeros((0, 3))
    P, idx = sample(A, step)
    w = area(A)[idx] / np.bincount(idx, minlength=len(A))[idx]
    if len(B) == 0:
        return float(w.sum()), P
    Q, _ = sample(B, step / 2)
    d, _ = cKDTree(Q).query(P)
    far = d > tol
    return float(w[far].sum()), P[far]


def check(rid):
    mine = view(rid)
    r = json.load(open("region_walls.json"))[rid]
    T = (np.array(r["tris"]).reshape(-1, 3) * FLIP).reshape(-1, 3, 3)
    ids = np.array(r["ids"])
    ref = {w: T[ids == w] for w in set(ids)}
    print(f"{rid}: view from the database's faces against the checked colouring")
    print(f"  {'wall':6s} {'view':>8s} {'checked':>8s} {'extra':>8s} {'missing':>8s}")
    for w in sorted(set(mine) | set(ref)):
        A, B = mine.get(w, np.zeros((0, 3, 3))), ref.get(w, np.zeros((0, 3, 3)))
        ex, pe = uncovered(A, B)
        mi, pm = uncovered(B, A)
        print(f"  {w:6s} {area(A).sum():8.5f} {area(B).sum():8.5f} {ex:8.5f} {mi:8.5f}")
        check.where = getattr(check, "where", {})
        check.where[w] = (pe, pm)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check(sys.argv[1])
    else:
        v = view(sys.argv[1])
        print({w: round(float(area(T).sum()), 5) for w, T in v.items()})
