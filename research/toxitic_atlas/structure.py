"""The orbifold of toxitic [3+,4,3+] (288) and oxitic [[3+,4,3+]] (576) on the 3-sphere: points, rings and areas
(structure_<sub>.json), to check against the descriptions.

Both groups are rotations. A ring is a great circle fixed pointwise by a cyclic subgroup (the elements whose fixed
plane it is); a point is a point of the sphere whose stabilizer is larger than any ring through it gives. For each:
how many, their stabilizer (order and kind), the orbit size of a generic point on it, and the faces of the chamber
(betas) where they lie.

    CONTIC_SUB=tox|ox python structure.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contic_atlas"))
import group as G  # noqa: E402

E = G.ELEMENTS
ORDER = len(E)
I = np.eye(4)


def fixed(g):
    u, s, vt = np.linalg.svd(g - I)
    return vt[s < 1e-9]


def plane_key(P):
    M = P.T @ P                                    # (projector onto the plane: a key independent of the basis)
    return tuple(np.round(M, 6).ravel())


def stab(x):
    return [g for g in E if np.abs(g @ x - x).max() < 1e-9]


def kind(H):
    """The rotation group H of the point: order and its element orders (as the 3D point group of the tangent space)."""
    o = Counter()
    for g in H:
        k = 1
        h = g.copy()
        while np.abs(h - I).max() > 1e-9:
            h = h @ g
            k += 1
        o[k] += 1
    n = len(H)
    name = {1: "C1", 2: "C2", 3: "C3", 4: "C4" if o[4] else "D2", 6: "D3" if o[2] == 3 else "C6",
            8: "D4" if o[2] == 5 else "C8", 12: "T" if o[3] == 8 else "D6", 24: "O" if o[4] else "D12"}.get(n, f"order {n}")
    return name, dict(sorted(o.items()))


# (rings)
rings = {}
for g in E:
    F = fixed(g)
    if len(F) == 2:
        rings.setdefault(plane_key(F), F)
ring_info = []
for k, F in rings.items():
    x = F[0] + 0.3719 * F[1]                    # (a generic point of the ring)
    x /= np.linalg.norm(x)
    H = stab(x)
    ring_info.append((len(H), kind(H)[0], F))
# (orbits of rings under the group)
def ring_orbit_id(F):
    return min(plane_key(F @ g.T) for g in E)
ring_classes = defaultdict(list)
for n, nm, F in ring_info:
    ring_classes[(n, nm, ring_orbit_id(F))].append(F)
# (points: where two rings cross, and their stabilizers)
pts = {}
keys = list(rings.values())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        M = np.vstack([keys[i], keys[j]])
        # (the common line of two planes: the null space of the complements' union)
        A = np.vstack([I - keys[i].T @ keys[i], I - keys[j].T @ keys[j]])
        u, s, vt = np.linalg.svd(A)
        if s[-1] < 1e-9 and s[-2] > 1e-9:
            x = vt[-1]
            for sg in (1, -1):
                y = sg * x
                pts.setdefault(tuple(np.round(y, 7)), y)
point_classes = defaultdict(list)
for y in pts.values():
    H = stab(y)
    oid = min(tuple(np.round(g @ y, 6)) for g in E)
    point_classes[(len(H), kind(H)[0], oid)].append(y)


# (the chamber's faces: where each ring class meets the chamber (betas, F4 frame))
FACES = {"b1 = b2 = 0": [0, 0, .37, .63], "b3 = b4 = 0": [.63, .37, 0, 0], "b2 = b3 = 0": [.37, 0, 0, .63],
         "b1 = b4, b2 = b3": [.31, .19, .19, .31], "b1 = b2 = b3 = 0": [0, 0, 0, 1], "b2 = b3 = b4 = 0": [1, 0, 0, 0],
         "b2 = b3 = 0, b1 = b4": [.5, 0, 0, .5]}


def where(x):
    x = x / np.linalg.norm(x)
    for nm, b in FACES.items():
        y = G.x4(b)
        y = y / np.linalg.norm(y)
        if any(np.abs(g @ y - x).max() < 1e-7 for g in G._elements()):
            return nm
    return None


def ring_where(F):
    for nm, b in FACES.items():
        y = G.x4(b)
        y = y / np.linalg.norm(y)
        for g in G._elements():
            z = g @ y
            if np.linalg.norm(F @ z) > 1 - 1e-9 and len(stab(z)) == len(stab(F[0] + 0.3719 * F[1])):
                return nm
    return None


out = {"group": G.SUB, "order": ORDER, "rings": [], "points": []}
print(f"{G.SUB}: order {ORDER}")
for (n, nm, _), L in sorted(ring_classes.items(), key=lambda t: (-t[0][0], -len(t[1]))):
    w = ring_where(L[0])
    out["rings"].append({"stabilizer": nm, "order": n, "rings": len(L), "generic_orbit": ORDER // n, "chamber": w})
    print(f"  {len(L):3} rings, stabilizer {nm} ({n}): a generic point's orbit {ORDER // n}; in the chamber {w}")
for (n, nm, _), L in sorted(point_classes.items(), key=lambda t: -t[0][0]):
    rs = sum(1 for F in rings.values() if np.linalg.norm(F @ L[0]) > 1 - 1e-9)
    w = where(L[0])
    out["points"].append({"stabilizer": nm, "order": n, "points": len(L), "rings_through": rs, "chamber": w})
    print(f"  {len(L):3} points, stabilizer {nm} ({n}), {rs} rings through each; in the chamber {w}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"structure_{G.SUB}.json"), "w"), indent=1)
