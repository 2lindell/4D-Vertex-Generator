"""The orbifolds of hyic [5,3,3] (14400), kihyic [5,3,3]+ (7200), ixitic 5[3+,4,3+] (1440) and ixoic 5[[3+,4,3+]]
(2880) on the 3-sphere: spheres, rings and points (structure_<group>.json), to check against the descriptions.

The groups are the generator page's (four_d_vertex_generator.library.named_symmetry: h4, h4+, ixitic, ixoic). Every
element's fixed subspace (a mirror sphere, a ring, or a line = two antipodal points), and the meets of two rings or of
a ring and a sphere, are collected; each keeps the elements fixing it pointwise (its stabilizer), and is a stratum when
that stabilizer is larger than that of every bigger subspace containing it. Counted: how many of each, their stabilizer
(order, with how many reflections), and the orbit of a generic point on it.

    python structure.py [h4 h4+ ixitic ixoic]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

HERE = os.path.dirname(os.path.abspath(__file__))
I = np.eye(4)


def basis(M, tol=1e-8):
    u, s, vt = np.linalg.svd(M)
    return vt[s < tol]


def key(B):
    P = B.T @ B
    return tuple(np.round(P, 6).ravel())


def analyse(name):
    E = np.array(group_elements(named_symmetry(name), 200_000))
    n = len(E)
    det = np.linalg.det(E)
    subs = {}
    for g in E:
        F = basis(g - I)
        if 0 < len(F) < 4:
            subs.setdefault(key(F), F)
    # (meets: rings / spheres pairwise, down to lines)
    for _ in range(2):
        L = [F for F in subs.values() if len(F) >= 2]
        P = [np.eye(4) - F.T @ F for F in L]
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                F = basis(np.vstack([P[i], P[j]]))
                if 0 < len(F) < min(len(L[i]), len(L[j])):
                    subs.setdefault(key(F), F)

    def stab(F):
        D = np.abs(np.einsum("nij,kj->nki", E, F) - F[None]).max(axis=(1, 2))
        return np.where(D < 1e-8)[0]
    info = {}
    for k, F in subs.items():
        s = stab(F)
        info[k] = (F, s)
    strata = []
    for k, (F, s) in info.items():
        # (a stratum: its stabilizer is not that of a bigger subspace containing it)
        bigger = [s2 for k2, (F2, s2) in info.items() if len(F2) > len(F) and np.abs(F @ (np.eye(4) - F2.T @ F2)).max() < 1e-8]
        if any(len(s2) == len(s) for s2 in bigger):
            continue
        strata.append((len(F), len(s), int(sum(det[s] < 0)), F))
    out = defaultdict(int)
    for dim, order, refl, F in strata:
        # (one class per orbit of the group, for the smaller groups: the least projector key over the images)
        oid = min(key(F @ g.T) for g in E) if n <= 3000 else None
        out[(dim, order, refl, oid)] += 1
    rows = []
    for (dim, order, refl, _), cnt in sorted(out.items(), key=lambda t: (-t[0][0], t[0][1], -t[1])):
        kind = {3: "spheres", 2: "rings", 1: "points"}[dim]
        count = cnt * (2 if dim == 1 else 1)              # (a fixed line meets the sphere in two points)
        rows.append({"kind": kind, "count": count, "stabilizer_order": order, "reflections": refl,
                     "generic_orbit": n // order})
    print(f"{name}: order {n}")
    for r in rows:
        print(f"  {r['count']:5} {r['kind']:7} stabilizer {r['stabilizer_order']:4} ({r['reflections']} reflections): "
              f"orbit {r['generic_orbit']}")
    json.dump({"group": name, "order": n, "strata": rows}, open(os.path.join(HERE, f"structure_{name.replace('+', '_plus')}.json"), "w"), indent=1)


if __name__ == "__main__":
    for nm in sys.argv[1:] or ["ixitic", "ixoic", "h4+", "h4"]:
        analyse(nm)
