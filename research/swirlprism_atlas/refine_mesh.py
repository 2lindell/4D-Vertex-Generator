"""A curved wall's mesh (xsurfaces.json) with its overlong triangles split: every edge longer than hmax (in betas)
gets its midpoint, put on the wall's exact surface, and each triangle is split along the edges of it that were split
(two, three or four triangles), so neighbours keep sharing their edges. Repeated until no edge is longer. A corner
fan reaching far across a curved wall is a flat chord off the surface (it shades wrongly) and too long for the page's
copies, which drop such a triangle and leave a hole; split, it follows the surface.

    python refine_mesh.py X22 [hmax]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import xsurface
from edge_snap import Poly, newton


def main(tid, hmax=0.02):
    X = json.load(open("xsurfaces.json"))
    sf = X[tid]
    s = xsurface.surface(tid)
    W = Poly(s["mons"], s["coeffs"], tid)
    T = np.array(sf["tris"], float)
    T = T / T.sum(2, keepdims=True)
    key = lambda p: tuple(np.round(p, 11))
    idx, V = {}, []
    F = []
    for t in T:
        f = []
        for p in t:
            k = key(p)
            if k not in idx:
                idx[k] = len(V)
                V.append(p)
            f.append(idx[k])
        F.append(f)
    V = list(V)
    n0 = len(F)
    count = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            count[frozenset((a, b))] = count.get(frozenset((a, b)), 0) + 1
    # the outline (exact edge): a boundary edge's midpoint goes onto it, not just onto the surface
    OL = [np.array([q["beta"] for q in l], float) for l in sf.get("outline", [])]
    SA = np.concatenate([L[:-1] for L in OL]) if OL else None
    SB = np.concatenate([L[1:] for L in OL]) if OL else None

    def on_outline(p):
        d = SB - SA
        t = np.clip(((p - SA) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
        Q = SA + t[:, None] * d
        i = np.argmin(np.linalg.norm(Q - p, axis=1))
        return Q[i], float(np.linalg.norm(Q[i] - p))

    for _ in range(12):
        mid = {}
        for f in F:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                e = frozenset((a, b))
                if e not in mid and np.linalg.norm(V[a] - V[b]) > hmax:
                    m = (V[a] + V[b]) / 2
                    if count.get(e, 2) == 1 and SA is not None:
                        q, dq = on_outline(m)
                        m = q if dq < 0.25 * np.linalg.norm(V[a] - V[b]) else newton([W], m)
                    else:
                        m = newton([W], m)
                    mid[e] = len(V)
                    V.append(m / m.sum())
        if not mid:
            break
        out = []
        for a, b, c in F:
            ab, bc, ca = mid.get(frozenset((a, b))), mid.get(frozenset((b, c))), mid.get(frozenset((c, a)))
            k = sum(x is not None for x in (ab, bc, ca))
            if k == 0:
                out.append([a, b, c])
            elif k == 3:
                out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
            else:
                # (rotated so the first split edge is a-b; one or two splits)
                for _r in range(3):
                    if ab is not None and (k == 1 or bc is not None):
                        break
                    a, b, c = b, c, a
                    ab, bc, ca = bc, ca, ab
                if k == 1:
                    out += [[a, ab, c], [ab, b, c]]
                else:
                    out += [[a, ab, c], [ab, b, bc], [ab, bc, c]]
        for e, m in mid.items():
            if count.get(e, 2) == 1:
                a, b = tuple(e)
                count[frozenset((a, m))] = count[frozenset((m, b))] = 1
        F = out
    V = np.array(V)
    # back onto the mirror β3 = β4 and the faces β = 0 where a projection stepped a hair past them
    d = V[:, 2] - V[:, 3]
    near = (d < 0) & (d > -1e-6)
    V[near, 2] = V[near, 3] = (V[near, 2] + V[near, 3]) / 2
    V[(V < 0) & (V > -1e-9)] = 0
    res = np.abs(W(V)).max()
    sf["tris"] = V[np.array(F)].tolist()
    sf.setdefault("refined", []).append({"hmax": hmax, "triangles_before": n0, "after": len(F)})
    json.dump(X, open("xsurfaces.json", "w"))
    L = max(np.linalg.norm(V[f[i]] - V[f[(i + 1) % 3]]) for f in F for i in range(3))
    print(f"{tid}: {n0} -> {len(F)} triangles, longest edge now {L:.4f}, surface residual {res:.1e}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.02)
