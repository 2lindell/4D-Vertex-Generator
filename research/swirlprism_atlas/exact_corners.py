"""Curved walls (xsurfaces.json) made to keep to their exact extent: no part past the half-cell, every corner reached.

  - A mesh running past a face of the cell or the splitting mirror is cut back exactly along it: each triangle crossing
    is split where the face cuts it, the new points put on both the wall's surface and the face, the part beyond
    dropped.
  - A named point lying on the wall's surface where the mesh narrows to it (a wedge whose two edges converge on the
    point: a corner of the wall), but stops short, is joined to the mesh: a fan from the point to the mesh's boundary
    between the wedge's two edges, and the point put into the outline. A singular corner (a cone's apex, A1, C1) is
    where the mesher stops short; the fan reaches it.

Nothing is traced or classified: the faces and the named points are known.

    python exact_corners.py [X33,X40,...]        (all curved walls when none are given; rewrites xsurfaces.json)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

import region_cells as rc
import region_walls as rw
import xsurface
from edge_snap import Poly, newton

FLIP = np.array([1, 1, -1.0])
HALF = [np.eye(4)[a] for a in range(4)] + [np.array([0, 0, 1, -1.0])]
GAP = (2e-4, 0.03)          # (a corner this far short of the mesh, in the cell's xyz, is joined)
WEDGE = 150.0               # (the mesh's boundary near the point within this angle seen from it: a wedge)


def lin(h):
    return Poly([(k,) for k in range(4)], h, "face")


def clip_half(W, T):
    """Triangles (n, 3, 4 betas) cut back to the half-cell, new points on the surface and the face."""
    out, n_cut = [], 0
    for t in T:
        polys = [list(t)]
        for h in HALF:
            nxt = []
            for poly in polys:
                f = [float(h @ p) for p in poly]
                if min(f) >= -1e-12:
                    nxt.append(poly)
                    continue
                if max(f) <= 1e-12:
                    continue
                n_cut += 1
                q = []
                for k in range(len(poly)):
                    p1, p2, f1, f2 = poly[k], poly[(k + 1) % len(poly)], f[k], f[(k + 1) % len(poly)]
                    if f1 >= 0:
                        q.append(p1)
                    if (f1 >= 0) != (f2 >= 0):
                        x = p1 + (p2 - p1) * (f1 / (f1 - f2))
                        q.append(newton([W, lin(h)], x / x.sum()))
                if len(q) >= 3:
                    nxt.append(q)
            polys = nxt
        for poly in polys:
            out += [np.array([poly[0], poly[m], poly[m + 1]]) for m in range(1, len(poly) - 1)]
    return np.array(out), n_cut


def boundary_loops(T):
    key = lambda p: tuple(np.round(p, 10))
    pos, cnt, nbr = {}, defaultdict(int), defaultdict(list)
    for t in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pos[key(t[a])] = t[a]
            cnt[frozenset((key(t[a]), key(t[b])))] += 1
    for e, n in cnt.items():
        if n == 1 and len(e) == 2:
            a, b = tuple(e)
            nbr[a].append(b)
            nbr[b].append(a)
    return pos, nbr


def fan_corner(T, K):
    """A fan from K to the boundary between the wedge's two edges, if the mesh narrows to K; else None."""
    pos, nbr = boundary_loops(T)
    keys = list(nbr)
    if not keys:
        return None
    Pb = np.array([pos[k] for k in keys])
    X = rc.xyz(Pb / Pb.sum(1, keepdims=True))
    xk = rc.xyz((K / K.sum())[None])[0]
    d = np.linalg.norm(X - xk, axis=1)
    i0 = int(d.argmin())
    gap = d[i0]
    R = gap + 0.01
    # (the boundary walked both ways from its nearest vertex while within R)
    idx = {k: i for i, k in enumerate(keys)}
    chain = [keys[i0]]
    for side in (0, 1):
        prev, cur = None, keys[i0]
        while True:
            nx = [m for m in nbr[cur] if m != prev and m not in chain]
            if len(nbr[cur]) != 2 or not nx or d[idx[nx[0]]] > R:
                break
            prev, cur = cur, nx[0]
            if side == 0:
                chain.append(cur)
            else:
                chain.insert(0, cur)
            if len(chain) > 400:
                break
    if len(chain) < 3:
        return None
    U = X[[idx[k] for k in chain]] - xk
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(U[0] @ U[-1], -1, 1)))
    # (a wedge: the two ends of the boundary near K go off from it in directions less than WEDGE apart, and the
    # boundary between keeps on the same side)
    if spread > WEDGE or spread < 5:              # (5: a spike out and back, not a wedge)
        return None
    fan = [np.array([K / K.sum(), pos[chain[k]], pos[chain[k + 1]]]) for k in range(len(chain) - 1)]
    return fan, gap, spread, (chain[0], chain[-1])


def main(which=None):
    X = json.load(open("xsurfaces.json"))
    d = rw.page_data()
    pts = {}
    for key in ("special", "regular", "uniform"):
        for q in d.get(key, []):
            if isinstance(q, dict) and "q" in q and not q.get("copy"):
                pts.setdefault(q["id"], np.array(q["q"], float) * FLIP)
    names = list(pts)
    BP = rc.betas(np.array([pts[k] for k in names]))
    for tid, sf in X.items():
        if which and tid not in which:
            continue
        s0 = xsurface.surface(tid)
        W = Poly(s0["mons"], s0["coeffs"], tid)
        T = np.array(sf["tris"], float)
        T = T / T.sum(2, keepdims=True)
        T, n_cut = clip_half(W, T)
        notes = [f"{n_cut} triangles cut back to the half-cell"] if n_cut else []
        g = W.grad(BP)
        g = g - g.mean(1, keepdims=True)
        on = np.abs(W(BP)) / np.maximum(np.linalg.norm(g, axis=1), 1e-300) < 1e-9
        V = rc.xyz(T.reshape(-1, 4))
        for i in np.flatnonzero(on):
            K = BP[i]
            gap = np.linalg.norm(V - rc.xyz(K[None])[0], axis=1).min()
            if not GAP[0] < gap < GAP[1]:
                continue
            got = fan_corner(T, K)
            if got is None:
                continue
            fan, gap, spread, (a, b) = got
            T = np.concatenate([T, np.array(fan)])
            V = rc.xyz(T.reshape(-1, 4))
            sf.setdefault("filled_corners", []).append({"corner": K.tolist(), "point": names[i], "triangles": len(fan)})
            # (the corner into the outline, between the outline points nearest the wedge's two ends)
            for loop in sf.get("outline", []):
                B = np.array([q["beta"] for q in loop], float)
                B = B / B.sum(1, keepdims=True)
                ia = int(np.linalg.norm(B - np.array(a), axis=1).argmin())
                ib = int(np.linalg.norm(B - np.array(b), axis=1).argmin())
                if np.linalg.norm(B[ia] - np.array(a)) > 1e-6 or np.linalg.norm(B[ib] - np.array(b)) > 1e-6:
                    continue
                lo, hi = sorted((ia, ib))
                if hi - lo <= len(loop) - 1 - hi + lo:          # (the short way round, between them)
                    loop[lo + 1:hi] = [{"beta": K.tolist(), "beyond": loop[lo]["beyond"]}]
                else:
                    loop[hi + 1:] = [{"beta": K.tolist(), "beyond": loop[hi]["beyond"]}, dict(loop[lo])] if lo == 0 else loop[hi + 1:]
                break
            notes.append(f"corner {names[i]} joined ({gap:.4f} short, wedge {spread:.0f}°, {len(fan)} fan triangles)")
        if notes:
            sf["tris"] = T.tolist()
            print(f"{tid}: " + "; ".join(notes), flush=True)
    json.dump(X, open("xsurfaces.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
