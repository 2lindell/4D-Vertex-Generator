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


def outline_to(sf, K, a, b, tol=0.003):
    """The corner K put into the wall's outline: the outline's run between the points nearest the wedge's two ends
    (the short way round) replaced by those ends and K."""
    K = K / K.sum()
    for loop in sf.get("outline", []):
        closed = len(loop) > 1 and np.allclose(loop[0]["beta"], loop[-1]["beta"])
        L = loop[:-1] if closed else loop
        B = np.array([q["beta"] for q in L], float)
        B = B / B.sum(1, keepdims=True)
        if np.linalg.norm(B - K, axis=1).min() < 1e-9:
            return True                               # (already there)
        da, db = np.linalg.norm(B - a / a.sum(), axis=1), np.linalg.norm(B - b / b.sum(), axis=1)
        ia, ib = int(da.argmin()), int(db.argmin())
        if da[ia] > tol or db[ib] > tol or ia == ib:
            continue
        n = len(L)
        fwd = (ib - ia) % n                           # (steps from ia to ib going forward)
        if fwd > n - fwd:
            ia, ib, fwd = ib, ia, n - fwd
        new = {"beta": K.tolist(), "beyond": L[ia]["beyond"]}
        idx = [(ia + k) % n for k in range(1, fwd)]   # (the run strictly between: replaced by K)
        keep = [L[k] for k in range(n) if k not in set(idx)]
        pos = keep.index(L[ia]) + 1
        keep.insert(pos, new)
        loop[:] = keep + ([keep[0]] if closed else [])
        return True
    return False


def snap_tip(sf, T, K, reach=0.01, straight=1.0):
    """A mesh ending in a tip just short of K along a straight edge pointing at K (the mesher stops short at a
    singular point): the tip vertex moved onto K, in the triangles and the outline. None if not so."""
    pos, nbr = boundary_loops(T)
    keys = list(nbr)
    if not keys:
        return None
    xk = rc.xyz((K / K.sum())[None])[0]
    Xb = {k: rc.xyz(np.array([pos[k]]))[0] for k in keys}
    k0 = min(keys, key=lambda k: np.linalg.norm(Xb[k] - xk))
    gap = float(np.linalg.norm(Xb[k0] - xk))
    if not 1e-9 < gap < reach or len(nbr[k0]) != 2:
        return None
    u = (Xb[k0] - xk) / gap
    for s_ in nbr[k0]:
        w = (Xb[s_] - xk) / np.linalg.norm(Xb[s_] - xk)
        if np.degrees(np.arccos(np.clip(u @ w, -1, 1))) < straight:
            break                                     # (an edge running straight on through the tip to K)
    else:
        return None
    old = pos[k0].copy()                         # (pos holds views into T)
    T[np.linalg.norm(T - old, axis=2) < 1e-9] = K / K.sum()
    xo = rc.xyz(old[None])[0]
    for loop in sf.get("outline", []):           # (the outline's own point at the tip: the nearest, within 2e-4)
        B = np.array([q["beta"] for q in loop], float)
        dd = np.linalg.norm(rc.xyz(B / B.sum(1, keepdims=True)) - xo, axis=1)
        for j in np.flatnonzero(dd < 2e-4):
            loop[j]["beta"] = (K / K.sum()).tolist()
    return gap


def ends_of_fan(T, K):
    """The outer ends of the fan at K: the vertices of K's triangles used by only one of them."""
    K = K / K.sum()
    at = [t for t in T if np.linalg.norm(t - K, axis=1).min() < 1e-9]
    from collections import Counter
    c = Counter(tuple(np.round(v, 10)) for t in at for v in t if np.linalg.norm(v - K) > 1e-9)
    ends = [np.array(k) for k, n in c.items() if n == 1]
    return ends if len(ends) == 2 else None


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
            tip = snap_tip(sf, T, K)
            if tip is not None:
                V = rc.xyz(T.reshape(-1, 4))
                sf.setdefault("filled_corners", []).append({"corner": K.tolist(), "point": names[i], "moved_tip": tip})
                notes.append(f"corner {names[i]}: the tip {tip:.4f} short of it moved onto it")
                continue
            got = fan_corner(T, K)
            if got is None:
                continue
            fan, gap, spread, (a, b) = got
            T = np.concatenate([T, np.array(fan)])
            V = rc.xyz(T.reshape(-1, 4))
            sf.setdefault("filled_corners", []).append({"corner": K.tolist(), "point": names[i], "triangles": len(fan)})
            outline_to(sf, K, np.array(a), np.array(b))
            notes.append(f"corner {names[i]} joined ({gap:.4f} short, wedge {spread:.0f}°, {len(fan)} fan triangles)")
        if notes:
            sf["tris"] = T.tolist()
            print(f"{tid}: " + "; ".join(notes), flush=True)
    json.dump(X, open("xsurfaces.json", "w"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--outlines":
        # (outlines of corners already joined, brought round them)
        X = json.load(open("xsurfaces.json"))
        for tid, sf in X.items():
            T = np.array(sf["tris"], float)
            T = T / T.sum(2, keepdims=True)
            for fc in sf.get("filled_corners", []):
                if "point" not in fc:
                    continue
                K = np.array(fc["corner"], float)
                e = ends_of_fan(T, K)
                print(tid, fc["point"], "outline brought round" if e and outline_to(sf, K, *e) else "not matched")
        json.dump(X, open("xsurfaces.json", "w"))
    else:
        main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
