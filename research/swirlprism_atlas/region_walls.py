"""A region's outer walls, coloured on the page with crisp edges (region_walls.json).

The region's walls are given (checked by hand: the walls the region really borders, not the walls inside it or past
its other walls). The half-cell is cut into voxels by those walls alone, every drawn piece of them at the page's own
resolution (the curved walls' copies made as the page makes them, wall_copies.py); the pockets holding the region's
samples, and no other region's, are its cell. A wall triangle with the cell on one side is coloured; a triangle where
that changes is cut along the other wall crossing it, so the colour stops on the exact line where the walls meet.

    python region_walls.py F2 T2,X3,X9,X10,X11,X14,X21      (reads ../../assets/symmetry_domain.html)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import region_cells as rc
from wall_copies import copies

FLIP = np.array([1, 1, -1.0])


def page_data(path="../../assets/symmetry_domain.html"):
    t = open(path).read()
    return json.loads(re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', t, re.S).group(1))


def pieces_of(d, walls):
    """(wall, kind, index, triangles in cell xyz) for every drawn piece of the walls."""
    out = []
    for wi, w in enumerate(d["xwalls"]):
        if w["id"] not in walls:
            continue
        V = np.array(w["verts"], float) * FLIP
        out.append((w["id"], "orig", wi, V[np.array(w["faces"])]))
        if w.get("copyOps"):
            C = copies(w, d)
            if C:
                out.append((w["id"], "copy", wi, rc.xyz(np.array(C).reshape(-1, 4)).reshape(-1, 3, 3)))
    if "T2" in walls:
        for pi, p in enumerate(d["tpatches"]):
            Q = np.array(p["pts"][:-1], float) * FLIP
            out.append(("T2", "tpatch", pi, np.array([[Q[0], Q[i], Q[i + 1]] for i in range(1, len(Q) - 1)])))
    return out


def cell(d, rid, pieces, h=0.0025):
    lo, hi = rc.CORNERS.min(axis=0) - 3 * h, rc.CORNERS.max(axis=0) + 3 * h
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    vox = lambda X: np.clip(np.floor((np.asarray(X) - lo) / h).astype(int), 0, np.array(shape) - 1)
    wall = np.zeros(shape, bool)
    for _, _, _, T in pieces:
        for t in T:
            e = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]), np.linalg.norm(t[0] - t[2]))
            n = max(1, int(np.ceil(e / (0.4 * h))))
            a_, b_ = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            k = a_ + b_ <= n
            a, b = a_[k] / n, b_[k] / n
            idx = vox(t[0] + a[:, None] * (t[1] - t[0]) + b[:, None] * (t[2] - t[0]))
            wall[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = np.stack(np.meshgrid(*[lo[k] + h * (np.arange(shape[k]) + 0.5) for k in range(3)], indexing="ij"), -1)
    Bg = rc.betas(g.reshape(-1, 3)).reshape(shape + (4,))
    lab, _ = ndimage.label(rc.inside(Bg) & ~wall)
    dims = {t["id"]: t["dim"] for t in d["types"]}
    per = defaultdict(Counter)
    for t, pts in d["samples"].items():
        if dims.get(t) != "region":
            continue
        for p in pts:
            v = lab[tuple(vox(np.array(p[:3]) * FLIP))]
            if v > 0:
                per[int(v)][t] += 1
    ins = sorted(p for p, c in per.items() if c[rid] and len(c) == 1)
    mixed = {p: dict(c) for p, c in per.items() if c[rid] and len(c) > 1}
    print(f"{rid}: pockets {ins}, mixed {mixed}", flush=True)
    return lab, lo, h, set(ins)


def paint(pieces, lab, lo, h, ins):
    shape = np.array(lab.shape)

    def side(X):
        i = np.floor((X - lo) / h).astype(int)
        ok = np.all((i >= 0) & (i < shape), axis=1)
        v = np.full(len(X), -1)
        v[ok] = lab[tuple(i[ok].T)]
        inh = rc.inside(rc.betas(X))
        out = np.where(inh & (v > 0), np.where(np.isin(v, list(ins)), 1, 2), 0)
        out[inh & (v <= 0)] = -1
        return out

    def sides(C, N):
        res = []
        for s in (1, -1):
            r = np.full(len(C), -1)
            for k in (1.5, 2.5, 3.5, 4.5):
                q = side(C + s * k * h * N)
                m = (r == -1) & (q != -1)
                r[m] = q[m]
            res.append(r)
        return np.stack(res, 1)

    coloured = lambda st: (st[0] == 1) != (st[1] == 1) and -1 not in st

    def clip(poly, p0, n):
        out = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            fa, fb = (a - p0) @ n, (b - p0) @ n
            if fa >= 0:
                out.append(a)
            if (fa >= 0) != (fb >= 0):
                out.append(a + (b - a) * fa / (fa - fb))
        return out
    allT = np.concatenate([T for *_, T in pieces])
    owner = np.concatenate([[k] * len(T) for k, (*_, T) in enumerate(pieces)])
    rad = np.max(np.linalg.norm(allT - allT.mean(axis=1)[:, None], axis=2), axis=1)
    small = rad < 0.03
    tree, sidx, big = cKDTree(allT[small].mean(axis=1)), np.flatnonzero(small), np.flatnonzero(~small)
    out, count = [], Counter()
    for k, (wid, kind, idx, T) in enumerate(pieces):
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ln = np.linalg.norm(N, axis=1)
        N = N / np.maximum(ln, 1e-300)[:, None]
        st = np.array([coloured(s) for s in sides(T.mean(1), N)]) & (ln > 1e-15)
        key = lambda p: tuple(np.round(p, 6))
        E = defaultdict(list)
        for j, t in enumerate(T):
            for a in range(3):
                E[tuple(sorted((key(t[a]), key(t[(a + 1) % 3]))))].append(j)
        front = {j for js in E.values() if len(js) == 2 and st[js[0]] != st[js[1]] for j in js}
        for j, t in enumerate(T):
            if j not in front:
                if st[j]:
                    out.append(t)
                    count[wid] += 1
                continue
            rt = np.max(np.linalg.norm(t - t.mean(0), axis=1))
            cand = [c for c in list(sidx[tree.query_ball_point(t.mean(0), rt + 0.03)]) + list(big) if owner[c] != k]
            polys = [list(t)]
            for c in cand:                         # (cut by every other wall's triangle crossing this one)
                u = allT[c]
                nu = np.cross(u[1] - u[0], u[2] - u[0])
                if np.linalg.norm(nu) < 1e-15:
                    continue
                fu, ft = (t - u[0]) @ nu, (u - t[0]) @ N[j]
                if fu.max() <= 0 or fu.min() >= 0 or ft.max() <= 0 or ft.min() >= 0:
                    continue
                polys = [q for pg in polys for q in (clip(pg, u[0], nu), clip(pg, u[0], -nu)) if len(q) >= 3]
                if len(polys) > 6:
                    break
            for pg in polys:
                pg = np.array(pg)
                if coloured(sides(pg.mean(0)[None], N[j][None])[0]):
                    out += [np.array([pg[0], pg[m], pg[m + 1]]) for m in range(1, len(pg) - 1)]
                    count[wid] += 1
    print(f"  coloured pieces: {dict(count)}", flush=True)
    return np.array(out)


def main(rid, walls):
    walls = set(walls.split(","))
    d = page_data()
    pieces = pieces_of(d, walls)
    lab, lo, h, ins = cell(d, rid, pieces)
    tris = paint(pieces, lab, lo, h, ins)
    m = np.isin(lab, list(ins)).ravel()
    dd = np.diff(np.r_[0, m.astype(np.int8), 0])
    runs = [[int(a), int(b - a)] for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1))]
    P = (tris.reshape(-1, 3) * FLIP).round(6)          # (page frame)
    try:
        allr = json.load(open("region_walls.json"))
    except FileNotFoundError:
        allr = {}
    allr[rid] = {"walls": sorted(walls), "pockets": len(ins), "tris": P.reshape(-1, 9).tolist(),
                 "grid": {"lo": lo.tolist(), "h": h, "shape": list(lab.shape), "runs": runs}}
    json.dump(allr, open("region_walls.json", "w"), separators=(",", ":"))
    print(f"{rid}: {len(tris)} triangles", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
