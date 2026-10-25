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
import os
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


def polygon_tris(Q):
    """A flat polygon (its corners in order) in triangles that stay inside it: a patch with a curved side is not
    convex, and a fan from one corner would reach past that side."""
    import shapely
    from shapely.geometry import Polygon
    o = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - o)
    e1, e2 = vt[0], vt[1]
    P = Polygon(np.c_[(Q - o) @ e1, (Q - o) @ e2])
    tris = []
    for t in shapely.constrained_delaunay_triangles(P).geoms:
        c = np.array(t.exterior.coords)[:3]
        tris.append(o + c[:, :1] * e1 + c[:, 1:2] * e2)
    return np.array(tris)


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
            out.append(("T2", "tpatch", pi, polygon_tris(Q)))
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


def cell_sides(lab, lo, h, ins):
    """sides(C, N) -> (n, 2): 1 in the cell, 2 elsewhere, 0 outside the half-cell, -1 not found, a few voxels off
    each side of the points C along N (cell xyz)."""
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
    return sides


def _clip(poly, p0, n):
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        fa, fb = (a - p0) @ n, (b - p0) @ n
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            out.append(a + (b - a) * fa / (fa - fb))
    return out


def _crosses(t, u, eps=3e-6):
    """Does triangle u cross triangle t's plane through t, or end on it (a wall ending on another touches it along
    its edge without crossing: the page's coordinates are kept to 1e-6)?"""
    n = np.cross(t[1] - t[0], t[2] - t[0])
    nu = np.cross(u[1] - u[0], u[2] - u[0])
    if np.linalg.norm(n) < 1e-15 or np.linalg.norm(nu) < 1e-15:
        return False
    n, nu = n / np.linalg.norm(n), nu / np.linalg.norm(nu)
    fu, ft = (t - u[0]) @ nu, (u - t[0]) @ n
    if abs(n @ nu) > 1 - 1e-9:                          # (parallel: no line to cut along)
        return False
    return fu.max() > eps and fu.min() < -eps and ft.max() >= -eps and ft.min() <= eps


def classified_sides(rid, cache_path="region_walls_cache.jsonl", step=0.0004, procs=4):
    """sides for a list of points by the classifier itself (cached): 1 where the shape a step off is the region,
    2 another shape, 0 outside the cell. Slow (seconds a point), so only for the few pieces of the flat walls."""
    import os
    from multiprocessing import Pool
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            k, v = json.loads(line)
            cache[k] = v
    key = lambda b: ",".join(f"{x:.10f}" for x in b / b.sum())

    def sides(C, N):
        B = [rc.betas((C + s * step * N))[:, None, :] for s in (1, -1)]
        B = np.concatenate(B, axis=1)                     # (n, 2, 4)
        todo = sorted({key(b) for b in B.reshape(-1, 4) if b.min() > 0 and b[2] >= b[3]} - set(cache))
        if todo:
            with Pool(procs) as pool:
                got = pool.map(rc._lab_job, [np.array([float(x) for x in k.split(",")]) for k in todo], chunksize=1)
            with open(cache_path, "a") as f:
                for k, l in zip(todo, got):
                    cache[k] = l
                    f.write(json.dumps([k, l]) + "\n")
        out = np.zeros(B.shape[:2], int)
        for i in range(len(B)):
            for j in range(2):
                b = B[i, j]
                # (past a face of the cell or the splitting mirror is outside the drawn half: what lies there is
                # drawn beside the copy, not here)
                out[i, j] = 0 if (b.min() <= 0 or b[2] < b[3]) else (1 if cache[key(b)] == rid else 2)
        return out
    return sides


def paint(pieces, sides, reach=0.03, max_parts=64, exact_sides=None, exact_walls=(), min_exact_area=1.5e-5, judge=None, near_face_reach=0.02):
    """The coloured triangles of the pieces [(wall, kind, index, triangles)]: every triangle another of the walls
    crosses is cut along that wall's triangles first, and each part is kept when the cell is on one side of it.
    The pieces of exact_walls (flat walls of a few large triangles) are judged by exact_sides instead."""
    coloured = lambda st: (st[0] == 1) != (st[1] == 1) and -1 not in st
    allT = np.concatenate([T for *_, T in pieces])
    wall_of = np.concatenate([[w] * len(T) for w, *_, T in pieces])
    rad = np.max(np.linalg.norm(allT - allT.mean(axis=1)[:, None], axis=2), axis=1)
    small = rad < reach
    tree, sidx, big = cKDTree(allT[small].mean(axis=1)), np.flatnonzero(small), np.flatnonzero(~small)
    parts = []                                          # (wall, polygon, normal)
    for wid, kind, idx, T in (pieces if judge is None else judge):
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        ln = np.linalg.norm(N, axis=1)
        N = N / np.maximum(ln, 1e-300)[:, None]
        for j, t in enumerate(T):
            if ln[j] < 1e-15:
                continue
            rt = np.max(np.linalg.norm(t - t.mean(0), axis=1))
            cand = [c for c in list(sidx[tree.query_ball_point(t.mean(0), rt + reach)]) + list(big)
                    if wall_of[c] != wid and _crosses(t, allT[c])]
            polys = [list(t)]
            for c in cand:                     # (cut along every other wall crossing it)
                u = allT[c]
                nu = np.cross(u[1] - u[0], u[2] - u[0])
                polys = [q for pg in polys for q in (_clip(pg, u[0], nu), _clip(pg, u[0], -nu)) if len(q) >= 3]
                if len(polys) > max_parts:
                    break
            for pg in polys:
                pg = np.array(pg)
                # (a sliver too thin to step off from is judged with its neighbours: skipped)
                if np.linalg.norm(sum(np.cross(pg[m] - pg[0], pg[m + 1] - pg[0]) for m in range(1, len(pg) - 1))) < 1e-12:
                    continue
                parts.append((wid, pg, N[j], kind))
    out, ids, kinds, count = [], [], [], Counter()
    area = lambda pg: 0.5 * np.linalg.norm(sum(np.cross(pg[m] - pg[0], pg[m + 1] - pg[0]) for m in range(1, len(pg) - 1)))
    def near_face(pg, r=near_face_reach):
        """Within r of a face of the cell or the splitting mirror (in betas): the voxels there are unreliable."""
        B = rc.betas(np.asarray(pg))
        return (B.min() < r) or ((B[:, 2] - B[:, 3]).min() < r)
    use_exact = [p[0] in exact_walls and area(p[1]) > min_exact_area for p in parts]
    if exact_sides is not None:                        # (near a face, where the voxel test gives up: the classifier)
        idx_ = [i for i, p in enumerate(parts) if not use_exact[i] and near_face(p[1])]
        if idx_:
            vs = sides(np.array([parts[i][1].mean(0) for i in idx_]), np.array([parts[i][2] for i in idx_]))
            for i, v in zip(idx_, vs):
                if -1 in v or (0 in v and 1 not in v):
                    use_exact[i] = True
    print(f"  {len(parts)} parts, {sum(use_exact)} judged by the classifier", flush=True)
    if os.environ.get("DRY"):
        return np.zeros((0, 3, 3))
    for ex in (False, True):
        sel = [p for p, e in zip(parts, use_exact) if e == ex]
        if not sel:
            continue
        fn = exact_sides if ex else sides
        st = fn(np.array([pg.mean(0) for _, pg, _, _ in sel]), np.array([n for _, _, n, _ in sel]))
        for (wid, pg, _, kd), s in zip(sel, st):
            if coloured(s):
                out += [np.array([pg[0], pg[m], pg[m + 1]]) for m in range(1, len(pg) - 1)]
                ids += [wid] * (len(pg) - 2)
                kinds += [kd] * (len(pg) - 2)
                count[wid] += 1
    print(f"  coloured pieces: {dict(count)}", flush=True)
    paint.ids, paint.kinds = ids, kinds
    return np.array(out)


def plane_of(T):
    """Centre and unit normal of the plane best through the triangles' corners."""
    V = T.reshape(-1, 3)
    c = V.mean(0)
    return c, np.linalg.svd(V - c)[2][-1]


def is_flat(T, tol=1e-5):
    c, n = plane_of(T)
    return np.abs((T.reshape(-1, 3) - c) @ n).max() < tol


def split_flat(wid, T, others, eps=3e-6, inset=0.0015, extend=1.5e-3):
    """A flat piece (triangles T of wall wid) split, in its own plane, along the traces the other walls leave on it
    (others: [(wall, triangles)]; where they cross it, end on it or lie in its plane). Returns its faces as
    (polygon in the plane, (origin, e1, e2), normal, a point well inside, too thin to step off from)."""
    from shapely.geometry import LineString, MultiLineString, Polygon
    from shapely.ops import unary_union
    out = []
    o, n = plane_of(T)
    e1 = np.linalg.svd(T.reshape(-1, 3) - o)[2][0]
    e2 = np.cross(n, e1)
    to2 = lambda X: np.c_[(X - o) @ e1, (X - o) @ e2]
    patch = unary_union([Polygon(to2(t)) for t in T if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-14])
    lo_, hi_ = T.reshape(-1, 3).min(0) - 1e-3, T.reshape(-1, 3).max(0) + 1e-3
    segs = []
    for w2, U in others:
        if w2 == wid:
            continue
        m = np.all((U.max(1) >= lo_) & (U.min(1) <= hi_), axis=1)
        for u in U[m]:
            f = (u - o) @ n
            on = np.abs(f) < eps
            if on.sum() == 3:                       # (a wall in the same plane: its edges split this one)
                segs += [to2(u[[a_, (a_ + 1) % 3]]) for a_ in range(3)]
            elif on.sum() == 2:                     # (an edge lying on the plane: a wall ending here)
                segs.append(to2(u[on]))
            elif f.max() > eps and f.min() < -eps:  # (crossing)
                pp = []
                for a_ in range(3):
                    b_ = (a_ + 1) % 3
                    if (f[a_] > 0) != (f[b_] > 0) and abs(f[a_] - f[b_]) > 0:
                        pp.append(u[a_] + f[a_] / (f[a_] - f[b_]) * (u[b_] - u[a_]))
                    elif on[a_]:
                        pp.append(u[a_])
                if len(pp) >= 2:
                    segs.append(to2(np.array(pp[:2])))
    segs = [s_ for s_ in segs if np.linalg.norm(s_[1] - s_[0]) > 1e-10]
    # (traces of different walls meet only to the meshes' precision: each segment is lengthened a little at both
    # ends so they join and close off faces; an overshoot only splits a face whose parts are judged alike)
    segs = [np.array([a_ - (b_ - a_) / np.linalg.norm(b_ - a_) * extend, b_ + (b_ - a_) / np.linalg.norm(b_ - a_) * extend])
            for a_, b_ in segs]
    cut = unary_union(MultiLineString([LineString(s_) for s_ in segs]).buffer(1e-7)) if segs else None
    parts = patch.difference(cut) if cut is not None else patch
    geoms = list(parts.geoms) if hasattr(parts, "geoms") else [parts]
    frame = (o, e1, e2)
    for g in geoms:
        if g.area < 1e-12:
            continue
        core = g.buffer(-inset)
        pt = (core if not core.is_empty else g).representative_point()
        P3 = o + pt.x * e1 + pt.y * e2
        out.append((g, frame, n, P3, core.is_empty))
    return out


def flat_paint(pieces, exact_sides, voxel_sides, eps=3e-6, inset=0.0015, extend=1.5e-3):
    """The coloured faces of the flat pieces: each piece, in its own plane, is split along the traces the region's
    other walls leave on it (where they cross it, and where they end on it), and every face is judged once by the
    classifier at a point well inside it. Returns (triangles, wall ids)."""
    import shapely
    from shapely.geometry import LineString, MultiLineString, Polygon
    from shapely.ops import unary_union
    flat = [p for p in pieces if is_flat(p[3])]
    # (a piece lying in the splitting mirror is not coloured: the mirror folds the drawn half, and what lies past it
    # is drawn beside the wall's copy)
    in_mirror = lambda T: np.abs(np.diff(rc.betas(T.reshape(-1, 3))[:, 2:], axis=1)).max() < 1e-6
    colourable = {id(p[3]) for p in flat if not in_mirror(p[3])}
    others = [(w, T) for w, _, _, T in pieces]
    faces = []                                         # (wall, 3D polygon list, normal, inner point)
    for wid, kind, idx, T in [p for p in flat if id(p[3]) in colourable]:
        for g, frame, n, P3, thin in split_flat(wid, T, others, eps, inset, extend):
            faces.append((wid, g, frame, n, P3, thin, kind))
    print(f"  flat walls: {len(faces)} faces", flush=True)
    st_exact = exact_sides(np.array([f[4] for f in faces]), np.array([f[3] for f in faces])) if faces else []
    st_vox = voxel_sides(np.array([f[4] for f in faces]), np.array([f[3] for f in faces])) if faces else []
    coloured = lambda st: (st[0] == 1) != (st[1] == 1) and -1 not in st
    out, ids, kinds, cnt = [], [], [], Counter()
    keep = defaultdict(list)                           # (the coloured faces of each piece, merged before filling)
    for i, ((wid, g, frame, n, P3, thin, kind), se, sv) in enumerate(zip(faces, st_exact, st_vox)):
        if coloured(sv if thin else se):
            cnt[wid] += 1
            keep[(wid, kind, id(frame))].append((g, frame))
    for (wid, kind, _), gs in keep.items():
        o, e1, e2 = gs[0][1]
        # (merged across the hair-thin cuts, then rid of the cut ends' extra vertices)
        m = unary_union([g.buffer(2e-7) for g, _ in gs]).buffer(-2e-7).simplify(1e-7, preserve_topology=True)
        for t in [t for part in (m.geoms if hasattr(m, "geoms") else [m]) if part.area > 0
                  for t in shapely.constrained_delaunay_triangles(part).geoms]:
            c = np.array(t.exterior.coords)[:3]
            out.append(o + c[:, :1] * e1 + c[:, 1:2] * e2)
            ids.append(wid)
            kinds.append(kind)
    print(f"  flat walls coloured faces: {dict(cnt)}", flush=True)
    flat_paint.kinds = kinds
    return out, ids


def main(rid, walls):
    walls = set(walls.split(","))
    d = page_data()
    pieces = pieces_of(d, walls)
    lab, lo, h, ins = cell(d, rid, pieces)
    vox = cell_sides(lab, lo, h, ins)
    ex = classified_sides(rid)
    curved = [p for p in pieces if not is_flat(p[3])]
    # (the curved walls, finely meshed: their triangles cut where other walls cross them, judged by the voxels;
    # the flat walls, a few large triangles: split along the other walls' traces, each face judged by the classifier)
    tris = paint(pieces, vox, judge=[p for p in curved], exact_sides=ex)
    ids = list(paint.ids)
    kinds0 = list(paint.kinds)
    ft, fids = flat_paint(pieces, ex, vox)
    tris = np.concatenate([tris.reshape(-1, 3, 3), np.array(ft).reshape(-1, 3, 3)])
    paint.ids = ids + fids
    paint.kinds = kinds0 + flat_paint.kinds
    m = np.isin(lab, list(ins)).ravel()
    dd = np.diff(np.r_[0, m.astype(np.int8), 0])
    runs = [[int(a), int(b - a)] for a, b in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1))]
    P = (tris.reshape(-1, 3) * FLIP).round(6)          # (page frame)
    try:
        allr = json.load(open("region_walls.json"))
    except FileNotFoundError:
        allr = {}
    allr[rid] = {"walls": sorted(walls), "pockets": len(ins), "tris": P.reshape(-1, 9).tolist(), "ids": paint.ids, "kinds": paint.kinds,
                 "grid": {"lo": lo.tolist(), "h": h, "shape": list(lab.shape), "runs": runs}}
    json.dump(allr, open("region_walls.json", "w"), separators=(",", ":"))
    print(f"{rid}: {len(tris)} triangles", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
