"""The regions as cells of the half tetrahedron cut by every wall, joined up on the 3-sphere, and drawn with the walls'
own meshes.

1. The half-cell is filled with a grid of voxels; every voxel a drawn wall piece passes through (its copies too: a copy
   is the same shape elsewhere, so it bounds as well) is a wall voxel, and the rest falls into connected pockets
   (6-connected).
2. Pockets are joined where they meet on the 3-sphere: the half-cell holds several images of each point under the
   group (the wall copies), and its faces are glued to images of themselves; a point just past a face, or a point of a
   pocket, brought back into the half-cell by each group element lands in a pocket that is the same place: the two are
   one cell (union-find).
3. Each cell is sampled at points spread over its pockets and classified, to see which region lives there (one region
   per cell when the walls close it off; two means a gap between walls).
4. Each cell's boundary is the wall triangles with that cell on one side (stepped off both ways onto the grid) and the
   parts of the half-cell's faces it reaches: drawn with the walls' own meshes.

    WALL_DUMP=walls.json python cell_atlas2.py ...      then      python region_cells.py walls.json [h]

Written to region_cells.json: {"h", "cells": [{id, pockets, voxels, samples: [[beta, shape], ...], tris (xyz,
unflipped), walls}], "regions": {shape: [cell ids]}}. Classifications are kept in REGION_CELLS_CACHE so a cut-off run
resumes.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from cell_atlas import P
from fexact import _lab

M_INV = np.linalg.inv(np.vstack([P.T, np.ones(4)]))
M = np.vstack([P.T, np.ones(4)])                   # beta -> (x, y, z, 1)
CACHE_PATH = os.environ.get("REGION_CELLS_CACHE", "region_cells_cache.jsonl")
CORNERS = np.array([P[0], P[1], P[2], (P[2] + P[3]) / 2])


def betas(X):
    B = np.c_[X, np.ones(len(X))] @ M_INV.T
    return B / B.sum(axis=1, keepdims=True)


def xyz(B):
    B = np.asarray(B, float)
    B = B / B.sum(axis=-1, keepdims=True)
    return (B @ M.T)[..., :3]


def inside(B, tol=0.0):
    return (B.min(axis=-1) > tol) & (B[..., 2] - B[..., 3] > tol)


def _lab_job(b):
    b = np.asarray(b, float)
    try:
        return _lab(b / b.sum())
    except Exception:
        return "ERR"


def subdivide(t, h):
    n = max(1, int(np.ceil(max(np.linalg.norm(t[i] - t[(i + 1) % 3]) for i in range(3)) / h)))
    at = lambda i, j: t[0] + (i / n) * (t[1] - t[0]) + (j / n) * (t[2] - t[0])
    out = []
    for i in range(n):
        for j in range(n - i):
            out.append([at(i, j), at(i + 1, j), at(i, j + 1)])
            if i + j < n - 1:
                out.append([at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
    return out


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def main(dump, h=0.0025, out="region_cells.json", per_pocket=4, min_voxels=8):
    import edge_snap
    h = float(h)
    W = json.load(open(dump))
    pieces = []                                       # (wall id, copy, triangles)
    for w in W:
        if "tris" in w:
            pieces.append((w["id"], bool(w.get("copy")), [np.asarray(t, float)[:, :3] for t in w["tris"]]))
        elif "poly" in w:
            Q = [np.asarray(q, float)[:3] for q in w["poly"]]
            pieces.append((w["id"], bool(w.get("copy")), [np.array([Q[0], Q[i], Q[i + 1]]) for i in range(1, len(Q) - 1)]))
    lo, hi = CORNERS.min(axis=0) - 3 * h, CORNERS.max(axis=0) + 3 * h
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    vox = lambda X: np.clip(np.floor((np.asarray(X) - lo) / h).astype(int), 0, np.array(shape) - 1)
    print(f"{sum(len(t) for _, _, t in pieces)} wall triangles in {len(pieces)} pieces; grid {shape} "
          f"({np.prod(shape) / 1e6:.1f}M voxels) at h = {h}", flush=True)
    wall = np.zeros(shape, bool)
    for _, _, T in pieces:
        for t in T:
            e = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]), np.linalg.norm(t[0] - t[2]))
            n = max(1, int(np.ceil(e / (0.4 * h))))
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            k = i + j <= n
            a, b = i[k] / n, j[k] / n
            idx = vox(t[0] + a[:, None] * (t[1] - t[0]) + b[:, None] * (t[2] - t[0]))
            wall[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    g = np.stack(np.meshgrid(*[lo[k] + h * (np.arange(shape[k]) + 0.5) for k in range(3)], indexing="ij"), -1)
    Bg = betas(g.reshape(-1, 3)).reshape(shape + (4,))
    free = inside(Bg) & ~wall
    lab, n_p = ndimage.label(free)
    sizes = np.bincount(lab.ravel())
    pockets = [c for c in range(1, n_p + 1) if sizes[c] >= min_voxels]
    print(f"{n_p} pockets, {len(pockets)} of at least {min_voxels} voxels", flush=True)
    pocket_of = lambda X: lab[tuple(vox(X).T)]

    rng = np.random.default_rng(0)
    objs = ndimage.find_objects(lab)
    spread = {}
    depth = ndimage.distance_transform_edt(free)
    for c in pockets:
        sl = objs[c - 1]
        m = lab[sl] == c
        d = np.where(m, depth[sl], 0)
        idx = np.argwhere(m & (d >= min(1.5, d.max())))
        picked = [idx[np.argmax(d[tuple(idx.T)])]]
        far = np.linalg.norm(idx - picked[0], axis=1)
        while len(picked) < per_pocket and far.max() > 2:
            picked.append(idx[np.argmax(far)])
            far = np.minimum(far, np.linalg.norm(idx - picked[-1], axis=1))
        offs = np.array([s.start for s in sl])
        spread[c] = np.array([Bg[tuple(p + offs)] for p in picked])
    # 3. samples and classification
    cache = {}
    if os.path.exists(CACHE_PATH):
        for line in open(CACHE_PATH):
            try:
                k, v = json.loads(line)
                cache[tuple(k)] = v
            except ValueError:
                pass
    key = lambda b: tuple(np.round(b, 11).tolist())
    todo = sorted({key(b) for c in pockets for b in spread[c]} - set(cache))
    print(f"{sum(len(spread[c]) for c in pockets)} samples, {len(todo)} to classify", flush=True)
    with Pool(4) as pool:
        for k, v in zip(todo, pool.imap(_lab_job, todo, chunksize=2)):
            cache[k] = v
            with open(CACHE_PATH, "a") as f:
                f.write(json.dumps([list(k), v]) + "\n")

    # (each pocket's shape: the one its samples all are; a pocket with two is a leak, kept apart)
    pure = {}
    for c in pockets:
        ls = {cache[key(b)] for b in spread[c]}
        pure[c] = ls.pop() if len(ls) == 1 else None
    # 2. joined on the 3-sphere
    G = edge_snap.group()                              # (acting on beta)
    dsu = DSU()
    for c in pockets:
        dsu.find(c)

    def join(B_from, from_pocket):
        """Each point brought back into the half-cell by every group element: its pocket there is the same place."""
        Y = np.einsum("gij,nj->ngi", G, B_from)
        s = Y.sum(axis=-1, keepdims=True)
        Y = Y / np.where(np.abs(s) < 1e-12, 1e-12, s)
        ok = inside(Y, 1e-6) & (s[..., 0] > 0)
        n_, g_ = np.nonzero(ok)
        if not len(n_):
            return 0
        q = pocket_of(xyz(Y[n_, g_]))
        k = 0
        for a, b in zip(from_pocket[n_], q):
            if a > 0 and b > 0 and sizes[a] >= min_voxels and sizes[b] >= min_voxels and dsu.find(a) != dsu.find(b) \
                    and pure.get(int(a)) is not None and pure.get(int(a)) == pure.get(int(b)):
                dsu.union(int(a), int(b))
                k += 1
        return k
    # (a) points of each pocket (several, spread), brought back by the images inside the half-cell
    Bs = np.vstack([spread[c] for c in pockets])
    Ps = np.concatenate([[c] * len(spread[c]) for c in pockets])
    print(f"  joined through images inside the half-cell: {join(Bs, Ps)}", flush=True)
    # (b) points just past each face of the half-cell, brought back by the group, against the pocket just inside
    faces = [[CORNERS[1], CORNERS[2], CORNERS[3]], [CORNERS[0], CORNERS[2], CORNERS[3]],
             [CORNERS[0], CORNERS[1], CORNERS[2]], [CORNERS[0], CORNERS[1], CORNERS[3]]]
    centre = CORNERS.mean(axis=0)
    face_tris = []
    k_total = 0
    for F in faces:
        F = [np.asarray(p, float) for p in F]
        nrm = np.cross(F[1] - F[0], F[2] - F[0])
        nrm /= np.linalg.norm(nrm)
        if (centre - F[0]) @ nrm < 0:
            nrm = -nrm                                   # (pointing into the half-cell)
        T = subdivide(F, 1.5 * h)
        C = np.array([np.mean(t, axis=0) for t in T])
        a_in = pocket_of(C + 1.5 * h * nrm)
        for t, a in zip(T, a_in):
            face_tris.append((t, a))
        k_total += join(betas(C - 0.5 * h * nrm), a_in)
    print(f"  joined across the faces: {k_total}", flush=True)
    cells = defaultdict(list)
    for c in pockets:
        cells[dsu.find(c)].append(c)
    print(f"{len(cells)} cells on the 3-sphere", flush=True)

    # 4. boundaries from the walls' own meshes (and the faces' parts)
    bound = defaultdict(list)
    walls_of = defaultdict(set)
    for wid, copy, T in pieces:
        for t0 in T:
            for t in subdivide(t0, 0.01):
                t = np.asarray(t)
                c0 = t.mean(axis=0)
                n = np.cross(t[1] - t[0], t[2] - t[0])
                ln = np.linalg.norm(n)
                if ln < 1e-14:
                    continue
                n /= ln
                sides = set()
                for s in (1, -1):
                    for k in (1.5, 2.5, 3.5):
                        p = pocket_of((c0 + s * k * h * n)[None])[0]
                        if p > 0:
                            sides.add(dsu.find(int(p)) if sizes[p] >= min_voxels else None)
                            break
                for cid in sides - {None}:
                    bound[cid].append(t.tolist())
                    walls_of[cid].add(wid + (" (copy)" if copy else ""))
    for t, a in face_tris:
        if a > 0 and sizes[a] >= min_voxels:
            bound[dsu.find(int(a))].append(np.asarray(t).tolist())
    out_cells, regions = [], defaultdict(list)
    for cid, ps in cells.items():
        samples = [[b.tolist(), cache[key(b)]] for c in ps for b in spread[c]]
        labs = {l for _, l in samples}
        # (boundary triangles kept only for the regions being drawn, REGION_CELLS_DRAW: the rest would be ~100 MB)
        draw = set(os.environ.get("REGION_CELLS_DRAW", "F2").split(","))
        tris = np.round(np.array(bound.get(cid, [])), 5).tolist() if len(labs) == 1 and labs <= draw else []
        out_cells.append({"id": int(cid), "pockets": [int(c) for c in ps], "voxels": int(sum(sizes[c] for c in ps)),
                          "samples": samples, "tris": tris, "walls": sorted(walls_of.get(cid, []))})
        for l in labs:
            regions[l].append(int(cid))
    json.dump({"h": h, "cells": out_cells, "regions": regions}, open(out, "w"), separators=(",", ":"))
    mixed = [c for c in out_cells if len({l for _, l in c["samples"]}) > 1]
    print(f"{len(out_cells)} cells; {len(regions)} shapes in them; {len(mixed)} with more than one", flush=True)
    for c in sorted(out_cells, key=lambda c: -c["voxels"]):
        print(f"  cell {c['id']}: {len(c['pockets'])} pockets, {c['voxels']} voxels, "
              f"{Counter(l for _, l in c['samples']).most_common()}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
