"""The cells of the half tetrahedron cut by every wall: the half-cell is filled with a grid of voxels, every voxel a
drawn wall piece passes through (its copies too: a copy is the same shape elsewhere, so it bounds as well) is a wall
voxel, and the rest falls into connected pockets (6-connected): the cells. Each cell is then sampled at the voxels
deepest inside it and classified, to see which region lives there (one region per cell when the walls close it off;
two in one cell means a gap between walls, a leak). Each cell's surface is meshed (marching cubes) for drawing.

    WALL_DUMP=walls.json python cell_atlas2.py ...      then      python region_cells.py walls.json [h]

Written to region_cells.json: {"h", "cells": [{id, voxels, volume, samples: [[beta, shape], ...], verts, faces}],
"regions": {shape: [cell ids]}}. Classifications are kept in REGION_CELLS_CACHE so a cut-off run resumes.
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
CACHE_PATH = os.environ.get("REGION_CELLS_CACHE", "region_cells_cache.jsonl")


def betas(X):
    B = np.c_[X, np.ones(len(X))] @ M_INV.T
    return B / B.sum(axis=1, keepdims=True)


def _lab_job(b):
    b = np.asarray(b, float)
    try:
        return _lab(b / b.sum())
    except Exception:
        return "ERR"


def main(dump, h=0.002, out="region_cells.json", per_cell=8, min_voxels=8):
    h = float(h)
    W = json.load(open(dump))
    tris = []
    for w in W:
        if "tris" in w:
            tris += [np.asarray(t, float)[:, :3] for t in w["tris"]]
        elif "poly" in w:                           # (a flat polygon: a fan from its first corner)
            Q = [np.asarray(q, float)[:3] for q in w["poly"]]
            tris += [np.array([Q[0], Q[i], Q[i + 1]]) for i in range(1, len(Q) - 1)]
    T = np.array(tris)
    corners = np.array([P[0], P[1], P[2], (P[2] + P[3]) / 2])
    lo, hi = corners.min(axis=0) - 2 * h, corners.max(axis=0) + 2 * h
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    print(f"{len(T)} wall triangles; grid {shape} ({np.prod(shape) / 1e6:.1f}M voxels) at h = {h}", flush=True)
    wall = np.zeros(shape, bool)
    # (each triangle sampled at under half a voxel, so the wall voxels form an unbroken sheet a 6-connected path
    # cannot slip through)
    for t in T:
        e = max(np.linalg.norm(t[1] - t[0]), np.linalg.norm(t[2] - t[1]), np.linalg.norm(t[0] - t[2]))
        n = max(1, int(np.ceil(e / (0.4 * h))))
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        X = t[0] + a[:, None] * (t[1] - t[0]) + b[:, None] * (t[2] - t[0])
        idx = np.floor((X - lo) / h).astype(int)
        wall[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    # outside the half tetrahedron (beta >= 0, beta3 >= beta4) is not part of any cell
    g = np.stack(np.meshgrid(*[lo[k] + h * (np.arange(shape[k]) + 0.5) for k in range(3)], indexing="ij"), -1)
    B = betas(g.reshape(-1, 3)).reshape(shape + (4,))
    inside = (B.min(axis=-1) > 0) & (B[..., 2] > B[..., 3])
    free = inside & ~wall
    lab, n_cells = ndimage.label(free)                # (6-connected)
    sizes = np.bincount(lab.ravel())
    keep_ids = [c for c in range(1, n_cells + 1) if sizes[c] >= min_voxels]
    print(f"{n_cells} pockets; {len(keep_ids)} of at least {min_voxels} voxels "
          f"({sum(sizes[c] for c in range(1, n_cells + 1) if sizes[c] < min_voxels)} voxels in the rest)", flush=True)
    # points spread over each cell (farthest-point sampling from its deepest voxel, among voxels not right against
    # a wall), so a cell that is two regions joined through a gap shows both
    depth = ndimage.distance_transform_edt(free)
    samples = {}
    objs = ndimage.find_objects(lab)
    for c in keep_ids:
        sl = objs[c - 1]
        m = lab[sl] == c
        d = np.where(m, depth[sl], 0)
        idx = np.argwhere(m & (d >= min(1.5, d.max())))
        picked = [idx[np.argmax(d[tuple(idx.T)])]]
        far = np.linalg.norm(idx - picked[0], axis=1)
        while len(picked) < per_cell and far.max() > 2:
            picked.append(idx[np.argmax(far)])
            far = np.minimum(far, np.linalg.norm(idx - picked[-1], axis=1))
        offs = np.array([s.start for s in sl])
        samples[c] = [B[tuple(p + offs)] for p in picked]
    cache = {}
    if os.path.exists(CACHE_PATH):
        for line in open(CACHE_PATH):
            try:
                k, v = json.loads(line)
                cache[tuple(k)] = v
            except ValueError:
                pass
    todo = sorted({tuple(np.round(b, 11).tolist()) for bs in samples.values() for b in bs} - set(cache))
    print(f"{sum(len(v) for v in samples.values())} samples, {len(todo)} to classify", flush=True)
    with Pool(4) as pool:
        for k, v in zip(todo, pool.imap(_lab_job, todo, chunksize=2)):
            cache[k] = v
            with open(CACHE_PATH, "a") as f:
                f.write(json.dumps([list(k), v]) + "\n")
    from skimage.measure import marching_cubes
    cells, regions = [], defaultdict(list)
    for c in keep_ids:
        sl = objs[c - 1]
        m = np.pad(lab[sl] == c, 1)
        try:
            V, F, _, _ = marching_cubes(m.astype(np.float32), 0.5, step_size=2 if sizes[c] > 4000 else 1)
        except (ValueError, RuntimeError):
            V, F = np.zeros((0, 3)), np.zeros((0, 3), int)
        offs = np.array([s.start for s in sl]) - 1
        Vx = lo + h * (V + offs + 0.5)
        labs = [cache[tuple(np.round(b, 11).tolist())] for b in samples[c]]
        cells.append({"id": c, "voxels": int(sizes[c]), "volume": float(sizes[c] * h ** 3),
                      "samples": [[b.tolist(), l] for b, l in zip(samples[c], labs)],
                      "verts": np.round(Vx, 5).tolist(), "faces": F.tolist()})
        for l in set(labs):
            regions[l].append(c)
    json.dump({"h": h, "cells": cells, "regions": regions}, open(out, "w"))
    mixed = [c for c in cells if len({l for _, l in c["samples"]}) > 1]
    print(f"{len(cells)} cells; {len(regions)} shapes found in them; {len(mixed)} cells with more than one", flush=True)
    for c in sorted(cells, key=lambda c: -c["voxels"]):
        print(f"  cell {c['id']}: {c['voxels']} voxels, {Counter(l for _, l in c['samples']).most_common()}", flush=True)
    for r, cs in sorted(regions.items()):
        print(f"  {r}: cells {cs}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
