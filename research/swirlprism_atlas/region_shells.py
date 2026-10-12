"""Each region's closed shell: the pieces of wall (and of the cell's faces) that the region touches. Every wall piece
drawn in the cell view (cell_atlas2.py with WALL_DUMP set: patches, curved meshes and their copies) is stepped off on
both sides and the regions there are recorded; a piece whose samples all agree is given whole to its two sides, a
mixed one (another wall ends on it, so the region beside it changes) triangle by triangle. The parts of the cell's
faces that regions reach (the face beta4 = 0 and the splitting mirror beta3 = beta4) close the shells, refined where
the region there changes. Written to region_shells.json as {region: [{piece, wall, tris (cell xyz, unflipped)}]}.

    WALL_DUMP=walls.json python cell_atlas2.py ...      then      python region_shells.py walls.json
(REGION_SHELLS_CACHE names the file of classifications kept, so an interrupted run resumes.)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from cell_atlas import P
from cell_atlas2 import ear_clip
from fexact import _lab

M_INV = np.linalg.inv(np.vstack([P.T, np.ones(4)]))
H = 0.03                        # the size of the triangles a mixed piece or a cell face is classified at


def beta(q):
    return M_INV @ np.r_[np.asarray(q, float)[:3], 1.0]


# every classification is kept in a cache file (REGION_SHELLS_CACHE), one line each, so a run cut off (the machine
# restarting) resumes where it stopped
import os
CACHE_PATH = os.environ.get("REGION_SHELLS_CACHE", "region_shells_cache.jsonl")
CACHE = {}
if os.path.exists(CACHE_PATH):
    for line in open(CACHE_PATH):
        try:
            k, v = json.loads(line)
            CACHE[tuple(k)] = v
        except ValueError:
            pass


def lab(q):
    b = beta(q)
    if b.min() < -1e-12 or b[2] < b[3] - 1e-12:
        return "OUT"
    b = b / b.sum()
    k = tuple(np.round(b, 11).tolist())
    if k in CACHE:
        return CACHE[k]
    v = _lab(b)
    with open(CACHE_PATH, "a") as f:
        f.write(json.dumps([list(k), v]) + "\n")
    return v


def _sides(args):
    """The shapes a small step off a triangle's centroid on either side, and on the centroid itself."""
    tri, eps, on = args
    T = np.asarray(tri, float)
    c = T.mean(axis=0)
    n = np.cross(T[1] - T[0], T[2] - T[0])
    n /= np.linalg.norm(n)
    return lab(c + eps * n), lab(c - eps * n), (lab(c) if on else None)


def subdivide(T, h=H):
    """A triangle (3 xyz corners) cut into n^2 similar ones with sides below h."""
    T = np.asarray(T, float)
    n = max(1, int(np.ceil(max(np.linalg.norm(T[i] - T[(i + 1) % 3]) for i in range(3)) / h)))
    at = lambda i, j: T[0] + (i / n) * (T[1] - T[0]) + (j / n) * (T[2] - T[0])
    out = []
    for i in range(n):
        for j in range(n - i):
            out.append([at(i, j), at(i + 1, j), at(i, j + 1)])
            if i + j < n - 1:
                out.append([at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)])
    return out


def coarse(tris, h=0.025):
    """A curved mesh coarsened on a grid of spacing h (as the dodecahedron views see it)."""
    acc = {}
    for t in tris:
        for v in t:
            a = acc.setdefault(tuple(int(np.floor(c / h)) for c in v), [np.zeros(3), 0])
            a[0] += v
            a[1] += 1
    out, seen = [], set()
    for t in tris:
        ks = [tuple(int(np.floor(c / h)) for c in v) for v in t]
        if len(set(ks)) < 3 or tuple(sorted(ks)) in seen:
            continue
        seen.add(tuple(sorted(ks)))
        out.append([acc[k][0] / acc[k][1] for k in ks])
    return out


def known(l):
    return l is not None and not l.startswith("new:") and l != "ERR"


def main(dump, out="region_shells.json"):
    W = json.load(open(dump))
    pieces = []                         # (wall id, copy, triangles, eps)
    for w in W:
        if "poly" in w:                 # (the T2 mirror patches come as polygons)
            C = [beta(q) for q in w["poly"]]
            nrm = np.linalg.svd(np.vstack([np.ones(4)] + [c - C[0] for c in C[1:]]))[2][-1]
            T = [[np.asarray(w["poly"][a], float), np.asarray(w["poly"][b], float), np.asarray(w["poly"][c], float)]
                 for a, b, c in ear_clip(C, nrm)]
            pieces.append((w["id"], w["copy"], T, 3e-5, None))
        else:
            T = [np.asarray(t, float) for t in w["tris"]]
            # (a curved mesh is drawn coarsened, but stepped off from its own triangles: a coarse triangle can sit off
            # the surface by more than the step)
            curved = len(T) > 60
            pieces.append((w["id"], w["copy"], coarse(T) if curved else T, 6e-4 if curved else 3e-5, T if curved else None))
    rng = np.random.default_rng(0)
    pool = Pool(4)
    # 1. a dozen samples per piece, spread by area
    jobs, owner = [], []
    for k, (wid, _, T, eps, fine) in enumerate(pieces):
        T = fine or T
        area = np.array([np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) for t in T])
        for i in rng.choice(len(T), size=min(12, len(T)), replace=False, p=area / area.sum()):
            jobs.append((T[i], eps, False))
            owner.append(k)
    res = pool.map(_sides, jobs, chunksize=4)
    votes = defaultdict(Counter)
    for k, (a, b, _) in zip(owner, res):
        if known(a) and known(b):
            votes[k][tuple(sorted((a, b)))] += 1
    whole, mixed = {}, []
    for k in range(len(pieces)):
        v = votes[k].most_common()
        if len(v) == 1:
            whole[k] = v[0][0]
        else:
            mixed.append(k)
    print(f"{len(pieces)} pieces: {len(whole)} with one pair of sides, {len(mixed)} mixed", flush=True)
    # 2. the mixed pieces triangle by triangle (flat ones cut small first)
    shells = defaultdict(list)
    for k, sides in whole.items():
        wid, copy, T, _, _ = pieces[k]
        for r in set(sides) - {wid, "OUT"}:
            shells[r].append({"piece": k, "wall": wid, "copy": copy, "tris": [np.asarray(t).tolist() for t in T]})
    for k in mixed:
        wid, copy, T, eps, fine = pieces[k]
        if fine:                         # each coarse triangle is stepped off from the fine one nearest its centre
            small = T
            fc = np.array([np.mean(t, axis=0) for t in fine])
            probe = [fine[int(np.argmin(np.linalg.norm(fc - np.mean(t, axis=0), axis=1)))] for t in small]
        else:
            small = probe = [s for t in T for s in subdivide(t)]
        res = pool.map(_sides, [(t, eps, False) for t in probe], chunksize=4)
        by = defaultdict(list)
        for t, (a, b, _) in zip(small, res):
            for r in {a, b} - {wid, "OUT"}:
                if known(r):
                    by[r].append(np.asarray(t).tolist())
        for r, tl in by.items():
            shells[r].append({"piece": k, "wall": wid, "copy": copy, "tris": tl})
        print(f"  piece {k} ({wid}{' copy' if copy else ''}): {len(small)} triangles, sides {Counter(l for ab in res for l in ab[:2]).most_common(5)}", flush=True)
    # 3. the cell's faces that regions reach: beta4 = 0 and the splitting mirror; refined twice where the region there
    # changes. A part with a wall on it is that wall's piece, already counted.
    faces = {"face b4=0": [P[0], P[1], P[2]], "mirror b3=b4": [P[0], P[1], (P[2] + P[3]) / 2]}
    inner = (P[0] + P[1] + P[2] + P[3]) / 4 + np.array([0, 0, 0.0])
    for name, F in faces.items():
        F = [np.asarray(p, float) for p in F]
        n = np.cross(F[1] - F[0], F[2] - F[0])
        if (inner - F[0]) @ n < 0:          # (wind each face so its normal points into the half cell)
            F = [F[0], F[2], F[1]]
        tris = subdivide(F, 2 * H)
        for level in range(3):
            res = pool.map(_sides, [(t, 3e-5, True) for t in tris], chunksize=4)
            got = [(t, a if a == on else None) for t, (a, _, on) in zip(tris, res)]
            if level == 2:
                break
            # cut again the triangles beside a triangle with another region (or with a wall) on the face
            vkey = lambda p: tuple(np.round(p, 7))
            at_vertex = defaultdict(set)
            for t, r in got:
                for p in t:
                    at_vertex[vkey(p)].add(r)
            nxt, keep = [], []
            for t, r in got:
                if any(len(at_vertex[vkey(p)]) > 1 for p in t):
                    nxt += subdivide(t, np.linalg.norm(t[1] - t[0]) / 1.9)
                else:
                    keep.append((t, r))
            for t, r in keep:
                if known(r):
                    shells[r].append({"piece": name, "wall": name, "copy": False, "tris": [np.asarray(t).tolist()]})
            tris = nxt
            print(f"  {name} level {level}: {len(keep)} settled, {len(nxt)} to refine", flush=True)
        for t, r in got:
            if known(r):
                shells[r].append({"piece": name, "wall": name, "copy": False, "tris": [np.asarray(t).tolist()]})
    pool.close()
    # merge each region's face pieces into one entry per face
    out_d = {}
    for r, parts in shells.items():
        merged = defaultdict(lambda: {"tris": []})
        for p in parts:
            m = merged[p["piece"]]
            m.update({"piece": p["piece"], "wall": p["wall"], "copy": p["copy"]})
            m["tris"] += p["tris"]
        out_d[r] = list(merged.values())
    json.dump(out_d, open(out, "w"))
    for r in sorted(out_d):
        print(r, sorted({p["wall"] for p in out_d[r]}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
