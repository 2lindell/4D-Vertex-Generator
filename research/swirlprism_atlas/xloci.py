"""Which atlas types fill 3D regions of the cell and which only live on walls, lines or points.

Step 1 (python xloci.py step1): for one sample of each type (the one deepest inside the cell), nudge the
seed by EPS in random directions, both ways, and record which nudges keep the type. A type that
survives nearly every nudge fills a region; one that survives few or none is lower-dimensional
(transitional) and is located further in step 2.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from cell_atlas2 import identify, load_refs
from probe import _one

EPS = 1e-5
N_DIRS = 16


def samples_by_type():
    refs = load_refs()[0]
    xids = json.load(open("atlas_xids.json"))
    out = defaultdict(list)
    for s in json.load(open("golden_22.json")):
        cid, xkey, _ = identify(s["sig"], refs)
        tid = cid or xids.get(xkey)
        if tid:
            b = np.array(s["beta"], float)
            out[tid].append(b / b.sum())
    return out


def directions(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((N_DIRS, 4))
    d -= d.mean(axis=1, keepdims=True)       # stay in the barycentric simplex (sum fixed)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _label(b):
    return _one(b)[0]


def step1():
    S = samples_by_type()
    jobs, meta = [], []
    D = directions()
    for tid, bs in sorted(S.items()):
        b = max(bs, key=lambda x: x.min())    # the sample deepest inside the cell
        meta.append((tid, b))
        jobs += [b + s * EPS * d for d in D for s in (1, -1)]
    with Pool(4) as pool:
        labels = pool.map(_label, jobs, chunksize=8)
    out, k = {}, 0
    for tid, b in meta:
        got = labels[k:k + 2 * N_DIRS]
        k += 2 * N_DIRS
        out[tid] = {"beta": b.tolist(), "kept": sum(g == tid for g in got), "of": len(got),
                    "neighbours": sorted({g for g in got if g != tid})}
    json.dump(out, open("xloci_step1.json", "w"), indent=1)
    for tid, r in out.items():
        print(f"{tid:5s} kept {r['kept']:2d}/{r['of']}  neighbours {r['neighbours'][:6]}")


PHI = (1 + 5 ** 0.5) / 2
_A, _B = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
GOLDEN = np.unique(np.round((_A + _B * PHI).ravel(), 12))     # a + b*phi, |a|, |b| <= 3


def golden_planes(b, tol=1e-10):
    """Planes n.beta = 0 with every n_i of the form a + b*phi (|a|, |b| <= 3) passing through b."""
    i4 = int(np.argmax(b))                    # solve for the largest weight's coefficient
    rest = [i for i in range(4) if i != i4]
    grid = np.stack(np.meshgrid(GOLDEN, GOLDEN, GOLDEN, indexing="ij"), -1).reshape(-1, 3)
    n4 = -(grid @ b[rest]) / b[i4]
    near = np.abs(GOLDEN[np.searchsorted(GOLDEN, n4).clip(1, len(GOLDEN) - 1) - 1] - n4)
    near = np.minimum(near, np.abs(GOLDEN[np.searchsorted(GOLDEN, n4).clip(0, len(GOLDEN) - 1)] - n4))
    found = {}
    for g, v in zip(grid[near < tol], n4[near < tol]):
        n = np.zeros(4); n[rest] = g; n[i4] = v
        if not n.any():
            continue
        n /= n[np.flatnonzero(np.abs(n) > 1e-9)[0]]
        key = tuple(np.round(n, 9))
        found[key] = n
    # simplest first: fewest non-zero entries, then smallest entries
    return sorted(found.values(), key=lambda n: ((np.abs(n) > 1e-9).sum(), np.abs(n).sum()))


def in_plane_dirs(normals, k):
    """k unit directions keeping beta's sum and n.beta fixed for every n in normals."""
    rng = np.random.default_rng(1)
    A = np.vstack([np.ones(4), *normals])
    basis = np.linalg.svd(A)[2][len(A):]       # null space
    d = rng.standard_normal((k, len(basis))) @ basis
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _keeps(tid, b, normals, k=4):
    """Do k in-plane nudges (both ways) all keep the type? Stops at the first one that does not."""
    for d in in_plane_dirs(normals, k):
        for s in (1, -1):
            if _one(b + s * EPS * d)[0] != tid:
                return False
    return True


def _locate(job):
    tid, b = job
    planes = golden_planes(b)[:30]
    res = {"beta": b.tolist()}
    for n in planes:
        if _keeps(tid, b, [n]):
            return tid, {**res, "kind": "wall", "normals": [n.tolist()]}
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            pair = [planes[i], planes[j]]
            if np.linalg.matrix_rank(np.vstack([np.ones(4), *pair]), 1e-8) == 3 and _keeps(tid, b, pair):
                return tid, {**res, "kind": "line", "normals": [n.tolist() for n in pair]}
    return tid, {**res, "kind": "point or curved"}


def step2():
    """For each type that does not fill a region, find the simplest golden plane (or pair of planes) along
    which nudges keep the type: a wall or a line. Types left over are points or lie on curved loci."""
    r1 = json.load(open("xloci_step1.json"))
    jobs = [(t, np.array(r["beta"])) for t, r in sorted(r1.items()) if r["kept"] < r["of"]]
    out = {}
    with Pool(4) as pool:
        for tid, res in pool.imap_unordered(_locate, jobs):
            out[tid] = res
            print(tid, res["kind"], np.round(res.get("normals", []), 4).tolist(), flush=True)
    json.dump(dict(sorted(out.items())), open("xloci_step2.json", "w"), indent=1)


SWAP = [1, 0, 3, 2]      # the exact half-turn that folds the cell onto the displayed half (beta3 >= beta4)


def _canon(n):
    n = np.asarray(n, float)
    n = n / n[np.flatnonzero(np.abs(n) > 1e-9)[0]]
    return n


def _section(normals):
    """Corners of {beta >= 0, sum = 1, n.beta = 0 for n in normals}: a polygon (one normal) or a segment (two)."""
    from itertools import combinations
    A = np.vstack([np.ones(4), *normals])
    need = 4 - len(A)                                  # how many beta_i must be 0 at a corner
    pts = []
    for zeros in combinations(range(4), need):
        M = np.vstack([A, np.eye(4)[list(zeros)]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        b = np.linalg.solve(M, np.r_[1.0, np.zeros(len(M) - 1)])
        if b.min() > -1e-12 and not any(np.allclose(b, q, atol=1e-10) for q in pts):
            pts.append(np.clip(b, 0, None))
    return pts


def _order(pts, normal):
    c = np.mean(pts, axis=0)
    basis = np.linalg.svd(np.vstack([np.ones(4), normal]))[2][2:]
    ang = [np.arctan2(*((p - c) @ basis.T)[::-1]) for p in pts]
    return [pts[k] for k in np.argsort(ang)]


def step3(n=28):
    """Label a triangular grid on each wall (and on its image under the half-turn), keeping the displayed half."""
    r2 = json.load(open("xloci_step2.json"))
    walls = {}
    for t, r in r2.items():
        if r["kind"] == "wall":
            for nrm in (_canon(r["normals"][0]), _canon(np.asarray(r["normals"][0])[SWAP])):
                walls.setdefault(tuple(np.round(nrm, 9)), nrm)
    jobs, meta = [], []
    for key, nrm in walls.items():
        poly = _order(_section([nrm]), nrm)
        c = np.mean(poly, axis=0)
        for k in range(len(poly)):                        # fan of triangles from the centre
            a, b = poly[k], poly[(k + 1) % len(poly)]
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    p = (i * a + j * b + (n - i - j) * c) / n
                    meta.append((key, k, i, j))
                    jobs.append(p)
    print(len(walls), "walls,", len(jobs), "grid points", flush=True)
    with Pool(4) as pool:
        labels = pool.map(_label, jobs, chunksize=8)
    out = {}
    for (key, k, i, j), p, lab in zip(meta, jobs, labels):
        w = out.setdefault(str(list(key)), {"normal": list(key), "n": n, "points": []})
        w["points"].append({"fan": k, "ij": [i, j], "beta": p.tolist(), "label": lab})
    json.dump(out, open("xloci_walls.json", "w"))


def step4(m=48, steps=30):
    """Each line type's segments: sample the line's chord through the cell, then bisect the ends of each run."""
    r2 = json.load(open("xloci_step2.json"))
    lines = {}
    for t, r in r2.items():
        if r["kind"] == "line":
            for ns in ([_canon(x) for x in r["normals"]], [_canon(np.asarray(x)[SWAP]) for x in r["normals"]]):
                ends = _section(ns)
                if len(ends) == 2:
                    lines.setdefault(t, []).append(ends)
    jobs, meta = [], []
    for t, chords in lines.items():
        for c, (a, b) in enumerate(chords):
            for k in range(m + 1):
                meta.append((t, c, k))
                jobs.append(a + (b - a) * k / m)
    with Pool(4) as pool:
        labels = pool.map(_label, jobs, chunksize=8)
        found = {}
        for (t, c, k), lab in zip(meta, labels):
            found.setdefault((t, c), []).append(lab)
        bis = []
        for (t, c), labs in found.items():
            a, b = lines[t][c]
            for k in range(m + 1):
                if labs[k] != t:
                    continue
                if k > 0 and labs[k - 1] != t:
                    bis.append((t, c, k, -1, a + (b - a) * (k - 1) / m, a + (b - a) * k / m))
                if k < m and labs[k + 1] != t:
                    bis.append((t, c, k, 1, a + (b - a) * k / m, a + (b - a) * (k + 1) / m))
        refined = pool.map(_bisect_end, [(t, lo, hi) for t, _, _, _, lo, hi in bis])
    out = {}
    for (t, c, k, side, _, _), p in zip(bis, refined):
        out.setdefault(t, {}).setdefault(str(c), {"chord": [x.tolist() for x in lines[t][c]], "ends": []})
        out[t][str(c)]["ends"].append({"k": k, "side": side, "beta": p})
    for (t, c), labs in found.items():
        out.setdefault(t, {}).setdefault(str(c), {"chord": [x.tolist() for x in lines[t][c]], "ends": []})
        out[t][str(c)]["labels"] = labs
        runs = [k for k, x in enumerate(labs) if x == t]
        print(t, c, "samples on the type:", runs[:3], "...", runs[-3:] if runs else "", flush=True)
    json.dump(out, open("xloci_lines.json", "w"), indent=1)


def _bisect_end(job, steps=30):
    """Bisect between a point of type t (one end) and a point of another type; return the last t point."""
    t, p, q = job
    lt = _label(p) == t
    good, bad = (p, q) if lt else (q, p)
    for _ in range(steps):
        mid = (good + bad) / 2
        if _label(mid) == t:
            good = mid
        else:
            bad = mid
    return good.tolist()


if __name__ == "__main__":
    {"step1": step1, "step2": step2, "step3": step3, "step4": step4}[sys.argv[1]]()
