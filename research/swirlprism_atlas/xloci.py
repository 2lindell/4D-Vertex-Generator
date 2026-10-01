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


if __name__ == "__main__":
    {"step1": step1, "step2": step2}[sys.argv[1]]()
