"""What bounds a shape: walked out from points inside it to where it ends (bounds_walk.json).

From each start point (inside the shape: checked), along a fixed, evenly spread set of directions (within the shape's
plane for a flat wall; all of the half-cell's three for a region), small steps are taken (2e-4 for the first 0.02, where thin
neighbours are, then 5e-4) until the classification changes; the change is bisected to 1e-12 and the point there is
classified as on a wall (the hull tolerance relaxed, as lab_edge): for a region that names the wall bounding it there,
for a wall the line (or the wall it ends on). Just past it, the shape beyond. A walk that leaves the half-cell first
ends on its face.

    python bounds_walk.py X101,X102,X103
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from core import shape_key

OUT = "bounds_walk.json"
STEP = 2e-4                 # (near the start, where thin neighbours are)
FAR, STEP_FAR = 0.02, 5e-4  # (past this, the step)
MAX_LEN = 0.5
CELL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])   # (b1, b2, b4, b3 - b4 >= 0)


def name(b, tol=1e-9):
    return shape_key(b, tol)[0]


def directions(U, n):
    """n directions spread evenly over the unit sphere of the span of U's rows (a circle for two, a Fibonacci sphere
    for three), as betas."""
    if len(U) == 2:
        a = np.pi * np.arange(n) / n * 2
        return np.cos(a)[:, None] * U[0] + np.sin(a)[:, None] * U[1]
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * k
    return (r * np.cos(a))[:, None] * U[0] + (r * np.sin(a))[:, None] * U[1] + z[:, None] * U[2]


def walk(task):
    shape, tol, b0, d = task
    b0, d = np.asarray(b0, float), np.asarray(d, float)
    d = d / np.linalg.norm(d)
    prev = b0
    ts = np.concatenate([np.arange(STEP, FAR, STEP), np.arange(FAR, MAX_LEN, STEP_FAR)])
    for t in ts:
        b = b0 + t * d
        if (b @ CELL.T < 0).any():
            return {"start": b0.tolist(), "dir": d.tolist(), "ends": "the half-cell's face", "at": prev.tolist()}
        n_ = name(b, tol)
        if n_ != shape:
            lo, hi = prev, b
            for _ in range(40):
                m = (lo + hi) / 2
                if name(m, tol) == shape:
                    lo = m
                else:
                    hi = m
            mid = (lo + hi) / 2
            return {"start": b0.tolist(), "dir": d.tolist(), "on": name(mid, 1e-7),
                    "beyond": name(hi + 1e-6 * d, tol), "at": mid.tolist(), "length": float(np.linalg.norm(mid - b0))}
        prev = b
    return {"start": b0.tolist(), "dir": d.tolist(), "ends": "not within the walk's length"}


def _walk_i(it):
    return it[0], walk(it[1])


def plan(shape, starts, normals=(), tol=1e-9, n_dir=None):
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if normals:
        N = np.array([np.asarray(v, float) for v in normals])
        _, sv, vt = np.linalg.svd(N @ U.T)
        U = vt[int((sv > 1e-9).sum()):] @ U
    D = directions(U, n_dir or (32 if len(U) == 2 else 60))
    good = [np.asarray(s, float) / np.sum(s) for s in starts]
    good = [s for s in good if name(s, tol) == shape]
    return [(shape, tol, s, d) for s in good for d in D], len(good), len(starts)


def summary(recs):
    out = Counter()
    pts = {}
    for r in recs:
        k = (r["on"], r["beyond"]) if "on" in r else (r["ends"], "")
        out[k] += 1
        pts.setdefault(k, r["at"])
    return [{"on": k[0], "beyond": k[1], "walks": n, "at": pts[k]} for k, n in out.most_common()]


def main(shapes):
    n26 = np.array([0, -1, 1, -1.0]); n26 = n26 - n26.mean(); n26 /= np.linalg.norm(n26)
    n79 = np.array([0, 1, 0, -1.0]); n79 = n79 - n79.mean(); n79 /= np.linalg.norm(n79)

    def at(b1, s26, s79, e):
        t = (1 - b1) / 4
        x = np.array([b1, t, 2 * t, t]); x = x / x.sum()
        return x + e * (s26 * n26 + s79 * n79)
    base = json.load(open("working_base.json"))["shapes"]
    quad = {"X72": (1, 1), "X103": (1, -1), "X73": (-1, 1), "X102": (-1, -1)}
    side = {"X101": 1, "X79": -1}
    tasks, info = [], {}
    for s in shapes:
        b1s = (0.64, 0.72, 0.82, 0.92)
        if s in quad:
            starts = [at(b1, *quad[s], e) for b1 in b1s for e in (1e-3, 4e-3)]
            starts.append(base[s]["seed"])
            t, ok, n = plan(s, starts)
        else:
            starts = [at(b1, side[s], 0, e) for b1 in b1s for e in (1e-3, 3e-3)]
            starts.append(base[s]["seed"])
            t, ok, n = plan(s, starts, normals=[[0, 1, 0, -1]], tol=1e-7)
        info[s] = {"starts_inside": ok, "starts_tried": n}
        tasks += t
    print(f"{len(tasks)} walks", flush=True)
    recs = [None] * len(tasks)
    with Pool() as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_walk_i, list(enumerate(tasks)), chunksize=2)):
            recs[i] = r
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(tasks)} walks done", flush=True)
    res = defaultdict(list)
    for t, r in zip(tasks, recs):
        res[t[0]].append(r)
    out = {"_about": __doc__.strip().split("\n\n")[0], "step": STEP,
           "shapes": {s: {**info[s], "bounds": summary(res[s])} for s in shapes}}
    json.dump(out, open(OUT, "w"), indent=1)
    for s in shapes:
        print(s, info[s], flush=True)
        for b in out["shapes"][s]["bounds"]:
            print(f"   on {b['on']:12s} beyond {b['beyond']:12s} walks {b['walks']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1].split(","))
