"""Exact extent of the named wall shapes (F1, F4, F5) inside their walls.

Each wall meets the half-cell in a polygon. A triangular grid on it is classified, every grid edge across which
the target shape starts or stops is bisected to its boundary point (with the class on the other side), and the
boundary points are then fitted with exact lines (great circles) in the wall.

    python fexact.py grid F1 b1=b2        # classify the grid and bisect the boundary (writes fexact_<wall>.json)
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from probe import _one

NORMALS = {"b1=b2": [1, -1, 0, 0]}
WALLS = {   # corners of the wall's polygon in the half-cell (barycentric beta)
    "b1=b2": [[0.5, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.5]],
}


def _lab(b):
    b = np.asarray(b, float)
    return _one(b / b.sum())[0]


def grid(corners, n):
    C = np.array(corners, float)
    pts, idx = [], {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            idx[(i, j)] = len(pts)
            pts.append((i * C[0] + j * C[1] + k * C[2]) / n)
    return pts, idx


def _bisect(job, steps=34):
    a, b, la, lb = job
    a, b = np.array(a), np.array(b)
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        m = (lo + hi) / 2
        if _lab(a + m * (b - a)) == la:
            lo = m
        else:
            hi = m
    return {"beta": (a + (lo + hi) / 2 * (b - a)).tolist(), "inside": la, "outside": _lab(a + hi * (b - a)),
            "far": lb, "t": [lo, hi]}


def run_grid(target, wall, n=24, procs=4):
    pts, idx = grid(WALLS[wall], n)
    with Pool(procs) as pool:
        labs = pool.map(_lab, pts, chunksize=4)
        edges = set()
        for (i, j), a in idx.items():
            for di, dj in ((1, 0), (0, 1), (-1, 1)):
                b = idx.get((i + di, j + dj))
                if b is not None:
                    edges.add((min(a, b), max(a, b)))
        jobs = []
        for a, b in sorted(edges):
            if (labs[a] == target) != (labs[b] == target):
                ins, out = (a, b) if labs[a] == target else (b, a)
                jobs.append((pts[ins].tolist(), pts[out].tolist(), target, labs[out]))
        print(len(jobs), "boundary crossings", flush=True)
        bnd = pool.map(_bisect, jobs, chunksize=2)
    json.dump({"target": target, "wall": wall, "n": n, "points": [p.tolist() for p in pts], "labels": labs,
               "boundary": bnd}, open(f"fexact_{target}_{wall}.json", "w"))
    from collections import Counter
    print("grid labels:", Counter(labs).most_common())
    print("neighbours across the boundary:", Counter(b["outside"] for b in bnd).most_common())


PHI = (1 + 5 ** 0.5) / 2


def golden_form(x, maxq=12, tol=1e-7):
    """x as (a + b phi)/q with small integers, or None."""
    best = None
    for q in range(1, maxq + 1):
        for b in range(-24, 25):
            a = round(x * q - b * PHI)
            if abs((a + b * PHI) / q - x) < tol:
                cand = (abs(a) + abs(b) + q, a, b, q)
                if best is None or cand < best:
                    best = cand
    return best and best[1:]


def fmt_golden(f):
    a, b, q = f
    if b == 0:
        s = f"{a}"
    elif a == 0:
        s = f"{'' if b == 1 else '-' if b == -1 else b}φ"
    else:
        s = f"{a}{'+' if b > 0 else '-'}{'' if abs(b) == 1 else abs(b)}φ"
    return s if q == 1 else f"({s})/{q}"


def plane_through(B, wall_normal):
    """The second plane (normal m, m . beta = 0) holding all points B of a great circle in the wall, snapped to
    golden coefficients; returns (m, residual, text)."""
    B = np.asarray(B, float)
    w = np.asarray(wall_normal, float) / np.linalg.norm(wall_normal)
    P = B - np.outer(B @ w, w)                         # remove the wall direction
    u, s, vt = np.linalg.svd(np.vstack([P, w]))
    m = vt[-1]
    m = m - (m @ w) * w
    m /= np.abs(m).max()
    m *= np.sign(m[np.argmax(np.abs(m))])
    res = float(np.abs(B @ m).max())
    k = np.argmax(np.abs(m))
    snapped = []
    for v in m / m[k]:
        f = golden_form(v)
        snapped.append(f)
    if all(f is not None for f in snapped):
        mm = np.array([(a + b * PHI) / q for a, b, q in snapped])
        terms = [f"{fmt_golden(f)}·β{i + 1}" for i, f in enumerate(snapped) if f[0] or f[1]]
        return mm, float(np.abs(B @ mm).max()), " + ".join(terms).replace("+ -", "− ") + " = 0"
    return m, res, "not golden"


def fit(target, wall, normal):
    d = json.load(open(f"fexact_{target}_{wall}.json"))
    from collections import defaultdict
    groups = defaultdict(list)
    for b in d["boundary"]:
        groups[b["outside"]].append(b["beta"])
    out = []
    for nb, B in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        B = np.array(B)
        sv = np.linalg.svd(B / np.linalg.norm(B, axis=1, keepdims=True), compute_uv=False)
        m, res, text = plane_through(B, normal)
        out.append({"neighbour": nb, "points": len(B), "rank_sv": sv.tolist(), "plane": m.tolist(), "residual": res, "equation": text})
        print(f"{target} | {nb:6s} {len(B):3d} pts  singular values {np.round(sv, 6).tolist()}  ->  {text}  (max off {res:.1e})")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        run_grid(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], NORMALS[sys.argv[3]])
