"""The equation of a wall shape found at a sample: points of the wall are found by bisecting short chords between its two
neighbouring regions at spots around the sample (out to a few hundredths), keeping the boundary points that classify as
the shape (loose classifier). They are fitted by a plane, or else a quadric, whose coefficients are snapped to golden
numbers and checked.

    python wall_fit.py out.json X15:A:B:b1,b2,b3,b4 [X25:...]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

from exact_conics import snap
from fexact import _lab, _lab_loose

U = np.linalg.svd(np.ones((1, 4)))[2][1:]
MONS = [(i, j) for i in range(4) for j in range(i, 4)]
rng = np.random.default_rng(13)


def N(v):
    v = np.asarray(v, float)
    return v / v.sum()


def lab(x):
    x = np.asarray(x, float)
    if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
        return "OUT"
    return _lab(N(x))


def _chord(args):
    tid, a, b, sides = args
    a, b = np.asarray(a), np.asarray(b)
    la, lb = lab(a), lab(b)
    if {la, lb} != set(sides):
        return None
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = (lo + hi) / 2
        if lab(a + m * (b - a)) == la:
            lo = m
        else:
            hi = m
    x = N(a + (lo + hi) / 2 * (b - a))
    return x.tolist() if _lab_loose(x) == tid else None


def row(x):
    return np.array([x[i] * x[j] for i, j in MONS])


def fit(P):
    P = np.array(P)
    sv = np.linalg.svd(P, compute_uv=False)
    n = np.linalg.svd(P)[2][-1]
    if sv[-1] < 1e-9 * sv[0]:
        vals, forms = snap(n, tol=1e-6)
        return {"kind": "plane", "normal": (n / n[np.argmax(np.abs(n))]).tolist(), "golden": forms,
                "residual": float(np.abs(P @ n).max())}
    A = np.array([row(x) for x in P])
    q = np.linalg.svd(A)[2][-1]
    res = float(np.abs(A @ q).max())
    vals, forms = snap(q, tol=1e-5)
    return {"kind": "quadric" if res < 1e-8 else "curved (not a quadric)", "coeffs": q.tolist(), "golden": forms,
            "residual": res, "golden_residual": None if vals is None else float(np.abs(A @ np.array(vals)).max()),
            "plane_residual": float(np.abs(P @ n).max())}


def main(out, specs):
    res = {}
    with Pool(4) as pool:
        for spec in specs:
            tid, A, B, beta = spec.split(":")
            q = N([float(v) for v in beta.split(",")])
            jobs = []
            for r in (0.002, 0.006, 0.015, 0.03):
                for _ in range(16):
                    c = q + r * (U.T @ rng.normal(size=3)) / 1.7
                    d = U.T @ rng.normal(size=3)
                    d = d / np.linalg.norm(d) * max(r / 4, 5e-4)
                    jobs.append((tid, (c - d).tolist(), (c + d).tolist(), (A, B)))
            pts = [p for p in pool.map(_chord, jobs, chunksize=2) if p]
            f = fit(pts) if len(pts) >= 10 else {"kind": "too few points"}
            f["points"] = pts
            res[tid] = f
            print(f"{tid}: {len(pts)} wall points; {f['kind']}; residual {f.get('residual')}; "
                  f"golden {f.get('golden')} (golden residual {f.get('golden_residual')})", flush=True)
            json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
