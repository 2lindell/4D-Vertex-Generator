"""Is a line shape with several crossing lines a surface? Find its equation.

All its traced lines lie on a family of quadrics (the null space of the lines' points in the 10 quadratic monomials).
At a point where two of its lines cross, the directions between them in their plane are probed: on a saddle surface
ruled by those lines, the four sectors alternate between the regions on its two sides. Points of the surface off the
lines are then found by bisecting, along the plane's normal, between those two regions; the member of the quadric
family through them is snapped to golden coefficients, and points pushed exactly onto it are classified.

    python surface_from_lines.py X48 out.json
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool

import numpy as np

from exact_conics import snap
from fexact import _lab

PHI = (1 + 5 ** 0.5) / 2
MONS = [(i, j) for i in range(4) for j in range(i, 4)]
POINT_LINES = {"X48": ([1, 1 + PHI, 2 + PHI, 1], [1, 1, 1, 1]), "X51": ([1, 1 + PHI, 2 + 2 * PHI, 1 + PHI], [1, 1, 1, 1])}


def N(v):
    v = np.asarray(v, float)
    return v / v.sum()


def row(x):
    return np.array([x[i] * x[j] for i, j in MONS])


def lines_of(tid):
    out = [tuple(map(N, POINT_LINES[tid]))] if tid in POINT_LINES else []
    for k, v in json.load(open("x_point_lines.json")).items():
        if v.get("id", k) == tid:
            out.append((N(v["a"]), N(v["b"])))
    return out


_G = {}


def _lab_job(x):
    return _lab(N(x))


def _bisect(args):
    x0, n, w, l0 = args
    lo, hi = -w, w
    for _ in range(36):
        m = (lo + hi) / 2
        if _lab(N(x0 + m * n)) == l0:
            lo = m
        else:
            hi = m
    return N(x0 + (lo + hi) / 2 * n).tolist()


def main(tid, out):
    L = lines_of(tid)
    A = np.array([row(a + t * (b - a)) for a, b in L for t in np.linspace(0, 1, 9)])
    sv = np.linalg.svd(A, compute_uv=False)
    V = np.linalg.svd(A)[2][-int((sv < 1e-12).sum()):] if (sv < 1e-12).any() else None
    print(f"{tid}: {len(L)} lines; quadrics through them: {0 if V is None else len(V)}", flush=True)
    # a crossing of two lines
    best = None
    for (a1, b1), (a2, b2) in itertools.combinations(L, 2):
        M = np.vstack([b1 - a1, -(b2 - a2)]).T
        st = np.linalg.lstsq(M, a2 - a1, rcond=None)[0]
        x = a1 + st[0] * (b1 - a1)
        if np.linalg.norm(x - (a2 + st[1] * (b2 - a2))) < 1e-9 and 0 < st[0] < 1 and 0 < st[1] < 1:
            best = (x, (b1 - a1) / np.linalg.norm(b1 - a1), (b2 - a2) / np.linalg.norm(b2 - a2))
            break
    e, d1, d2 = best
    d2 = d2 - (d2 @ d1) * d1
    d2 /= np.linalg.norm(d2)
    n = np.linalg.svd(np.vstack([np.ones(4), d1, d2]))[2][-1]
    print(f"{tid}: crossing at {np.round(e / e.max(), 6).tolist()}", flush=True)
    with Pool(4) as pool:
        ths = np.radians([45, 135, 225, 315])
        sect = pool.map(_lab_job, [e + 1e-4 * (np.cos(t) * d1 + np.sin(t) * d2) for t in ths])
        print(f"{tid}: sectors at 1e-4: {sect}", flush=True)
        if len(set(sect)) != 2 or sect[0] != sect[2] or sect[1] != sect[3]:
            print(f"{tid}: not the alternating pattern of a surface", flush=True)
            return
        jobs = []
        for t, l0 in zip(ths, sect):
            for r in (0.002, 0.006, 0.02):
                jobs.append((e + r * (np.cos(t) * d1 + np.sin(t) * d2), n, 3 * r * r, l0))
        # (each bisection starts from the sector's own region; check the far end is the other region)
        far = pool.map(_lab_job, [x0 + w * nn for x0, nn, w, _ in jobs])
        far2 = pool.map(_lab_job, [x0 - w * nn for x0, nn, w, _ in jobs])
        jobs = [(x0, nn if f2 == l0 else -nn, w, l0) for (x0, nn, w, l0), f, f2 in zip(jobs, far, far2) if (f == l0) != (f2 == l0)]
        pts = pool.map(_bisect, jobs)
        print(f"{tid}: {len(pts)} surface points off the lines", flush=True)
        R = np.array([[row(np.array(x)) @ v for v in V] for x in pts])
        lam = np.linalg.svd(R)[2][-1]
        q = lam @ V
        print(f"{tid}: fit residual {np.abs(R @ lam).max():.1e}", flush=True)
        vals, forms = snap(q, tol=5e-5)
        if vals is None:
            print(f"{tid}: no golden snap: {np.round(q / np.abs(q).max(), 6).tolist()}", flush=True)
            json.dump({"tid": tid, "coeffs": q.tolist(), "points": pts}, open(out, "w"))
            return
        vals = np.array(vals)
        names = ["β1", "β2", "β3", "β4"]
        text = " + ".join(f"({f})·{names[i]}{names[j] if i != j else '²'}" if f not in ("1", "-1") else
                          f"{'-' if f == '-1' else ''}{names[i]}{names[j] if i != j else '²'}"
                          for f, (i, j) in zip(forms, MONS) if f != "0").replace("+ -", "− ") + " = 0"
        on_lines = float(np.abs(A @ vals).max())

        def proj(b):
            b = np.array(b, float)
            for _ in range(30):
                g = np.array([sum(c * (x[j] if i == k else 0) + c * (x[i] if j == k else 0)
                                  for c, (i, j) in zip(vals, MONS) for x in [b]) for k in range(4)])
                g -= g.mean()
                b = b - (row(b) @ vals) / (g @ g) * g
            return N(b)
        exact = [proj(x) for x in pts]
        labs = pool.map(_lab_job, exact)
        print(f"{tid}: {text}  (on the lines to {on_lines:.1e}); surface points read {sorted(set(labs))}", flush=True)
        json.dump({"tid": tid, "coeffs": vals.tolist(), "mons": MONS, "text": text, "sides": sorted(set(sect)),
                   "points": [x.tolist() for x in exact], "labels": labs}, open(out, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
