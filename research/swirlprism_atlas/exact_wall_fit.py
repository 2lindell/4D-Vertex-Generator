"""The equation of a curved wall from exact points: at a sample of the wall, the coincidence that makes it a wall (a
vertex exactly in a cell's plane, surface_points.coincidence) is found; its roots along many lines crossing the wall
near the sample are exact points of the wall's algebraic surface, which are fitted by the lowest-degree polynomial
that vanishes on them all, snapped to golden coefficients.

    python exact_wall_fit.py out.json X42:b1,b2,b3,b4 [X53:...]
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool

import numpy as np

from exact_conics import snap
from surface_points import coincidence, root_along

_CO = None


def _root(args):
    p, d, co = args
    r = root_along(np.array(p), np.array(d), co, span=0.03)
    return None if r is None else (r / r.sum()).tolist()


def main(out, specs, n=120, radius=0.06):
    res = {}
    rng = np.random.default_rng(3)
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    for spec in specs:
        tid, beta = spec.split(":")
        s = np.array([float(v) for v in beta.split(",")])
        s /= s.sum()
        co = coincidence(s)
        if co is None:
            print(f"{tid}: no coincidence found at the sample", flush=True)
            continue
        jobs = []
        for _ in range(n):
            p = s + radius * (U.T @ rng.normal(size=3)) * rng.uniform(0.2, 1)
            d = U.T @ rng.normal(size=3)
            jobs.append((p.tolist(), (d / np.linalg.norm(d)).tolist(), co))
        with Pool(4) as pool:
            pts = [q for q in pool.map(_root, jobs) if q]
        P = np.array(pts)
        r = {"coincidence": [list(co[0]), co[1]], "points": pts}
        for deg in (1, 2, 3, 4):
            mons = list(itertools.combinations_with_replacement(range(4), deg))
            if len(mons) >= len(P) - 5:
                break
            A = np.array([[np.prod(x[list(m)]) for m in mons] for x in P])
            sv = np.linalg.svd(A, compute_uv=False)
            if sv[-1] < 1e-10 * sv[0] and sv[-2] > 1e-6 * sv[0]:
                q = np.linalg.svd(A)[2][-1]
                vals, forms = snap(q, tol=1e-6)
                r.update({"degree": deg, "mons": [list(m) for m in mons], "coeffs": q.tolist(), "golden": forms,
                          "residual": float(np.abs(A @ q).max())})
                break
        res[tid] = r
        print(f"{tid}: {len(P)} exact points; degree {r.get('degree')}; golden {r.get('golden')}", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
