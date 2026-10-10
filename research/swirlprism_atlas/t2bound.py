"""Exact T2 boundaries: radial bisection from an interior T2 point inside each plane that holds T2 samples."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from probe import _one

STEPS = 18


def plane_basis(normals: list[np.ndarray]) -> np.ndarray:
    """Orthonormal directions in the barycentric simplex (sum 0) orthogonal to the given plane normals."""
    M = np.vstack([np.ones(4)] + [np.asarray(n, float) for n in normals])
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > 1e-12))
    return vt[rank:]


def ray(args):
    c, d, target = args
    c, d = np.asarray(c), np.asarray(d)
    neg = d < -1e-15
    tmax = float(np.min(-c[neg] / d[neg])) if neg.any() else 1.0   # stay inside the cell
    lab_end = _one(c + tmax * d)[0]
    if lab_end == target:
        return tmax, tmax, target, "cell boundary"
    lo, hi = 0.0, tmax
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        if _one(c + mid * d)[0] == target:
            lo = mid
        else:
            hi = mid
    return lo, tmax, target, _one(c + hi * d)[0]


def trace(center, normals, n_dirs=36, target="T2", procs=4):
    c = np.asarray(center, float); c = c / c.sum()
    basis = plane_basis(normals)
    if len(basis) == 1:
        dirs = [basis[0], -basis[0]]
    else:
        dirs = [np.cos(a) * basis[0] + np.sin(a) * basis[1] for a in np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)]
    with Pool(procs) as pool:
        res = pool.map(ray, [(c, d, target) for d in dirs])
    return [{"dir": d.tolist(), "t": t, "tmax": tm, "beyond": beyond, "point": (c + t * d).tolist()}
            for d, (t, tm, _, beyond) in zip(dirs, res)]


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    out = {}
    for name, item in spec.items():
        out[name] = {"center": item["center"], "normals": item["normals"],
                     "rays": trace(item["center"], item["normals"], item.get("dirs", 36))}
        print(name, "done", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
