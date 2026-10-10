"""Refine boundary points between two labels of a plane map and fit an exact conic to them.

A conic here is a homogeneous quadratic form in the barycentric weights restricted to the plane;
coefficients are matched to a + b*phi when possible.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
from cellframe import golden_form
from probe import _one
from t2bound import plane_basis


def refine(pt, pair, normals, width=2e-6, steps=34):
    """Re-bisect a boundary point to double precision, crossing along an in-plane direction."""
    p = np.asarray(pt, float); p = p / p.sum()
    for d in plane_basis(normals):
        lo, hi = p - width * d, p + width * d
        la, lb = _one(lo)[0], _one(hi)[0]
        if {la, lb} != set(pair):
            continue
        for _ in range(steps):
            m = (lo + hi) / 2
            lab = _one(m)[0]
            if lab == la:
                lo = m
            elif lab == lb:
                hi = m
            else:
                break
        q = (lo + hi) / 2
        return q / q.sum()
    return None


def fit(points, coords):
    """Fit a quadratic form in the given coordinate indices; returns (coefficients, names, residual)."""
    P = np.asarray(points)[:, coords]
    terms = list(itertools.combinations_with_replacement(range(len(coords)), 2))
    A = np.stack([P[:, i] * P[:, j] for i, j in terms], axis=1)
    _, s, vt = np.linalg.svd(A)
    c = vt[-1]
    c = c / c[np.argmax(np.abs(c) > 1e-3)]
    names = [f"b{coords[i] + 1}·b{coords[j] + 1}" for i, j in terms]
    return c, names, float(np.abs(A @ c).max()), s


if __name__ == "__main__":
    plane_file, lab1, lab2, normals, coords, out = sys.argv[1:7]
    D = json.load(open(plane_file))
    normals = json.loads(normals)
    coords = json.loads(coords)
    pts = [b["pt"] for b in D["boundary"] if sorted(b["between"]) == sorted([lab1, lab2]) and not b["third"]]
    uniq = []
    for p in pts:
        if not any(np.allclose(p, u, atol=1e-9) for u in uniq):
            uniq.append(p)
    ref = [r for r in (refine(p, (lab1, lab2), normals) for p in uniq) if r is not None]
    json.dump([r.tolist() for r in ref], open(out, "w"))
    c, names, res, s = fit(ref, coords)
    print(f"{len(ref)} points; singular values {np.array2string(s, precision=2)}; residual {res:.1e}")
    print("  ", ", ".join(f"{n}: {golden_form(v, maxc=12)}" for n, v in zip(names, c)))
