"""Exact points on a curved wall: at a sample of the wall shape some vertex of the orbit lies exactly in the plane of
a cell (a coincidence that nearby seeds off the wall lack). That vertex's signed distance from that plane, as a
function of beta, vanishes on the whole wall, so its roots along lines crossing the wall are exact wall points.

    from surface_points import coincidence, root_along
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

from cellframe import T
from dodeca_view import E

GROUP = np.array(E, float)
TM = np.array(T, float)


def orbit(beta):
    return np.einsum("gij,j->gi", GROUP, np.asarray(beta, float) @ TM)


def _dist(V, simplex, j):
    P = V[list(simplex)]
    n = np.linalg.svd(P[1:] - P[0])[2][-1]
    if n @ P[0] < 0:
        n = -n
    return float(n @ (V[j] - P[0]) / np.linalg.norm(P[0]))


def coincidence(sample, h=1e-3, seed=0):
    """(simplex, vertex) whose distance is ~0 at the sample but not at nearby points in two random directions."""
    s = np.asarray(sample, float) / np.sum(sample)
    rng = np.random.default_rng(seed)
    d1, d2 = (rng.normal(size=4) for _ in range(2))
    d1 -= d1.mean(); d2 -= d2.mean()
    p1, p2 = s + h * d1 / np.linalg.norm(d1), s + h * d2 / np.linalg.norm(d2)
    V0, V1, V2 = orbit(s), orbit(p1), orbit(p2)
    hull = ConvexHull(V1)
    best = None
    for simplex in hull.simplices:
        for j in range(len(V0)):
            if j in simplex:
                continue
            g0 = _dist(V0, simplex, j)
            if abs(g0) > 1e-11:
                continue
            g1, g2 = _dist(V1, simplex, j), _dist(V2, simplex, j)
            if min(abs(g1), abs(g2)) > 1e-9 and (best is None or abs(g1) < best[0]):
                best = (abs(g1), tuple(int(v) for v in simplex), j)
        if best is not None and best[0] < 1e-3:
            break
    return None if best is None else (best[1], best[2])


def g(beta, co):
    return _dist(orbit(beta), co[0], co[1])


def root_along(p, d, co, span=0.05):
    """The root of the coincidence nearest to p along the line p + t d (None if it does not cross within span)."""
    f = lambda t: g(p + t * d, co)
    ts = np.linspace(-span, span, 41)
    vals = [f(t) for t in ts]
    best = None
    for a, b, fa, fb in zip(ts[:-1], ts[1:], vals[:-1], vals[1:]):
        if fa == 0:
            return p + a * d
        if fa * fb < 0:
            t = brentq(f, a, b, xtol=1e-16, rtol=1e-15)
            if best is None or abs(t) < abs(best):
                best = t
    return None if best is None else p + best * d
