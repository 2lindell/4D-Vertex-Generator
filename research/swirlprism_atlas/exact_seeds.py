"""The most even seed of a shape, solved exactly (exact_seeds.json).

A shape's polytope is the orbit of its seed p = beta T / |beta T| (T: the cell's vertices, exact in Q(sqrt 5)). Every
edge joins the seed's vertex to an image g p, and edges of one orbit have one length; with |p| = 1 its squared length
is 2 - 2 p.gp, so the ratio of two edges' squared lengths is the ratio of two quadratic forms in the betas,
    q_g(beta) = beta T (I - (g + g^T) / 2) T^T beta^T,
exact in Q(sqrt 5) (the group's elements are, as are T's entries); the normalisation cancels. While the seed stays in
the shape, its edges are the same orbits, so the ratio of its longest edge to its shortest is
    R(beta) = sqrt(max_g q_g(beta) / min_g q_g(beta)),
a function of the betas alone, on the shape's exact locus (free for a region, its surface for a wall).

The most even seed minimises R. Found in three steps:
  1. the edge orbits at a point of the shape (the hull there), each with its group element and its exact form;
  2. the minimum located numerically on the forms alone (no classification inside the search), which says which
     orbits are longest and which shortest there (the active ones: within 1e-7 of the extremes);
  3. solved exactly: the longest active orbits all of one length, the shortest all of one length, the locus, and, where
     these leave freedom, the minimum's own condition (the gradient of max/min lies in the span of the constraints'
     gradients), by Newton's method at 60 digits.
Then checked: the equations hold to 1e-45; the point is the shape (its name and the key of how its cells fit together,
classify_fast.py); every other orbit lies strictly between the shortest and longest; and R rises at every point
of a ring of small steps round it on the locus (from the exact forms). A minimum on the edge of the shape (where the
search leaves it) is reported, not hidden.

    python exact_seeds.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations

import mpmath as mp
import numpy as np
from scipy.optimize import minimize

from cellframe import T as T_FLOAT, seed_from_beta
from classify import ACTION
from classify_fast import _elements
from core import shape_key
from exact import exact_vec
from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

mp.mp.dps = 60
OUT = "exact_seeds.json"
T_EX = exact_vec(T_FLOAT, scale=False)


def edge_orbits(b, hull_tol):
    """The edge orbits at the seed's vertex: for each, a group element g with g p the other end (exact) and its form."""
    p = seed_from_beta(b)
    V = generate_vertices_from_seed(p, ACTION, tol=1e-9)
    faces, _ = compute_convex_hull(V, tol=hull_tol)
    v0 = int(np.argmin(np.linalg.norm(V - p, axis=1)))
    E = _elements()
    forms, seen = [], []
    for e in hull_edges(faces):
        e = tuple(e)
        if v0 not in e:
            continue
        w = e[1] if e[0] == v0 else e[0]
        gi = int(np.argmin(np.linalg.norm(E @ p - V[w], axis=1)))
        g = E[gi]
        M = T_FLOAT @ (np.eye(4) - (g + g.T) / 2) @ T_FLOAT.T
        if any(np.abs(M - m).max() < 1e-9 for m in seen):
            continue
        seen.append(M)
        g_ex = exact_vec(g, scale=False)
        I = mp.eye(4)
        M_ex = T_EX * (I - (g_ex + g_ex.T) / 2) * T_EX.T
        forms.append({"g": gi, "M": M, "M_ex": M_ex})
    return forms


def qf(M, b):
    return float(b @ M @ b)


def qx(M, b):
    return (b.T * M * b)[0]


def basis(normals):
    """An orthonormal basis of the directions along the locus: sum-zero, orthogonal to each of its plane normals."""
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if not len(normals):
        return U
    N = np.array([np.asarray(n, float) for n in normals])
    _, sv, vt = np.linalg.svd(N @ U.T)
    null = vt[int((sv > 1e-9).sum()):]
    return null @ U


def _float_surface(name):
    """A surface's value at a point (betas), floating point: a library surface, a face, or name@g (an image)."""
    import exact as ex
    if "@" in name:
        base, gi = name.rsplit("@", 1)
        s, gi = ex.FLIB[base], int(gi)
        ex.near_images([0.25] * 4, 1e-300)            # (loads the float group)
        GI = ex._GF[1][gi]
        return lambda x: float(np.ravel(s.f((GI @ x)[None]))[0])
    s = ex.FLIB[name]
    return lambda x: float(np.ravel(s.f(np.asarray(x)[None]))[0])


def _exact_surface(name):
    import exact as ex
    if "@" in name:
        base, gi = name.rsplit("@", 1)
        ex.load_group()
        return ex.EImage(ex.SURF[base], int(gi))
    return ex.SURF[name]


def _numeric_min(Ms, b0, locus):
    """The minimum of max q / min q on the locus near b0, floating point (SLSQP on the epigraph: variables the betas,
    log of the longest and log of the shortest)."""
    fs = [_float_surface(n) for n in locus]
    v0 = np.array([qf(M, b0) for M in Ms])
    z0 = np.concatenate([b0, [np.log(v0.max()), np.log(v0.min())]])
    cons = [{"type": "eq", "fun": lambda z: z[:4].sum() - 1}]
    cons += [{"type": "eq", "fun": (lambda z, f=f: f(z[:4]))} for f in fs]
    cons += [{"type": "ineq", "fun": (lambda z, M=M: np.exp(z[4]) - qf(M, z[:4]))} for M in Ms]
    cons += [{"type": "ineq", "fun": (lambda z, M=M: qf(M, z[:4]) - np.exp(z[5]))} for M in Ms]
    r = minimize(lambda z: z[4] - z[5], z0, method="SLSQP", constraints=cons,
                 options={"ftol": 1e-15, "maxiter": 2000})
    return r.x[:4] / r.x[:4].sum()


def _exact_min(forms, b, locus):
    """The exact minimum near b: the active longest and shortest orbits equal among themselves, the locus, and the
    minimum's condition where they leave freedom; Newton at 60 digits. Returns (x, residual, A, B, free)."""
    Ms = [f["M"] for f in forms]
    v = np.array([qf(M, b) for M in Ms])
    hi, lo = v.max(), v.min()
    A = [i for i in range(len(Ms)) if v[i] > hi * (1 - 1e-6)]
    B = [i for i in range(len(Ms)) if v[i] < lo * (1 + 1e-6)]
    S = [_exact_surface(n) for n in locus]
    MA, MB = forms[A[0]]["M_ex"], forms[B[0]]["M_ex"]

    def base(x):
        out = [mp.fsum(x) - 1] + [s.f(x) for s in S]
        out += [qx(forms[i]["M_ex"], x) - qx(MA, x) for i in A[1:]]
        out += [qx(forms[i]["M_ex"], x) - qx(MB, x) for i in B[1:]]
        return out

    def grads(x):
        G = [mp.matrix([1, 1, 1, 1])] + [mp.matrix(s.grad(x)) for s in S]
        G += [2 * (forms[i]["M_ex"] - MA) * x for i in A[1:]]
        G += [2 * (forms[i]["M_ex"] - MB) * x for i in B[1:]]
        return G
    xb = mp.matrix([mp.mpf(float(t)) for t in b])
    nb = len(base(xb))
    free = 4 - nb

    # (everything is homogeneous in the betas, so the sum's multiplier is 0: the condition is taken in the sum-zero
    # directions only (three components), with a multiplier for each other constraint)
    P = mp.matrix([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]])

    def F(*z):
        x = mp.matrix(z[:4])
        out = base(x)
        if free <= 0:
            return out[:4]
        stat = qx(MB, x) * (2 * MA * x) - qx(MA, x) * (2 * MB * x)
        G = grads(x)[1:]
        for k, l_ in enumerate(z[4:]):
            stat = stat - l_ * G[k]
        ps = P * stat
        return out + [ps[i] for i in range(3)]
    z0 = [mp.mpf(float(t)) for t in b]
    if free > 0:
        G = np.array([[float(t) for t in g] for g in grads(xb)[1:]]).T
        Pf = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1.0]])
        stat = qf(Ms[B[0]], b) * (2 * Ms[A[0]] @ b) - qf(Ms[A[0]], b) * (2 * Ms[B[0]] @ b)
        lam = np.linalg.lstsq(Pf @ G, Pf @ stat, rcond=None)[0] if G.size else np.zeros(0)
        z0 += [mp.mpf(float(t)) for t in lam]
    sol = mp.findroot(F, z0, tol=mp.mpf(10) ** -55, maxsteps=200)
    sol = list(sol) if hasattr(sol, "__len__") else [sol]
    x = mp.matrix(sol[:4])
    res = max(abs(r) for r in F(*sol) + base(x))
    return x, res, A, B, max(free, 0)


def _boundary(name, key, b_in, b_out, hull_tol, locus):
    """Where the shape ends between b_in (in it) and b_out (not), to 1e-12 (moved along the locus); and the surfaces
    through that point that are not already in the locus."""
    import exact as ex
    fs = [_float_surface(n) for n in locus]

    def on(t):
        """The point a fraction t of the way, put back on the locus by least-change Newton steps (floating point)."""
        b = (1 - t) * b_in + t * b_out
        for _ in range(30):
            if not fs:
                break
            F_ = np.array([f(b) for f in fs] + [b.sum() - 1])
            if np.abs(F_).max() < 1e-15:
                break
            J = np.array([[(f(b + 1e-7 * e) - f(b - 1e-7 * e)) / 2e-7 for e in np.eye(4)] for f in fs] + [[1, 1, 1, 1.0]])
            b = b - J.T @ np.linalg.solve(J @ J.T, F_)
        return b
    lo_, hi_ = 0.0, 1.0
    for _ in range(45):
        m = (lo_ + hi_) / 2
        if shape_key(on(m), hull_tol) == (name, key):
            lo_ = m
        else:
            hi_ = m
    bb = on(lo_)
    cands = [n for n in ex.near(bb, 1e-7) if n not in locus]
    if not cands:
        cands = [f"{n}@{gi}" for n, gi in ex.near_images(bb, 1e-7)]
        cands = [c for c in cands if c not in locus]
    # (the one closest to the point)
    dist = lambda n: abs(_float_surface(n)(bb))
    cands.sort(key=dist)
    return bb, on(lo_ * 0.999), cands


def solve(name, key, start, locus=(), hull_tol=1e-9, rounds=6):
    b_in = np.asarray(start, float)
    b_in = b_in / b_in.sum()
    assert shape_key(b_in, hull_tol) == (name, key), "the start is not the shape"
    forms = edge_orbits(b_in, hull_tol)
    Ms = [f["M"] for f in forms]
    locus = list(locus)
    history = []
    for _ in range(rounds):
        bn = _numeric_min(Ms, b_in, locus)
        x, res, A, B, free = _exact_min(forms, bn, locus)
        bx = np.array([float(t) for t in x])
        inside = shape_key(bx, hull_tol) == (name, key)
        history.append({"locus": list(locus), "inside": inside, "ratio": float(mp.sqrt(max(qx(f['M_ex'], x) for f in forms) / min(qx(f['M_ex'], x) for f in forms)))})
        if inside:
            break
        bb, b_in, cands = _boundary(name, key, b_in, bx, hull_tol, locus)
        if not cands:
            history.append({"stuck": "no surface found at the boundary", "at": bb.tolist()})
            break
        locus.append(cands[0])
        history[-1]["boundary"] = cands[0]
    # (checks)
    vals = [qx(f["M_ex"], x) for f in forms]
    hiX, loX = max(vals), min(vals)
    Rx = mp.sqrt(hiX / loX)
    tiny = mp.mpf(10) ** -45
    others = all(loX + tiny < v_ < hiX - tiny for i, v_ in enumerate(vals) if i not in A and i not in B)
    # (every nearby point that is still the shape has a higher ratio: steps all round, in the full 3-space)
    rng = np.random.default_rng(0)
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    worse_in, tried_in = 0, 0
    for h in (1e-7, 1e-5, 1e-3):
        for d in rng.normal(size=(60, 3)):
            bb = bx + h * (d / np.linalg.norm(d)) @ U
            if shape_key(bb, hull_tol) != (name, key):
                continue
            tried_in += 1
            vv = [qf(M, bb) for M in Ms]
            if np.sqrt(max(vv) / min(vv)) < float(Rx) * (1 - 1e-12):
                worse_in += 1
    return {"name": name, "key": key, "betas": [mp.nstr(t, 50) for t in x], "betas_float": bx.tolist(),
            "seed": seed_from_beta(bx).tolist(), "edge_ratio": mp.nstr(Rx, 40), "orbits": len(forms),
            "longest": A, "shortest": B, "locus": locus, "free": free, "residual": mp.nstr(res, 5), "rounds": history,
            "checks": {"equations_hold": bool(res < tiny), "others_strictly_between": others,
                       "is_the_shape": bool(shape_key(bx, hull_tol) == (name, key)),
                       "nearby_points_in_the_shape_tried": tried_in, "nearby_points_in_the_shape_lower": worse_in}}


if __name__ == "__main__":
    print("use from a script", file=sys.stderr)


def corner_infimum(name, key, inner, corner, normals=(), hull_tol=1e-9, n_dir=400, radii=(1e-6, 1e-5, 1e-4)):
    """The shape's lowest edge ratio where it is reached at a corner of the shape (a point it closes on, not in it).

    The forms are the shape's own (its edge orbits at `inner`, a point of it); the ratio at the corner (exact
    betas, 40 digits) is the limit of the shape's ratio there. It is the minimum over the shape's closure when the ratio
    rises along every direction from the corner into the shape: checked to first order exactly (the derivative of
    log max q - log min q, from the forms' exact gradients at the corner, the extremes taken over the orbits active
    there) over evenly spread directions that enter the shape (classified a small step along each), and by the ratio
    itself at those steps."""
    import bounds_walk as bw
    forms = edge_orbits(np.asarray(inner, float) / np.sum(inner), hull_tol)
    x = mp.matrix([mp.mpf(t) for t in corner])
    x = x / mp.fsum(x)
    xf = np.array([float(t) for t in x])
    vals = [qx(f["M_ex"], x) for f in forms]
    hi, lo = max(vals), min(vals)
    tiny = mp.mpf(10) ** -30
    A = [i for i, v in enumerate(vals) if hi - v < tiny]
    B = [i for i, v in enumerate(vals) if v - lo < tiny]
    R = mp.sqrt(hi / lo)
    gA = [np.array([float(t) for t in (2 * forms[i]["M_ex"] * x)]) / float(hi) for i in A]
    gB = [np.array([float(t) for t in (2 * forms[i]["M_ex"] * x)]) / float(lo) for i in B]
    Ub = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if normals:
        N = np.array([np.asarray(v, float) for v in normals])
        _, sv, vt = np.linalg.svd(N @ Ub.T)
        Ub = vt[int((sv > 1e-9).sum()):] @ Ub
    D = bw.directions(Ub, n_dir)
    entering, rises, slope_min, lower = 0, 0, np.inf, 0
    for d in D:
        ins = [r for r in radii if shape_key(xf + r * d, hull_tol) == (name, key)]
        if not ins:
            continue
        entering += 1
        slope = max(g @ d for g in gA) - min(g @ d for g in gB)         # (d/dt of log(max q / min q) at the corner)
        slope_min = min(slope_min, slope)
        rises += slope > 0
        for r in ins:
            b = xf + r * d
            vv = [qf(f["M"], b) for f in forms]
            if np.sqrt(max(vv) / min(vv)) < float(R) * (1 - 1e-13):
                lower += 1
    return {"name": name, "key": key, "corner": [mp.nstr(t, 40) for t in x], "edge_ratio_at_corner": mp.nstr(R, 40),
            "orbits": len(forms), "longest_there": A, "shortest_there": B,
            "checks": {"directions_entering_the_shape": entering, "of_them_ratio_rising_first_order": rises,
                       "smallest_first_order_slope": float(slope_min), "points_in_the_shape_with_a_lower_ratio": lower}}
