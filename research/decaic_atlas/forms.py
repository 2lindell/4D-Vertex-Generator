"""Exact wall forms of the decaic domain: where a seed's polychoron changes.

A polychoron changes only where one of its cells stops being supporting: a vertex of a neighbouring cell reaches the
cell's 3-flat. Every vertex is a signed rearrangement eps sigma(x(beta)) of the seed, x linear in beta with integer
coefficients (group.W5), so for four vertices of a cell and one of its neighbour's,

    f(beta) = det[v_b - v_a, v_c - v_a, v_d - v_a, v_e - v_a]      (in the first four coordinates of R^5: an
                                                                     isomorphism of the sum-zero hyperplane)

is a quartic form in beta with integer coefficients. It is found from its values at 35 integer points (an exact
interpolation: the values are integers), checked at more integer points with exact integer arithmetic, reduced
(content, sign) and factored (sympy). A region is the set where every such form of its polychoron keeps the sign it
has at the region's seed; a wall is a factor that vanishes where a region ends.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

import group as G

B = sp.symbols("b1 b2 b3 b4")
MONS = [m for m in itertools.product(range(5), repeat=4) if sum(m) == 4]
_PTS = None


def _points():
    global _PTS
    if _PTS is None:
        rng = np.random.default_rng(5)
        while True:
            P = rng.integers(1, 9, size=(len(MONS), 4))
            M = np.prod(P[:, None, :].astype(float) ** np.array(MONS)[None], axis=2)
            if abs(np.linalg.det(M)) > 1e-3:
                _PTS = (P, M)
                break
    return _PTS


def vertex_labels(beta):
    """For each vertex of the polychoron (group.polytope order): (sign, permutation) with v = sign * x[perm]."""
    V5 = G.vertices5(beta)
    x = G.x5(beta)
    out = []
    for v in V5:
        for s in (1, -1):
            for p in G.PERMS:
                if np.abs(s * x[p] - v).max() < 1e-9 * max(1, np.abs(x).max()):
                    out.append((s, tuple(p)))
                    break
            else:
                continue
            break
    return out


def _vert_int(lbl, b):
    s, p = lbl
    x = np.asarray(b, np.int64) @ G.W5.astype(np.int64)
    return [int(s * x[i]) for i in p]


def _det_int(rows):
    # (integer entries of a few hundred at most: the float determinant is exact after rounding, |det| < 1e12)
    return int(round(np.linalg.det(np.array(rows, float))))


def form_of(labels5):
    """The quartic form (integer coefficients on MONS) of five vertices given by their labels."""
    P, M = _points()
    vals = []
    for b in P:
        V = [_vert_int(l, b) for l in labels5]
        rows = [[V[k][i] - V[0][i] for i in range(4)] for k in range(1, 5)]
        vals.append(_det_int(rows))
    c = np.linalg.solve(M, np.array(vals, float))
    c = np.round(c).astype(np.int64)
    # (checked exactly at more integer points)
    rng = np.random.default_rng(11)
    for b in rng.integers(-6, 7, size=(6, 4)):
        V = [_vert_int(l, b) for l in labels5]
        rows = [[V[k][i] - V[0][i] for i in range(4)] for k in range(1, 5)]
        exact = _det_int(rows)
        got = int(sum(int(ci) * int(np.prod([int(bb) ** e for bb, e in zip(b, m)])) for ci, m in zip(c, MONS)))
        if exact != got:
            raise ValueError("interpolation not exact")
    g = np.gcd.reduce(np.abs(c[c != 0])) if (c != 0).any() else 1
    c = c // g
    first = c[np.flatnonzero(c)[0]] if (c != 0).any() else 1
    return tuple(int(v) for v in (c if first > 0 else -c))


def poly(coeffs):
    return sum(int(c) * sp.prod([v ** e for v, e in zip(B, m)]) for c, m in zip(coeffs, MONS) if c)


@lru_cache(maxsize=None)
def factors(coeffs):
    """The distinct irreducible factors of a form, each as text (sign fixed by its first term)."""
    f = sp.factor_list(sp.expand(poly(coeffs)))[1]
    out = []
    for fac, _ in f:
        if fac.free_symbols:
            fac = sp.expand(fac)
            lead = sp.Poly(fac, *B).coeffs()[0]
            if lead < 0:
                fac = -fac
            out.append(str(fac))
    return tuple(sorted(set(out)))


def evaluate(coeffs, Bt):
    """Values of a form at betas (n, 4)."""
    Bt = np.atleast_2d(np.asarray(Bt, float))
    return np.prod(Bt[:, None, :] ** np.array(MONS)[None], axis=2) @ np.asarray(coeffs, float)


def region_conditions(beta):
    """The forms of the seed's polychoron: for each pair of neighbouring cells, four vertices of one spanning its 3-flat
    and a vertex of the other off it; each with its sign at the seed. Distinct forms only."""
    V5, V, faces, cells = G.polytope(beta)
    labels = vertex_labels(beta)
    cv = [sorted({v for f in c for v in faces[f]}) for c in cells]
    face_cells = {}
    for ci, c in enumerate(cells):
        for f in c:
            face_cells.setdefault(f, []).append(ci)
    out = {}
    for f, cs in face_cells.items():
        if len(cs) != 2:
            continue
        a, b = cs
        for A, Bc in ((a, b), (b, a)):
            pts = cv[A]
            # (four of A's vertices spanning its 3-flat)
            base = [pts[0]]
            for v in pts[1:]:
                M = V[base[1:] + [v]] - V[base[0]] if len(base) > 1 else (V[[v]] - V[base[0]])
                if np.linalg.matrix_rank(np.vstack([V[u] - V[base[0]] for u in base[1:]] + [V[v] - V[base[0]]]), 1e-9) == len(base):
                    base.append(v)
                if len(base) == 4:
                    break
            off = next(v for v in cv[Bc] if v not in set(pts))
            co = form_of([labels[i] for i in base + [off]])
            if not any(co):
                continue
            val = evaluate(co, beta)[0]
            out[co] = int(np.sign(val))
    return out
