"""Exact wall forms of the contic domain (as decaic_atlas/forms.py, over Q(sqrt2)).

Each vertex of a seed's polychoron is g x(beta) = (g X) beta for a group element g, with g X exact in Q(sqrt2). For four
vertices spanning a cell's 3-flat and one vertex of a neighbouring cell, det[v_b - v_a, .., v_e - v_a] is a quartic form
in beta over Q(sqrt2), computed symbolically and factored over Q(sqrt2). One face per class of faces (the group carries
the rest). On a mirror (some beta_k = 0) the form is restricted there before factoring.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

import group as G
from qsqrt2 import K, from_sympy

B = sp.symbols("b1 b2 b3 b4")
BV = sp.Matrix(B)


@lru_cache(maxsize=None)
def coeff_matrix(g):
    """g X exactly (vertex = this times beta)."""
    return G.exact(G.ELEMENTS[g] @ G.X)


def form(labels5):
    C = [coeff_matrix(int(g)) * BV for g in labels5]
    M = sp.Matrix([[sp.expand(C[k][i] - C[0][i]) for i in range(4)] for k in range(1, 5)])
    return sp.expand(M.det(method="berkowitz"))


@lru_cache(maxsize=None)
def _factor(expr_text):
    f = sp.sympify(expr_text)
    return [fac for fac, _ in sp.factor_list(f, *B, extension=sp.sqrt(2))[1] if fac.free_symbols]


class NonLinear(ValueError):
    pass


def linear(fac):
    """A linear factor as a normalized tuple of K (first nonzero coefficient 1)."""
    p = sp.Poly(fac, *B)
    if p.total_degree() != 1 or p.coeff_monomial(1) != 0:
        raise NonLinear(str(fac))
    c = [from_sympy(p.coeff_monomial(v)) for v in B]
    first = next(x for x in c if x != K(0))
    return tuple(x / first for x in c)


def region_forms(beta):
    """The forms of the seed's polychoron, one per face class and side, restricted to the seed's mirrors."""
    info = G.classify(beta)
    V, idx, faces, cells, perms, cc, fc = info["_hull"]
    Vn = V / np.linalg.norm(V[0])
    cv = [sorted({v for f in c for v in faces[f]}) for c in cells]
    face_cells = {}
    for ci, c in enumerate(cells):
        for f in c:
            face_cells.setdefault(f, []).append(ci)
    zero = {v: 0 for v, b in zip(B, beta) if abs(b) < 1e-12}
    out = []
    for cl in fc:
        f = cl[0]
        cs = face_cells[f]
        if len(cs) != 2:
            continue
        for A, Bc in ((cs[0], cs[1]), (cs[1], cs[0])):
            pts = cv[A]
            base = [pts[0]]
            for v in pts[1:]:
                if np.linalg.matrix_rank(np.vstack([Vn[u] - Vn[base[0]] for u in base[1:]] + [Vn[v] - Vn[base[0]]]), 1e-9) == len(base):
                    base.append(v)
                if len(base) == 4:
                    break
            off = next(v for v in cv[Bc] if v not in set(pts))
            fx = form([idx[i] for i in base + [off]])
            if zero:
                fx = sp.expand(fx.subs(zero))
            if fx != 0:
                out.append(fx)
    return out


def candidates(beta, log=print):
    """Mixed-sign linear factors (planes crossing the open chamber) of the seed's forms."""
    out = set()
    for fx in region_forms(beta):
        for fac in _factor(str(fx)):
            try:
                c = linear(fac)
            except NonLinear:
                log(f"  NONLINEAR at {np.round(beta, 6).tolist()}: {fac}")
                continue
            s = [x.sign() for x in c]
            if 1 in s and -1 in s:
                out.add(c)
    return out
