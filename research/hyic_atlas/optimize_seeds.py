"""The decaic optimiser (decaic_atlas/optimize_seeds.py) and centre finder (decaic_atlas/central_seeds.py) run on the
hyic family's groups (HYIC_SUB), as the contic atlas runs them: the same objective (R = longest / shortest edge on the
shape's locus while its sign conditions hold) with this atlas's edge forms (an edge from x to g x has squared length
beta^T M^T M beta, M = X - g X), sign conditions (forms.region_forms over Q(sqrt5), on the quartic monomials) and
shape test (the arrangement key). Imported by ixitic_offs.py.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import forms as F  # noqa: E402
import group as G  # noqa: E402

sys.path.insert(1, os.path.join(ROOT, "decaic_atlas"))
G.W5 = np.zeros((4, 5))                     # (read at the decaic optimiser's import; not used here)
_spec = importlib.util.spec_from_file_location("decaic_optimize", os.path.join(ROOT, "decaic_atlas", "optimize_seeds.py"))
O = importlib.util.module_from_spec(_spec)
sys.modules["decaic_optimize"] = O
_spec.loader.exec_module(O)
assert O.G is G and O.F is F
_cspec = importlib.util.spec_from_file_location("decaic_central", os.path.join(ROOT, "decaic_atlas", "central_seeds.py"))
CS = importlib.util.module_from_spec(_cspec)
sys.modules["decaic_central"] = CS
_cspec.loader.exec_module(CS)

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402

B = O.B
F.MONS = [m for m in itertools.product(range(5), repeat=4) if sum(m) == 4]
F.poly = lambda e: e


def edge_forms(seed, hull_tol):
    V, Vn, idx, faces, cells = G.polytope(seed, hull_tol)
    Qs = {}
    for e in hull_edges(faces):
        e = tuple(e)
        if 0 not in e:
            continue
        w = e[1] if e[0] == 0 else e[0]
        M = G.X - G.ELEMENTS[idx[w]] @ G.X
        Q = M.T @ M
        Qs[tuple(np.round(Q, 9).ravel())] = Q
    return list(Qs.values())


def sign_conditions(seed):
    out = []
    for fx in F.region_forms(seed):
        p = sp.Poly(sp.expand(fx), *B)
        co = np.array([float(p.coeff_monomial(sp.prod([v ** k for v, k in zip(B, m)]))) for m in F.MONS])
        scale = np.abs(co).sum()
        if scale == 0:
            continue
        v = float(fx.subs(dict(zip(B, seed))))
        if abs(v) > 1e-9 * scale:
            out.append((co / scale, int(np.sign(v)), sp.expand(fx)))
    return out


def same_shape(b, seed, key0, hull_tol):
    try:
        return G.label(b, hull_tol)[1] == key0
    except Exception:
        return False


O.edge_forms, O.sign_conditions, O.same_shape = edge_forms, sign_conditions, same_shape
CS.O, CS.G, CS.F = O, G, F
