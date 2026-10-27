"""The core every step of the pipeline uses, and nothing else: no step imports the old atlas, page or region modules.

  * the cell (cellframe: its four corners), betas and seeds; the fold and the half-cell;
  * the symmetry group: the swirlprism group G (the generator library) and the half-turn Q that normalizes it
    (normalizer_q.npy), so the group with Q, as maps of the betas;
  * the classifier: a seed's polytope (its orbit's hull, classify_fast) and its name: the wiki's list (catalog.py,
    the table below), the reference shapes (references.json) and the X names (atlas_xids.json), naming tables only;
  * the exact surfaces the walls lie on (surfaces.json);
  * the grid the walls are meshed on (marching tetrahedra), and text forms of betas and seeds.
"""
from __future__ import annotations

import json
from itertools import permutations

import numpy as np

from catalog import match, parse_signature
from cellframe import PHI, T, TINV, golden_form, seed_from_beta
from classify import ACTION, signature
from classify_fast import classify
from four_d_vertex_generator.generation import group_elements

FOLD = np.eye(4)[[1, 0, 3, 2]]
HALF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def in_half(x, tol=1e-12):
    x = np.asarray(x, float)
    return bool((x > -tol).all() and x[2] >= x[3] - tol)


def to_upper(beta):
    """A seed moved into the displayed half-cell (beta3 >= beta4) with the exact half-turn (not normalised)."""
    b = np.asarray(beta, float)
    return b[[1, 0, 3, 2]] if b[2] < b[3] - 1e-12 else b


def home(b):
    """Normalised and moved into the half-cell (beta3 >= beta4) by the fold, which keeps every shape."""
    b = np.asarray(b, float)
    b = b / b.sum()
    b = np.where(np.abs(b) < 1e-15, 0.0, b)              # (on a face of the half-cell: exactly on it)
    return b @ FOLD.T if b[2] < b[3] else b


# ------------------------------------------------------------------------------------------------------------ group

def group():
    """The swirlprism group G as maps of the betas (1200)."""
    E = np.stack(group_elements(ACTION))
    return np.einsum("ij,gjk,kl->gil", TINV, E, np.array(T, float).T)


def extended():
    """G and its coset G.Q (Q the half-turn normalizing G, normalizer_q.npy) as maps of the betas (2400)."""
    E = np.stack(group_elements(ACTION))
    E = np.concatenate([E, E @ np.load("normalizer_q.npy")])
    return np.einsum("ij,gjk,kl->gil", TINV, E, np.array(T, float).T)


# ------------------------------------------------------------------------------------------------------- classifier

CANON = [
    ("A1", "Hexacosichoron (600-cell)", ("ref", "Hexacosichoron (600-cell)")),
    ("C1", "Hecatonicosachoron (120-cell)", ("ref", "Hecatonicosachoron (120-cell)")),
    ("B1", "Icosafold icosaswirlchoron", ("ref", "Icosafold icosaswirlchoron (main ring, 240)")),
    ("C2a", "Subsymmetrical icosafold icosidodecaswirlchoron", ("ref", "Cross ring: antiprisms split 120+120 (not in the wiki list)")),
    ("C2b", "Subsymmetrical icosafold icosidodecaswirlchoron", ("ref", "Cross ring: tetrahedra split 600+600 (not in the wiki list)")),
    ("C3", "Subsymmetrical pentagonal-gyroprismatic triacosihexecontachoron", ("ref", "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron")),
    ("D1", "Polychoron with 120+120+1200 cells (600+1200 4-valent edges)", ("ref", "Cross ring: 120+120+1200 cells (600+1200 4-valent edges)")),
    ("D2", "Polychoron with 120+600+600+600+1200 cells (600 3-valent edges)", ("ref", "Cross ring: 120+600+600+600+1200 cells (600 3-valent edges)")),
    ("D3", "Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges)", ("ref", "Cross ring: 120+600+600+600+1200 cells (1200 3-valent edges)")),
    ("C4", "Bigyroprismatic transitional didecafold icosidodecaswirlchoron", ("ref", "Bigyroprismatic transitional didecafold icosidodecaswirlchoron")),
    ("C5", "Partially-rectified small swirlprism", ("ref", "Partially-rectified small swirlprism")),
    ("C6", "Swirlprismatodiminished rectified hexacosichoron", ("ref", "Swirlprismatodiminished rectified hexacosichoron")),
    ("F1", "Polychoron with 120+120+600+600+600+600+1200 cells", ("wiki", "W1")),
    ("F2", "Polychoron with 120+120+240+600+600+1200+1200 cells", ("wiki", "W2")),
    ("F3", "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+1200×9 faces)", ("wiki", "W3")),
    ("F4", "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+600+1200×9 faces)", ("wiki", "W4")),
    ("F5", "Polychoron with 120+120+600+600+600+600+1200+1200+1200 cells", ("wiki", "W5")),
    ("E1", "Bi-hecatonicosadiminished truncated hexacosichoron", ("counts", "120+120+600|240+600+600+600+1200|600+600+600+600+1200")),
    ("E2", "Swirlprismatodiminished truncated hecatonicosachoron", ("counts", "120+120|240+600+600|600+600+1200")),
    ("T1", "Transitional polychoron with 120+120+600+600 cells", ("wiki", "T1")),
    ("T2", "Transitional polychoron with 120+120+600+600+1200 cells", ("wiki", "T2")),
]

CANON_VALENCE = {"F2": "3:2400,4:3000,6:1200,8:600", "T2": "3:2400,4:3000,6:600",
                 "F5": "3:2400,4:3000,5:1200,6:1200,8:600,10:600"}


def counts_key(sig):
    p = parse_signature(sig)
    return None if p is None else "|".join("+".join(map(str, x)) for x in p[1:])


REFS = json.load(open("references.json"))               # signature (with valences) -> reference name
XIDS = json.load(open("atlas_xids.json"))               # x-key -> X name


def identify(sig):
    """(canonical id or None, x-key, reference name or None)."""
    if sig.startswith("ERR"):
        return None, "ERR", None
    ref = REFS.get(sig)
    val = sig.split(" | val ")[-1].strip() if " | val " in sig else None
    for cid, _, (kind, arg) in CANON:
        if kind == "ref" and ref == arg:
            return cid, None, ref
        if kind in ("wiki", "counts") and val and CANON_VALENCE.get(cid, val) != val:
            continue
        if kind == "wiki" and match(sig) == arg:
            return cid, None, ref
        if kind == "counts" and counts_key(sig) == arg:
            return cid, None, ref
    return None, (sig if ref else counts_key(sig) + "|" + sig.split(" | val")[-1]), ref


def label_of_sig(sig):
    cid, xkey, _ = identify(sig)
    return cid or XIDS.get(xkey, "new:" + sig)


def _label(b, hull_tol=1e-9):
    try:
        info = classify(seed_from_beta(np.asarray(b, float) / np.sum(b)), hull_tol=hull_tol)
        sig = signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    except Exception as exc:  # degenerate hulls
        return "ERR", repr(exc)
    return label_of_sig(sig), sig


def label(b):
    """The shape a seed (betas) makes: (name, signature)."""
    return _label(b)


def lab_edge(b):
    """The shape at a point on a wall or line, the hull tolerance relaxed (nearly coplanar faces kept together)."""
    return _label(b, 1e-7)[0]


# --------------------------------------------------------------------------------------------------------- surfaces

class Plane:
    def __init__(self, n, name):
        self.n, self.name = np.asarray(n, float), name

    def f(self, B):
        return np.atleast_2d(B) @ self.n

    def grad(self, b):
        return self.n


class Curved:
    """A polynomial in the betas, taken at A beta."""

    def __init__(self, name, mons, coeffs, A):
        self.name, self.mons, self.c, self.A = name, [list(m) for m in mons], np.asarray(coeffs, float), np.asarray(A, float)

    def f(self, B):
        Y = np.atleast_2d(B) @ self.A.T
        return sum(c * np.prod(Y[:, m], axis=1) for c, m in zip(self.c, self.mons))

    def grad(self, b, e=1e-7):
        b = np.asarray(b, float)
        return np.array([(self.f(b + e * np.eye(4)[k])[0] - self.f(b - e * np.eye(4)[k])[0]) / (2 * e)
                         for k in range(4)])


def library():
    """The exact surfaces (surfaces.json): (planes, curved)."""
    S = json.load(open("surfaces.json"))
    return ([Plane(p["n"], p["name"]) for p in S["planes"]],
            [Curved(c["name"], c["mons"], c["coeffs"], c["A"]) for c in S["curved"]])


# ------------------------------------------------------------------------------------------------------------- mesh

U = np.linalg.svd(np.ones((1, 4)))[2][1:]            # sum-zero directions: beta = C0 + U^T y
C0 = np.full(4, 0.25)
CUBE = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
TETS = []                                            # (the cube cut into six tetrahedra along its main diagonal)
for perm in permutations(range(3)):
    path = [np.zeros(3, int)]
    for ax in perm:
        nxt = path[-1].copy()
        nxt[ax] = 1
        path.append(nxt)
    TETS.append([int(np.flatnonzero((CUBE == p).all(axis=1))[0]) for p in path])


# ------------------------------------------------------------------------------------------------------------- text

_s = 0.34
_d = _s / np.sqrt(2)
CHART = np.array([[_d, _s, 0.0], [_d, -_s, 0.0], [-_d, 0.0, _s], [-_d, 0.0, -_s]])   # (the old cell chart)


def beta_text(beta):
    b = np.asarray(beta, float)
    nz = b[b > 1e-9]
    bb = b / nz.min()
    return "(" + ", ".join(golden_form(x) for x in bb) + ")"


def fmt17(c):
    c = float(c)
    if abs(c) < 1e-13:
        return "0"
    s = f"{c:.17g}"
    return f"{c:.20f}".rstrip("0") if "e" in s else s


def seed_text(beta):
    return ", ".join(fmt17(c) for c in seed_from_beta(beta))
