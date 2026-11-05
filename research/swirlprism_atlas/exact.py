"""Exact arithmetic for the atlas (no rounding).

Every number the atlas is built on lies in Q(sqrt 5): each plane's normal and each curved surface's coefficients and
matrix (surfaces.json), and each symmetry of the 2400-element group as a map of the betas, up to a scale that does not
change the map. They are recognised here from their stored values as (a + b sqrt 5) / c with small integers, checked
against those values, and kept that way. Points are solved on them with mpmath at 60 digits, so a meeting point is
found to about 50 digits and two names for the same point agree to that precision.

    from exact import SURF, GROUP, solve, through
"""
from __future__ import annotations

import json
from functools import lru_cache

import mpmath as mp
import numpy as np

mp.mp.dps = 60
SQ5 = mp.sqrt(5)
_s5 = 5 ** 0.5
DENOMS = (1, 2, 4, 5, 10, 20, 3, 6, 8, 12, 40)


def recognise(x, tol=1e-8):
    """x as (a, b, c): x = (a + b sqrt 5) / c, small integers; None if it is not one."""
    x = float(x)
    if abs(x) < 1e-12:
        return (0, 0, 1)
    b = np.arange(-40, 41)
    for c in DENOMS:
        a = np.round(x * c - b * _s5)
        err = np.abs(x - (a + b * _s5) / c)
        k = int(np.argmin(err))
        if err[k] < tol * max(1.0, abs(x)) and abs(a[k]) < 400:
            return (int(a[k]), int(b[k]), c)
    return None


def q(t):
    a, b, c = t
    return (mp.mpf(a) + mp.mpf(b) * SQ5) / c


def exact_vec(v, scale=True):
    """A vector (or matrix) of stored values made exact, up to scale when scale (divided by its largest entry)."""
    v = np.asarray(v, float)
    s = v.ravel()[np.argmax(np.abs(v.ravel()))] if scale else 1.0
    out = []
    for x in (v / s).ravel():
        t = recognise(x)
        if t is None:
            raise ValueError(f"{x} is not in Q(sqrt 5)")
        out.append(q(t))
    return mp.matrix(out) if v.ndim == 1 else mp.matrix([out[i * v.shape[1]:(i + 1) * v.shape[1]] for i in range(v.shape[0])])


class EPlane:
    def __init__(self, name, n, ex=None):
        self.name = name
        self.n = mp.matrix([q(t) for t in ex]) if ex else exact_vec(n)

    def f(self, x):
        return mp.fsum(self.n[i] * x[i] for i in range(4))

    def grad(self, x):
        return [self.n[i] for i in range(4)]


class ECurved:
    """A homogeneous polynomial in the betas, taken at A beta."""

    def __init__(self, name, mons, coeffs, A, ex=None):
        self.name, self.mons = name, [list(m) for m in mons]
        if ex:                                         # (the exact numbers stored with the surface)
            self.c = [q(t) for t in ex["coeffs"]]
            self.A = mp.matrix([[q(t) for t in row] for row in ex["A"]])
        else:
            self.c = [q(recognise(c)) for c in np.asarray(coeffs, float) / np.abs(coeffs).max()]
            self.A = exact_vec(np.asarray(A, float), scale=False)

    def _y(self, x):
        return [mp.fsum(self.A[i, j] * x[j] for j in range(4)) for i in range(4)]

    def f(self, x):
        y = self._y(x)
        return mp.fsum(c * mp.fprod(y[i] for i in m) for c, m in zip(self.c, self.mons))

    def grad(self, x):
        y = self._y(x)
        gy = [mp.mpf(0)] * 4
        for c, m in zip(self.c, self.mons):
            for k in range(len(m)):
                gy[m[k]] += c * mp.fprod(y[m[i]] for i in range(len(m)) if i != k)
        return [mp.fsum(gy[i] * self.A[i, j] for i in range(4)) for j in range(4)]


class EImage:
    """A surface carried by a symmetry g: f(x) = s(g^-1 x) (the surfaces are homogeneous, so no scale is needed)."""

    def __init__(self, s, gi):
        self.s, self.gi, self.ginv, self.name = s, gi, GROUP_INV[gi], f"{s.name}@{gi}"

    def f(self, x):
        return self.s.f(self.ginv * mp.matrix(list(x)))

    def grad(self, x):
        g = self.s.grad(self.ginv * mp.matrix(list(x)))
        return [mp.fsum(g[i] * self.ginv[i, j] for i in range(4)) for j in range(4)]


FACES = {"face b1=0": [1, 0, 0, 0], "face b2=0": [0, 1, 0, 0], "face b3=0": [0, 0, 1, 0], "face b4=0": [0, 0, 0, 1],
         "mirror b3=b4": [0, 0, 1, -1]}


def _load():
    S = json.load(open("surfaces.json"))
    out = {p["name"]: EPlane(p["name"], p["n"], p.get("exact")) for p in S["planes"]}
    out.update({c["name"]: ECurved(c["name"], c["mons"], c["coeffs"], c["A"], c.get("exact")) for c in S["curved"]})
    for k, n in FACES.items():
        out[k] = EPlane(k, n)
    return out


def _group():
    from core import FOLD, extended
    E = extended()
    GG = np.concatenate([E, E @ FOLD])
    return [exact_vec(g) for g in GG]


SURF = _load()
GROUP, GROUP_INV = [], []


def load_group():
    """The symmetries, exact (loaded on first use: 4800 matrices)."""
    if not GROUP:
        GROUP.extend(_group())
        GROUP_INV.extend(g ** -1 for g in GROUP)
    return GROUP


def _float_lib():
    import core as osd
    planes, curved = osd.library()
    lib = {s.name: s for s in list(planes) + list(curved)}
    for k, n in FACES.items():
        lib[k] = osd.Plane(np.array(n, float), k)
    return lib


FLIB = _float_lib()


def near(x, tol=1e-6):
    """The surfaces through x to tol, tested in floating point first (the exact test follows in corner)."""
    x = np.asarray([float(v) for v in x])
    out = []
    for n, s in FLIB.items():
        f = float(np.ravel(s.f(x))[0])
        g = np.asarray(s.n, float) if hasattr(s, "n") else s.grad(x)
        g = g - g.mean()
        if abs(f) / (np.linalg.norm(g) + 1e-300) < tol:
            out.append(n)
    return out


def carry(x, gi):
    """The point x carried by the symmetry gi, normalised (sum of betas 1)."""
    load_group()
    y = GROUP[gi] * mp.matrix(list(x))
    s = mp.fsum(y)
    return [y[i] / s for i in range(4)]


def value(s, x):
    """|f| / |grad f| at x: the distance from x to the surface, to first order."""
    g = s.grad(x)
    gm = mp.sqrt(mp.fsum((gi - mp.fsum(g) / 4) ** 2 for gi in g))
    return abs(s.f(x)) / (gm if gm else mp.mpf(1))


def solve(x0, surfs, steps=60):
    """The point near x0 on all the surfaces (sum of betas 1), by Newton steps at full precision (least change when
    there are more equations than unknowns). Returns (point, residual)."""
    x = mp.matrix([mp.mpf(float(v)) for v in x0])
    x = x / mp.fsum(x)
    for _ in range(steps):
        F = mp.matrix([s.f(x) for s in surfs] + [mp.fsum(x) - 1])
        if mp.norm(F) < mp.mpf(10) ** (-(mp.mp.dps - 5)):
            break
        J = mp.matrix([s.grad(x) for s in surfs] + [[1, 1, 1, 1]])
        dx = mp.lu_solve(J.T * J, J.T * F) if J.rows > 4 else mp.lu_solve(J, F)
        x = x - dx
    res = max([abs(s.f(x)) for s in surfs] + [abs(mp.fsum(x) - 1)])
    return [x[i] for i in range(4)], res


def through(x, names=None, tol=1e-6):
    """The surfaces (and faces) through the point x (to tol)."""
    return [n for n, s in SURF.items() if (names is None or n in names) and value(s, x) < tol]


def corner(x0, names=None, tol=1e-6):
    """The exact point near x0 where the surfaces through it meet: three of them with the best-conditioned crossing
    solved together, then every surface through x0 checked at full precision. Returns (point, surfaces on it) or None
    when no three cross there."""
    x0f = [mp.mpf(float(v)) for v in x0]
    cand = [n for n in near(x0, tol) if names is None or n in names]
    surf = {n: SURF[n] for n in cand}
    if len(cand) < 3 or _rank(cand, surf, x0f) < 3:
        # (the third surface is an image of a library surface under the group)
        load_group()
        for n, gi in near_images(x0, tol):
            k = f"{n}@{gi}"
            if k not in surf:
                surf[k] = EImage(SURF[n], gi)
                cand.append(k)
    if len(cand) < 3:
        return None
    G = []
    for n in cand:
        g = np.array([float(v) for v in surf[n].grad(x0f)])
        g -= g.mean()
        G.append(g / (np.linalg.norm(g) or 1))
    best, pick = -1, None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            for k in range(j + 1, len(cand)):
                sv = np.linalg.svd(np.vstack([G[i], G[j], G[k]]), compute_uv=False)[-1]
                if sv > best:
                    best, pick = sv, (i, j, k)
    if best < 1e-9:
        return None
    x, res = solve(x0, [surf[cand[i]] for i in pick])
    on = [n for n in cand if value(surf[n], x) < mp.mpf(10) ** -40]
    if len(on) < 3:
        print('corner: not converged', best, [float(v) for v in x0], flush=True)
        return None
    return x, on


def _rank(cand, surf, x):
    G = []
    for n in cand:
        g = np.array([float(v) for v in surf[n].grad(x)])
        g -= g.mean()
        G.append(g / (np.linalg.norm(g) or 1))
    return int(np.linalg.matrix_rank(np.array(G), tol=1e-6)) if G else 0


_GF = None


def near_images(x, tol=1e-6):
    """The images of library surfaces under the group through x (to tol), one per direction across x, as
    (surface name, symmetry index): for a point whose third surface is not itself in the library."""
    global _GF
    if _GF is None:
        from core import FOLD, extended
        E = extended()
        G = np.concatenate([E, E @ FOLD])
        _GF = (G, np.linalg.inv(G))
    G, GI = _GF
    x = np.asarray([float(v) for v in x])
    fx = lambda s, X: np.ravel(s.f(X))
    found = []
    for n, s in FLIB.items():
        if n in FACES:
            continue
        X = np.einsum("gij,j->gi", GI, x)
        f = fx(s, X)
        e = 1e-7
        Gr = np.stack([(fx(s, np.einsum("gij,j->gi", GI, x + e * u)) - fx(s, np.einsum("gij,j->gi", GI, x - e * u)))
                       / (2 * e) for u in np.eye(4)], 1)
        Gr = Gr - Gr.mean(1, keepdims=True)
        nG = np.linalg.norm(Gr, axis=1)
        d = np.abs(f) / (nG + 1e-300)
        for i in np.flatnonzero(d < tol):
            found.append((float(d[i]), n, int(i), Gr[i] / nG[i]))
    out = []
    for d, n, i, g in sorted(found, key=lambda t: t[0]):
        if not any(abs(abs(g @ h) - 1) < 1e-9 for _, _, h in out):
            out.append((n, i, g))
    return [(n, i) for n, i, _ in out]


def text(x, digits=40):
    return [mp.nstr(v, digits) for v in x]


@lru_cache(maxsize=None)
def as_float(t):
    return float(q(t))
