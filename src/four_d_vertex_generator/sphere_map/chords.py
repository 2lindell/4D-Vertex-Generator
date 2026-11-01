"""Explore the half-cell with random chords: exact crossings with every library surface.

Each chord is a straight segment through the domain (a great-circle arc on the sphere), ending on
two of its faces. Along it every surface is a polynomial in the chord parameter (degree 1, 2 or 4),
so its crossings are polynomial roots, polished by Newton. Between crossings the sign of every
surface is constant: that sign vector names the piece of the arrangement the chord passes through.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frame import DOMAIN_FACES, surface_values, surfaces

FACE_NAMES = list(DOMAIN_FACES)
FACE_N = np.stack([DOMAIN_FACES[k] for k in FACE_NAMES])

# Chebyshev-Lobatto nodes on [-1, 1] and the inverse Vandermonde for exact polynomial fits.
_NODES = {d: np.cos(np.pi * np.arange(d + 1) / d) for d in (2, 4)}
_VINV = {d: np.linalg.inv(np.vander(_NODES[d], d + 1, increasing=True)) for d in (2, 4)}


def random_domain_points(n: int, rng: np.random.Generator) -> np.ndarray:
    b = rng.dirichlet(np.ones(4), n)
    swap = b[:, 2] < b[:, 3]
    b[swap] = b[swap][:, [1, 0, 3, 2]]
    return b


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=(n, 4))
    d -= d.mean(axis=1, keepdims=True)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def chord_ends(b0: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parameters t0 < 0 < t1 where b0 + t d leaves the domain, and the faces it leaves through."""
    nb = b0 @ FACE_N.T                 # >= 0 inside
    nd = d @ FACE_N.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -nb / nd
    t_hi = np.where(nd < 0, t, np.inf)
    t_lo = np.where(nd > 0, t, -np.inf)
    f1 = t_hi.argmin(axis=1)
    f0 = t_lo.argmax(axis=1)
    rows = np.arange(len(b0))
    return t_lo[rows, f0], t_hi[rows, f1], f0, f1


@dataclass
class ChordResult:
    """Crossings of one batch of chords."""

    start_beta: np.ndarray                  # (M, 4)
    end_beta: np.ndarray
    start_face: np.ndarray                  # (M,)
    end_face: np.ndarray
    start_signs: np.ndarray                 # (M, S) int8
    # flattened crossings, sorted by chord then by position along it
    chord: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    u: np.ndarray = field(default_factory=lambda: np.zeros(0))          # position in [0, 1]
    surface: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    beta: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # chord passed the end check


def _roots_on_segment(coef: np.ndarray, deg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Real roots in (-1, 1) of polynomials (rows of increasing-power coefficients)."""
    out_rows, out_u = [], []
    if deg == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -coef[:, 0] / coef[:, 1]
        ok = np.isfinite(r) & (r > -1) & (r < 1)
        return [(np.flatnonzero(ok), r[ok])]
    scale = np.abs(coef).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    c = coef / scale
    lead = c[:, deg]
    small = np.abs(lead) < 1e-9
    for rows, d in ((np.flatnonzero(~small), deg), (np.flatnonzero(small), deg - 1)):
        if not len(rows):
            continue
        if d == 0:
            continue
        cc = c[rows, : d + 1]
        if d == 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = -cc[:, 0] / cc[:, 1]
            roots = r[:, None].astype(complex)
        else:
            comp = np.zeros((len(rows), d, d))
            comp[:, 1:, :-1] = np.eye(d - 1)
            comp[:, :, -1] = -cc[:, :d] / cc[:, d : d + 1]
            roots = np.linalg.eigvals(comp)
        real = (np.abs(roots.imag) < 1e-7) & (roots.real > -1) & (roots.real < 1)
        ri, rj = np.nonzero(real)
        out_rows.append(rows[ri])
        out_u.append(roots.real[ri, rj])
    if not out_rows:
        return [(np.zeros(0, int), np.zeros(0))]
    return [(np.concatenate(out_rows), np.concatenate(out_u))]


def cast(b0: np.ndarray, d: np.ndarray, surfs=None) -> ChordResult:
    """Crossings of every library surface along the chords through b0 with directions d."""
    t0, t1, f0, f1 = chord_ends(b0, d)
    a = b0 + t0[:, None] * d
    b = b0 + t1[:, None] * d
    a[np.abs(a) < 1e-15] = 0.0
    b[np.abs(b) < 1e-15] = 0.0
    mid, half = (a + b) / 2, (b - a) / 2          # beta(v) = mid + v * half, v in [-1, 1]
    surfs = surfs or surfaces()
    rows_all, v_all, s_all = [], [], []
    for s in surfs:
        deg = s.degree if s.kind == "curved" else 1
        if deg == 1:
            f_mid, f_dir = mid @ s.normal, half @ s.normal
            coef = np.stack([f_mid, f_dir], axis=1)
        else:
            nodes = _NODES[deg]
            vals = np.stack([s.f(mid + v * half) for v in nodes], axis=1)
            coef = vals @ _VINV[deg].T
        for rows, v in _roots_on_segment(coef, deg):
            if not len(rows):
                continue
            # Newton polish on the true surface along the chord
            for _ in range(3):
                pts = mid[rows] + v[:, None] * half[rows]
                fv = s.f(pts)
                gv = np.einsum("ij,ij->i", s.grad(pts), half[rows])
                step = np.where(np.abs(gv) > 1e-300, fv / np.where(gv == 0, 1, gv), 0.0)
                v = v - step
            keep = (v > -1 + 1e-12) & (v < 1 - 1e-12)
            rows_all.append(rows[keep])
            v_all.append(v[keep])
            s_all.append(np.full(int(keep.sum()), s.index))
    chord = np.concatenate(rows_all) if rows_all else np.zeros(0, int)
    v = np.concatenate(v_all) if v_all else np.zeros(0)
    surf = np.concatenate(s_all) if s_all else np.zeros(0, int)
    order = np.lexsort((v, chord))
    chord, v, surf = chord[order], v[order], surf[order]
    beta = mid[chord] + v[:, None] * half[chord]

    # sign vectors at the start, a hair inside; then check flips against the end
    eps = 1e-9
    start_signs = np.sign(surface_values(a + eps * (b - a), surfs)).astype(np.int8)
    end_signs = np.sign(surface_values(b - eps * (b - a), surfs)).astype(np.int8)
    propagated = start_signs.copy()
    np.multiply.at(propagated, (chord, surf), -1)
    valid = (propagated == end_signs).all(axis=1) & (start_signs != 0).all(axis=1)
    return ChordResult(a, b, f0, f1, start_signs, chord, (v + 1) / 2, surf, beta, valid)
