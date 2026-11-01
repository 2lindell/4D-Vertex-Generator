"""Small exact helpers shared by the exact-data scripts (line_ends.py, exact_lines.py): betas normalised,
a surface's value and gradient, Newton steps onto surfaces, the half-cell test, and the extended group."""
from __future__ import annotations

import numpy as np

import core as osd
from core import FOLD, extended

G = extended()
GG = np.concatenate([G, G @ FOLD])


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def fval(s, x):
    return float(np.atleast_1d(s.f(np.atleast_2d(x)).ravel())[0])


def grad(s, x):
    if isinstance(s, osd.Plane):
        g = s.n.astype(float)
    else:
        e = 1e-7
        g = np.array([(fval(s, x + e * u) - fval(s, x - e * u)) / (2 * e) for u in np.eye(4)])
    return g - g.mean()


def project(surfs, x, steps=30):
    """x moved onto all the surfaces (sum(b) = 1 kept), by Newton steps (least change)."""
    x = np.asarray(x, float).copy()
    for _ in range(steps):
        F = np.array([fval(s, x) for s in surfs])
        if np.abs(F).max() < 1e-14:
            break
        J = np.array([grad(s, x) for s in surfs])
        x = x - J.T @ np.linalg.lstsq(J @ J.T, F, rcond=None)[0]
    return x


def in_half(x, tol=1e-12):
    return (x > -tol).all() and x[2] >= x[3] - tol
