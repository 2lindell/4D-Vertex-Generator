"""Search for a Dirichlet-domain centre whose 2400 domain lies in the half-cell and holds all the axes."""
from __future__ import annotations

import numpy as np
from cellframe import TINV, T, seed_from_beta
from dirichlet import dirichlet
from normalizer import coset_axes, extended_group

N = extended_group()
AXES = coset_axes()
AXIS_PTS = np.array([a + (b - a) * t for a, b in AXES for t in np.linspace(0, 1, 41)])
AXIS_X = AXIS_PTS @ T
AXIS_X /= np.linalg.norm(AXIS_X, axis=1, keepdims=True)


def score(beta_p):
    p = seed_from_beta(np.asarray(beta_p, float))
    if any(np.allclose(g @ p, p, atol=1e-9) for g in N[1:]):
        return None
    D = dirichlet(p, N)
    B = np.array([TINV @ x for x in D["corners"]])
    B = B / B.sum(axis=1, keepdims=True)
    outside = max(0.0, float(-B.min()), float(-(B[:, 2] - B[:, 3]).min()))
    images = N @ p
    near = (AXIS_X @ p) >= (AXIS_X @ images.T).max(axis=1) - 1e-10
    return {"outside": outside, "axes_on_surface": float(near.mean()), "corners": len(B), "faces": len(D["faces"])}


def grid(n):
    for i in range(1, n):
        for j in range(1, n - i):
            for k in range(1, n - i - j):
                m = n - i - j - k
                if m >= 1 and k > m:
                    yield np.array([i, j, k, m], float) / n
