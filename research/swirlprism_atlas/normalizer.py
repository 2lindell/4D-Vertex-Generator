"""The half-turn Q outside H4 that normalizes h4_swirlprism, found from the congruent E2 pair."""
from __future__ import annotations

import os

import numpy as np
from cellframe import PHI, seed_from_beta
from scipy.spatial import cKDTree

from four_d_vertex_generator.generation import generate_vertices_from_seed, group_elements
from four_d_vertex_generator.library import named_symmetry

G = named_symmetry("h4_swirlprism")


def _find_q() -> np.ndarray:
    """An orthogonal map sending the orbit of beta (2+2φ)(1,1,1)+V4 onto the orbit of (2+2φ, 1+2φ, φ, 1)."""
    a = seed_from_beta([2 + 2 * PHI, 1 + 2 * PHI, PHI, 1])
    b = seed_from_beta([2 + 2 * PHI] * 3 + [1])
    A = generate_vertices_from_seed(a, G)
    B = generate_vertices_from_seed(b, G)
    tree = cKDTree(A)
    nb = np.argsort(np.linalg.norm(B - b, axis=1))[1:8]
    FB = np.vstack([b, B[nb]])
    DA = np.linalg.norm(A[:, None] - A[None], axis=2)
    DF = np.linalg.norm(FB[:, None] - FB[None], axis=2)

    def ext(sel):
        k = len(sel)
        if k == len(FB):
            Q = np.linalg.lstsq(FB, A[sel], rcond=None)[0].T
            if np.allclose(Q @ Q.T, np.eye(4), atol=1e-10) and tree.query(B @ Q.T)[0].max() < 1e-8:
                return Q
            return None
        for j in np.where(np.all(np.abs(DA[:, sel] - DF[k, :k]) < 1e-9, axis=1))[0]:
            if int(j) not in sel:
                r = ext(sel + [int(j)])
                if r is not None:
                    return r
        return None

    for a0 in range(len(A)):
        Q = ext([a0])
        if Q is not None:
            return Q
    raise RuntimeError("no congruence found")


def halfturn() -> np.ndarray:
    path = "normalizer_q.npy"
    if not os.path.exists(path):
        np.save(path, _find_q())
    return np.load(path)


if __name__ == "__main__":
    Q = halfturn()
    E = np.stack(group_elements(G))
    keys = {tuple(np.round(g, 7).ravel()) for g in E}
    print("orthogonal:", np.allclose(Q @ Q.T, np.eye(4)), " normalizes G:",
          all(tuple(np.round(Q @ g @ Q.T, 7).ravel()) in keys for g in E), " Q^2 = 1:", np.allclose(Q @ Q, np.eye(4)))


_E = None


def qcopies(beta, tol: float = 1e-10) -> list[np.ndarray]:
    """Barycentric positions (normalised, displayed half beta3 >= beta4) of the Q-image of a seed."""
    from cell_atlas import to_upper
    from cellframe import TINV
    global _E
    if _E is None:
        _E = np.stack(group_elements(G))
    y = halfturn() @ seed_from_beta(np.asarray(beta, float))
    out: list[np.ndarray] = []
    for x in _E @ y:
        b = TINV @ x
        if np.all(b >= -tol):
            b = to_upper(np.clip(b, 0, None) / np.clip(b, 0, None).sum())
            if not any(np.allclose(b, o, atol=1e-9) for o in out):
                out.append(b)
    return out
