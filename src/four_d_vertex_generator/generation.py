from __future__ import annotations

from collections import deque

import numpy as np

from .symmetry import SymmetryAction


def _key(v: np.ndarray, tol: float) -> tuple[int, int, int, int]:
    if tol <= 0:
        raise ValueError("tol must be positive")
    q = np.round(v / tol).astype(int)
    return int(q[0]), int(q[1]), int(q[2]), int(q[3])


def generate_vertices_from_seed(
    seed: np.ndarray,
    action: SymmetryAction,
    *,
    tol: float = 1e-8,
    max_vertices: int = 20000,
) -> np.ndarray:
    """Generate the orbit of a seed point under generator closure via BFS.

    Repeatedly applies all generators to newly discovered points until closure
    (within tolerance-quantized keys) is reached.
    """
    s = np.asarray(seed, dtype=float)
    if s.shape != (4,):
        raise ValueError(f"Expected seed shape (4,), got {s.shape}")

    discovered: dict[tuple[int, int, int, int], np.ndarray] = {_key(s, tol): s}
    q: deque[np.ndarray] = deque([s])

    while q:
        cur = q.popleft()
        for img in action.apply(cur):
            k = _key(img, tol)
            if k in discovered:
                continue
            discovered[k] = img
            q.append(img)
            if len(discovered) > max_vertices:
                raise RuntimeError(
                    f"Exceeded max_vertices={max_vertices}. "
                    "Group may be very large/infinite under current generators."
                )

    verts = np.vstack(list(discovered.values()))
    return verts
