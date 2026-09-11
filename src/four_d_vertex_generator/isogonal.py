from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .symmetry import SymmetryAction


def _key(v: np.ndarray, tol: float) -> tuple[int, int, int, int]:
    """Quantize a 4D vector to a hashable key using tolerance-scaled rounding."""
    if tol <= 0:
        raise ValueError("tol must be positive")
    q = np.round(v / tol).astype(int)
    return int(q[0]), int(q[1]), int(q[2]), int(q[3])


@dataclass(frozen=True)
class OrbitPartition:
    """Container for orbit partition results.

    Attributes:
        orbit_ids: For each input vertex index i, orbit_ids[i] gives the orbit id.
        num_orbits: Number of discovered orbits.
    """

    orbit_ids: list[int]
    num_orbits: int


def compute_orbits(vertices: np.ndarray, action: SymmetryAction, tol: float = 1e-8) -> OrbitPartition:
    """Compute orbit partition (isogonal grouping) for vertices under a symmetry action.

    Notes:
        - This initial implementation applies provided generators directly during BFS.
        - For many groups, you'll want closure generation/caching for full robust action.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 4:
        raise ValueError(f"Expected vertices shape (n,4), got {verts.shape}")

    key_to_index: dict[tuple[int, int, int, int], int] = {}
    for i, v in enumerate(verts):
        key_to_index[_key(v, tol)] = i

    n = len(verts)
    orbit_ids = [-1] * n
    orbit_counter = 0

    for i in range(n):
        if orbit_ids[i] != -1:
            continue

        orbit_ids[i] = orbit_counter
        q: deque[int] = deque([i])

        while q:
            cur = q.popleft()
            for img in action.apply(verts[cur]):
                j = key_to_index.get(_key(img, tol))
                if j is None:
                    continue
                if orbit_ids[j] == -1:
                    orbit_ids[j] = orbit_counter
                    q.append(j)

        orbit_counter += 1

    return OrbitPartition(orbit_ids=orbit_ids, num_orbits=orbit_counter)
