from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class SymmetryAction:
    """A finite symmetry action in R^4 represented by linear maps.

    Attributes:
        generators: Iterable of shape-(4,4) arrays. These are assumed to act on
            column vectors in R^4.
    """

    generators: tuple[ArrayLike, ...]

    @staticmethod
    def from_iterable(mats: Iterable[ArrayLike]) -> "SymmetryAction":
        normalized: list[ArrayLike] = []
        for m in mats:
            a = np.asarray(m, dtype=float)
            if a.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix, got {a.shape}")
            normalized.append(a)
        if not normalized:
            raise ValueError("At least one generator matrix is required")
        return SymmetryAction(generators=tuple(normalized))

    def apply(self, v: ArrayLike) -> tuple[ArrayLike, ...]:
        """Apply all generators to a 4D vertex and return transformed vertices."""
        x = np.asarray(v, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"Expected vertex shape (4,), got {x.shape}")
        return tuple(g @ x for g in self.generators)
