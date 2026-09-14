"""Top-level package for 4D vertex isogonality enumeration."""

from .generation import generate_vertices_from_seed
from .isogonal import OrbitPartition, compute_orbits, detect_symmetries, split_by_orbits
from .off import compute_convex_hull, parse_4off, to_4off
from .symmetry import SymmetryAction

__all__ = [
    "SymmetryAction",
    "generate_vertices_from_seed",
    "OrbitPartition",
    "compute_orbits",
    "detect_symmetries",
    "split_by_orbits",
    "compute_convex_hull",
    "parse_4off",
    "to_4off",
]
