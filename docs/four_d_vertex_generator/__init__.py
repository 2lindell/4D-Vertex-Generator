"""Top-level package for 4D vertex isogonality enumeration."""

from .generation import generate_vertices_from_seed
from .isogonal import OrbitPartition, compute_orbits, detect_symmetries, split_by_orbits
from .off import compute_convex_hull, parse_4off, to_4off
from .polychoron import (
    Corner,
    EdgeOrbit,
    Polychoron,
    build_polychoron,
    compare_to_hull,
    corner_classes,
    edge_orbits,
    find_closed_polychora,
    hull_edge_orbit_indices,
    reflection_edge_orbit_indices,
    rotation_edge_orbit_indices,
)
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
    "Corner",
    "EdgeOrbit",
    "Polychoron",
    "build_polychoron",
    "compare_to_hull",
    "corner_classes",
    "edge_orbits",
    "find_closed_polychora",
    "hull_edge_orbit_indices",
    "reflection_edge_orbit_indices",
    "rotation_edge_orbit_indices",
]
