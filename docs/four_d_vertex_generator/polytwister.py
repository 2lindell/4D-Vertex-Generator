"""Hard polytwisters: intersections of cycloplanes, built from their Hopf quotient.

A point y = (z1, z2) of C^2 maps to H(y) = (2 Re z1 conj(z2), 2 Im z1 conj(z2), |z1|^2 - |z2|^2)
in R^3, with |H| = |y|^2, and the global phase rotation y -> e^(it) y is exactly the fibre of
that map. A direction d in S^2 has a Hopf lift q(d) in C^2, and the cycloplane of d is the
cylinder |<y, q(d)>| <= 1, the dual of the ring (unit circle) through q(d). Because
|<y, q(d)>|^2 = (|H| + d.H) / 2, each cycloplane becomes the paraboloid |H| + d.H <= 2 with its
focus at the origin. The hard polytwister is the intersection of the cycloplanes of the
chosen directions, so modulo phase it is a curved polyhedron in R^3 whose
  - vertices (three paraboloids meet) lift to its rings,
  - conic edges (two paraboloids meet) lift to its strips,
  - faces (one paraboloid) lift to its twisters.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

# Four directions of a regular tetrahedron, rotated so that no strip passes through the
# polar axis where the Hopf lift used for drawing is discontinuous.
_TETRAHEDRON = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
) / np.sqrt(3)


def _rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


_ORIENTATION = _rotation((0.70015, 0.66366, 1.97247), 2.38229)
TETRAHEDRON_DIRECTIONS = _TETRAHEDRON @ _ORIENTATION.T


def hopf_lift(vectors: np.ndarray) -> np.ndarray:
    """Lift points H of R^3 to y in C^2 (as real 4-vectors) with H(y) = H and |y|^2 = |H|."""
    vectors = np.asarray(vectors, dtype=float)
    radius = np.linalg.norm(vectors, axis=-1)
    safe = np.where(radius > 0, radius, 1.0)
    polar = np.arccos(np.clip(vectors[..., 2] / safe, -1.0, 1.0))
    azimuth = np.arctan2(vectors[..., 1], vectors[..., 0])
    scale = np.sqrt(radius)
    first = scale * np.cos(polar / 2) * np.exp(1j * azimuth)
    second = scale * np.sin(polar / 2)
    return np.stack((first.real, first.imag, second, np.zeros_like(second)), axis=-1)


def rotate_phase(points: np.ndarray, angle: float | np.ndarray) -> np.ndarray:
    """Apply the global phase rotation y -> e^(i angle) y to lifted points."""
    points = np.asarray(points, dtype=float)
    cos, sin = np.cos(angle), np.sin(angle)
    first = (points[..., 0] + 1j * points[..., 1]) * (cos + 1j * sin)
    second = (points[..., 2] + 1j * points[..., 3]) * (cos + 1j * sin)
    return np.stack((first.real, first.imag, second.real, second.imag), axis=-1)


def cycloplane_values(points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """|<y, q(d)>|^2 for each 4D point and direction; the polytwister is where all are <= 1."""
    points = np.asarray(points, dtype=float)
    z1 = points[..., 0] + 1j * points[..., 1]
    z2 = points[..., 2] + 1j * points[..., 3]
    hopf = np.stack(
        (2 * (z1 * z2.conj()).real, 2 * (z1 * z2.conj()).imag, abs(z1) ** 2 - abs(z2) ** 2),
        axis=-1,
    )
    return (np.linalg.norm(hopf, axis=-1)[..., None] + hopf @ np.asarray(directions).T) / 2


@dataclass(frozen=True)
class HardPolytwister:
    """Rings and strips of a hard polytwister, stored by their Hopf quotient in R^3."""

    directions: np.ndarray
    ring_vectors: np.ndarray
    ring_constraints: tuple[tuple[int, ...], ...]
    strip_curves: tuple[np.ndarray, ...]
    strip_constraints: tuple[tuple[int, int], ...]

    def ring_points(self, samples: int = 120) -> np.ndarray:
        """Return (rings, samples, 4): each ring is a circle e^(it) y0 about the origin."""
        base = hopf_lift(self.ring_vectors)
        angles = 2 * np.pi * np.arange(samples) / samples
        return np.stack([rotate_phase(point, angles) for point in base])

    def strip_lines(self, phase_lines: int = 24) -> list[np.ndarray]:
        """Polylines across every strip, one per phase, each running ring to ring."""
        angles = 2 * np.pi * np.arange(phase_lines) / phase_lines
        lines = []
        for curve in self.strip_curves:
            lifted = hopf_lift(curve)
            lines.extend(rotate_phase(lifted, angle) for angle in angles)
        return lines

    def strip_circles(self, samples: int = 120, count: int = 6) -> list[np.ndarray]:
        """Phase circles through evenly spaced points of every strip (its ruling by rings)."""
        angles = 2 * np.pi * np.arange(samples + 1) / samples
        circles = []
        for curve in self.strip_curves:
            lifted = hopf_lift(curve)
            for index in np.linspace(0, len(curve) - 1, count + 2)[1:-1].astype(int):
                circles.append(rotate_phase(lifted[index], angles))
        return circles


def hard_polytwister(
    directions: np.ndarray = TETRAHEDRON_DIRECTIONS,
    *,
    curve_samples: int = 721,
    tol: float = 1e-9,
) -> HardPolytwister:
    """Build the intersection of the cycloplanes of the given unit directions."""
    directions = np.asarray(directions, dtype=float)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    count = len(directions)

    def inside(vectors: np.ndarray) -> np.ndarray:
        values = np.linalg.norm(vectors, axis=-1)[..., None] + vectors @ directions.T
        return np.all(values <= 2 + 1e-7, axis=-1)

    rings: list[np.ndarray] = []
    ring_constraints: list[tuple[int, ...]] = []
    for triple in combinations(range(count), 3):
        block = directions[list(triple)]
        if abs(np.linalg.det(block)) < tol:
            continue
        direction = np.linalg.solve(block, np.ones(3))
        length = np.linalg.norm(direction)
        # |H| + d.H = 2 for the three directions gives H = (2 - r) u with |H| = r.
        radius = 2 * length / (1 + length)
        vector = (2 - radius) * direction
        if inside(vector):
            rings.append(vector)
            ring_constraints.append(triple)

    strips: list[np.ndarray] = []
    strip_constraints: list[tuple[int, int]] = []
    phi = np.linspace(-np.pi, np.pi, curve_samples, endpoint=False)
    for a, b in combinations(range(count), 2):
        mean = directions[a] + directions[b]
        mu = np.linalg.norm(mean) / 2
        cross = np.cross(directions[a], directions[b])
        if mu < tol or np.linalg.norm(cross) < tol:
            continue
        mean = mean / (2 * mu)
        cross = cross / np.linalg.norm(cross)
        # In the bisector plane the intersection is the conic r = 2 / (1 + mu cos phi).
        radius = 2 / (1 + mu * np.cos(phi))
        curve = radius[:, None] * (np.cos(phi)[:, None] * mean + np.sin(phi)[:, None] * cross)
        keep = inside(curve) & (radius > 0)
        if not keep.any():
            continue
        start = int(np.argmin(keep))
        curve, keep = np.roll(curve, -start, axis=0), np.roll(keep, -start)
        edges = np.flatnonzero(np.diff(np.r_[False, keep, False].astype(int)))
        for first, last in zip(edges[::2], edges[1::2]):
            if last - first >= 3:
                strips.append(curve[first:last])
                strip_constraints.append((a, b))

    ring_array = np.array(rings)
    for curve in strips:
        # The sampled conic stops just short of each ring, so close it exactly.
        for end in (0, -1):
            gaps = np.linalg.norm(ring_array - curve[end], axis=1)
            if gaps.min() < 0.05:
                curve[end] = ring_array[int(np.argmin(gaps))]

    return HardPolytwister(
        directions=directions,
        ring_vectors=ring_array,
        ring_constraints=tuple(ring_constraints),
        strip_curves=tuple(strips),
        strip_constraints=tuple(strip_constraints),
    )
