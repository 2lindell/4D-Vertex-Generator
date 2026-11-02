from __future__ import annotations

from functools import cache

import numpy as np

# Polytwister symmetries: continuous forms of the tetrahedral, octahedral and
# icosahedral swirlprism symmetries. The "ki" names use the rotation subgroup.
POLYTWISTER_SYMMETRIES = (
    "tetteric",
    "kitetteric",
    "cubiteric",
    "kicubiteric",
    "doteric",
    "kidoteric",
)
CONTINUOUS_SYMMETRIES = ("glomic", "kiglomic", "hopfic", "kihopfic") + POLYTWISTER_SYMMETRIES
DEFAULT_SAMPLES_PER_ANGLE = 120


def continuous_approximation_vertex_count(
    name: str, samples_per_angle: int = DEFAULT_SAMPLES_PER_ANGLE
) -> int:
    """Return the maximum point count for a generic sampled orbit."""
    name = name.strip().lower()
    _validate_request(name, samples_per_angle)
    if name in ("glomic", "kiglomic"):
        return (samples_per_angle - 2) * samples_per_angle**2 + 2 * samples_per_angle
    if name in POLYTWISTER_SYMMETRIES:
        return _polytwister_group_order(name) * samples_per_angle
    return samples_per_angle**2


def generate_continuous_approximation(
    seed: np.ndarray,
    name: str,
    *,
    samples_per_angle: int = DEFAULT_SAMPLES_PER_ANGLE,
    max_vertices: int = 2_000_000,
) -> np.ndarray:
    """Sample a continuous glome or Hopf orbit using a finite angular grid."""
    name = name.strip().lower()
    _validate_request(name, samples_per_angle)
    if max_vertices < 1:
        raise ValueError("max_vertices must be positive")

    point = np.asarray(seed, dtype=float)
    if point.shape != (4,):
        raise ValueError(f"Expected seed shape (4,), got {point.shape}")
    if not np.isfinite(point).all():
        raise ValueError("Seed coordinates must be finite")
    if not np.any(point):
        return np.zeros((1, 4))

    estimate = continuous_approximation_vertex_count(name, samples_per_angle)
    if estimate > max_vertices:
        raise RuntimeError(
            f"The {name} approximation requires up to {estimate:,} vertices at "
            f"{samples_per_angle} samples per angle, exceeding max_vertices={max_vertices:,}."
        )

    angles = 2.0 * np.pi * np.arange(samples_per_angle) / samples_per_angle
    cosine, sine = np.cos(angles), np.sin(angles)

    if name in ("glomic", "kiglomic"):
        return _sample_glome(float(np.linalg.norm(point)), cosine, sine)
    if name in POLYTWISTER_SYMMETRIES:
        return _sample_polytwister_rings(point, name, angles)
    return _sample_hopf_torus(point, cosine, sine)


def _polytwister_group_order(name: str) -> int:
    return len(_polyhedral_point_group(name))


def _polyhedral_generators(name: str) -> list[np.ndarray]:
    kind = name.removeprefix("ki").removesuffix("teric")
    chiral = name.startswith("ki")
    if kind == "tet":
        generators = [_axis_rotation((1, 1, 1), 2 * np.pi / 3), _axis_rotation((0, 0, 1), np.pi)]
        if not chiral:
            generators.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float))
    elif kind == "cubi":
        generators = [
            _axis_rotation((1, 1, 1), 2 * np.pi / 3),
            _axis_rotation((0, 0, 1), np.pi / 2),
        ]
        if not chiral:
            generators.append(-np.eye(3))
    else:
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        generators = [
            _axis_rotation((0, 1, golden), 2 * np.pi / 5),
            _axis_rotation((1, 1, 1), 2 * np.pi / 3),
        ]
        if not chiral:
            generators.append(-np.eye(3))
    return generators


@cache
def _polyhedral_point_group(name: str) -> tuple[np.ndarray, ...]:
    """Return the 3D point group matrices behind a polytwister symmetry."""
    generators = _polyhedral_generators(name)
    elements = {_matrix_key(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        new_frontier = []
        for element in frontier:
            for generator in generators:
                product = generator @ element
                key = _matrix_key(product)
                if key not in elements:
                    elements[key] = product
                    new_frontier.append(product)
        frontier = new_frontier
    return tuple(elements.values())


def _matrix_key(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple((np.round(matrix, 6) + 0.0).ravel())


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _sample_polytwister_rings(seed: np.ndarray, name: str, angles: np.ndarray) -> np.ndarray:
    """Sample the Hopf rings over the polyhedral orbit of the seed's Hopf image."""
    first = complex(seed[0], seed[1])
    second = complex(seed[2], seed[3])
    pair = first * second.conjugate()
    base = np.array([2 * pair.real, 2 * pair.imag, abs(first) ** 2 - abs(second) ** 2])

    orbit: dict[tuple[float, ...], np.ndarray] = {}
    for matrix in _polyhedral_point_group(name):
        image = matrix @ base
        orbit.setdefault(_matrix_key(image), image)

    phase = np.exp(1j * angles)
    rings = []
    for image in orbit.values():
        # Lift a point of S2 to a point of S3 whose Hopf image it is.
        scale = float(np.linalg.norm(image))
        polar = np.arccos(np.clip(image[2] / scale, -1.0, 1.0))
        azimuth = np.arctan2(image[1], image[0])
        lifted_first = np.sqrt(scale) * np.cos(polar / 2) * np.exp(1j * azimuth)
        lifted_second = np.sqrt(scale) * np.sin(polar / 2)
        ring_first = lifted_first * phase
        ring_second = lifted_second * phase
        rings.append(
            np.column_stack((ring_first.real, ring_first.imag, ring_second.real, ring_second.imag))
        )
    return np.concatenate(rings)


def _sample_glome(radius: float, cosine: np.ndarray, sine: np.ndarray) -> np.ndarray:
    """Sample S3 in Hopf coordinates, omitting duplicate pole circles."""
    samples = len(cosine)
    latitude = np.linspace(0.0, np.pi / 2.0, samples)
    cos_theta, cos_phi = np.meshgrid(cosine, cosine, indexing="ij")
    sin_theta, sin_phi = np.meshgrid(sine, sine, indexing="ij")
    points = [
        np.column_stack((radius * cosine, radius * sine, np.zeros(samples), np.zeros(samples))),
        np.column_stack((np.zeros(samples), np.zeros(samples), radius * cosine, radius * sine)),
    ]

    for angle in latitude[1:-1]:
        cos_latitude, sin_latitude = np.cos(angle), np.sin(angle)
        points.append(
            np.column_stack(
                (
                    (radius * cos_latitude * cos_theta).ravel(),
                    (radius * cos_latitude * sin_theta).ravel(),
                    (radius * sin_latitude * cos_phi).ravel(),
                    (radius * sin_latitude * sin_phi).ravel(),
                )
            )
        )
    return np.concatenate(points)


def _sample_hopf_torus(
    seed: np.ndarray, cosine: np.ndarray, sine: np.ndarray
) -> np.ndarray:
    """Rotate the two orthogonal seed-plane components independently."""
    first_radius = float(np.linalg.norm(seed[:2]))
    second_radius = float(np.linalg.norm(seed[2:]))
    if first_radius <= 1e-14:
        return np.column_stack(
            (
                np.zeros(len(cosine)),
                np.zeros(len(cosine)),
                second_radius * cosine,
                second_radius * sine,
            )
        )
    if second_radius <= 1e-14:
        return np.column_stack(
            (
                first_radius * cosine,
                first_radius * sine,
                np.zeros(len(cosine)),
                np.zeros(len(cosine)),
            )
        )

    first_cosine, second_cosine = np.meshgrid(cosine, cosine, indexing="ij")
    first_sine, second_sine = np.meshgrid(sine, sine, indexing="ij")
    return np.column_stack(
        (
            (seed[0] * first_cosine - seed[1] * first_sine).ravel(),
            (seed[0] * first_sine + seed[1] * first_cosine).ravel(),
            (seed[2] * second_cosine - seed[3] * second_sine).ravel(),
            (seed[2] * second_sine + seed[3] * second_cosine).ravel(),
        )
    )


def _validate_request(name: str, samples_per_angle: int) -> None:
    if name not in CONTINUOUS_SYMMETRIES:
        raise ValueError(
            f"Unknown continuous symmetry '{name}'. Choose one of: "
            f"{', '.join(CONTINUOUS_SYMMETRIES)}"
        )
    if isinstance(samples_per_angle, bool) or not isinstance(
        samples_per_angle, (int, np.integer)
    ):
        raise ValueError("samples_per_angle must be an integer")
    if samples_per_angle < 3:
        raise ValueError("samples_per_angle must be at least 3")
