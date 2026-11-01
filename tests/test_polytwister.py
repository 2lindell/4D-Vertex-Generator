from __future__ import annotations

import numpy as np

from four_d_vertex_generator.polytwister import cycloplane_values, hard_polytwister


def test_hard_tetter_has_four_rings_and_six_strips() -> None:
    shape = hard_polytwister()
    assert len(shape.ring_vectors) == 4
    assert len(shape.strip_curves) == 6
    assert sorted(shape.strip_constraints) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def test_rings_lie_on_three_cycloplanes_and_are_equivalent() -> None:
    shape = hard_polytwister()
    rings = shape.ring_points(48)
    values = cycloplane_values(rings.reshape(-1, 4), shape.directions)
    assert np.all(values <= 1 + 1e-9)
    assert np.all((np.abs(values - 1) < 1e-9).sum(axis=1) == 3)
    assert np.allclose(np.linalg.norm(rings, axis=-1), np.linalg.norm(rings[0, 0]))


def test_strip_points_lie_on_exactly_two_cycloplanes() -> None:
    shape = hard_polytwister()
    points = np.concatenate(shape.strip_lines(8))
    values = cycloplane_values(points, shape.directions)
    assert values.max() <= 1 + 1e-9
    interior = points[(np.abs(values - 1) < 1e-9).sum(axis=1) == 2]
    assert len(interior) > 0.9 * len(points)
