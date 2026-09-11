import numpy as np

from four_d_vertex_generator.isogonal import compute_orbits
from four_d_vertex_generator.symmetry import SymmetryAction


def test_identity_action_gives_singletons() -> None:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    identity = np.eye(4)
    action = SymmetryAction.from_iterable([identity])

    part = compute_orbits(vertices, action)

    assert part.num_orbits == 3
    assert part.orbit_ids == [0, 1, 2]


def test_swap_connects_two_vertices() -> None:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    swap_xy = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    action = SymmetryAction.from_iterable([np.eye(4), swap_xy])
    part = compute_orbits(vertices, action)

    assert part.num_orbits == 2
    assert part.orbit_ids[0] == part.orbit_ids[1]
    assert part.orbit_ids[2] != part.orbit_ids[0]
