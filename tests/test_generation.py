import numpy as np

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import to_4off


def test_hyperoctahedral_seed_axis_generates_8_vertices() -> None:
    action = named_symmetry("hyperoctahedral")
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    verts = generate_vertices_from_seed(seed, action)

    # Orbit of e1 under signed permutations in 4D has 8 points: +/- e_i
    assert len(verts) == 8


def test_4off_header() -> None:
    verts = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    txt = to_4off(verts)
    lines = txt.strip().splitlines()
    assert lines[0] == "4OFF"
    assert lines[1] == "2 0 0"
