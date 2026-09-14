from __future__ import annotations

import numpy as np
import pytest

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull, to_4off


def test_compute_convex_hull_tesseract() -> None:
    # 8-cell (tesseract): 16 vertices, 24 square faces, 8 cubic cells
    action = named_symmetry("hyperoctahedral")
    seed = np.array([1.0, 1.0, 1.0, 1.0])
    verts = generate_vertices_from_seed(seed, action)

    faces, cells = compute_convex_hull(verts)

    assert len(verts) == 16
    assert len(faces) == 24
    assert len(cells) == 8

    # Each cell in a tesseract has 6 faces
    for cell in cells:
        assert len(cell) == 6

    # Each face in a tesseract has 4 vertices
    for face in faces:
        assert len(face) == 4


def test_compute_convex_hull_16cell() -> None:
    # 16-cell: 8 vertices, 32 triangular faces, 16 tetrahedral cells
    action = named_symmetry("hyperoctahedral")
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    verts = generate_vertices_from_seed(seed, action)

    faces, cells = compute_convex_hull(verts)

    assert len(verts) == 8
    assert len(faces) == 32
    assert len(cells) == 16

    # Each cell is a tetrahedron (4 faces)
    for cell in cells:
        assert len(cell) == 4

    # Each face is a triangle (3 vertices)
    for face in faces:
        assert len(face) == 3


def test_compute_convex_hull_24cell() -> None:
    # 24-cell: 24 vertices, 96 triangular faces, 24 octahedral cells
    action = named_symmetry("f4")
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    verts = generate_vertices_from_seed(seed, action)

    faces, cells = compute_convex_hull(verts)

    assert len(verts) == 24
    assert len(faces) == 96
    assert len(cells) == 24

    # Each cell is an octahedron (8 faces)
    for cell in cells:
        assert len(cell) == 8


def test_compute_convex_hull_5cell() -> None:
    # 5-cell (simplex): 5 vertices, 10 triangular faces, 5 tetrahedral cells
    v_5 = np.array([
        [1.0, 1.0, 1.0, -1.0 / np.sqrt(5)],
        [1.0, -1.0, -1.0, -1.0 / np.sqrt(5)],
        [-1.0, 1.0, -1.0, -1.0 / np.sqrt(5)],
        [-1.0, -1.0, 1.0, -1.0 / np.sqrt(5)],
        [0.0, 0.0, 0.0, 4.0 / np.sqrt(5)],
    ])

    faces, cells = compute_convex_hull(v_5)

    assert len(faces) == 10
    assert len(cells) == 5


def test_compute_convex_hull_too_few_vertices_raises_value_error() -> None:
    few_verts = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="At least 5 vertices"):
        compute_convex_hull(few_verts)


def test_compute_convex_hull_degenerate_vertices_raises_value_error() -> None:
    # Vertices lying in a 3D subspace (w=0)
    flat_verts = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
    ])
    with pytest.raises(ValueError, match="Vertices do not span 4D space"):
        compute_convex_hull(flat_verts)


def test_to_4off_with_explicit_faces_and_cells() -> None:
    verts = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    faces = [[0, 1, 2], [0, 1, 3]]
    cells = [[0, 1]]

    off_str = to_4off(verts, faces, cells)
    lines = off_str.strip().splitlines()

    assert lines[0] == "4OFF"
    assert lines[1] == "5 2 0 1"
    # 5 vertex lines
    assert lines[2] == "0 0 0 0"
    assert lines[6] == "0 0 0 1"
    # 2 face lines
    assert lines[7] == "3 0 1 2"
    assert lines[8] == "3 0 1 3"
    # 1 cell line
    assert lines[9] == "2 0 1"


def test_to_4off_with_compute_hull() -> None:
    action = named_symmetry("hyperoctahedral")
    seed = np.array([1.0, 1.0, 1.0, 1.0])
    verts = generate_vertices_from_seed(seed, action)

    off_str = to_4off(verts, compute_hull=True)
    lines = off_str.strip().splitlines()

    assert lines[0] == "4OFF"
    assert lines[1] == "16 24 0 8"
    # Total lines: 1 (header) + 1 (counts) + 16 (verts) + 24 (faces) + 8 (cells) = 50 lines
    assert len(lines) == 50


def test_compute_convex_hull_swirlprism_every_face_in_two_cells() -> None:
    from collections import defaultdict
    from four_d_vertex_generator.library import h4_swirlprism_anchor_seed

    action = named_symmetry("h4_swirlprism+")
    seed = h4_swirlprism_anchor_seed()
    verts = generate_vertices_from_seed(seed, action)

    faces, cells = compute_convex_hull(verts)

    face_counts = defaultdict(int)
    for cell in cells:
        for f_idx in cell:
            face_counts[f_idx] += 1

    # Every face in the hull must belong to exactly 2 cells
    assert len(face_counts) == len(faces)
    assert set(face_counts.values()) == {2}

