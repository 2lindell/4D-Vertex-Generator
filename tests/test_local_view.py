from __future__ import annotations

import numpy as np
import pytest

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import fundamental_chamber_roots, named_symmetry
from four_d_vertex_generator.local_view import (
    edge_length_classes,
    hull_edges,
    local_view,
    nearest_neighbour_edges,
)
from four_d_vertex_generator.off import compute_convex_hull


def _polytope(symmetry: str, seed: list[float]) -> np.ndarray:
    return generate_vertices_from_seed(np.array(seed, dtype=float), named_symmetry(symmetry))


# (symmetry, seed, total edges, edges per vertex, vertex-figure edges)
REGULAR = [
    ("hyperoctahedral", [1, 1, 1, 1], 32, 4, 6),  # tesseract: tetrahedral vertex figure
    ("hyperoctahedral", [1, 0, 0, 0], 24, 6, 12),  # 16-cell: octahedron
    ("f4", [1, 0, 0, 0], 96, 8, 12),  # 24-cell: cube
    ("h4_icosian", [1, 0, 0, 0], 720, 12, 30),  # 600-cell: icosahedron
]


@pytest.mark.parametrize(("symmetry", "seed", "n_edges", "degree", "n_figure"), REGULAR)
def test_regular_polytope_vertex_figures(
    symmetry: str, seed: list[float], n_edges: int, degree: int, n_figure: int
) -> None:
    vertices = _polytope(symmetry, seed)
    faces, _ = compute_convex_hull(vertices)
    edges = hull_edges(faces)

    assert len(edges) == n_edges
    assert edges == nearest_neighbour_edges(vertices)

    view = local_view(vertices, 0, edges, faces)
    assert len(view.neighbours) == degree
    assert len(view.figure_edges) == n_figure
    assert view.positions.shape == (degree, 3)
    assert np.allclose(view.edge_lengths, view.edge_lengths[0])


def test_tangent_projection_keeps_edge_directions_orthogonal_to_the_radius() -> None:
    vertices = _polytope("f4", [1, 0, 0, 0])
    view = local_view(vertices, 3, nearest_neighbour_edges(vertices))
    # Neighbours of a vertex of a centred polytope all sit at the same radial
    # depth, so their projected positions are equidistant from the origin.
    radii = np.linalg.norm(view.positions, axis=1)
    assert np.allclose(radii, radii[0])


def test_omnitruncated_edges_have_one_length_class_per_ringed_node() -> None:
    roots = fundamental_chamber_roots("a4")
    seed = np.linalg.solve(roots, np.array([0.2, 0.4, 0.6, 0.8]))
    vertices = generate_vertices_from_seed(seed, named_symmetry("a4"))
    faces, _ = compute_convex_hull(vertices)

    view = local_view(vertices, 0, hull_edges(faces), faces)
    assert len(view.neighbours) == 4
    assert sorted(edge_length_classes(view.edge_lengths).tolist()) == [0, 1, 2, 3]


def test_local_view_rejects_bad_index() -> None:
    vertices = _polytope("hyperoctahedral", [1, 0, 0, 0])
    with pytest.raises(IndexError):
        local_view(vertices, 99, set())
