"""The convex hull against polytopes whose vertex, face and cell counts are known independently.

Face counts come from the cells: e.g. the truncated 600-cell has 600 truncated tetrahedra
(4 triangles, 4 hexagons) and 120 icosahedra (20 triangles), each face in two cells, so 1200
hexagons + 2400 triangles. A Wythoff seed's combinatorics depend only on which chamber walls it
is off, so any positive weights do.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from hull_checks import check_hull

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import fundamental_chamber_roots, named_symmetry
from four_d_vertex_generator.off import compute_convex_hull

# (group, ringed chamber nodes) -> (vertices, faces, cells)
WYTHOFF = {
    ("h4", (0,)): (600, 720, 120),            # 120-cell
    ("h4", (3,)): (120, 1200, 600),           # 600-cell
    ("h4", (1,)): (1200, 3120, 720),          # rectified 120-cell
    ("h4", (2,)): (720, 3600, 720),           # rectified 600-cell
    ("h4", (0, 1)): (2400, 3120, 720),        # truncated 120-cell
    ("h4", (2, 3)): (1440, 3600, 720),        # truncated 600-cell
    ("h4", (1, 2)): (3600, 4320, 720),        # bitruncated 120-cell
    ("h4", (0, 2)): (3600, 9120, 1920),       # cantellated 120-cell
    ("h4", (1, 3)): (3600, 8640, 1440),       # cantellated 600-cell
    ("h4", (0, 3)): (2400, 7440, 2640),       # runcinated 120-cell
    ("h4", (0, 1, 2, 3)): (14400, 17040, 2640),  # omnitruncated 120-cell
    ("b4", (3,)): (16, 24, 8),                # tesseract
    ("b4", (0,)): (8, 32, 16),                # 16-cell
    ("b4", (1,)): (24, 96, 24),               # 24-cell (rectified 16-cell)
    ("b4", (2,)): (32, 88, 24),               # rectified tesseract
    ("b4", (2, 3)): (64, 88, 24),             # truncated tesseract
    ("b4", (0, 1, 2, 3)): (384, 464, 80),     # omnitruncated tesseract
    ("a4", (0,)): (5, 10, 5),                 # 5-cell
    ("a4", (1,)): (10, 30, 10),               # rectified 5-cell
    ("a4", (0, 1, 2, 3)): (120, 150, 30),     # omnitruncated 5-cell
    ("f4", (0,)): (24, 96, 24),               # 24-cell
    ("f4", (0, 1, 2, 3)): (1152, 1392, 240),  # omnitruncated 24-cell
    ("duoprism_5_6", (0, 1, 2, 3)): (120, 142, 22),  # 10-12 duoprism: 120 squares, 10 + 12 gons
}


def wythoff_vertices(group: str, rings: tuple[int, ...]) -> np.ndarray:
    roots = fundamental_chamber_roots(group)
    weights = np.zeros(4)
    weights[list(rings)] = 1.0 + 0.1 * np.arange(len(rings))  # deliberately not uniform
    seed = np.linalg.solve(roots, weights)
    return generate_vertices_from_seed(seed / np.linalg.norm(seed), named_symmetry(group))


@pytest.mark.parametrize(("group", "rings"), list(WYTHOFF))
def test_wythoff_polytopes_have_their_known_counts(group: str, rings: tuple[int, ...]) -> None:
    verts = wythoff_vertices(group, rings)
    faces, cells = compute_convex_hull(verts)
    assert (len(verts), len(faces), len(cells)) == WYTHOFF[(group, rings)]
    check_hull(verts, faces, cells, named_symmetry(group).generators)


def test_grand_antiprism() -> None:
    # The 600-cell minus two orthogonal rings of 10: 20 pentagonal antiprisms and 300 tetrahedra.
    verts = wythoff_vertices("h4", (3,))
    a = verts[0]
    edge = np.min(np.linalg.norm(verts[1:] - a, axis=1))
    b = verts[np.argmin(np.abs(np.linalg.norm(verts - a, axis=1) - edge))]
    basis = np.linalg.qr(np.stack([a, b]).T)[0].T          # the great circle through an edge
    in_plane = np.linalg.norm(verts - (verts @ basis.T) @ basis, axis=1) < 1e-9
    off_plane = np.linalg.norm(verts @ basis.T, axis=1) < 1e-9  # the ring in the orthogonal plane
    assert in_plane.sum() == off_plane.sum() == 10
    keep = verts[~in_plane & ~off_plane]
    faces, cells = compute_convex_hull(keep)
    assert (len(keep), len(faces), len(cells)) == (100, 720, 320)
    check_hull(keep, faces, cells)


@pytest.mark.parametrize("n", [12, 40])
def test_prism_over_a_random_polyhedron(n: int) -> None:
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(n)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    tri = len(ConvexHull(p).simplices)                        # random points: all faces triangles
    edges = 3 * tri // 2
    verts = np.vstack([np.c_[p, -np.ones(n) * 0.7], np.c_[p, np.ones(n) * 0.7]])
    faces, cells = compute_convex_hull(verts)
    # two caps and a triangular prism per triangle; faces: each triangle twice, a square per edge
    assert (len(faces), len(cells)) == (2 * tri + edges, tri + 2)
    check_hull(verts, faces, cells)


@pytest.mark.parametrize("seed", range(3))
def test_random_points_give_a_simplicial_hull(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((300, 4))
    pts[:200] /= np.linalg.norm(pts[:200], axis=1, keepdims=True)  # 100 points stay inside
    faces, cells = compute_convex_hull(pts)
    assert all(len(f) == 3 for f in faces) and all(len(c) == 4 for c in cells)
    assert len(faces) == 2 * len(cells)
    check_hull(pts, faces, cells)


def test_every_orbit_of_a_random_seed_closes_up_symmetrically() -> None:
    rng = np.random.default_rng(7)
    for name in ("a4", "b4", "d4", "f4", "h4_swirlprism", "h4_icosian+", "duoprism_4_6"):
        seed = rng.standard_normal(4)
        verts = generate_vertices_from_seed(seed, named_symmetry(name))
        faces, cells = compute_convex_hull(verts)
        check_hull(verts, faces, cells, named_symmetry(name).generators)


def test_ring_choices_are_exhaustive_for_a4() -> None:
    # all 15 Wythoff ring sets of A4 close up symmetrically
    for k in range(1, 5):
        for rings in itertools.combinations(range(4), k):
            verts = wythoff_vertices("a4", rings)
            faces, cells = compute_convex_hull(verts)
            check_hull(verts, faces, cells, named_symmetry("a4").generators)
