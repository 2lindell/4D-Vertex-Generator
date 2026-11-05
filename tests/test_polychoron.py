from __future__ import annotations

import numpy as np
import pytest

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import parse_4off, to_4off
from scipy.spatial.distance import pdist

from four_d_vertex_generator.polychoron import (
    build_polychoron,
    corner_classes,
    compare_to_hull,
    edge_orbits,
    find_closed_polychora,
    hull_edge_orbit_indices,
    reflection_edge_orbit_indices,
    rotation_edge_orbit_indices,
)


def _setup(name: str, seed: list[float]):
    action = named_symmetry(name)
    verts = generate_vertices_from_seed(np.array(seed, dtype=float), action)
    return action, verts, edge_orbits(verts, action)


def test_edge_orbits_tesseract() -> None:
    _, verts, orbits = _setup("hyperoctahedral", [1, 1, 1, 1])
    assert [len(o.edges) for o in orbits] == [32, 48, 32, 8]
    assert orbits[0].length == pytest.approx(2.0)
    assert sum(len(o.edges) for o in orbits) == 16 * 15 // 2


@pytest.mark.parametrize(
    ("name", "seed", "faces", "cells"),
    [
        ("hyperoctahedral", [1, 1, 1, 1], 24, 8),
        ("f4", [1, 0, 0, 0], 96, 24),
        ("h4", [1, 0, 0, 0], 1200, 600),
        ("hyperoctahedral", [2, 1, 1, 0], 240, 48),
        ("hyperoctahedral", [3, 2, 1, 0.5], 464, 80),
    ],
)
def test_hull_edge_orbits_reproduce_convex_hull(name, seed, faces, cells) -> None:
    action, verts, orbits = _setup(name, seed)
    poly = build_polychoron(verts, action, hull_edge_orbit_indices(verts, orbits), orbits=orbits)
    assert poly.problems == []
    assert poly.euler == 0
    assert (len(poly.faces), len(poly.cells)) == (faces, cells)
    assert compare_to_hull(poly) == []


def test_swirlprism_seed_matches_hull() -> None:
    action, verts, orbits = _setup("h4_swirlprism", [1.0, 0.31, 0.17, 0.05])
    poly = build_polychoron(verts, action, hull_edge_orbit_indices(verts, orbits), orbits=orbits)
    assert poly.is_closed
    assert compare_to_hull(poly) == []


def test_search_finds_hull_edge_set() -> None:
    action, verts, orbits = _setup("f4", [1, 0, 0, 0])
    found = find_closed_polychora(verts, action, max_orbits=2)
    assert tuple(hull_edge_orbit_indices(verts, orbits)) in [combo for combo, _ in found]


def test_unclosed_edge_set_is_reported_not_raised() -> None:
    action, verts, orbits = _setup("hyperoctahedral", [2, 1, 1, 0])
    poly = build_polychoron(verts, action, [len(orbits) - 1], orbits=orbits)
    assert not poly.is_closed


def test_result_exports_to_4off() -> None:
    action, verts, orbits = _setup("hyperoctahedral", [1, 1, 1, 1])
    poly = build_polychoron(verts, action, [0], orbits=orbits)
    text = to_4off(poly.vertices, poly.faces, poly.cells)
    assert text.splitlines()[1] == "16 24 0 8"
    assert len(parse_4off(text)) == 16


def _edge_lengths(verts: np.ndarray, face: list[int]) -> set[float]:
    return {
        round(float(np.linalg.norm(verts[a] - verts[b])), 3)
        for a, b in zip(face, face[1:] + face[:1])
    }


def test_star_duoprism_5_2_5() -> None:
    # {5/2}x{5}: pentagram chords in one plane (orbit 4), pentagon edges in the other (orbit 0)
    action, verts, orbits = _setup("duoprism_5_5", [1, 0, 1, 0])
    chord, edge = orbits[4], orbits[0]
    assert chord.length == pytest.approx(edge.length * (1 + 5**0.5) / 2)

    # cells of a star prism lie on hyperplanes that cut through the polytope
    assert not build_polychoron(verts, action, [0, 4], orbits=orbits).is_closed

    poly = build_polychoron(verts, action, [0, 4], orbits=orbits, convex=False)
    assert poly.is_closed
    assert (len(poly.edges), len(poly.faces), len(poly.cells), poly.euler) == (50, 35, 10, 0)
    assert sorted(len(f) for f in poly.faces).count(4) == 25
    pentagrams = [f for f in poly.faces if len(f) == 5 and _edge_lengths(verts, f) == {chord.length.__round__(3)}]
    assert len(pentagrams) == 5
    assert all(len(c) == 7 for c in poly.cells)
    assert compare_to_hull(poly) != []


def test_star_duoprism_exports_to_4off() -> None:
    action, verts, orbits = _setup("duoprism_5_5", [1, 0, 1, 0])
    poly = build_polychoron(verts, action, [0, 4], orbits=orbits, convex=False)
    assert to_4off(poly.vertices, poly.faces, poly.cells).splitlines()[1] == "25 35 0 10"


def _h4_star(orbit: int, corner_angle_deg: int, choice: int):
    action, verts, orbits = _setup("h4", [1, 0, 0, 0])
    classes = corner_classes(verts, action, orbits[orbit].edges)
    corner = next(i for i, c in enumerate(classes) if round(np.degrees(c.angle)) == corner_angle_deg)
    poly = build_polychoron(
        verts, action, [orbit], orbits=orbits, convex=False, face_corners=[corner], cell_choice=choice
    )
    return verts, orbits[orbit], poly


def _cell_edge_ratio(verts: np.ndarray, poly, edge_length: float) -> float:
    cell = poly.cells[0]
    ids = sorted({x for f in cell for x in poly.faces[f]})
    return edge_length / float(pdist(verts[ids]).min())


@pytest.mark.parametrize(
    ("orbit", "choice", "cells", "chi", "ratio"),
    [
        (0, 0, 120, 480, 1.0),  # icosahedral 120-cell {3,5,5/2}: convex icosahedra
        (4, 0, 120, 480, 1.618),  # great icosahedral 120-cell {3,5/2,5}: great icosahedra
        (4, 1, 600, 0, 1.0),  # grand 600-cell {3,3,5/2}: tetrahedra
    ],
)
def test_h4_star_polychora_on_the_600_cell_vertices(orbit, choice, cells, chi, ratio) -> None:
    verts, edge_orbit, poly = _h4_star(orbit, 60, choice)
    assert poly.is_closed
    assert poly.cell_choices == 2
    assert (len(verts), len(poly.edges), len(poly.faces), len(poly.cells)) == (120, 720, 1200, cells)
    assert poly.euler == chi
    assert _cell_edge_ratio(verts, poly, edge_orbit.length) == pytest.approx(ratio, abs=1e-3)


def _non_uniform_duoprism(orbit_lengths: tuple[float, ...]):
    # seed (1, 0.3, 1, 0): a decagon with alternating edge lengths times a pentagon
    action, verts, orbits = _setup("duoprism_5_5", [1, 0.3, 1, 0])
    indices = [next(i for i, o in enumerate(orbits) if o.length == pytest.approx(x, abs=5e-4)) for x in orbit_lengths]
    edges = sorted({e for i in indices for e in orbits[i].edges})
    corners = list(range(len(corner_classes(verts, action, edges))))
    poly = build_polychoron(
        verts, action, indices, orbits=orbits, convex=False, face_corners=corners
    )
    return verts, orbits, indices, poly


@pytest.mark.parametrize(
    ("lengths", "differs_from_hull"),
    [
        ((0.6, 0.69, 1.176), False),  # convex decagon x pentagon, unequal edges in the decagon
        ((0.6, 0.69, 1.902), True),  # decagon x pentagram
        ((0.6, 1.661, 1.176), True),  # alternating edges, one a decagon chord
        ((1.661, 1.717, 1.902), True),  # decagon of two chord lengths x pentagram
    ],
)
def test_non_uniform_duoprism_with_unequal_edges(lengths, differs_from_hull) -> None:
    verts, _, _, poly = _non_uniform_duoprism(lengths)
    assert poly.is_closed
    assert poly.euler == 0
    assert (len(poly.edges), len(poly.faces), len(poly.cells)) == (100, 65, 15)
    assert sorted(len(f) for f in poly.faces).count(10) == 5
    assert (compare_to_hull(poly) != []) == differs_from_hull


def test_reflection_edge_orbits_prune_the_search() -> None:
    action, verts, orbits = _setup("duoprism_5_5", [1, 0.3, 1, 0])
    reflections = reflection_edge_orbit_indices(verts, action, orbits)
    assert (len(orbits), len(reflections)) == (23, 7)
    found = find_closed_polychora(verts, action, max_orbits=3, reflections_only=True, star=True)
    assert len(found) == 20
    assert all((len(p.edges), len(p.faces), len(p.cells)) == (100, 65, 15) for _, p in found)


def test_swirlprism_group_has_no_reflections() -> None:
    action, verts, orbits = _setup("h4_swirlprism", [1.0, 0.31, 0.17, 0.05])
    assert reflection_edge_orbit_indices(verts, action, orbits) == []


def test_rotation_edge_orbits_need_no_mirrors() -> None:
    action, verts, orbits = _setup("duoprism_5_5+", [1, 0.3, 1, 0.2])
    assert (len(orbits), len(rotation_edge_orbit_indices(verts, action, orbits))) == (12, 4)
    found = find_closed_polychora(verts, action, max_orbits=3, rotations_only=True, star=True)
    # pentagon x pentagon plus the three pentagram variants
    assert len(found) == 4
    assert [bool(compare_to_hull(p)) for _, p in found].count(False) == 1
    assert all((len(p.edges), len(p.faces), len(p.cells)) == (50, 35, 10) for _, p in found)


def test_swirlprism_has_rotation_edge_candidates() -> None:
    action, verts, orbits = _setup("h4_swirlprism+", [1.0, 0.31, 0.17, 0.05])
    assert 0 < len(rotation_edge_orbit_indices(verts, action, orbits)) < 20
