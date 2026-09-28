import numpy as np
import pytest

from four_d_vertex_generator.generation import (
    generate_vertices_from_seed,
    group_order,
    is_subgroup,
)
from four_d_vertex_generator.isogonal import compute_orbits, detect_symmetries, matches_symmetry
from four_d_vertex_generator.library import (
    SYMMETRY_ALIASES,
    available_symmetries,
    fundamental_chamber_roots,
    h4_swirlprism_anchor_seed,
    h4_swirlprism_predefined_seed,
    named_symmetry,
)
from four_d_vertex_generator.off import to_4off
from four_d_vertex_generator.symmetry import SymmetryAction


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
    assert lines[1] == "2 0 0 0"


def test_4off_uses_precision_without_exponent_notation() -> None:
    verts = np.array([[1.2345678901234567, 1e-8, 1e20, -1.2e-7]])
    lines = to_4off(verts).strip().splitlines()

    assert lines[1] == "1 0 0 0"
    assert "e" not in lines[2].lower()
    assert lines[2].split() == [
        "1.2345678901234567",
        "0.00000001",
        "100000000000000000000",
        "-0.00000011999999999999999",
    ]


def test_additional_symmetries_resolve_and_have_expected_axis_orbits() -> None:
    seed = np.array([1.0, 0.0, 0.0, 0.0])
    expected_sizes = {
        "cyclic_coordinate_rotations": 4,
        "dihedral_coordinate_symmetries": 4,
        "global_inversion": 2,
    }

    for name, expected_size in expected_sizes.items():
        assert name in available_symmetries()
        vertices = generate_vertices_from_seed(seed, named_symmetry(name))
        assert len(vertices) == expected_size


def test_four_dimensional_coxeter_symmetries_are_available() -> None:
    for name in ("a4", "d4", "f4", "h4"):
        action = named_symmetry(name)
        assert name in available_symmetries()
        assert len(action.generators) == 4


def test_chiral_coxeter_symmetries_are_orientation_preserving() -> None:
    for name in ("a4+", "b4+", "d4+", "f4+", "h4+"):
        action = named_symmetry(name)
        assert name in available_symmetries()
        assert all(np.linalg.det(generator) > 0 for generator in action.generators)


def test_h4_swirlprism_anchor_seed_aligns_with_600_cell_vertex() -> None:
    anchor = h4_swirlprism_anchor_seed()
    assert np.isclose(np.linalg.norm(anchor), 1.0)
    six_hundred_cell = generate_vertices_from_seed(anchor, named_symmetry("h4_icosian"))
    assert len(six_hundred_cell) == 120


def test_h4_swirlprism_predefined_seed_starts_at_anchor() -> None:
    seed = h4_swirlprism_predefined_seed(0.0, 0.0, 0.0)
    assert np.allclose(seed, h4_swirlprism_anchor_seed())
    assert len(generate_vertices_from_seed(seed, named_symmetry("h4_swirlprism+"), tol=1e-6)) == 120


def test_h4_swirlprism_cross_ring_sliders_split_the_anchor_orbit() -> None:
    action = named_symmetry("h4_swirlprism+")
    for values in ((30.0, 0.0, 0.0), (0.0, 30.0, 0.0), (30.0, 30.0, 0.0)):
        seed = h4_swirlprism_predefined_seed(*values)
        assert len(generate_vertices_from_seed(seed, action, tol=1e-6)) == 600


def test_h4_swirlprism_main_ring_slider_does_not_split_the_anchor_orbit() -> None:
    seed = h4_swirlprism_predefined_seed(0.0, 0.0, 30.0)
    assert len(generate_vertices_from_seed(seed, named_symmetry("h4_swirlprism+"), tol=1e-6)) == 120


def test_h4_pentagonal_swirl_is_a_genuine_h4_subgroup() -> None:
    # h4_pentagonal_swirl/_ring are verified subgroups of H4: they must
    # actually preserve a genuine 600-cell vertex set.
    assert group_order(named_symmetry("h4_pentagonal_swirl")) == 50
    assert group_order(named_symmetry("h4_pentagonal_swirl_ring")) == 10

    six_hundred_cell = generate_vertices_from_seed(
        np.array([1.0, 0.0, 0.0, 0.0]), named_symmetry("h4_icosian"), max_vertices=20000
    )
    assert len(six_hundred_cell) == 120

    for name in ("h4_pentagonal_swirl", "h4_pentagonal_swirl_ring"):
        action = named_symmetry(name)
        assert matches_symmetry(six_hundred_cell, action, tol=1e-6)

    # The ring subgroup partitions the 600-cell into exactly the 12 rings of
    # 10 vertices described by the classic pentagonal-swirl decomposition.
    ring_partition = compute_orbits(
        six_hundred_cell, named_symmetry("h4_pentagonal_swirl_ring"), tol=1e-6
    )
    assert ring_partition.num_orbits == 12
    counts: dict[int, int] = {}
    for orbit_id in ring_partition.orbit_ids:
        counts[orbit_id] = counts.get(orbit_id, 0) + 1
    assert set(counts.values()) == {10}


def test_h4_swirlprism_has_order_1200_and_is_vertex_transitive_on_600_cell() -> None:
    # This is the real "small swirlprism" [5,3:5] symmetry: h4_swirlprism/+
    # are verified subgroups of H4 that share all of the 600-cell's vertices
    # as a single orbit.
    assert group_order(named_symmetry("h4_swirlprism+")) == 600
    assert group_order(named_symmetry("h4_swirlprism")) == 1200

    six_hundred_cell = generate_vertices_from_seed(
        np.array([1.0, 0.0, 0.0, 0.0]), named_symmetry("h4_icosian"), max_vertices=20000
    )
    assert len(six_hundred_cell) == 120

    for name in ("h4_swirlprism+", "h4_swirlprism"):
        action = named_symmetry(name)
        assert matches_symmetry(six_hundred_cell, action, tol=1e-6)
        partition = compute_orbits(six_hundred_cell, action, tol=1e-6)
        assert partition.num_orbits == 1


def test_hyperoctahedral_chiral_symmetry_uses_b4_name() -> None:
    action = named_symmetry("b4+")
    assert len(action.generators) == 3
    assert all(np.linalg.det(generator) > 0 for generator in action.generators)


def test_duoprism_symmetries_are_available_for_orders_three_through_six() -> None:
    for p in range(3, 7):
        for q in range(p, 7):
            basic_name = f"duoprism_{p}_{q}+"
            extended_name = f"duoprism_{p}_{q}"
            assert basic_name in available_symmetries()
            assert extended_name in available_symmetries()
            assert len(named_symmetry(basic_name).generators) == 3
            assert len(named_symmetry(extended_name).generators) == 5


def test_triangular_duoprism_product_vertex_has_nine_vertices() -> None:
    seed = np.array([1.0, 0.0, 1.0, 0.0])
    vertices = generate_vertices_from_seed(seed, named_symmetry("duoprism_3_3"))

    assert len(vertices) == 9


def test_duoprism_chambers_are_available_for_supported_orders() -> None:
    for p in range(3, 7):
        for q in range(p, 7):
            roots = fundamental_chamber_roots(f"duoprism_{p}_{q}")
            assert roots is not None
            assert roots.shape == (4, 4)
            assert np.isfinite(roots).all()


def test_a4_extension_doubles_order_but_b4_has_no_larger_linear_extension() -> None:
    def group_order(name: str) -> int:
        action = named_symmetry(name)
        identity = np.eye(4)
        seen = {tuple(identity.round(8).ravel())}
        pending = [identity]
        while pending:
            current = pending.pop()
            for generator in action.generators:
                transformed = generator @ current
                key = tuple(transformed.round(8).ravel())
                if key not in seen:
                    seen.add(key)
                    pending.append(transformed)
        return len(seen)

    assert group_order("a4") == 120
    assert group_order("a4_extended") == 240
    assert group_order("hyperoctahedral") == 384
    assert group_order("b4_extended") == 384


def test_new_subgroup_generators_have_reference_group_orders() -> None:
    expected_orders = {
        "f4_double_diminished": 288,
        "f4_extended_double_diminished": 576,
        "h4_ionic": 120,
    }

    for name, expected_order in expected_orders.items():
        action = named_symmetry(name)
        identity = np.eye(4)
        seen = {tuple(identity.round(8).ravel())}
        pending = [identity]
        while pending:
            current = pending.pop()
            for generator in action.generators:
                transformed = generator @ current
                key = tuple(transformed.round(8).ravel())
                if key not in seen:
                    seen.add(key)
                    pending.append(transformed)
        assert len(seen) == expected_order


def test_generated_coordinates_below_ten_digits_are_zeroed() -> None:
    action = named_symmetry("identity")
    seed = np.array([1.0e-11, -1.0e-10, 1.0e-9, 1.0])
    vertices = generate_vertices_from_seed(seed, action)

    assert vertices.tolist() == [[0.0, -1.0e-10, 1.0e-9, 1.0]]


def _chamber_seed(name: str, weights: list[float]) -> np.ndarray:
    return np.linalg.solve(fundamental_chamber_roots(name), np.array(weights, dtype=float))


CHAMBER_NAMES = [
    "a4",
    "b4",
    "hyperoctahedral",
    "d4",
    "f4",
    "h4",
    "b4_prismatic_octahedral",
    "b4_prismatic_tetrahedral",
    "h4_prismatic",
    *(f"duoprism_{p}_{q}" for p in range(3, 7) for q in range(p, 7)),
]


@pytest.mark.parametrize("name", CHAMBER_NAMES)
def test_chamber_roots_are_mirrors_of_their_own_group(name: str) -> None:
    roots = fundamental_chamber_roots(name)
    reflections = SymmetryAction.from_iterable(
        np.eye(4) - 2.0 * np.outer(root, root) / (root @ root) for root in roots
    )
    action = named_symmetry(name)
    assert is_subgroup(reflections, action)
    assert is_subgroup(action, reflections)


@pytest.mark.parametrize(
    ("name", "weights", "expected"),
    [
        ("hyperoctahedral", [1, 0, 0, 0], 8),  # 16-cell
        ("hyperoctahedral", [0, 0, 0, 1], 16),  # tesseract
        ("h4", [0, 0, 0, 1], 120),  # 600-cell
        ("h4", [1, 0, 0, 0], 600),  # 120-cell
        ("duoprism_3_3", [0, 1, 0, 1], 9),
        ("duoprism_3_5", [0, 1, 0, 1], 15),
        ("duoprism_5_5", [0, 1, 0, 1], 25),
    ],
)
def test_chamber_corners_give_regular_vertex_counts(
    name: str, weights: list[float], expected: int
) -> None:
    vertices = generate_vertices_from_seed(_chamber_seed(name, weights), named_symmetry(name))
    assert len(vertices) == expected


def test_is_subgroup_distinguishes_coordinate_bases() -> None:
    assert is_subgroup(named_symmetry("b4_ionic"), named_symmetry("hyperoctahedral"))
    assert is_subgroup(named_symmetry("b4+"), named_symmetry("b4"))
    # Same abstract group, different coordinate embeddings.
    assert not is_subgroup(named_symmetry("hyperoctahedral"), named_symmetry("b4"))
    assert not is_subgroup(named_symmetry("h4_icosian"), named_symmetry("h4"))


def test_aliases_resolve_to_the_same_group() -> None:
    for alias, target in SYMMETRY_ALIASES.items():
        assert group_order(named_symmetry(alias)) == group_order(named_symmetry(target))
    assert "a4_basic" not in available_symmetries(include_aliases=False)
    assert "duoprism_3_3_chiral" not in available_symmetries(include_aliases=False)
    assert "a4_basic" in available_symmetries()


def test_h4_ionic_is_not_the_chiral_prismatic_group() -> None:
    ionic = named_symmetry("h4_ionic")
    assert group_order(ionic) == 120
    assert any(np.linalg.det(g) < 0 for g in ionic.generators)
    assert is_subgroup(ionic, named_symmetry("h4_prismatic"))
    assert not is_subgroup(ionic, named_symmetry("h4_prismatic_chiral"))


def test_detect_symmetries_skips_aliases_by_default() -> None:
    tesseract = generate_vertices_from_seed(
        np.array([1.0, 1.0, 1.0, 1.0]), named_symmetry("hyperoctahedral")
    )
    detected = detect_symmetries(tesseract)
    assert detected[0] == "hyperoctahedral"
    assert not any(name in SYMMETRY_ALIASES for name in detected)


def test_generated_coordinates_have_no_floating_point_dust() -> None:
    vertices = generate_vertices_from_seed(np.array([1.0, 0.0, 0.0, 0.0]), named_symmetry("f4"))
    nonzero = np.abs(vertices[vertices != 0.0])
    assert nonzero.min() > 1e-10
