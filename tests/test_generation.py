import numpy as np

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import (
    available_symmetries,
    dodecaswirl_significant_seeds,
    fundamental_chamber_roots,
    named_symmetry,
)
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


def test_4off_uses_precision_without_exponent_notation() -> None:
    verts = np.array([[1.2345678901234567, 1e-8, 1e20, -1.2e-7]])
    lines = to_4off(verts).strip().splitlines()

    assert lines[1] == "1 0 0"
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


def test_decafold_dodecaswirlchoric_has_order_1200() -> None:
    name = "decafold_dodecaswirlchoric"
    action = named_symmetry(name)
    assert name in available_symmetries()
    assert all(np.linalg.det(generator) > 0 for generator in action.generators)

    seen = {tuple(np.eye(4).round(8).ravel())}
    pending = [np.eye(4)]
    while pending:
        current = pending.pop()
        for generator in action.generators:
            transformed = generator @ current
            key = tuple(transformed.round(8).ravel())
            if key not in seen:
                seen.add(key)
                pending.append(transformed)

    assert len(seen) == 1200


def test_dodecaswirl_significant_seeds_have_verified_orbit_sizes() -> None:
    action = named_symmetry("decafold_dodecaswirlchoric")
    expected_sizes = {
        "Cross-ring seed (600 vertices)": 600,
        "Icosahedral vertex-ring seed (240 vertices)": 240,
    }
    for label, seed in dodecaswirl_significant_seeds().items():
        assert np.isclose(np.linalg.norm(seed), 1.0)
        vertices = generate_vertices_from_seed(seed, action)
        assert len(vertices) == expected_sizes[label]


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
