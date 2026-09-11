from __future__ import annotations

import numpy as np

from .symmetry import SymmetryAction


def _identity() -> np.ndarray:
    return np.eye(4)


def _quaternion_left_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]],
        dtype=float,
    )


def _quaternion_right_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]],
        dtype=float,
    )


def _dodecaswirlchoric_generators() -> list[np.ndarray]:
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inverse_golden_ratio = 1.0 / golden_ratio
    binary_icosahedral = (
        (golden_ratio / 2.0, 0.5, inverse_golden_ratio / 2.0, 0.0),
        (golden_ratio / 2.0, -0.5, inverse_golden_ratio / 2.0, 0.0),
    )
    angle = 2.0 * np.pi / 20.0
    right_fivefold = (np.cos(angle), np.sin(angle), 0.0, 0.0)
    return [
        _quaternion_left_matrix(quaternion)
        for quaternion in binary_icosahedral
    ] + [_quaternion_right_matrix(right_fivefold)]


def dodecaswirl_significant_seeds() -> dict[str, np.ndarray]:
    """Return representative unit seeds for the main dodecaswirl orbit strata."""
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    axis = np.array([-golden_ratio / np.sqrt(golden_ratio**2 + 1.0),
                     -1.0 / np.sqrt(golden_ratio**2 + 1.0), 0.0])
    dot_with_x = axis[0]
    vertex_seed = np.array(
        [np.sqrt((1.0 + dot_with_x) / 2.0), 0.0, 0.0,
         axis[1] / np.sqrt(2.0 * (1.0 + dot_with_x))]
    )
    return {
        "Cross-ring seed (600 vertices)": np.array([1.0, 0.0, 0.0, 0.0]),
        "Icosahedral vertex-ring seed (240 vertices)": vertex_seed,
    }


def _swap(i: int, j: int) -> np.ndarray:
    m = np.eye(4)
    m[[i, j]] = m[[j, i]]
    return m


def _sign_flip(i: int) -> np.ndarray:
    m = np.eye(4)
    m[i, i] = -1.0
    return m


def _cycle() -> np.ndarray:
    m = np.zeros((4, 4))
    m[0, 3] = 1.0
    m[1, 0] = 1.0
    m[2, 1] = 1.0
    m[3, 2] = 1.0
    return m


def _reverse_coordinates() -> np.ndarray:
    m = np.zeros((4, 4))
    m[0, 3] = 1.0
    m[1, 2] = 1.0
    m[2, 1] = 1.0
    m[3, 0] = 1.0
    return m


def _planar_rotation(order: int) -> np.ndarray:
    angle = 2.0 * np.pi / order
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:2, :2] = ((cosine, -sine), (sine, cosine))
    return matrix


def _planar_reflection(start: int) -> np.ndarray:
    matrix = np.eye(4)
    matrix[start + 1, start + 1] = -1.0
    return matrix


def _duoprism_generators(p: int, q: int, extended: bool) -> list[np.ndarray]:
    generators = [_identity(), _planar_rotation(p), np.block([
        [np.eye(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), _planar_rotation(q)[:2, :2]],
    ])]
    if extended:
        generators.extend((_planar_reflection(0), _planar_reflection(2)))
    return generators


def _with_inversion(generators: list[np.ndarray]) -> list[np.ndarray]:
    return [*generators, -_identity()]


def _duoprism_factor_swap() -> np.ndarray:
    matrix = np.zeros((4, 4))
    matrix[0, 2] = 1.0
    matrix[1, 3] = 1.0
    matrix[2, 0] = 1.0
    matrix[3, 1] = 1.0
    return matrix


def _duoprism_orders() -> tuple[tuple[int, int], ...]:
    return tuple((p, q) for p in range(3, 7) for q in range(p, 7))


def _coxeter_roots(edges: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    gram = np.eye(4)
    for left, right, label in edges:
        value = -np.cos(np.pi / label)
        gram[left, right] = value
        gram[right, left] = value
    return np.linalg.cholesky(gram)


def _coxeter_reflections(edges: tuple[tuple[int, int, int], ...]) -> list[np.ndarray]:
    roots = _coxeter_roots(edges)
    identity = np.eye(4)
    return [identity - 2.0 * np.outer(root, root) for root in roots]


def _coxeter_rotation_generators(
    edges: tuple[tuple[int, int, int], ...],
) -> list[np.ndarray]:
    reflections = _coxeter_reflections(edges)
    return [reflections[0] @ reflection for reflection in reflections[1:]]


def _coxeter_diagram_automorphism(
    edges: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    roots = _coxeter_roots(edges)
    permutation = np.arange(4)[::-1]
    return roots[permutation].T @ np.linalg.inv(roots.T)


def _b4_ionic_generators() -> list[np.ndarray]:
    swaps = [_swap(0, 1), _swap(1, 2), _swap(2, 3)]
    sign_flips = [_sign_flip(index) for index in range(4)]
    three_cycles = [swaps[0] @ swaps[1], swaps[1] @ swaps[2]]
    return [_identity(), *sign_flips, *three_cycles]


def _h4_ionic_generators() -> list[np.ndarray]:
    return _coxeter_rotation_generators(((0, 1, 5), (1, 2, 3), (2, 3, 2)))


def fundamental_chamber_roots(name: str) -> np.ndarray | None:
    """Return simple roots for a built-in spherical Coxeter chamber."""
    edges_by_name = {
        "a4": ((0, 1, 3), (1, 2, 3), (2, 3, 3)),
        "b4": ((0, 1, 3), (1, 2, 3), (2, 3, 4)),
        "hyperoctahedral": ((0, 1, 3), (1, 2, 3), (2, 3, 4)),
        "d4": ((0, 1, 3), (1, 2, 3), (1, 3, 3)),
        "f4": ((0, 1, 3), (1, 2, 4), (2, 3, 3)),
        "h4": ((0, 1, 5), (1, 2, 3), (2, 3, 3)),
    }
    normalized_name = name.strip().lower()
    if normalized_name.startswith("duoprism_"):
        parts = normalized_name.removeprefix("duoprism_").split("_")
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            p, q = (int(part) for part in parts)
            if (p, q) in _duoprism_orders():
                edges_by_name[normalized_name] = ((0, 1, p), (1, 2, 2), (2, 3, q))
    edges = edges_by_name.get(normalized_name)
    return None if edges is None else _coxeter_roots(edges)


def named_symmetry(name: str) -> SymmetryAction:
    """Return built-in symmetry action by name."""
    name = name.strip().lower()

    aliases = {
        "a4_basic": "a4",
        "a4_chiral": "a4+",
        "a4_extended": "a4_extended",
        "a4_chiral_extended": "a4_chiral_extended",
        "a4_extended_chiral": "a4_extended_chiral",
        "b4": "hyperoctahedral",
        "b4_basic": "hyperoctahedral",
        "b4_chiral": "b4+",
        "b4_extended": "hyperoctahedral",
        "b4_chiral_extended": "b4+",
        "d4_basic": "d4",
        "d4_chiral": "d4+",
        "f4_basic": "f4",
        "f4_chiral": "f4+",
        "h4_basic": "h4",
        "h4_chiral": "h4+",
    }
    name = aliases.get(name, name)

    if name == "identity":
        gens = [_identity()]
    elif name == "coordinate_permutations":
        # Adjacent swaps generate S4 on coordinates.
        gens = [_identity(), _swap(0, 1), _swap(1, 2), _swap(2, 3)]
    elif name == "hyperoctahedral":
        # Signed permutations B4 generated by adjacent swaps + one sign flip.
        # (With swaps, one sign flip can be conjugated to all coordinates.)
        gens = [_identity(), _swap(0, 1), _swap(1, 2), _swap(2, 3), _sign_flip(0)]
    elif name == "cyclic_coordinate_rotations":
        # The 4-cycle (0 1 2 3) generates C4.
        gens = [_identity(), _cycle()]
    elif name == "dihedral_coordinate_symmetries":
        # A 4-cycle and a coordinate reversal generate D4.
        gens = [_identity(), _cycle(), _reverse_coordinates()]
    elif name == "global_inversion":
        # Central inversion maps every coordinate to its negative.
        gens = [_identity(), -_identity()]
    elif name == "decafold_dodecaswirlchoric":
        gens = _dodecaswirlchoric_generators()
    elif name == "a4_extended":
        gens = _with_inversion(_coxeter_reflections(((0, 1, 3), (1, 2, 3), (2, 3, 3))))
    elif name == "a4_chiral_extended":
        gens = _with_inversion(
            _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (2, 3, 3)))
        )
    elif name == "a4_extended_chiral":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (2, 3, 3)))
    elif name == "d4_extended":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
    elif name == "d4_extended_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
    elif name == "b4_ionic":
        gens = _b4_ionic_generators()
    elif name == "b4_prismatic_octahedral":
        gens = _coxeter_reflections(((0, 1, 4), (1, 2, 3), (2, 3, 2)))
    elif name == "b4_prismatic_octahedral_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 4), (1, 2, 3), (2, 3, 2)))
    elif name == "b4_prismatic_tetrahedral":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (2, 3, 2)))
    elif name == "b4_prismatic_tetrahedral_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (2, 3, 2)))
    elif name == "b4_half":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (1, 3, 3)))
    elif name == "b4_half_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (1, 3, 3)))
    elif name == "f4_extended":
        edges = ((0, 1, 3), (1, 2, 4), (2, 3, 3))
        gens = [*_coxeter_reflections(edges), _coxeter_diagram_automorphism(edges)]
    elif name == "f4_chiral_extended":
        edges = ((0, 1, 3), (1, 2, 4), (2, 3, 3))
        gens = [*_coxeter_rotation_generators(edges), _coxeter_diagram_automorphism(edges)]
    elif name == "f4_double_diminished":
        reflections = _coxeter_reflections(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
        gens = [reflections[0] @ reflections[1], reflections[2] @ reflections[3]]
    elif name == "f4_extended_double_diminished":
        edges = ((0, 1, 3), (1, 2, 4), (2, 3, 3))
        reflections = _coxeter_reflections(edges)
        gens = [reflections[0] @ reflections[1], reflections[2] @ reflections[3],
                _coxeter_diagram_automorphism(edges)]
    elif name == "h4_prismatic":
        gens = _coxeter_reflections(((0, 1, 5), (1, 2, 3), (2, 3, 2)))
    elif name == "h4_prismatic_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 5), (1, 2, 3), (2, 3, 2)))
    elif name == "h4_ionic":
        gens = _h4_ionic_generators()
    elif name == "h4_half":
        gens = _coxeter_reflections(((0, 1, 5), (1, 2, 3), (2, 3, 2)))
    elif name == "h4_half_chiral":
        gens = _coxeter_rotation_generators(((0, 1, 5), (1, 2, 3), (2, 3, 2)))
    elif name.startswith("duoprism_"):
        suffix = ""
        for candidate in ("_chiral_extended", "_extended", "_chiral"):
            if name.endswith(candidate):
                suffix = candidate
                name = name[: -len(candidate)]
                break
        is_chiral = name.endswith("+") or suffix in ("_chiral", "_chiral_extended")
        name = name.removesuffix("+")
        parts = name.removeprefix("duoprism_").split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid duoprism symmetry '{name}'")
        p, q = (int(part) for part in parts)
        if (p, q) not in _duoprism_orders():
            raise ValueError(f"Unsupported duoprism orders ({p}, {q})")
        gens = _duoprism_generators(p, q, not is_chiral)
        if suffix in ("_extended", "_chiral_extended"):
            if p != q:
                raise ValueError("Duoprism factor-swap extensions require p == q")
            gens.append(_duoprism_factor_swap())
    elif name == "a4":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (2, 3, 3)))
    elif name == "d4":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (1, 3, 3)))
    elif name == "f4":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
    elif name == "h4":
        gens = _coxeter_reflections(((0, 1, 5), (1, 2, 3), (2, 3, 3)))
    elif name == "a4+":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (2, 3, 3)))
    elif name == "b4+":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (2, 3, 4)))
    elif name == "d4+":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 3), (1, 3, 3)))
    elif name == "f4+":
        gens = _coxeter_rotation_generators(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
    elif name == "h4+":
        gens = _coxeter_rotation_generators(((0, 1, 5), (1, 2, 3), (2, 3, 3)))
    else:
        raise ValueError(
            f"Unknown symmetry '{name}'. "
            f"Choose one of: {', '.join(available_symmetries())}"
        )

    return SymmetryAction.from_iterable(gens)


def available_symmetries() -> list[str]:
    symmetries = [
        "identity",
        "coordinate_permutations",
        "cyclic_coordinate_rotations",
        "dihedral_coordinate_symmetries",
        "global_inversion",
        "decafold_dodecaswirlchoric",
        "a4",
        "hyperoctahedral",
        "d4",
        "f4",
        "h4",
        "a4+",
        "b4+",
        "d4+",
        "f4+",
        "h4+",
        "a4_basic",
        "a4_chiral",
        "a4_extended",
        "a4_chiral_extended",
        "a4_extended_chiral",
        "b4",
        "b4_basic",
        "b4_chiral",
        "b4_extended",
        "b4_chiral_extended",
        "d4_basic",
        "d4_chiral",
        "d4_extended",
        "d4_extended_chiral",
        "f4_basic",
        "f4_chiral",
        "f4_extended",
        "f4_chiral_extended",
        "f4_double_diminished",
        "f4_extended_double_diminished",
        "h4_basic",
        "h4_chiral",
        "h4_prismatic",
        "h4_prismatic_chiral",
        "h4_ionic",
        "h4_half",
        "h4_half_chiral",
        "b4_ionic",
        "b4_half",
        "b4_half_chiral",
        "b4_prismatic_octahedral",
        "b4_prismatic_octahedral_chiral",
        "b4_prismatic_tetrahedral",
        "b4_prismatic_tetrahedral_chiral",
    ]
    for p, q in _duoprism_orders():
        symmetries.extend((f"duoprism_{p}_{q}", f"duoprism_{p}_{q}+"))
        if p == q:
            symmetries.extend(
                (
                    f"duoprism_{p}_{q}_chiral",
                    f"duoprism_{p}_{q}_extended",
                    f"duoprism_{p}_{q}_chiral_extended",
                )
            )
    return symmetries
