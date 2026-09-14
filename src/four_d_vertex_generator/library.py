from __future__ import annotations

from functools import lru_cache

import numpy as np

from .generation import _finite_group_elements
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


def _h4_icosian_chiral_generators() -> list[np.ndarray]:
    """Rotation subgroup (order 7200) of H4 on standard 600-cell icosian coordinates."""
    binary_icosahedral = (
        (-0.809016994374947, -0.309016994374947, 0.0, 0.5),
        (0.809016994374947, -0.309016994374947, 0.0, -0.5),
    )
    return [_quaternion_left_matrix(q) for q in binary_icosahedral] + [
        _quaternion_right_matrix(q) for q in binary_icosahedral
    ]


def _h4_icosian_generators() -> list[np.ndarray]:
    """Full H4 (order 14400) on standard 600-cell icosian coordinates.

    This is a different basis/embedding than "h4" (built from Coxeter simple
    roots), so a vertex set using the classic (0,+-1,+-phi,+-1/phi)-style
    600-cell coordinates matches this candidate rather than "h4" directly.
    """
    return _h4_icosian_chiral_generators() + [np.diag([1.0, -1.0, -1.0, -1.0])]


# A genuine order-10 element of the binary icosahedral group 2I (a 5-fold
# icosahedral rotation lifted to its double cover, so its quaternion order is
# 10 rather than 5). Verified by BFS closure: 2I has exactly 24 such elements.
_ICOSIAN_ORDER_TEN = (0.809016994374947, -0.309016994374947, 0.0, -0.5)


def _pentagonal_swirl_generators() -> list[np.ndarray]:
    """Z10 x Z10 pentagonal swirl subgroup of H4 (icosian basis, order 50).

    Independent left/right multiplication by an order-10 element of 2I. This
    is a genuine subgroup of h4/h4_icosian. Splitting a 600-cell-derived
    vertex set by this action traces its pentagonal swirl rings.
    """
    return [
        _quaternion_left_matrix(_ICOSIAN_ORDER_TEN),
        _quaternion_right_matrix(_ICOSIAN_ORDER_TEN),
    ]


def _pentagonal_swirl_ring_generators() -> list[np.ndarray]:
    """Single Z10 (order 10) one-sided icosian multiplication.

    This is the subgroup whose orbits are exactly the 12 rings of 10 vertices
    that partition the classic 600-cell (right-coset decomposition of 2I by
    a Z10 subgroup).
    """
    return [_quaternion_right_matrix(_ICOSIAN_ORDER_TEN)]


def _h4_swirlprism_chiral_generators() -> list[np.ndarray]:
    """Order-600 chiral "small swirlprism" subgroup of H4 (icosian basis).

    Left multiplication by the full 2I (icosahedral rotation double cover)
    combined with right multiplication by a single genuine 2I order-10
    element. Vertex-transitive on the classic 600-cell -- i.e. it genuinely
    shares the 600-cell's vertices, matching the "small swirlprism" [5,3:5]
    construction.
    """
    binary_icosahedral = (
        (-0.809016994374947, -0.309016994374947, 0.0, 0.5),
        (0.809016994374947, -0.309016994374947, 0.0, -0.5),
    )
    return [_quaternion_left_matrix(q) for q in binary_icosahedral] + [
        _quaternion_right_matrix(_ICOSIAN_ORDER_TEN)
    ]


def _h4_swirlprism_generators() -> list[np.ndarray]:
    """Full order-1200 "small swirlprism" subgroup of H4 (icosian basis)."""
    return _h4_swirlprism_chiral_generators() + [
        _quaternion_right_matrix((0.0, 0.0, 1.0, 0.0))
    ]


def h4_swirlprism_anchor_seed() -> np.ndarray:
    """Return a verified seed that aligns exactly with a classic 600-cell vertex.

    This seed generates all 120 vertices of the 600-cell under h4_icosian,
    and is the anchor point used by `h4_swirlprism_predefined_seed` to
    explore the small swirlprism's main-ring/cross-ring structure.
    """
    return np.array([1.0, 0.0, 0.0, 0.0])


@lru_cache(maxsize=1)
def _h4_swirlprism_ring_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cross_ring_one, cross_ring_two, main_ring) at the anchor seed.

    Derived from the eigenstructure of a non-trivial stabilizer of the anchor
    seed under h4_swirlprism+ (a verified order-5 rotation fixing the seed):
    its real fixed axis (orthogonal to the seed) is the main ring direction,
    and its complex-eigenvalue rotating plane gives the two cross-ring
    directions -- together with the seed, an orthonormal basis of R^4.
    """
    seed = h4_swirlprism_anchor_seed()
    action = SymmetryAction.from_iterable(_h4_swirlprism_chiral_generators())
    generator_data = tuple(generator.tobytes() for generator in action.generators)
    elements = _finite_group_elements(generator_data, 5000)
    if elements is None:
        raise RuntimeError("h4_swirlprism+ closure did not converge")

    stabilizer = next(
        element
        for element in elements
        if np.allclose(element @ seed, seed, atol=1e-6)
        and not np.allclose(element, np.eye(4), atol=1e-6)
    )
    eigvals, eigvecs = np.linalg.eig(stabilizer)

    main_ring: np.ndarray | None = None
    cross_vec: np.ndarray | None = None
    for eigval, eigvec in zip(eigvals, eigvecs.T):
        if abs(eigval.imag) < 1e-6 and abs(eigval.real - 1.0) < 1e-6:
            candidate = np.real(eigvec)
            candidate = candidate - np.dot(candidate, seed) * seed
            if np.linalg.norm(candidate) > 1e-6:
                main_ring = candidate / np.linalg.norm(candidate)
        elif eigval.imag > 1e-6:
            cross_vec = eigvec
    if main_ring is None or cross_vec is None:
        raise RuntimeError("Could not find pentagonal-swirl ring basis at anchor seed")

    cross_one = np.real(cross_vec)
    cross_one = cross_one - np.dot(cross_one, seed) * seed - np.dot(cross_one, main_ring) * main_ring
    cross_one = cross_one / np.linalg.norm(cross_one)

    cross_two = np.imag(cross_vec)
    cross_two = (
        cross_two
        - np.dot(cross_two, seed) * seed
        - np.dot(cross_two, main_ring) * main_ring
        - np.dot(cross_two, cross_one) * cross_one
    )
    cross_two = cross_two / np.linalg.norm(cross_two)

    return cross_one, cross_two, main_ring


def h4_swirlprism_predefined_seed(
    cross_ring_one_degrees: float,
    cross_ring_two_degrees: float,
    main_ring_degrees: float,
) -> np.ndarray:
    """Move the verified 120-point anchor seed around its cross rings and main ring."""
    anchor = h4_swirlprism_anchor_seed()
    cross_ring_one, cross_ring_two, main_ring = _h4_swirlprism_ring_basis()
    seed = anchor.copy()

    for angle_degrees, direction in (
        (cross_ring_one_degrees, cross_ring_one),
        (cross_ring_two_degrees, cross_ring_two),
        (main_ring_degrees, main_ring),
    ):
        angle = np.deg2rad(angle_degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        seed_component = float(np.dot(seed, anchor))
        direction_component = float(np.dot(seed, direction))
        seed = (
            seed
            + (cosine - 1.0) * (seed_component * anchor + direction_component * direction)
            + sine * (seed_component * direction - direction_component * anchor)
        )

    return seed / np.linalg.norm(seed)


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
        "b4_basic": "b4",
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
    elif name == "b4":
        # Full [4,3,3] reflection group in the same Coxeter root basis as b4+,
        # so full B4 symmetry can be detected regardless of coordinate embedding.
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (2, 3, 4)))
    elif name == "d4":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 3), (1, 3, 3)))
    elif name == "f4":
        gens = _coxeter_reflections(((0, 1, 3), (1, 2, 4), (2, 3, 3)))
    elif name == "h4":
        gens = _coxeter_reflections(((0, 1, 5), (1, 2, 3), (2, 3, 3)))
    elif name == "h4_icosian":
        gens = _h4_icosian_generators()
    elif name == "h4_icosian+":
        gens = _h4_icosian_chiral_generators()
    elif name == "h4_pentagonal_swirl":
        gens = _pentagonal_swirl_generators()
    elif name == "h4_pentagonal_swirl_ring":
        gens = _pentagonal_swirl_ring_generators()
    elif name == "h4_swirlprism":
        gens = _h4_swirlprism_generators()
    elif name == "h4_swirlprism+":
        gens = _h4_swirlprism_chiral_generators()
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
        "h4_icosian",
        "h4_icosian+",        "h4_pentagonal_swirl",
        "h4_pentagonal_swirl_ring",
        "h4_swirlprism",
        "h4_swirlprism+",        "a4_basic",
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
