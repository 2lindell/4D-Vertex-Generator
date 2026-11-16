import itertools

import numpy as np

from four_d_vertex_generator import generate_vertices_from_seed
from four_d_vertex_generator.five_d import _frame, has_5d_coordinates, to_5d
from four_d_vertex_generator.library import named_symmetry


def _seed(p):
    p = np.asarray(p, float)
    return _frame().T @ (p - p.mean())


def _key(P):
    return sorted(map(tuple, np.round(P, 8)))


def test_frame_is_an_isometry_onto_the_sum_zero_hyperplane():
    F = _frame()
    assert np.allclose(F.T @ F, np.eye(4))
    assert np.allclose(F.sum(0), 0)


def test_known_polychora_get_their_5d_coordinates():
    rap = to_5d(generate_vertices_from_seed(_seed([0, 0, 0, 1, 1]), named_symmetry("a4")))
    assert _key(rap) == _key(np.array(sorted(set(itertools.permutations([0, 0, 0, 1, 1]))), float))
    deca = to_5d(generate_vertices_from_seed(_seed([0, 0, 1, 2, 2]), named_symmetry("a4_extended")))
    assert len(deca) == 30 and _key(deca) == _key(np.array(sorted(set(itertools.permutations([0, 0, 1, 2, 2]))), float))


def test_orthant_facet_and_symmetries():
    rng = np.random.default_rng(0)
    for sym in ("a4", "a4_extended", "a4_chiral_extended", "iodecaic", "a4+"):
        assert has_5d_coordinates(sym)
        P = to_5d(generate_vertices_from_seed(rng.normal(size=4), named_symmetry(sym)))
        assert P.min() == 0 and np.ptp(P.sum(1)) < 1e-9
        c = P.sum(1)[0] / 5
        assert (_key(P[:, [1, 0, 2, 3, 4]]) == _key(P)) == (sym in ("a4", "a4_extended"))
        assert (_key(2 * c - P) == _key(P)) == (sym in ("a4_extended", "a4_chiral_extended"))
    assert not has_5d_coordinates("f4")
