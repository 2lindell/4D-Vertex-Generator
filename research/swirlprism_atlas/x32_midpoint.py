"""The midpoint of the regular-antiprism piece inside X32: where it crosses a purple 2400-symmetry axis, the
seed fixed by the extra half-turn that reverses the piece. Writes examples/X32_regular_antiprisms_axis_point_swirlprism_1200.off."""
import numpy as np
from cellframe import seed_from_beta
from classify import classify, signature
from normalizer import coset_axes, qcopies
from probe import _one
from regular_cells import antiprism_residuals, hull
from scipy.spatial import cKDTree
from supergroups import supergroup

from four_d_vertex_generator.off import to_4off

F = (1 + 5 ** 0.5) / 2
f2 = F * F
a = np.array([2 * f2, 1, f2, f2])
a /= a.sum()
b = qcopies(a)[0]
best = None
for c, d in coset_axes():
    s, t = np.linalg.lstsq(np.column_stack([b - a, -(d - c)]), c - a, rcond=None)[0]
    p, q = a + s * (b - a), c + t * (d - c)
    if best is None or np.linalg.norm(p - q) < best[0]:
        best = (np.linalg.norm(p - q), p, s, t)
gap, p, s, t = best
p = p / p.sum()
print(f"crossing at s = {s:.12f} along the piece, gap to the axis {gap:.1e}")
print("beta", repr(p.tolist()))
x = seed_from_beta(p)
print("seed", ", ".join(f"{v:.17g}" for v in x))
print("label", _one(p)[0], "|", signature(classify(x)))
v, faces, cells = hull(p)
res = antiprism_residuals(v, faces, cells)
print("regular antiprisms:", sum(max(abs(r[1]), abs(r[2])) < 1e-9 for r in res), "of", len(res), "antiprisms")
T = cKDTree(v)
for k in (2, 3):
    E = np.stack(supergroup(k))
    print(f"G_{k} ({len(E)}):", sum(T.query(v @ g.T)[0].max() < 1e-7 for g in E), "preserve the vertex set")
open("examples/X32_regular_antiprisms_axis_point_swirlprism_1200.off", "w").write(to_4off(v, faces, cells))
print(len(v), "vertices,", len(faces), "faces,", len(cells), "cells")
