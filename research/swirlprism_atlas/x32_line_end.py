"""OFF file for the end of the short regular-antiprism line inside X32 (regular_lines.json, the straight piece).

The line runs through X32 from beta (2phi^2, 1, phi^2, phi^2) to (3+2phi, 1, 2+phi, phi^2); the two ends are
copies of each other under the extra half-turn, so they give the same polytope (class X19). There the regular
antiprisms stop being cells: each one's realm takes in a second antiprism and five tetrahedra, merging into one
15-vertex cell (a pentagon, a decagon and fifteen triangles). Checked with the independent hull checker.
"""
import sys

import numpy as np
from cellframe import seed_from_beta
from classify import classify, signature
from probe import label_of_sig

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull, to_4off

sys.path.insert(0, "../../tests")
from hull_checks import check_hull  # noqa: E402

PHI = (1 + 5 ** 0.5) / 2
G = named_symmetry("h4_swirlprism")
beta = np.array([2 * PHI ** 2, 1, PHI ** 2, PHI ** 2])
x = seed_from_beta(beta / beta.sum())
info = classify(x)
tid = label_of_sig(signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items()))
V = generate_vertices_from_seed(x, G, tol=1e-9)
faces, cells = compute_convex_hull(V, tol=1e-9)
check_hull(V, faces, cells, margin=1e-9)
path = f"examples/{tid}_x32_antiprism_line_end_swirlprism_{len(V)}.off"
open(path, "w").write(to_4off(V, faces, cells))
sizes = {k: sum(1 for f in faces if len(f) == k) for k in sorted({len(f) for f in faces})}
print(f"{tid}: {path}  {len(V)} vertices, {len(faces)} faces {sizes}, {len(cells)} cells")
print("signature:", signature(info), "| valence", info["valence"])
print("seed", ", ".join(f"{v:.17g}" for v in x))
