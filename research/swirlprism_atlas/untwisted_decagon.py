"""OFF file for the point on the regular-decagon line where the decagon's cell is untwisted: beta (phi^2, 1, 2, 1).

Along the dashed decagon line (decagon_twist.py) the cell holding each regular decagon also has a pentagon face,
turned against the decagon by an angle that grows from about 8 to past 18 degrees and back. Exactly at
beta (phi^2, 1, 2, 1) (class X6) it is 18 degrees: the pentagon's edges line up with the decagon's, the twisted
pairs of triangles flatten into trapezoids, and the cell is an untwisted pentagonal cupola (5 triangles, 5
isosceles trapezoids, the pentagon and the regular decagon; pentagon edge = phi x decagon edge). Its copy under
the extra half-turn, on the solid decagon line, is beta (phi^2, 1, 1, 1). Checked with the independent hull checker.
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
beta = np.array([PHI ** 2, 1, 2, 1])
x = seed_from_beta(beta / beta.sum())
info = classify(x)
tid = label_of_sig(signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items()))
V = generate_vertices_from_seed(x, G, tol=1e-9)
faces, cells = compute_convex_hull(V, tol=1e-9)
check_hull(V, faces, cells, margin=1e-9)
path = f"examples/{tid}_untwisted_decagon_cupola_swirlprism_{len(V)}.off"
open(path, "w").write(to_4off(V, faces, cells))
sizes = {k: sum(1 for f in faces if len(f) == k) for k in sorted({len(f) for f in faces})}
print(f"{tid}: {path}  {len(V)} vertices, {len(faces)} faces {sizes}, {len(cells)} cells")
print("signature:", signature(info), "| valence", info["valence"])
print("seed", ", ".join(f"{v:.17g}" for v in x))
