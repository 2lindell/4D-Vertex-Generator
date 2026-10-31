"""OFF files for the points where the purple axis crosses transitional classes (axis_crossings.py), checked with
the independent hull checker. The points lie within ~1e-15 of walls, so the hulls are built with a merge
tolerance of 1e-8 (relative), as they are classified."""
import json
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

G = named_symmetry("h4_swirlprism")
xid = json.load(open("atlas_xids.json"))
for c in json.load(open("axis_crossings.json")):
    if not c.get("interior"):
        continue
    x = seed_from_beta(np.array(c["beta"]))
    info = classify(x, hull_tol=1e-8)
    sig = signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    tid = label_of_sig(sig)
    V = generate_vertices_from_seed(x, G, tol=1e-9)
    faces, cells = compute_convex_hull(V, tol=1e-8)
    check_hull(V, faces, cells, margin=1e-8)
    path = f"examples/{tid}_purple_axis_crossing_swirlprism_{len(V)}.off"
    open(path, "w").write(to_4off(V, faces, cells))
    sizes = {k: sum(1 for f in faces if len(f) == k) for k in sorted({len(f) for f in faces})}
    print(f"{tid}: {path}  {len(V)} vertices, {len(faces)} faces {sizes}, {len(cells)} cells; seed",
          ", ".join(f"{v:.17g}" for v in x))
