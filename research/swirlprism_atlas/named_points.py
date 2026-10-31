"""Exact positions of named 1200-vertex shapes built from uniform H4 truncations."""
import json
from collections import Counter

import numpy as np
from named2 import A, c0, n1, n2, nb120, truncation_point
from survey import BASIS, FULL, in_cell

from four_d_vertex_generator.generation import group_elements

E = np.stack(group_elements(FULL))
def images_in_cell(p):
    out = []
    for x in E @ p:
        if x @ A <= 0: continue
        q = BASIS @ (x / (x @ A))
        if in_cell(q, 1e-9) and not any(np.allclose(q, o, atol=1e-6) for o in out): out.append(q)
    return out
pts = []
p, _ = truncation_point(A, n1, n2)
for q in images_in_cell(p): pts.append({"id": "btt600", "q": [round(float(c), 6) for c in q]})
for i, kind in ((0, "sdt120a"), (1, "sdt120b")):
    others = [u for j, u in enumerate(nb120) if j != i]
    p, _ = truncation_point(c0, nb120[i], others[0])
    for q in images_in_cell(p): pts.append({"id": kind, "q": [round(float(c), 6) for c in q]})
json.dump(pts, open("named_points.json", "w"))
print(Counter(p["id"] for p in pts))
