import json

import numpy as np
from cellframe import TINV, golden_form, in_half
from uniform import NAMES, S  # simple roots in diagram order (runs the uniform setup quietly)

from four_d_vertex_generator.generation import generate_vertices_from_seed as g
from four_d_vertex_generator.library import named_symmetry

H4 = named_symmetry("h4_icosian")
out = []
for rings, uname in NAMES.items():
    w = np.array([float(c) for c in rings]); seed = np.linalg.solve(S, w); seed /= np.linalg.norm(seed)
    V = g(seed, H4, tol=1e-7)
    B = (TINV @ V.T).T; B = B / B.sum(axis=1, keepdims=True)
    inside = [b for b in B if in_half(b)]
    uniq = []
    for b in inside:
        if not any(np.allclose(b, u, atol=1e-9) for u in uniq): uniq.append(b)
    print(f"{uname:<26} {len(uniq)} point(s) in the half-cell")
    for b in uniq:
        # scale so the smallest non-zero weight is 1, then write each weight in the golden field
        nz = b[b > 1e-9]; bb = b / nz.min()
        print("     beta ∝ (" + ", ".join(golden_form(x) for x in bb) + ")")
        out.append({"uniform": uname, "rings": rings, "beta": b.tolist()})
json.dump(out, open("uniform_in_cell.json", "w"))
