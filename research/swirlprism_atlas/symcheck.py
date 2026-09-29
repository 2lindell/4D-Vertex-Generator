"""Full symmetry order of each type's example polytope, and H4 mirrors through the cell."""
import itertools
import json
import re

import numpy as np
from scipy.spatial import cKDTree

from four_d_vertex_generator.generation import generate_vertices_from_seed as g
from four_d_vertex_generator.library import named_symmetry

FULL = named_symmetry("h4_swirlprism")

def full_symmetry_order(V):
    """|Sym(V)| for a vertex-transitive set: |V| x (isometries fixing V[0])."""
    T = cKDTree(V); d0 = np.linalg.norm(V - V[0], axis=1); nb = np.argsort(d0)[1:13]
    X = np.array([V[0], *V[nb[:3]]]); mats = []
    for combo in itertools.permutations(nb, 3):
        if not np.allclose(d0[list(combo)], d0[nb[:3]], atol=1e-7): continue
        Y = np.array([V[0], *V[list(combo)]])
        M = np.linalg.lstsq(X, Y, rcond=None)[0].T; U, _, Vt = np.linalg.svd(M); M = U @ Vt
        if T.query(V @ M.T)[0].max() < 1e-7 and not any(np.allclose(M, m, atol=1e-6) for m in mats):
            mats.append(M)
    return len(V) * len(mats), mats

if __name__ == "__main__":
    html = open("atlas.html").read()
    data = json.loads(re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    rows = []
    for k, t in sorted(data["types"].items(), key=lambda kv: -kv[1]["samples"]):
        seed = np.array([float(x) for x in t["example"].split(",")])
        V = g(seed, FULL, tol=1e-6)
        order, mats = full_symmetry_order(V)
        refl = sum(1 for m in mats if np.linalg.det(m) < 0)
        rows.append((k, order, refl, t["planes"], t["volumeSamples"]))
        print(f"{k:<4} symmetry order {order:>5}  (vertex stabilizer {order // 1200}, of which {refl} reflections)  "
              f"{'volume' if t['volumeSamples'] else 'plane ' + ','.join(t['planes'])}", flush=True)
