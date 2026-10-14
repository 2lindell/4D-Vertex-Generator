"""Break each uniform H4 polytope's vertex set into h4_swirlprism orbits and classify them."""
import itertools
import json

import numpy as np
from axes import name
from catalog import match
from classify import classify, signature
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from four_d_vertex_generator.generation import generate_vertices_from_seed as g
from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

H4 = named_symmetry("h4_icosian"); G = named_symmetry("h4_swirlprism")
# unit mirror normals of H4 in these coordinates
normals = []
for e in group_elements(H4):
    if np.linalg.det(e) < 0 and np.isclose(np.trace(e), 2.0):
        w, v = np.linalg.eigh((e + e.T) / 2); n = v[:, np.isclose(w, -1.0)][:, 0]
        if not any(abs(abs(n @ m) - 1) < 1e-9 for m in normals): normals.append(n)
N = np.array(normals)
v0 = np.array([0.9, 0.31, 0.17, 0.23]); N = N * np.sign(N @ v0)[:, None]  # positive roots for a generic point
walls = []
for i in range(len(N)):  # a root is a chamber wall if it can't be dropped
    others = np.delete(N, i, axis=0)
    r = linprog(N[i], A_ub=-others, b_ub=np.zeros(len(others)), bounds=[(-1, 1)] * 4, method="highs")
    if r.status == 0 and r.fun < -1e-9: walls.append(N[i])
S = np.array(walls); assert len(S) == 4, len(S)
gram = np.round(S @ S.T, 6); print("simple-root Gram matrix:\n", gram)
# order the nodes along the Coxeter diagram: the 5-branch is between the pair with -cos(36°)
labels = {}
for i, j in itertools.combinations(range(4), 2):
    c = -gram[i, j]; labels[(i, j)] = {0.809017: 5, 0.5: 3, 0.0: 2}.get(round(c, 6), c)
ends = [i for i in range(4) if sum(labels.get(tuple(sorted((i, j))), 2) != 2 for j in range(4) if j != i) == 1]
five_end = next(i for i in ends if any(labels.get(tuple(sorted((i, j)))) == 5 for j in range(4) if j != i))
order = [five_end]
while len(order) < 4:
    order.append(next(j for j in range(4) if j not in order and labels.get(tuple(sorted((order[-1], j)))) != 2))
S = S[order]; print("diagram order [5,3,3]:", [labels.get(tuple(sorted((order[k], order[k + 1])))) for k in range(3)])
NAMES = {"1000": "120-cell", "0100": "rectified 120-cell", "0010": "rectified 600-cell", "0001": "600-cell",
         "1100": "truncated 120-cell", "1010": "cantellated 120-cell", "1001": "runcinated 120-cell", "0110": "bitruncated 120-cell",
         "0101": "cantellated 600-cell", "0011": "truncated 600-cell", "1110": "cantitruncated 120-cell", "1101": "runcitruncated 120-cell",
         "1011": "runcitruncated 600-cell", "0111": "cantitruncated 600-cell", "1111": "omnitruncated 120-cell"}
results = []
for rings, uname in NAMES.items():
    w = np.array([float(c) for c in rings]); seed = np.linalg.solve(S, w); seed /= np.linalg.norm(seed)
    V = g(seed, H4, tol=1e-7); T = cKDTree(V); left = np.ones(len(V), bool); orbits = []
    while left.any():
        i = int(np.argmax(left)); orbit = g(V[i], G, tol=1e-6)
        _, idx = T.query(orbit); left[idx] = False; orbits.append(V[i])
    print(f"\n{uname} ({rings}): {len(V)} vertices -> {len(orbits)} swirlprism orbits", flush=True)
    for rep in orbits:
        try: s = signature(classify(rep)); nm = match(s) or name(s)
        except Exception as exc: s, nm = f"ERR {exc}", "ERR"
        print(f"   {s.split(':')[0]:>5} vertices  {nm if not nm.startswith('NEW') else 'NEW'}   {s.split(' | val')[0] if nm.startswith('NEW') or nm in ('T1','T2') else ''}", flush=True)
        results.append({"uniform": uname, "rings": rings, "seed": rep.tolist(), "sig": s, "name": nm})
json.dump(results, open("uniform_orbits.json", "w"))
