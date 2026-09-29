"""H4 mirror hyperplanes, as planes n.q = c in the gnomonic dodecahedron."""
import json

import numpy as np
from survey import BASIS, A, in_cell, stabilizer_q_maps

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

H4 = group_elements(named_symmetry("h4_icosian"))
normals = []
for e in H4:
    if np.linalg.det(e) < 0 and np.isclose(np.trace(e), 2.0):      # reflections
        w, v = np.linalg.eigh((e + e.T) / 2); n = v[:, np.isclose(w, -1.0)][:, 0]
        if not any(abs(abs(n @ m) - 1) < 1e-9 for m in normals): normals.append(n)
print("H4 mirrors:", len(normals))
# plane in gnomonic coordinates: n.(A + B^T q) = 0  ->  (B n).q = -(n.A)
cell_pts = np.array([[x, y, z] for x in np.linspace(-.45, .45, 31) for y in np.linspace(-.45, .45, 31) for z in np.linspace(-.45, .45, 31)])
cell_pts = cell_pts[[in_cell(q) for q in cell_pts]]
planes = []
for n in normals:
    nq, c = BASIS @ n, -(n @ A)
    side = cell_pts @ nq - c
    if side.min() < -1e-9 and side.max() > 1e-9:                      # crosses the cell interior
        s = np.linalg.norm(nq); planes.append((nq / s, c / s))
print("mirrors crossing the dodecahedron:", len(planes), "| through the centre:", sum(abs(c) < 1e-9 for _, c in planes))
for nq, c in planes:
    if abs(c) < 1e-9:
        az = np.degrees(np.arctan2(-nq[0], nq[1])) % 180               # direction of the plane's trace in xy
        print(f"   through centre: normal {np.round(nq, 3)}  contains z-axis: {abs(nq[2]) < 1e-9}" + (f"  azimuth {az:.1f}°" if abs(nq[2]) < 1e-9 else ""))
# classes under the anchor stabilizer (10 maps)
maps = stabilizer_q_maps(); classes = []
for nq, c in planes:
    imgs = [(m @ nq, c) for m in maps]
    if not any(any((np.allclose(i, k) or np.allclose(i, -k)) and np.isclose(abs(c), abs(ck)) for i, _ in imgs) for k, ck in classes): classes.append((nq, c))
print("classes of crossing mirrors under the 10 anchor symmetries:", len(classes), [round(float(c), 4) for _, c in classes])
json.dump([{"n": list(map(float, nq)), "c": float(c)} for nq, c in planes], open("mirrors.json", "w"))
