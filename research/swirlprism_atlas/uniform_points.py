import json

import numpy as np
from survey import BASIS, FULL, A, in_cell

from four_d_vertex_generator.generation import group_elements

E = np.stack(group_elements(FULL))
mir = [np.array(m["n"]) for m in json.load(open("mirrors.json"))]
halfturn = {"0° plane": np.array([0, 1.0, 0]), "36° plane": np.array([-np.sin(np.radians(36)), np.cos(np.radians(36)), 0])}
R = json.load(open("uniform_orbits.json"))
pts = []
for r in R:
    if r["name"] not in ("T1", "T2"): continue
    p = np.array(r["seed"]); imgs = []
    for x in E @ p:
        if x @ A <= 0: continue
        q = BASIS @ (x / (x @ A))
        if in_cell(q, 1e-9) and not any(np.allclose(q, o, atol=1e-6) for o in imgs): imgs.append(q)
    q = min(imgs, key=lambda q: (-q[2], q[0]))
    on = sum(abs(n @ q) < 1e-7 for n in mir)
    extra = [k for k, n in halfturn.items() if abs(n @ q) < 1e-7]
    ang = np.degrees(np.arctan(np.linalg.norm(q)))
    print(f"{r['name']} from {r['uniform']}: {len(imgs)} images in the cell; e.g. q={np.round(q, 4)} ({ang:.2f}° from centre), on {on} H4 mirror(s) {extra}")
    for q in imgs: pts.append({"id": r["name"], "q": [round(float(c), 6) for c in q], "from": r["uniform"]})
uniq = []
for p in pts:
    if not any(p["id"] == u["id"] and np.allclose(p["q"], u["q"], atol=1e-6) for u in uniq): uniq.append(p)
json.dump(uniq, open("uniform_points.json", "w")); print(len(uniq), "T1/T2 points saved")
