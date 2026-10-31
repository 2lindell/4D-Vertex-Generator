"""Scan the rays from the centre along the icosahedral axes (H4 mirror intersections)."""
import json
import re
import sys
from multiprocessing import Pool

import numpy as np
from build_atlas import cell_edges
from catalog import match, parse_signature
from classify import classify, signature
from survey import BASIS, NEIGHBOURS, in_cell, seed_from_q, stabilizer_q_maps

edges = [(np.array(a), np.array(b)) for a, b in cell_edges()]
corners = []
for a, b in edges:
    for v in (a, b):
        if not any(np.allclose(v, c) for c in corners): corners.append(v)
mids = [(a + b) / 2 for a, b in edges]
# face centres (5-fold): the directions of the 12 face normals
faces5 = [BASIS @ u for u in NEIGHBOURS]  # directions of the 12 face normals
rays = [("3-fold", c) for c in corners] + [("2-fold", m) for m in mids] + [("5-fold", f) for f in faces5]
maps = stabilizer_q_maps(); reps = []
for kind, r in rays:
    r = r / np.linalg.norm(r)
    if not any(k == kind and any(np.allclose(m @ r, s) for m in maps) for k, s in reps): reps.append((kind, r))
print("ray classes:", [(k, np.round(r, 3).tolist()) for k, r in reps], flush=True)
d = json.loads(re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', open("atlas.html").read(), re.S).group(1))
key = {(t["cells"], t["faces"], t["edges"]): k for k, t in d["types"].items()}
def name(sig):
    p = parse_signature(sig)
    if p is None: return "ERR"
    return match(sig) or key.get(tuple("+".join(map(str, x)) for x in p[1:]), "NEW[" + sig.split(" | val")[0] + "]")
def job(item):
    i, s, q = item
    try: return i, s, name(signature(classify(seed_from_q(q)))), q
    except Exception: return i, s, "ERR", q
if __name__ == "__main__":
    steps = int(sys.argv[1]); jobs = []
    for i, (kind, r) in enumerate(reps):
        smax = 0.0
        while in_cell((smax + 1e-4) * r, 1e-12): smax += 1e-4
        for s in np.linspace(0, smax, steps + 1)[1:]: jobs.append((i, float(s), s * r))
    with Pool(4) as pool: res = pool.map(job, jobs)
    out = []
    for i, (kind, r) in enumerate(reps):
        seq = sorted((s, t, q) for j, s, t, q in res if j == i); runs = []
        for s, t, q in seq:
            if runs and runs[-1][2] == t: runs[-1][1] = s
            else: runs.append([s, s, t, q.tolist()])
        print(f"\n{kind} ray {np.round(r, 3).tolist()}:")
        for a, b, t, q in runs: print(f"   {np.degrees(np.arctan(a)):6.2f}° .. {np.degrees(np.arctan(b)):6.2f}°  {t}")
        out.append({"kind": kind, "dir": r.tolist(), "runs": runs})
    json.dump(out, open("axes.json", "w"))
