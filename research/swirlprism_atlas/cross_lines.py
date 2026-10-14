"""Scan lines where the special half-turn planes (az0, az36) cross the tilted H4 mirrors."""
import json
from multiprocessing import Pool

import numpy as np
from axes import name
from classify import classify, signature
from survey import in_cell, seed_from_q, stabilizer_q_maps

mir = json.load(open("mirrors.json"))
tilted = [np.array(m["n"]) for m in mir if abs(m["n"][2]) > 1e-6]
halfturn_planes = {"az0": np.array([0.0, 1.0, 0.0]), "az36": np.array([-np.sin(np.radians(36)), np.cos(np.radians(36)), 0.0])}
maps = stabilizer_q_maps(); rays = []
for pname, pn in halfturn_planes.items():
    for tn in tilted:
        d = np.cross(pn, tn)
        if np.linalg.norm(d) < 1e-9: continue
        d /= np.linalg.norm(d)
        for r in (d, -d):
            if not any(any(np.allclose(m @ r, s) for m in maps) for _, s in rays): rays.append((pname, r))
print(len(rays), "distinct rays", flush=True)
def job(item):
    i, s, q = item
    try: return i, s, name(signature(classify(seed_from_q(q))))
    except Exception: return i, s, "ERR"
jobs = []
for i, (_, r) in enumerate(rays):
    smax = 0.0
    while in_cell((smax + 1e-4) * r, 1e-12): smax += 1e-4
    for s in np.linspace(0, smax, 101)[1:-1]: jobs.append((i, float(s), s * r))
with Pool(4) as pool: res = pool.map(job, jobs)
for i, (pname, r) in enumerate(rays):
    seq = sorted((s, t) for j, s, t in res if j == i); runs = []
    for s, t in seq:
        if t == "ERR": continue
        if runs and runs[-1][2] == t: runs[-1][1] = s
        else: runs.append([s, s, t])
    print(f"\n{pname} ∩ tilted mirror, ray {np.round(r, 3).tolist()}:")
    for a, b, t in runs: print(f"   {np.degrees(np.arctan(a)):6.2f}° .. {np.degrees(np.arctan(b)):6.2f}°  {t}")
