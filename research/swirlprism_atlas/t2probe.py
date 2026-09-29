"""Probe the neighbourhood of known T2 points to find the dimension of the T2 locus."""
import json
import sys
from multiprocessing import Pool

import numpy as np
from catalog import match
from classify import classify, signature
from survey import seed_from_q, stabilizer_q_maps

maps = stabilizer_q_maps()

EPS, TOL = float(sys.argv[1]), float(sys.argv[2])
pts = [p for p in json.load(open("uniform_points.json")) if p["id"] == "T2"]
# one representative per source, preferring the fundamental wedge (z >= 0, azimuth 0..72)
def wedge_score(q):
    az = np.degrees(np.arctan2(q[1], q[0])) % 360
    return (0 if q[2] >= -1e-9 else 1, 0 if az < 72 + 1e-6 else 1)
reps = {}
for p in pts:
    q = np.array(p["q"]); key = p["from"] + str(len([r for r in reps if r.startswith(p["from"])]))
    cands = [m @ q for m in maps]
    best = min(cands, key=wedge_score)
    if not any(np.allclose(best, r, atol=1e-6) for r in reps.values()):
        reps[f"{p['from']} #{len(reps)}"] = best
reps = dict(list(reps.items()))
print(len(reps), "distinct T2 representatives:")
for k, q in reps.items(): print(f"   {k}: q={np.round(q, 5)}")
# probe directions: 98 roughly uniform directions on the sphere (Fibonacci)
n = 98; i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5**0.5) * i
DIRS = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
def job(item):
    k, j, eps, q = item
    try: s = signature(classify(seed_from_q(q), hull_tol=TOL))
    except Exception: return k, j, eps, "ERR"
    return k, j, eps, match(s) or s.split(" | val")[0]
if __name__ == "__main__":
    jobs = [(k, -1, 0.0, q) for k, q in reps.items()]
    for k, q in reps.items():
        for j, d in enumerate(DIRS):
            for eps in (EPS,):
                jobs.append((k, j, eps, q + eps * d))
    with Pool(4) as pool: res = pool.map(job, jobs)
    out = {}
    for k in reps:
        centre = [t for kk, j, e, t in res if kk == k and j == -1][0]
        keep = [j for kk, j, e, t in res if kk == k and j >= 0 and t == "T2"]
        types = {}
        for kk, j, e, t in res:
            if kk == k and j >= 0: types[t[:40]] = types.get(t[:40], 0) + 1
        D = DIRS[keep]
        # fit: if the surviving directions lie on a great circle, their normal is the smallest singular vector
        if len(D) >= 3:
            _, sv, vt = np.linalg.svd(D); planarity = sv[-1] / sv[0]
        else: sv, vt, planarity = None, None, None
        print(f"\n{k}: centre classifies as {centre}; T2 in {len(keep)}/{len(DIRS)} directions")
        print("   neighbourhood types:", dict(sorted(types.items(), key=lambda kv: -kv[1])))
        if planarity is not None: print(f"   surviving directions: singular values {np.round(sv, 3)} (a surface gives one ~0)")
        out[k] = {"q": reps[k].tolist(), "keep": [DIRS[j].tolist() for j in keep], "types": types}
    json.dump(out, open(f"t2probe_{EPS:g}.json", "w"))
