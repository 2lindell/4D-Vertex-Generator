"""Walk around each T2 point inside its boundary surface to see whether T2 continues as a curve."""
import json
import sys
from multiprocessing import Pool

import numpy as np
from catalog import match
from classify import classify, signature
from survey import seed_from_q

EPS, TOL = 1e-4, 1e-6
def lab(q):
    try: s = signature(classify(seed_from_q(q), hull_tol=TOL))
    except Exception: return "ERR"
    return match(s) or s.split(" | val")[0]
def run(q):
    return lab(q)
if __name__ == "__main__":
    reps = json.load(open("t2probe_0.0001.json"))
    names = sys.argv[1:] or list(reps)
    n = 98; i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5**0.5) * i
    DIRS = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    out = {}
    with Pool(4) as pool:
        for k in names:
            q0 = np.array(reps[k]["q"])
            labels = pool.map(run, [q0 + EPS * d for d in DIRS])
            kinds = [t for t, _ in sorted(((t, labels.count(t)) for t in set(labels)), key=lambda x: -x[1])][:2]
            a = DIRS[[lb == kinds[0] for lb in labels]].mean(0); b = DIRS[[lb == kinds[1] for lb in labels]].mean(0)
            nrm = (a - b) / np.linalg.norm(a - b)                       # surface normal estimate
            u = np.cross(nrm, [1.0, 0, 0]); u = u if np.linalg.norm(u) > 0.1 else np.cross(nrm, [0, 1.0, 0]); u /= np.linalg.norm(u); v = np.cross(nrm, u)
            # refine the normal: find the offset along nrm where the side flips, at a few in-surface points
            angs = np.radians(np.arange(0, 360, 1.0))
            ring = [q0 + EPS * (np.cos(t) * u + np.sin(t) * v) for t in angs]
            ring_labels = pool.map(run, ring)
            runs = []
            for t, lb in zip(np.degrees(angs), ring_labels):
                if runs and runs[-1][2] == lb: runs[-1][1] = t
                else: runs.append([t, t, lb])
            if len(runs) > 1 and runs[0][2] == runs[-1][2]: runs[0][0] = runs[-1][0] - 360; runs.pop()
            print(f"\n{k}: sides {kinds[0]} | {kinds[1]}; normal {np.round(nrm, 3)}")
            for s0, s1, lb in runs: print(f"   {s0:6.0f}° .. {s1:4.0f}°  {lb}")
            out[k] = {"q": q0.tolist(), "normal": nrm.tolist(), "u": u.tolist(), "v": v.tolist(), "runs": runs}
    json.dump(out, open("t2surface.json", "w"))
