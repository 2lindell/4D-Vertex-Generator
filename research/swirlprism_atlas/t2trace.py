"""Trace T2 curves inside the H4 mirror planes that contain them (predictor-corrector)."""
import json
import sys
from multiprocessing import Pool

import numpy as np
from catalog import match
from classify import classify, signature
from survey import BASIS, A, in_cell, seed_from_q

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

H4 = group_elements(named_symmetry("h4_icosian")); normals = []
for e in H4:
    if np.linalg.det(e) < 0 and np.isclose(np.trace(e), 2.0):
        w, v = np.linalg.eigh((e + e.T) / 2); n = v[:, np.isclose(w, -1.0)][:, 0]
        if not any(abs(abs(n @ m) - 1) < 1e-9 for m in normals): normals.append(n)
PLANES = []
for n in normals:
    nq, c = BASIS @ n, -(n @ A); s = np.linalg.norm(nq)
    if s > 1e-9: PLANES.append((nq / s, c / s))

def lab(q):
    try: s = signature(classify(seed_from_q(q)))
    except Exception: return "ERR"
    return match(s) or s.split(" | val")[0]

def trace(args):
    name, q0, t0, h, max_steps = args
    q0, t = np.array(q0), np.array(t0)
    nq, c = min(PLANES, key=lambda pc: abs(pc[0] @ q0 - pc[1]) + 10 * abs(pc[0] @ t))  # the mirror holding q0 and t
    path = [q0.tolist()]; q = q0; status = "max steps"
    for _ in range(max_steps):
        pred = q + h * t
        pred = pred - (nq @ pred - c) * nq                     # stay exactly on the mirror
        if not in_cell(pred, 1e-9): status = "left the region"; break
        w = np.cross(nq, t); w /= np.linalg.norm(w)
        R = 0.25 * h; lo, hi = pred - R * w, pred + R * w
        llo, lhi = lab(lo), lab(hi)
        if lab(pred) == "T2": new = pred
        else:
            tries = 0
            while llo == lhi and tries < 3:
                R *= 2; lo, hi = pred - R * w, pred + R * w; llo, lhi = lab(lo), lab(hi); tries += 1
            if llo == lhi: status = f"lost the curve (both sides {llo})"; break
            new = None
            for _ in range(24):
                mid = (lo + hi) / 2; lm = lab(mid)
                if lm == "T2": new = mid; break
                if lm == llo: lo = mid
                elif lm == lhi: hi = mid
                else: hi = mid; lhi = lm            # a third label: narrow towards it
            if new is None: status = f"no T2 between {llo} and {lhi}"; break
        t = (new - q) / np.linalg.norm(new - q); q = new; path.append(q.tolist())
    return name, path, status, lab(q)

if __name__ == "__main__":
    S = json.load(open("t2surface.json"))
    probe = json.load(open("t2probe_0.0001.json"))
    jobs = []
    for k, d in S.items():
        q0, u, v = np.array(d["q"]), np.array(d["u"]), np.array(d["v"])
        for a, b, lb in d["runs"]:
            if lb != "T2": continue
            w = np.radians((a + b) / 2); t = np.cos(w) * u + np.sin(w) * v
            jobs.append((f"{k} dir {(a + b) / 2:.0f}°", q0.tolist(), t.tolist(), float(sys.argv[1]), int(sys.argv[2])))
    with Pool(4) as pool:
        out = []
        for name, path, status, last in pool.imap_unordered(trace, jobs):
            L = sum(np.linalg.norm(np.subtract(path[i + 1], path[i])) for i in range(len(path) - 1))
            print(f"{name}: {len(path)} points, length {L:.4f}, stopped: {status} (last label {last})", flush=True)
            out.append({"name": name, "path": path, "status": status})
            json.dump(out, open("t2trace.json", "w"))
