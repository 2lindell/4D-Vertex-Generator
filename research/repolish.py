"""Re-optimise the optimized seeds the old optimiser may have stopped at a mirror or the fold (it kept the betas >= 0
and b1 >= b4; it now crosses them: decaic_atlas/optimize_seeds.CROSS_MIRRORS), keeping the better R.

Rows taken: status "infimum on the shape's edge" or "no progress", or an optimum with a zero beta the seed did not have
or on the fold b1 = b4 when the seed was not. Each is optimised again from its current optimum and from its seed, on its
own sheet's group; where R drops (by more than 1e-9) the row and its optimized .off file are updated.

    python repolish.py ixitic|toxitic        (toxitic: the toxitic, oxitic and icopyritic rows)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP = r'''
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, %(atlas)r)
import optimize_seeds as OS
O = OS.O
G = OS.G
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import to_4off
rows = json.loads(sys.stdin.read())


def job(r):
    tol = 1e-7 if r["kind"] == "wall" else 1e-9
    best = (r.get("opt_R") or r.get("off_R") or 1e99, None, None)
    for start in (r["optimized"], r["seed"]):
        try:
            o = O.optimise({"name": r["name"], "kind": r["kind"], "degrees": r["degrees"], "seed": start,
                            "locus": r.get("locus", []), "hull_tol": tol})
        except Exception as e:
            continue
        b = np.asarray(o.get("off_seed", o["best"]), float)
        bb = b / b.sum()
        if o["R"] is None or G.label(bb, tol)[1] != r["key"]:
            continue
        # (the true R, from the polytope's own edges: across a seam the optimiser can reach a congruent copy of the
        #  shape whose edges are other group elements than the seed's, where its edge forms no longer apply)
        V, Vn, idx, faces, cells = G.polytope(bb, tol)
        E = [np.linalg.norm(Vn[x] - Vn[y]) for x, y in (tuple(e) for e in hull_edges(faces))]
        Rt = float(max(E) / min(E))
        if Rt < best[0] - 1e-9:
            best = (Rt, b.tolist(), o["status"] + ("" if abs(Rt - o["R"]) < 1e-6 else
                                                   f" (the optimiser's own R {o['R']:.6f}; the polytope's {Rt:.6f})"))
    return best


with Pool(4) as pool:
    R = pool.map(job, rows, chunksize=1)
out = []
for r, (Rb, b, st) in zip(rows, R):
    if b is None:
        out.append(None)
        continue
    tol = 1e-7 if r["kind"] == "wall" else 1e-9
    bb = np.asarray(b) / np.sum(b)
    V, Vn, idx, faces, cells = G.polytope(bb, tol)
    E = [np.linalg.norm(Vn[x] - Vn[y]) for x, y in (tuple(e) for e in hull_edges(faces))]
    open(os.path.join(%(root)r, r["opt_off"]), "w").write(to_4off(Vn, faces, cells))
    out.append({"optimized": bb.tolist(), "R": float(max(E) / min(E)), "status": st})
    print(f"{r['name']:28} {r.get('opt_R') or r.get('off_R'):.6f} -> {out[-1]['R']:.6f}  {st[:50]}", flush=True)
print(json.dumps(out))
'''


def wanted(r):
    s, b = r["seed"], r["optimized"]
    st = r.get("optimized_status", "")
    return ("infimum" in st or "no progress" in st
            or any(b[k] < 1e-9 <= s[k] for k in range(4))
            or (abs(b[0] - b[3]) < 1e-9 and abs(s[0] - s[3]) > 1e-9))


def run(atlas, env, files):
    for f in files:
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        by = {}
        for i, r in enumerate(d["shapes"]):
            if wanted(r):
                by.setdefault(r.get("sheet") or env[1], []).append(i)
        for sheet, idx in by.items():
            rows = [d["shapes"][i] for i in idx]
            res = subprocess.run([sys.executable, "-c", STEP % {"atlas": os.path.join(HERE, atlas), "root": HERE}],
                                 input=json.dumps(rows), capture_output=True, text=True, cwd=os.path.join(HERE, atlas),
                                 env={**os.environ, env[0]: sheet})
            lines = res.stdout.strip().splitlines()
            for line in lines[:-1]:
                print(f"{f} {sheet}: {line}", flush=True)
            try:
                upd = json.loads(lines[-1])
            except Exception:
                print(f, sheet, "FAILED", res.stderr[-600:], flush=True)
                continue
            for i, u in zip(idx, upd):
                if u:
                    r = d["shapes"][i]
                    old = r.get("opt_R") or r.get("off_R")
                    r["optimized"] = u["optimized"]
                    r["opt_R" if "opt_R" in r else "off_R"] = u["R"]
                    r["optimized_status"] = (u["status"] + f"; re-optimised crossing the mirrors and the fold "
                                             f"(was R {old:.6f})")
            json.dump(d, open(p, "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "ixitic":
        run("hyic_atlas", ("HYIC_SUB", "ix0"), ["hyic_atlas/opt_central_ixitic.json"])
    else:
        run("contic_atlas", ("CONTIC_SUB", "tox"), [f"toxitic_atlas/opt_central_{s}.json" for s in ("tox", "ox", "oxb", "icp", "icpb")])
