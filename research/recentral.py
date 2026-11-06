"""Re-find the central seeds of the shapes in seam groups (oxitic and its mirror image, icopyritic's two sheets,
ixitic's ten): the centre finder (decaic_atlas/central_seeds.py) no longer counts a seam mirror or a seam fold as a
boundary (its reflection does not normalise the group, so the shape continues past it). Updates each row's central
seed and its .off file; a centre that is not the shape (another arrangement key) keeps the shape's seed.

    python recentral.py
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
if hasattr(OS, "CS"):
    CS = OS.CS
else:
    import central_seeds as CSR
    CS = CSR.CS
    CS.O, CS.G, CS.F = OS.O, OS.G, OS.F
G = OS.G
from four_d_vertex_generator.off import to_4off
rows = json.loads(sys.stdin.read())


def job(r):
    tol = 1e-7 if r["kind"] == "wall" else 1e-9
    CS.OPT = {r["name"]: {"best": r["optimized"], "off_seed": r["optimized"]}}
    try:
        c = CS.centre({"name": r["name"], "kind": r["kind"], "degrees": r["degrees"], "seed": r["seed"],
                       "locus": r.get("locus", []), "hull_tol": tol})
    except Exception as e:
        return None
    b = np.asarray(c["centre"], float)
    if G.label(b / b.sum(), tol)[1] != r["key"]:
        return None
    return c


with Pool(4) as pool:
    C = pool.map(job, rows, chunksize=1)
out = []
for r, c in zip(rows, C):
    if c is None:
        out.append(None)
        continue
    tol = 1e-7 if r["kind"] == "wall" else 1e-9
    b = np.asarray(c["centre"], float); b = b / b.sum()
    V, Vn, idx, faces, cells = G.polytope(b, tol)
    open(os.path.join(%(root)r, r["central_off"]), "w").write(to_4off(Vn, faces, cells))
    out.append({"central": b.tolist(), "depth": c.get("depth"), "status": c["status"]})
    print(f"{r['name']:28} depth {c.get('depth') or 0:.3e}  {[round(v, 5) for v in b]}", flush=True)
print(json.dumps(out))
'''


def run(atlas, var, path, default):
    d = json.load(open(path))
    by = {}
    for i, r in enumerate(d["shapes"]):
        by.setdefault(r.get("sheet") or default, []).append(i)
    for sheet, idx in by.items():
        rows = [d["shapes"][i] for i in idx]
        res = subprocess.run([sys.executable, "-c", STEP % {"atlas": os.path.join(HERE, atlas), "root": HERE}],
                             input=json.dumps(rows), capture_output=True, text=True, cwd=os.path.join(HERE, atlas),
                             env={**os.environ, var: sheet})
        lines = res.stdout.strip().splitlines()
        for line in lines[:-1]:
            print(f"{os.path.basename(path)} {sheet}: {line}", flush=True)
        try:
            upd = json.loads(lines[-1])
        except Exception:
            print(path, sheet, "FAILED", res.stderr[-600:], flush=True)
            continue
        for i, u in zip(idx, upd):
            if u:
                r = d["shapes"][i]
                r.update(central=u["central"], depth=u["depth"],
                         central_status=u["status"] + "; seams (mirrors / fold that do not bound the shape) not counted")
                r.pop("central_note", None)
        json.dump(d, open(path, "w"), indent=1)


if __name__ == "__main__":
    for s in ("ox", "oxb", "icp", "icpb"):
        run("contic_atlas", "CONTIC_SUB", os.path.join(HERE, "toxitic_atlas", f"opt_central_{s}.json"), s)
    run("hyic_atlas", "HYIC_SUB", os.path.join(HERE, "hyic_atlas", "opt_central_ixitic.json"), "ix0")
