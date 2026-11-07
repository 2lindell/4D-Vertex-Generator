"""Seed, optimized and central .off files of the ixitic shapes (opt_central_ixitic.json; new_shapes/off/ixitic,
optimized_off/ixitic, central_off/ixitic).

Each shape of ixitic_shapes.json is run on the sheet it was found on (HYIC_SUB = ix0..ix9) with the decaic optimiser
and centre finder (optimize_seeds.py); its locus: free for a region, the zero betas for a shape on a mirror face or edge
(kept zero), else the map's surfaces through its seed. Named as in the accepted list (ixitic_shapes.json), the rest
numbered (ixitic_region_NN, ...). A seed the optimiser or the centre finder moves to another polytope is not used: the
shape's own seed is written instead, and noted.

    python ixitic_offs.py [gather | new]      (new: only the shapes not yet in opt_central_ixitic.json)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STEP = r'''
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, %(here)r)
import optimize_seeds as OS
G = OS.G
from snub_map import Surface
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import to_4off
S = json.loads(sys.stdin.read())
M = json.load(open(os.path.join(%(here)r, G.fname("snub_map"))))
surfs = [Surface(t) for t in M["surfaces"]]
for s in S:
    b = np.asarray(s["seed"], float); b = b / b.sum()
    if s["kind"] == "region" or b.min() < 1e-12:
        s["locus"] = []
    else:
        s["locus"] = [x.text for x in surfs if x.dist(b) < 1e-7]


def job(s):
    try:
        r = OS.O.optimise(s)
    except Exception as e:
        r = {"best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None, "status": f"optimiser failed: {e}"}
    OS.CS.OPT = {s["name"]: r}
    try:
        c = OS.CS.centre(s)
    except Exception as e:
        c = {"centre": s["seed"], "depth": None, "status": f"centre finder failed: {e}"}
    return r, c


with Pool(4) as pool:
    R = pool.map(job, S, chunksize=1)
out = []
for s, (r, c) in zip(S, R):
    row = {k: s[k] for k in ("name", "kind", "degrees", "key", "locus", "sheet")}
    row.update(seed=list(map(float, s["seed"])), start_R=r.get("start_R"), R=r.get("R"),
               optimized=list(map(float, r.get("off_seed", r.get("best", s["seed"])))), optimized_status=r["status"],
               central=list(map(float, c["centre"])), depth=c.get("depth"), central_status=c["status"])
    for folder, key, tag in (("new_shapes/off", "seed", "seed"), ("optimized_off", "optimized", "opt"),
                             ("central_off", "central", "central")):
        b = np.asarray(row[key], float); b = b / b.sum()
        if tag != "seed" and G.label(b, s["hull_tol"])[1] != s["key"]:
            row[tag + "_note"] = "moved to another polytope: the seed is written instead"
            b = np.asarray(row["seed"]) / np.sum(row["seed"])
        V, Vn, idx, faces, cells = G.polytope(b, s["hull_tol"])
        d = os.path.join(%(root)r, folder, "ixitic"); os.makedirs(d, exist_ok=True)
        f = os.path.join(d, (f"ixitic_{s['kind']}_" if tag == "seed" else "") + f"{s['name']}_{len(V)}v_{len(cells)}c_{tag}.off")
        open(f, "w").write(to_4off(Vn, faces, cells))
        row[tag + "_off"] = os.path.relpath(f, %(root)r)
        E = [np.linalg.norm(Vn[a] - Vn[c_]) for a, c_ in (tuple(e) for e in hull_edges(faces))]
        row[tag + "_R"] = float(max(E) / min(E))
    out.append(row)
    print(f"{row['name']:28} {row['kind']:6} R {row['seed_R']:.4f} -> {row['opt_R']:.6f}  {row['optimized_status'][:45]:45} "
          f"central {row['central_status'][:28]}", flush=True)
print(json.dumps(out))
'''


def main(only_new=False):
    d = json.load(open(os.path.join(HERE, "ixitic_shapes.json")))["shapes"]
    count, todo = {}, {}
    rows = []
    if only_new:
        rows = json.load(open(os.path.join(HERE, "opt_central_ixitic.json")))["shapes"]
    done = {r["key"] for r in rows}
    for s in d:
        count[s["kind"]] = count.get(s["kind"], 0) + 1
        name = s.get("name") or f"ixitic_{s['kind']}_{count[s['kind']]:02d}"
        if s["key"] in done:
            continue
        todo.setdefault(s["sheet"], []).append({
            "name": name, "kind": s["kind"], "degrees": {"region": 3, "wall": 2, "line": 1, "point": 0}[s["kind"]],
            "seed": s["seed"], "key": s["key"], "sheet": s["sheet"],
            "hull_tol": 1e-9 if s["kind"] in ("region", "line", "point") else 1e-7})
    for sub in sorted(todo):
        r = subprocess.run([sys.executable, "-c", STEP % {"here": HERE, "root": ROOT}], input=json.dumps(todo[sub]),
                           capture_output=True, text=True, cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        lines = r.stdout.strip().splitlines()
        for line in lines[:-1]:
            print(f"{sub} {line}", flush=True)
        try:
            rows += json.loads(lines[-1])
        except Exception:
            print(sub, "FAILED", r.stderr[-800:], flush=True)
        json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": rows},
                  open(os.path.join(HERE, "opt_central_ixitic.json"), "w"), indent=1)
    gather(rows)


def gather(rows=None):
    rows = rows or json.load(open(os.path.join(HERE, "opt_central_ixitic.json")))["shapes"]
    for folder, tag, title in (("optimized_off", "opt", "Optimized"), ("central_off", "central", "Central")):
        L = [f"# {title} ixitic seeds", "", "Made by research/hyic_atlas/ixitic_offs.py; shapes from "
             "research/hyic_atlas/ixitic_shapes.md (sheet: the conjugate of ixitic the shape was found on).", ""]
        if tag == "opt":
            L += ["| shape | kind | sheet | R seed | R | status | beta | file |", "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {r['seed_R']:.6f} | {r['opt_R']:.8f} | "
                         f"{r['optimized_status']}{'; ' + r['opt_note'] if 'opt_note' in r else ''} | "
                         f"{[round(v, 9) for v in r['optimized']]} | {r['opt_off'].split('/')[-1]} |")
        else:
            L += ["| shape | kind | sheet | depth | status | beta | file |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                dd = "-" if r["depth"] is None else f"{r['depth']:.2e}"
                L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {dd} | {r['central_status']}"
                         f"{'; ' + r['central_note'] if 'central_note' in r else ''} | "
                         f"{[round(v, 9) for v in r['central']]} | {r['central_off'].split('/')[-1]} |")
        os.makedirs(os.path.join(ROOT, folder, "ixitic"), exist_ok=True)
        open(os.path.join(ROOT, folder, "ixitic", "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    gather() if arg == "gather" else main(only_new=arg == "new")
