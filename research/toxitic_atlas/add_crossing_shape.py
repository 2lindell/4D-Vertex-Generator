"""Add a shape found where interior walls cross (a line or point the map's strata do not cover) to the oxitic atlas:
crossings_<sheet>.json (read by shapes.py), toxitic_shapes.json / .md, opt_central_<sheet>.json, and its seed and
optimized .off files (its central seed: hyic_atlas/recentre.py --oxitic <name>).

    python add_crossing_shape.py <sheet> <name> '<beta json>' '<surface>' ['<surface>' ...] [--source '<text>']
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTIC = os.path.join(ROOT, "contic_atlas")

CODE = r'''
import json, sys, os
import numpy as np
sys.path.insert(0, %(contic)r); sys.path.insert(0, %(here)r)
os.chdir(%(contic)r)
import group as G
from strata import symmetry
from four_d_vertex_generator.off import to_4off
from four_d_vertex_generator.local_view import hull_edges
import optimize_central as OC
a = json.loads(sys.argv[1])
b = np.asarray(a["seed"], float); b = b / b.sum()
kind = "point" if len(a["locus"]) >= 3 else "line" if len(a["locus"]) == 2 else "wall"
sg, k = G.label(b, 1e-7)
n, imp = symmetry(b)
s = {"group": "oxitic", "name": a["name"], "kind": kind, "degrees": {"wall": 2, "line": 1, "point": 0}[kind],
     "seed": list(map(float, b)), "key": k, "hull_tol": 1e-9, "locus": a["locus"]}
r, c = OC.job(s)
out = {"key": k, "signature": sg, "kind": kind, "symmetry": int(n), "improper": int(imp), "opt": r}
for folder, beta, tag in (("new_shapes/off/oxitic", b, "seed"), ("optimized_off/oxitic", r.get("off_seed", r["best"]), "opt")):
    beta = np.asarray(beta, float); beta = beta / beta.sum()
    if G.label(beta, 1e-9)[1] != k:
        out[tag + "_note"] = "moved to another polytope: the seed is written instead"
        beta = b
    V, Vn, idx, faces, cells = G.polytope(beta, 1e-9)
    name = (f"oxitic_{kind}_" if tag == "seed" else "") + f"{a['name']}_{len(V)}v_{len(cells)}c_{tag}.off"
    f = os.path.join(%(root)r, folder, name)
    open(f, "w").write(to_4off(Vn, faces, cells))
    E = [np.linalg.norm(Vn[x] - Vn[y]) for x, y in (tuple(e) for e in hull_edges(faces))]
    out[tag + "_off"] = os.path.relpath(f, %(root)r)
    out[tag + "_R"] = float(max(E) / min(E))
    out[tag + "_counts"] = [len(V), len(cells)]
print(json.dumps(out))
'''


def main():
    args = sys.argv[1:]
    src = "found where walls cross"
    if "--source" in args:
        i = args.index("--source")
        src = args[i + 1]
        args = args[:i] + args[i + 2:]
    sheet, name, beta, locus = args[0], args[1], json.loads(args[2]), args[3:]
    p = subprocess.run([sys.executable, "-c", CODE % {"contic": CONTIC, "here": HERE, "root": ROOT},
                        json.dumps({"name": name, "seed": beta, "locus": locus})], capture_output=True, text=True,
                       cwd=CONTIC, env={**os.environ, "CONTIC_SUB": sheet})
    if p.returncode:
        sys.exit(p.stderr[-2000:])
    o = json.loads(p.stdout.strip().splitlines()[-1])
    seed = [b / sum(beta) for b in beta]
    sg = o["signature"]
    # (crossings_<sheet>.json, read by shapes.py on a rerun)
    cf = os.path.join(HERE, f"crossings_{sheet}.json")
    C = json.load(open(cf)) if os.path.exists(cf) else {"_about": "shapes where interior walls cross", "shapes": []}
    C["shapes"] = [t for t in C["shapes"] if t["key"] != o["key"]] + [
        {"key": o["key"], "name": name, "signature": sg, "seed": seed, "kind": o["kind"], "locus": locus, "source": src}]
    json.dump(C, open(cf, "w"), indent=1)
    # (toxitic_shapes.json / .md)
    D = json.load(open(os.path.join(HERE, "toxitic_shapes.json")))
    import re
    from collections import Counter
    e = Counter()
    for n_, k_ in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k_)] += int(n_)
    row = {"key": o["key"], "signature": sg, "kind": o["kind"], "seed": seed, "sheet": sheet, "symmetry": o["symmetry"],
           "improper": o["improper"], "sources": [f"{sheet} {src}"], "sheets": [sheet], "name": None,
           "vertices": int(sg.split(":")[0]), "cells": sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0])),
           "edges": dict(sorted(e.items()))}
    D["oxitic"] = [s for s in D["oxitic"] if s["key"] != o["key"]] + [row]
    json.dump(D, open(os.path.join(HERE, "toxitic_shapes.json"), "w"), indent=1)
    md = os.path.join(HERE, "toxitic_shapes.md")
    L = open(md).read().rstrip("\n").split("\n")
    L = [x for x in L if f"({name})" not in x]
    L.append(f"- {name} (added: {src}): {o['kind']}, {row['vertices']} vertices, {row['cells']} cells, edges {row['edges']}, "
             f"symmetry {o['symmetry']}, seed {[round(v, 6) for v in seed]} ({sheet}); on {' and '.join(locus)} = 0")
    open(md, "w").write("\n".join(L) + "\n")
    # (opt_central_<sheet>.json)
    oc = os.path.join(HERE, f"opt_central_{sheet}.json")
    O = json.load(open(oc))
    r = o["opt"]
    O["shapes"] = [s for s in O["shapes"] if s["name"] != name] + [{
        "group": "oxitic", "name": name, "kind": o["kind"], "degrees": {"wall": 2, "line": 1, "point": 0}[o["kind"]],
        "key": o["key"], "locus": locus, "sheet": sheet, "seed": seed, "start_R": r.get("start_R"), "R": r.get("R"),
        "optimized": list(map(float, r.get("off_seed", r.get("best", seed)))), "optimized_status": r["status"],
        "central": seed, "depth": None, "central_status": "pending (recentre.py --oxitic)", "opt_off": o["opt_off"],
        "opt_counts": o["opt_counts"], "off_R": o["opt_R"], "central_off": o["opt_off"].replace("optimized_off", "central_off").replace("_opt.off", "_central.off"),
        "central_counts": o["seed_counts"], **({"opt_note": o["opt_note"]} if "opt_note" in o else {})}]
    json.dump(O, open(oc, "w"), indent=1)
    print(json.dumps({k: v for k, v in o.items() if k != "opt"}, indent=1), "\noptimiser:", r["status"], r.get("R"))


if __name__ == "__main__":
    main()
