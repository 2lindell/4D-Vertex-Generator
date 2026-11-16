"""Add the new wall shapes a region closure check found (<atlas>/snub_closure_<sheet>.json, "new_wall_shapes") to
their atlas, as crossing_add.py adds the crossing search's: each is run through the atlas's optimiser (on its wall
surface) and centre finder and gets a seed, optimized and central .off file. Written to
<atlas>/closure_walls_<sheet>.json (the pages read it). Names are given here, numbered after the atlas's own.

    python closure_add.py contic ki wall_27=f7c174f6a037 wall_28=...      (runs the shapes in parallel, NPROC at a time)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from crossing_add import CODE, HERE, TARGET

SRC = "found by the region closure check (snub_closure.py)"


def run_one(adir, var, sheet, group, s):
    p = subprocess.run([sys.executable, "-c", CODE % {"root": HERE, "adir": adir}],
                       input=json.dumps({"group": group, "shapes": [s]}), capture_output=True, text=True,
                       cwd=os.path.join(HERE, adir), env={**os.environ, var: sheet})
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])[s["key"]]


def main():
    atlas, sheet, *pairs = sys.argv[1:]
    adir, var, group = TARGET[f"{atlas}_{sheet}"]
    C = json.load(open(os.path.join(HERE, adir, f"snub_closure_{sheet}.json")))["new_wall_shapes"]
    out = os.path.join(HERE, adir, f"closure_walls_{sheet}.json")
    X = json.load(open(out)) if os.path.exists(out) else {"_about": "new wall shapes found by the region closure "
                                                          "check (closure_add.py)", "shapes": []}
    have = {t["key"] for t in X["shapes"]}
    shapes = []
    for pr in pairs:
        nm, key = pr.split("=")
        if key in have:
            continue
        w = C[key]
        nv = int(w["signature"].split(":")[0])
        nc = sum(int(a) for a in re.findall(r"(\d+)x\[", w["signature"].split("|")[0]))
        shapes.append({"name": nm, "key": key, "kind": "wall", "degrees": 2, "seed": w["seed"], "locus": [w["surface"]],
                       "hull_tol": 1e-7, "signature": w["signature"], "sides": w["sides"],
                       "file": f"{group}_{nm}_{nv}v_{nc}c"})
    with ThreadPoolExecutor(int(os.environ.get("NPROC", 2))) as ex:
        res = list(ex.map(lambda s: run_one(adir, var, sheet, group, s), shapes))
    for s, o in zip(shapes, res):
        r, c = o["opt"], o["centre"]
        X["shapes"].append({"key": s["key"], "name": s["name"], "group": group, "signature": s["signature"],
                            "kind": "wall", "degrees": 2, "seed": s["seed"], "locus": s["locus"], "sides": s["sides"],
                            "source": SRC, "start_R": r.get("start_R"), "R": r.get("R"),
                            "optimized": o["opt"]["beta"], "optimized_status": r.get("status"),
                            "central": o["central"]["beta"], "depth": c.get("depth"), "central_status": c.get("status"),
                            "seed_off": o["seed"]["off"], "seed_R": o["seed"]["R"], "opt_off": o["opt"]["off"],
                            "opt_R": o["opt"]["R"], "central_off": o["central"]["off"], "central_R": o["central"]["R"],
                            "vertices": o["seed"]["counts"][0], "cells": o["seed"]["counts"][1],
                            **{k: v for k, v in o.items() if k.endswith("_note")}})
        print(s["name"], s["key"], "R", r.get("R"), r.get("status"), "| centre", c.get("status"), flush=True)
    json.dump(X, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
