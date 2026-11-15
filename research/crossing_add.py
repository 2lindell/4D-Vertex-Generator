"""Add the crossing search's confirmed shapes (crossing_search/verified.json) to their atlases.

Each confirmed shape is run through its atlas's optimiser (R = longest / shortest edge, minimised on its locus: the
walls whose crossing it lies on) and centre finder (the decaic ones, as every atlas runs them), and gets a seed,
optimized and central .off file (new_shapes/off/<group>/, optimized_off/<group>/, central_off/<group>/). It is written
to <atlas>/crossings_<sheet>.json, which the pages read (make_contic_page.py, make_decaic_page.py, make_hyic_page.py).
Oxitic shapes go through toxitic_atlas/add_crossing_shape.py instead (as oxitic_line_01 did; their central seeds then
from hyic_atlas/recentre.py --oxitic). Unnamed shapes are numbered after the group's existing ones. A point (three
walls) has no freedom: its seed is its optimized and central seed.

    python crossing_add.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "found where walls cross (crossing_search.py)"

# (atlas_sheet -> atlas dir, env var, group, file-name style)
TARGET = {"contic_ki": ("contic_atlas", "CONTIC_SUB", "kicontic"), "contic_io": ("contic_atlas", "CONTIC_SUB", "iocontic"),
          "decaic_ki": ("decaic_atlas", "DECAIC_SUB", "kidecaic"), "decaic_io": ("decaic_atlas", "DECAIC_SUB", "iodecaic"),
          "hyic_ix0": ("hyic_atlas", "HYIC_SUB", "ixitic")}
for i in range(1, 10):
    TARGET[f"hyic_ix{i}"] = ("hyic_atlas", "HYIC_SUB", "ixitic")

CODE = r'''
import json, os, sys
import numpy as np
ROOT = %(root)r
ADIR = os.path.join(ROOT, %(adir)r)
sys.path.insert(0, ADIR)
os.chdir(ADIR)
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
from four_d_vertex_generator.off import to_4off
from four_d_vertex_generator.local_view import hull_edges
if %(adir)r == "contic_atlas":
    import central_seeds as CSR
    O, CS = CSR.C.O, CSR.CS
    CS.O, CS.G, CS.F = CSR.C.O, CSR.C.G, CSR.C.F
elif %(adir)r == "hyic_atlas":
    import optimize_seeds as H
    O, CS = H.O, H.CS
else:
    import optimize_seeds as O
    import central_seeds as CS
import group as G
A = json.loads(sys.stdin.read())


def off(beta, path):
    beta = np.asarray(beta, float)
    beta = beta / beta.sum()
    P = G.polytope(beta, 1e-9)
    V, faces, cells = (P[1], P[3], P[4]) if len(P) == 5 else (P[1], P[2], P[3])
    V = np.asarray(V, float)
    V = V / np.linalg.norm(V, axis=1).max()
    open(path, "w").write(to_4off(V, faces, cells))
    E = [np.linalg.norm(V[x] - V[y]) for x, y in (tuple(e) for e in hull_edges(faces))]
    return len(V), len(cells), float(max(E) / min(E))


out = {}
for s in A["shapes"]:
    key = s["key"]
    try:
        r = O.optimise(s)
    except Exception as e:
        r = {"best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None, "status": f"optimiser failed: {e}"}
    CS.OPT = {s["name"]: r}
    try:
        c = CS.centre(s)
    except Exception as e:
        c = {"centre": s["seed"], "depth": None, "status": f"centre finder failed: {e}"}
    o = {"opt": r, "centre": c}
    for tag, beta, folder, fname in (("seed", s["seed"], "new_shapes/off", s["file"] + "_seed.off"),
                                     ("opt", r.get("off_seed", r.get("best", s["seed"])), "optimized_off", s["file"] + "_opt.off"),
                                     ("central", c.get("centre", s["seed"]), "central_off", s["file"] + "_central.off")):
        b = np.asarray(beta, float)
        b = b / b.sum()
        if G.label(b, s["hull_tol"])[1] != key:              # (moved off the shape: keep the seed)
            o[tag + "_note"] = "left the shape: the seed is written instead"
            b = np.asarray(s["seed"], float)
        d = os.path.join(ROOT, folder, A["group"])
        os.makedirs(d, exist_ok=True)
        nv, nc, R = off(b, os.path.join(d, fname))
        o[tag] = {"beta": (b / b.sum()).tolist(), "off": os.path.relpath(os.path.join(d, fname), ROOT), "R": R,
                  "counts": [nv, nc]}
    out[key] = o
print(json.dumps(out))
'''


def number(group, kind, existing):
    """The next number for an unnamed shape of this group and kind, after the atlas's own (kidecaic, kicontic and
    iocontic number all their kinds in one series, as new_shapes does; ixitic and oxitic each kind on its own)."""
    pat = rf"(?:^|_){kind}_(\d+)" if group in ("ixitic", "oxitic") else r"(?:^|_)(?:region|wall|line|point)_(\d+)"
    nums = [int(m.group(1)) for n in existing for m in [re.search(pat, n)] if m]
    return max(nums, default=0) + 1


def existing_names(adir, group, sheet):
    names = []
    if group in ("kicontic", "iocontic"):
        names += [r["name"] for r in json.load(open(os.path.join(HERE, "contic_atlas", f"optimized_seeds_{sheet}.json")))["shapes"]]
        sys.path.insert(0, os.path.join(HERE, "contic_atlas"))
        import given_names                          # (the atlas's numbered names that were then given list names)
        names += list({"kicontic": given_names.KICONTIC, "iocontic": given_names.IOCONTIC}[group])
    if group in ("kidecaic", "iodecaic", "kicontic", "iocontic"):
        names += [r["name"].replace(group + "_", "") for r in json.load(open(os.path.join(HERE, "new_shapes", "new_shapes.json")))["shapes"]
                  if r["group"] == group]
    if group == "ixitic":
        names += [r["name"] for r in json.load(open(os.path.join(HERE, "hyic_atlas", "opt_central_ixitic.json")))["shapes"]]
    for f in os.listdir(os.path.join(HERE, adir)):
        if f.startswith("crossings_") and f.endswith(".json"):
            names += [t["name"] for t in json.load(open(os.path.join(HERE, adir, f)))["shapes"]]
    return names


def main():
    V = json.load(open(os.path.join(HERE, "crossing_search", "verified.json")))
    report = []
    for name, cands in V.items():
        atlas, sheet = name.split("_", 1)
        C = json.load(open(os.path.join(HERE, "crossing_search", f"{name}.json")))["candidates"]
        conf = {k: r for k, r in cands.items() if r["verdict"] == "confirmed"}
        if not conf:
            continue
        if name == "contic_ox":                         # (oxitic: its own adder)
            done = {t["key"] for t in json.load(open(os.path.join(HERE, "toxitic_atlas", "crossings_ox.json")))["shapes"]}
            D = json.load(open(os.path.join(HERE, "toxitic_atlas", "toxitic_shapes.json")))["oxitic"]
            for k, r in conf.items():
                if k in done:
                    continue
                kind = r["kind"]
                n = len([s for s in D if s["kind"] == kind and s["symmetry"] <= 576]) + 1
                nm = f"oxitic_{kind}_{n:02d}"
                p = subprocess.run([sys.executable, "add_crossing_shape.py", "ox", nm, json.dumps(r["seed"])]
                                   + C[k]["surfaces"] + ["--source", SRC], cwd=os.path.join(HERE, "toxitic_atlas"),
                                   capture_output=True, text=True)
                print(name, k, nm, "added" if p.returncode == 0 else "FAILED " + p.stderr[-800:], flush=True)
                D = json.load(open(os.path.join(HERE, "toxitic_atlas", "toxitic_shapes.json")))["oxitic"]
                report.append((atlas, sheet, nm, k, kind, r["degrees"], r["seed"]))
            continue
        adir, var, group = TARGET[name]
        cf = os.path.join(HERE, adir, f"crossings_{sheet}.json")
        X = json.load(open(cf)) if os.path.exists(cf) else {"_about": "shapes where interior walls cross (crossing_add.py)",
                                                              "shapes": []}
        have = {t["key"] for t in X["shapes"]}
        shapes = []
        for k, r in conf.items():
            if k in have:
                continue
            ex = existing_names(adir, group, sheet) + [s["name"] for s in shapes]
            nv = int(r["signature"].split(":")[0])
            ncell = sum(int(a) for a in re.findall(r"(\d+)x\[", r["signature"].split("|")[0]))
            num = number(group, r["kind"], ex)
            nm = (f"{group}_" if group == "ixitic" else "") + f"{r['kind']}_{num:02d}"
            shapes.append({"name": nm, "key": k, "kind": r["kind"], "degrees": r["degrees"], "seed": r["seed"],
                           "locus": C[k]["surfaces"], "hull_tol": 1e-7, "signature": r["signature"],
                           "file": f"{group}_{r['kind']}_{num:02d}_{nv}v_{ncell}c"})
        if not shapes:
            continue
        p = subprocess.run([sys.executable, "-c", CODE % {"root": HERE, "adir": adir}],
                           input=json.dumps({"group": group, "shapes": shapes}), capture_output=True, text=True,
                           cwd=os.path.join(HERE, adir), env={**os.environ, var: sheet})
        if p.returncode:
            print(name, "FAILED", p.stderr[-2000:], flush=True)
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        for s in shapes:
            o = res[s["key"]]
            r, c = o["opt"], o["centre"]
            X["shapes"].append({"key": s["key"], "name": s["name"], "group": group, "signature": s["signature"],
                                "kind": s["kind"], "degrees": s["degrees"], "seed": s["seed"], "locus": s["locus"],
                                "source": SRC, "start_R": r.get("start_R"), "R": r.get("R"),
                                "optimized": o["opt"]["beta"], "optimized_status": r.get("status"),
                                "central": o["central"]["beta"], "depth": c.get("depth"), "central_status": c.get("status"),
                                "seed_off": o["seed"]["off"], "seed_R": o["seed"]["R"], "opt_off": o["opt"]["off"],
                                "opt_R": o["opt"]["R"], "central_off": o["central"]["off"], "central_R": o["central"]["R"],
                                "vertices": o["seed"]["counts"][0], "cells": o["seed"]["counts"][1],
                                **{k: v for k, v in o.items() if k.endswith("_note")}})
            print(name, s["key"], s["name"], "R", r.get("R"), r.get("status"), "| centre", c.get("status"), flush=True)
            report.append((atlas, sheet, s["name"], s["key"], s["kind"], s["degrees"], s["seed"]))
        json.dump(X, open(cf, "w"), indent=1)
    if report:
        md = os.path.join(HERE, "crossing_search", "README.md")
        L = open(md).read().rstrip("\n")
        if "## Added to the atlases" not in L:
            L += ("\n\n## Added to the atlases\n\nConfirmed by crossing_verify.py (verified.json: same key at hull "
                  "tolerances 1e-7 to 1e-9; kept along the locus to 1e-3; changed 1e-5 off every wall) and checked to "
                  "have exactly their group's symmetry (fullsym.py). None is on the accepted lists. Added by "
                  "crossing_add.py.\n\n| atlas | sheet | name | key | kind (degrees) | seed |\n|---|---|---|---|---|---|")
        for a, sh, nm, k, kind, dg, seed in report:
            L += f"\n| {a} | {sh} | {nm} | {k} | {kind} ({dg}) | {[round(x, 6) for x in seed]} |"
        open(md, "w").write(L + "\n")


if __name__ == "__main__":
    main()
