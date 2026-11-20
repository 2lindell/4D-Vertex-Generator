"""Seed, optimized and central .off files of the ixitic shapes (opt_central_ixitic.json; new_shapes/off/ixitic,
optimized_off/ixitic, central_off/ixitic).

Each shape of ixitic_shapes.json is run on the sheet it was found on (HYIC_SUB = ix0..ix9) with the decaic optimiser
and centre finder (optimize_seeds.py); its locus: free for a region, the zero betas for a shape on a mirror face or edge
(kept zero), else the map's surfaces through its seed. Named as in the accepted list (ixitic_shapes.json), the rest
numbered (ixitic_region_NN, ...). A seed the optimiser or the centre finder moves to another polytope is not used: the
shape's own seed is written instead, and noted.

    python ixitic_offs.py [gather | new] [ixoic]   (new: only the shapes not yet in opt_central_<group>.json;
                                                    ixoic: every ixoic shape, as pages/ixoic_match.py gathers them)
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


MODE = %(mode)r                                   # ("": all; "seed": seed files only; "nocentre": no centre finder)


def job(s):
    if MODE == "seed":
        return ({"best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None, "status": "seed only"},
                {"centre": s["seed"], "depth": None, "status": "not computed"})
    try:
        r = OS.O.optimise(s)
    except Exception as e:
        r = {"best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None, "status": f"optimiser failed: {e}"}
    OS.CS.OPT = {s["name"]: r}
    if MODE == "nocentre":
        return r, {"centre": s["seed"], "depth": None, "status": "not computed (about 30 min a shape)"}
    try:
        c = OS.CS.centre(s)
    except Exception as e:
        c = {"centre": s["seed"], "depth": None, "status": f"centre finder failed: {e}"}
    return r, c


import time
from multiprocessing import TimeoutError as _Timeout
with Pool(4) as pool:                             # (a worker that dies (a crash in native code) loses its task and
    A = [pool.apply_async(job, (s,)) for s in S]  # pool.map waits forever: each shape is given 30 min at most, then
    t0, R = time.time(), []                       # its seed is written instead)
    for s, a in zip(S, A):
        try:
            R.append(a.get(timeout=max(1.0, t0 + 1800 - time.time())))
        except _Timeout:
            R.append(({"best": s["seed"], "off_seed": s["seed"], "R": None, "start_R": None,
                       "status": "optimiser crashed (scipy minimize) or ran over 30 min: the seed is written"},
                      {"centre": s["seed"], "depth": None, "status": "not computed"}))
    pool.terminate()
out = []
for s, (r, c) in zip(S, R):
    row = {k: s[k] for k in ("name", "kind", "degrees", "key", "locus", "sheet")}
    row.update(seed=list(map(float, s["seed"])), locus_used=r.get("locus_used"), start_R=r.get("start_R"), R=r.get("R"),
               optimized=list(map(float, r.get("off_seed", r.get("best", s["seed"])))), optimized_status=r["status"],
               central=list(map(float, c["centre"])), depth=c.get("depth"), central_status=c["status"])
    for folder, key, tag in (("new_shapes/off", "seed", "seed"), ("optimized_off", "optimized", "opt"),
                             ("central_off", "central", "central")):
        if (MODE == "seed" and tag != "seed") or (MODE == "nocentre" and tag == "central"):
            continue
        b = np.asarray(row[key], float); b = b / b.sum()
        if tag != "seed" and G.label(b, s["hull_tol"])[1] != s["key"]:
            row[tag + "_note"] = "moved to another polytope: the seed is written instead"
            b = np.asarray(row["seed"]) / np.sum(row["seed"])
        for tol in (s["hull_tol"], 1e-8, 1e-7):  # (a hull ambiguous at 1e-9, as some closure regions: 1e-8)
            try:
                V, Vn, idx, faces, cells = G.polytope(b, tol)
                text = to_4off(Vn, faces, cells)
                break
            except ValueError:
                if tol == 1e-7:
                    raise
        if tol != s["hull_tol"]:
            row[tag + "_hull_tol"] = tol
        d = os.path.join(%(root)r, folder, %(group)r); os.makedirs(d, exist_ok=True)
        f = os.path.join(d, (f"%(group)s_{s['kind']}_" if tag == "seed" and not s['name'].startswith("%(group)s_") else "")
                         + f"{s['name']}_{len(V)}v_{len(cells)}c_{tag}.off")
        open(f, "w").write(text)
        row[tag + "_off"] = os.path.relpath(f, %(root)r)
        E = [np.linalg.norm(Vn[a] - Vn[c_]) for a, c_ in (tuple(e) for e in hull_edges(faces))]
        row[tag + "_R"] = float(max(E) / min(E))
    out.append(row)
    print(f"{row['name']:28} {row['kind']:6} R {row['seed_R']:.4f} -> {row.get('opt_R', row['seed_R']):.6f}  {row['optimized_status'][:45]:45} "
          f"central {row['central_status'][:28]}", flush=True)
print(json.dumps(out))
'''


GROUP = "ixitic"
MODE = ""


def shapes():
    """Every shape of the group, named: from the accepted list where it is named, else numbered by kind."""
    d = json.load(open(os.path.join(HERE, f"{GROUP}_shapes.json")))["shapes"]
    nm = {}
    if GROUP == "ixoic":                        # (every ixoic shape: merged, the targeted search's confirmed ones
        sys.path.insert(0, os.path.join(ROOT, "pages"))   # (ixoic_crossings.json), the half-turns' fixed lines and
        from ixoic_match import atlas                       # the closure check's regions: pages/ixoic_match.py)
        d = atlas()
        try:                                    # (names from the accepted list: pages/ixoic_match.py)
            N = json.load(open(os.path.join(ROOT, "pages", "ixoic_names.json")))
        except FileNotFoundError:
            N = {}
        first = {k: v[0]["name"].lower() for k, v in N.items()}
        twice = {n for n in first.values() if list(first.values()).count(n) > 1}
        nm = {k: n + (f"_{s['vertices']}" if n in twice else "") for s in d for k, n in first.items() if s["key"] == k}
    count, out = {}, []
    for s in d:
        count[s["kind"]] = count.get(s["kind"], 0) + 1
        out.append({"name": s.get("name") or nm.get(s["key"]) or f"{GROUP}_{s['kind']}_{count[s['kind']]:02d}",
                    "kind": s["kind"], "degrees": {"region": 3, "wall": 2, "line": 1, "point": 0}[s["kind"]],
                    "seed": s["seed"], "key": s["key"], "sheet": s["sheet"],
                    "hull_tol": 1e-9 if s["kind"] in ("region", "line", "point") else 1e-7})
    return out


def main(only_new=False):
    rows, todo = [], {}
    if only_new:
        rows = json.load(open(os.path.join(HERE, f"opt_central_{GROUP}{'_seeds' if MODE == 'seed' else ''}.json")))["shapes"]
    done = {r["key"] for r in rows}
    for s in shapes():
        if s["key"] not in done:
            todo.setdefault(s["sheet"], []).append(s)
    todo = {(sub, i): L[i:i + 4] for sub, L in todo.items() for i in range(0, len(L), 4)}   # (saved every 4 shapes)
    for sub, i in sorted(todo):
        r = subprocess.run([sys.executable, "-c", STEP % {"here": HERE, "root": ROOT, "group": GROUP, "mode": MODE}], input=json.dumps(todo[sub, i]),
                           capture_output=True, text=True, cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        lines = r.stdout.strip().splitlines()
        for line in lines[:-1]:
            print(f"{sub} {line}", flush=True)
        try:
            rows += json.loads(lines[-1])
        except Exception:
            print(sub, "FAILED", r.stderr[-800:], flush=True)
        json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": rows},
                  open(os.path.join(HERE, f"opt_central_{GROUP}{'_seeds' if MODE == 'seed' else ''}.json"), "w"),
                  indent=1)
    if MODE != "seed":
        gather(rows)


def gather(rows=None):
    rows = rows or json.load(open(os.path.join(HERE, f"opt_central_{GROUP}.json")))["shapes"]
    for folder, tag, title in (("optimized_off", "opt", "Optimized"), ("central_off", "central", "Central")):
        L = [f"# {title} {GROUP} seeds", "", "Made by research/hyic_atlas/ixitic_offs.py; shapes from "
             f"research/hyic_atlas/{GROUP}_shapes.md (sheet: the conjugate of {GROUP} the shape was found on).", ""]
        if tag == "opt":
            L += ["| shape | kind | sheet | R seed | R | status | beta | file |", "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {r['seed_R']:.6f} | {r['opt_R']:.8f} | "
                         f"{r['optimized_status']}{'; ' + r['opt_note'] if 'opt_note' in r else ''} | "
                         f"{[round(v, 9) for v in r['optimized']]} | {r['opt_off'].split('/')[-1]} |")
        else:
            L += ["| shape | kind | sheet | depth | status | beta | file |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                if "central_off" not in r:
                    continue
                dd = "-" if r["depth"] is None else f"{r['depth']:.2e}"
                L.append(f"| {r['name']} | {r['kind']} | {r['sheet']} | {dd} | {r['central_status']}"
                         f"{'; ' + r['central_note'] if 'central_note' in r else ''} | "
                         f"{[round(v, 9) for v in r['central']]} | {r['central_off'].split('/')[-1]} |")
        os.makedirs(os.path.join(ROOT, folder, GROUP), exist_ok=True)
        open(os.path.join(ROOT, folder, GROUP, "README.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    for m in ("seed", "nocentre"):
        if m in args:
            MODE = m
            args.remove(m)
    if "ixoic" in args:
        GROUP = "ixoic"
        args.remove("ixoic")
    arg = args[0] if args else ""
    gather() if arg == "gather" else main(only_new=arg == "new")
