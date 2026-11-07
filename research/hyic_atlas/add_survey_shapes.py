"""Add the surface survey's new polytopes (surface_survey_ix*.json) to ixitic_shapes.json / .md.

Each new polytope was found on one of the map's wall surfaces (a sample on the surface, classified with the wall
tolerance), so it is a wall shape; it gets its full symmetry and the ixoic elements kept as in ixitic_shapes.py, and a
name from the accepted list where its edges and cells match one.

    python add_survey_shapes.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ixitic_shapes as IS  # noqa: E402

NAMES = {"ce1564d83d55": "awosbitdy"}
NOTES = {"awosbitdy": "awosbitdy: found by the surface survey on sheet ix9, on the wall b1 b3 = (sqrt(5) - 1)/2 b2 b4: "
                      "cells exactly the list's (120x2 snits, 480 trigypos, 720x2 phywdows, 1440+1440 tets and 1440 "
                      "hexflanges, 6480) and edges 88 12,4^4,3^10 exactly"}


def main():
    path = os.path.join(HERE, "ixitic_shapes.json")
    D = json.load(open(path))
    known = {s["key"] for s in D["shapes"]}
    new = {}
    for i in range(10):
        f = os.path.join(HERE, f"surface_survey_ix{i}.json")
        if not os.path.exists(f):
            continue
        for k, t in json.load(open(f))["new"].items():
            if k not in known and k not in new:
                new[k] = (f"ix{i}", t)
    if not new:
        print("nothing new")
        return
    added = []
    for sub, items in sorted({v[0] for v in new.values()} and
                             {s: [(k, t) for k, (s_, t) in new.items() if s_ == s] for s in {v[0] for v in new.values()}}.items()):
        r = subprocess.run([sys.executable, "-c", IS.CODE % HERE], input=json.dumps([[t["at"], "wall"] for _, t in items]),
                           capture_output=True, text=True, cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        if r.returncode:
            print(sub, r.stderr[-800:])
            continue
        for (k, t), (sg, k2, n, imp, keeps) in zip(items, json.loads(r.stdout)):
            if k2 != k:
                print(f"{k}: reclassified as {k2}, skipped")
                continue
            s = {"key": k, "signature": sg, "kind": "wall", "seed": t["at"], "sheet": sub, "symmetry": n, "improper": imp,
                 "other_group_elements_kept": keeps, "sources": [f"{sub} surface survey: {t['on']}"[:120]], "sheets": [sub],
                 "vertices": int(sg.split(":")[0]),
                 "cells": sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))}
            e = Counter()
            for n_, k_ in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
                e[int(k_)] += int(n_)
            s["edges"] = dict(sorted(e.items()))
            s["name"] = NAMES.get(k)
            D["shapes"].append(s)
            added.append(s)
            print(f"added {k} {s['name'] or ''} {s['vertices']}v {s['cells']}c sym {n} edges {s['edges']}")
    json.dump(D, open(path, "w"), indent=1)
    # (the .md: rows after the last wall row, totals, notes)
    mp = os.path.join(HERE, "ixitic_shapes.md")
    L = open(mp).read().split("\n")
    last = max(i for i, x in enumerate(L) if x.startswith("| ") and "| wall |" in x)
    rows = [f"| {s['name'] or ''} | wall | {s['vertices']} | {s['cells']} | {s['edges']} | {s['symmetry']} | "
            f"{s['other_group_elements_kept']} |  | 1 | {[round(v, 5) for v in s['seed']]} ({s['sheet']}) |" for s in added]
    L[last + 1:last + 1] = rows
    cnt = Counter(s["kind"] for s in D["shapes"])
    L = [f"Totals: {dict(cnt)}." if x.startswith("Totals:") else x for x in L]
    L = [x.replace("not found: awosbitdy (6480 cells: the two 6480-cell atlas regions have other edges), sastithi",
                   "not found: sastithi") if "awosbitdy" in NAMES.values() else x for x in L]
    i = next(i for i, x in enumerate(L) if x.startswith("- not found"))
    L[i:i] = ["- " + NOTES[s["name"]] for s in added if s["name"] in NOTES]
    L.append("- added from the surface survey (add_survey_shapes.py): " + ", ".join(
        f"{s['name'] or s['key']} ({s['sheet']}, {s['cells']} cells)" for s in added))
    open(mp, "w").write("\n".join(L).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
