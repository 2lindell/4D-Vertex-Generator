"""Which ixoic shapes are also ixitic shapes (ixitic_relations.json, ixitic_relations.md).

Each ixitic sheet's group ixN lies inside the ixoic sheet's group ixoN (checked here). An ixoic shape whose seed has a
stabiliser (fewer than 2880 vertices) has, under ixN, an orbit inside its ixoic one: the same vertices when the
stabilising half-turn is not in ixitic (then it is the same polytope, isogonal under ixitic too), half of them when it is
(a different, smaller polytope). Each ixoic shape below 2880 vertices (ixoic_match.atlas()) is labelled at its seed under
ixoN at the hull tolerance its own ixoic key needs, and called the same polytope when the vertices and the edges at each
valence agree; the ixitic shape is named from opt_central_ixitic.json / the crossing shapes where it has a name.

    python ixitic_relations.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES = os.path.join(os.path.dirname(HERE), "pages")

CODE = r'''
import sys, json, re
import numpy as np
from collections import Counter
sys.path.insert(0, %r)
import group as G
def edges(sg):
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += int(n)
    return {str(k): v for k, v in sorted(e.items())}
mode, out = sys.argv[1], {}
for it in json.loads(sys.stdin.read()):
    b = np.asarray(it["seed"], float); b = b / b.sum(); r = None
    for t in (1e-9, 1e-8, 1e-7, 1e-6):
        try:
            info = G.classify(b, t)
        except Exception:
            info = None
        if info and ((mode == "ix" and t == it.get("tol")) or (mode == "ixo" and info["key"] == it["key"])):
            sg = G.signature(info)
            r = [t, info["key"], edges(sg), int(sg.split(":")[0])]
            break
    out[it["key"]] = r
print(json.dumps(out))
'''


def run(mode, items, sub):
    p = subprocess.run([sys.executable, "-c", CODE % HERE, mode], input=json.dumps(items), capture_output=True, text=True,
                       cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    sys.path.insert(0, PAGES)
    import ixoic_match as M
    A = M.atlas()
    N = json.load(open(os.path.join(PAGES, "ixoic_names.json")))
    items = [{"key": a["key"], "sheet": a["sheet"], "seed": a["seed"], "V": a["vertices"], "kind": a["kind"],
              "names": [n["name"] for n in N.get(a["key"], [])]} for a in A if a["vertices"] < 2880]
    ixname = {o["key"]: o["name"] for o in json.load(open(os.path.join(HERE, "opt_central_ixitic.json")))["shapes"] if o.get("key")}
    for i in range(10):
        try:
            ixname.update({t["key"]: t["name"] for t in json.load(open(os.path.join(HERE, f"crossings_ix{i}.json")))["shapes"]})
        except FileNotFoundError:
            pass
    ixkind = {s["key"]: s["kind"] for s in json.load(open(os.path.join(HERE, "ixitic_shapes.json")))["shapes"]}
    by = {}
    for it in items:
        by.setdefault(it["sheet"], []).append(it)
    rows = []
    for sh, its in sorted(by.items()):
        a = run("ixo", its, sh)
        for it in its:
            it["tol"] = a[it["key"]][0] if a[it["key"]] else None
        b = run("ix", its, "ix" + sh[3:])
        for it in its:
            x, y = a[it["key"]], b[it["key"]]
            rel = ("same" if x and y and x[2] == y[2] and x[3] == y[3] else
                   "half" if x and y and 2 * y[3] == x[3] else "unlabelled")
            rows.append({"ixoic": it["key"], "names": it["names"], "kind": it["kind"], "vertices": it["V"], "sheet": sh,
                         "tol": it["tol"], "ixitic": y[1] if y else None, "ixitic_vertices": y[3] if y else None,
                         "ixitic_kind": ixkind.get(y[1]) if y else None, "ixitic_name": ixname.get(y[1]) if y else None,
                         "relation": rel})
    # (a line whose recorded seed sits too near a change to label: a seed mid-way along its stretch of the half-turn
    #  line traced whole, fixed_circles.json, in sheet ixo0)
    try:
        import numpy as np
        FC = json.load(open(os.path.join(HERE, "fixed_circles.json")))
        sys.path.insert(0, HERE)
        os.environ["HYIC_SUB"] = "ixo0"
        import group as G0
        for r in rows:
            if r["relation"] != "unlabelled":
                continue
            for c in FC["circles"]:
                st = [x for x in c["stretches"] if x["key"] == r["ixoic"]]
                if st:
                    u, v = map(np.array, c["plane"])
                    th = st[0]["from"] + np.pi / FC["samples"] * st[0]["samples"] / 2
                    b = np.linalg.solve(G0.X, np.cos(th) * u + np.sin(th) * v)
                    it = [{"key": r["ixoic"], "seed": (b / b.sum()).tolist()}]
                    x = run("ixo", it, "ixo0")[r["ixoic"]]
                    if x:
                        it[0]["tol"] = x[0]
                        y = run("ix", it, "ix0")[r["ixoic"]]
                        if y and x[2] == y[2] and x[3] == y[3]:
                            r.update(ixitic=y[1], ixitic_vertices=y[3], ixitic_kind=ixkind.get(y[1]),
                                     ixitic_name=ixname.get(y[1]), relation="same", sheet="ixo0 (mid-stretch)")
                    break
    except FileNotFoundError:
        pass
    rows.sort(key=lambda r: (r["relation"], r["vertices"], r["ixoic"]))
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "pairs": rows}, open(os.path.join(HERE, "ixitic_relations.json"), "w"), indent=1)
    L = ["# Ixoic shapes that are also ixitic", "", __doc__.strip().split("\n\n")[1], "",
         "| ixoic | name | vertices | kind | ixitic | ixitic name | ixitic kind | relation |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['ixoic']} | {', '.join(r['names']) or '—'} | {r['vertices']} | {r['kind']} | {r['ixitic'] or '—'} | "
                 f"{r['ixitic_name'] or '—'} | {r['ixitic_kind'] or '—'} | {r['relation']}"
                 + (f" ({r['ixitic_vertices']} vertices)" if r["relation"] == "half" else "") + " |")
    open(os.path.join(HERE, "ixitic_relations.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[4:]))


if __name__ == "__main__":
    main()
