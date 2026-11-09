"""Hidhi and matethi are points, not lines (line_check.py: each is an isolated change point on its line -- hidhi at
b1 = b4 on b2 = b3 = 0, matethi at b2 / (b2 + b3) = (sqrt(5) - 1)/2 on b1 = b4 = 0). Their seeds are put at the exact
points, their kind and degrees corrected (ixitic_shapes.json / .md, opt_central_ixitic.json), and their seed, optimized
and central .off files (all three the same polychoron: a point has no freedom) rewritten from the exact seed with the
wall tolerance (at a change point the region tolerance splits the hull's flat cells).

    python fix_points.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G5 = (5 ** 0.5 - 1) / 2
POINTS = {"hidhi": [0.5, 0.0, 0.0, 0.5], "matethi": [0.0, G5, 1 - G5, 0.0]}
CODE = r'''
import json, sys, numpy as np
sys.path.insert(0, %r)
import group as G
from four_d_vertex_generator.off import to_4off
from four_d_vertex_generator.local_view import hull_edges
a = json.loads(sys.argv[1])
b = np.asarray(a["seed"], float)
V, Vn, idx, faces, cells = G.polytope(b, 1e-7)
k = G.label(b, 1e-7)[1]
E = [np.linalg.norm(Vn[x] - Vn[y]) for x, y in (tuple(e) for e in hull_edges(faces))]
out = {}
for f in a["files"]:
    open(f, "w").write(to_4off(Vn, faces, cells))
print(json.dumps({"key": k, "V": len(V), "C": len(cells), "R": float(max(E) / min(E))}))
'''


def main():
    sj = os.path.join(HERE, "ixitic_shapes.json")
    D = json.load(open(sj))
    oc = os.path.join(HERE, "opt_central_ixitic.json")
    O = json.load(open(oc))
    for s in D["shapes"]:
        if s.get("name") not in POINTS:
            continue
        n = s["name"]
        s["kind"], s["seed"] = "point", POINTS[n]
        row = next(r for r in O["shapes"] if r["name"] == n)
        tag = f"{n}_{s['vertices']}v_{s['cells']}c"
        files = {"seed": f"new_shapes/off/ixitic/ixitic_point_{tag}_seed.off", "opt": f"optimized_off/ixitic/{tag}_opt.off",
                 "central": f"central_off/ixitic/{tag}_central.off"}
        r = subprocess.run([sys.executable, "-c", CODE % HERE, json.dumps({"seed": POINTS[n], "files": [
            os.path.join(ROOT, f) for f in files.values()]})], capture_output=True, text=True, cwd=HERE,
            env={**os.environ, "HYIC_SUB": s["sheet"]})
        res = json.loads(r.stdout.strip().splitlines()[-1])
        assert res["key"] == s["key"] and (res["V"], res["C"]) == (s["vertices"], s["cells"]), (n, res)
        for t in ("seed", "opt", "central"):
            old = row.get(f"{t}_off")
            if old and old != files[t] and os.path.exists(os.path.join(ROOT, old)):
                os.remove(os.path.join(ROOT, old))
            row[f"{t}_off"], row[f"{t}_R"] = files[t], res["R"]
            row.pop(f"{t}_note", None)
        row.update(kind="point", degrees=0, locus=[], seed=POINTS[n], optimized=POINTS[n], central=POINTS[n], R=res["R"],
                   start_R=res["R"], optimized_status="a point: no freedom (the exact seed)",
                   central_status="a point: the exact seed", depth=None)
        print(n, res)
    json.dump(D, open(sj, "w"), indent=1)
    json.dump(O, open(oc, "w"), indent=1)
    mp = os.path.join(HERE, "ixitic_shapes.md")
    L = open(mp).read().split("\n")
    for i, x in enumerate(L):
        for n, b in POINTS.items():
            if x.startswith(f"| {n} | line |"):
                c = x.split(" | ")
                c[1] = "point"
                c[-1] = f"{[round(v, 5) for v in b]} ({c[-1].split('(')[1]}"
                L[i] = " | ".join(c)
        if x.startswith("Totals:"):
            from collections import Counter
            L[i] = f"Totals: {dict(Counter(s['kind'] for s in D['shapes']))}."
    L.append("- hidhi and matethi: points, not lines (line_check.py: each is an isolated change point on its line, "
             "hidhi at b1 = b4 on b2 = b3 = 0 and matethi at b2 / (b2 + b3) = (sqrt(5) - 1)/2 on b1 = b4 = 0), as the "
             "accepted list has them; fix_points.py")
    open(mp, "w").write("\n".join(L).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
