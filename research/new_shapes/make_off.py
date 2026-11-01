"""The new shapes found by the decaic-family and contic maps, as .off files with their seeds (new_shapes.md).

  * kidecaic / iodecaic: every shape of wall_shapes_<sub>.json (the map's regions and its cleanly classified walls) not
    on the accepted list (decaic_atlas/names_sub.py's
    names, matched by edge valences: decaic_atlas/edges.py). Regions have 3 degrees of freedom, walls 2. The seed is
    the map's example point (a grid point for a region, a point put exactly on the wall's surface for a wall). Every
    file is checked to rebuild with the map's vertex and cell counts.
  * contic: every shape of contic_atlas (no accepted list yet), with its degrees from the exact arrangement.

Seeds: beta (the map's coordinates) and the seed vector itself. Decaic family: x in R^5 (coordinates summing to 0),
vertices every sign * x[perm] of the group (kidecaic: even perms, both signs; iodecaic: even perms with +, odd perms
with -). Contic: x in R^4 in the standard F4 frame, vertices its orbit under F4 and the 24-cell/dual swap.

    python make_off.py
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
from four_d_vertex_generator.off import to_4off  # noqa: E402

# (the accepted list's kidecaic / iodecaic shapes by their edge valences: decaic_atlas/edges.py)
LIST_EDGES = {"ki": [{3: 120, 5: 120, 8: 60, 4: 420}, {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60},
                     {3: 360, 4: 300, 8: 120}, {3: 300, 4: 300, 7: 60}, {3: 240, 4: 360, 8: 60},
                     {3: 300, 4: 120, 5: 60, 6: 60}, {3: 420, 4: 60, 5: 60, 6: 60}],
              "io": [{3: 120, 4: 300, 5: 240}, {3: 360, 4: 60, 5: 240, 6: 60}, {3: 240, 4: 240}, {3: 360, 4: 180}]}


def listed(sub, t):
    e = {int(k): v for k, v in t["edges_by_valence"].items()}
    return any(e == L for L in LIST_EDGES[sub])


def load_group(path, sub=None):
    for m in ("group", "forms", "qsqrt2"):
        sys.modules.pop(m, None)
    if sub:
        os.environ["DECAIC_SUB"] = sub
    sys.path.insert(0, path)
    G = importlib.import_module("group")
    sys.path.pop(0)
    return G


def edges(sg):
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += int(n)
    return dict(sorted(e.items()))


def cell_text(faces, cells):
    c = Counter()
    for cl in cells:
        vs = {v for f in cl for v in faces[f]}
        sizes = Counter(len(faces[f]) for f in cl)
        c[(len(vs), "+".join(f"{n}x{k}" for k, n in sorted(sizes.items())))] += 1
    return " + ".join(f"{n}x{v}v({p})" for (v, p), n in sorted(c.items(), key=lambda r: (-r[0][0], r[0][1])))


def main():
    rows = []
    out_dir = os.path.join(HERE, "off")
    os.makedirs(out_dir, exist_ok=True)
    dec = os.path.join(ROOT, "decaic_atlas")
    for sub, title in (("ki", "kidecaic"), ("io", "iodecaic")):
        G = load_group(dec, sub)
        d = json.load(open(os.path.join(dec, f"wall_shapes_{sub}.json")))["shapes"]
        k = 0
        for t in d:
            if t["edges_by_valence"] and listed(sub, t):
                continue
            region = t["kind"] == "region"
            b = np.array(t["seed_beta"])
            _, V, faces, cells = G.polytope(b, 1e-9 if region else 1e-7)
            if len(V) != t["vertices"] or len(cells) != t["cells"]:
                raise RuntimeError(f"{sub} {t['kind']} {t['cells']}: rebuilt with {len(cells)} cells")
            k += 1
            name = f"{title}_{'region' if region else 'wall'}_{k:02d}_{len(V)}v_{len(cells)}c"
            open(os.path.join(out_dir, name + ".off"), "w").write(to_4off(V, faces, cells))
            x = G.x5(b)
            rows.append({"name": name, "group": title, "kind": "region" if region else "wall",
                         "degrees": 3 if region else 2, "vertices": len(V), "cells": len(cells),
                         "cell_types": cell_text(faces, cells), "edges_by_valence": t["edges_by_valence"],
                         "surface": next(iter(t["surfaces"]), "") if not region else "",
                         "beta": [round(float(v), 15) for v in b], "seed_vector": [round(float(v), 15) for v in x / np.abs(x).max()]})
    C = load_group(os.path.join(ROOT, "contic_atlas"))
    cs = json.load(open(os.path.join(ROOT, "contic_atlas", "contic_shapes.json")))["shapes"]
    exact = json.load(open(os.path.join(ROOT, "contic_atlas", "shapes.json")))["shapes"]
    for s, e in zip(cs, exact):
        b = np.array([float(Fraction(a)) + float(Fraction(c)) * 2 ** 0.5 for a, c in e["example"]])
        if (len(e["example_float"]), s["vertices"]) != (4, e["vertices"]):
            raise RuntimeError("contic shape lists out of step")
        V, Vn, idx, faces, cells = C.polytope(b)
        if len(V) != s["vertices"] or len(cells) != s["n_cells"]:
            raise RuntimeError(f"contic {s['decaic_counterpart']}: rebuilt with {len(cells)} cells, not {s['n_cells']}")
        name = f"contic_{s['kind']}_{s['decaic_counterpart']}-like_{len(V)}v_{len(cells)}c"
        open(os.path.join(out_dir, name + ".off"), "w").write(to_4off(Vn, faces, cells))
        x = C.x4(b)
        rows.append({"name": name, "group": "contic", "kind": s["kind"], "degrees": s["degrees"],
                     "vertices": len(V), "cells": len(cells), "cell_types": cell_text(faces, cells),
                     "beta": [round(float(v), 15) for v in b], "beta_exact": s["seed_beta"],
                     "seed_vector": [round(float(v), 15) for v in x / np.abs(x).max()],
                     "decaic_counterpart": s["decaic_counterpart"]})
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": rows}, open(os.path.join(HERE, "new_shapes.json"), "w"), indent=1)
    tot = Counter((r["group"], r["kind"]) for r in rows)
    L = ["# New shapes, with seeds and .off files", "",
         "Each .off file (off/) is the convex hull of the seed's orbit (4OFF: vertices, faces, cells), vertices scaled to "
         "unit radius.", "",
         "Seeds:",
         "* **beta** = the map's coordinates (decaic family: beta_k = x_k - x_{k+1}; contic: distances to the four F4 "
         "mirrors), any positive multiple is the same shape.",
         "* **seed vector**: decaic family, x in R^5 (summing to 0), the vertices are sign * x[perm] for the group "
         "(kidecaic: every even perm with both signs; iodecaic: even perms with +, odd perms with -). Contic: x in R^4 in "
         "the standard F4 frame (roots e_i +- e_j, e_i, (+-1/2)^4), the orbit under F4 and the 24-cell/dual swap.", "",
         "Totals: " + "; ".join(f"{g} {k}s: {n}" for (g, k), n in sorted(tot.items())) + f"; **{len(rows)} in all**.", ""]
    for g in ("kidecaic", "iodecaic", "contic"):
        L += [f"## {g}", "", "| file | kind (degrees) | vertices | cells | beta | seed vector |", "|---|---|---|---|---|---|"]
        for r in rows:
            if r["group"] == g:
                L.append(f"| {r['name']}.off | {r['kind']} ({r['degrees']}) | {r['vertices']} | {r['cells']}: {r['cell_types']} | "
                         f"{r['beta']} | {r['seed_vector']} |")
        L.append("")
    open(os.path.join(HERE, "new_shapes.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    for r in rows:
        print(f"{r['name']:60} {r['kind']:6} deg {r['degrees']}  beta {np.round(r['beta'], 6).tolist()}")


if __name__ == "__main__":
    main()
