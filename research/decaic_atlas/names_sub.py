"""Names of the kidecaic and iodecaic shapes (the accepted list of pennic and decaic isogonals), matched to
shapes_ki.json / shapes_io.json (decaic_ki_shapes.md / decaic_io_shapes.md).

Shapes the subgroup shares with decaic are matched by the decaic profiles (names.NAMES). The rest by kind, vertex count
and number of cells (the list's cells added up: e.g. sabosid 10 snits + 40 gyro-traps + 390 tets = 440). Mebosid and
gabosid have the same counts ("almost identical"); they are told apart by the list's place: mebosid closer to gabred,
gabosid closer to mabred.

    DECAIC_SUB=ki python names_sub.py ; DECAIC_SUB=io python names_sub.py
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

import group as G
from names import NAMES, integer_seed, plane_text

DEC = {"deca", "spid", "bideca", "apid", "bited", "biped", "bimted", "respid", "redeca", "mabred", "sabred", "gabred",
       "sabipard", "mabipard", "gabipard", "tispid", "tabipard", "sobcated", "mobcated", "gobcated", "tadeca", "tabcated"}
DEG = {"deca": 0, "spid": 0, "bideca": 0, "apid": 0, "respid": 0, "redeca": 0, "gippid": 1, "bited": 1, "biped": 1,
       "bimted": 1, "mabred": 1, "sabred": 1, "gabred": 1, "tispid": 1, "tabipard": 1, "tadeca": 1, "tabcated": 1,
       "sabipard": 2, "mabipard": 2, "gabipard": 2, "sobcated": 2, "mobcated": 2, "gobcated": 2}
# (name: (degrees, vertices, cells))
OWN = {"ki": {"snad": (1, 60, 90), "sabosid": (3, 120, 440), "mebosid": (3, 120, 560), "apbosid": (3, 120, 420),
              "gabosid": (3, 120, 560), "sigwebosid": (2, 120, 290), "gigwebosid": (2, 120, 320),
              "tetbosid": (2, 120, 240), "petbosid": (2, 120, 230)},
       "io": {"gippid": (1, 120, 30), "sosbid": (3, 120, 380), "gosbid": (3, 120, 360), "tetosbid": (2, 120, 180),
              "petosbid": (2, 120, 170)}}


def main():
    sub = G.SUB
    S = json.load(open(G.fname("shapes")))["shapes"]
    A = json.load(open(G.fname("arrangement")))
    planes = [tuple(b) for b in A["boundary"]] + [tuple(p) for p in A["planes"] if list(p) not in A["boundary"]]
    gab = np.array([1, 0, 7, 0], float); gab /= gab.sum()          # (gabred, mabred seeds: decaic_shapes)
    mab = np.array([37, 0, 11, 0], float); mab /= mab.sum()
    expected = {n: (DEG[n], *NAMES[n]) for n in DEC} | ({"gippid": (1, *NAMES["gippid"])} if sub == "io" else {})
    out = []
    for s in S:
        names = [n for n in DEC if NAMES[n] == (s["vertices"], s["cells_text"])]
        if not names:
            names = [n for n, (d, v, c) in OWN[sub].items() if v == s["vertices"] and c == s["n_cells"] and d == s["degrees"]
                     and n not in DEC]
            if sub == "io" and "gippid" in names and NAMES["gippid"] != (s["vertices"], s["cells_text"]):
                names.remove("gippid")
        out.append({"names": names, "s": s})
    # (mebosid / gabosid: the one nearer gabred is mebosid)
    pair = [o for o in out if set(o["names"]) == {"mebosid", "gabosid"}]
    if len(pair) == 2:
        d = [min(np.linalg.norm(np.array(o["s"]["example_float"]) - gab), 1e9) - np.linalg.norm(np.array(o["s"]["example_float"]) - mab)
             for o in pair]
        i = int(np.argmin(d))
        pair[i]["names"], pair[1 - i]["names"] = ["mebosid"], ["gabosid"]
    rows, used = [], set()
    for o in out:
        s = o["s"]
        name = o["names"][0] if len(o["names"]) == 1 else ("UNMATCHED" if not o["names"] else "/".join(o["names"]))
        used.add(name)
        want = OWN[sub].get(name, (DEG.get(name),))[0]
        on = [plane_text(planes[i]) for i, v in enumerate(s["example_signs"]) if v == 0]
        rows.append({"name": name, "kind": s["kind"], "degrees": s["degrees"], "expected_degrees": want,
                     "vertices": s["vertices"], "n_cells": s["n_cells"], "cells": s["cells_text"], "pieces": s["pieces"],
                     "seed_beta": integer_seed(s["example"]), "example_face_lies_on": on})
    listed = set(DEC) | set(OWN[sub])
    missing = sorted(listed - used)
    extra = [r for r in rows if r["name"] not in listed]
    ok = not missing and not extra and all(r["degrees"] == r["expected_degrees"] for r in rows)
    title = {"ki": "Kidecaic ([[3,3,3]]+, order 120)", "io": "Iodecaic ([[3,3,3]+], order 120)"}[sub]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "all_match": ok, "missing": missing,
               "not_on_list": [r["name"] + " " + r["cells"] for r in extra], "shapes": rows},
              open(G.fname("decaic_shapes"), "w"), indent=1)
    kind = {3: "region", 2: "wall", 1: "line", 0: "point"}
    L = [f"# {title} shapes", "",
         f"{len(rows)} shapes found; on the list: {len(listed)}; missing: {missing or 'none'}; not on the list: "
         f"{len(extra)}; all names and degrees match: **{ok}**.", "",
         "| name | kind (degrees) | vertices | cells | pieces | seed beta | example face lies on |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["degrees"], r["vertices"], r["name"])):
        L.append(f"| {r['name']} | {kind[r['degrees']]} ({r['degrees']}) | {r['vertices']} | {r['n_cells']}: {r['cells']} | "
                 f"{r['pieces']} | ({', '.join(map(str, r['seed_beta']))}) | {'; '.join(r['example_face_lies_on']) or '(open region)'} |")
    open(G.fname("decaic_shapes", "md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
