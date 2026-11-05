"""Names of the decaic shapes (the accepted list of decaic isogonals), matched to shapes.json by vertex count and cells
(each cell class by its vertex count and face sizes), with each shape's exact seeds and the planes it lies on
(decaic_shapes.json, decaic_shapes.md).

    python names.py
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

# (name, vertices, cells by our description: the accepted list's cells, combinatorially)
NAMES = {
    "bideca": (10, "30x4v(4x3)"),
    "spid": (20, "20x6v(2x3+3x4) + 10x4v(4x3)"),
    "apid": (20, "20x6v(8x3) + 10x4v(4x3)"),
    "deca": (30, "10x12v(4x3+4x6)"),
    "redeca": (60, "10x18v(16x3+4x6) + 30x4v(4x3)"),
    "respid": (60, "20x9v(8x3+3x4) + 30x6v(8x3)"),
    "bited": (40, "20x6v(8x3) + 100x4v(4x3)"),
    "biped": (40, "30x8v(6x4) + 20x6v(2x3+3x4) + 10x4v(4x3)"),
    "bimted": (60, "10x12v(4x3+4x6) + 20x12v(6x4+2x6)"),
    "gabred": (60, "10x12v(8x3+6x4) + 20x6v(8x3) + 60x5v(4x3+1x4) + 30x4v(4x3)"),
    "mabred": (60, "20x6v(2x3+3x4) + 30x6v(8x3) + 60x5v(4x3+1x4) + 30x4v(4x3)"),
    "sabred": (60, "70x6v(8x3) + 30x4v(4x3)"),
    "tabipard": (120, "10x24v(4x3+12x4+4x6) + 30x8v(6x4) + 20x6v(2x3+3x4)"),
    "gippid": (120, "10x24v(6x4+8x6) + 20x12v(6x4+2x6)"),
    "tadeca": (120, "10x36v(12x3+4x6+4x12) + 30x4v(4x3)"),
    "tabcated": (120, "20x12v(2x3+9x4) + 10x12v(4x3+4x6) + 20x12v(6x4+2x6)"),
    "tispid": (120, "20x18v(6x3+2x6+3x8) + 10x12v(4x3+4x6) + 20x6v(8x3)"),
    "gabipard": (120, "10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 30x8v(6x4) + 60x6v(2x3+3x4) + 20x6v(8x3)"),
    "mobcated": (120, "10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 40x9v(4x3+3x4+1x6) + 30x4v(4x3)"),
    "sobcated": (120, "10x12v(4x3+4x6) + 20x12v(6x4+2x6) + 80x6v(2x3+3x4) + 30x4v(4x3)"),
    "mabipard": (120, "10x12v(4x3+4x6) + 40x9v(4x3+3x4+1x6) + 30x8v(6x4) + 20x6v(2x3+3x4) + 20x6v(8x3)"),
    "sabipard": (120, "10x12v(8x3+6x4) + 30x8v(6x4) + 80x6v(2x3+3x4) + 20x6v(8x3)"),
    "gobcated": (120, "10x24v(6x4+8x6) + 20x12v(6x4+2x6) + 60x6v(2x3+3x4) + 30x4v(4x3)"),
    "petbotid": (240, "10x24v(6x4+8x6) + 20x24v(9x4+2x6+3x8) + 20x12v(6x4+2x6)"),
    "tetbotid": (240, "10x48v(18x4+4x6+4x12) + 20x12v(6x4+2x6) + 30x8v(6x4)"),
    "sabotid": (240, "10x24v(6x4+8x6) + 40x12v(6x4+2x6) + 90x8v(6x4)"),
    "gabotid": (240, "10x24v(6x4+8x6) + 80x12v(6x4+2x6) + 30x8v(6x4)"),
}
EXPECTED = {"sabotid": 3, "gabotid": 3, "tetbotid": 2, "petbotid": 2, "sabipard": 2, "mabipard": 2, "gabipard": 2,
            "sobcated": 2, "mobcated": 2, "gobcated": 2, "gippid": 1, "bited": 1, "biped": 1, "bimted": 1, "mabred": 1,
            "sabred": 1, "gabred": 1, "tispid": 1, "tabipard": 1, "tadeca": 1, "tabcated": 1, "deca": 0, "spid": 0,
            "bideca": 0, "apid": 0, "respid": 0, "redeca": 0}
VARS = ["b1", "b2", "b3", "b4"]


def plane_text(c):
    pos = " + ".join((f"{a}*" if a != 1 else "") + v for a, v in zip(c, VARS) if a > 0)
    neg = " + ".join((f"{-a}*" if a != -1 else "") + v for a, v in zip(c, VARS) if a < 0)
    return f"{pos} = {neg}" if neg else f"{pos} = 0"


def integer_seed(point):
    q = [Fraction(v) for v in point]
    d = np.lcm.reduce([x.denominator for x in q])
    v = [int(x * d) for x in q]
    g = np.gcd.reduce([abs(x) for x in v if x])
    return [int(x // g) for x in v]


def main():
    S = json.load(open("shapes.json"))["shapes"]
    A = json.load(open("arrangement.json"))
    planes = [tuple(b) for b in A["boundary"]] + [tuple(p) for p in A["planes"] if list(p) not in A["boundary"]]
    faces = A["faces"]
    out = []
    used = set()
    for s in S:
        name = next((n for n, (v, c) in NAMES.items() if v == s["vertices"] and c == s["cells_text"]), None)
        used.add(name)
        # (the planes the shape's top faces lie on: every face of the shape of top dimension)
        on = set()
        for f in faces:
            if f["dim"] == s["degrees"] and f.get("signs") and s["example_signs"] and f["signs"] == s["example_signs"]:
                on |= {planes[i] for i, v in enumerate(f["signs"]) if v == 0}
        out.append({"name": name or "UNMATCHED", "degrees": s["degrees"], "expected_degrees": EXPECTED.get(name),
                    "vertices": s["vertices"], "cells": s["cells_text"], "n_cells": s["n_cells"],
                    "faces_by_dim": s["faces_by_dim"], "seed_beta": integer_seed(s["example"]),
                    "seed_beta_exact": s["example"], "lies_on": [plane_text(c) for c in sorted(on)]})
    missing = sorted(set(NAMES) - used)
    ok = all(o["degrees"] == o["expected_degrees"] for o in out) and not missing and len(out) == len(NAMES)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "all_match": ok, "missing": missing, "shapes": out},
              open("decaic_shapes.json", "w"), indent=1)
    kind = {3: "region", 2: "wall", 1: "line", 0: "point"}
    L = ["# Decaic shapes ([[3,3,3]], order 240)", "",
         f"{len(out)} shapes found; accepted list: {len(NAMES)}; all names and degrees match: **{ok}**.", "",
         "Seeds are beta = (b1, b2, b3, b4) in the fundamental-weight basis (beta_k = x_k - x_{k+1}); domain beta >= 0, "
         "b1 >= b4.", "",
         "| name | kind (degrees) | vertices | cells | seed beta | example face lies on |", "|---|---|---|---|---|---|"]
    for o in out:
        L.append(f"| {o['name']} | {kind[o['degrees']]} ({o['degrees']}) | {o['vertices']} | {o['n_cells']}: {o['cells']} "
                 f"| ({', '.join(map(str, o['seed_beta']))}) | {'; '.join(o['lies_on']) or '(open region)'} |")
    open("decaic_shapes.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("missing:", missing)


if __name__ == "__main__":
    main()
