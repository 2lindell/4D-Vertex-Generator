"""The contic shapes, each with its decaic counterpart (the decaic shape in the same place of the domain, with the same
kinds of cells: tet <-> tet, tut <-> tic, hip <-> hip or op, oct/trap <-> square antiprism, ...), exact seeds and the
planes its faces lie on (contic_shapes.json, contic_shapes.md). Not an accepted list: contic names are not confirmed.

    python names.py
"""
from __future__ import annotations

import json
from fractions import Fraction

from qsqrt2 import K

# (decaic counterpart: (kind, vertices, n_cells, a cell profile the shape must have))
COUNTERPART = [
    ("sabotid", "region", 2304, 1248, None), ("gabotid", "region", 2304, 1056, None),
    ("petbotid", "wall", 2304, 384, None), ("tetbotid", "wall", 2304, 528, None),
    ("gabipard", "wall", 1152, 1248, "24v(6x4+8x6)"), ("sobcated", "wall", 1152, 1248, "24v(8x3+6x8)"),
    ("sabipard", "wall", 1152, 1248, "24v(8x3+18x4)"), ("mobcated", "wall", 1152, 864, None),
    ("mabipard", "wall", 1152, 1056, "9v(4x3+3x4+1x6)"), ("gobcated", "wall", 1152, 1056, "48v(12x4+8x6+6x8)"),
    ("biped", "line", 288, 528, None), ("bited", "line", 384, 1056, None), ("mabred", "line", 576, 1248, None),
    ("sabred", "line", 576, 864, None), ("gabred", "line", 576, 1056, None), ("bimted", "line", 576, 192, None),
    ("tispid", "line", 1152, 384, "18v(6x3+2x6+3x8)"), ("tabcated", "line", 1152, 384, "12v(2x3+9x4)"),
    ("gippid", "line", 1152, 240, None), ("tabipard", "line", 1152, 528, None), ("tadeca", "line", 1152, 336, None),
    ("bideca", "point", 48, 288, None), ("spid", "point", 144, 240, None), ("apid", "point", 192, 192, None),
    ("deca", "point", 288, 48, None), ("respid", "point", 576, 384, None), ("redeca", "point", 576, 336, None),
]
VARS = ["b1", "b2", "b3", "b4"]


def kparse(v):
    return K(Fraction(v[0]), Fraction(v[1]))


def plane_text(c):
    pos = " + ".join(f"{x}*{v}" if x != K(1) else v for x, v in zip(c, VARS) if x.sign() > 0)
    neg = " + ".join(f"{-x}*{v}" if x != K(-1) else v for x, v in zip(c, VARS) if x.sign() < 0)
    return f"{pos} = {neg}" if neg else f"{pos} = 0"


def main():
    S = json.load(open("shapes.json"))["shapes"]
    A = json.load(open("arrangement.json"))
    planes = [tuple(kparse(v) for v in p) for p in A["boundary"] + A["planes"]]
    out, used = [], set()
    for s in S:
        m = [n for n, kind, v, c, prof in COUNTERPART if kind == s["kind"] and v == s["vertices"] and c == s["n_cells"]
             and (prof is None or prof in s["cells_text"])]
        name = m[0] if len(m) == 1 else "UNMATCHED"
        used.add(name)
        on = [plane_text(planes[i]) for i, v in enumerate(s["example_signs"]) if v == 0]
        out.append({"decaic_counterpart": name, "kind": s["kind"], "degrees": s["degrees"], "vertices": s["vertices"],
                    "n_cells": s["n_cells"], "cells": s["cells_text"], "faces_by_dim": s["faces_by_dim"],
                    "seed_beta": [repr(kparse(v)) for v in s["example"]], "seed_float": s["example_float"],
                    "example_face_lies_on": on})
    missing = sorted({c[0] for c in COUNTERPART} - used)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "missing": missing, "shapes": out},
              open("contic_shapes.json", "w"), indent=1)
    L = ["# Contic shapes ([[3,4,3]], order 2304)", "",
         f"{len(out)} shapes; each paired with its decaic counterpart (missing: {missing or 'none'}).", "",
         "Seeds: beta = distances to the four mirrors of the F4 chamber (b1, b2 long-root mirrors, b3, b4 short); "
         "domain beta >= 0, b1 >= b4. Exact values a + b*sqrt2; the float shown is normalized to sum 1.", "",
         "| decaic counterpart | kind | vertices | cells | seed beta (float) | example face lies on |", "|---|---|---|---|---|---|"]
    for o in out:
        L.append(f"| {o['decaic_counterpart']} | {o['kind']} ({o['degrees']}) | {o['vertices']} | {o['n_cells']}: {o['cells']} "
                 f"| {o['seed_float']} | {'; '.join(o['example_face_lies_on']) or '(open region)'} |")
    open("contic_shapes.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
