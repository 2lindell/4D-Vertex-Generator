"""The tessic shapes named by the accepted list (tessic_shapes.md). Under [4,3,3] the chamber has no walls inside: each
shape is one face of the chamber, named by which betas are nonzero (beta_1: the hexadecachoron's node, beta_4: the
tesseract's).

    TESSIC_SUB=full python names.py
"""
import json

import numpy as np

import group as G

BY_SUPPORT = {(1, 0, 0, 0): "hex", (0, 1, 0, 0): "ico", (0, 0, 1, 0): "rit", (0, 0, 0, 1): "tes",
              (1, 1, 0, 0): "thex", (1, 0, 1, 0): "rico", (1, 0, 0, 1): "sidpith", (0, 1, 1, 0): "tah",
              (0, 1, 0, 1): "srit", (0, 0, 1, 1): "tat", (1, 1, 1, 0): "tico", (1, 1, 0, 1): "proh",
              (1, 0, 1, 1): "prit", (0, 1, 1, 1): "grit", (1, 1, 1, 1): "gidpith"}
LIST_DEGREES = {"tes": 0, "tat": 1, "srit": 1, "grit": 2, "sidpith": 1, "prit": 2, "proh": 2, "gidpith": 3, "hex": 0,
                "thex": 1, "rit": 0, "tah": 1, "ico": 0, "rico": 1, "tico": 2}


def main():
    S = json.load(open(G.fname("shapes")))["shapes"]
    rows = []
    for s in S:
        sup = tuple(int(v > 1e-12) for v in s["example_float"])
        n = BY_SUPPORT[sup]
        rows.append({"name": n, "degrees": s["degrees"], "list_degrees": LIST_DEGREES[n], "vertices": s["vertices"],
                     "cells": s["cells_text"], "n_cells": s["n_cells"], "seed": s["example_float"]})
    ok = all(r["degrees"] == r["list_degrees"] for r in rows) and len({r["name"] for r in rows}) == 15
    json.dump({"all_match": ok, "shapes": rows}, open("tessic_shapes.json", "w"), indent=1)
    kind = {3: "region", 2: "face (wall)", 1: "line", 0: "point"}
    L = ["# Tessic shapes ([4,3,3], order 384)", "",
         f"15 shapes, one on each face of the chamber (the arrangement has no walls inside it); names and degrees of "
         f"freedom all match the list: **{ok}**.", "",
         "Seeds: beta = distances to the four mirrors (b1: e1 - e2, b2: e2 - e3, b3: e3 - e4 long; b4: e4 short); "
         "(1,0,0,0) is the hexadecachoron, (0,0,0,1) the tesseract.", "",
         "| name | kind (degrees) | vertices | cells | seed beta |", "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (-r["degrees"], r["vertices"])):
        L.append(f"| {r['name']} | {kind[r['degrees']]} ({r['degrees']}) | {r['vertices']} | {r['n_cells']}: {r['cells']} | {r['seed']} |")
    open("tessic_shapes.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
