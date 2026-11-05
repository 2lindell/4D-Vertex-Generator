"""The central seed of every contic, kicontic and iocontic shape (central_seeds<tag>.json, central .off files).

decaic_atlas/central_seeds.py run on the contic groups, with this atlas's shapes, loci and sign conditions (as patched
for Q(sqrt2) by optimize_seeds.py): the point of each shape's locus deepest inside it, kept only if it rebuilds as the
same polytope with the same vertex and cell counts.

    CONTIC_SUB=full|ki|io python central_seeds.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimize_seeds as C  # noqa: E402  (contic group, forms and the patched optimiser)

ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("decaic_central", os.path.join(ROOT, "decaic_atlas", "central_seeds.py"))
CS = importlib.util.module_from_spec(spec)
sys.modules["decaic_central"] = CS
spec.loader.exec_module(CS)


def main():
    G = C.G
    CS.O, CS.G, CS.F = C.O, G, C.F
    try:
        CS.OPT = {r["name"]: r for r in json.load(open(os.path.join(HERE, G.fname("optimized_seeds"))))["shapes"]}
    except FileNotFoundError:
        CS.OPT = {}
    shapes = C.contic_shapes() if G.SUB == "full" else C.snub_shapes()
    out = os.path.join(ROOT, "central_off", {"full": "contic", "ki": "kicontic", "io": "iocontic"}[G.SUB])

    def hull(b, tol):
        V, Vn, idx, faces, cells = G.polytope(b, tol)
        return Vn, faces, cells
    CS.run(shapes, out, G.x4, hull, os.path.join(HERE, G.fname("central_seeds")))


if __name__ == "__main__":
    main()
