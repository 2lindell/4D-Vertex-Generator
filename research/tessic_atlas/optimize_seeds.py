"""The most even seed of every tessic-family shape (optimized_seeds<tag>.json, optimized .off files).

The decaic optimiser (decaic_atlas/optimize_seeds.py) run on the tessic groups (as contic_atlas/optimize_seeds.py): the same objective (R = longest edge /
shortest edge, minimised on the shape's locus while its sign conditions hold, with multi-starts, a polish of the active
equations, a ring check, and a seed just inside where the infimum is on the shape's edge). What differs:

  * an edge from x to g x (g in the group, x = X beta) has squared length beta^T M^T M beta, M = X - g X, with
    entries in Q(sqrt2) (used in floating point: the polish then solves the active equations to double precision,
    not 40 digits);
  * the sign conditions are this atlas's forms (forms.region_forms, over Q(sqrt2), restricted to the seed's mirrors),
    written on the quartic monomials;
  * staying in the shape is judged by the arrangement key alone (the incidence comparison is too slow at 1152-2304
    vertices).

    TESSIC_SUB=full|ki|pyr|demi|kidemi python optimize_seeds.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import forms as F  # noqa: E402  (this atlas's forms and group)
import group as G  # noqa: E402

sys.path.insert(1, os.path.join(ROOT, "decaic_atlas"))
_spec = importlib.util.spec_from_file_location("decaic_optimize", os.path.join(ROOT, "decaic_atlas", "optimize_seeds.py"))
O = importlib.util.module_from_spec(_spec)
sys.modules["decaic_optimize"] = O         # (so pool workers find its functions)
G.W5 = np.zeros((4, 5))                     # (read at its import; not used here)
_spec.loader.exec_module(O)                 # (its "group" and "forms" are this atlas's: already imported above)
assert O.G is G and O.F is F

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402
from four_d_vertex_generator.off import to_4off  # noqa: E402

B = O.B
F.MONS = O.F.MONS if hasattr(O.F, "MONS") else [m for m in __import__("itertools").product(range(5), repeat=4) if sum(m) == 4]
F.poly = lambda e: e                        # (a condition is kept as its sympy expression)
OFF_DIR = os.path.join(ROOT, "optimized_off", {"full": "tessic", "ki": "kitessic", "pyr": "tessipyritic", "demi": "demitessic", "kidemi": "kidemitessic"}[G.SUB])
O.OFF_DIR = OFF_DIR


def edge_forms(seed, hull_tol):
    V, Vn, idx, faces, cells = G.polytope(seed, hull_tol)
    assert np.abs(V[0] - G.x4(seed)).max() < 1e-9
    Qs = {}
    for e in hull_edges(faces):
        e = tuple(e)
        if 0 not in e:
            continue
        w = e[1] if e[0] == 0 else e[0]
        M = G.X - G.ELEMENTS[idx[w]] @ G.X
        Q = M.T @ M
        Qs[tuple(np.round(Q, 9).ravel())] = Q
    return list(Qs.values())


def sign_conditions(seed):
    out = []
    for fx in F.region_forms(seed):
        p = sp.Poly(sp.expand(fx), *B)
        co = np.array([float(p.coeff_monomial(sp.prod([v ** k for v, k in zip(B, m)]))) for m in F.MONS])
        scale = np.abs(co).sum()
        if scale == 0:
            continue
        v = float(fx.subs(dict(zip(B, seed))))
        if abs(v) > 1e-9 * scale:
            out.append((co / scale, int(np.sign(v)), sp.expand(fx)))
    return out


def same_shape(b, seed, key0, hull_tol):
    try:
        return G.label(b, hull_tol)[1] == key0
    except Exception:
        return False


O.edge_forms, O.sign_conditions, O.same_shape = edge_forms, sign_conditions, same_shape


GROUP = {"full": "tessic", "ki": "kitessic", "pyr": "tessipyritic", "demi": "demitessic", "kidemi": "kidemitessic"}
# (the demitessic map's seven polytopes, by vertex and cell count)
DEMI_NAMES = {(192, 48): "tico", (96, 48): "rico", (96, 24): "tah", (32, 24): "rit", (48, 24): "thex", (8, 16): "hex",
              (24, 24): "ico"}


def tessic_shapes():
    """The shapes of this group: tessic, the 15 Wythoffian shapes (tessic_shapes.json); demitessic, its 7 polytopes, each
    with every face of the chamber it fills (its pieces); kitessic, tessipyritic and kidemitessic, their own shapes --
    the one polytope filling the chamber, and the face b4 = 0 where tessipyritic's orbits are halved (sadi). Shapes on
    the other faces are the tessic (or demitessic) ones: their files are in that group's folder."""
    out = []
    if G.SUB == "full":
        for r in json.load(open(os.path.join(HERE, "tessic_shapes.json")))["shapes"]:
            out.append({"name": r["name"], "kind": {3: "region", 2: "wall", 1: "line", 0: "point"}[r["degrees"]],
                        "degrees": r["degrees"], "seed": r["seed"], "locus": [], "hull_tol": 1e-9})
        return out
    if G.SUB == "demi":
        S = json.load(open(os.path.join(HERE, G.fname("shapes"))))["shapes"]
        A = json.load(open(os.path.join(HERE, G.fname("arrangement"))))
        for s in S:
            n = DEMI_NAMES[(s["vertices"], s["n_cells"])]
            out.append({"name": n, "kind": s["kind"], "degrees": s["degrees"], "seed": s["example_float"], "locus": [],
                        "hull_tol": 1e-9})
        O.add_pieces(out, [(f["dim"], f["point_float"], [f"b{i + 1}" for i, v in enumerate(f["point_float"]) if abs(v) < 1e-12])
                           for f in A["faces"]],
                     {k: DEMI_NAMES[(s["vertices"], s["n_cells"])] for s in S for k in s["keys"]})
        return out
    W = json.load(open(os.path.join(HERE, G.fname("wall_shapes"))))["shapes"]
    interior = {"ki": "snet", "pyr": "pysnet", "kidemi": "sadi"}[G.SUB]
    for t in W:
        if t["kind"] == "region":
            out.append({"name": interior, "kind": "region", "degrees": 3, "seed": t["seed_beta"], "locus": [],
                        "hull_tol": 1e-9})
    if G.SUB == "pyr":                           # (the face b4 = 0: sadi, from the face survey)
        f = json.load(open(os.path.join(HERE, G.fname("face_survey_b4"))))["types"][0]
        out.append({"name": "sadi", "kind": "wall", "degrees": 2, "seed": f["example"], "locus": [], "hull_tol": 1e-9})
    return out


def main():
    os.makedirs(OFF_DIR, exist_ok=True)
    shapes = tessic_shapes()
    with Pool(4) as pool:
        R = pool.map(O.optimise_best, shapes, chunksize=1)
    for s, r in zip(shapes, R):
        if "exact_digits" in r:
            r["exact_digits"] = "double precision (the edge forms are used in floating point)"
        b = np.asarray(r["off_seed"], float)
        V, Vn, idx, faces, cells = G.polytope(b, s["hull_tol"])
        name = f"{r['name']}_{len(V)}v_{len(cells)}c_opt"
        open(os.path.join(OFF_DIR, name + ".off"), "w").write(to_4off(Vn, faces, cells))
        r["off"] = os.path.relpath(os.path.join(OFF_DIR, name + ".off"), ROOT)
        r["vertices"], r["cells"] = len(V), len(cells)
        x = G.x4(b)
        r["seed_vector"] = (x / np.abs(x).max()).tolist()
        print(f"{r['name']:34} {r['kind']:6} R {r['start_R']:.6f} -> {r['R']:.9f}  {r['status'][:40]:40} "
              f"beta {np.round(r['best'], 9).tolist()}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": R}, open(os.path.join(HERE, G.fname("optimized_seeds")), "w"), indent=1)


if __name__ == "__main__":
    main()
