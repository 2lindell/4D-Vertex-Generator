"""The most even seed of every contic, kicontic and iocontic shape (optimized_seeds<tag>.json, optimized .off files).

The decaic optimiser (decaic_atlas/optimize_seeds.py) run on the contic groups: the same objective (R = longest edge /
shortest edge, minimised on the shape's locus while its sign conditions hold, with multi-starts, a polish of the active
equations, a ring check, and a seed just inside where the infimum is on the shape's edge). What differs:

  * an edge from x to g x (g in the contic group, x = X beta) has squared length beta^T M^T M beta, M = X - g X, with
    entries in Q(sqrt2) (used in floating point: the polish then solves the active equations to double precision,
    not 40 digits);
  * the sign conditions are the contic forms (forms.region_forms, over Q(sqrt2), restricted to the seed's mirrors),
    written on the quartic monomials;
  * staying in the shape is judged by the arrangement key alone (the incidence comparison is too slow at 1152-2304
    vertices).

    CONTIC_SUB=full|ki|io python optimize_seeds.py
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
import forms as F  # noqa: E402  (the contic forms and group)
import given_names  # noqa: E402  (the accepted names)
import group as G  # noqa: E402

sys.path.insert(1, os.path.join(ROOT, "decaic_atlas"))
_spec = importlib.util.spec_from_file_location("decaic_optimize", os.path.join(ROOT, "decaic_atlas", "optimize_seeds.py"))
O = importlib.util.module_from_spec(_spec)
sys.modules["decaic_optimize"] = O         # (so pool workers find its functions)
G.W5 = np.zeros((4, 5))                     # (read at its import; not used for contic)
_spec.loader.exec_module(O)                 # (its "group" and "forms" are this atlas's: already imported above)
assert O.G is G and O.F is F

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402
from four_d_vertex_generator.off import to_4off  # noqa: E402

B = O.B
F.MONS = O.F.MONS if hasattr(O.F, "MONS") else [m for m in __import__("itertools").product(range(5), repeat=4) if sum(m) == 4]
F.poly = lambda e: e                        # (a condition is kept as its sympy expression)
OFF_DIR = os.path.join(ROOT, "optimized_off", {"full": "contic", "ki": "kicontic", "io": "iocontic", "ico": "icoic",
                                                "tox": "toxitic", "ox": "oxitic"}.get(G.SUB, G.SUB))
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


def ktext(c):
    return " + ".join(f"({Fraction(a)} + {Fraction(b)}*sqrt(2))*{v}" for (a, b), v in zip(c, ["b1", "b2", "b3", "b4"]))


def contic_shapes():
    """The 27 contic shapes: the exact seed (a + b sqrt2) and the planes of the arrangement its face lies on."""
    S = json.load(open(os.path.join(HERE, "shapes.json")))["shapes"]
    N = json.load(open(os.path.join(HERE, "contic_shapes.json")))["shapes"]
    A = json.load(open(os.path.join(HERE, "arrangement.json")))
    planes = A["boundary"] + A["planes"]
    out = []
    for s, n in zip(S, N):
        seed = [float(Fraction(a)) + float(Fraction(b)) * 2 ** 0.5 for a, b in s["example"]]
        zero = [ktext(planes[i]) for i, v in enumerate(s["example_signs"]) if v == 0]
        out.append({"name": given_names.contic(n["decaic_counterpart"]), "kind": s["kind"], "degrees": s["degrees"], "seed": seed,
                    "locus": O.independent(zero, seed), "hull_tol": 1e-9})
    O.add_pieces(out, [(f["dim"], f["point_float"], [ktext(planes[i]) for i, v in enumerate(f["signs"]) if v == 0])
                       for f in A["faces"]],
                 {k: given_names.contic(n["decaic_counterpart"]) for s, n in zip(S, N) for k in s["keys"]})
    return out


def snub_shapes():
    """The kicontic / iocontic shapes of their own: the map's regions and walls, the edge survey's new regions, and
    the line b1 = b4, b2 = b3 (snad-like / gippid-like)."""
    W = json.load(open(os.path.join(HERE, G.fname("wall_shapes"))))["shapes"]
    out, k = [], {"region": 0, "wall": 0}
    seen = set()
    for t in W:
        kind = "region" if t["kind"] == "region" else "wall"
        k[kind] += 1
        if t["kind"] == "wall (fold)":
            locus = ["b1 - b4"]
        elif kind == "wall":
            locus = [max(t["surfaces"], key=t["surfaces"].get)]
        else:
            locus = []
            seen.add(G.label(np.asarray(t["seed_beta"]) / np.sum(t["seed_beta"]))[1])
        out.append({"name": f"{kind}_{k[kind]:02d}_{t['vertices']}v_{t['cells']}c", "kind": kind,
                    "degrees": 3 if kind == "region" else 2, "seed": t["seed_beta"], "locus": locus,
                    "hull_tol": 1e-9 if kind == "region" else 1e-7})
    try:
        zones = json.load(open(os.path.join(HERE, G.fname("edge_survey"))))["zones"]
    except FileNotFoundError:
        zones = {}
    for z in zones.values():
        for t in z["types"]:
            if t["known_region"] is None:
                kk = G.label(t["example"])[1]
                if kk in seen:
                    continue
                seen.add(kk)
                sg = t["signature"]
                k["region"] += 1
                nc = sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))
                out.append({"name": f"region_{k['region']:02d}_{sg.split(':')[0]}v_{nc}c", "kind": "region",
                            "degrees": 3, "seed": t["example"], "locus": [], "hull_tol": 1e-9})
    out.append({"name": "snad-like" if G.SUB == "ki" else "gippid-like", "kind": "line", "degrees": 1,
                "seed": (np.array([1, 11, 11, 1]) / 24).tolist(), "locus": ["b1 - b4", "b2 - b3"], "hull_tol": 1e-9})
    for s in out:
        s["atlas_name"], s["name"] = s["name"], given_names.snub(G.SUB, s["name"])
    return out


def main():
    os.makedirs(OFF_DIR, exist_ok=True)
    shapes = contic_shapes() if G.SUB == "full" else snub_shapes()
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
