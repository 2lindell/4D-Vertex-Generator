"""The six pennic shapes (no decaic versions): their optimized and central seeds, and kipennic-only (chiral) examples.

Pennic symmetry is [3,3,3] = S5 (order 120): vertices x[perm] of the seed x = beta W5. Its fundamental domain is the
whole chamber beta >= 0 (no swap). The six shapes are the Wythoffian ones not fixed by the swap, each on a face of the
chamber (its locus: the mirrors beta_k = 0 it lies on):

    pen (1,0,0,0) point, rap (0,1,0,0) point, tip (a,b,0,0) line, srip (a,0,b,0) line, grip (a,b,c,0) face,
    prip (a,b,0,c) face.

Each is optimized (optimize_seeds.optimise: R = longest / shortest edge, minimised) and centred (central_seeds.centre:
the deepest point of its locus) from a deliberately uneven start, so both are found, not assumed.

Kipennic symmetry is [3,3,3]+ = A5 (order 60): the even permutations. Each of the six is also kipennic (A5 already acts
transitively on its vertices). A seed inside the chamber has a 60-vertex A5 orbit with no mirror symmetry: a chiral
polychoron -- and every such seed gives the same one combinatorially (6000 points of the chamber, one arrangement key):
the kipennic snad family (5+5 snits, 10+10 gyro-traps, 60 tets; three degrees of freedom). So the six have no chiral
versions of their own; pushed off their mirrors each becomes a snad of this family that still resembles it. The
example for each: its uniform seed with the betas that were 0 raised to 0.06, 0.1, 0.14 (distinct, keeping it off the
line b1 = b4, b2 = b3 where it would gain the decaic swap and be snad itself). Not optimized: making the edges as even
as possible runs into that line. Its symmetry is counted (the elements of S5 x {+-1} keeping the vertex set): 60 means
kipennic only.

    python pennic_shapes.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SHAPES = [("pen", "point", 0, [1, 0, 0, 0]), ("rap", "point", 0, [0, 1, 0, 0]),
          ("tip", "line", 1, [0.8, 0.2, 0, 0]), ("srip", "line", 1, [0.75, 0, 0.25, 0]),
          ("grip", "wall", 2, [0.55, 0.3, 0.15, 0]), ("prip", "wall", 2, [0.55, 0.3, 0, 0.15])]
PUSH = [0.06, 0.1, 0.14]            # (the betas a chiral example raises off the mirrors: distinct, so off the snad line)
UNIFORM = {"pen": [1, 0, 0, 0], "rap": [0, 1, 0, 0], "tip": [1, 1, 0, 0], "srip": [1, 0, 1, 0], "grip": [1, 1, 1, 0],
           "prip": [1, 1, 0, 1]}


def pennic():
    """Optimized and central seeds of the six, under pennic symmetry (run with DECAIC_SUB=pen)."""
    import numpy as np
    import central_seeds as CS
    import optimize_seeds as O
    from four_d_vertex_generator.off import to_4off
    G = O.G
    CS.O, CS.G, CS.F = O, G, O.F
    shapes = [{"name": n, "kind": k, "degrees": d, "seed": s, "locus": [], "hull_tol": 1e-9} for n, k, d, s in SHAPES]
    out = []
    for s in shapes:
        r = O.optimise(s)
        CS.OPT = {s["name"]: r}
        c = CS.centre(s)
        row = {"name": s["name"], "kind": s["kind"], "degrees": s["degrees"], "start": r["start"],
               "optimized": r.get("best", r["start"]), "R": r["R"], "optimized_status": r["status"],
               "central": c["centre"], "depth": c["depth"], "central_status": c["status"]}
        row["same"] = bool(np.allclose(row["optimized"], row["central"], atol=1e-6))
        u = np.array(UNIFORM[s["name"]], float)
        row["uniform"] = (u / u.sum()).tolist()
        row["optimized_is_uniform"] = bool(np.allclose(row["optimized"], row["uniform"], atol=1e-6))
        row["central_is_uniform"] = bool(np.allclose(row["central"], row["uniform"], atol=1e-6))
        for folder, b in (("optimized_off", row["optimized"]), ("central_off", row["central"])):
            d = os.path.join(ROOT, folder, "pennic")
            os.makedirs(d, exist_ok=True)
            _, V, faces, cells = G.polytope(np.asarray(b), 1e-9)
            tag = "opt" if folder == "optimized_off" else "central"
            open(os.path.join(d, f"{s['name']}_{len(V)}v_{len(cells)}c_{tag}.off"), "w").write(to_4off(V, faces, cells))
            row[f"{tag}_counts"] = [len(V), len(cells)]
        x = G.x5(np.asarray(row["optimized"]))
        row["seed_vector"] = (x / np.abs(x).max()).tolist()
        out.append(row)
        print(f"{s['name']:5} R {r['start_R']:.4f} -> {r['R']:.6f}  optimized {np.round(row['optimized'], 6).tolist()}  "
              f"central {np.round(row['central'], 6).tolist()}  same {row['same']}  uniform {row['optimized_is_uniform']}/"
              f"{row['central_is_uniform']}  {row['opt_counts']}", flush=True)
    return out


def kipennic():
    """Chiral (kipennic-only) examples of the six (run with DECAIC_SUB=kipen)."""
    import numpy as np
    import optimize_seeds as O
    from four_d_vertex_generator.local_view import hull_edges
    from four_d_vertex_generator.off import to_4off
    from margin import margin
    G = O.G
    out = []
    d = os.path.join(ROOT, "optimized_off", "kipennic")
    os.makedirs(d, exist_ok=True)
    for name, u in UNIFORM.items():
        u = np.array(u, float)
        u = u / u.sum()
        fill = iter(PUSH)
        s = np.array([v if v > 0 else next(fill) for v in u])
        s = s / s.sum()
        V5, V, faces, cells = G.polytope(s, 1e-9)
        key = {tuple(np.round(v, 9)) for v in V5}
        sym = sum(1 for sg in (1, -1) for p in G.PERMS if all(tuple(np.round(sg * v[p], 9)) in key for v in V5))
        info = G.classify(s)
        E = [np.linalg.norm(V[a_] - V[b_]) for a_, b_ in (tuple(e) for e in hull_edges(faces))]
        fname = f"{name}_chiral_{len(V)}v_{len(cells)}c.off"
        open(os.path.join(d, fname), "w").write(to_4off(V, faces, cells))
        x = G.x5(s)
        row = {"name": f"{name}_chiral", "from": name, "seed": s.tolist(), "seed_vector": (x / np.abs(x).max()).tolist(),
               "vertices": len(V), "cells": len(cells), "symmetry_order": sym, "key": info["key"],
               "edge_ratio": float(max(E) / min(E)), "margin": float(margin(s)), "signature": G.signature(info),
               "off": os.path.relpath(os.path.join(d, fname), ROOT)}
        out.append(row)
        print(f"{name}_chiral  seed {np.round(s, 6).tolist()}  {len(V)}v {len(cells)}c  symmetry {sym}  key {info['key']}  "
              f"R {row['edge_ratio']:.3f}  margin {row['margin']:.1e}", flush=True)
    return out


def main():
    if len(sys.argv) > 1:
        res = pennic() if sys.argv[1] == "pen" else kipennic()
        json.dump(res, open(os.path.join(HERE, f"pennic_{sys.argv[1]}.json"), "w"), indent=1)
        return
    for sub in ("pen", "kipen"):
        subprocess.run([sys.executable, __file__, sub], cwd=HERE, env={**os.environ, "DECAIC_SUB": sub}, check=True)
    pen = json.load(open(os.path.join(HERE, "pennic_pen.json")))
    ki = json.load(open(os.path.join(HERE, "pennic_kipen.json")))
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "pennic": pen, "kipennic_chiral": ki},
              open(os.path.join(HERE, "pennic_shapes.json"), "w"), indent=1)
    for f in ("pennic_pen.json", "pennic_kipen.json"):
        os.remove(os.path.join(HERE, f))


if __name__ == "__main__":
    main()
