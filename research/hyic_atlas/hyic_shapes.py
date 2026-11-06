"""The fifteen hyic shapes ([5,3,3], order 14400): seed, optimized and central .off files (hyic_shapes.json).

Hyic is a reflection group with no diagram symmetry, so the whole chamber is the domain and the polychoron is the same
(Wythoff) on each open face: one shape per face, 4 points, 6 lines, 4 walls and the region (the 120-cell family).
Names: the accepted list's (hi = 120-cell, ex = 600-cell, ...; its prahi has srids, tuts, pips and ditrips, its prix tids,
ratets, dipips and trips, and so they are placed).

  seed       a generic, uneven point of the face (betas 0.6 : 0.4, 0.5 : 0.3 : 0.2, 0.4 : 0.3 : 0.2 : 0.1);
  optimized  R = longest / shortest edge minimised: an edge joins x to its image in a mirror it is off, length 2 beta_k,
             so every edge is equal exactly when the face's betas are equal -- the uniform polychoron (R = 1, checked);
  central    the face's deepest point (least distance to the mirrors bounding it, with the betas summing to 1): with
             no fold and no other walls this is also where the betas are equal, the uniform seed (as for pennic).

    python hyic_shapes.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import group as G  # noqa: E402

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402
from four_d_vertex_generator.off import to_4off  # noqa: E402

SHAPES = [("hi", (1, 0, 0, 0)), ("rahi", (0, 1, 0, 0)), ("rox", (0, 0, 1, 0)), ("ex", (0, 0, 0, 1)),
          ("thi", (1, 1, 0, 0)), ("srahi", (1, 0, 1, 0)), ("sidpixhi", (1, 0, 0, 1)), ("xhi", (0, 1, 1, 0)),
          ("srix", (0, 1, 0, 1)), ("tex", (0, 0, 1, 1)),
          ("grahi", (1, 1, 1, 0)), ("prix", (1, 1, 0, 1)), ("prahi", (1, 0, 1, 1)), ("grix", (0, 1, 1, 1)),
          ("gidpixhi", (1, 1, 1, 1))]
UNEVEN = {1: [1.0], 2: [0.6, 0.4], 3: [0.5, 0.3, 0.2], 4: [0.4, 0.3, 0.2, 0.1]}
KIND = {1: "point", 2: "line", 3: "wall", 4: "region"}


def ratio(Vn, faces):
    E = [np.linalg.norm(Vn[a] - Vn[b]) for a, b in (tuple(e) for e in hull_edges(faces))]
    return float(max(E) / min(E))


def cell_text(faces, cells):
    from collections import Counter
    c = Counter()
    for cl in cells:
        vs = {v for f in cl for v in faces[f]}
        c[(len(vs), len(cl))] += 1
    return " + ".join(f"{n}x[{v}v{f}f]" for (v, f), n in sorted(c.items(), key=lambda t: -t[1]))


def write(folder, fname, Vn, faces, cells):
    d = os.path.join(ROOT, folder, "hyic")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, fname)
    open(p, "w").write(to_4off(Vn, faces, cells))
    return os.path.relpath(p, ROOT)


def main():
    out = []
    for name, ring in SHAPES:
        on = [i for i in range(4) if ring[i]]
        m = len(on)
        seed = np.zeros(4)
        seed[on] = UNEVEN[m]
        uni = np.array(ring, float) / m
        row = {"name": name, "kind": KIND[m], "degrees": m - 1, "ringed": list(ring), "seed": seed.tolist(),
               "optimized": uni.tolist(), "central": uni.tolist()}
        V, Vn, idx, faces, cells = G.polytope(seed)
        row["seed_R"] = ratio(Vn, faces)
        row["seed_off"] = write(os.path.join("new_shapes", "off"), f"hyic_{KIND[m]}_{name}_{len(V)}v_{len(cells)}c_seed.off",
                                Vn, faces, cells)
        V2, Vn2, idx2, faces2, cells2 = G.polytope(uni)
        if (len(V2), len(cells2)) != (len(V), len(cells)):
            raise RuntimeError(f"{name}: the uniform seed rebuilds differently")
        row["R"] = ratio(Vn2, faces2)
        row["vertices"], row["cells"], row["cell_types"] = len(V2), len(cells2), cell_text(faces2, cells2)
        row["opt_off"] = write("optimized_off", f"{name}_{len(V2)}v_{len(cells2)}c_opt.off", Vn2, faces2, cells2)
        row["central_off"] = write("central_off", f"{name}_{len(V2)}v_{len(cells2)}c_central.off", Vn2, faces2, cells2)
        x = G.x4(uni)
        row["seed_vector"] = (x / np.abs(x).max()).tolist()
        out.append(row)
        print(f"{name:9} {row['kind']:6} {row['vertices']:6}v {row['cells']:5}c  seed R {row['seed_R']:.4f} -> R {row['R']:.9f}  "
              f"{row['cell_types']}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out}, open(os.path.join(HERE, "hyic_shapes.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
