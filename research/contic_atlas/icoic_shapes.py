"""The nine icoic shapes ([3,4,3] = F4, order 1152): seed, optimized and central .off files (icoic_shapes.json).

F4 is this atlas's group without the swap (CONTIC_SUB=ico); the swap phi normalises it, so the half-chamber b1 >= b4
is still the shape domain. Each shape lies on a face of the chamber (its zero betas):

    ico (1,0,0,0) point, rico (0,1,0,0) point, sidpii (a,0,0,d) line, tico (a,b,0,0) line, srico (a,0,c,0) line,
    icoi (0,b,c,0) line, grico (a,b,c,0) wall, prico (a,b,0,d) wall, gidpii (a,b,c,d) region.

The seed .off is the uniform member where there is one (ico, rico, tico, srico, grico, prico) and otherwise a point of
the face away from the fold; the optimized seed minimises R = longest / shortest edge on the face (optimize_seeds),
the central seed is the face's deepest point (central_seeds). Sidpii, icoi and gidpii become contic shapes on the fold
(b1 = b4 with b2 = b3 where needed: spic, cont, gippic), where their two sets of 24 cells are equal; their optimized
seeds go there if that is where the edges are most even, and that is reported.

    python icoic_shapes.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ["CONTIC_SUB"] = "ico"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import central_seeds as CSR  # noqa: E402  (loads the decaic centre finder)
import optimize_seeds as C  # noqa: E402

from four_d_vertex_generator.local_view import hull_edges  # noqa: E402
from four_d_vertex_generator.off import to_4off  # noqa: E402

G = C.G
CS = CSR.CS
SHAPES = [("ico", "point", 0, [1, 0, 0, 0], [1, 0, 0, 0]), ("rico", "point", 0, [0, 1, 0, 0], [0, 1, 0, 0]),
          ("sidpii", "line", 1, [0.7, 0, 0, 0.3], None), ("tico", "line", 1, [0.8, 0.2, 0, 0], [1, 1, 0, 0]),
          ("srico", "line", 1, [0.75, 0, 0.25, 0], [1, 0, 1, 0]), ("icoi", "line", 1, [0, 0.35, 0.65, 0], None),
          ("grico", "wall", 2, [0.55, 0.3, 0.15, 0], [1, 1, 1, 0]), ("prico", "wall", 2, [0.45, 0.3, 0, 0.25], [1, 1, 0, 1]),
          ("gidpii", "region", 3, [0.4, 0.3, 0.2, 0.1], None)]


def edge_ratio(b):
    V, Vn, idx, faces, cells = G.polytope(np.asarray(b, float), 1e-9)
    E = [np.linalg.norm(Vn[a] - Vn[c]) for a, c in (tuple(e) for e in hull_edges(faces))]
    return float(max(E) / min(E))


def write(folder, name, tag, b):
    d = os.path.join(ROOT, folder, "icoic")
    os.makedirs(d, exist_ok=True)
    V, Vn, idx, faces, cells = G.polytope(np.asarray(b, float), 1e-9)
    f = os.path.join(d, f"{name}_{len(V)}v_{len(cells)}c_{tag}.off")
    open(f, "w").write(to_4off(Vn, faces, cells))
    return os.path.relpath(f, ROOT), len(V), len(cells)


def main():
    CS.O, CS.G, CS.F = C.O, G, C.F
    out = []
    for name, kind, deg, start, uniform in SHAPES:
        s = {"name": name, "kind": kind, "degrees": deg, "seed": start, "locus": [], "hull_tol": 1e-9}
        seed = np.asarray(uniform if uniform is not None else start, float)
        seed = seed / seed.sum()
        row = {"name": name, "kind": kind, "degrees": deg, "seed": seed.tolist(), "seed_R": edge_ratio(seed)}
        row["seed_off"], nv, nc = write(os.path.join("new_shapes", "off"), f"icoic_{kind}_{name}", "seed", seed)
        row["vertices"], row["cells"] = nv, nc
        if deg == 0:
            row.update(optimized=seed.tolist(), R=row["seed_R"], optimized_status="point (no freedom)",
                       central=seed.tolist(), depth=None, central_status="point (no freedom)")
        else:
            r = C.O.optimise(s)
            CS.OPT = {name: r}
            c = CS.centre(s)
            if name == "icoi":
                # (on b1 = b4 = 0 the swap folds the line at b2 = b3 (cont): the centre finder does not see that end, so
                #  the deepest point of the half-line between rico (b3 = 0, distance b3 sqrt2) and cont (distance
                #  (b2 - b3) / sqrt2) is put here: b2 = 3 b3)
                c = {"centre": [0, 0.75, 0.25, 0], "depth": 0.25 * 2 ** 0.5,
                     "status": "deepest point of the half-line between rico and cont (b2 = 3 b3)"}
            ob = np.asarray(r.get("off_seed", r.get("best", r["start"])), float)
            row.update(optimized=ob.tolist(), R=r["R"], optimized_status=r["status"], central=c["centre"],
                       depth=c["depth"], central_status=c["status"])
        for folder, key, tag in (("optimized_off", "optimized", "opt"), ("central_off", "central", "central")):
            row[f"{tag}_off"], nv, nc = write(folder, name, tag, row[key])
            row[f"{tag}_counts"] = [nv, nc]
        b = np.asarray(row["optimized"])
        row["optimized_on_fold"] = bool(abs(b[0] - b[3]) < 1e-7)
        x = G.x4(b)
        row["seed_vector"] = (x / np.abs(x).max()).tolist()
        out.append(row)
        print(f"{name:7} {kind:6} {row['vertices']:5}v {row['cells']:4}c  seed R {row['seed_R']:.4f}  optimized R "
              f"{row['R']:.6f} {np.round(row['optimized'], 6).tolist()} {row['opt_counts']}  central "
              f"{np.round(row['central'], 6).tolist()} {row['central_counts']}  {row['optimized_status'][:50]}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out},
              open(os.path.join(HERE, "icoic_shapes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
