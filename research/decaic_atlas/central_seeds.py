"""The central seed of every shape: the point of its locus deepest inside it (central_seeds<tag>.json, central .off).

Where the optimized seed makes the edges as even as possible (and often sits on or near the shape's edge), the
central seed keeps as far as possible from every place the shape changes, so its cells are clearly formed. A shape is
the set of seeds on its locus (its mirror / wall / line equations) where every sign condition of its polychoron
(forms.region_conditions / region_forms: where a vertex reaches a cell's 3-flat) keeps its sign, inside the domain
(beta >= 0, b1 >= b4). The depth of a seed is its least distance to any of those boundaries, each measured to first
order (f / |grad f| along the locus, beta summing to 1). The central seed maximises the depth (SLSQP from the start
seed, the optimized seed and their midpoint), and is kept only if it is still the shape: the same arrangement key, and
the polychoron rebuilt with the shape's vertex and cell counts. A point has no freedom: its seed is its centre.

    DECAIC_SUB=full|ki|io python central_seeds.py      (contic_atlas/central_seeds.py for the contic groups)
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import sympy as sp
from scipy.optimize import minimize

O = None            # (the optimiser module whose shapes, loci and sign conditions are used: set by the runner)
G = None
F = None
OPT = {}            # (the optimized seeds by name, as extra starting points)


def centre(shape):
    seed = np.asarray(shape["seed"], float)
    seed = seed / seed.sum()
    tol = shape["hull_tol"]
    key0 = G.label(seed, tol)[1]
    res = {"name": shape["name"], "kind": shape["kind"], "degrees": shape["degrees"], "locus": shape["locus"],
           "start": seed.tolist()}
    if shape["degrees"] == 0:
        res.update({"centre": seed.tolist(), "depth": None, "status": "point (no freedom): the seed itself"})
        return res
    zero = [i for i in range(4) if seed[i] < 1e-12]
    free = [i for i in range(4) if i not in zero]
    loc = [sp.sympify(t) for t in shape["locus"]]
    loc = [e for e in loc if sp.expand(e.subs({O.B[i]: 0 for i in zero})) != 0]
    loc_f = [sp.lambdify(O.B, e, "numpy") for e in loc]
    loc_g = [[sp.lambdify(O.B, sp.diff(e, v), "numpy") for v in O.B] for e in loc]
    conds = O.sign_conditions(seed)
    Mx = np.array(F.MONS, float)
    C = np.array([c[0] for c in conds]) if conds else np.zeros((0, len(Mx)))
    Sg = np.array([c[1] for c in conds], float)

    def full(z):
        b = np.zeros(4)
        b[free] = z[:len(free)]
        return b

    def tangent(g, b):
        """The gradient's part along the locus: beta summing to 1, the mirrors' zeros kept, the locus equations."""
        N = [np.ones(4)] + [np.eye(4)[i] for i in zero]
        N += [np.array([gi(*b) for gi in lg]) for lg in loc_g]
        N = np.array(N)
        P = np.eye(4) - np.linalg.pinv(N) @ N
        return g @ P.T

    def depths(b):
        out = []
        if len(C):
            with np.errstate(divide="ignore", invalid="ignore"):
                mono = np.prod(b[None, :] ** Mx, axis=1)
                val = Sg * (C @ mono)
                dm = np.zeros((len(Mx), 4))
                for j in range(4):
                    e = Mx.copy()
                    e[:, j] -= 1
                    dm[:, j] = np.where(Mx[:, j] > 0, Mx[:, j] * np.prod(b[None, :] ** np.maximum(e, 0), axis=1)
                                        * np.where(e[:, j] < 0, 0, 1), 0)
            g = C @ dm
            gt = tangent(g, b)
            out.append(val / (np.linalg.norm(gt, axis=1) + 1e-300))
        dom = []
        for k in free:                                  # (the mirrors the shape is not on)
            dom.append(b[k] / (np.linalg.norm(tangent(np.eye(4)[k][None], b)) + 1e-300))
        if 0 in free and 3 in free and not any(sp.expand(e - (O.B[0] - O.B[3])) == 0 for e in loc):
            dom.append((b[0] - b[3]) / (np.linalg.norm(tangent(np.array([[1, 0, 0, -1.0]]), b)) + 1e-300))
        out.append(np.array(dom, float))
        return np.concatenate(out)

    nf = len(free)
    cons = [{"type": "ineq", "fun": lambda z: depths(full(z)) - z[nf]},
            {"type": "eq", "fun": lambda z: np.array([full(z).sum() - 1] + [f(*full(z)) for f in loc_f])}]
    starts = [seed]
    o = OPT.get(shape["name"])
    if o is not None:
        b2 = np.asarray(o.get("off_seed", o["best"]), float)
        starts += [b2, (seed + b2) / 2]
    best, bestd = None, -np.inf
    for p in starts:
        p = O.project(p, loc_f, loc_g, zero)
        z0 = np.concatenate([p[free], [max(depths(p).min(), 0)]])
        r = minimize(lambda z: -z[nf], z0, method="SLSQP", constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
        b = full(r.x)
        if b.sum() <= 0:
            continue
        b = b / b.sum()
        if not G.in_domain(b[None], 1e-12)[0]:
            continue
        d = float(depths(b).min())
        if d > bestd and G.label(b, tol)[1] == key0:
            best, bestd = b, d
    d0 = float(depths(seed).min())
    if best is None or bestd < d0:
        best, bestd, status = seed, d0, "no deeper point found: the seed itself"
    else:
        status = "deepest point of the shape"
    res.update({"centre": best.tolist(), "depth": bestd, "start_depth": d0, "status": status,
                "boundaries": int(len(depths(best)))})
    return res


def run(shapes, out_dir, x_vector, hull, json_path):
    """Find every shape's centre, write its .off (checked: same vertex and cell counts as the shape's seed)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".off"):
            os.remove(os.path.join(out_dir, f))
    with Pool(4) as pool:
        R = pool.map(centre, shapes, chunksize=1)
    for s, r in zip(shapes, R):
        V0, _, cells0 = hull(np.asarray(r["start"]), s["hull_tol"])
        b = np.asarray(r["centre"], float)
        V, faces, cells = hull(b, s["hull_tol"])
        if (len(V), len(cells)) != (len(V0), len(cells0)):
            r["status"] += f"; REBUILT DIFFERENTLY ({len(V)}v {len(cells)}c, the shape has {len(V0)}v {len(cells0)}c): seed kept"
            b = np.asarray(r["start"], float)
            V, faces, cells = hull(b, s["hull_tol"])
        name = f"{r['name']}_{len(V)}v_{len(cells)}c_central"
        from four_d_vertex_generator.off import to_4off
        open(os.path.join(out_dir, name + ".off"), "w").write(to_4off(V, faces, cells))
        r["off"] = os.path.relpath(os.path.join(out_dir, name + ".off"), os.path.dirname(os.path.dirname(out_dir)))
        r["vertices"], r["cells"] = len(V), len(cells)
        x = x_vector(b)
        r["seed_vector"] = (x / np.abs(x).max()).tolist()
        print(f"{r['name']:34} {r['kind']:6} depth {r['start_depth'] if r.get('start_depth') is not None else 0:.2e} -> "
              f"{r['depth'] if r['depth'] is not None else 0:.2e}  {r['status'][:60]:60} {len(V)}v {len(cells)}c  "
              f"beta {np.round(b, 6).tolist()}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": R}, open(json_path, "w"), indent=1)


def main():
    global O, G, F, OPT
    import optimize_seeds as OS
    O, G, F = OS, OS.G, OS.F
    try:
        OPT = {r["name"]: r for r in json.load(open(G.fname("optimized_seeds")))["shapes"]}
    except FileNotFoundError:
        OPT = {}
    shapes = OS.decaic_shapes() if G.SUB == "full" else OS.snub_shapes()
    out = os.path.join(OS.ROOT, "central_off", {"full": "decaic", "ki": "kidecaic", "io": "iodecaic"}[G.SUB])

    def hull(b, tol):
        _, V, faces, cells = G.polytope(b, tol)
        return V, faces, cells
    run(shapes, out, G.x5, hull, G.fname("central_seeds"))


if __name__ == "__main__":
    main()
