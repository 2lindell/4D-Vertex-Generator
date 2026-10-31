"""Which wall lies on each located boundary (boundary_points.json, find_boundaries.py): the boundary points put exactly
on the surface they lie on (the wall's equation carried by the exact element of the extended group nearest the page's
copy matrix, with the fold where the surface's name has it) and classified there. On a wall's exact surface the
classifier names the wall; a little off it (under 1e-6) it can name a passing signature, so only points exactly on
the surface are used.

    python boundary_walls.py          (prints a line per boundary; writes boundary_walls.json)
"""
from __future__ import annotations

import json
from collections import Counter
from multiprocessing import Pool

import numpy as np

import region_cells as rc
import region_walls as rw
import xsurface
from edge_snap import Poly, newton
from region_db import home
from t2_pieces import extended

FOLD = np.eye(4)[[1, 0, 3, 2]]


class Carried:
    """f(A beta): a wall's equation carried by the matrix A."""

    def __init__(self, f, A):
        self.f, self.A = f, A

    def __call__(self, B):
        return self.f(np.atleast_2d(B) @ self.A.T)

    def grad(self, B):
        return self.f.grad(np.atleast_2d(B) @ self.A.T) @ self.A


def exact_surface(name, d, G):
    """The named surface (arrangement.surfaces naming: wall, wall:g, wall:fold, wall:g:fold, or a plane) exactly."""
    parts = name.split(":")
    tid = parts[0]
    curved = set(json.load(open("xsurface_equations.json"))) | {"X10", "X38", "X83", "X48", "X51", "X33"}
    if tid not in curved or (len(parts) > 1 and not (parts[1].isdigit() or parts[1] == "fold")):
        return None                                   # (a plane or a face of the cell: exact as it is)
    s = xsurface.surface(tid)
    f = Poly(s["mons"], s["coeffs"], tid)
    A = np.eye(4)
    if len(parts) >= 2 and parts[1].isdigit():
        M = np.asarray(d["copyMats"][parts[1]], float)
        g = min(G, key=lambda g_: np.abs(g_ / np.abs(g_).max() - M / np.abs(M).max()).max())
        A = np.linalg.inv(g)
    if parts[-1] == "fold":
        A = A @ FOLD
    return Carried(f, A)


def lab(b):
    return rc._lab_job(b) if rc.inside(b[None], -1e-12)[0] else rc._lab_job(home(b[None])[0])


def main():
    d = rw.page_data()
    G = extended()
    out = []
    with Pool(4) as pool:
        for r in json.load(open("boundary_points.json")):
            if not r["boundary_points"] or not r["best_surfaces"] or r["best_surfaces"][0][0] > 2e-4:
                continue
            name = r["best_surfaces"][0][1]
            W = exact_surface(name, d, G)
            if W is not None:
                P = [newton([W], b) for b in rc.betas(np.array(r["boundary_points"]))]
                res = max(abs(W(np.array(p)[None])[0]) for p in P)
            else:
                from arrangement import DOMAIN, project, surfaces
                S = {s_.name: s_ for s_ in surfaces() + DOMAIN}[name]
                X = project(np.array(r["boundary_points"]), [S])
                P = list(rc.betas(X))
                res = float(np.abs(S.at(X)).max())
            L = Counter(l if not l.startswith("new") else "new" for l in pool.map(lab, P))
            rec = {**{k: r[k] for k in ("wall", "section", "side", "between")}, "surface": name,
                   "on_it": dict(L), "residual": res, "nearest_mesh": r.get("nearest_mesh")}
            out.append(rec)
            print(f"{r['wall']} s{r['section']} side {r['side']} {r['between']}: on {name} exactly ({res:.0e}): "
                  f"{dict(L)}; page meshes {r.get('nearest_mesh')}", flush=True)
    json.dump(out, open("boundary_walls.json", "w"), indent=1)


if __name__ == "__main__":
    main()
