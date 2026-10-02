"""Where the seed stops lying on the cell at each corner of the domain.

At a corner c of the displayed half-cell, the polytope of a nearby seed has a cell whose hyperplane (realm) is
perpendicular to c, and the seed vertex lies on it. The seed x lies in that cell's hyperplane exactly when it
maximises x . c over its orbit:  (g x) . c <= x . c  for every g, i.e.  x . (c - g^T c) >= 0.
Each condition is a plane through the origin in seed space, so a plane in barycentric coordinates beta
(x is proportional to T^T beta). The region where the seed is on the corner cell is the Dirichlet domain of c's
orbit; its faces are where the seed breaks off from the corner cell. Some of these planes cross the displayed
half-cell, others only bound the region outside it.

    python corner_cells.py
"""
from __future__ import annotations

import json

import numpy as np
from cellframe import T, golden_form, seed_from_beta
from scipy.spatial import HalfspaceIntersection

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

G = np.stack(group_elements(named_symmetry("h4_swirlprism")))
CORNERS = {                       # the displayed half-cell (beta3 >= beta4)
    "V1": [1, 0, 0, 0], "V2": [0, 1, 0, 0], "V3": [0, 0, 1, 0], "M34": [0, 0, 1, 1],
}
CELL = [np.eye(4)[i] for i in range(4)] + [np.array([0, 0, 1, -1.0])]     # beta_i >= 0, beta3 >= beta4
BASIS = np.linalg.svd(np.ones((1, 4)))[2][1:]                              # sum(beta) fixed


def _nice(n):
    n = np.asarray(n, float)
    n = n / n[np.flatnonzero(np.abs(n) > 1e-9)[0]]
    return n


def eq_text(n):
    n = _nice(n)
    c = lambda v: golden_form(v)  # noqa: E731
    pos = " + ".join(("" if abs(v - 1) < 1e-9 else f"({c(v)})·") + f"β{i + 1}" for i, v in enumerate(n) if v > 1e-9)
    neg = " + ".join(("" if abs(v + 1) < 1e-9 else f"({c(-v)})·") + f"β{i + 1}" for i, v in enumerate(n) if v < -1e-9)
    return f"{pos} = {neg or 0}"


def corner_planes(c_beta):
    """Distinct planes n . beta >= 0 (n = T (c - g^T c)) that keep the seed on the corner cell."""
    c = seed_from_beta(c_beta)
    planes = []
    for g in G:
        d = c - g.T @ c
        if np.linalg.norm(d) < 1e-9:
            continue                                   # g fixes c
        n = T @ d                                      # x = T^T beta (up to scale), so x . d = beta . (T d)
        n = n / np.linalg.norm(n)
        if not any(np.allclose(n, m, atol=1e-9) for m in planes):
            planes.append(n)
    return planes


def region(planes, extra=CELL):
    """The polytope {beta: n . beta >= 0 for all planes and the cell constraints, sum beta = 1} in 3D, with the
    planes that carry a face."""
    allp = list(planes) + list(extra)
    # in coordinates u: beta = b0 + BASIS^T u, b0 = (1/4,...)
    b0 = np.full(4, 0.25)
    rows = np.array([np.append(-(BASIS @ n), -(n @ b0)) for n in allp])      # -n.(b0 + B^T u) <= 0
    # an interior point: least violated point by a small LP-free search
    from scipy.optimize import linprog
    A = np.hstack([rows[:, :3], np.linalg.norm(rows[:, :3], axis=1)[:, None]])
    lp = linprog([0, 0, 0, -1], A_ub=A, b_ub=-rows[:, 3], bounds=[(None, None)] * 3 + [(0, 1)])
    if not lp.success or lp.x[3] < 1e-12:
        return None
    hs = HalfspaceIntersection(rows, lp.x[:3])
    pts = []
    for u in hs.intersections:
        if not any(np.linalg.norm(u - w) < 1e-10 for w in pts):
            pts.append(u)
    pts = np.array(pts)
    betas = b0 + pts @ BASIS
    faces = []
    for k, n in enumerate(allp):
        on = np.flatnonzero(np.abs(betas @ n) < 1e-10)
        if len(on) >= 3:
            faces.append({"plane": k, "is_cell_wall": k >= len(planes), "corners": on.tolist()})
    return {"corners": betas, "faces": faces}


def _order(P, n):
    c = P.mean(axis=0)
    B = np.linalg.svd(np.vstack([np.ones(4), n]))[2][2:]
    ang = np.arctan2(*((P - c) @ B.T).T[::-1])
    return P[np.argsort(ang)]


def main():
    from probe import _one
    out = {}
    for name, cb in CORNERS.items():
        cb = np.array(cb, float) / sum(cb)
        planes = corner_planes(cb)
        R = region(planes)
        full = region(planes, extra=[np.eye(4)[i] for i in range(4)])    # in the whole cell, both halves
        rec = {"corner": cb.tolist(), "shape_at_corner": _one(cb)[0], "planes_total": len(planes), "faces": []}
        inside_planes = set()
        if R is not None:
            for f in R["faces"]:
                if f["is_cell_wall"]:
                    continue
                n = planes[f["plane"]]
                inside_planes.add(f["plane"])
                poly = _order(R["corners"][f["corners"]], n)
                rec["faces"].append({"equation": eq_text(n), "normal": _nice(n).tolist(), "polygon": poly.tolist()})
        rec["region_corners"] = [] if R is None else R["corners"].tolist()
        if full is not None:
            rec["outside_planes"] = [eq_text(planes[f["plane"]]) for f in full["faces"]
                                     if not f["is_cell_wall"] and f["plane"] not in inside_planes]
        out[name] = rec
        print(f"{name} at β ∝ {np.round(cb / cb.max(), 4).tolist()} ({rec['shape_at_corner']}): {len(planes)} planes; "
              f"the region in the displayed half-cell has {len(rec['faces'])} breaking faces:")
        for f in rec["faces"]:
            print("    ", f["equation"])
        if rec.get("outside_planes"):
            print("   bounding the region only outside the displayed half:", rec["outside_planes"])
    json.dump(out, open("corner_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
