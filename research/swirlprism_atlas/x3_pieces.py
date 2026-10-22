"""X3's pieces for the region colouring: its two flat patches (beta1 = beta2, and beta2 = beta3 + phi^2 beta4, which is
the first's copy under the extended group, the two sharing the crease X1), each cut along the curve where X10 ends on
it: on the second patch the conic where X10's copy surface meets it, on the first the same curve carried over by the
element taking the first patch onto the second. Each piece is named from the region database's voxels beside it.

    python x3_pieces.py          (writes sec_pieces/X3.json.gz)
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import split

import region_cells as rc
import region_walls as rw
from arrangement import surfaces
from t2_pieces import FOLD, extended, frame, nrm
from wall_sections import Grid

PHI = (1 + 5 ** 0.5) / 2


def split_by(P, f):
    """The flat patch with corners P (betas) split along f(betas) = 0: [(triangles in cell xyz, normal)]."""
    o, e1, e2, nn = frame(P)
    X = rc.xyz(P)
    poly = Polygon(np.c_[(X - o) @ e1, (X - o) @ e2])
    b = poly.buffer(0.05).bounds
    m = 600
    gx, gy = np.meshgrid(np.linspace(b[0], b[2], m), np.linspace(b[1], b[3], m))
    vals = f(nrm(rc.betas(o + gx.ravel()[:, None] * e1 + gy.ravel()[:, None] * e2))).reshape(gx.shape)
    parts = [poly]
    from skimage.measure import find_contours
    for c in find_contours(vals, 0.0):
        xy = np.c_[b[0] + c[:, 1] / (m - 1) * (b[2] - b[0]), b[1] + c[:, 0] / (m - 1) * (b[3] - b[1])]
        line = LineString(xy).simplify(2e-5)         # (the contour's dense points thinned: no needle triangles)
        if not line.intersects(poly):
            continue
        parts = [q for p in parts for q in split(p, line).geoms]
    out = []
    for p in parts:
        if p.area < 1e-12:
            continue
        T = np.array([o + np.array(t.exterior.coords)[:3, :1] * e1 + np.array(t.exterior.coords)[:3, 1:2] * e2
                      for t in shapely.constrained_delaunay_triangles(p).geoms])
        out.append((T, nn))
    return out


def main():
    d = rw.page_data()
    S = {s.name: s for s in surfaces()}
    s10 = S["X10:1200"]
    f10 = lambda B: s10.f(B)
    X1a, X1b = nrm(np.array([2 + PHI, 2 + PHI, 1, 1.0])), nrm(np.array([1, 1, 1, 0.0]))
    p1 = np.array([X1a, X1b, nrm(np.array([1, 1, 1, 1.0]))])               # (beta1 = beta2)
    p2 = np.array([X1a, X1b, nrm(np.array([1 + PHI, 1, 1, 0.0]))])         # (beta2 = beta3 + phi^2 beta4)
    # (the element taking patch 1 onto patch 2, corner to corner)
    A = None
    for M in extended():
        for F in (np.eye(4), FOLD):
            Y = p1 @ (F @ M).T
            if (np.abs(Y.sum(1)) < 1e-12).any():
                continue
            Yn = nrm(Y)
            if all(np.abs(Yn - q).max(1).min() < 1e-7 for q in p2) and all(np.abs(p2 - y).max(1).min() < 1e-7 for y in Yn):
                A = F @ M                         # (the corners in any order: X1's ends may swap)
                break
        if A is not None:
            break
    print("element taking patch 1 onto patch 2:", "found" if A is not None else "NONE")
    grid = Grid()
    faces = []
    for name, P, f in (("beta1 = beta2", p1, (lambda B: f10(B @ A.T)) if A is not None else None),
                       ("beta2 = beta3 + phi^2 beta4", p2, f10)):
        for T, n in (split_by(P, f) if f is not None else [(None, None)]):
            Ar = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
            if Ar.sum() < 1e-8:
                continue                          # (slivers where the curve's contour grazes the patch's edge)
            rng = np.random.default_rng(0)
            j = rng.choice(len(T), 300, p=Ar / Ar.sum())
            u, v = rng.random(300), rng.random(300)
            fl = u + v > 1
            u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
            X = T[j, 0] + u[:, None] * (T[j, 1] - T[j, 0]) + v[:, None] * (T[j, 2] - T[j, 0])
            sides = []
            for sg in (1, -1):
                first = np.zeros(len(X), int)
                for depth in (1.5, 2.5, 3.5, 4.5, 5.5):
                    lab = grid.at(X + sg * depth * grid.h * n)
                    take = (first == 0) & (lab > 0)
                    first[take] = lab[take]
                c = np.bincount(first[first > 0]) if (first > 0).any() else None
                sides.append(grid.names[int(c.argmax())] if c is not None else "unknown")
            print(f"  patch {name}: piece of area {Ar.sum():.5f}, sides {sides[0]} | {sides[1]}")
            if sides[0] != sides[1]:
                faces.append({"sides": sides, "kind": "orig", "mirror": False, "cut": [False, False],
                              "tris": T.reshape(-1, 9).round(7).tolist()})
    json.dump(faces, gzip.open(os.path.join("sec_pieces", "X3.json.gz"), "wt"), separators=(",", ":"))


if __name__ == "__main__":
    main()
