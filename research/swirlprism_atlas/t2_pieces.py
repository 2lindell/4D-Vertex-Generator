"""T2's pieces for the region colouring: its patches as the page draws them (the cell's edges split the view, not the
wall), the sliver patch left out, cut only along the lines other walls meet it in: L, where X11's plane
beta3 = beta2 + beta4 meets T2's plane beta1 = beta3 (from C5 to the end of the line X11 on X4), and L's images on the
patches that are copies under the extra half-turn (the same element of the extended group that takes the copy back
onto its original takes the cutting plane with it). Each piece is named from the region database's voxels beside it.

    python t2_pieces.py          (writes sec_pieces/T2.json.gz)
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
from shapely.geometry import Polygon

import region_cells as rc
import region_walls as rw
from wall_sections import Grid

FL = np.array([1, 1, -1.0])
FOLD = np.eye(4)[[1, 0, 3, 2]]
DROP = {4}                                   # (patch 5, a sliver: not a piece of T2)
CUT = {0: np.array([0, -1, 1, -1.0])}        # (patch 1: beta3 - beta2 - beta4 = 0, X11's plane)


def nrm(X):
    return X / X.sum(axis=-1, keepdims=True)


def frame(B):
    """A 2D frame of the patch's plane (in cell xyz)."""
    X = rc.xyz(B)
    o = X.mean(0)
    u, s, vt = np.linalg.svd(X - o)
    return o, vt[0], vt[1], vt[2]


def extended():
    from cellframe import T as _T
    from normalizer import extended_group
    import cell_atlas2 as ca
    return np.einsum("ij,gjk,kl->gil", ca.TINV, np.array(extended_group(), float), np.array(_T, float).T)


def pieces():
    d = rw.page_data()
    patches = [nrm(rc.betas(np.array(t["pts"], float) * FL)) for t in d["tpatches"]]
    G = extended()
    cuts = dict(CUT)
    # (each copy patch: the element taking it back into an original patch with a cut; that cut, carried over)
    for k, P in enumerate(patches):
        if k in cuts or k in DROP or not d["tpatches"][k].get("copy"):
            continue
        for src, n in CUT.items():
            o, e1, e2, nn = frame(patches[src])
            poly = Polygon(np.c_[(rc.xyz(patches[src]) - o) @ e1, (rc.xyz(patches[src]) - o) @ e2]).buffer(1e-6)
            hit = None
            for M in G:
                for F in (np.eye(4), FOLD):
                    A = np.linalg.inv(F @ M)
                    Y = P @ A.T
                    if (np.abs(Y.sum(1)) < 1e-12).any():
                        continue
                    Y = nrm(Y)
                    X = rc.xyz(Y)
                    if np.abs((X - o) @ nn).max() > 1e-6:
                        continue
                    if all(poly.contains(Polygon(np.c_[(X - o) @ e1, (X - o) @ e2]).representative_point()) for _ in [0]) \
                            and poly.buffer(1e-4).contains(Polygon(np.c_[(X - o) @ e1, (X - o) @ e2])):
                        hit = A
                        break
                if hit is not None:
                    break
            if hit is not None:
                cuts[k] = n @ hit                # (beta -> n . (A beta): the cut plane on the copy)
                print(f"  patch {k + 1} is a copy of patch {src + 1}: L carried over")
    out = []
    for k, P in enumerate(patches):
        if k in DROP:
            continue
        o, e1, e2, nn = frame(P)
        X = rc.xyz(P)
        poly = Polygon(np.c_[(X - o) @ e1, (X - o) @ e2]).buffer(0)
        parts = [poly]
        if k in cuts:
            # (the cut plane, met with the patch's plane: a line in its 2D frame, the patch split along it)
            n = cuts[k]
            big = poly.buffer(0.05).bounds
            gx, gy = np.meshgrid(np.linspace(big[0], big[2], 400), np.linspace(big[1], big[3], 400))
            Q3 = o + gx.ravel()[:, None] * e1 + gy.ravel()[:, None] * e2
            vals = (nrm(rc.betas(Q3)) @ n).reshape(gx.shape)
            pos = Polygon([(big[0], big[1]), (big[2], big[1]), (big[2], big[3]), (big[0], big[3])])
            # (the line through two points of the zero set, the plane being linear in betas and so in the frame)
            from skimage.measure import find_contours
            cs = find_contours(vals, 0.0)
            if cs:
                c = max(cs, key=len)
                to_xy = lambda rc_: (big[0] + rc_[1] / 399 * (big[2] - big[0]), big[1] + rc_[0] / 399 * (big[3] - big[1]))
                a, b = np.array(to_xy(c[0])), np.array(to_xy(c[-1]))
                dvec = (b - a) / np.linalg.norm(b - a)
                nrm2 = np.array([-dvec[1], dvec[0]])
                far = 10
                half = Polygon([a - far * dvec, b + far * dvec, b + far * dvec + far * nrm2, a - far * dvec + far * nrm2])
                parts = [g for g in (poly.intersection(half), poly.difference(half)) if g.area > 1e-12]
        for g in parts:
            for gg in (g.geoms if hasattr(g, "geoms") else [g]):
                if gg.area <= 1e-12:
                    continue
                import shapely
                tris = []
                for t in shapely.constrained_delaunay_triangles(gg).geoms:
                    c2 = np.array(t.exterior.coords)[:3]
                    tris.append(o + c2[:, :1] * e1 + c2[:, 1:2] * e2)
                out.append((k, d["tpatches"][k].get("copy", False), np.array(tris), nn))
    return out


def main():
    grid = Grid()
    faces = []
    for k, copy, T, n in pieces():
        A = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        rng = np.random.default_rng(k)
        j = rng.choice(len(T), 300, p=A / A.sum())
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
            if not (first > 0).any():
                from region_db import home
                for depth in (1.5, 2.5, 3.5, 4.5, 5.5):
                    Bh = home(rc.betas(X + sg * depth * grid.h * n))
                    ok = ~np.isnan(Bh).any(1)
                    lab = np.zeros(len(X), int)
                    lab[ok] = grid.at(rc.xyz(Bh[ok]))
                    take = (first == 0) & (lab > 0)
                    first[take] = lab[take]
            c = np.bincount(first[first > 0]) if (first > 0).any() else None
            sides.append(grid.names[int(c.argmax())] if c is not None else "unknown")
        print(f"  patch {k + 1}{' (copy)' if copy else ''}: piece of area {A.sum():.5f}, sides {sides[0]} | {sides[1]}")
        if sides[0] != sides[1]:
            faces.append({"sides": sides, "kind": "copy" if copy else "orig", "mirror": False, "cut": [False, False],
                          "tris": T.reshape(-1, 9).round(7).tolist()})
    json.dump(faces, gzip.open(os.path.join("sec_pieces", "T2.json.gz"), "wt"), separators=(",", ":"))


if __name__ == "__main__":
    main()
