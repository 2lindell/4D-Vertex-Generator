"""The lines and points on each region's boundary, checked against its walls. A region's boundary is its walls (from
region_defs.json) and the parts of the half-cell's faces around it; the lines and points it has are those of its
walls (boundaries.json) that lie on the part of each wall facing it, and those on the faces beside it. So a wall that
bounds the region only in part (region_db.json "partial") gives only the lines and points of that part.

Each candidate (a line or point bounding one of the region's walls, or on a face of the half-cell) is kept where its
geometry (the page's traced lines and samples, copies included) lies on the region's own wall sections (the coloured
outer walls, region_walls.json from region_view.py --defs), or on a face of the half-cell next to the region's voxels.
A line counts when a stretch of it does (not just an end touching).

    python region_bounds.py        (writes region_bounds.json and prints what changes against the page's lists)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from wall_sections import Grid

FLIP = np.array([1, 1, -1.0])
ON_WALL = 6e-4              # (a line or point this close to the region's wall sections lies on them)
STRETCH = 0.008             # (a line's points on the region span at least this much: more than an end touching)


def geometry(d):
    """Cell points (xyz) of every line and point shape: the traced lines (copies too), each a polyline densified to
    1e-3, and the samples apart."""
    runs, samp = defaultdict(list), defaultdict(list)
    for key in ("xlines", "rings", "segments", "main"):
        for x in d.get(key, []):
            if isinstance(x, dict) and x.get("pts") and not x.get("outline"):
                P = np.array(x["pts"], float) * FLIP
                Q = [P[0]]
                for a, b in zip(P[:-1], P[1:]):
                    n = max(2, int(np.linalg.norm(b - a) / 1e-3) + 1)
                    Q += list(a + np.linspace(0, 1, n)[1:, None] * (b - a))
                runs[x["id"]].append(np.array(Q))
    for k, v in d["samples"].items():
        samp[k] += [np.array(p[:3], float) * FLIP for p in v]
    for key in ("uniform", "special"):
        for x in d.get(key, []):
            if isinstance(x, dict) and "q" in x and x.get("id"):
                samp[x["id"]].append(np.array(x["q"], float) * FLIP)
    return runs, {k: np.array(v) for k, v in samp.items()}


def longest_run(P, hit):
    """The longest stretch (arc length) of consecutive points of a polyline that are all hits."""
    best = cur = 0.0
    for k in range(1, len(P)):
        cur = cur + np.linalg.norm(P[k] - P[k - 1]) if hit[k] and hit[k - 1] else 0.0
        best = max(best, cur)
    return best


def on_face(X, tol=2e-4):
    B = rc.betas(X)
    B = B / B.sum(1, keepdims=True)
    return (np.abs(B) < tol).any(1) | (np.abs(B[:, 2] - B[:, 3]) < tol)


def main():
    d = rw.page_data()
    dims = {t["id"]: t.get("dim") for t in d["types"]}
    page_bounds = {t["id"]: t.get("bounds") or [] for t in d["types"]}
    bnd = json.load(open("boundaries.json"))
    defs = json.load(open("region_defs.json"))
    partial = json.load(open("region_db.json")).get("partial", {})
    runs, samp = geometry(d)
    grid = Grid()
    RW = json.load(open("region_walls.json"))             # (each region's wall sections: region_view.py --defs)
    vox_of = {n: k for k, n in enumerate(grid.names) if n}
    out, report = {}, []
    for r, v in defs.items():
        if r.startswith("_") or not isinstance(v, dict) or r not in RW:
            continue
        T = np.array(RW[r]["tris"], float).reshape(-1, 3, 3) * FLIP
        # (the region's wall sections, sampled)
        from cut_walls import sample_tris
        wall_tree = cKDTree(sample_tris(T, 7e-4)) if len(T) else None
        # (the region's voxels, for the faces of the half-cell beside it)
        idx = np.argwhere(grid.lab == vox_of.get(r, -1))
        vox_tree = cKDTree(grid.lo + (idx + 0.5) * grid.h) if len(idx) else None
        walls = v.get("walls", [])
        cand = set()
        for w in walls:
            cand |= {b for b in bnd.get(w, []) if dims.get(b) in ("line", "point")}
        cand |= {b for b in page_bounds.get(r, []) if dims.get(b) in ("line", "point")}
        cand |= {b for b in bnd.get(r, []) if dims.get(b) in ("line", "point")}
        keep, why = set(), {}
        def hits(G):
            onw = wall_tree.query(G)[0] < ON_WALL if wall_tree is not None else np.zeros(len(G), bool)
            onf = on_face(G) & (vox_tree.query(G)[0] < 1.6 * grid.h) if vox_tree is not None else np.zeros(len(G), bool)
            return onw, onf
        for c in sorted(cand):
            if dims[c] == "line" and runs.get(c):
                # (a stretch of one of its traced pieces along the region, not points scattered near a corner)
                best, how = 0.0, None
                for P in runs[c]:
                    onw, onf = hits(P)
                    L = longest_run(P, onw | onf)
                    if L > best:
                        best, how = L, ("on its walls" if longest_run(P, onw) >= L else "on a face beside it")
                if best < STRETCH:
                    continue
                keep.add(c)
                why[c] = f"{how} ({best:.3f} of it)"
                continue
            G = samp.get(c)
            if G is None or not len(G):
                continue
            onw, onf = hits(G)
            if dims[c] == "line" and (onw | onf).sum() < 5:
                continue
            if not (onw | onf).any():
                continue
            keep.add(c)
            why[c] = "on its walls" if onw.any() else "on a face beside it"
        old = {b for b in page_bounds.get(r, []) if dims.get(b) in ("line", "point")}
        out[r] = {"walls": walls, "partial": partial.get(r, []), "lines": sorted(x for x in keep if dims[x] == "line"),
                  "points": sorted(x for x in keep if dims[x] == "point"), "how": why}
        gone, new = sorted(old - keep), sorted(keep - old)
        if gone or new:
            report.append(f"{r} (walls {', '.join(walls)}; partial {', '.join(partial.get(r, [])) or '-'}): "
                          f"drop {gone or '-'}; add {new or '-'}")
    json.dump(out, open("region_bounds.json", "w"), indent=1)
    print("\n".join(report) or "no changes")


if __name__ == "__main__":
    main()
