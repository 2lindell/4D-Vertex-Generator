"""Which stretches of a wall's drawn outline are drawn, and how (used by cell_atlas2.py when it builds the page).

A wall's outline (each drawn piece of it: the main piece and each copy) is split into runs:
  edge  on one of the wall's edges in working_data.json: a line (any image of it in the half-cell) or another wall
        (any of its pieces); drawn in the wall's colour (solid, dashed on a copy);
  gap   on something else that is not one of its edges: a named line or another wall, where the wall goes on across
        it (to its own copy, or past a wall that ends on it or crosses it); drawn white, to show the break;
  none  on nothing (a face of the half-cell or the splitting mirror where the wall goes on into a copy drawn
        elsewhere, and nothing lies there): not drawn.
So no line is drawn on a wall but its own edges, and the white lines where it is cut.

The points of every wall and line in the half-cell (all images under the extended group) are kept in
edge_display_cache.npz, rebuilt when missing.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial import cKDTree

import edge_curves as ec

TOL_LINE = 2.5e-3
CACHE = "edge_display_cache.npz"


def _points(work):
    """(walls: {id: (points, h)}, lines: {id: points}), all images in the half-cell."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["walls"].item(), z["lines"].item()
    import region_walls as rw
    d = rw.page_data()
    X = json.load(open("xsurfaces.json"))
    fp = [p for p in (lambda x: x if isinstance(x, list) else x.values())(json.load(open("fexact_patches.json")))
          if isinstance(p, dict)]
    walls = {}
    for w in work["walls"]:
        W = ec.Wall(w, X, fp)
        walls[w] = (W.V.astype(np.float32), W.h)
    L0 = ec.page_lines(d)
    lines = {l: ec.line_images(P).astype(np.float32) for l, P in L0.items()}
    np.savez_compressed(CACHE, walls=np.array(walls, dtype=object), lines=np.array(lines, dtype=object))
    return walls, lines


class Classifier:
    def __init__(self):
        self.work = json.load(open("working_data.json"))
        walls, lines = _points(self.work)
        self.wt = {w: (cKDTree(V), max(TOL_LINE, 0.7 * h)) for w, (V, h) in walls.items() if len(V)}
        self.lt = {l: cKDTree(P) for l, P in lines.items() if len(P)}

    def kinds(self, w, B):
        """For each point of wall w's outline (betas, n x 4): 'edge', 'gap' or None."""
        B = np.asarray(B, float)
        edges = set(self.work["walls"].get(w, {}).get("edges", []))
        out = np.array([None] * len(B), dtype=object)
        near = lambda shape: (self.lt[shape].query(B)[0] < TOL_LINE) if shape in self.lt else (
            (self.wt[shape][0].query(B)[0] < self.wt[shape][1]) if shape in self.wt else np.zeros(len(B), bool))
        on_edge = np.zeros(len(B), bool)
        for e in edges:
            on_edge |= near(e)
        out[on_edge] = "edge"
        rest = ~on_edge
        if rest.any():
            other = np.zeros(len(B), bool)
            for s in list(self.lt) + list(self.wt):
                if s != w and s not in edges:
                    other[rest] |= near(s)[rest]
            out[rest & other] = "gap"
        return out


def runs(pts, kinds):
    """Consecutive runs of the same kind (each run includes the point where the next begins, so they join)."""
    out, cur, k0 = [], [], None
    for p, k in zip(pts, kinds):
        if k != k0 and cur:
            if k0 is not None and len(cur) >= 2:
                out.append((k0, cur + [p]))
            cur = []
        cur.append(p)
        k0 = k
    if k0 is not None and len(cur) >= 2:
        out.append((k0, cur))
    return out
