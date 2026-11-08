"""The star of vertex 0 (the hull cells through it, as vertex-index sets): a fast, exact test of "same polytope".

The vertices are the orbit of one point in a fixed order (group.vertices), and every cell of the hull is an image of a
cell through vertex 0, so two seeds with the same vertex indexing and the same star have the same hull, index for
index, and hence the same arrangement key. The star needs Qhull's triangulation and the simplices around vertex 0
(grown over neighbours whose hyperplanes agree within the hull tolerance, as compute_convex_hull groups them) -- not
the whole hull's faces, cells, checks and classes -- so it costs a small part of a full classification. Simplices are
joined only when their hyperplanes agree to within 1/1000 of the hull tolerance and kept apart only when they differ by
1000x it; anything between is ambiguous (None: the caller falls back to the full classification), so the grouping is
the one compute_convex_hull makes under any reading of its tolerance. A star with a cell inside another (a flat sliver
of Qhull's triangulation, which compute_convex_hull folds away) is ambiguous too.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

import group as G


def star(beta, hull_tol=1e-9):
    try:
        V, keep = G.vertices(np.asarray(beta, float))
        if len(V) < 5:
            return None
        Vn = V / np.linalg.norm(V[0])
        h = ConvexHull(Vn)
    except Exception:
        return None
    eq, nb, sx = h.equations, h.neighbors, h.simplices
    at0 = np.flatnonzero((sx == 0).any(1))
    seen, cells = set(), []
    for s0 in at0:
        if s0 in seen:
            continue
        group, todo = {s0}, [s0]
        seen.add(s0)
        while todo:
            s = todo.pop()
            for t in nb[s]:
                if t in group:
                    continue
                d = np.abs(eq[s] - eq[t]).max()
                if d < hull_tol / 1000:
                    group.add(t)
                    seen.add(t)
                    todo.append(t)
                elif d < 1000 * hull_tol:
                    return None
        cells.append(frozenset(int(v) for s in group for v in sx[s]))
    if any(a < b for a in cells for b in cells):
        return None
    return len(V), keep.tobytes(), frozenset(cells)
