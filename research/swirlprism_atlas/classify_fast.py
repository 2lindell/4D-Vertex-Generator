"""classify.classify from one vertex: the same info (orbit sizes of cells, faces and edges, edge valences) at a
fraction of the cost.

The polytope is the orbit of one seed, so every vertex is like every other: each orbit of cells, faces or edges has
members at vertex 0, and an orbit's size follows from how many it has there (counting (item, vertex) pairs:
size * |item| = |vertices| * members at vertex 0). So only the hull's cells at vertex 0 are built (the Qhull
simplices around it, grouped into cells by the same rule as off.compute_convex_hull), with the faces and edges
there, and the group sorts them into orbits.

    python classify_fast.py check [n]        (compares with classify.classify at n points of the label cache)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from classify import ACTION
from four_d_vertex_generator.generation import generate_vertices_from_seed, group_elements
from four_d_vertex_generator.off import _check_cells

_E = None


def _elements():
    global _E
    if _E is None:
        _E = np.stack(group_elements(ACTION))
    return _E


def classify(seed: np.ndarray, hull_tol: float = 1e-9) -> dict:
    verts = generate_vertices_from_seed(seed, ACTION, tol=1e-9)
    if len(verts) < 5:
        raise ValueError("fewer than 5 vertices")
    try:
        hull = ConvexHull(verts)
    except QhullError as err:
        raise ValueError("degenerate") from err
    tol = hull_tol
    radius = float(np.max(np.linalg.norm(verts - verts.mean(axis=0), axis=1)))
    eps = tol * max(radius, 1e-300)
    eqs = hull.equations / np.linalg.norm(hull.equations[:, :4], axis=1)[:, None]
    S, NB = hull.simplices, hull.neighbors

    def same_plane(i, j):
        d = np.abs(eqs[i] - eqs[j])
        return d[:4].max() <= tol and d[4] <= eps

    # cells: simplices grouped from a founder by its hyperplane (as compute_convex_hull), for the simplices at
    # vertex 0 and one ring past them (for the sliver fold below)
    owner = {}
    groups = {}

    def grow(start):
        owner[start] = start
        q, members = [start], [start]
        while q:
            c = q.pop()
            for n in NB[c]:
                n = int(n)
                if n not in owner and same_plane(start, n):
                    owner[n] = start
                    q.append(n)
                    members.append(n)
        groups[start] = members

    at0 = np.flatnonzero((S == 0).any(axis=1))
    for s in at0:
        if int(s) not in owner:
            grow(int(s))
    first = list(groups)
    for g in first:
        for c in groups[g]:
            for n in NB[c]:
                if int(n) not in owner:
                    grow(int(n))
    vs = {g: set(S[m].ravel().tolist()) for g, m in groups.items()}
    # (a sliver group whose vertices all lie in a neighbouring group belongs to it)
    root = {g: g for g in groups}

    def find(g):
        while root[g] != g:
            root[g] = root[root[g]]
            g = root[g]
        return g
    changed = True
    while changed:
        changed = False
        for g in list(groups):
            if find(g) != g:
                continue
            for c in groups[g]:
                for n in NB[c]:
                    h = owner.get(int(n))
                    if h is None:
                        continue
                    a, b = find(g), find(h)
                    if a != b and (vs[a] <= vs[b] or vs[b] <= vs[a]):
                        small, big = (a, b) if len(vs[a]) <= len(vs[b]) else (b, a)
                        root[small] = big
                        vs[big] |= vs.pop(small)
                        groups[big] = groups[big] + groups[small]
                        changed = True
    cell_of = {s: find(g) for s, g in owner.items()}
    cells = {g: frozenset(vs[g]) for g in {find(g) for g in groups} if 0 in vs[g]}
    _check_cells(verts, list(cells.values()), eps)
    # faces at vertex 0: two cells there meeting across a simplex facet
    faces = {}
    for g in cells:
        for c in groups[g]:
            for n in NB[c]:
                h = cell_of.get(int(n))
                if h is not None and h != g and h in cells:
                    key = (min(g, h), max(g, h))
                    if key not in faces:
                        shared = sorted(cells[g] & cells[h])
                        if len(shared) < 3:
                            raise ValueError("a face has fewer than 3 vertices")
                        faces[key] = shared
    # edges at vertex 0: its two neighbours round each face there
    edge_cells = defaultdict(set)
    cell_edge_uses = defaultdict(lambda: defaultdict(int))
    face_sets = []
    for (g, h), shared in faces.items():
        pts = verts[shared]
        mid = pts.mean(axis=0)
        vh = np.linalg.svd(pts - mid)[2]
        pl = (pts - mid) @ vh[:2].T
        ring = [shared[int(i)] for i in np.argsort(np.arctan2(pl[:, 1], pl[:, 0]))]
        k = ring.index(0)
        for w in (ring[k - 1], ring[(k + 1) % len(ring)]):
            e = frozenset((0, w))
            edge_cells[e] |= {g, h}
            cell_edge_uses[g][e] += 1
            cell_edge_uses[h][e] += 1
        face_sets.append(frozenset(shared))
    for g in cells:                                    # (each cell closes up round vertex 0)
        if len(cell_edge_uses[g]) < 3 or any(k != 2 for k in cell_edge_uses[g].values()):
            raise ValueError("a cell is not closed")
    # orbits: an image of an item at vertex 0 that is again at vertex 0 is g.X with g sending some u in X to 0
    E = _elements()
    tree = cKDTree(verts)
    perm_cache = {}

    def perm(gi):
        if gi not in perm_cache:
            d, idx = tree.query(verts @ E[gi].T)
            if d.max() > 1e-6:
                raise ValueError("orbit not closed")
            perm_cache[gi] = idx
        return perm_cache[gi]
    to0 = {}

    def senders(u):
        if u not in to0:
            to0[u] = np.flatnonzero(np.linalg.norm(E @ verts[u] - verts[0], axis=1) < 1e-10)   # (near-twin vertices sit ~1e-7 apart)
        return to0[u]

    def orbits(items):
        index = {x: i for i, x in enumerate(items)}
        parent = list(range(len(items)))

        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i, x in enumerate(items):
            for u in x:
                for gi in senders(u):
                    p = perm(int(gi))
                    j = index.get(frozenset(int(p[v]) for v in x))
                    if j is None:
                        raise ValueError("hull not symmetric (merge tolerance issue)")
                    a, b = f(i), f(j)
                    if a != b:
                        parent[a] = b
        cls = defaultdict(list)
        for i in range(len(items)):
            cls[f(i)].append(i)
        out = []
        for members in cls.values():
            n = len(verts) * len(members)
            size = len(items[members[0]])
            if n % size:
                raise ValueError("orbit count not whole")
            out.append((n // size, members))
        return out
    cell_list = list(cells.values())
    edge_list = list(edge_cells)
    co, fo, eo = orbits(cell_list), orbits(face_sets), orbits(edge_list)
    valence = defaultdict(int)
    for size, members in eo:
        valence[len(edge_cells[edge_list[members[0]]])] += size
    return {
        "vertices": len(verts),
        "cells": sorted(s for s, _ in co),
        "faces": sorted(s for s, _ in fo),
        "edges": sorted(s for s, _ in eo),
        "valence": dict(sorted(valence.items())),
    }


def _sig(seed, fn):
    from classify import signature
    try:
        info = fn(seed)
        return signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    except Exception as exc:
        return "ERR " + type(exc).__name__


def _pair(b):
    import time
    import classify as slow
    from cellframe import seed_from_beta
    s = seed_from_beta(np.asarray(b, float))
    t = time.time()
    a = _sig(s, slow.classify)
    t1 = time.time()
    f = _sig(s, classify)
    t2 = time.time()
    return a, f, t1 - t, t2 - t1


if __name__ == "__main__" and sys.argv[1] == "check":
    from multiprocessing import Pool
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rows = [json.loads(l) for l in open("region_walls_cache.jsonl")]
    rng = np.random.default_rng(1)
    pick = rng.choice(len(rows), n, replace=False)
    B = [[float(x) for x in rows[i][0].split(",")] for i in pick]
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
        res = pool.map(_pair, B)
    bad = [(b, a, f) for b, (a, f, _, _) in zip(B, res) if a != f]
    ts, tf = sum(r[2] for r in res), sum(r[3] for r in res)
    print(f"{n} points: {n - len(bad)} agree; slow {ts / n:.2f}s, fast {tf / n:.3f}s a point")
    for b, a, f in bad[:10]:
        print("  ", np.round(b, 6).tolist(), "\n     slow", a[:160], "\n     fast", f[:160])


if __name__ == "__main__" and sys.argv[1] == "check_shapes":
    # points on walls, lines and points too (the page's samples of every shape), where hulls are degenerate
    from multiprocessing import Pool
    import region_walls as rw
    import region_cells as rc
    d = rw.page_data()
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rng = np.random.default_rng(2)
    B, names = [], []
    for t, pts in d["samples"].items():
        for i in rng.permutation(len(pts))[:per]:
            B.append(rc.betas((np.array(pts[i][:3]) * rw.FLIP)[None])[0].tolist())
            names.append(t)
    with Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 2) as pool:
        res = pool.map(_pair, B)
    bad = [(t, b, a, f) for t, b, (a, f, _, _) in zip(names, B, res) if a != f]
    print(f"{len(B)} shape samples: {len(B) - len(bad)} agree")
    for t, b, a, f in bad[:15]:
        print("  ", t, np.round(b, 6).tolist(), "\n     slow", a[:160], "\n     fast", f[:160])
