"""Name the polytope a seed makes under h4_swirlprism.

A port of the atlas branch's ``classify_fast`` (research/swirlprism_atlas): the orbit's convex hull
is built only around vertex 0, and the group sorts its cells, faces and edges into symmetry classes.
The result is the signature the atlas uses (vertex count, class sizes of cells / faces / edges, and
edge valences). Names come from the atlas's own shape list: each named shape's example seed is
classified once, and its signature is the key.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from ..generation import generate_vertices_from_seed
from ..library import named_symmetry
from .frame import SYMMETRY, group, load_json, point_from_beta

ACTION = named_symmetry(SYMMETRY)


def _check_cells(verts: np.ndarray, cells_v: list[frozenset[int]], eps: float) -> None:
    bad = "Convex hull is ambiguous at this tolerance"
    for cell in cells_v:
        members = np.fromiter(cell, dtype=int)
        pts = verts[members]
        mid = pts.mean(axis=0)
        normal = np.linalg.svd(pts - mid)[2][3]
        dist = (verts - mid) @ normal
        if dist[members].max() > eps or dist[members].min() < -eps:
            raise ValueError(f"{bad} (a cell is not flat).")
        if dist.max() > eps and dist.min() < -eps:
            raise ValueError(f"{bad} (vertices lie on both sides of a cell).")
        if dist.max() > eps:
            dist = -dist
        others = np.setdiff1d(np.flatnonzero(dist >= -eps * 1e-3), members)
        if len(others):
            gaps = np.min(np.linalg.norm(verts[others][:, None] - pts[None], axis=2), axis=1)
            if gaps.max() > eps:
                raise ValueError(f"{bad} (a vertex lies on a cell it is not part of).")


def classify(seed: np.ndarray, hull_tol: float = 1e-9) -> dict:
    """Vertex count, class sizes of cells / faces / edges, and edge valences of the seed's hull."""
    verts = generate_vertices_from_seed(np.asarray(seed, float), ACTION, tol=1e-9)
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
    simplices, neighbours = hull.simplices, hull.neighbors

    def same_plane(i, j):
        d = np.abs(eqs[i] - eqs[j])
        return d[:4].max() <= tol and d[4] <= eps

    owner: dict[int, int] = {}
    groups: dict[int, list[int]] = {}

    def grow(start):
        owner[start] = start
        queue, members = [start], [start]
        while queue:
            c = queue.pop()
            for n in neighbours[c]:
                n = int(n)
                if n not in owner and same_plane(start, n):
                    owner[n] = start
                    queue.append(n)
                    members.append(n)
        groups[start] = members

    for s in np.flatnonzero((simplices == 0).any(axis=1)):
        if int(s) not in owner:
            grow(int(s))
    for g in list(groups):
        for c in groups[g]:
            for n in neighbours[c]:
                if int(n) not in owner:
                    grow(int(n))
    vs = {g: set(simplices[m].ravel().tolist()) for g, m in groups.items()}
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
                for n in neighbours[c]:
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
    faces: dict[tuple[int, int], list[int]] = {}
    for g in cells:
        for c in groups[g]:
            for n in neighbours[c]:
                h = cell_of.get(int(n))
                if h is not None and h != g and h in cells:
                    key = (min(g, h), max(g, h))
                    if key not in faces:
                        shared = sorted(cells[g] & cells[h])
                        if len(shared) < 3:
                            raise ValueError("a face has fewer than 3 vertices")
                        faces[key] = shared
    edge_cells: dict[frozenset, set] = defaultdict(set)
    cell_edge_uses: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    face_sets = []
    face_edges: dict[frozenset, list] = defaultdict(list)
    for (g, h), shared in faces.items():
        pts = verts[shared]
        mid = pts.mean(axis=0)
        vh = np.linalg.svd(pts - mid)[2]
        pl = (pts - mid) @ vh[:2].T
        ring = [shared[int(i)] for i in np.argsort(np.arctan2(pl[:, 1], pl[:, 0]))]
        k = ring.index(0)
        for w in (ring[k - 1], ring[(k + 1) % len(ring)]):
            e = frozenset((0, w))
            face_edges[frozenset(shared)].append(e)
            edge_cells[e] |= {g, h}
            cell_edge_uses[g][e] += 1
            cell_edge_uses[h][e] += 1
        face_sets.append(frozenset(shared))
    for g in cells:
        if len(cell_edge_uses[g]) < 3 or any(k != 2 for k in cell_edge_uses[g].values()):
            raise ValueError("a cell is not closed")
    elements = group()
    tree = cKDTree(verts)
    perm_cache: dict[int, np.ndarray] = {}

    def perm(gi):
        if gi not in perm_cache:
            d, idx = tree.query(verts @ elements[gi].T)
            if d.max() > 1e-6:
                raise ValueError("orbit not closed")
            perm_cache[gi] = idx
        return perm_cache[gi]

    to0: dict[int, np.ndarray] = {}

    def senders(u):
        if u not in to0:
            to0[u] = np.flatnonzero(np.linalg.norm(elements @ verts[u] - verts[0], axis=1) < 1e-10)
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
        cls: dict[int, list[int]] = defaultdict(list)
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

    cell_list, edge_list = list(cells.values()), list(edge_cells)
    co, fo, eo = orbits(cell_list), orbits(face_sets), orbits(edge_list)
    valence: dict[int, int] = defaultdict(int)
    for size, members in eo:
        valence[len(edge_cells[edge_list[members[0]]])] += size
    return {
        "vertices": len(verts),
        "cells": sorted(s for s, _ in co),
        "faces": sorted(s for s, _ in fo),
        "edges": sorted(s for s, _ in eo),
        "valence": dict(sorted(valence.items())),
        "vfig": _vertex_key(cells, cell_list, face_sets, edge_list, faces, face_edges,
                            edge_cells, co, fo, eo),
    }


def _vertex_key(cells, cell_list, face_sets, edge_list, faces, face_edges, edge_cells, co, fo, eo):
    """How the cells, faces and edges fit together at vertex 0 (every vertex is alike): each labelled
    by its size and orbit, refined four times over its neighbours, hashed. Two polytopes with the
    same counts can differ here. The same key as the atlas's classify_fast (``vfig``)."""
    orbit_of = {}
    for kind, orb in (("c", co), ("f", fo), ("e", eo)):
        for size, members in orb:
            for i in members:
                orbit_of[(kind, i)] = size
    lab = {("c", i): f"c{len(x)}:{orbit_of[('c', i)]}" for i, x in enumerate(cell_list)}
    lab.update({("f", i): f"f{len(x)}:{orbit_of[('f', i)]}" for i, x in enumerate(face_sets)})
    lab.update({("e", i): f"e{len(edge_cells[x])}:{orbit_of[('e', i)]}"
                for i, x in enumerate(edge_list)})
    nbr = defaultdict(list)
    cell_ix = {g: i for i, g in enumerate(cells)}
    face_ix = {x: i for i, x in enumerate(face_sets)}
    edge_ix = {x: i for i, x in enumerate(edge_list)}
    for (g, h), shared in faces.items():
        f_ = ("f", face_ix[frozenset(shared)])
        for c_ in (("c", cell_ix[g]), ("c", cell_ix[h])):
            nbr[f_].append(c_)
            nbr[c_].append(f_)
        for e in face_edges[frozenset(shared)]:
            e_ = ("e", edge_ix[e])
            nbr[f_].append(e_)
            nbr[e_].append(f_)
    for _ in range(4):
        lab = {n: hashlib.sha1((lab[n] + "|" + ",".join(sorted(lab[m] for m in nbr[n])))
                               .encode()).hexdigest()[:16] for n in lab}
    return hashlib.sha1(",".join(sorted(lab.values())).encode()).hexdigest()[:12]


def signature(info: dict) -> str:
    def j(xs):
        return "+".join(map(str, xs))

    val = ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    return (f"{info['vertices']}: cells {j(info['cells'])} | faces {j(info['faces'])} | "
            f"edges {j(info['edges'])} | val {val}")


def key_of_beta(beta: np.ndarray, hull_tol: float = 1e-9) -> tuple[str, str | None]:
    """(signature, vertex key) of the polytope a seed (betas) makes."""
    try:
        info = classify(point_from_beta(np.asarray(beta, float)), hull_tol=hull_tol)
        return signature(info), info["vfig"]
    except Exception as exc:  # noqa: BLE001
        return f"ERR {type(exc).__name__}: {exc}", None


@lru_cache(maxsize=1)
def splits() -> dict[str, dict[str, str]]:
    """Names covering more than one polytope (source/shape_splits.json): name -> {key: part name}."""
    try:
        data = load_json("shape_splits.json")
    except FileNotFoundError:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_")}


def split_parent(name: str) -> str:
    """The name a split part was split from (itself if it is not a part)."""
    for parent, parts in splits().items():
        if name in parts.values():
            return parent
    return name


def split_name(name: str, key: str | None) -> str:
    """The part of a split name a polytope with this vertex key goes by."""
    parent = split_parent(name)
    return splits().get(parent, {}).get(key, name) if key else name


def signature_of_beta(beta: np.ndarray, hull_tol: float = 1e-9) -> str:
    try:
        return signature(classify(point_from_beta(np.asarray(beta, float)), hull_tol=hull_tol))
    except Exception as exc:  # noqa: BLE001 - degenerate hulls are reported, not raised
        return f"ERR {type(exc).__name__}: {exc}"


@lru_cache(maxsize=1)
def shape_signatures() -> dict[str, str]:
    """Signature of every named shape, from its example seed in the working data."""
    shapes = load_json("working_data.json")["shapes"]
    out = {}
    for name, shape in shapes.items():
        tol = 1e-9 if shape["dim"] == "region" else 1e-7
        out[name] = signature_of_beta(np.asarray(shape["seed"], float), hull_tol=tol)
    return out


def counts_match(name: str, sig: str) -> bool:
    """Whether a signature has the vertex / cell / face / edge counts the working data records."""
    shape = load_json("working_data.json")["shapes"][name]
    parts = dict(p.strip().split(" ", 1) for p in sig.split(": ", 1)[1].split(" | ") if " " in p.strip())
    want = {"cells": shape["cells"], "faces": shape["faces"], "edges": shape["edges"]}
    head = sig.split(":", 1)[0]
    norm = lambda s: "+".join(sorted(s.split("+"), key=int))  # noqa: E731
    return head == str(shape["vertices"]) and all(norm(parts[k]) == norm(v) for k, v in want.items())
