"""Combinatorial fingerprints of the polytopes in the atlas, to spot types that are the same polytope.

The swirlprism group G can sit inside a polytope's full symmetry group in more than one way, so the same
polytope can appear at different places in the cell with different class counts under G. Class counts
then differ, but the polytope (its face lattice) does not. The fingerprint here ignores G: it is colour
refinement (Weisfeiler-Lehman) on the incidence graph of vertices, edges, faces and cells, plus the f-vector
and the cell types. Equal fingerprints mean the face lattices cannot be told apart by refinement; congruence
is then checked separately.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from cellframe import seed_from_beta

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull

G = named_symmetry("h4_swirlprism")


def lattice(beta):
    verts = generate_vertices_from_seed(seed_from_beta(np.asarray(beta, float)), G, tol=1e-9)
    faces, cells = compute_convex_hull(verts)
    edges = sorted(hull_edges(faces))
    return verts, edges, faces, cells


def fingerprint(beta, rounds: int = 8) -> dict:
    verts, edges, faces, cells = lattice(beta)
    nv, ne, nf, nc = len(verts), len(edges), len(faces), len(cells)
    edge_index = {e: i for i, e in enumerate(edges)}
    # incidence graph: nodes are (rank, index); rank 0 vertices, 1 edges, 2 faces, 3 cells
    nbr: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def link(a, b):
        nbr[a].append(b)
        nbr[b].append(a)

    for i, (u, w) in enumerate(edges):
        link((1, i), (0, u))
        link((1, i), (0, w))
    for f, face in enumerate(faces):
        for u, w in zip(face, face[1:] + face[:1]):
            link((2, f), (1, edge_index[(min(u, w), max(u, w))]))
    for c, cell in enumerate(cells):
        for f in cell:
            link((3, c), (2, f))
    colour = {node: str(node[0]) for node in nbr}
    for _ in range(rounds):
        new = {}
        for node, ns in nbr.items():
            sig = colour[node] + "|" + ",".join(sorted(colour[n] for n in ns))
            new[node] = hashlib.sha1(sig.encode()).hexdigest()[:16]
        if len(set(new.values())) == len(set(colour.values())):
            colour = new
            break
        colour = new
    wl = hashlib.sha1(json.dumps(sorted(Counter(colour.values()).items())).encode()).hexdigest()[:12]
    cell_types = Counter()
    for cell in cells:
        sizes = tuple(sorted(len(faces[f]) for f in cell))
        cv = len({v for f in cell for v in faces[f]})
        cell_types[(cv, sizes)] += 1
    face_sizes = Counter(len(f) for f in faces)
    return {"f": [nv, ne, nf, nc], "faces": sorted(face_sizes.items()),
            "cells": sorted([[k[0], list(k[1]), v] for k, v in cell_types.items()]), "wl": wl}


def _job(item):
    tid, beta = item
    try:
        return tid, beta, fingerprint(beta)
    except Exception as exc:
        return tid, beta, {"error": repr(exc)}


if __name__ == "__main__":
    from cell_atlas2 import identify, load_refs
    refs = load_refs()[0]
    xids = json.load(open("atlas_xids.json"))
    samples = json.load(open("golden_22.json"))
    per_type = defaultdict(list)
    for s in samples:
        cid, xkey, _ = identify(s["sig"], refs)
        tid = cid or xids.get(xkey)
        if tid and len(per_type[tid]) < int(sys.argv[1] if len(sys.argv) > 1 else 3):
            if not any(np.allclose(s["beta"], b, atol=1e-12) for b in per_type[tid]):
                per_type[tid].append(s["beta"])
    jobs = [(t, b) for t, bs in per_type.items() for b in bs]
    with Pool(4) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = defaultdict(list)
    for tid, beta, fp in res:
        out[tid].append({"beta": beta, **fp})
    json.dump(out, open("fingerprints.json", "w"), indent=0)
    print(len(jobs), "polytopes fingerprinted")


def distance_profile(beta, k: int = 40) -> np.ndarray:
    """Sorted distances from one vertex to its k nearest others, scaled by the circumradius."""
    verts = generate_vertices_from_seed(seed_from_beta(np.asarray(beta, float)), G, tol=1e-9)
    d = np.sort(np.linalg.norm(verts - verts[0], axis=1))[1:k + 1]
    return d / np.linalg.norm(verts[0])


def congruent(beta_a, beta_b, tol: float = 1e-7) -> bool:
    """Same vertex count and the same full set of pairwise distances (up to scale)."""
    va = generate_vertices_from_seed(seed_from_beta(np.asarray(beta_a, float)), G, tol=1e-9)
    vb = generate_vertices_from_seed(seed_from_beta(np.asarray(beta_b, float)), G, tol=1e-9)
    if len(va) != len(vb):
        return False
    da = np.sort(np.linalg.norm(va - va[0], axis=1)) / np.linalg.norm(va[0])
    db = np.sort(np.linalg.norm(vb - vb[0], axis=1)) / np.linalg.norm(vb[0])
    return bool(np.abs(da - db).max() < tol)
