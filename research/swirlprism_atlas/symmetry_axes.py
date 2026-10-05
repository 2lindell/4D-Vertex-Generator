"""Every great circle in the half-cell where the polytope gains symmetry beyond the swirlprism group.

Candidates are the fixed circles of the elements outside the group of: H4 (rotations and reflections), its coset
by the extra half-turn Q, and the 3600-element group of the green girdle. Each circle's pieces in the half-cell
are found, and the polytope's symmetry is counted at a point on each piece (every element of those sets that maps
the 1200 vertices onto themselves), and at the piece's neighbourhood off the circle for comparison. Writes
symmetry_axes.json.

    python symmetry_axes.py
"""
from __future__ import annotations

import json

import numpy as np
from cellframe import TINV, seed_from_beta
from dodeca_view import E, clip_images
from normalizer import halfturn
from scipy.spatial import cKDTree
from supergroups import supergroup

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

H4 = np.stack(group_elements(named_symmetry("h4_icosian")))
Q = halfturn()
G3 = np.stack(supergroup(3))
CAND = np.concatenate([H4, H4 @ Q, G3])
SOURCE = ["H4"] * len(H4) + ["H4.Q"] * len(H4) + ["girdle group"] * len(G3)
H_CELL = [TINV[i] for i in range(4)] + [TINV[2] - TINV[3]]
_G = cKDTree(E.reshape(len(E), -1))


def in_G(k):
    return _G.query(k.ravel())[0] < 1e-8


def symmetries(x):
    """Elements of the candidate sets mapping the vertex set G.x onto itself (counted once each)."""
    V = E @ x
    kd = cKDTree(V)
    first = kd.query(CAND @ x)[0] < 1e-8
    keys = set()
    for k in CAND[first]:
        if np.all(kd.query(V @ k.T)[0] < 1e-8):
            keys.add(tuple(np.round(k, 7).ravel()))
    return len(keys)


def circles():
    """Distinct fixed 2-planes of candidate elements outside the group (as projector keys), with their source."""
    out = {}
    for k, src in zip(CAND, SOURCE):
        if in_G(k):
            continue
        u, s, vt = np.linalg.svd(k - np.eye(4))
        if np.sum(s > 1e-9) != 2:
            continue
        F = vt[2:]
        key = tuple(np.round(F.T @ F, 6).ravel())
        out.setdefault(key, (F, set()))[1].add(src)
    return list(out.values())


def beta(x):
    b = TINV @ x
    return b / b.sum()


def main():
    cs = circles()
    print(len(cs), "fixed circles", flush=True)
    pieces, seen = [], set()
    for F, srcs in cs:
        th = np.linspace(0, 2 * np.pi, 1441)
        X = np.outer(np.cos(th), F[0]) + np.outer(np.sin(th), F[1])
        for pc in clip_images(X, H_CELL, elements=np.eye(4)[None]):
            a, b = beta(pc[0]), beta(pc[-1])
            if np.linalg.norm(a - b) < 1e-6:
                continue
            key = frozenset([tuple(np.round(a, 6)), tuple(np.round(b, 6))])
            if key in seen:
                continue
            seen.add(key)
            pieces.append((pc, sorted(srcs)))
    print(len(pieces), "pieces in the half-cell", flush=True)
    rng = np.random.default_rng(5)
    out = []
    for pc, srcs in pieces:
        m = pc[len(pc) // 3]
        m = m / np.linalg.norm(m)
        on = symmetries(m)
        off = max(symmetries((m + 1e-3 * d) / np.linalg.norm(m + 1e-3 * d)) for d in rng.normal(size=(3, 4)))
        out.append({"a": beta(pc[0]).tolist(), "b": beta(pc[-1]).tolist(), "sources": srcs, "on": on, "off": off,
                    "seeds": [p.tolist() for p in pc[:: max(1, len(pc) // 24)]] + [pc[-1].tolist()]})
        print(np.round(out[-1]["a"], 4).tolist(), "->", np.round(out[-1]["b"], 4).tolist(), srcs, "on", on, "near", off, flush=True)
    json.dump(out, open("symmetry_axes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
