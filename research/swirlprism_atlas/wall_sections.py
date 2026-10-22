"""Each wall's mesh cut into sections along the lines where other walls meet it, and each section's two sides named
from the region database.

A wall's mesh (the page's) already ends at the lines bounding it. Inside it, other walls meet it (meetings.py): along
a traced line, along the edge of a wall that ends on it, or where a wall passes through it. The mesh is cut along each
such wall's exact surface (arrangement.py's surfaces), across the whole mesh, so a crossing that stops part way is
carried on to the mesh's edge as an invisible line. The sections are the neighbouring triangles on the same side of
every one of those surfaces. Each section has one region on each side: the one most of the database's voxels
(region_grid.json) beside it on that side belong to, counting only voxels on the section's own side of every cutting
wall. No classification.

    python wall_sections.py X3,X7,X10,X83,T2,X11     (writes sec_pieces/<wall>.json.gz, as region_pieces/)
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from arrangement import Mesh, carriers, surfaces
from meetings import mesh_edges, sample_segs, tri_tri

OUT = "sec_pieces"
TOL = 2e-3
SLIVER = 2e-4


def weld(T, nd=7):
    """A triangle soup (n, 3, 3) as vertices and index triangles."""
    P = T.reshape(-1, 3)
    key = np.round(P, nd)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[first], inv.reshape(-1, 3)


def surface_of(T, surfs):
    """The exact surface most of the piece lies on."""
    rng = np.random.default_rng(0)
    j = rng.integers(0, len(T), 300)
    u, v = rng.random(300), rng.random(300)
    fl = u + v > 1
    u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
    V = T[j, 0] + u[:, None] * (T[j, 1] - T[j, 0]) + v[:, None] * (T[j, 2] - T[j, 0])
    best = None
    for s in surfs:
        g = np.median(np.abs(s.at(V)) / (np.linalg.norm(s.grad(V), axis=1) + 1e-30))
        if best is None or g < best[0]:
            best = (g, s)
    return best[1] if best[0] < 3e-4 else None


class Grid:
    """The region database's voxels: the region at any point (0 where none)."""

    def __init__(self):
        g = json.load(open("region_grid.json"))
        self.sh, self.lo, self.h = tuple(g["shape"]), np.array(g["lo"]), g["h"]
        self.names = [None] + sorted(g["runs"])
        lab = np.zeros(int(np.prod(self.sh)), np.int16)
        for k, r in enumerate(self.names[1:], 1):
            for a, n in g["runs"][r]:
                lab[a:a + n] = k
        self.lab = lab.reshape(self.sh)

    def at(self, X):
        i = np.floor((X - self.lo) / self.h).astype(int)
        ok = ((i >= 0) & (i < np.array(self.sh))).all(1)
        out = np.zeros(len(X), int)
        out[ok] = self.lab[tuple(i[ok].T)]
        return out


def main(walls):
    want = set(walls.split(","))
    d = rw.page_data()
    names = {w["id"] for w in d["xwalls"]} | {"T2"}
    pieces = rw.pieces_of(d, names)
    surfs = surfaces()
    carry = carriers(surfs)
    surfs = [s for s in surfs if s.name in carry]
    grid = Grid()
    # (meetings: every crossing triangle pair of two walls' meshes)
    allT = np.concatenate([T for *_, T in pieces])
    owner = np.concatenate([[pid] * len(T) for pid, (*_, T) in enumerate(pieces)])
    wall_of = np.array([pieces[p][0] for p in owner])
    C = allT.mean(1)
    R = np.linalg.norm(allT - C[:, None], axis=2).max(1)
    rs = float(np.quantile(R, 0.95))
    small = np.flatnonzero(R <= rs)
    pairs = small[cKDTree(C[small]).query_pairs(2 * rs, output_type="ndarray")]
    tree = cKDTree(C)
    extra = [(b, j) for b in np.flatnonzero(R > rs) for j in tree.query_ball_point(C[b], R[b] + rs)]
    if extra:
        pairs = np.vstack([pairs, np.array(extra)])
    pairs = pairs[wall_of[pairs[:, 0]] != wall_of[pairs[:, 1]]]
    seg, ok = tri_tri(allT[pairs[:, 0]], allT[pairs[:, 1]])
    seg, pairs = seg[ok], pairs[ok]
    os.makedirs(OUT, exist_ok=True)
    by_wall = defaultdict(list)
    for pid, (w, kind, i, T) in enumerate(pieces):
        if w not in want:
            continue
        own = sample_segs(mesh_edges(T))
        own_tree = cKDTree(own) if len(own) else None
        me = np.isin(owner[pairs[:, 0]], [pid]) | np.isin(owner[pairs[:, 1]], [pid])
        partners = set()
        for (a, b), sg in zip(pairs[me], seg[me]):
            q = owner[b] if owner[a] == pid else owner[a]
            P3 = np.stack([sg[0], sg.mean(0), sg[1]])
            if own_tree is not None and own_tree.query(P3)[0].max() <= TOL:
                continue                                  # (along this wall's own edge)
            partners.add(q)
        S_me = surface_of(T, surfs)
        cutters = []
        for q in sorted(partners):
            s = surface_of(pieces[q][3], surfs)
            if s is not None and s is not S_me and all(s is not c for c in cutters):
                cutters.append(s)
        V, F = weld(T)
        m = Mesh(V, F, S_me)
        for k, s in enumerate(cutters):
            m.cut(s, k)
        # (sections: neighbouring triangles on the same side of every cutting surface, by their furthest corner)
        Ti = np.array(m.T)
        S = np.zeros((len(Ti), len(cutters)), np.int8)
        for k, s in enumerate(cutters):
            g = s.at(m.V)[Ti]
            S[:, k] = np.sign(g[np.arange(len(Ti)), np.abs(g).argmax(1)])
        parent = list(range(len(Ti)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        by_edge = defaultdict(list)
        for t, (a, b, c) in enumerate(Ti):
            for e in ((a, b), (b, c), (c, a)):
                by_edge[(min(e), max(e))].append(t)
        for ts in by_edge.values():
            for t in ts[1:]:
                if (S[t] == S[ts[0]]).all():
                    parent[find(t)] = find(ts[0])
        comp = np.array([find(t) for t in range(len(Ti))])
        Tri = m.V[Ti]
        A = 0.5 * np.linalg.norm(np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0]), axis=1)
        N = np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0])
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-300
        rng = np.random.default_rng(1)
        faces = []
        for c in np.unique(comp):
            idx = np.flatnonzero(comp == c)
            a = A[idx]
            if a.sum() < 1e-12:
                continue
            # (points spread over the section, and the voxels a few voxels off it on each side)
            j = idx[rng.choice(len(idx), min(400, 20 * len(idx)), p=a / a.sum())]
            u, v = rng.random(len(j)), rng.random(len(j))
            fl = u + v > 1
            u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
            X = Tri[j, 0] + u[:, None] * (Tri[j, 1] - Tri[j, 0]) + v[:, None] * (Tri[j, 2] - Tri[j, 0])
            n = N[j] * np.sign(N[j] @ N[idx[a.argmax()]])[:, None]           # (one orientation over the section)
            sides = []
            sec_sign = S[idx[a.argmax()]]
            for sgn in (1, -1):
                votes = Counter()
                for k in (1.5, 2.5, 3.5, 4.5):
                    Y = X + sgn * k * grid.h * n
                    lab = grid.at(Y)
                    # (a voxel counts only on the section's own side of every wall cutting this one: a vote never
                    # reaches across a wall the section ends on)
                    same = np.ones(len(Y), bool)
                    for kk, cs in enumerate(cutters):
                        if sec_sign[kk]:
                            same &= np.sign(cs.at(Y)) == sec_sign[kk]
                    votes.update(int(x) for x in lab[same] if x)
                if not votes:
                    # (a side past the half-cell's faces: the same places brought into the half-cell by the group,
                    # as the database names them)
                    from region_db import home
                    for k in (1.5, 2.5, 3.5, 4.5):
                        Bh = home(rc.betas(X + sgn * k * grid.h * n))
                        okh = ~np.isnan(Bh).any(1)
                        votes.update(int(x) for x in grid.at(rc.xyz(Bh[okh])) if x)
                sides.append(grid.names[votes.most_common(1)[0][0]] if votes else "unknown")
            if sides[0] == sides[1]:
                continue
            # (a section never more than SLIVER from a wall cutting it is this mesh's edge poking past that wall, where
            # it ends on it: not a part of the wall)
            V_ = np.concatenate([Tri[idx].mean(1), (Tri[idx] + Tri[idx][:, [1, 2, 0]]).reshape(-1, 3) / 2])   # (inside the triangles,
            # not their corners, which may lie on the cutting wall where the section is a large flat triangle)
            if any((np.abs(cs.at(V_)) / (np.linalg.norm(cs.grad(V_), axis=1) + 1e-30)).max() < SLIVER for cs in cutters):
                continue
            nn = N[idx[a.argmax()]]
            cc = Tri[idx[a.argmax()]].mean(0)
            faces.append({"sides": sides, "kind": kind, "mirror": False,
                          "cut": [bool(~rc.inside(rc.betas((cc + sg_ * 4e-4 * nn)[None]))[0]) for sg_ in (1, -1)],
                          "tris": Tri[idx].reshape(-1, 9).round(7).tolist()})
        by_wall[w] += faces
        print(f"  {w} ({kind}): cut along {len(cutters)} surfaces ({', '.join(c.name for c in cutters)}); "
              f"{len(np.unique(comp))} sections, {len(faces)} of them between two regions", flush=True)
    for w, faces in by_wall.items():
        json.dump(faces, gzip.open(os.path.join(OUT, f"{w}.json.gz"), "wt"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
