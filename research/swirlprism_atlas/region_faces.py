"""Whole faces for the region colouring: every curved wall's pieces (region_pieces/) grouped into the faces between
the other surfaces crossing it, and every piece given its face's sides.

The sides of each small piece were looked up separately (region_db.py), from voxel pockets a few voxels off the wall;
near a line, and near the faces of the cell, the lookup lands in the wrong pocket often enough that the colour frays
and whole patches carry a neighbour's shape. A face of the wall between the surfaces crossing it has the same two
sides throughout, so here the pieces are joined into faces: pieces next to each other on the same side of every other
surface crossing the wall, each surface carried on past where its mesh stops by its nearest triangle's plane (an
invisible line, so a wall ending part way across never lets the faces either side of it join round its end). The
faces making up most of the wall (COVER of its area) are classified once on each side, off the middle of their biggest
piece; the small rest take the sides most of their area has. The pieces' own labels are kept as "sides_piece".

    python region_faces.py [walls]          (all walls in region_pieces/ by default; then region_view.py --export)
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_walls as rw
import region_cells as rc
from region_db import Labels, load_pieces, save_pieces

COVER = 0.97          # (the share of each wall's area whose faces are classified)
STEP = 4e-4


def polygon(f):
    """A piece's polygon from its fan of triangles (all from the first corner)."""
    T = np.array(f["tris"], float).reshape(-1, 3, 3)
    return np.vstack([T[0, 0], T[0, 1], T[:, 2]])


def oriented(T):
    """Triangles T (n, 3, 3) of one meshed surface wound consistently (flipped across shared edges as needed, per
    connected patch), and their unit normals."""
    T = np.array(T, float)
    key = lambda p: tuple(np.round(p, 6))
    by_edge = defaultdict(list)
    for i, t in enumerate(T):
        for a in range(3):
            by_edge[frozenset((key(t[a]), key(t[(a + 1) % 3])))].append(i)
    flip = np.zeros(len(T), bool)
    seen = np.zeros(len(T), bool)
    for s0 in range(len(T)):
        if seen[s0]:
            continue
        seen[s0] = True
        stack = [s0]
        while stack:
            i = stack.pop()
            t = T[i][::-1] if flip[i] else T[i]
            for a in range(3):
                A, B = key(t[a]), key(t[(a + 1) % 3])
                for k in by_edge[frozenset((A, B))]:
                    if seen[k]:
                        continue
                    u = T[k]
                    # (a neighbour wound the same way runs the shared edge the other way)
                    same = any(key(u[c]) == B and key(u[(c + 1) % 3]) == A for c in range(3))
                    flip[k] = not same
                    seen[k] = True
                    stack.append(k)
    T[flip] = T[flip][:, ::-1]
    N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return T, N / np.maximum(np.linalg.norm(N, axis=1), 1e-300)[:, None]


def side_keys(C, wid, pieces, curved):
    """For each point C (the pieces' centres on wall wid): the side it lies on of every other surface crossing the
    wall, the surface carried on past where it stops by its nearest triangle's plane (the "invisible lines"). Points
    with the same key lie in the same cell of the crossing surfaces."""
    mine = np.concatenate([T for _, _, T in curved])
    cols = []
    for pid, (w, kind, idx, T) in enumerate(pieces):
        if w == wid:
            continue
        # (only surfaces that reach the wall: some triangle of theirs crosses one of its triangles)
        lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
        if (lo > mine.reshape(-1, 3).max(0) + 1e-9).any() or (hi < mine.reshape(-1, 3).min(0) - 1e-9).any():
            continue
        tree = cKDTree(T.mean(1))
        near = cKDTree(mine.mean(1)).query_ball_tree(tree, 0.03)
        if not any(rw._crosses(mine[i], T[k]) for i, ks in enumerate(near) for k in ks[:40]):
            continue
        OT, ON = oriented(T)
        k = cKDTree(OT.mean(1)).query(C, k=min(4, len(OT)))[1].reshape(len(C), -1)
        # (of the few nearest triangles, the one whose plane is nearest)
        h = np.einsum("nki,nki->nk", C[:, None] - OT[k][:, :, 0], ON[k])
        best = np.abs(h).argmin(1)
        cols.append(np.sign(h[np.arange(len(C)), best]).astype(np.int8))
    return [tuple(r) for r in np.stack(cols, 1)] if cols else [()] * len(C), len(cols)


def _segs(pg, A, d, L2, tol):
    """The edges of polygon pg lying along the line A + t d: [(t0, t1)]."""
    tp = (pg - A) @ d / L2
    on = np.linalg.norm(pg - A - tp[:, None] * d, axis=1) < tol
    out = []
    for m in range(len(pg)):
        m2 = (m + 1) % len(pg)
        if on[m] and on[m2] and abs(tp[m2] - tp[m]) > 1e-6:
            out.append((min(tp[m], tp[m2]), max(tp[m], tp[m2])))
    return out


def join(parts, piece_T, keys, tol=1e-6, share=0.25):
    """Parts joined into faces: two parts sharing a good stretch of an edge (across a mesh edge between triangles, or
    across a clipping line inside one) and lying on the same side of every crossing surface (keys) are one face.
    Returns each part's face and whether its normal is flipped against the face's."""
    parent, par = list(range(len(parts))), [False] * len(parts)

    def find(x):
        p = False
        while parent[x] != x:
            p ^= par[x]
            x = parent[x]
        return x, p

    def link(i, k):
        if keys[i] != keys[k]:
            return
        ri, pi = find(i)
        rk, pk = find(k)
        if ri != rk:
            parent[ri] = rk
            par[ri] = pi ^ pk ^ bool(parts[i][3] @ parts[k][3] < 0)

    def pairs(segs, A, d):
        for x in range(len(segs)):
            for y in range(x + 1, len(segs)):
                i, a0, a1 = segs[x]
                k, b0, b1 = segs[y]
                if i != k and min(a1, b1) - max(a0, b0) > share * min(a1 - a0, b1 - b0):
                    link(i, k)
    # (across mesh edges: the parts' edges along each triangle's edges, matched by the edge's corners)
    on_edge = defaultdict(list)
    line_of = {}
    by_tri = defaultdict(list)
    for i, (_, _, pg, n, pid, j) in enumerate(parts):
        by_tri[(pid, j)].append(i)
        t = piece_T[pid][j]
        for a in range(3):
            A, B = t[a], t[(a + 1) % 3]
            ka, kb = tuple(np.round(A, 6)), tuple(np.round(B, 6))
            if kb < ka:
                A, B, ka, kb = B, A, kb, ka
            d = B - A
            L2 = d @ d
            if L2 < 1e-20:
                continue
            line_of[(pid, ka, kb)] = (A, d)
            on_edge[(pid, ka, kb)] += [(i, t0, t1) for t0, t1 in _segs(pg, A, d, L2, tol)]
    for key, segs in on_edge.items():
        pairs(segs, *line_of[key])
    # (inside a triangle: the parts' other edges, along the clipping lines)
    for idx in by_tri.values():
        if len(idx) < 2:
            continue
        lines = []
        for i in idx:
            pg = parts[i][2]
            for m in range(len(pg)):
                A, B = pg[m], pg[(m + 1) % len(pg)]
                d = B - A
                if d @ d > 1e-20:
                    lines.append((A, d / np.linalg.norm(d)))
        done = []
        for A, u in lines:
            if any(np.linalg.norm(np.cross(A - A2, u2)) < tol and abs(u @ u2) > 1 - 1e-9 for A2, u2 in done):
                continue
            done.append((A, u))
            segs = [(i, t0, t1) for i in idx for t0, t1 in _segs(parts[i][2], A, u, 1.0, tol)]
            if len({s[0] for s in segs}) > 1:
                pairs(segs, A, u)
    out = [find(i) for i in range(len(parts))]
    return [r for r, _ in out], [p for _, p in out]


def area(pg):
    return 0.5 * np.linalg.norm(sum(np.cross(pg[m] - pg[0], pg[m + 1] - pg[0]) for m in range(1, len(pg) - 1)))


def main(walls=None):
    d = rw.page_data()
    names = {w["id"] for w in d["xwalls"]} | {"T2"}
    pieces = rw.pieces_of(d, names)
    fp = load_pieces(set(walls.split(",")) if walls else None)
    labels = Labels()
    for wid, faces in sorted(fp.items()):
        curved = [(pid, kind, T) for pid, (w, kind, idx, T) in enumerate(pieces) if w == wid and not rw.is_flat(T)]
        if not curved:
            continue
        allT = np.concatenate([T for _, _, T in curved])
        owner = np.concatenate([[(pid, j) for j in range(len(T))] for pid, _, T in curved])
        tree = cKDTree(allT.mean(1))
        parts, which = [], []
        for i, f in enumerate(faces):
            if not f["tris"]:
                continue
            pg = polygon(f)
            c = pg.mean(0)
            n = np.cross(pg[1] - pg[0], pg[2] - pg[0])
            if np.linalg.norm(n) < 1e-18:
                continue
            n /= np.linalg.norm(n)
            # (the wall triangle the piece was cut from: the near one holding its centre)
            found = None
            for k in tree.query(c, k=min(12, len(allT)))[1]:
                t = allT[k]
                M = np.c_[t[1] - t[0], t[2] - t[0]]
                uv, res, *_ = np.linalg.lstsq(M, c - t[0], rcond=None)
                off = np.linalg.norm(M @ uv - (c - t[0]))
                if off < 1e-6 and uv.min() > -1e-5 and uv.sum() < 1 + 1e-5:     # (stored corners carry 7 decimals)
                    found = k
                    break
            if found is None:
                continue                           # (a flat wall's face, or a piece not traced back: left as it is)
            pid, j = owner[found]
            parts.append((wid, f["kind"], pg, n, int(pid), int(j)))
            which.append(i)
        if not parts:
            continue
        piece_T = {pid: T for pid, _, T in curved}
        keys, n_s = side_keys(np.array([p[2].mean(0) for p in parts]), wid, pieces, curved)
        comp, flip = join(parts, piece_T, keys)
        votes = defaultdict(Counter)
        fa = defaultdict(float)
        lead = {}
        for (w, kind, pg, n, pid, j), i, c, fl in zip(parts, which, comp, flip):
            s = faces[i].get("sides_piece", faces[i]["sides"])
            a = area(pg)
            votes[c][tuple(s[::-1] if fl else s)] += a
            fa[c] += a
            if c not in lead or a > lead[c][0]:
                lead[c] = (a, pg, n, fl)
        # (the faces making up most of the wall: classified once each side, a short step off the middle of the
        # face's biggest piece; the small rest keep the sides most of their pieces have)
        big = sorted(fa, key=lambda c: -fa[c])
        cs = np.cumsum([fa[c] for c in big]) / sum(fa.values())
        big = big[:int(np.searchsorted(cs, COVER)) + 1]
        C = np.array([lead[c][1].mean(0) for c in big])
        N = np.array([lead[c][2] for c in big])
        got = labels(np.concatenate([rc.betas(C + STEP * N), rc.betas(C - STEP * N)])) if big else []
        best = {c: votes[c].most_common(1)[0][0] for c in votes}
        n_cls = 0
        for m, c in enumerate(big):
            sn = (got[m], got[len(big) + m])
            if not any(x is None or str(x).startswith(("mixed", "TIMEOUT", "ERR")) for x in sn):
                best[c] = sn[::-1] if lead[c][3] else sn          # (in the face's orientation)
                n_cls += 1
        changed = 0.0
        total = 0.0
        for (w, kind, pg, n, pid, j), i, c, fl in zip(parts, which, comp, flip):
            new = list(best[c][::-1] if fl else best[c])
            f = faces[i]
            f.setdefault("sides_piece", f["sides"])
            a = area(pg)
            total += a
            if new != f["sides_piece"]:
                changed += a
            f["sides"] = new
        print(f"{wid}: {n_s} crossing surfaces; {len(parts)} pieces in {len(set(comp))} faces ({n_cls} "
              f"classified); {changed / max(total, 1e-30):.1%} of the area relabelled to its face's sides", flush=True)
        save_pieces(fp, [wid])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
