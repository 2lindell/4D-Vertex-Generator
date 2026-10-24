"""Where two walls meet away from the lines already defined: the invisible lines a wall's mesh still needs.

Each wall's mesh (the page's, original and copies) already ends at its defined lines. Two walls' meshes are intersected
triangle by triangle; a piece of their intersection is already defined when it lies along a traced line of the atlas,
or along either mesh's own edge (where a wall ends on another). What is left are the meetings not yet defined: a wall
passing through another, or ending part way across it. Each is a curve on both walls' exact surfaces (arrangement.py
lists them, with their equations), so its equations are the two surfaces'.

    python meetings.py [walls]           (all walls by default; writes meetings.json)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_walls as rw

FLIP = np.array([1, 1, -1.0])
TOL = 2e-3                    # (a meeting within this of a defined line or mesh edge is that line)


def tri_tri(T, U):
    """Intersection segments of triangle pairs T[i], U[i] (n, 3, 3): (n, 2, 3) and whether they meet."""
    n = len(T)
    nT = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nU = np.cross(U[:, 1] - U[:, 0], U[:, 2] - U[:, 0])
    dU = np.einsum("nki,ni->nk", U - T[:, :1], nT)          # (U's corners against T's plane)
    dT = np.einsum("nki,ni->nk", T - U[:, :1], nU)
    ok = ((dU.max(1) > 0) & (dU.min(1) < 0)) & ((dT.max(1) > 0) & (dT.min(1) < 0))

    def chord(P, d):
        """Where triangle P crosses the other's plane: the two points on its edges with d = 0."""
        out = np.zeros((len(P), 2, 3))
        cnt = np.zeros(len(P), int)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            s = (d[:, a] > 0) != (d[:, b] > 0)
            t = d[:, a] / np.where(s, d[:, a] - d[:, b], 1)
            p = P[:, a] + t[:, None] * (P[:, b] - P[:, a])
            for k in (0, 1):
                m = s & (cnt == k)
                out[m, k] = p[m]
            cnt += s
        return out
    A, B = chord(T, dT), chord(U, dU)
    L = np.cross(nT, nU)
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-300
    ta, tb = np.einsum("nki,ni->nk", A, L), np.einsum("nki,ni->nk", B, L)
    a0, a1 = ta.min(1), ta.max(1)
    b0, b1 = tb.min(1), tb.max(1)
    lo, hi = np.maximum(a0, b0), np.minimum(a1, b1)
    ok &= hi > lo
    base = A[:, 0] - ta[:, :1] * L
    seg = np.stack([base + lo[:, None] * L, base + hi[:, None] * L], 1)
    return seg, ok


def mesh_edges(T):
    """A mesh's boundary edges (used by one triangle only), as segments."""
    key = lambda p: tuple(np.round(p, 6))
    cnt = defaultdict(list)
    for t in T:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            cnt[frozenset((key(t[a]), key(t[b])))].append((t[a], t[b]))
    return np.array([v[0] for v in cnt.values() if len(v) == 1]).reshape(-1, 2, 3)


def sample_segs(S, h=5e-4):
    P = []
    for a, b in S:
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / h)) + 1)
        P.append(a + np.linspace(0, 1, n)[:, None] * (b - a))
    return np.concatenate(P) if P else np.zeros((0, 3))


def main(only=None):
    d = rw.page_data()
    names = {w["id"] for w in d["xwalls"]} | {"T2"}
    pieces = rw.pieces_of(d, names)
    # (the traced lines, each sampled, with its name; and every piece's own mesh edges)
    traced = []
    for l in list(d["xlines"]) + list(d.get("segments", [])):
        p = np.array(l["pts"], float) * FLIP
        if len(p) > 1:
            traced.append((l["id"], sample_segs(np.stack([p[:-1], p[1:]], 1))))
    T_pts = np.concatenate([p for _, p in traced])
    T_name = np.concatenate([[n] * len(p) for n, p in traced])
    t_tree = cKDTree(T_pts)
    edge_tree = {}
    for pid, (w, kind, i, T) in enumerate(pieces):
        E = sample_segs(mesh_edges(T))
        edge_tree[pid] = cKDTree(E) if len(E) else None
    allT = np.concatenate([T for *_, T in pieces])
    owner = np.concatenate([[pid] * len(T) for pid, (*_, T) in enumerate(pieces)])
    wall_of = np.array([pieces[p][0] for p in owner])
    C = allT.mean(1)
    R = np.linalg.norm(allT - C[:, None], axis=2).max(1)
    rs = float(np.quantile(R, 0.95))
    small = np.flatnonzero(R <= rs)
    pairs = cKDTree(C[small]).query_pairs(2 * rs, output_type="ndarray")
    pairs = small[pairs]
    tree = cKDTree(C)
    extra = [(b, j) for b in np.flatnonzero(R > rs) for j in tree.query_ball_point(C[b], R[b] + rs)]
    if extra:
        pairs = np.vstack([pairs, np.array(extra)])
    pairs = pairs[wall_of[pairs[:, 0]] != wall_of[pairs[:, 1]]]
    if only:
        pairs = pairs[np.isin(wall_of[pairs[:, 0]], only) | np.isin(wall_of[pairs[:, 1]], only)]
    print(f"{len(pieces)} wall pieces, {len(allT)} triangles; {len(pairs)} close pairs from different walls", flush=True)
    seg, ok = tri_tri(allT[pairs[:, 0]], allT[pairs[:, 1]])
    seg, pr = seg[ok], pairs[ok]
    print(f"{len(seg)} crossing triangle pairs", flush=True)
    # (for each wall, every meeting not along its own edge is a line it must be cut along: a traced line, the edge of
    # the other wall ending on it, or a crossing not yet defined)
    P3 = np.stack([seg[:, 0], seg.mean(1), seg[:, 1]], 1)
    dt, it = t_tree.query(P3.reshape(-1, 3))
    dt, it = dt.reshape(-1, 3).max(1), it.reshape(-1, 3)[:, 1]
    cuts = defaultdict(lambda: defaultdict(float))
    for side in (0, 1):
        me, other = pr[:, side], pr[:, 1 - side]
        for pid in np.unique(owner[me]):
            m = owner[me] == pid
            if edge_tree[pid] is None:
                own = np.full(m.sum(), np.inf)
            else:
                own = edge_tree[pid].query(P3[m].reshape(-1, 3))[0].reshape(-1, 3).max(1)
            L = np.linalg.norm(seg[m, 1] - seg[m, 0], axis=1)
            for k, (o, ln) in enumerate(zip(other[m], L)):
                if own[k] <= TOL:
                    continue                                # (along this wall's own edge)
                q = owner[o]
                j = np.flatnonzero(m)[k]
                if dt[j] <= TOL:
                    kind = f"traced line {T_name[it[j]]}"
                elif edge_tree[q] is not None and edge_tree[q].query(P3[j])[0].max() <= TOL:
                    kind = f"edge of {pieces[q][0]} ending on it"
                else:
                    kind = f"crossing {pieces[q][0]} (not defined)"
                cuts[(pieces[pid][0], pieces[pid][1])][kind] += ln
    out = {f"{w} ({k})": {kind: round(v, 4) for kind, v in sorted(c.items(), key=lambda kv: -kv[1]) if v > 3 * TOL}
           for (w, k), c in sorted(cuts.items())}
    json.dump(out, open("meetings.json", "w"), indent=1)
    for w, c in out.items():
        if only and w.split()[0] not in only:
            continue
        print(f"  {w}:")
        for kind, v in c.items():
            print(f"      {kind:40s} length {v:.4f}")
    return out


if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
