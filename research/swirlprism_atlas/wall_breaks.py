"""Where another wall passes through a wall: the curves along which the two meshes cross, inside the wall (away from
its own edges), drawn on it as thin white lines so the break in the shape there shows.

Used by cell_atlas2.py on the cell view's wall pieces (xyz triangles): wall_breaks(walls) gives, per piece index,
a list of polylines.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

CELL = 0.02          # grid for pairing triangles whose boxes overlap
EDGE_GAP = 0.004     # crossings closer than this to the wall's own edge are where it ends, not a break


def _segment(A, B, eps=1e-12):
    """The segment where triangles A and B (3x3 arrays) cross, or None."""
    na = np.cross(A[1] - A[0], A[2] - A[0])
    nb = np.cross(B[1] - B[0], B[2] - B[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if la < eps or lb < eps:
        return None
    na, nb = na / la, nb / lb
    d = np.cross(na, nb)
    if np.linalg.norm(d) < 1e-6:                     # (parallel or coplanar: no crossing line)
        return None
    sa = (A - B[0]) @ nb                            # A's corners against B's plane
    sb = (B - A[0]) @ na
    if (sa > eps).all() or (sa < -eps).all() or (sb > eps).all() or (sb < -eps).all():
        return None

    def cut(T, s):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if abs(s[i]) <= eps:
                pts.append(T[i])
            if (s[i] > eps and s[j] < -eps) or (s[i] < -eps and s[j] > eps):
                t = s[i] / (s[i] - s[j])
                pts.append(T[i] + t * (T[j] - T[i]))
        if len(pts) < 2:
            return None
        P = np.array(pts)
        t = P @ d
        return P[np.argmin(t)], P[np.argmax(t)]
    ca, cb = cut(A, sa), cut(B, sb)
    if ca is None or cb is None:
        return None
    ta = sorted((ca[0] @ d, ca[1] @ d))
    tb = sorted((cb[0] @ d, cb[1] @ d))
    lo, hi = max(ta[0], tb[0]), min(ta[1], tb[1])
    if hi - lo < 1e-9:
        return None
    # (the two cut segments lie on one line: the overlap, as points on A's cut)
    p, q = ca if ca[0] @ d <= ca[1] @ d else (ca[1], ca[0])
    span = q @ d - p @ d
    if span < 1e-12:
        return None
    return p + (q - p) * ((lo - p @ d) / span), p + (q - p) * ((hi - p @ d) / span)


def _boundary_edges(T):
    """A mesh's edges used by one triangle only (its outline), as segment endpoint arrays."""
    key = lambda p: tuple(np.round(p, 7))
    count, at = defaultdict(int), {}
    for t in T:
        for i in range(3):
            a, b = key(t[i]), key(t[(i + 1) % 3])
            e = (a, b) if a < b else (b, a)
            count[e] += 1
            at[a], at[b] = t[i], t[(i + 1) % 3]
    E = [(at[a], at[b]) for (a, b), n in count.items() if n == 1]
    return (np.array([e[0] for e in E]), np.array([e[1] for e in E])) if E else (np.zeros((0, 3)), np.zeros((0, 3)))


def _dist_to_segments(p, S0, S1):
    if not len(S0):
        return np.inf
    v = S1 - S0
    t = np.clip(((p - S0) * v).sum(1) / np.maximum((v * v).sum(1), 1e-30), 0, 1)
    return np.min(np.linalg.norm(S0 + t[:, None] * v - p, axis=1))


def _chain(segs, tol=1e-6):
    """Segments joined end to end into polylines."""
    key = lambda p: tuple(np.round(np.asarray(p) / tol).astype(int))
    ends = defaultdict(list)
    for k, (a, b) in enumerate(segs):
        ends[key(a)].append(k)
        ends[key(b)].append(k)
    used, out = set(), []
    for k in range(len(segs)):
        if k in used:
            continue
        used.add(k)
        line = [segs[k][0], segs[k][1]]
        for forward in (True, False):
            while True:
                tip = line[-1] if forward else line[0]
                nxt = [j for j in ends[key(tip)] if j not in used]
                if not nxt:
                    break
                j = nxt[0]
                used.add(j)
                a, b = segs[j]
                new = b if key(a) == key(tip) else a
                if forward:
                    line.append(new)
                else:
                    line.insert(0, new)
        out.append(line)
    return out


def wall_breaks(walls):
    """walls: [{id, tris (n,3,3 xyz), copy}]; returns {index: [polyline, ...]} for the pieces another wall's piece
    passes through (a piece of another shape, or a copy of another shape; not the wall's own copies)."""
    Ts = [np.asarray(w["tris"], float).reshape(-1, 3, 3) for w in walls]
    grid = defaultdict(list)                          # cell -> (piece, triangle)
    for k, T in enumerate(Ts):
        lo = np.floor(T.min(axis=1) / CELL).astype(int)
        hi = np.floor(T.max(axis=1) / CELL).astype(int)
        for i in range(len(T)):
            for x in range(lo[i, 0], hi[i, 0] + 1):
                for y in range(lo[i, 1], hi[i, 1] + 1):
                    for z in range(lo[i, 2], hi[i, 2] + 1):
                        grid[(x, y, z)].append((k, i))
    out = {}
    for k, T in enumerate(Ts):
        if walls[k].get("copy"):
            continue
        S0, S1 = _boundary_edges(T)
        segs, seen = [], set()
        lo = np.floor(T.min(axis=1) / CELL).astype(int)
        hi = np.floor(T.max(axis=1) / CELL).astype(int)
        for i in range(len(T)):
            cand = set()
            for x in range(lo[i, 0], hi[i, 0] + 1):
                for y in range(lo[i, 1], hi[i, 1] + 1):
                    for z in range(lo[i, 2], hi[i, 2] + 1):
                        cand.update(c for c in grid[(x, y, z)] if walls[c[0]]["id"] != walls[k]["id"])
            for c in cand:
                if (i, c) in seen:
                    continue
                seen.add((i, c))
                s = _segment(T[i], Ts[c[0]][c[1]])
                if s is None:
                    continue
                m = (s[0] + s[1]) / 2
                if _dist_to_segments(m, S0, S1) < EDGE_GAP:
                    continue
                segs.append(s)
        if segs:
            out[k] = [[p.tolist() for p in line] for line in _chain(segs) if len(line) >= 2]
    return out
