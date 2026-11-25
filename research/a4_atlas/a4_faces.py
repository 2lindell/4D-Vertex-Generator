"""Wall pieces from the exact curve graph, as the swirlprism atlas builds them (sphere_map/domain_pieces.py): on each
surface the exact segments of a4_exact (the curves where it meets the other surfaces and the mirrors, cut at the exact
points) form a graph; going round each point in the surface's tangent plane gives its faces, each bounded by exact
loops, so every corner is an exact point and every edge an exact curve. Each face is triangulated inside its loops
(laid flat across the surface's normal, quality triangles, every inner vertex solved onto the surface).

A face takes the place of the marching piece it lies in (a4_exact's pieces carry the classification: shape, sides);
a face in no wall piece is dropped. A piece whose faces do not cover it (area off by more than AREA_TOL) keeps its
marching mesh.

    python a4_faces.py iodecaic          (reads a4_pieces_<group>.pkl, writes it back)
"""
from __future__ import annotations

import os
import pickle
import time
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_pieces as P  # noqa: E402
import a4_exact as E  # noqa: E402

BZ = P.BZ
P.MAX_AREA = float(os.environ.get("A4_FACE_AREA", "6e-4"))   # (about the marching grid's triangles)
AREA_TOL = 0.08
H = 1.6 / E.GRID                                 # (the marching grid's step)


def poly_dist(x, Q):
    """The distance from x to the polyline Q."""
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return float(np.linalg.norm(Q[0] - x))
    A_, d_ = Q[:-1], Q[1:] - Q[:-1]
    u = np.clip(np.einsum("ij,ij->i", x - A_, d_) / (np.einsum("ij,ij->i", d_, d_) + 1e-300), 0, 1)
    return float(np.linalg.norm(A_ + u[:, None] * d_ - x, axis=1).min())


def frame_at(S, x):
    """An orthonormal frame (e1, e2) of the surface's tangent plane at x (in the 3-space sum(beta) = 0), turning the
    way its gradient does."""
    n = P.normal(S, x) @ BZ.T
    n /= np.linalg.norm(n)
    a = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


class _Ball:
    """|x - c|^2 = r^2, as a surface for a4_exact's Newton steps and traces."""

    def __init__(self, c, r):
        self.c, self.r = np.asarray(c, float), r

    def f(self, x):
        return float((x - self.c) @ (x - self.c) - self.r ** 2)

    def g(self, x):
        return 2 * (x - self.c)


SURFS = None          # (all the surfaces, when known: link points are then solved exactly on their curve)


def link_groups(S, v, hs, r=2e-3, pair_of=None):
    """At a singular point v of S: the curves leaving it in turn on each sheet. Each curve's point at distance r is
    found; the curve where the sphere of radius r round v meets S is traced through them (each component a sheet's
    round, cut by the mirrors into arcs); along each, the curves' points in order, turning the way S's normal there
    does."""
    mirrors = [E.Surf(E.B[i], f"b{i + 1}") for i in range(4)]
    ball = _Ball(v, r)
    at = {}
    for i, fw, Q in hs:
        Q = np.asarray(Q, float)
        d = np.linalg.norm(Q - v, axis=1)
        j = int(np.argmax(d >= r)) if (d >= r).any() else len(Q) - 1
        if j == 0:
            p = Q[0]
        else:
            t = (r - d[j - 1]) / max(d[j] - d[j - 1], 1e-300)
            p = Q[j - 1] + np.clip(t, 0, 1) * (Q[j] - Q[j - 1])
        pr = pair_of(i, fw, p) if pair_of else None
        if pr is not None:
            q, ok = E.newton(p, [pr[0], pr[1], ball])     # (exactly on the curve: tangent curves lie close there)
            if ok and np.linalg.norm(q - p) < r / 20:
                p = q
            else:
                p, ok = E.newton(p, [S, ball])
        else:
            p, ok = E.newton(p, [S, ball])
        at[(i, fw)] = p
    left = dict(at)
    comps = []
    while left:
        he, p = next(iter(left.items()))
        pts, closed = E.trace(p, S, ball, mirrors, h=r / 200)
        C = np.array(pts) if len(pts) >= 2 else np.array([p, p])
        mine = []
        A_, D_ = C[:-1], C[1:] - C[:-1]
        for he2, q in list(left.items()):
            # (its place along the round to a fraction of a step: curves leaving the point all but tangent come
            #  within a step of each other there)
            u = np.clip(np.einsum("ij,ij->i", q - A_, D_) / (np.einsum("ij,ij->i", D_, D_) + 1e-300), 0, 1)
            dd = np.linalg.norm(A_ + u[:, None] * D_ - q, axis=1)
            j = int(np.argmin(dd))
            if dd[j] < r / 10:
                mine.append((j + float(u[j]), he2))
                del left[he2]
        if he in left:
            del left[he]
            mine.append((0, he))
        mine.sort()
        order = [h for _, h in mine]
        # (turning with the normal: going along the round, the point moves anticlockwise seen down S's normal)
        if len(C) >= 3:
            m = len(C) // 2
            tdir = (C[min(m + 1, len(C) - 1)] - C[max(m - 1, 0)]) @ BZ.T
            n = P.normal(S, C[m]) @ BZ.T
            if np.cross((C[m] - v) @ BZ.T, tdir) @ n < 0:
                order = order[::-1]
        comps.append(order)
    return comps


def leave_dir(Q, d=0.02):
    """The direction a polyline leaves its first point: the point a little way along, from the start."""
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = min(d, cum[-1] / 3)
    i = int(np.searchsorted(cum, d))
    i = min(max(i, 1), len(Q) - 1)
    return Q[i] - Q[0]


def exact_leave(k, S, v, Q, on, d=0.02):
    """The direction a curve leaves v, from its point at distance exactly d (or a third of its length), solved on the
    curve: curves leaving v all but tangent differ there by less than a polyline's points' spacing turns the chord."""
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    d = min(d, seg.sum() / 3)
    r = np.linalg.norm(Q - v, axis=1)
    j = int(np.argmax(r >= d)) if (r >= d).any() else len(Q) - 1
    if j == 0:
        return leave_dir(Q)
    t = np.clip((d - r[j - 1]) / max(r[j] - r[j - 1], 1e-300), 0, 1)
    p = Q[j - 1] + t * (Q[j] - Q[j - 1])
    others = [SURFS[m] for m in on if m != k]
    if others:
        C = max(others, key=lambda C_: E.crossing_angle(p, S, C_))
        q, ok = E.newton(p, [S, C, _Ball(v, d)])
        if ok and np.linalg.norm(q - p) < d / 20:
            p = q
    return p - v


def thin_seg(Q):
    """A curve's points about half a triangle's side apart, both its ends (exact points) kept."""
    step = 0.5 * np.sqrt(P.MAX_AREA)
    keep = [0]
    for i in range(1, len(Q) - 1):
        if np.linalg.norm(Q[i] - Q[keep[-1]]) >= step and np.linalg.norm(Q[-1] - Q[i]) >= 0.5 * step:
            keep.append(i)
    keep.append(len(Q) - 1)
    return Q[keep]


def faces_on(k, S, segs, Pts):
    """The faces of the curve graph on surface k: (loops [outer, holes...], half-edge cycles) each."""
    own = [i for i, s in enumerate(segs) if k in s["on"]]
    if not own:
        return []
    # (one copy of each curve: two segments with the same ends running together are one)
    uniq = []
    for i in own:
        s = segs[i]
        Q = np.asarray(s["pts"], float)
        dup = False
        for j in uniq:
            t = segs[j]
            if set(t["ends"]) == set(s["ends"]):
                if poly_dist(Q[len(Q) // 2], t["pts"]) < 1e-5:
                    dup = True
                    break
        if not dup:
            uniq.append(i)
    out = defaultdict(list)
    for i in uniq:
        Q = np.asarray(segs[i]["pts"], float)
        a, b = segs[i]["ends"]
        out[a].append((i, True, Q))
        out[b].append((i, False, Q[::-1]))
    # (round each point, the curves leaving it in turn: in the surface's tangent plane there, or, where the surface
    #  is singular (two sheets crossing, a cone), along the curve a small sphere round the point cuts from it, each
    #  sheet its own round)
    groups, pos = {}, {}
    junc = [v for v, hs in out.items() if len(hs) > 2]
    near_junc = None
    if len(junc) > 1:
        from scipy.spatial import cKDTree
        jt = cKDTree(np.array([Pts[v] for v in junc]))

        def near_junc(v):
            d, _ = jt.query(np.asarray(Pts[v], float), k=2)
            return float(d[1]) if d[0] < 1e-12 else float(d[0])
    for v, hs in out.items():
        x = np.asarray(Pts[v], float)
        g = S.g(x)
        if np.linalg.norm(g - g.mean()) < 1e-7 and len(hs) > 2:
            # (the small sphere well inside every curve leaving the point: one ending inside it has no place on it)
            if SURFS is not None:
                # (points within 3e-3 of a hub are merged into it, so curves' crossings that near are folded into
                #  it: their order is read just outside, 3.5e-3 out, a curve ending sooner followed on along the
                #  curve it continues in; each point solved exactly on its curve)
                r_ = 3.5e-3
                # (but within half the way to the nearest other point where curves branch: one closer than 7e-3 is a
                #  real point of its own, not merged into this one, and the sphere must not take it in)
                if near_junc:
                    r_ = min(r_, near_junc(v) / 2)
                hs2, src = [], {}
                for i, fw, Q in hs:
                    parts, segi = [np.asarray(Q, float)], [i]
                    L = np.linalg.norm(np.diff(parts[0], axis=0), axis=1).sum()
                    cur, end, on = i, (segs[i]["ends"][1] if fw else segs[i]["ends"][0]), set(segs[i]["on"])
                    while L < 2 * r_:
                        nxt = [h for h in out.get(end, []) if h[0] != cur]
                        if not nxt:
                            break
                        j, fw2, R = max(nxt, key=lambda h: len(on & set(segs[h[0]]["on"])))
                        R = np.asarray(R, float)
                        parts.append(R[1:])
                        segi += [j] * (len(R) - 1)
                        L += np.linalg.norm(np.diff(R, axis=0), axis=1).sum()
                        cur, end = j, (segs[j]["ends"][1] if fw2 else segs[j]["ends"][0])
                    QQ = np.vstack(parts)
                    hs2.append((i, fw, QQ))
                    src[(i, fw)] = (QQ, [i] * len(parts[0]) + segi[1:])

                def pair_of(i, fw, p, src=src):
                    QQ, si = src[(i, fw)]
                    j = si[min(int(np.argmin(np.linalg.norm(QQ - p, axis=1))), len(si) - 1)]
                    others = [SURFS[m] for m in segs[j]["on"] if m != k]
                    if not others:
                        return None
                    return S, max(others, key=lambda C: E.crossing_angle(p, S, C))
                gr = link_groups(S, x, hs2, r=r_, pair_of=pair_of)
            else:
                short = min(np.linalg.norm(np.diff(np.asarray(Q, float), axis=0), axis=1).sum() for _, _, Q in hs)
                gr = link_groups(S, x, hs, r=min(2e-3, short / 3))
        else:
            e1, e2 = frame_at(S, x)
            ang = {}
            # (every curve read at the same distance: a third of the shortest's length, at most 0.02. Read at
            #  different distances, curves leaving where the surface is all but singular turn past each other)
            dd = min([0.02] + [np.linalg.norm(np.diff(np.asarray(Q, float), axis=0), axis=1).sum() / 3
                               for _, _, Q in hs])
            for i, fw, Q in hs:
                d = (exact_leave(k, S, x, np.asarray(Q, float), segs[i]["on"], d=dd) if SURFS is not None
                     else leave_dir(Q)) @ BZ.T
                ang[(i, fw)] = np.arctan2(d @ e2, d @ e1)
            gr = [[(h[0], h[1]) for h in sorted(hs, key=lambda h: ang[(h[0], h[1])])]]
        groups[v] = gr
        pos[v] = {he: (gi, j) for gi, grp in enumerate(gr) for j, he in enumerate(grp)}

    def head(i, fw):
        a, b = segs[i]["ends"]
        return b if fw else a
    seen, cycles, hcyc = set(), [], []
    for v, hs in out.items():
        for h in hs:
            he = (h[0], h[1])
            if he in seen:
                continue
            cyc, cur = [], he
            for _ in range(10 * len(uniq) + 10):
                if cur in seen:
                    break
                seen.add(cur)
                cyc.append(cur)
                w = head(*cur)
                if (cur[0], not cur[1]) not in pos[w]:
                    break
                gi, j = pos[w][(cur[0], not cur[1])]
                grp = groups[w][gi]
                cur = grp[(j - 1) % len(grp)]
            if cur != he:
                continue
            pts = []
            for i, fw in cyc:
                Q = thin_seg(np.asarray(segs[i]["pts"], float))
                Q = Q if fw else Q[::-1]
                pts.extend(Q[:-1])
            cycles.append(np.array(pts))
            hcyc.append(cyc)

    # (every cycle is a face, on its left, once split where it walks a curve both ways (a slit joining two faces)
    #  or passes a point twice (faces touching only there); each part is a face, unless it is a cycle round the
    #  outside, which has the outside of the chamber on its left)
    parts = []
    for cyc in hcyc:
        for c in unbridge(segs, cyc)[1]:
            parts += cycle_lobes(segs, c)
    out = []
    for cyc in parts:
        pts = []
        for i, fw in cyc:
            Q = thin_seg(np.asarray(segs[i]["pts"], float))
            Q = Q if fw else Q[::-1]
            pts.extend(Q[:-1])
        if len(pts) < 3:
            continue
        L = np.array(pts)
        mir = [sorted(set(segs[i]["on"]) & ({0, 1, 2, 3} - {k})) for i, _ in cyc]
        if not all(mir):                         # (a curve inside the chamber in it: an inside face)
            out.append(([L], [cyc]))
            continue
        # (all on the chamber's faces: inside, or the cycle round the outside. Steps in from many points of its
        #  mirror curves decide, by vote: where a sheet of the surface lies along a mirror, nearly touching it, one
        #  step can land either side)
        votes = []
        for (i, fw), ms in zip(cyc, mir):
            if len(ms) != 1:
                continue
            Q = np.asarray(segs[i]["pts"], float)
            Q = Q if fw else Q[::-1]
            for f in (0.2, 0.35, 0.5, 0.65, 0.8):
                m = min(max(int(f * (len(Q) - 1)), 0), len(Q) - 2)
                x = Q[m] + (Q[m + 1] - Q[m]) * (f * (len(Q) - 1) - m)
                d = Q[m + 1] - Q[m]
                n = P.normal(S, x) @ BZ.T
                side = np.cross(n, d @ BZ.T) @ BZ
                nn = np.linalg.norm(side)
                if nn < 1e-14:
                    continue
                y = P.lift_normal(S, x + 1e-3 * side / nn)  # (the walk keeps the face on its left)
                if y is not None:
                    votes.append(y[ms[0]] > 0)
        keep = bool(votes) and np.mean(votes) > 0.5
        if os.environ.get("A4_DEBUG_FACES") == str(k):
            print("   all-mirror cycle", [i for i, _ in cyc], "votes", votes, "kept", keep)
        if keep:
            out.append(([L], [cyc]))
    # (a lobe of the cycle round the outside of a cluster joined at one point runs round the same points as the
    #  face inside it, the other way: the same face, kept once)
    seen, uniq = set(), []
    for f in out:
        key = frozenset(map(tuple, np.round(f[0][0], 9)))
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq



def cycle_lobes(segs, cyc):
    """A cycle through one point twice split there into the cycles of its lobes (each a face of its own)."""
    todo, done = [list(cyc)], []
    while todo:
        c = todo.pop()
        start = [segs[i]["ends"][0] if fw else segs[i]["ends"][1] for i, fw in c]
        first, hit = {}, None
        for j, v in enumerate(start):
            if v in first:
                hit = (first[v], j)
                break
            first[v] = j
        if hit is None:
            done.append(c)
            continue
        a, b = hit
        todo += [x for x in (c[a:b], c[b:] + c[:a]) if x]
    return done


def unbridge(segs, cyc):
    """A cycle that walks a curve both ways (a bridge: the face on both sides of it) split there into the loops it
    joins: (loops, [cycles])."""
    stack, done = [list(cyc)], []
    while stack:
        c = stack.pop()
        pos = {}
        for a, (i, fw) in enumerate(c):
            if i in pos:
                b = pos[i]
                inner, outer = c[b + 1:a], c[:b] + c[a + 1:]
                stack += [x for x in (inner, outer) if x]
                break
            pos[i] = a
        else:
            done.append(c)
    loops = []
    for c in done:
        pts = []
        for i, fw in c:
            Q = thin_seg(np.asarray(segs[i]["pts"], float))
            Q = Q if fw else Q[::-1]
            pts.extend(Q[:-1])
        if len(pts) >= 3:
            loops.append(np.array(pts))
    return loops, done


def cut(k, S, segs, Pts, surfs, hcycles, step=3e-3):
    """A face too curved to lay flat, cut across by the curve where its surface meets a plane (traced exactly, its
    ends solved on the face's edges): the face's edges split there, the cut added as curves of the graph."""
    L = np.vstack([np.asarray(segs[i]["pts"], float) for cyc in hcycles for i, _ in cyc])
    c0 = L.mean(0)
    axes = np.linalg.svd(L - c0, full_matrices=False)[2][:2]
    for w in axes:                               # (planes across the face's longest and next longest ways, through
        w = w - w.mean()                         #  its middle or a little to either side)
        w /= np.linalg.norm(w)
        v = L @ w
        for c in (np.median(v), np.quantile(v, 0.4), np.quantile(v, 0.6), np.quantile(v, 0.3), np.quantile(v, 0.7)):
            n0, p0 = len(segs), len(Pts)
            saved = [dict(x) for x in segs]
            if _cut_with(k, S, segs, Pts, surfs, hcycles, w, c, step):
                return True
            segs[:] = saved                      # (nothing traced: the edges left as they were)
            del Pts[p0:]
    return False


def _cut_with(k, S, segs, Pts, surfs, hcycles, w, c, step):
    if True:
        pass
    plane = E.Surf(sum((float(w[i]) - c) * E.B[i] for i in range(4)), "cut")
    # crossings of the face's edges with the plane
    cross = []                                   # (segment, index along it, point, direction along the walk)
    for cyc in hcycles:
        for i, fw in cyc:
            Q = np.asarray(segs[i]["pts"], float)
            v = Q @ w - c
            for j in np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0):
                x0 = Q[j] + (Q[j + 1] - Q[j]) * v[j] / (v[j] - v[j + 1])
                other = [surfs[m] for m in sorted(segs[i]["on"]) if m != k][:1]
                x, ok = E.newton(x0, [S] + other + [plane])
                if not ok or np.linalg.norm(x - x0) > 1e-3:
                    x, ok = E.newton(x0, [S, plane])
                if not ok:
                    continue
                d = (Q[j + 1] - Q[j]) * (1 if fw else -1)
                cross.append((i, j, x, d))
    uniq = []
    for cr in cross:                             # (an edge walked both ways crosses once)
        if not any(np.linalg.norm(cr[2] - u[2]) < 1e-9 for u in uniq):
            uniq.append(cr)
    cross = uniq
    if len(cross) < 2:
        return False
    # (split the edges at the crossings: new exact points)
    ids = []
    for i, j, x, d in cross:
        Pts.append(x)
        ids.append(len(Pts) - 1)
    by_seg = defaultdict(list)
    for (i, j, x, d), pid in zip(cross, ids):
        by_seg[i].append((j, x, pid))
    for i, cs in by_seg.items():
        s = segs[i]
        Q = list(np.asarray(s["pts"], float))
        cs.sort(key=lambda t: t[0])
        parts, start, a = [], 0, s["ends"][0]
        for j, x, pid in cs:
            parts.append((a, pid, Q[start:j + 1] + [x]))
            Q[j] = Q[j]
            start = j + 1
            a = pid
            Q_ = Q
        parts.append((a, s["ends"][1], [cs[-1][1]] + Q[start:]))
        for n_, (a_, b_, pts) in enumerate(parts):
            seg = dict(s, ends=(a_, b_), pts=np.array(pts))
            if n_ == 0:
                segs[i] = seg
            else:
                segs.append(seg)
    # (trace inward from each crossing to the next it reaches)
    used = set()
    pos = np.array([x for _, _, x, _ in cross])
    for q, (i, j, x, d) in enumerate(cross):
        if q in used:
            continue
        t0 = E.tangent(x, S, plane)
        end = None
        for sgn in (1, -1):                  # (the way into the face: the way that stays in the chamber and reaches
            t, pts, y = sgn * t0, [x], x.copy()          # another crossing)
            for _ in range(4000):
                z, ok = E.newton(y + step * t, [S, plane])
                if not ok or z.min() < -1e-9:
                    break
                t2 = E.tangent(z, S, plane)
                t = t2 if t2 @ t > 0 else -t2
                y = z
                pts.append(y)
                dd = np.linalg.norm(pos - y, axis=1)
                dd[q] = np.inf
                r = int(np.argmin(dd))
                if dd[r] < 3 * step and len(pts) > 2:
                    end = r
                    break
            if end is not None:
                break
        if end is None or end in used:
            continue
        pts[-1] = pos[end]
        used |= {q, end}
        segs.append({"ends": (ids[q], ids[end]), "pts": np.array(pts), "on": {k}, "key": None, "sig": None})
    return bool(used)


def inside(q, Q):
    x, y = q
    c = False
    for i in range(len(Q)):
        (x1, y1), (x2, y2) = Q[i], Q[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1:
            c = not c
    return c


def mesh_area(V, T):
    V = np.asarray(V, float)
    T = np.asarray(T, int)
    if not len(T):
        return 0.0
    e1 = (V[T[:, 1]] - V[T[:, 0]]) @ BZ.T
    e2 = (V[T[:, 2]] - V[T[:, 0]]) @ BZ.T
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def disk_layout(F, S):
    """The marching piece laid on the unit disk (Tutte: its rim round the circle by length, each inner vertex the
    mean of its neighbours): (V, T, uv), or None when the piece is not a disk."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    G = {"surface": F["surface"], "mesh": F["mesh"], "filled": F.get("filled", [])}
    P.weld(G)
    R = P.rim_loops(G, S, [], 1.0, ids=True)
    if R is None or len(R) != 1:
        return None
    loop = [int(v) for v in R[0][0]]
    V = np.asarray(G["mesh"]["verts"], float)
    fl = set(G.get("filled", []))
    T = np.array([t for t in G["mesh"]["faces"] if not fl.intersection(t)], int)
    seg = [np.linalg.norm(V[loop[(i + 1) % len(loop)]] - V[loop[i]]) for i in range(len(loop))]
    ang = np.concatenate([[0], np.cumsum(seg)[:-1]]) / sum(seg) * 2 * np.pi
    bnd = {v: (np.cos(a), np.sin(a)) for v, a in zip(loop, ang)}
    used = sorted({int(a) for t in T for a in t})
    inner = [v for v in used if v not in bnd]
    nb = defaultdict(set)
    for t in T:
        for a in t:
            nb[int(a)].update(int(b) for b in t if b != a)
    kx = {v: j for j, v in enumerate(inner)}
    uv = np.zeros((len(V), 2))
    if inner:
        A = lil_matrix((len(inner), len(inner)))
        rhs = np.zeros((len(inner), 2))
        for v in inner:
            A[kx[v], kx[v]] = len(nb[v])
            for w in nb[v]:
                if w in kx:
                    A[kx[v], kx[w]] -= 1
                elif w in bnd:
                    rhs[kx[v]] += bnd[w]
        sol = spsolve(A.tocsr(), rhs).reshape(-1, 2)
        for v, j in kx.items():
            uv[v] = sol[j]
    for v, q in bnd.items():
        uv[v] = q
    return V, T, uv


def onto_mesh(V, T, x):
    """The point of the mesh nearest x: (triangle, barycentric coordinates)."""
    best = None
    X3 = V @ BZ.T
    y = x @ BZ.T
    for j in range(len(T)):
        a, b, c = X3[T[j]]
        # (closest point on a triangle: clamped barycentric, refined over its edges)
        e0, e1, d = b - a, c - a, y - a
        M = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]])
        try:
            s_, t_ = np.linalg.solve(M, [e0 @ d, e1 @ d])
        except np.linalg.LinAlgError:
            continue
        lam = np.array([1 - s_ - t_, s_, t_])
        if lam.min() < 0:
            lam = np.clip(lam, 0, None)
            lam /= lam.sum()
        dist = np.linalg.norm(lam @ X3[T[j]] - y)
        if best is None or dist < best[0]:
            best = (dist, j, lam)
    return best[1], best[2]


def tutte_exact(F, S, loop, planar):
    """A face too curved to lay flat, on its marching piece laid on a disk: its exact loop put on the disk through
    the nearest points of the piece, triangulated there (refined by area on the surface), the inner vertices carried
    back through the piece and solved onto the surface; the loop's own points kept exact."""
    D = disk_layout(F, S)
    if D is None:
        return None
    V, T, uv = D
    L = P.thin(np.asarray(loop, float), 0.5 * np.sqrt(P.MAX_AREA))
    cen = cKDTree((V[T].mean(1)) @ BZ.T)
    Lu = []
    for x in L:
        js = cen.query(x @ BZ.T, k=min(16, len(T)))[1]
        js = np.atleast_1d(js)
        j, lam = onto_mesh(V, T[js], x)
        Lu.append(lam @ uv[T[js][j]])
    Lu = np.array(Lu)
    keep = [0] + [i for i in range(1, len(Lu)) if np.linalg.norm(Lu[i] - Lu[i - 1]) > 1e-9]
    L, Lu = L[keep], Lu[keep]
    if len(L) < 3 or P.crossings(Lu):
        return None
    if P.area2(Lu) < 0:
        L, Lu = L[::-1], Lu[::-1]
    m = len(Lu)
    a_, b_, c_ = uv[T][:, 0], uv[T][:, 1], uv[T][:, 2]
    det = (b_[:, 1] - c_[:, 1]) * (a_[:, 0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (a_[:, 1] - c_[:, 1])
    ok = np.abs(det) > 1e-300
    det = np.where(ok, det, 1.0)

    def to3(q):
        l1 = ((b_[:, 1] - c_[:, 1]) * (q[0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (q[1] - c_[:, 1])) / det
        l2 = ((c_[:, 1] - a_[:, 1]) * (q[0] - c_[:, 0]) + (a_[:, 0] - c_[:, 0]) * (q[1] - c_[:, 1])) / det
        l3 = 1 - l1 - l2
        worst = np.where(ok, np.minimum(np.minimum(l1, l2), l3), -np.inf)
        j = int(np.argmax(worst))
        if worst[j] < -0.02:
            return None
        lam = np.clip([l1[j], l2[j], l3[j]], 0, None)
        lam /= lam.sum()
        y = lam @ V[T[j]]
        return y if planar else P.lift_normal(S, y)
    A3 = mesh_area(V, T)
    amax = P.MAX_AREA * abs(P.area2(Lu)) / max(A3, 1e-12)
    try:
        Tr = P.tr.triangulate({"vertices": np.ascontiguousarray(Lu, dtype=np.float64),
                               "segments": np.array([[i, (i + 1) % m] for i in range(m)], np.int32)},
                              f"pYq20a{amax:.3g}")
    except Exception:
        return None
    if "triangles" not in Tr or len(Tr["vertices"]) < m:
        return None
    for rnd in range(5):
        Xv = list(L)
        for q in Tr["vertices"][m:]:
            y = to3(q)
            if y is None:
                return None
            Xv.append(y)
        Xv = np.array(Xv)
        Ft = np.asarray(Tr["triangles"])
        e1 = (Xv[Ft[:, 1]] - Xv[Ft[:, 0]]) @ BZ.T
        e2 = (Xv[Ft[:, 2]] - Xv[Ft[:, 0]]) @ BZ.T
        a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        if a3.max() <= 1.6 * P.MAX_AREA or rnd == 4:
            break
        U = Tr["vertices"]
        ad = 0.5 * np.abs((U[Ft[:, 1], 0] - U[Ft[:, 0], 0]) * (U[Ft[:, 2], 1] - U[Ft[:, 0], 1])
                          - (U[Ft[:, 1], 1] - U[Ft[:, 0], 1]) * (U[Ft[:, 2], 0] - U[Ft[:, 0], 0]))
        lim = np.where(a3 > P.MAX_AREA, ad * P.MAX_AREA / np.maximum(a3, 1e-300), -1.0)
        try:
            Tr = P.tr.triangulate({"vertices": U, "triangles": Ft, "segments": Tr["segments"],
                                   "triangle_max_area": lim}, "rpYq20a")
        except Exception:
            return None
    return Xv, np.asarray(Tr["triangles"])


def clean_graph(X, surfs, log=print, tol=1e-7):
    """The exact curve graph made whole: exact points closer than tol made one (curves ending at one point from
    different traces were given different copies of it, so the graph fell apart there); the swap's axis, where
    many curves end (traces stop where their surfaces turn tangent) and many overlapping copies of it were traced,
    rebuilt as one chain through its distinct points, on every surface containing it; copies of a curve made one."""
    P_ = np.array([np.asarray(p, float) for p in X["points"]])
    tree = cKDTree(P_)
    rep = np.arange(len(P_))
    for i, j in sorted(tree.query_pairs(tol)):
        a, b = rep[i], rep[j]
        while rep[a] != a:
            a = rep[a]
        while rep[b] != b:
            b = rep[b]
        if a != b:
            rep[max(a, b)] = min(a, b)
    for i in range(len(rep)):
        r = i
        while rep[r] != r:
            r = rep[r]
        rep[i] = r
    on_ax = lambda Q: np.abs(Q[:, 0] - Q[:, 3]).max() < 1e-9 and np.abs(Q[:, 1] - Q[:, 2]).max() < 1e-9
    ax = lambda t: np.array([t, 0.5 - t, 0.5 - t, t])
    ax_surfs = {k for k, S in enumerate(surfs)
                if all(abs(S.f(ax(t))) / (np.linalg.norm(S.g(ax(t))) + 1e-300) < 1e-11 for t in (0.05, 0.21, 0.37))}
    segs, seen = [], {}
    n_ax = 0
    for s in X["segments"]:
        Q = np.asarray(s["pts"], float)
        if ax_surfs and on_ax(Q):
            n_ax += 1
            continue
        a, b = int(rep[s["ends"][0]]), int(rep[s["ends"][1]])
        if a == b and np.linalg.norm(Q[-1] - Q[0]) < 1e-9 and len(Q) < 4:
            continue
        key = (min(a, b), max(a, b), tuple(np.round(Q[len(Q) // 2], 7)))
        key2 = (min(a, b), max(a, b), tuple(np.round(Q[(len(Q) - 1) - len(Q) // 2], 7)))
        hit = seen.get(key) or seen.get(key2)
        if hit is not None:
            hit["on"] = set(hit["on"]) | set(s["on"])
            continue
        Q = Q.copy()
        Q[0], Q[-1] = P_[a], P_[b]
        t = dict(s, ends=(a, b), pts=Q)
        seen[key] = t
        segs.append(t)
    if ax_surfs:
        used = {int(rep[e]) for s in X["segments"] for e in s["ends"]}
        axp = sorted({i for i in used if abs(P_[i][0] - P_[i][3]) < 1e-9 and abs(P_[i][1] - P_[i][2]) < 1e-9},
                     key=lambda i: P_[i][0])
        for i, j in zip(axp, axp[1:]):
            t0, t1 = P_[i][0], P_[j][0]
            if t1 - t0 < 1e-12:
                continue
            m = max(2, int(np.ceil((t1 - t0) / 2e-3)) + 1)
            Q = np.array([ax(t) for t in np.linspace(t0, t1, m)])
            Q[0], Q[-1] = P_[i], P_[j]
            segs.append({"ends": (i, j), "pts": Q, "on": set(ax_surfs), "key": None, "sig": None, "axis": True})
    # a curve can lie on more surfaces than the pair it was traced on: every surface it lies on is a side of it
    grown = 0
    for s in segs:
        Q = np.asarray(s["pts"], float)
        samp = Q[np.unique(np.linspace(0, len(Q) - 1, min(len(Q), 7)).round().astype(int))]
        if len(Q) > 2:
            samp = samp[1:-1] if len(samp) > 2 else Q[1:-1]
        else:
            samp = (Q[:1] + Q[1:]) / 2
        on = set(s["on"])
        for m, Sm in enumerate(surfs):
            if m in on:
                continue
            if all(abs(Sm.f(x)) / (np.linalg.norm(Sm.g(x)) + 1e-300) < 1e-8 for x in samp):
                on.add(m)
        grown += len(on) > len(s["on"])
        s["on"] = on
    log(f"  graph: {len(set(rep.tolist()))} distinct points of {len(rep)}; {len(X['segments'])} segments -> "
        f"{len(segs)} ({n_ax} along the swap's axis rebuilt as one chain on {len(ax_surfs)} surfaces; "
        f"{grown} found on more surfaces)")
    X["segments"] = segs


def safe(fn, *args, timeout=120):
    """fn(*args) in a process of its own: triangle can hang or exit on a bad loop; then None."""
    import select
    import time
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = pickle.dumps(fn(*args))
        except Exception:  # noqa: BLE001
            out = pickle.dumps(None)
        with os.fdopen(w, "wb") as f:
            f.write(out)
        os._exit(0)
    os.close(w)
    buf, t0 = b"", time.time()
    while True:
        if select.select([r], [], [], 1.0)[0]:
            chunk = os.read(r, 1 << 20)
            if not chunk:
                break
            buf += chunk
        elif time.time() - t0 > timeout:
            os.kill(pid, 9)
            break
    os.close(r)
    os.waitpid(pid, 0)
    try:
        return pickle.loads(buf)
    except Exception:  # noqa: BLE001
        return None


def dense(Q, step=1e-3):
    """A polyline's points with more put in along its long steps (straight curves are kept as their two ends)."""
    Q = np.asarray(Q, float)
    out = [Q[:1]]
    for a, b in zip(Q[:-1], Q[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    return np.vstack(out)


def lobes(L):
    """A loop that passes through one point twice: its lobes, each a simple loop. (A spur, a dangling curve the walk
    goes out along and back, is dropped first; a pinch, where a face's outline touches itself at a corner, splits
    the loop in two.)"""
    L = np.asarray(L, float)
    keep = [0] + [i for i in range(1, len(L)) if np.linalg.norm(L[i] - L[i - 1]) > 1e-10]
    L = L[keep]
    if len(L) > 1 and np.linalg.norm(L[-1] - L[0]) <= 1e-10:
        L = L[:-1]
    key = [tuple(r) for r in np.round(L, 11)]
    # (spurs: ... a b a ... -> ... a ...)
    st = []
    for i, k in enumerate(key):
        st.append(i)
        while len(st) >= 3 and key[st[-1]] == key[st[-3]]:
            del st[-2:]
    out, todo = [], [L[st]]
    while todo:
        L = todo.pop()
        first, hit = {}, None
        for j, r in enumerate(map(tuple, np.round(L, 11))):
            if r in first:
                hit = (first[r], j)
                break
            first[r] = j
        if hit is None:
            if len(L) >= 3:
                out.append(L)
            continue
        i, j = hit
        todo += [L[i:j], np.vstack([L[j:], L[:i]])]
    return out


def tri_face(S, loops, planar):
    """triangulate, or for a pinched single loop, each lobe triangulated and the meshes put together."""
    r_ = safe(P.triangulate, S, loops, planar)
    if r_ is not None or len(loops) != 1:
        return r_
    parts = lobes(loops[0])
    if len(parts) < 2:
        return None
    V, T, off = [], [], 0
    for L in parts:
        q = safe(P.triangulate, S, [L], planar)
        if q is None:
            return None
        V.append(np.asarray(q[0]))
        T.append(np.asarray(q[1]) + off)
        off += len(q[0])
    return np.vstack(V), np.vstack(T).tolist()


def complete_graph(X, surfs, log=print, near=3e-3):
    """The curves a4_exact did not trace, added: wherever a marching piece's rim runs along two surfaces (its cut
    vertices were solved onto both) with no traced curve of that pair near, the curve is traced exactly from there
    (a4_exact.trace), cut where any other surface crosses it (each point solved on all three), and the curves it
    meets there are cut there too, so the graph stays joined."""
    mirrors = surfs[:4]
    Pts = X["points"]
    segs = X["segments"]
    on_axis = lambda x: abs(x[0] - x[3]) < 1e-9 and abs(x[1] - x[2]) < 1e-9
    pair_pts = defaultdict(list)
    for s_ in segs:
        on = sorted(s_["on"])
        for a in on:
            for b in on:
                if a < b:
                    pair_pts[(a, b)].append(dense(s_["pts"]))
    trees = {p: cKDTree(np.vstack(v)) for p, v in pair_pts.items()}
    seeds = defaultdict(list)
    for F in X["faces"]:
        k = F["surface"]
        G = {"mesh": F["mesh"], "filled": F.get("filled", [])}
        P.weld(G)
        V = np.asarray(G["mesh"]["verts"], float)
        for v in sorted({a for e in P.rim_of(G) for a in e}):
            x = V[v]
            if on_axis(x):
                continue
            for j, Sj in enumerate(surfs):
                if j == k or abs(Sj.f(x)) / (np.linalg.norm(Sj.g(x)) + 1e-300) > 1e-9:
                    continue
                p = (min(k, j), max(k, j))
                if p in trees and trees[p].query(x)[0] < near:
                    continue
                seeds[p].append(x)
    log(f"  completing: {sum(len(v) for v in seeds.values())} rim points on {len(seeds)} pairs need curves")
    curves = []
    t_ = time.time()
    for (a, b), xs in seeds.items():
        xs = np.array(xs)
        left = np.ones(len(xs), bool)
        for _ in range(40):
            if not left.any():
                break
            i = int(np.flatnonzero(left)[0])
            x0, ok = E.newton(xs[i], [surfs[a], surfs[b]])
            left[i] = False
            if not ok or not E.inside(x0, 1e-10):
                continue
            pts, closed = E.trace(x0, surfs[a], surfs[b], mirrors)
            if len(pts) < 3:
                continue
            Q = np.array(pts)
            left &= cKDTree(dense(Q)).query(xs)[0] >= near
            curves.append(((a, b), Q, closed))
    # the points on each new curve: other surfaces crossing it, and its ends
    tree_p = cKDTree(np.array(Pts))

    def point_id(x):
        nonlocal tree_p
        d, i = tree_p.query(x)
        if d < 1e-7:
            return int(i)
        Pts.append(np.asarray(x, float))
        tree_p = cKDTree(np.array(Pts))
        return len(Pts) - 1

    by_surf = defaultdict(set)

    def add_seg(s_):
        for m in s_["on"]:
            by_surf[m].add(len(segs))
        segs.append(s_)

    for i_, s_ in enumerate(segs):
        for m in s_["on"]:
            by_surf[m].add(i_)

    def split_existing(x, pid, pairs):
        """Existing curves on any of these surface pairs through x: cut at x (the new point)."""
        cand = set()
        for p_, q_ in pairs:
            cand |= by_surf[p_] & by_surf[q_]
        for i_ in sorted(cand):
            s_ = segs[i_]
            if pid in s_["ends"]:
                continue
            Q = np.asarray(s_["pts"], float)
            A_, B_ = Q[:-1], Q[1:]
            d_ = B_ - A_
            u = np.clip(np.einsum("ij,ij->i", x - A_, d_) / (np.einsum("ij,ij->i", d_, d_) + 1e-300), 0, 1)
            dist = np.linalg.norm(A_ + u[:, None] * d_ - x, axis=1)
            j = int(np.argmin(dist))
            if dist[j] > 1e-6 or min(np.linalg.norm(Q[0] - x), np.linalg.norm(Q[-1] - x)) < 1e-6:
                continue
            first = dict(s_, ends=(s_["ends"][0], pid), pts=np.vstack([Q[:j + 1], x]))
            second = dict(s_, ends=(pid, s_["ends"][1]), pts=np.vstack([x, Q[j + 1:]]))
            segs[i_] = first
            add_seg(second)
    # a traced curve that runs along a curve already in the graph (or one traced before it) is that curve there:
    # only its runs off the graph are new, each ending where it leaves the graph (snapped onto the curve it leaves,
    # which is cut there), and a curve of the graph it covers gains the pair's surfaces
    tol, step = 3e-4, 2e-4
    allp = [dense(s_["pts"], step) for s_ in segs]
    all_tree = cKDTree(np.vstack(allp))
    all_of = np.concatenate([[i] * len(q) for i, q in enumerate(allp)])
    new_pts, new_tree = [], None

    def near_graph(Q):
        d = all_tree.query(Q)[0]
        if new_tree is not None:
            d = np.minimum(d, new_tree.query(Q)[0])
        return d < tol

    def snap(x):
        """x put on the nearest curve of the graph (the old one), that curve cut there; the point's id."""
        _, i0 = all_tree.query(x)
        s_ = segs[all_of[i0]]
        on = sorted(s_["on"])
        best = None
        for i_ in set().union(*[by_surf[m] for m in on[:1]]):
            t_ = segs[i_]
            if not set(on) <= set(t_["on"]):
                continue
            Q = np.asarray(t_["pts"], float)
            A_, d_ = Q[:-1], Q[1:] - Q[:-1]
            u = np.clip(np.einsum("ij,ij->i", x - A_, d_) / (np.einsum("ij,ij->i", d_, d_) + 1e-300), 0, 1)
            y = A_ + u[:, None] * d_
            dist = np.linalg.norm(y - x, axis=1)
            j_ = int(np.argmin(dist))
            if best is None or dist[j_] < best[0]:
                best = (dist[j_], y[j_])
        if best is None or best[0] > 2 * tol:
            return None
        z = best[1]
        pid = point_id(z)
        split_existing(z, pid, [(on[0], o) for o in on[1:]] or [(on[0], on[0])])
        return pid

    kept, merged, trimmed = [], 0, 0
    for (a, b), Q, closed in curves:
        old = near_graph(Q)
        if old.mean() > 0.8:
            d, i = all_tree.query(Q)
            for i_ in set(all_of[i[d < tol]].tolist()):
                cov = np.mean(cKDTree(Q).query(allp[i_])[0] < tol) if len(allp[i_]) else 0
                if cov > 0.8:
                    segs[i_]["on"] = set(segs[i_]["on"]) | {a, b}
            merged += 1
            continue
        n0 = len(kept)
        if not old.any():
            kept.append(((a, b), Q, closed, (None, None)))
        else:
            trimmed += 1
            idx = np.flatnonzero(~old)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for r in runs:
                i0, i1 = r[0], r[-1]
                e0 = i0 - 1 if i0 > 0 else (len(Q) - 2 if closed and old[-2] else None)
                e1 = i1 + 1 if i1 < len(Q) - 1 else (1 if closed and old[1] else None)
                sub = Q[i0:i1 + 1]
                ends = [None, None]
                if e0 is not None:
                    ends[0] = Q[e0]
                    sub = np.vstack([Q[e0], sub])
                if e1 is not None:
                    ends[1] = Q[e1]
                    sub = np.vstack([sub, Q[e1]])
                if len(sub) >= 3:
                    kept.append(((a, b), sub, False, tuple(ends)))
        new_pts += [dense(c[1], step) for c in kept[n0:]]
        if new_pts:
            new_tree = cKDTree(np.vstack(new_pts))
    # (ends where a run leaves the graph: on the curve it leaves, cut there)
    curves = []
    snapped = 0
    for (a, b), Q, closed, ends in kept:
        Q = Q.copy()
        for side, e in ((0, ends[0]), (-1, ends[1])):
            if e is None:
                continue
            pid = snap(e)
            if pid is not None:
                Q[side] = Pts[pid]
                snapped += 1
        curves.append(((a, b), Q, closed))
    by_surf.clear()
    for i_, s_ in enumerate(segs):
        for m in s_["on"]:
            by_surf[m].add(i_)
    added = n_cuts = 0
    t_ = time.time()
    for c_i, ((a, b), Q, closed) in enumerate(curves):
        if c_i % 25 == 0:
            log(f"    cutting curve {c_i}/{len(curves)}: {n_cuts} cuts, {len(segs)} segments, {time.time() - t_:.0f}s")
        A_, B_ = surfs[a], surfs[b]
        cuts = []
        on = {a, b}
        for m, C in enumerate(surfs):
            if m in (a, b):
                continue
            fv = np.array([C.f(p) for p in Q])
            sc = np.array([np.linalg.norm(C.g(p)) + 1e-300 for p in Q])
            # (on it all along, but perhaps at an end snapped or run on a little way)
            if np.mean(np.abs(fv / sc) < 1e-9) > 0.9 and np.abs(fv / sc).max() < 1e-4:
                on.add(m)
                continue
            for i in range(len(Q) - 1):
                if fv[i] * fv[i + 1] < 0 and max(abs(fv[i]) / sc[i], abs(fv[i + 1]) / sc[i + 1]) > 1e-7:
                    t = fv[i] / (fv[i] - fv[i + 1])
                    z, ok = E.newton(Q[i] + t * (Q[i + 1] - Q[i]), [A_, B_, C])
                    if ok and E.inside(z, 1e-10):
                        cuts.append((i + t, z, m))
            for i in (0, len(Q) - 1):
                if abs(fv[i]) / sc[i] < 1e-10:
                    cuts.append((float(i), Q[i], m))
        if not closed:
            cuts += [(0.0, Q[0], None), (len(Q) - 1.0, Q[-1], None)]
        cuts.sort(key=lambda c: c[0])
        n_cuts += len(cuts)
        if len(cuts) > 100:
            log(f"    curve {c_i} on {a},{b} (also {sorted(on - {a, b})}): {len(cuts)} cuts by "
                f"{Counter(c[2] for c in cuts).most_common(4)}")
        ids = []
        for t, z, m in cuts:
            pid = point_id(z)
            if ids and ids[-1][1] == pid:
                continue
            ids.append((t, pid))
            if m is not None:
                split_existing(z, pid, [(a, m), (b, m)] + [(m, o) for o in on if o not in (a, b)])
        for (t0, p0), (t1, p1) in zip(ids, ids[1:]):
            i0, i1 = int(np.ceil(t0 + 1e-9)), int(np.floor(t1 - 1e-9))
            pts = np.vstack([Pts[p0], Q[i0:i1 + 1], Pts[p1]])
            if p0 == p1 and len(pts) < 4:
                continue
            add_seg({"ends": (p0, p1), "pts": pts, "on": set(on), "key": None, "sig": None, "added": True})
            added += 1
    log(f"  graph completed: {sum(len(v) for v in seeds.values())} rim points on {len(seeds)} surface pairs "
        f"with no curve; {len(curves) + merged} curves traced ({merged} already there, {trimmed} partly; {snapped} ends snapped), {n_cuts} cuts, {added} segments added ({time.time() - t_:.0f}s)")

    # loose ends: a trace stops short of a singular point (where its two surfaces turn tangent, often a corner
    # where many surfaces meet), so its end hangs a step or so from where it should join. Each loose end is run
    # on to its true end: a point of the graph on all of the curve's surfaces just ahead of it, else the point
    # ahead where a third surface crosses (solved exactly), which the curves through it are cut at.
    rel = lambda S_, x: abs(S_.f(x)) / (np.linalg.norm(S_.g(x)) + 1e-300)
    R = 1.5e-2
    joined = solved = done = 0
    loose = []
    for rnd in range(3):
        deg = defaultdict(Counter)
        for s_ in segs:
            for m in s_["on"]:
                deg[m][s_["ends"][0]] += 1
                deg[m][s_["ends"][1]] += 1
        loose = []
        for i_, s_ in enumerate(segs):
            for side in (0, 1):
                p = s_["ends"][side]
                if s_["ends"][0] != s_["ends"][1] and any(deg[m][p] == 1 for m in s_["on"]):
                    loose.append((i_, side))
        if not loose:
            break
        tree_p = cKDTree(np.array(Pts))
        done = 0
        for i_, side in loose:
            s_ = segs[i_]
            p = s_["ends"][side]
            if not any(deg[m][p] == 1 for m in s_["on"]):
                continue
            Q = np.asarray(s_["pts"], float)
            x, prev = (Q[-1], Q[-2]) if side == 1 else (Q[0], Q[1])
            fwd = x - prev
            fwd = fwd / (np.linalg.norm(fwd) + 1e-300)
            on = sorted(s_["on"])
            best = None
            for q in tree_p.query_ball_point(x, R):
                y = Pts[q]
                v = y - x
                if q == p or q in s_["ends"] or v @ fwd <= 0.2 * np.linalg.norm(v):
                    continue
                if all(rel(surfs[m], y) < 1e-8 for m in on):
                    d = np.linalg.norm(v)
                    if best is None or d < best[0]:
                        best = (d, q, None)
            if best is None:
                pair = sorted(((E.crossing_angle(x, surfs[a], surfs[b]), a, b) for a in on for b in on if a < b),
                              reverse=True)
                if pair:
                    _, a, b = pair[0]
                    for m, C in enumerate(surfs):
                        if m in on:
                            continue
                        for st in (0.0, 0.5 * R):
                            z, ok = E.newton(x + st * fwd, [surfs[a], surfs[b], C])
                            v = z - x
                            d = np.linalg.norm(v)
                            if ok and E.inside(z, 1e-10) and 1e-9 < d < R and v @ fwd > 0.2 * d and \
                                    all(rel(surfs[o], z) < 1e-8 for o in on):
                                if best is None or d < best[0]:
                                    best = (d, None, (z, m))
                                break
            if best is None:
                continue
            d, q, zm = best
            if q is None:
                z, m = zm
                q = point_id(z)
                split_existing(z, q, [(o, m) for o in on])
                solved += 1
            else:
                joined += 1
            y = Pts[q]
            n_ = max(1, int(np.ceil(d / 2e-3)))
            ext = x + (y - x) * (np.arange(1, n_ + 1) / n_)[:, None]
            ext[-1] = y
            if side == 1:
                s_["pts"] = np.vstack([Q, ext])
                s_["ends"] = (s_["ends"][0], q)
            else:
                s_["pts"] = np.vstack([ext[::-1], Q])
                s_["ends"] = (q, s_["ends"][1])
            for m in s_["on"]:
                deg[m][p] -= 1
                deg[m][q] += 1
            done += 1
        if not done:
            break
    log(f"  loose ends: {joined} joined to a point of the graph, {solved} run on to a crossing solved there; "
        f"{len(loose) - done} left")


def rework(group, log=print):
    path = os.path.join(HERE, f"a4_pieces_{group}.pkl")
    X = pickle.load(open(path, "rb"))
    surfs = [P.surf_of(n) for n in X["surfaces"]]
    X["points"] = [np.asarray(p, float) for p in X["points"]]
    Pts = X["points"]                            # (cuts add exact points to it)
    clean_graph(X, surfs, log)
    complete_graph(X, surfs, log)
    by_surf = defaultdict(list)
    for fi, F in enumerate(X["faces"]):
        by_surf[F["surface"]].append(fi)
    new = defaultdict(list)
    stats = Counter()
    def surf_job(k):
        fis = by_surf[k]
        out, stats = [], Counter()
        S = surfs[k]
        planar = k < 4 or S.expr.as_poly().total_degree() == 1
        faces = faces_on(k, S, X["segments"], Pts)
        meshes = {}
        for rnd in range(int(os.environ.get("A4_CUT_ROUNDS", "1"))):   # (cuts: off unless asked)
            bad = []
            for q, (loops, hc) in enumerate(faces):
                r_ = tri_face(S, loops, planar)
                if r_ is None:
                    bad.append(hc)
                else:
                    meshes[q] = r_
            if not bad or rnd == int(os.environ.get("A4_CUT_ROUNDS", "1")) - 1:
                break
            did = [cut(k, S, X["segments"], Pts, surfs, hc) for hc in bad]
            stats["cuts"] += sum(did)
            if not any(did):
                break
            faces, meshes = faces_on(k, S, X["segments"], Pts), {}
        # (the marching pieces on this surface, to say which piece each face is)
        cent, owner = [], []
        for fi in fis:
            F = X["faces"][fi]
            V = np.asarray(F["mesh"]["verts"], float)
            T = np.asarray([t for t in F["mesh"]["faces"] if not set(F.get("filled", [])) & set(t)], int)
            if len(T):
                cent.append(V[T].mean(1))
                owner += [fi] * len(T)
        if not cent:
            return out, stats
        tree = cKDTree(np.vstack(cent))
        owner = np.array(owner)
        for q, (loops, hc) in enumerate(faces):
            r_ = meshes.get(q)
            if r_ is None and len(loops) == 1:
                # (too curved to lay flat: on the marching piece its loop runs round, laid on a disk)
                Lp = loops[0][:: max(1, len(loops[0]) // 300)]
                best = None
                for fi in fis:
                    Vm = np.asarray(X["faces"][fi]["mesh"]["verts"], float)
                    if not len(Vm):
                        continue
                    dm = cKDTree(Vm).query(Lp)[0].mean()
                    if best is None or dm < best[0]:
                        best = (dm, fi)
                if best is not None and best[0] < H:
                    tu = safe(tutte_exact, X["faces"][best[1]], S, loops[0], planar)
                    if tu is not None:
                        out.append((best[1], tu[0], np.asarray(tu[1])))
                        stats["faces laid on their piece's disk"] += 1
                        continue
            if r_ is None:
                stats["face not triangulated"] += 1
                continue
            V, T = np.asarray(r_[0]), np.asarray(r_[1])
            C = V[T].mean(1)
            d, j = tree.query(C)
            near = d < 0.5 * H
            if near.mean() < 0.5:
                stats["face in no wall piece"] += 1
                continue
            fi = Counter(owner[j[near]]).most_common(1)[0][0]
            # (most of the face, by area, on that piece: else it reaches over others, a face bigger than any piece
            #  or the cycle round the outside of a cluster of curves)
            e1, e2 = V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]]
            ta = np.sqrt(np.maximum((e1 * e1).sum(1) * (e2 * e2).sum(1) - (e1 * e2).sum(1) ** 2, 0))
            if ta[near & (owner[j] == fi)].sum() < 0.85 * ta.sum():
                stats["face over several pieces"] += 1
                continue
            out.append((fi, V, T))
            stats["faces"] += 1
        return out, stats

    def piece_job(fi):
        F = X["faces"][fi]
        out, stats = [], Counter()
        k = F["surface"]
        S = surfs[k]
        planar = k < 4 or S.expr.as_poly().total_degree() == 1
        rim = P.rim_of(dict(F))
        if not rim:
            return out, stats
        Vm = np.asarray(P.weld(G := {"mesh": F["mesh"], "filled": F.get("filled", [])}) or G["mesh"]["verts"], float)
        R = [Vm[a] + u * (Vm[b] - Vm[a]) for a, b in P.rim_of(G) for u in np.linspace(0, 1, 6)]
        tree = cKDTree(np.array(R))
        h = H
        sub_ids = set()
        for i_, s_ in enumerate(X["segments"]):
            if k not in s_["on"]:
                continue
            Q = np.asarray(s_["pts"], float)
            d = tree.query(Q[1:-1] if len(Q) > 2 else Q)[0]
            if len(d) and np.mean(d < h) > 0.85:
                sub_ids.add(i_)
        if not sub_ids:
            return out, stats
        view = [dict(s_, on=(s_["on"] if i in sub_ids else set(s_["on"]) - {k})) for i, s_ in enumerate(X["segments"])]
        old_a = mesh_area(G["mesh"]["verts"], [t for t in G["mesh"]["faces"] if not set(G.get("filled", [])) & set(t)])
        for loops, hc in faces_on(k, S, view, Pts):
            Lp = loops[0][:: max(1, len(loops[0]) // 300)]
            if cKDTree(Vm).query(Lp)[0].mean() > 0.25 * h:
                continue
            r_ = tri_face(S, loops, planar) or safe(tutte_exact, F, S, loops[0], planar)
            if r_ is None:
                continue
            if abs(mesh_area(*r_) - old_a) < AREA_TOL * old_a:
                out.append((fi, np.asarray(r_[0]), np.asarray(r_[1])))
                stats["faces from the piece's own rim curves"] += 1
                break
        return out, stats

    def gather(res):
        for r in res.values():
            if r[0] is None:
                stats[f"surface or piece lost ({r[1]})"] += 1
                continue
            for fi, V, T in r[0]:
                new[fi].append((np.asarray(V), np.asarray(T)))
            stats.update(r[1])
    # (each surface in a worker of its own: triangle can crash on a bad loop)
    gather(P.run_forked(sorted(by_surf), surf_job, timeout=3000, log=log))
    # (a piece the walk gave no face: its graph walked again on the curves along its own rim only)
    gather(P.run_forked([fi for fi in range(len(X["faces"])) if fi not in new], piece_job, timeout=600, log=log))

    for fi, F in enumerate(X["faces"]):
        parts = new.get(fi)
        old_a = mesh_area(F["mesh"]["verts"], [t for t in F["mesh"]["faces"]
                                               if not set(F.get("filled", [])) & set(t)])
        if not parts:
            stats["piece kept (no exact face)"] += 1
            continue
        Vs, Ts = [], []
        for V, T in parts:
            o = len(Vs)
            Vs.extend(V)
            Ts.extend((T + o).tolist())
        a = mesh_area(Vs, Ts)
        if abs(a - old_a) > AREA_TOL * max(old_a, 1e-9):
            stats["piece kept (area off)"] += 1
            log(f"  piece {fi} ({F['key'][:6]}): exact faces {a:.5f}, marching {old_a:.5f}: kept")
            continue
        F["mesh"] = {"verts": list(Vs), "faces": Ts}
        F["filled"] = []
        F["exact_loop"] = True
        stats["piece from exact faces"] += 1
    log(f"{group}: {dict(stats)}")
    pickle.dump(X, open(path, "wb"))


if __name__ == "__main__":
    for g in sys.argv[1:] or ["iodecaic"]:
        rework(g)
