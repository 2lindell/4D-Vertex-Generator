"""Exact edges and corners for a curved wall (xsurfaces.json), from equations instead of the classifier.

Each run of a wall's outline (one shape past it) lies on the wall's surface and on a second known surface: another
wall's surface, a cell face or mirror, a plane or cone of an exact conic, or an image of one of these under the
2400-element group. That second surface is found by testing every candidate on the run's points; the edge is then the
exact intersection of the two (points by Newton on both equations), each corner the exact meeting of the wall with the
surfaces of the two runs either side, and the edge between two corners is marched along the intersection. The mesh is
then cut back from the edge and the strip between it and the exact edge refilled.

The classifier is not needed at all, so a wall takes seconds rather than hours (the mesher's own edge points are only
bisected 6 times, ~2e-4: the cause of the ragged corners).

    python edge_snap.py X42 [out.json]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

import xsurface
from cell_atlas2 import ear_clip

PHI = (1 + 5 ** 0.5) / 2
FILL = os.environ.get("FILL") == "1"
NAMED = os.environ.get("NAMED") == "1"   # pick each edge's second surface from the shape named on it (edge_lines.json)
CORNERS = []                             # the exact corners of the wall being done (kept by fill_mesh)
DROPPED_LOOPS = []                       # outline loops fill_mesh found to be fragments inside the main one     # rebuild the wall's mesh inside its exact outline (fill_mesh)


# ------------------------------------------------------------------------------------------------ polynomials
class Poly:
    """A homogeneous polynomial in beta: monomials (index tuples) and coefficients."""

    def __init__(self, mons, coeffs, name):
        self.mons = [tuple(m) for m in mons]
        self.coeffs = np.asarray(coeffs, float)
        self.name = name

    def __call__(self, B):
        B = np.atleast_2d(B)
        out = np.zeros(len(B))
        for c, m in zip(self.coeffs, self.mons):
            out += c * np.prod(B[:, list(m)], axis=1) if m else c
        return out

    def grad(self, B):
        B = np.atleast_2d(B)
        g = np.zeros_like(B)
        for c, m in zip(self.coeffs, self.mons):
            for k in range(len(m)):
                rest = list(m[:k] + m[k + 1:])
                g[:, m[k]] += c * (np.prod(B[:, rest], axis=1) if rest else 1.0)
        return g


class Image:
    """A surface's image under a group element: x -> p(Ginv x)."""

    def __init__(self, p, Ginv, name):
        self.p, self.Ginv, self.name = p, Ginv, name

    def __call__(self, B):
        return self.p(np.atleast_2d(B) @ self.Ginv.T)

    def grad(self, B):
        return self.p.grad(np.atleast_2d(B) @ self.Ginv.T) @ self.Ginv


def group():
    from cellframe import TINV, T as T_cell
    from normalizer import extended_group
    return np.einsum("ij,gjk,kl->gil", TINV, np.array(extended_group(), float), np.array(T_cell, float).T)


def base_surfaces():
    """Every known surface: the curved walls', the planes of the exact patches and their edges, the cell's faces and
    splitting mirror, and the cones of the exact conics."""
    out, seen = [], set()

    def add(p):
        key = tuple(sorted((m, round(c / np.max(np.abs(p.coeffs)), 9)) for m, c in zip(p.mons, p.coeffs)))
        key2 = tuple(sorted((m, -v) for m, v in key))
        if key in seen or key2 in seen:
            return
        seen.add(key)
        out.append(p)
    tids = list(json.load(open("xsurfaces.json")))
    if os.path.exists("xsurface_equations.json"):
        tids += list(json.load(open("xsurface_equations.json")))
    for tid in dict.fromkeys(tids):
        sf = xsurface.surface(tid)
        add(Poly(sf["mons"], sf["coeffs"], f"{tid}'s surface"))
    for i in range(4):
        add(Poly([(i,)], [1.0], f"face β{i + 1} = 0"))
    add(Poly([(2,), (3,)], [1.0, -1.0], "mirror β3 = β4"))
    P = json.load(open("fexact_patches.json"))
    for key, pt in P.items():
        n = json.loads(pt["normal"]) if isinstance(pt["normal"], str) else pt["normal"]
        add(Poly([(i,) for i in range(4) if abs(n[i]) > 1e-12], [n[i] for i in range(4) if abs(n[i]) > 1e-12], f"plane of {key}"))
        edges = json.loads(pt["edges"]) if isinstance(pt["edges"], str) else pt["edges"]
        for e in edges:
            m = e[0]
            add(Poly([(i,) for i in range(4) if abs(m[i]) > 1e-12], [m[i] for i in range(4) if abs(m[i]) > 1e-12], f"plane {e[2]}"))
        for c in pt.get("conics") or []:
            v = c["vars"]
            mons = [(v[0], v[0]), (v[1], v[1]), (v[2], v[2]), (v[0], v[1]), (v[0], v[2]), (v[1], v[2])]
            add(Poly(mons, c["coeffs"], f"cone of the {c['label']} conic"))
    return out


# ------------------------------------------------------------------------------------------------ geometry
def _golden_points():
    """Points of the cell with golden coordinates from a small set (its corners, centres, A1 = (0, 0, 1, 1), ...)."""
    import itertools
    V = [0.0, 1 / PHI ** 2, 1 / PHI, 0.5, 1.0, PHI, PHI ** 2, 2.0]
    out = {}
    for t in itertools.product(V, repeat=4):
        if sum(t) > 0:
            b = np.array(t) / sum(t)
            out[tuple(np.round(b, 12))] = b
    return list(out.values())


GOLDEN = _golden_points()


def dist(s, X):
    """Distance estimate of points X from surface s (|f| / |grad f| in the sum-zero directions)."""
    g = s.grad(X)
    g = g - g.mean(axis=1, keepdims=True)
    return np.abs(s(X)) / np.maximum(np.linalg.norm(g, axis=1), 1e-300)


def newton(surfs, x, iters=60):
    """The point on all the surfaces (and sum = 1) nearest x: minimal-norm Newton steps."""
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        r = np.array([s(x)[0] for s in surfs] + [x.sum() - 1])
        J = np.vstack([s.grad(x)[0] for s in surfs] + [np.ones(4)])
        if np.abs(r).max() < 1e-15:
            break
        dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        x += dx
        if np.abs(dx).max() < 1e-16:
            break
    return x


def grad1(W, x):
    return W.grad(np.atleast_2d(x))[0]


def project_w(W, x):
    """x moved onto the surface W (sum = 1 kept)."""
    return newton([W], x)


def in_half(b):
    return b.min() > -1e-7 and b[2] >= b[3] - 1e-7



def extend_to_meet(W, SA, SB, start, toward, step=0.002, max_len=0.6):
    """Follow edge A (W and SA) on from start, in the direction it was heading (toward), until it meets edge B (W
    and SB): the corner the mesh never reached (the mesher broke up in a thin tip). The path and the corner, or None."""
    x = newton([W, SA], start)
    path, d = [x], np.asarray(toward, float)
    for _ in range(int(max_len / step)):
        J = np.vstack([W.grad(x)[0], SA.grad(x)[0], np.ones(4)])
        t = np.linalg.svd(J)[2][-1]
        t = t if t @ d > 0 else -t
        y = newton([W, SA], x + step * t)
        if not np.isfinite(y).all() or np.linalg.norm(y - x) > 3 * step or not in_half(y):
            return None
        q = newton([W, SB], y)
        if np.isfinite(q).all() and np.linalg.norm(q - y) < 2 * step:
            c = newton([W, SA, SB], (q + y) / 2)
            if np.linalg.norm(c - y) < 4 * step and in_half(c):
                path.append(c)
                return path, c
        d = y - x
        path.append(y)
        x = y
    return None

def march(W, S, a, b, step=0.003, guide=None):
    """Points along the curve W ∩ S from a toward b (both on it); None if it does not arrive."""
    pts = [a]
    x = a.copy()
    total = np.linalg.norm(b - a)
    for _ in range(int(6 * max(total, 0.01) / step) + 50):
        if np.linalg.norm(b - x) < 1.5 * step:
            pts.append(b)
            return pts
        J = np.vstack([W.grad(x)[0], S.grad(x)[0], np.ones(4)])
        t = np.linalg.svd(J)[2][-1]
        want = (guide(x) if guide is not None else b) - x
        if t @ want < 0:
            t = -t
        y = newton([W, S], x + step * t)
        if np.linalg.norm(y - x) > 3 * step or not np.isfinite(y).all():
            if np.linalg.norm(b - x) < 8 * step:      # (into a singular corner such as A1: finish at it)
                pts.append(b)
                return pts
            return None
        pts.append(y)
        x = y
    return None



def tip(W, c, SA, SB, la, lb, pa, pb, tris, outline):
    """A sharp corner rebuilt: the mesh is cut off across the wall at the line from pa (on edge A) to pb (on edge B),
    and the tip beyond is filled with rows of points running from the cut into the corner c, blended between the two
    exact edges and put on the wall's surface; the outline follows the exact edges into the corner."""
    m0 = (pa + pb) / 2
    u = c - m0
    u /= np.linalg.norm(u)
    rad = 1.5 * max(np.linalg.norm(c - pa), np.linalg.norm(c - pb))     # (only near the tip: the wall may come
    zone = lambda p: (np.asarray(p) - m0) @ u > 1e-9 and np.linalg.norm(np.asarray(p) - c) < rad   # back past it)
    keep = [t for t in tris if not any(zone(v) for v in t)]
    key = lambda p: tuple(np.round(p, 9))
    cnt, at = {}, {}
    for t in keep:
        for i in range(3):
            e = tuple(sorted((key(t[i]), key(t[(i + 1) % 3]))))
            cnt[e] = cnt.get(e, 0) + 1
            at[key(t[i])] = np.asarray(t[i])
    adj = {}
    for e, n in cnt.items():
        if n == 1 and all((at[v] - m0) @ u > -0.03 for v in e):
            adj.setdefault(e[0], []).append(e[1])
            adj.setdefault(e[1], []).append(e[0])
    if not adj:
        return tris, outline
    near = lambda q: min(adj, key=lambda v: np.linalg.norm(at[v] - q))
    s0, s1 = near(pa), near(pb)
    # the cut: the shortest path between them along the kept mesh's edge near the cut
    import heapq
    dist_, prev, hq = {s0: 0.0}, {}, [(0.0, s0)]
    while hq:
        d, v = heapq.heappop(hq)
        if v == s1:
            break
        if d > dist_.get(v, 1e9):
            continue
        for w in adj[v]:
            nd = d + np.linalg.norm(at[w] - at[v])
            if nd < dist_.get(w, 1e9):
                dist_[w], prev[w] = nd, v
                heapq.heappush(hq, (nd, w))
    if s1 not in dist_:
        return tris, outline
    chain = [s1]
    while chain[-1] != s0:
        chain.append(prev[chain[-1]])
    chain = [at[v] for v in chain[::-1]]
    EA = march(W, SA, chain[0], c) or [chain[0], c]
    EB = march(W, SB, chain[-1], c) or [chain[-1], c]

    def resample(P, n):
        P = np.asarray(P)
        sl = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        out_ = []
        for t in np.linspace(0, sl[-1], n + 1):
            k = min(np.searchsorted(sl, t, side="right") - 1, len(P) - 2)
            f = (t - sl[k]) / max(sl[k + 1] - sl[k], 1e-300)
            out_.append(P[k] + f * (P[k + 1] - P[k]))
        return out_
    L = max(np.linalg.norm(c - chain[0]), np.linalg.norm(c - chain[-1]))
    N = max(8, int(np.ceil(L / 0.004)))
    A_, B_ = resample(EA, N), resample(EB, N)
    Cs = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.array(chain), axis=0), axis=1))]
    sj = Cs / max(Cs[-1], 1e-300)
    base = [chain[j] - ((1 - sj[j]) * chain[0] + sj[j] * chain[-1]) for j in range(len(chain))]
    rows = []
    for r in range(N + 1):
        t = r / N
        row = []
        for j in range(len(chain)):
            q = (1 - sj[j]) * A_[r] + sj[j] * B_[r] + (1 - t) * base[j]
            row.append(c if r == N else (chain[j] if r == 0 else newton([W], q)))
        if 0 < r < N:
            row[0], row[-1] = A_[r], B_[r]
        rows.append(row)
    new_t = []
    for r in range(N):
        for j in range(len(chain) - 1):
            a, b, cc, d = rows[r][j], rows[r][j + 1], rows[r + 1][j + 1], rows[r + 1][j]
            new_t.append(np.array([a, b, cc]))
            if r < N - 1:
                new_t.append(np.array([a, cc, d]))
    # the outline: its points in the cut-off tip go, the exact edges into the corner come in
    out_lines = []
    for l in outline:
        P = [np.asarray(q["beta"]) for q in l]
        inz = [zone(p) for p in P]
        if not any(inz):
            out_lines.append(l)
            continue
        body = l[:-1] if len(l) > 1 and np.allclose(P[0], P[-1]) else l
        inz = inz[:len(body)]
        n = len(body)
        if all(inz):                                             # (a scrap wholly in the tip: the new rows cover it)
            continue
        k = next(i for i in range(n) if not inz[i] and inz[(i + 1) % n])   # last point before the tip
        first = body[k]
        side_a = np.linalg.norm(np.asarray(first["beta"]) - chain[0]) < np.linalg.norm(np.asarray(first["beta"]) - chain[-1])
        ins = ([{"beta": p.tolist(), "beyond": la} for p in A_[1:]] + [{"beta": p.tolist(), "beyond": lb} for p in B_[::-1][1:-1]]
               if side_a else
               [{"beta": p.tolist(), "beyond": lb} for p in B_[1:]] + [{"beta": p.tolist(), "beyond": la} for p in A_[::-1][1:-1]])
        rot = body[k + 1:] + body[:k + 1]                        # start just after the last point before the tip
        rest = [q for q, z in zip(rot, inz[k + 1:] + inz[:k + 1]) if not z]
        new = rest[-1:] + ins + rest[:-1]
        out_lines.append(new + new[:1])
    print(f"    tip at {np.round(c / c.max(), 5).tolist()}: cut back {L:.3f}, {len(new_t)} triangles in {N} rows", flush=True)
    return keep + new_t, out_lines


def fill_mesh(W, outline, T, h=0.012):
    """A fresh mesh inside the exact outline: the outline loops and an even grid of points inside them, in the plane
    that fits the wall best, joined by a Delaunay triangulation (triangles whose middle is outside the outline are
    dropped), then every inner point lifted onto the wall's surface along that plane's normal. No triangle overlaps
    another, and the edge is the exact outline. Falls back to the old mesh if the wall does not lie over that plane."""
    from scipy.spatial import Delaunay
    V = np.asarray(T, float).reshape(-1, 4)
    c0 = V.mean(axis=0)
    basis = np.linalg.svd(np.vstack([np.ones(4)]))[2][1:]               # sum-zero space
    Q = (V - c0) @ basis.T
    E = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2]
    ax2, nrm = E[:2] @ basis, E[2] @ basis                               # chart axes and normal (in beta)
    chart = lambda X: (np.asarray(X) - c0) @ ax2.T
    # the wall must lie over the chart plane: every triangle facing the same way
    G = np.array([W.grad(t.mean(axis=0))[0] for t in np.asarray(T, float)])
    if np.mean(np.sign(G @ nrm) == np.sign(np.median(G @ nrm))) < 0.995:
        print("    fill: the wall folds over its best plane; old mesh kept", flush=True)
        return [np.asarray(t) for t in T]
    loops = []
    for l in outline:
        Pb = np.array([q["beta"] for q in l])
        if len(Pb) > 3 and np.allclose(Pb[0], Pb[-1]):
            Pb = Pb[:-1]
        # (edge points no closer than h/3, so the triangles along the edge are not slivers)
        iscorner = lambda p: any(np.linalg.norm(p - c_) < 1e-12 for c_ in CORNERS)
        keep = [Pb[0]]
        for p in Pb[1:]:
            if iscorner(p):                       # (corners always stay; a point just before one gives way)
                if np.linalg.norm(p - keep[-1]) < h / 6 and not iscorner(keep[-1]) and len(keep) > 1:
                    keep.pop()
                keep.append(p)
            elif np.linalg.norm(p - keep[-1]) > h / 3:
                keep.append(p)
        loops.append(np.array(keep))
    # (a smaller loop lying wholly inside the largest one is a fragment of the old mesh in a stretch the largest now
    # covers, not a hole: it goes)
    if len(loops) > 1:
        big = max(range(len(loops)), key=lambda i: len(loops[i]))
        L2 = chart(loops[big])

        def inside_big(P2):
            res = np.zeros(len(P2), bool)
            x1, y1 = L2[:, 0], L2[:, 1]
            x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
            for a, b, c_, d in zip(x1, y1, x2, y2):
                cr = ((b > P2[:, 1]) != (d > P2[:, 1]))
                xi = a + (P2[:, 1] - b) * (c_ - a) / np.where(d != b, d - b, 1e-300)
                res ^= cr & (P2[:, 0] < xi)
            return res
        keep_i = [i for i, L in enumerate(loops) if i == big or not inside_big(chart(L)).all()]
        loops = [loops[i] for i in keep_i]
        DROPPED_LOOPS[:] = [i for i in range(len(outline)) if i not in keep_i]
    # (near a sharp corner the two edges can still cross: the points closest to the corner go until no two
    # segments of the outline cross)
    def crossings(L2):
        n_ = len(L2)
        segs_ = [(L2[i], L2[(i + 1) % n_]) for i in range(n_)]
        bad = set()
        for i in range(n_):
            for j in range(i + 2, n_):
                if i == 0 and j == n_ - 1:
                    continue
                (p1, p2), (p3, p4) = segs_[i], segs_[j]
                d1 = np.cross(p2 - p1, p3 - p1); d2 = np.cross(p2 - p1, p4 - p1)
                d3 = np.cross(p4 - p3, p1 - p3); d4 = np.cross(p4 - p3, p2 - p3)
                if d1 * d2 < 0 and d3 * d4 < 0:
                    bad.update([i, (i + 1) % n_, j, (j + 1) % n_])
        return bad
    fixed = []
    for L in loops:
        L = list(L)
        for _ in range(40):
            bad = crossings(chart(np.array(L)))
            if not bad:
                break
            # drop the non-corner point of a crossing segment nearest a corner
            cand = [k for k in bad if not any(np.linalg.norm(L[k] - c_) < 1e-12 for c_ in CORNERS)]
            if not cand or len(L) <= 4:
                break
            k = min(cand, key=lambda k: min([np.linalg.norm(L[k] - c_) for c_ in CORNERS] or [0]))
            L.pop(k)
        fixed.append(np.array(L))
    loops = fixed
    # (spurs, an outline doubling straight back on itself, are cut out: the triangulation needs simple loops)
    clean = []
    for L in loops:
        out_ = list(L)
        changed = True
        while changed and len(out_) > 3:
            changed = False
            for i in range(len(out_)):
                a_, b_, c_ = out_[i - 1], out_[i], out_[(i + 1) % len(out_)]
                u_, v_ = chart(b_[None])[0] - chart(a_[None])[0], chart(c_[None])[0] - chart(b_[None])[0]
                if u_ @ v_ < -0.95 * np.linalg.norm(u_) * np.linalg.norm(v_) and not any(np.linalg.norm(b_ - c_) < 1e-12 for c_ in CORNERS):
                    out_.pop(i)                   # (a sharp corner is not a spur: it stays)
                    changed = True
                    break
        clean.append(np.array(out_))
    loops = clean
    # (no point twice, in the chart: the triangulator crashes on a repeated vertex)
    seen_, dedup = set(), []
    for L in loops:
        out_ = []
        for p_ in L:
            k_ = tuple(np.round(chart(p_[None])[0], 10))
            if k_ in seen_:
                continue
            seen_.add(k_)
            out_.append(p_)
        if len(out_) >= 3:
            dedup.append(np.array(out_))
    loops = dedup
    bpts = np.vstack(loops)
    B2 = chart(bpts)
    import triangle as tr
    segs, o = [], 0
    for L in loops:
        n_ = len(L)
        segs += [[o + i, o + (i + 1) % n_] for i in range(n_)]
        o += n_
    # quality triangulation inside the loops: no angle under 25 degrees, no triangle bigger than an h-sided one,
    # no new points on the outline (Y): the edge stays the exact one
    try:
        res_ = tr.triangulate({"vertices": B2, "segments": np.array(segs)}, f"pq25Ya{h * h * 0.43:.8f}")
    except Exception as ex:
        print(f"    fill: triangulation failed ({ex}); old mesh kept", flush=True)
        return [np.asarray(t) for t in T]
    P2, tri = res_["vertices"], res_["triangles"]
    # lift: the outline's points are exact already; each new point goes onto the surface from the old mesh
    from scipy.spatial import cKDTree
    Vc = chart(V)
    tree = cKDTree(Vc)
    P4 = []
    for k, p2 in enumerate(P2):
        if k < len(bpts):
            P4.append(bpts[k])
            continue
        # (start on the old mesh, interpolated from its nearest vertices in the chart, then the shortest move onto
        # the surface: along a fixed line the surface can have other sheets, and a point could jump to one)
        # straight along the chart normal (so the point keeps its place in the chart and no triangle folds over),
        # from the height of the old mesh there; only if that fails, the shortest move from the old mesh
        dd, ii = tree.query(p2, k=4)
        w_ = 1.0 / np.maximum(dd, 1e-12)
        xo = (V[ii] * w_[:, None]).sum(axis=0) / w_.sum()
        x0 = c0 + p2[0] * ax2[0] + p2[1] * ax2[1]
        t0 = t = (xo - x0) @ nrm
        ok = False
        for _ in range(60):
            x = x0 + t * nrm
            f = W(x)[0]
            g = W.grad(x)[0] @ nrm
            if abs(f) < 1e-15:
                ok = True
                break
            if abs(g) < 1e-300:
                break
            t -= f / g
        if not ok or abs(t - t0) > 0.03:
            x = newton([W], xo / xo.sum())
        P4.append(x)
    P4 = np.array(P4)
    G2 = P2[len(bpts):]
    print(f"    fill: {len(bpts)} edge points, {len(G2)} inner points, {len(tri)} triangles", flush=True)
    return [P4[t] for t in tri]

# ------------------------------------------------------------------------------------------------ named edges
def named_edges(tid, every=False):
    """The shape found on each edge of the wall (edge_lines.json, edge_relabel/edge_fill_missing): label past the
    edge -> the shape on it, the most often seen (every: all of them, most seen first)."""
    out = {}
    for past, on, *n in json.load(open("edge_lines.json")).get(tid, []):
        if on in (tid, "ERR") or on.startswith("new:"):
            continue
        out.setdefault(past, {})
        out[past][on] = out[past].get(on, 0) + (n[0] if n else 1)
    if every:
        return {k: sorted(v, key=lambda o: -v[o]) for k, v in out.items()}
    return {k: max(v, key=v.get) for k, v in out.items()}


def shape_points():
    """Points of the drawn line shapes and the dims of all shapes, from the built page (betas from its xyz, ~1e-6)."""
    import re
    page = open("../../assets/symmetry_domain.html").read()
    m = re.search(r'<script[^>]*id="atlas-data"[^>]*>', page)
    D = json.loads(page[m.end():page.index("</script>", m.end())])
    Minv = np.array(D["cellInv"], float)
    tob = lambda q: (lambda b: b / b.sum())(Minv @ np.r_[q[0], q[1], -q[2], 1])
    dims = {t["id"]: t.get("dim") for t in D["types"]}
    pts = {}
    for l in D["xlines"] + D["segments"] + D["rings"] + D["main"]:
        if (dims.get(l["id"]) != "line" and l not in D["rings"] + D["main"]) or len(l["pts"]) < 2:
            continue
        B = np.array([tob(q) for q in l["pts"]])
        dense = [a + t * (b - a) for a, b in zip(B[:-1], B[1:]) for t in np.linspace(0, 1, 6)[:-1]] + [B[-1]]
        pts.setdefault(l["id"], []).extend(dense)
    if os.path.exists("xcurves.json"):
        for lid, cv in json.load(open("xcurves.json")).items():
            pts.setdefault(lid, []).extend(np.asarray(cv["pts"], float))
    return {k: np.array(v) for k, v in pts.items()}, dims


# ------------------------------------------------------------------------------------------------ main
def main(tid, out="xsurfaces.json"):
    X = json.load(open(os.environ.get("EDGE_SRC", "xsurfaces.json")))   # (EDGE_SRC: read the meshes from elsewhere)
    sf = X[tid]
    h = sf.get("h", 0.012)
    s0 = xsurface.surface(tid)
    W = Poly(s0["mons"], s0["coeffs"], f"{tid}'s surface")
    G = group()
    Ginv = np.linalg.inv(G)
    bases = base_surfaces()
    T = np.array(sf["tris"], float)
    V = T.reshape(-1, 4)
    rng = np.random.default_rng(0)
    inner = V[rng.choice(len(V), size=min(40, len(V)), replace=False)]

    GT = np.transpose(Ginv, (0, 2, 1))

    def dists_all(b, P):
        """Distances of points P (n, 4) from every image of base surface b: (len(G), n)."""
        Y = np.einsum("nj,gjk->gnk", P, GT)                 # Ginv x for every element
        f = b(Y.reshape(-1, 4)).reshape(len(G), -1)
        gy = b.grad(Y.reshape(-1, 4)).reshape(len(G), -1, 4)
        gx = np.einsum("gnj,gjk->gnk", gy, Ginv)           # chain rule: grad_x = Ginv^T grad_y
        gx = gx - gx.mean(axis=2, keepdims=True)
        return np.abs(f) / np.maximum(np.linalg.norm(gx, axis=2), 1e-300)

    def degree(b):
        return max(len(m) for m in b.mons)

    EXCLUDE = None

    def best_surface(P):
        """The known surface (or image) the run's points lie on, not the wall's own: of those fitting well (every
        point within 2.5e-4), the one of lowest degree (a conic's plane before its cone: the cone meets the wall in
        other branches too, which throw the corners off), then the closest."""
        good, loose = [], None
        for b in bases:
            D = dists_all(b, P)
            mx, med = D.max(axis=1), np.median(D, axis=1)
            for gi in np.argsort(med)[:8]:
                if med[gi] > 2e-3:
                    break
                img = Image(b, Ginv[gi], b.name + (f" (image {gi})" if gi else ""))
                if np.median(dist(img, inner)) < 1e-6:          # (the wall's own surface)
                    continue
                if EXCLUDE is not None and np.median(dist(img, EXCLUDE)) < 1e-4:   # (fits the whole outline: no use)
                    continue
                cand = (float(med[gi]), float(mx[gi]), img)
                if mx[gi] <= 2.5e-4:
                    good.append((degree(b), cand))
                if loose is None or cand[0] < loose[0]:
                    loose = cand
                break
        if good:
            return min(good, key=lambda dc: (dc[0], dc[1][0]))[1]
        return loose

    named, (LPTS, DIMS) = (named_edges(tid), shape_points()) if NAMED else ({}, ({}, {}))
    named_all = named_edges(tid, every=True) if NAMED else {}
    for past_, ons in named_all.items():     # (a region found on an edge is a sample just off it: a line or wall
        better = [o for o in ons if DIMS.get(o) != "region"]      # found there names the edge)
        if better and DIMS.get(named.get(past_)) == "region":
            named[past_] = better[0]
    # walls on this wall's own surface (X39 and X52 on X10's quadric): a stretch labelled with one is not an edge
    # with it, only the mesher seeing the neighbouring piece of the same surface along the true edge there
    OWN, OWN_LINES = set(), {}
    if NAMED:
        eqs = set(json.load(open("xsurfaces.json"))) | set(json.load(open("xsurface_equations.json")))
        for w in eqs - {tid}:
            sf_ = xsurface.surface(w)
            if np.min(np.median(dists_all(Poly(sf_["mons"], sf_["coeffs"], w), inner), axis=1)) < 1e-9:
                OWN.add(w)
        if OWN:
            print(f"  on {tid}'s own surface too: {sorted(OWN)}", flush=True)
        # (the lines either piece measured on the stretch between them: X29 between X39 and X52)
        EL = json.load(open("edge_lines.json"))
        OWN_LINES = {w: list(dict.fromkeys([e[1] for e in EL.get(tid, []) if e[0] == w and DIMS.get(e[1]) == "line"] +
                                           [e[1] for e in EL.get(w, []) if e[0] == tid and DIMS.get(e[1]) == "line"]))
                     for w in OWN}
    planes = [Poly([(i,)], [1.0], f"face β{i + 1} = 0") for i in range(4)] + \
        [Poly([(i,), (j,)], [1.0, -1.0], f"mirror β{i + 1} = β{j + 1}") for i in range(4) for j in range(i + 1, 4)]

    def named_surface(lab, P):
        """The second surface of an edge from the shape named on it: a line's own exact points pick the surface
        holding them (with the wall's: an image of a known surface); a wall named there gives its own surface's
        images; a cell face gives the faces and mirrors. None when the name does not settle it."""
        on = lab                                  # (labels are the named shapes already: canon below)
        if lab.startswith("face:"):
            cands = [(1, Image(b, np.eye(4), b.name)) for b in planes]
            D_ = [(float(np.median(dist(c, P))), float(dist(c, P).max()), c) for _, c in cands]
            best = min(D_, key=lambda d: d[0])
            return best if best[1] < 1e-3 else None
        if on in LPTS:
            Lp = LPTS[on]
            near = Lp[np.min(np.linalg.norm(Lp[:, None] - P[None], axis=2), axis=1) < 0.01]
            if len(near) >= 3:
                good = []
                for b in bases + planes:
                    Dm = dists_all(b, near).max(axis=1)
                    for gi in np.argsort(Dm)[:6]:
                        if Dm[gi] > 2e-5:
                            break
                        img = Image(b, Ginv[gi], b.name + (f" (image {gi})" if gi else ""))
                        if np.median(dist(img, inner)) < 1e-6:          # (the wall's own surface)
                            continue
                        good.append((degree(b), float(np.median(dist(img, P))), float(Dm[gi]), img))
                # (the run's own points must lie on it too: a few of the line's points, bunched at a corner, lie on
                # many surfaces)
                fit = [g for g in good if g[1] <= 2.5e-4]
                if fit or good:
                    d_, med, mx, img = min(fit, key=lambda g: (g[0], g[1])) if fit else min(good, key=lambda g: g[1])
                    print(f"    edge past {lab}: {on}'s points ({len(near)}) lie on {img.name} (max {mx:.1e})", flush=True)
                    return (med, float(dist(img, P).max()), img)
        if DIMS.get(on) == "wall" and (on in json.load(open("xsurfaces.json")) or on in json.load(open("xsurface_equations.json"))):
            sf_ = xsurface.surface(on)
            b = Poly(sf_["mons"], sf_["coeffs"], f"{on}'s surface")
            Dm = np.median(dists_all(b, P), axis=1)
            gi = int(np.argmin(Dm))
            img = Image(b, Ginv[gi], b.name + (f" (image {gi})" if gi else ""))
            print(f"    edge past {lab}: on the wall {on}, {img.name} (median {Dm[gi]:.1e})", flush=True)
            return (float(Dm[gi]), float(dist(img, P).max()), img) if Dm[gi] < 2e-3 else None
        return None

    new_outline, info = [], []
    MOVES, FANS, TIPS = {}, [], []
    CINFO = []                                 # (FILL) each solved corner with the surfaces and labels of its two edges
    ALLC = []                                  # every solved corner, with the labels past the two edges meeting there
    CORNERS.clear()
    key = lambda p: tuple(np.round(p, 9))
    for li, line in enumerate(sf["outline"]):
        B = np.array([q["beta"] for q in line])
        # (named: a stretch is known by the shape on its edge, so a line in a cell face, labelled now by the face and
        # now by the line, is one edge, and a stretch past a region but along a wall is that wall's edge)
        canon = lambda l_: named.get(l_, l_[5:] if l_.startswith("face:") and l_[5:] in LPTS else l_)
        L = [canon(q["beyond"]) if NAMED else q["beyond"] for q in line]

        LORIG = [q["beyond"] for q in line]                 # (the shapes past the edge, kept for the outline)
        # (other shapes named on the edges a canonical name stands for: a run along two lines is split between them)
        ALT = {}
        for q in line:
            ALT.setdefault(canon(q["beyond"]), [])
            for o in named_all.get(q["beyond"], []):
                if o not in ALT[canon(q["beyond"])]:
                    ALT[canon(q["beyond"])].append(o)
        closed = len(B) > 3 and np.allclose(B[0], B[-1])
        if not closed:
            new_outline.append(line)
            continue
        B, L = B[:-1], L[:-1]
        n = len(B)
        # runs of one label; scraps (< 4 points) join the run before
        runs = []
        for k in range(n):
            if runs and L[k] == runs[-1][0]:
                runs[-1][1].append(k)
            else:
                runs.append([L[k], [k]])
        if len(runs) > 1 and runs[0][0] == runs[-1][0]:
            runs[0][1] = runs.pop()[1] + runs[0][1]
        changed = True
        while changed and len(runs) > 1:
            changed = False
            for r in range(len(runs)):
                if len(runs[r][1]) < 4:
                    runs[r - 1][1] += runs[r][1]
                    runs.pop(r)
                    changed = True
                    break
        # each run is fitted on its own; a run fitting no surface well is a scrap of the mesher in a tight corner: its
        # points join the neighbouring run (not used for that run's fit), and runs left side by side with one label
        # are joined. Then the fits are made on the good points of each run.
        # runs go together by the surface they lie on, not by the label past them (a wall ending on this one's edge
        # changes the label without making a corner): a run lying on the previous run's surface joins it; one
        # fitting no surface well is a scrap of the mesher (it joins the run before, not used for fitting)
        # (a surface through the whole outline, all its edges at once, cannot tell them apart: left out)
        EXCLUDE = None
        whole = best_surface(B[:: max(1, len(B) // 40)])
        if whole is not None and whole[0] <= 2e-5 and len(B) > 40:
            EXCLUDE = B[:: max(1, len(B) // 40)]
            print(f"  outline {li}: {whole[2].name} holds the whole outline; left out", flush=True)
        # (named: a stretch past a piece of the wall's own surface is an edge only along a line measured there whose
        # surface holds it; otherwise the mesher saw the neighbouring piece along the edge it is on, and it takes
        # that edge's label; then neighbours with one label are one run)
        if NAMED and OWN:
            for r_, (lab_, ix) in enumerate(runs):
                if lab_ not in OWN:
                    continue
                Pr = B[ix][:: max(1, len(ix) // 25)]
                new_lab = None
                for ln in OWN_LINES.get(lab_, []):
                    f2 = named_surface(ln, Pr)
                    if f2 is not None and np.median(dist(f2[2], B[ix])) <= 2.5e-4:
                        new_lab = ln
                        break
                if new_lab is None:
                    # (to the neighbouring edge whose surface it lies nearer)
                    best_nb = None
                    for d in (1, -1, 2, -2):
                        l_, ix_ = runs[(r_ + d) % len(runs)]
                        if l_ in OWN:
                            continue
                        Pn = B[ix_][:: max(1, len(ix_) // 25)]
                        f_ = named_surface(l_, Pn) or (best_surface(Pn) if len(ix_) >= 12 else None)
                        dd = float(np.median(dist(f_[2], B[ix]))) if f_ is not None else 1.0
                        if best_nb is None or dd < best_nb[0]:
                            best_nb = (dd, l_)
                    new_lab = best_nb[1] if best_nb else lab_
                runs[r_][0] = new_lab
                print(f"    stretch past {lab_} ({len(ix)} points): {'along ' + new_lab if new_lab in OWN_LINES.get(lab_, []) else 'part of the edge past ' + new_lab}", flush=True)
            joined = []
            for lab_, ix in runs:
                if joined and joined[-1][0] == lab_:
                    joined[-1][1] += ix
                else:
                    joined.append([lab_, list(ix)])
            if len(joined) > 1 and joined[0][0] == joined[-1][0]:
                joined[0][1] = joined.pop()[1] + joined[0][1]
            runs = joined
        if os.environ.get("DEBUG_RUNS"):
            print("  runs:", [(lab_, len(ix)) for lab_, ix in runs], flush=True)
        merged = []                               # [label, all indices, good indices, surface]
        for lab, idx in runs:
            P = B[idx]
            Ps = P[:: max(1, len(P) // 25)]
            prev = next((m for m in reversed(merged) if m[3] is not None), None)

            def turns(prev_idx, idx):
                """Does the edge turn sharply (> 20 degrees) from the previous run into this one?"""
                if len(prev_idx) < 4 or len(idx) < 4:
                    return False
                u = B[prev_idx[-1]] - B[prev_idx[-4]]
                v = B[idx[3]] - B[idx[0]]
                return u @ v < np.cos(np.radians(20)) * np.linalg.norm(u) * np.linalg.norm(v)
            same_name = not NAMED or prev is None or prev[0] == lab
            if prev is not None and same_name and dist(prev[3], Ps).max() <= 5e-4 and not turns(prev[2] or prev[1], idx):
                prev[1] += idx
                prev[2] += idx
                continue
            # (a short run, under 12 points, is not an edge of its own: with so many candidate surfaces one would
            # always fit it by chance)
            f = named_surface(lab, Ps) if NAMED else None
            if NAMED and f is None and len(idx) >= 12:
                # (no surface from the name, a flat wall's say: the known surface holding most of the run, the rest
                # split off below)
                fb = best_surface(Ps)
                if fb is not None and np.median(dist(fb[2], B[idx])) <= 2.5e-4:
                    f = fb
            if f is not None:
                ok_ = dist(f[2], B[idx]) <= 2.5e-4
                if ok_.all() or ok_.sum() < 4:
                    merged.append([lab, list(idx), list(idx), f[2]])
                    continue
                # (the named surface holds only part of the run: the run follows two lines, and the rest is on
                # another shape named on that edge, or else on whatever known surface holds it)
                parts = []
                for k_, g_ in zip(idx, ok_):
                    if parts and parts[-1][0] == g_:
                        parts[-1][1].append(k_)
                    else:
                        parts.append([g_, [k_]])
                for g_, sub in parts:
                    if g_:
                        merged.append([lab, sub, sub, f[2]])
                        continue
                    Psub = B[sub][:: max(1, len(sub) // 25)]
                    alt = None
                    for on2 in ALT.get(lab, []):
                        if on2 != lab:
                            f2 = named_surface(on2, Psub)
                            if f2 is not None and dist(f2[2], B[sub]).max() <= 2.5e-4:
                                alt = (on2, f2)
                                break
                    if alt is None and len(sub) >= 12:
                        f2 = best_surface(Psub)
                        if f2 is not None and f2[1] <= 2.5e-4:
                            alt = (lab + "'", f2)
                    if alt is not None:
                        print(f"    run past {lab}: {len(sub)} of its points are on {alt[0]} ({alt[1][2].name})", flush=True)
                        merged.append([alt[0], sub, sub, alt[1][2]])
                    elif merged:
                        merged[-1][1] += sub
                    else:
                        merged.append([lab, sub, [], None])
                continue
            f = best_surface(Ps) if len(idx) >= 12 or not merged else None
            if f is None or f[1] > 2.5e-4:
                if merged:
                    merged[-1][1] += idx
                else:
                    merged.append([lab, list(idx), [], None])
                continue
            merged.append([lab, list(idx), list(idx), f[2]])
        # (the outline is a loop: the last run may continue the first)
        if len(merged) > 1 and merged[0][3] is not None and merged[-1][3] is not None:
            P = B[merged[-1][2]] if merged[-1][2] else B[merged[-1][1]]
            if dist(merged[0][3], P[:: max(1, len(P) // 25)]).max() <= 5e-4 and merged[0][3] is merged[-1][3]:
                last = merged.pop()
                merged[0] = [merged[0][0], last[1] + merged[0][1], last[2] + merged[0][2], merged[0][3]]
        if merged and merged[0][3] is None and len(merged) > 1:
            first = merged.pop(0)
            merged[-1][1] += first[1]
        runs = [m[:3] for m in merged]
        orig = lambda r_: Counter(LORIG[k] for k in runs[r_][1]).most_common(1)[0][0] if NAMED else runs[r_][0]
        fits = [None if m[3] is None else (float(np.median(dist(m[3], B[m[2]]))), float(dist(m[3], B[m[2]]).max()), m[3])
                for m in merged]
        surf = []
        for (lab, idx, *_), bs in zip(runs, fits):
            surf.append(bs[2] if bs else None)
            info.append((li, lab, len(idx), None if bs is None else (bs[2].name, f"{bs[0]:.1e}", f"{bs[1]:.1e}")))
            print(f"  outline {li}: run past {lab} ({len(idx)} points): "
                  f"{'no surface found' if bs is None else bs[2].name + f', median distance {bs[0]:.1e}, max {bs[1]:.1e}'}", flush=True)
        if len(runs) == 1:
            if surf[0] is None:
                new_outline.append(line)
                continue
            for k in runs[0][2]:
                q = newton([W, surf[0]], B[k])
                if np.linalg.norm(q - B[k]) < 1e-3:
                    MOVES[key(B[k])] = q
            new_outline.append(line)
            continue
        # corners: the wall and the surfaces of the two runs meeting there
        R = len(runs)
        corners = []
        for r in range(R):
            a, b = runs[r], runs[(r + 1) % R]
            guess = (B[a[1][-1]] + B[b[1][0]]) / 2
            ok = False
            if surf[r] is not None and surf[(r + 1) % R] is not None:
                S3 = [W, surf[r], surf[(r + 1) % R]]
                c = newton(S3, guess)
                # (a corner the mesh never reached, a sharp tip, can be well away from the runs' ends: accepted if
                # both edges lead to it, checked when they are marched; and only if it is on all three surfaces:
                # Newton only creeps up on a singular point, A1 say, or wanders along a line two of them share)
                on3 = lambda x: np.isfinite(x).all() and max(float(dist(s3, x)[0]) for s3 in S3) < 1e-10
                ok = np.linalg.norm(c - guess) < 0.15 and in_half(c) and on3(c)
                # (an exact golden point on all three surfaces, within 0.01 of the corner found, or within 0.03 of
                # where the two runs meet when none was found: a corner at a special point of the cell, A1 say,
                # which Newton only creeps up on)
                near = [g_ for g_ in GOLDEN if (ok and np.linalg.norm(g_ - c) < 0.01 or not ok and np.linalg.norm(g_ - guess) < 0.03)
                        and in_half(g_) and on3(g_)]
                if near:
                    c = min(near, key=lambda g_: np.linalg.norm(g_ - (c if ok else guess)))
                    ok = True
                    print(f"    corner at the golden point {np.round(c / c.max(), 5).tolist()}", flush=True)
                print(f"    newton corner {np.round(c, 6).tolist()} moved {np.linalg.norm(c - guess):.3g} in half {in_half(c)}", flush=True)
                if not ok:
                    # (two edges meeting at a sharp tip are nearly tangent there and Newton on all three surfaces
                    # fails: alternate between the two edges instead, from the ends of their good stretches)
                    q = B[a[2][-1]] if len(a) > 2 and a[2] else B[a[1][-1]]
                    for _ in range(300):
                        p_ = newton([W, surf[r]], q)
                        q = newton([W, surf[(r + 1) % R]], p_)
                        if np.linalg.norm(p_ - q) < 1e-11:
                            break
                    c = (p_ + q) / 2
                    print(f"    alternating corner {np.round(c, 6).tolist()} gap {np.linalg.norm(p_ - q):.2g} moved {np.linalg.norm(c - guess):.3g}", flush=True)
                    ok = np.linalg.norm(p_ - q) < 1e-8 and np.linalg.norm(c - guess) < 0.15 and in_half(c) and \
                        max(float(dist(s3, c)[0]) for s3 in S3) < 1e-8
            else:
                c = guess
            if ok:                                # (at a singular point, A1 say, Newton only creeps up on it: snap
                c = c.copy()                      # coordinates within 1e-4 of the cell's own values onto them)
                c[np.abs(c) < 1e-4] = 0.0
                if abs(c[2] - c[3]) < 1e-4:
                    c[2] = c[3] = (c[2] + c[3]) / 2
                c = c / c.sum()
            corners.append(c if ok else None)
            if ok:
                ALLC.append([c.tolist(), orig(r), orig((r + 1) % R)])
            print(f"  corner {li}.{r} ({a[0][:12]} | {b[0][:12]}): "
                  f"{np.round(c / c.max(), 5).tolist() if ok else 'not solved'}", flush=True)
        if FILL:
            # a corner not solved where the wall runs out in a thin tip the mesh broke up in: edge A is followed on
            # until it meets edge B, and that is the corner
            extended = set()
            # (also a solved corner with pieces of the broken-up mesh just past it: the wall goes on past them)
            others = [np.array([q["beta"] for q in l]) for j, l in enumerate(sf["outline"]) if j != li]
            others = np.vstack(others) if others else np.zeros((0, 4))
            for r in range(R):
                c_ = corners[r]
                if c_ is None or not len(others) or len(runs[r][2]) <= 3:
                    continue
                A_ = [B[k] for k in runs[r][2]]
                ahead = (others - c_) @ (A_[-1] - A_[-4])
                near_ = np.linalg.norm(others - c_, axis=1)
                if np.any((near_ < 0.05) & (ahead > 0)):
                    corners[r] = None
            for r in range(R):
                if corners[r] is None and surf[r] is not None and surf[(r + 1) % R] is not None and len(runs[r][2]) > 3:
                    A_ = [B[k] for k in runs[r][2]]
                    e = extend_to_meet(W, surf[r], surf[(r + 1) % R], A_[-1], A_[-1] - A_[-4])
                    if e is not None:
                        corners[r] = e[1]
                        ALLC.append([e[1].tolist(), orig(r), orig((r + 1) % R)])
                        extended.update([r, (r + 1) % R])
                        print(f"  corner {li}.{r} reached by following the edge on: {np.round(e[1] / e[1].max(), 5).tolist()}", flush=True)
            # the outline drawn afresh: each run's exact edge marched from corner to corner (or its own points put
            # on both surfaces where it cannot be marched)
            loop = []
            for r in range(R):
                lab_, s_ = runs[r][0], surf[r]
                P = [B[k] for k in runs[r][1]]
                ca, cb = corners[r - 1], corners[r]
                if s_ is None:
                    seg = ([ca] if ca is not None else []) + P
                else:
                    Pp = [newton([W, s_], p_) for p_ in (runs[r][2] and [B[k] for k in runs[r][2]] or P)]
                    st, en = (ca if ca is not None else Pp[0]), (cb if cb is not None else Pp[-1])
                    # (a marched edge must follow the run: from a point where branches of the intersection cross,
                    # the cell's centre say, it can set off along the wrong one)
                    Pa = np.array(P)
                    follows = lambda sg: sg is not None and max(np.min(np.linalg.norm(np.array(sg) - q_, axis=1)) for q_ in Pa) < 3 * h \
                        and (r in extended or max(np.min(np.linalg.norm(Pa - q_, axis=1)) for q_ in sg) < 3 * h)
                    seg = march(W, s_, st, en)
                    if not follows(seg):
                        seg = march(W, s_, en, st)
                        seg = seg[::-1] if seg is not None else None
                    if not follows(seg):
                        seg = [st] + Pp + [en]
                        print(f"    run past {lab_}: its own points kept (the marched edge strays)", flush=True)
                    seg = seg[:-1]
                lab_ = orig(r)                    # (the shape past the edge, as the mesher found it there)
                loop += [{"beta": np.asarray(p_).tolist(), "beyond": lab_} for p_ in seg]
            # (close to a corner two edges meeting at a sharp angle can cross before they meet: their points within
            # 0.015 of a corner go, so both run straight into it)
            cs = [c_ for c_ in corners if c_ is not None]
            CORNERS.extend(cs)
            CINFO.extend((corners[r], surf[r], surf[(r + 1) % R], orig(r), orig((r + 1) % R))
                         for r in range(R) if corners[r] is not None and surf[r] is not None and surf[(r + 1) % R] is not None)
            is_c = lambda p_: any(np.linalg.norm(p_ - c_) < 1e-12 for c_ in cs)
            loop = [q for q in loop if is_c(np.asarray(q["beta"])) or all(np.linalg.norm(np.asarray(q["beta"]) - c_) > 0.015 for c_ in cs)]
            loop.append(loop[0])
            new_outline.append(loop)
            continue
        # the mesh is kept as it is: each edge point of a run (they are the mesh's own boundary vertices) moves onto
        # the exact edge, and each solved corner joins the mesh by a fan to the edge points between the two runs'
        # good stretches (the scraps there are dropped from the outline: the fan covers them)
        pos = {k: n_ for n_, k in enumerate(range(n))}
        for r in range(R):
            s_ = surf[r]
            if s_ is None:
                continue
            for k in runs[r][2]:
                q = newton([W, s_], B[k])
                if np.linalg.norm(q - B[k]) < 1e-3:
                    MOVES[key(B[k])] = q
        skip, insert = set(), {}
        for r in range(R):
            c = corners[r]
            nxt = runs[(r + 1) % R]
            if c is None or not runs[r][2] or not nxt[2]:
                continue
            k0, k1 = runs[r][2][-1], nxt[2][0]
            chain = [k0]
            while chain[-1] != k1 and len(chain) <= n // 2:
                chain.append((chain[-1] + 1) % n)
            if chain[-1] != k1:
                continue
            # the two exact edges, from the mesh's last good edge points into the corner
            pa = MOVES.get(key(B[k0]), B[k0])
            pb = MOVES.get(key(B[k1]), B[k1])
            ea = march(W, surf[r], pa, c) if surf[r] is not None else None
            if ea is None and surf[r] is not None:
                ea = march(W, surf[r], c, pa)
                ea = ea[::-1] if ea is not None else None
            eb = march(W, surf[(r + 1) % R], c, pb) if surf[(r + 1) % R] is not None else None
            if eb is None and surf[(r + 1) % R] is not None:
                eb = march(W, surf[(r + 1) % R], pb, c)
                eb = eb[::-1] if eb is not None else None
            ea = ea if ea is not None else [pa, c]
            eb = eb if eb is not None else [c, pb]
            # a sharp corner (the two edges meet at under 30 degrees, a tip): rebuilt whole afterwards (tip())
            if len(ea) > 1 and len(eb) > 1 and surf[r] is not None and surf[(r + 1) % R] is not None:
                u_, v_ = np.asarray(ea[-2]) - c, np.asarray(eb[1]) - c
                if u_ @ v_ > np.cos(np.radians(30)) * np.linalg.norm(u_) * np.linalg.norm(v_):
                    # the cut goes where the wall is first 0.03 wide, walking back from the tip along edge A
                    PA = [MOVES.get(key(B[k]), B[k]) for k in runs[r][2]][::-1]
                    PB = np.array([MOVES.get(key(B[k]), B[k]) for k in nxt[2]])
                    A0 = next((p_ for p_ in PA if np.min(np.linalg.norm(PB - p_, axis=1)) >= 0.03), PA[-1])
                    ud = (c - A0) / np.linalg.norm(c - A0)       # (straight across: as far along as A0 toward c)
                    B0 = PB[np.argmin(np.abs((PB - A0) @ ud) + 0.1 * np.linalg.norm(PB - A0, axis=1))]
                    TIPS.append((c, surf[r], surf[(r + 1) % R], runs[r][0], runs[(r + 1) % R][0], A0, B0))
                    continue
            # the sliver between them and the mesh's old edge there: filled in the tangent plane, its triangles cut
            # small with their new points put on the wall's surface
            poly = [np.asarray(p_) for p_ in ea] + [np.asarray(p_) for p_ in eb[1:]] + [B[k] for k in chain[::-1][1:-1]]
            clean = []
            for p_ in poly:
                if not clean or np.linalg.norm(p_ - clean[-1]) > 1e-10:
                    clean.append(p_)
            while len(clean) > 1 and np.linalg.norm(clean[0] - clean[-1]) < 1e-10:
                clean.pop()
            if len(clean) >= 3:
                cut = [[clean[a_], clean[b_], clean[c_]] for a_, b_, c_ in ear_clip(clean, grad1(W, np.mean(clean, axis=0)))]
                for _ in range(2):
                    nxt_ = []
                    for t_ in cut:
                        if max(np.linalg.norm(t_[i_] - t_[(i_ + 1) % 3]) for i_ in range(3)) < 0.01:
                            nxt_.append(t_)
                            continue
                        m_ = [project_w(W, (t_[i_] + t_[(i_ + 1) % 3]) / 2) for i_ in range(3)]
                        nxt_ += [[t_[0], m_[0], m_[2]], [m_[0], t_[1], m_[1]], [m_[2], m_[1], t_[2]], [m_[0], m_[1], m_[2]]]
                    cut = nxt_
                FANS.extend(cut)
            skip.update(chain[1:-1])
            insert[k0] = ([np.asarray(p_) for p_ in ea[1:]] + [np.asarray(p_) for p_ in eb[1:-1]], runs[r][0], runs[(r + 1) % R][0])
        out_line = []
        for k in range(n):
            if k in skip:
                continue
            out_line.append({"beta": B[k].tolist(), "beyond": L[k]})
            if k in insert:
                pts_, la_, lb_ = insert[k]
                half = len(pts_) // 2
                out_line += [{"beta": q.tolist(), "beyond": la_ if i_ <= half else lb_} for i_, q in enumerate(pts_)]
        out_line.append(out_line[0])
        new_outline.append(out_line)

    def moved(p):
        return MOVES.get(key(p), p)
    tris = [np.array([moved(v) for v in t]) for t in T] + [np.array([moved(np.asarray(v)) for v in f]) for f in FANS]
    new_outline = [[{"beta": np.asarray(moved(np.asarray(q["beta"]))).tolist(), "beyond": q["beyond"]} for q in l]
                   for l in new_outline]
    if FILL:
        DROPPED_LOOPS.clear()
        tris = fill_mesh(W, new_outline, T)
        new_outline = [l for i, l in enumerate(new_outline) if i not in DROPPED_LOOPS]
        TIPS = []
        # (at a corner where the wall is singular, A1 say, points lifted along the chart's normal can land on another
        # sheet and fold the mesh over: the tip there is rebuilt in rows between its two exact edges, from where the
        # wall is 0.03 wide)
        U_ = np.linalg.svd(np.ones((1, 4)))[2][1:]

        def folded(tris_):
            T_ = np.asarray(tris_)
            P_ = T_ @ U_.T
            n_ = np.cross(P_[:, 1] - P_[:, 0], P_[:, 2] - P_[:, 0])
            sg = np.sign((n_ * (W.grad(T_.mean(1)) @ U_.T)).sum(1))
            return T_.mean(1)[sg != np.sign(np.sum(sg))]
        for c, SA, SB, la, lb in CINFO:
            bad = folded(tris)
            if not len(bad) or np.min(np.linalg.norm(bad - c, axis=1)) > 0.04:
                continue
            ol = [np.asarray(q["beta"]) for l in new_outline for q in l]
            labs = [q["beyond"] for l in new_outline for q in l]
            reach = max(0.03, float(np.max(np.linalg.norm(bad[np.linalg.norm(bad - c, axis=1) < 0.04] - c, axis=1))) + 0.01)
            pick = lambda lab_: min((p_ for p_, l_ in zip(ol, labs) if l_ == lab_), default=None,
                                    key=lambda p_: abs(np.linalg.norm(p_ - c) - reach))
            pa, pb = pick(la), pick(lb)
            if pa is None or pb is None:
                continue
            n0 = len(bad)
            t2, o2 = tip(W, c, SA, SB, la, lb, pa, pb, tris, new_outline)
            n1 = len(folded(t2))
            print(f"    tip at a singular corner: {n0} folded triangles -> {n1}{'' if n1 < n0 else ' (not kept)'}", flush=True)
            if n1 < n0:                           # (kept only if it helps)
                tris, new_outline = t2, o2
    for c, SA, SB, la, lb, pa, pb in TIPS:
        tris, new_outline = tip(W, c, SA, SB, la, lb, moved(pa), moved(pb), tris, new_outline)
    fill = FANS
    res = json.load(open(out)) if os.path.exists(out) else {}
    res[tid] = {**sf, "tris": [np.asarray(t).tolist() for t in tris], "outline": new_outline, "snapped": True,
                "corner_points": ALLC,
                "edge_surfaces": [[li, lab, n_, s] for li, lab, n_, s in info]}
    print(f"{tid}: {len(MOVES)} edge points moved onto exact edges, {len(FANS)} corner-fan triangles", flush=True)
    json.dump(res, open(out, "w"))
    print(f"{tid}: {len(T)} -> {len(tris)} triangles ({len(fill)} in the stitched strip)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
