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
    zone = lambda p: (np.asarray(p) - m0) @ u > 1e-9
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

# ------------------------------------------------------------------------------------------------ main
def main(tid, out="xsurfaces.json"):
    X = json.load(open("xsurfaces.json"))
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
                cand = (float(med[gi]), float(mx[gi]), img)
                if mx[gi] <= 2.5e-4:
                    good.append((degree(b), cand))
                if loose is None or cand[0] < loose[0]:
                    loose = cand
                break
        if good:
            return min(good, key=lambda dc: (dc[0], dc[1][0]))[1]
        return loose

    new_outline, info = [], []
    MOVES, FANS, TIPS = {}, [], []
    key = lambda p: tuple(np.round(p, 9))
    for li, line in enumerate(sf["outline"]):
        B = np.array([q["beta"] for q in line])
        L = [q["beyond"] for q in line]
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
            if prev is not None and dist(prev[3], Ps).max() <= 5e-4 and not turns(prev[2] or prev[1], idx):
                prev[1] += idx
                prev[2] += idx
                continue
            # (a short run, under 12 points, is not an edge of its own: with so many candidate surfaces one would
            # always fit it by chance)
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
                c = newton([W, surf[r], surf[(r + 1) % R]], guess)
                # (a corner the mesh never reached, a sharp tip, can be well away from the runs' ends: accepted if
                # both edges lead to it, checked when they are marched)
                ok = np.linalg.norm(c - guess) < 0.15 and in_half(c)
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
                    ok = np.linalg.norm(p_ - q) < 1e-8 and np.linalg.norm(c - guess) < 0.15 and in_half(c)
            else:
                c = guess
            if ok:                                # (at a singular point, A1 say, Newton only creeps up on it: snap
                c = c.copy()                      # coordinates within 1e-4 of the cell's own values onto them)
                c[np.abs(c) < 1e-4] = 0.0
                if abs(c[2] - c[3]) < 1e-4:
                    c[2] = c[3] = (c[2] + c[3]) / 2
                c = c / c.sum()
            corners.append(c if ok else None)
            print(f"  corner {li}.{r} ({a[0][:12]} | {b[0][:12]}): "
                  f"{np.round(c / c.max(), 5).tolist() if ok else 'not solved'}", flush=True)
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
    for c, SA, SB, la, lb, pa, pb in TIPS:
        tris, new_outline = tip(W, c, SA, SB, la, lb, moved(pa), moved(pb), tris, new_outline)
    fill = FANS
    res = json.load(open(out)) if os.path.exists(out) else {}
    res[tid] = {**sf, "tris": [np.asarray(t).tolist() for t in tris], "outline": new_outline, "snapped": True,
                "edge_surfaces": [[li, lab, n_, s] for li, lab, n_, s in info]}
    print(f"{tid}: {len(MOVES)} edge points moved onto exact edges, {len(FANS)} corner-fan triangles", flush=True)
    json.dump(res, open(out, "w"))
    print(f"{tid}: {len(T)} -> {len(tris)} triangles ({len(fill)} in the stitched strip)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
