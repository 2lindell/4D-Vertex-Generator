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

    def best_surface(P):
        """The known surface (or image) the run's points lie on, not the wall's own."""
        best = None
        for b in bases:
            D = dists_all(b, P)
            med = np.median(D, axis=1)
            for gi in np.argsort(med)[:6]:
                if med[gi] > 2e-3 or (best is not None and med[gi] >= best[0]):
                    break
                img = Image(b, Ginv[gi], b.name + (f" (image {gi})" if gi else ""))
                if np.median(dist(img, inner)) < 1e-6:          # (the wall's own surface)
                    continue
                best = (float(med[gi]), float(D[gi].max()), img)
                break
        return best

    new_outline, info = [], []
    MOVES, FANS = {}, []
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
            f = best_surface(Ps)
            if f is None or f[1] > 5e-4:
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
            for u, v in zip(chain, chain[1:]):
                FANS.append([c, B[u], B[v]])
            skip.update(chain[1:-1])
            insert[k0] = (c, runs[r][0])
        out_line = []
        for k in range(n):
            if k in skip:
                continue
            out_line.append({"beta": B[k].tolist(), "beyond": L[k]})
            if k in insert:
                out_line.append({"beta": insert[k][0].tolist(), "beyond": insert[k][1]})
        out_line.append(out_line[0])
        new_outline.append(out_line)

    def moved(p):
        return MOVES.get(key(p), p)
    tris = [np.array([moved(v) for v in t]) for t in T] + [np.array([f[0], moved(f[1]), moved(f[2])]) for f in FANS]
    new_outline = [[{"beta": np.asarray(moved(np.asarray(q["beta"]))).tolist(), "beyond": q["beyond"]} for q in l]
                   for l in new_outline]
    fill = FANS
    res = json.load(open(out)) if os.path.exists(out) else {}
    res[tid] = {**sf, "tris": [np.asarray(t).tolist() for t in tris], "outline": new_outline, "snapped": True,
                "edge_surfaces": [[li, lab, n_, s] for li, lab, n_, s in info]}
    print(f"{tid}: {len(MOVES)} edge points moved onto exact edges, {len(FANS)} corner-fan triangles", flush=True)
    json.dump(res, open(out, "w"))
    print(f"{tid}: {len(T)} -> {len(tris)} triangles ({len(fill)} in the stitched strip)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
