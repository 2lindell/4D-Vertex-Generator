"""Each wall's edges on other walls and its cuts, from the exact data on the whole 3-sphere (exact_curves.json).

For a wall w and another wall Y, every image of Y's surface under the group is laid against w's surface. Where an image
meets it along a curve through the exact meeting points (exact_points.json, 40 digits, carried by every symmetry over
the whole sphere), the curve is traced exactly (Newton onto both surfaces at every step), on the sphere itself, so a
stretch crossing a face of the cell is followed straight through, from one meeting point to the next. A stretch is
  * w's edge along Y when both its ends lie on w's other edges,
  * Y's cut on w (where Y ends on w, between w's pieces) when both its ends lie on Y's other edges:
each closed edge set runs from one of its meeting points to the next. No stored curve and no mesh is used.

Points are kept as unit vectors of betas (the surfaces are homogeneous, so each point stands for its ray).

    python exact_curves.py [walls]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import core as osd
from core import FOLD, extended

E_ = extended()
GG = np.concatenate([E_, E_ @ FOLD])
GINV = np.linalg.inv(GG)
planes, curved = osd.library()
LIB = {s.name: s for s in list(planes) + list(curved)}
H = 2e-3                                                   # (step along a curve, on the unit sphere)


def unit(X):
    X = np.atleast_2d(np.asarray(X, float))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    s = np.sign(X[np.arange(len(X)), np.argmax(np.abs(X) > 1e-12, axis=1)])   # (one of each antipodal pair)
    return X * s[:, None]


def fv(s, X):
    return np.ravel(s.f(np.atleast_2d(X)))


def gradient(s, x):
    """The exact gradient: a plane's normal; a curved surface's polynomial differentiated (through A, and g^-1 for an
    image)."""
    if isinstance(s, Img):
        return gradient(s.s, s.ginv @ x) @ s.ginv
    if isinstance(s, osd.Plane):
        return np.asarray(s.n, float)
    y = s.A @ x
    gy = np.zeros(4)
    for c, m in zip(s.c, s.mons):
        for k in range(len(m)):
            gy[m[k]] += c * np.prod([y[m[i]] for i in range(len(m)) if i != k])
    return gy @ s.A


def dist(s, X):
    """Distance from points (unit vectors) to the surface, to first order."""
    X = np.atleast_2d(X)
    f = fv(s, X)
    e = 1e-7
    G = np.stack([(fv(s, X + e * u) - fv(s, X - e * u)) / (2 * e) for u in np.eye(4)], 1)
    G = G - (G * X).sum(1, keepdims=True) * X              # (the part along the sphere)
    return np.abs(f) / (np.linalg.norm(G, axis=1) + 1e-300)


class Img:
    """The surface s carried by a symmetry: f(x) = s(g^-1 x) (homogeneous: no scale needed)."""

    def __init__(self, s, gi):
        self.s, self.ginv, self.gi = s, GINV[gi], gi

    def f(self, X):
        return self.s.f(np.atleast_2d(X) @ self.ginv.T)


def onto(surfs, x, steps=40):
    """x moved onto the surfaces and the unit sphere, by Newton steps (least change)."""
    x = np.asarray(x, float) / np.linalg.norm(x)
    for _ in range(steps):
        F = np.array([fv(s, x)[0] for s in surfs] + [0.5 * (x @ x - 1)])
        if np.abs(F).max() < 1e-15:
            break
        J = np.array([gradient(s, x) for s in surfs] + [x])
        x = x - J.T @ np.linalg.lstsq(J @ J.T, F, rcond=None)[0]
    return x / np.linalg.norm(x)


def loop(S, T, x0, max_steps=8000):
    """The whole closed curve of S and T on the sphere through x0 (a great circle exactly when both are planes)."""
    if isinstance(S, osd.Plane) and isinstance(getattr(T, "s", T), osd.Plane) and not isinstance(T, Img) or (
            isinstance(S, osd.Plane) and isinstance(T, Img) and isinstance(T.s, osd.Plane)):
        n1 = np.asarray(S.n, float)
        n2 = gradient(T, x0)
        B = np.linalg.svd(np.vstack([n1, n2]))[2][2:]          # (the plane of the circle)
        u = x0 / np.linalg.norm(x0)
        v = B[0] - (B[0] @ u) * u if abs(B[0] @ u) < 0.9 else B[1] - (B[1] @ u) * u
        v /= np.linalg.norm(v)
        th = np.linspace(0, 2 * np.pi, int(2 * np.pi / H) + 1)
        return np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
    x, pts, prev = np.asarray(x0, float), [np.asarray(x0, float)], None
    for _ in range(max_steps):
        M = np.vstack([gradient(S, x), gradient(T, x), x])
        t = np.linalg.svd(M)[2][-1]
        t = t if prev is None or t @ prev > 0 else -t
        prev = t
        x = onto([S, T], x + H * t / np.linalg.norm(t))
        pts.append(x)
        if len(pts) > 30 and np.linalg.norm(x - x0) < 1.5 * H:
            pts.append(np.asarray(x0, float))
            return np.array(pts)
    return None


def follow_to(S, T, x0, t0, targets, max_steps=4000):
    """The curve of S and T on the sphere from x0, setting out along t0, to the first of the targets it reaches (the run
    ends exactly on it): (run, index of the target), or (None, None) when it comes back round or runs too long."""
    x, pts, prev = np.asarray(x0, float), [np.asarray(x0, float)], np.asarray(t0, float)
    far = np.linalg.norm(targets - x0, axis=1) > 1e-9
    for _ in range(max_steps):
        M = np.vstack([gradient(S, x), gradient(T, x), x])
        t = np.linalg.svd(M)[2][-1]
        t = t if t @ prev > 0 else -t
        prev = t
        y = onto([S, T], x + H * t / np.linalg.norm(t))
        d = y - x
        s_ = np.clip(((targets - x) @ d) / max(d @ d, 1e-30), 0, 1)
        e = np.linalg.norm(x + s_[:, None] * d - targets, axis=1)
        e[~far] = 1
        k = int(np.argmin(e))
        if e[k] < H / 4:
            pts.append(targets[k])
            return np.array(pts), k
        x = y
        pts.append(x)
        if len(pts) > 30 and np.linalg.norm(x - x0) < H:
            return None, None
    return None, None


def follow_to_surface(S, T, x0, t0, Ws, max_steps=4000):
    """The curve of S and T on the sphere from x0, setting out along t0, to where it first crosses one of the surfaces
    Ws (away from x0): the run, its end solved exactly onto S, T and that surface; or None."""
    x, pts, prev = np.asarray(x0, float), [np.asarray(x0, float)], np.asarray(t0, float)
    for _ in range(max_steps):
        M = np.vstack([gradient(S, x), gradient(T, x), x])
        t = np.linalg.svd(M)[2][-1]
        t = t if t @ prev > 0 else -t
        prev = t
        y = onto([S, T], x + H * t / np.linalg.norm(t))
        if np.linalg.norm(x - x0) > 3 * H:
            for W in Ws:
                fa, fb = fv(W, x)[0], fv(W, y)[0]
                if fa != 0 and np.sign(fa) != np.sign(fb):
                    z = onto([S, T, W], x + fa / (fa - fb) * (y - x))
                    if max(dist(s_, z[None])[0] for s_ in (S, T, W)) < 1e-10:
                        pts.append(z)
                        return np.array(pts)
        x = y
        pts.append(x)
        if len(pts) > 30 and np.linalg.norm(x - x0) < H:
            return None
    return None


def cut_at(P, targets):
    """The closed curve P divided at the targets it passes (each put exactly into it): [(i, j, run)] between
    consecutive targets i and j."""
    marks = []
    for k, v in enumerate(targets):
        A, B = P[:-1], P[1:]
        d = B - A
        t = np.clip(((v - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
        e = np.linalg.norm(A + t[:, None] * d - v, axis=1)
        for s_ in np.flatnonzero(e < H / 4):
            marks.append((s_ + t[s_], k))
    marks = sorted(set((round(a, 6), k) for a, k in marks))
    if len(marks) < 2:
        return []
    out = []
    n = len(P) - 1
    for (a, i), (b, j) in zip(marks, marks[1:] + [(marks[0][0] + n, marks[0][1])]):
        idx = [int(np.floor(a)) + 1 + q for q in range(max(0, int(np.floor(b)) - int(np.floor(a))))]
        mid = [P[q % n] for q in idx]
        run = np.array([targets[i]] + mid + [targets[j]])
        if len(run) >= 2 and i != j or len(mid) > 3:
            out.append((i, j, run))
    return out


class Data:
    def __init__(self):
        self.ex = json.load(open("exact_defs.json"))
        # (the lines' points put exactly on their surfaces, exact_lines.py)
        for L, v in json.load(open("exact_lines.json"))["lines"].items():
            self.ex["lines"][L]["points"] = v["points"]
        self.work = json.load(open("working_data.json"))["walls"]
        self.points = [p for p in json.load(open("exact_points.json"))["points"] if p["exact"]]
        self.edge_corners = json.load(open("working_base.json"))["corrections"].get("edge_corners", {})
        P = np.array([p["float"] for p in self.points])
        V = unit(np.einsum("gij,nj->gni", GG, P).reshape(-1, 4))
        self.V = np.unique(V.round(12), axis=0)            # (every meeting point over the whole sphere)
        H_ = np.einsum("gij,nj->gni", GG, P).reshape(-1, 4)
        sm = H_.sum(1)
        H_ = H_[np.abs(sm) > 1e-9] / sm[np.abs(sm) > 1e-9, None]
        H_ = H_[(H_ > -1e-9).all(1) & (H_[:, 2] >= H_[:, 3] - 1e-9)]
        self.VH = np.unique(H_.round(12), axis=0)          # (the same, in the half-cell)
        self.segs, self.cache = {}, {}

    def line_segs(self, L):
        """The line's verified pieces (in the half-cell, betas summing to 1) as segments (A, B)."""
        if L not in self.segs:
            A, B = [], []
            for P in self.ex["lines"][L]["points"]:
                P = np.asarray(P, float)
                if len(P) >= 2:
                    A.append(P[:-1])
                    B.append(P[1:])
            self.segs[L] = (np.vstack(A), np.vstack(B)) if A else (np.zeros((0, 4)), np.zeros((0, 4)))
        return self.segs[L]

    def named_points(self, names):
        """The exact points of these names (the classifier's names in exact_points.json), every image in the half-cell,
        as unit vectors."""
        pick = [p["float"] for p in self.points if p["name"] in names]
        if not pick:
            return np.zeros((0, 4))
        Y = np.einsum("gij,nj->gni", GG, np.array(pick)).reshape(-1, 4)
        sm = Y.sum(1)
        Y = Y[np.abs(sm) > 1e-9] / sm[np.abs(sm) > 1e-9, None]
        Y = Y[(Y > -1e-9).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-9)]
        return unit(np.unique(Y.round(12), axis=0))

    def named_corners(self, owner, e):
        """The owner's corners by name (exact_points.json's sources: "end of L on M"), every image in the half-cell,
        as unit vectors: L one of the owner's lines and M e itself or one of e's edges or points, so the corner lies
        on e (for a wall with no lines for edges, the exact points named as its points)."""
        w, y = self.work[owner], self.work.get(e, {})
        lines_ = {x for x in w["edges"] if x in self.ex["lines"]}
        ok_on = set(y.get("edges", [])) | set(y.get("points", [])) | {e}     # (a corner on e: ending on e's own edges)
        pick = []
        for p in self.points:
            tags = [f[7:].split(" on ", 1) for f in p["from"] if f.startswith("end of ") and " on " in f]
            if any(L in lines_ and M in ok_on for L, M in tags) or (not lines_ and p["name"] in w.get("points", [])):
                pick.append(p["float"])
        if not pick:
            return np.zeros((0, 4))
        Y = np.einsum("gij,nj->gni", GG, np.array(pick)).reshape(-1, 4)
        sm = Y.sum(1)
        Y = Y[np.abs(sm) > 1e-9] / sm[np.abs(sm) > 1e-9, None]
        Y = Y[(Y > -1e-9).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-9)]
        return unit(np.unique(Y.round(12), axis=0))

    def line_ends(self, Ls):
        """The ends of the lines' verified pieces, every image in the half-cell (betas summing to 1)."""
        P = np.array([np.asarray(Q, float)[k] for L in Ls for Q in self.ex["lines"][L]["points"] if len(Q) >= 2
                      for k in (0, -1)])
        if not len(P):
            return np.full((1, 4), 9.0)
        Y = np.einsum("gij,nj->gni", GG, P).reshape(-1, 4)
        sm = Y.sum(1)
        Y = Y[np.abs(sm) > 1e-9] / sm[np.abs(sm) > 1e-9, None]
        return Y[(Y > -1e-7).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-7)]

    def on_edge(self, v, e, tol=5e-6):
        """The meeting point v lies on the edge e: an image of v in the half-cell lies on one of the line's pieces, or v
        on an image of the wall's surface."""
        key_ = (tuple(np.round(v, 9)), e)
        if key_ in self.cache:
            return self.cache[key_]
        if e in self.ex["lines"]:
            A, B = self.line_segs(e)
            Y = GG @ v
            sm = Y.sum(1)
            Y = Y[np.abs(sm) > 1e-9] / sm[np.abs(sm) > 1e-9, None]
            Y = Y[(Y > -1e-7).all(1) & (Y[:, 2] >= Y[:, 3] - 1e-7)]
            got = False
            d = B - A
            for y in Y:
                t = np.clip(((y - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
                if len(A) and np.linalg.norm(A + t[:, None] * d - y, axis=1).min() < tol:
                    got = True
                    break
        else:
            got = any(n in LIB and dist_images(LIB[n], v).min() < 1e-9
                      for n in self.ex["walls"].get(e, {}).get("surface", []))
        self.cache[key_] = got
        return got


def dist_images(T, v):
    """Distance from the point v to every image of T."""
    X = np.einsum("gij,j->gi", GINV, v)
    f = fv(T, X)
    e = 1e-7
    G = np.stack([(fv(T, np.einsum("gij,j->gi", GINV, v + e * u)) - fv(T, np.einsum("gij,j->gi", GINV, v - e * u)))
                  / (2 * e) for u in np.eye(4)], 1)
    G = G - (G @ v)[:, None] * v
    return np.abs(f) / (np.linalg.norm(G, axis=1) + 1e-300)


def edge_runs(D, owner, e):
    """The owner wall's edge along the wall e (the same curve is e's cut by the owner, where the owner ends on it).

    Its corners on e are named by the 3-sphere data: the ends of the owner's lines where they end on e or on one of
    e's edges or points (exact points). Between two of them, on one of the owner's surfaces, the curve where it meets
    e's surface, traced exactly on the sphere from one corner to the next meeting point it reaches. Which copy of e:
    first the copies placed in the half-cell (the library's surfaces of e, as the walls lie there); only if none of
    them joins two corners, every image of e's surfaces. Returns [runs (unit vectors)]."""
    edges = D.work[owner]["edges"]
    bases = {n.split(":")[0] for n in D.ex["walls"].get(e, {}).get("surface", [])}
    placed = [s_ for n, s_ in LIB.items() if n.split(":")[0] in bases]
    out = []
    for sn in D.ex["walls"][owner]["surface"]:
        S = LIB.get(sn)
        if S is None:
            continue
        spec = D.edge_corners.get(owner, {}).get(e)
        meets = []
        if spec:                                            # (corners given: named points, and "meets W")
            names = [n for n in spec if not n.startswith(("meets ", "ends at "))]
            meets = [n[6:] for n in spec if n.startswith("meets ")]
            ends_at = [n[8:] for n in spec if n.startswith("ends at ")]
            C = [v for v in D.named_points(names) if dist(S, v[None])[0] < 1e-9 and D.on_edge(v, e)]
            if meets and not C:
                # (the copy of the wall in the half-cell does not reach the corner, where the walls touch: the edge is
                # traced back to the corner from the point where the three walls meet)
                got = from_triple_point(D, owner, e, S, names, meets, out)
                if ends_at:                                 # (only the stretches ending at the given point)
                    E_ = unit(np.einsum("gij,nj->gni", GG, D.named_points(ends_at)).reshape(-1, 4))
                    E_ = np.concatenate([E_, -E_])
                    got = [r for r in got if np.linalg.norm(E_ - np.asarray(r)[-1], axis=1).min() < 1e-7]
                out += got
                continue
        else:
            C = [v for v in D.named_corners(owner, e) if dist(S, v[None])[0] < 1e-9 and D.on_edge(v, e)]
        if len(C) < (1 if meets else 2):
            continue
        C = np.array(C)
        if meets:
            Ws = [s_ for n, s_ in LIB.items()
                  if n.split(":")[0] in {m.split(":")[0] for W_ in meets for m in D.ex["walls"].get(W_, {}).get("surface", [])}]
            for T in [T for T in placed if (dist(T, C) < 1e-9).any()]:
                if same_surface(S, T):
                    continue
                for i in np.flatnonzero(dist(T, C) < 1e-9):
                    t0 = np.linalg.svd(np.vstack([gradient(S, C[i]), gradient(T, C[i]), C[i]]))[2][-1]
                    for sgn in (1, -1):
                        run = follow_to_surface(S, T, C[i], sgn * t0, Ws)
                        if run is None or any(near(run[len(run) // 2], q) for q in out):
                            continue
                        m = run[len(run) // 2]
                        if any(D.on_edge(m, L, 1e-7) for L in edges if L in D.ex["lines"]):
                            continue
                        out.append(run)
            continue
        for everywhere in (False, True):
            cands = []
            if not everywhere:
                cands = [T for T in placed if (dist(T, C) < 1e-9).sum() >= 2]
            else:
                for yn in D.ex["walls"].get(e, {}).get("surface", []):
                    if yn in LIB:
                        Dc = np.array([dist_images(LIB[yn], c) for c in C]) < 1e-9
                        cands += [Img(LIB[yn], gi) for gi in np.flatnonzero(Dc.sum(0) >= 2)]
            found = len(out)
            for T in cands:
                if same_surface(S, T):
                    continue
                Vt = D.V[(dist(S, D.V) < 1e-9) & (dist(T, D.V) < 1e-9)]      # (every meeting point on the curve)
                for i in np.flatnonzero(dist(T, C) < 1e-9):
                    t0 = np.linalg.svd(np.vstack([gradient(S, C[i]), gradient(T, C[i]), C[i]]))[2][-1]
                    for sgn in (1, -1):
                        run, k = follow_to(S, T, C[i], sgn * t0, Vt)
                        if run is None:
                            continue
                        end = Vt[k]
                        if np.linalg.norm(C - end, axis=1).min() > 1e-9:
                            continue                        # (not to another corner)
                        m = run[len(run) // 2]
                        if any(near(m, q) for q in out):
                            continue
                        if any(D.on_edge(m, L, 1e-7) for L in edges if L in D.ex["lines"]):
                            continue                        # (along one of its own lines: that line)
                        out.append(run)
            if len(out) > found:
                break
    return out


def surface_images_through(T0, pts, tol=1e-9):
    """The images of the surface T0 holding all the points."""
    ok = np.ones(len(GG), bool)
    for p in pts:
        ok &= dist_images(T0, p) < tol
    return np.flatnonzero(ok)


def branches(S, T, c, Wsets):
    """Where S and T touch at c: the curve's branches leaving c, each started from points a few steps off c on both
    surfaces and followed until it crosses an image of one of Wsets (end solved exactly there): runs from c."""
    n = gradient(S, c)
    B = np.linalg.svd(np.vstack([n, c]))[2][2:]           # (the tangent plane of S at c, on the sphere)
    starts = []
    for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p = onto([S, T], c + 3 * H * (np.cos(ang) * B[0] + np.sin(ang) * B[1]))
        r_ = np.linalg.norm(p - c)
        if H < r_ < 8 * H and max(dist(S, p[None])[0], dist(T, p[None])[0]) < 1e-10 \
                and not any(np.linalg.norm(p - q) < H for q in starts):
            starts.append(p)
    runs = []
    for p in starts:
        run = follow_to_sets(S, T, p, p - c, Wsets, avoid=c)
        if run is not None:
            runs.append(np.vstack([c[None], run]))
    return runs


def from_triple_point(D, owner, e, S, names, meets, out):
    """The owner's edge along e, from the named corner (where the owner touches e, so its branches are started just
    off it) to the point where the owner, e and the wall in meets come together: each branch of the owner's surface
    against an image of e's through the corner is followed until it crosses an image of that wall's surface, and the
    owner's edge along that wall likewise until it crosses an image of e's; the point is the end both share (exact)."""
    found = []
    W = meets[0]
    sets = {x: [ImgSet(LIB[m]) for m in D.ex["walls"].get(x, {}).get("surface", []) if m in LIB] for x in (e, W)}
    for c in unit(np.vstack([D.named_points(names)])):
        gs = np.flatnonzero(dist_images(S, c) < 1e-9)
        if not len(gs):
            continue
        So = Img(S, gs[0])
        fam = {}
        for x, other in ((e, W), (W, e)):
            fam[x] = []
            normals = []
            for yn in D.ex["walls"].get(x, {}).get("surface", []):
                if yn not in LIB:
                    continue
                for gi in np.flatnonzero(dist_images(LIB[yn], c) < 1e-9):
                    T = Img(LIB[yn], gi)
                    m_ = gradient(T, c)
                    m_ = m_ - (m_ @ c) * c
                    m_ /= np.linalg.norm(m_)
                    if any(abs(m_ @ q) > 1 - 1e-9 for q in normals):
                        continue
                    normals.append(m_)
                    fam[x] += branches(So, T, c, sets[other])
        ends_W = [r[-1] for r in fam[W]]
        for r in fam[e]:
            if any(np.linalg.norm(r[-1] - q) < 1e-7 for q in ends_W) and not any(near(r[len(r) // 2], q)
                                                                                 for q in out + found):
                found.append(r)
    return found


class ImgSet:
    """Every image of a surface at once: f(x) for all g (the sign of each)."""

    def __init__(self, s):
        self.s = s

    def values(self, x):
        return fv(self.s, np.einsum("gij,j->gi", GINV, x))


def given_from_image(D, owner, e, S, names, meets, edges, out):
    """The owner's edge along e from given corners (names) to where it meets the walls in meets, on the image of the
    owner's surface S through a corner, against each image of e's surfaces through it (one per direction there)."""
    found = []
    corners = unit(np.vstack([D.named_points(names)]))
    # (the point where the three walls meet lies on the edge where e and the other wall meet: that edge, exact, carried
    # over the whole sphere)
    target = []
    for W in meets:
        for a, b in ((W, e), (e, W)):
            if b in D.work.get(a, {}).get("edges", []):
                for r in edge_runs(D, a, b):
                    target.append((np.asarray(r, float), a, b))
    if not target:
        return []
    pts, src = [], []
    for k, (r, a, b) in enumerate(target):
        Y = np.einsum("gij,nj->gni", GG, r)
        Y = Y / np.linalg.norm(Y, axis=-1, keepdims=True)
        pts.append(np.concatenate([Y, -Y]).reshape(-1, 4))
    from scipy.spatial import cKDTree
    tree = cKDTree(np.vstack(pts))
    Wsurf = [LIB[m] for W in meets for m in D.ex["walls"].get(W, {}).get("surface", []) if m in LIB]
    for c in corners:
        for gs in np.flatnonzero(dist_images(S, c) < 1e-9)[:1]:
            S2 = Img(S, gs)
            for yn in D.ex["walls"].get(e, {}).get("surface", []):
                if yn not in LIB:
                    continue
                dirs = []
                for gi in np.flatnonzero(dist_images(LIB[yn], c) < 1e-9):
                    T = Img(LIB[yn], gi)
                    t = np.linalg.svd(np.vstack([gradient(S2, c), gradient(T, c), c]))[2][-1]
                    if np.linalg.norm(gradient(T, c) - (gradient(T, c) @ gradient(S2, c)) / (gradient(S2, c) @ gradient(S2, c))
                                      * gradient(S2, c)) < 1e-9 * np.linalg.norm(gradient(T, c)):
                        continue                            # (T is S itself there)
                    if any(abs(t @ u) > 1 - 1e-9 for u, _ in dirs):
                        continue
                    dirs.append((t, T))
                for t, T in dirs:
                    for sgn in (1, -1):
                        run = follow_to_curve(S2, T, c, sgn * t, tree, Wsurf)
                        if run is None or any(near(run[len(run) // 2], q) for q in out + found):
                            continue
                        found.append(run)
    return found


def follow_to_curve(S, T, x0, t0, tree, Wsurf, max_steps=4000):
    """The curve of S and T on the sphere from x0, setting out along t0, to where it reaches the curve held in tree
    (away from x0): the end solved exactly onto S, T and the image of one of Wsurf through it; or None."""
    x, pts, prev = np.asarray(x0, float), [np.asarray(x0, float)], np.asarray(t0, float)
    for _ in range(max_steps):
        M = np.vstack([gradient(S, x), gradient(T, x), x])
        t = np.linalg.svd(M)[2][-1]
        t = t if t @ prev > 0 else -t
        prev = t
        y = onto([S, T], x + H * t / np.linalg.norm(t))
        if np.linalg.norm(y - x0) > 3 * H and tree.query(y)[0] < H:
            for W0 in Wsurf:
                for gi in np.flatnonzero(dist_images(W0, y) < 5e-3):
                    W = Img(W0, gi)
                    z = onto([S, T, W], y)
                    if max(dist(s_, z[None])[0] for s_ in (S, T, W)) < 1e-10 and np.linalg.norm(z - y) < 2 * H \
                            and tree.query(z)[0] < H / 2:
                        pts.append(z)
                        return np.array(pts)
        x = y
        pts.append(x)
        if len(pts) > 30 and np.linalg.norm(x - x0) < H:
            return None
    return None


def follow_to_sets(S, T, x0, t0, Wsets, max_steps=4000, avoid=None):
    """As follow_to_surface, the surfaces being every image of each of Wsets (the first one crossed, away from x0; the
    images through the point avoid, the corner the curve leaves, left out)."""
    x, pts, prev = np.asarray(x0, float), [np.asarray(x0, float)], np.asarray(t0, float)
    skip = [np.abs(Ws.values(avoid)) < 1e-9 if avoid is not None else np.zeros(len(GG), bool) for Ws in Wsets]
    for _ in range(max_steps):
        M = np.vstack([gradient(S, x), gradient(T, x), x])
        t = np.linalg.svd(M)[2][-1]
        t = t if t @ prev > 0 else -t
        prev = t
        y = onto([S, T], x + H * t / np.linalg.norm(t))
        if np.linalg.norm(x - x0) > 3 * H:
            for Ws, sk in zip(Wsets, skip):
                fa, fb = Ws.values(x), Ws.values(y)
                hit = np.flatnonzero((fa != 0) & (np.sign(fa) != np.sign(fb)) & ~sk)
                for gi in hit:
                    W = Img(Ws.s, gi)
                    z = onto([S, T, W], x + fa[gi] / (fa[gi] - fb[gi]) * (y - x))
                    if max(dist(s_, z[None])[0] for s_ in (S, T, W)) < 1e-10:
                        pts.append(z)
                        return np.array(pts)
        x = y
        pts.append(x)
        if len(pts) > 30 and np.linalg.norm(x - x0) < H:
            return None
    return None


def same_surface(S, T):
    """T is S itself (the same zeros along a few points of S)."""
    pts = [onto([S], x) for x in (np.array([.4, .3, .2, .1]), np.array([.1, .2, .3, .4]), np.array([.25, .3, .25, .2]))]
    return all(dist(T, p[None])[0] < 1e-9 for p in pts)


def near(p, Q, tol=1e-6):
    A, B = Q[:-1], Q[1:]
    d = B - A
    t = np.clip(((p - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
    return float(np.linalg.norm(A + t[:, None] * d - p, axis=1).min()) < tol


def one(w):
    """The wall's edges along other walls: {Y: [runs]}."""
    D = Data()
    r = {}
    for e in D.work[w]["edges"]:
        if e not in D.ex["lines"] and e in D.ex["walls"]:
            r[e] = [q.round(12).tolist() for q in edge_runs(D, w, e)]
    return w, r


def main(only=None):
    walls = [w for w in json.load(open("working_data.json"))["walls"] if not only or w in only]
    out = json.load(open("exact_curves.json")) if only else {}
    with Pool() as pool:
        for w, r in pool.imap_unordered(one, walls):
            out[w] = r
            print(w, {y: len(v) for y, v in r.items()}, flush=True)
            json.dump(out, open("exact_curves.json", "w"))           # (kept as each wall is done)


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
