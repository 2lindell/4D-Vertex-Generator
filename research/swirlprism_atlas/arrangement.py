"""Every wall's extent from where the walls lie on the 3-sphere (arrangement.json).

No corner is named by hand and no point is classified. For each wall W and each of its surfaces as placed in the
half-cell (exact_defs.json):

  1. The surface is laid over the half-cell (and a little past its faces) as a scaffold of points, each solved exactly
     onto it (marching tetrahedra; the scaffold only tells which points are next to which).
  2. Every copy (image under the 4800 symmetries) of every surface of the walls W meets (its edges and the walls that
     cut it, working_data.json) that crosses it is found, and so is every copy of each of W's lines (the copy of the
     line's second surface along it). W's other planes cross it too (seams). At each scaffold point, the side of each
     copy is exact (the sign of its polynomial).
  3. W is grown from its seed (working_base.json) between neighbouring points, stopped only where a real edge lies
     between them: a copy of an edge wall Y is real where Y itself is there (the crossing point, carried back into the
     half-cell by the group onto Y's own surface, lies on Y as grown); a copy of a line is real where the line's
     verified pieces run; a seam always stops it; a cut never does. Where W reaches a face of the half-cell, the point
     just past it is carried back into the half-cell by the group, onto one of W's surfaces, and W goes on from there.
  4. All walls are grown together, each from the others' extents, until nothing changes.

    python arrangement.py [walls]        (meshes cached in $ARR_CACHE)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import core as osd
from core import FOLD, TETS, extended
from domain_page import carry, images

E_ = extended()
GG = np.concatenate([E_, E_ @ FOLD])
GINV = np.linalg.inv(GG)
planes_, curved_ = osd.library()
LIB = {s.name: s for s in list(planes_) + list(curved_)}
SURF = dict(LIB)                                          # (the library's surfaces, and copies found as walls grow)
N = int(os.environ.get("ARR_N", 128))
H = 1.0 / N
CACHE = os.environ.get("ARR_CACHE", "/tmp/arr_cache")
STRICT = os.environ.get("ARR_STRICT", "1") == "1"         # (every copy of a wall it ends on stops a wall: see Wall.present)
CELL = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])   # (the half-cell: CELL @ b >= 0)
CUBE8 = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])

# (marching tetrahedra: for each pattern of signs at a tetrahedron's four corners, its triangles as pairs of corners)
TABLE = {}
for bits in range(16):
    pos = [i for i in range(4) if bits >> i & 1]
    neg = [i for i in range(4) if not bits >> i & 1]
    if len(pos) in (0, 4):
        TABLE[bits] = []
    elif len(pos) == 1 or len(neg) == 1:
        a, rest = (pos[0], neg) if len(pos) == 1 else (neg[0], pos)
        TABLE[bits] = [[(a, rest[0]), (a, rest[1]), (a, rest[2])]]
    else:
        (a, b), (c, d) = pos, neg
        TABLE[bits] = [[(a, c), (a, d), (b, d)], [(a, c), (b, d), (b, c)]]


def fvals(s, X):
    return np.ravel(s.f(np.atleast_2d(X)))


class Img:
    """The surface s carried by the symmetry gi: zero where s(g^-1 x) is."""

    def __init__(self, s, gi):
        self.s, self.gi, self.ginv = s, gi, GINV[gi]

    def f(self, X):
        return self.s.f(np.atleast_2d(X) @ self.ginv.T)


def surface(sn):
    """A surface by name: one of the library's, or a copy of one, named "<library name>@<symmetry>"."""
    if sn not in SURF:
        base, gi = sn.rsplit("@", 1)
        SURF[sn] = Img(LIB[base], int(gi))
    return SURF[sn]


def grads(s, X, e=1e-7):
    """Gradients along the betas' sum-zero directions, at each point."""
    G = np.stack([(fvals(s, X + e * u) - fvals(s, X - e * u)) / (2 * e) for u in np.eye(4)], 1)
    return G - G.mean(1, keepdims=True)


def onto(surfs, X, steps=8):
    """Points moved onto all the surfaces (betas summing to 1 kept), by Newton steps (least change)."""
    X = np.array(X, float)
    for _ in range(steps):
        F = np.stack([fvals(s, X) for s in surfs], 1)
        if np.abs(F).max() < 1e-15:
            break
        J = np.stack([grads(s, X) for s in surfs], 1)                       # (n, k, 4)
        JJ = J @ J.transpose(0, 2, 1)
        X = X - (J.transpose(0, 2, 1) @ np.linalg.pinv(JJ, rcond=1e-12) @ F[..., None])[..., 0]   # (tangent: least squares)
    return X


def off(s, X):
    X = np.atleast_2d(X)
    return np.abs(fvals(s, X)) / (np.linalg.norm(grads(s, X), axis=1) + 1e-300)


def mesh(sn):
    """The scaffold of the surface sn over the half-cell and a little past it: points (betas, on the surface), the pairs
    of neighbouring points, and which points are in the half-cell."""
    path = os.path.join(CACHE, hashlib.md5(sn.encode()).hexdigest()[:12] + f"_{N}s.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["V"], z["E"], z["T"]
    S = surface(sn)
    pad = 3
    # (the grid shifted by odd amounts on each axis, so no grid point lies on any of the exact planes, the half-cell's
    # faces among them)
    r = np.arange(-pad, N + pad + 1) * H
    I, J, K = np.meshgrid(r + 0.3127 * H, r + 0.4419 * H, r + 0.2733 * H, indexing="ij")
    Q = np.stack([I, J, K], -1).reshape(-1, 3)
    Bt = np.column_stack([Q, 1 - Q.sum(1)])
    near = ((Bt @ CELL.T) / np.linalg.norm(CELL, axis=1) > -2.5 * H).all(1)
    F = np.full(len(Bt), np.nan)
    F[near] = fvals(S, Bt[near])
    n = len(r)
    F = F.reshape(n, n, n)
    tris_keys = []
    c0 = np.stack(np.meshgrid(np.arange(n - 1), np.arange(n - 1), np.arange(n - 1), indexing="ij"), -1).reshape(-1, 3)
    corners = c0[:, None, :] + CUBE8[None]                                  # (cubes, 8, 3)
    Fc = F[corners[..., 0], corners[..., 1], corners[..., 2]]
    ok = np.isfinite(Fc).all(1) & ((Fc > 0).any(1) & (Fc <= 0).any(1))
    corners, Fc = corners[ok], Fc[ok]
    gid = (corners[..., 0] * n + corners[..., 1]) * n + corners[..., 2]     # (global point index)
    for t in TETS:
        g4, f4 = gid[:, t], Fc[:, t]
        bits = ((f4 > 0) * (1 << np.arange(4))).sum(1)
        for b, tl in TABLE.items():
            if not tl:
                continue
            sel = bits == b
            if not sel.any():
                continue
            for tri in tl:
                ks = [np.sort(np.stack([g4[sel, a], g4[sel, c]], 1), 1) for a, c in tri]
                tris_keys.append(np.stack(ks, 1))                           # (m, 3, 2)
    TK = np.concatenate(tris_keys)
    keys, inv = np.unique(TK.reshape(-1, 2), axis=0, return_inverse=True)
    T = inv.reshape(-1, 3)
    Ff = F.reshape(-1)
    fa, fb = Ff[keys[:, 0]], Ff[keys[:, 1]]
    t_ = fa / (fa - fb)
    V = Bt[keys[:, 0]] + t_[:, None] * (Bt[keys[:, 1]] - Bt[keys[:, 0]])
    V = onto([S], V)
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [0, 2]]]), 1), axis=0)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, V=V, E=E, T=T)
    return V, E, T


def inside(V, tol=1e-12):
    return (V @ CELL.T >= -tol).all(1)


def copies_crossing(base, V, ins, skip=None):
    """The copies of the surface base (one per distinct surface) changing sign over the points V[ins]."""
    X = V[ins]
    if len(X) > 6000:
        X = X[np.random.default_rng(0).choice(len(X), 6000, replace=False)]
    out, seen = [], set()
    for c in range(0, len(GG), 400):
        Xg = np.einsum("gij,nj->gni", GINV[c:c + 400], X)
        vals = fvals(base, Xg.reshape(-1, 4)).reshape(len(Xg), -1)
        for k in range(len(vals)):
            v = vals[k]
            nv = np.linalg.norm(v)
            if nv == 0 or not (v.min() < -1e-9 * nv and v.max() > 1e-9 * nv):
                continue
            u = v / nv
            u = u * np.sign(u[np.argmax(np.abs(u))])
            key = tuple(np.round(u[:: max(1, len(u) // 64)], 6))
            if key in seen:
                continue
            if skip is not None and np.abs(u - skip).max() < 1e-6:
                continue
            seen.add(key)
            out.append(c + k)
    return out


def line_copies(L, lines, S, V, ins):
    """Each copy of the line L lying on S in the half-cell: (its second surface there, its points)."""
    out = []
    for P, pair in zip(lines[L]["points"], lines[L]["surfaces"]):
        P = np.asarray(P, float)
        if len(P) < 2 or len(pair) < 2 or not all(n in LIB for n in pair):
            continue
        for gi in images(P):
            Y = carry(P.copy(), GG[gi])
            Y = Y[np.isfinite(Y).all(1)]
            if len(Y) < 2:
                continue
            k = [0, len(Y) // 2, len(Y) - 1]
            if off(S, Y[k]).max() > 1e-7:
                continue
            A, Bs = Img(LIB[pair[0]], gi), Img(LIB[pair[1]], gi)
            m = Y[len(Y) // 2]
            # (the second surface: the one of the pair crossing S, the more steeply)
            gs = grads(S, m[None])[0]
            best = None
            for T in (A, Bs):
                gt = grads(T, m[None])[0]
                sn_ = np.sqrt(max(0, 1 - (gs @ gt) ** 2 / ((gs @ gs) * (gt @ gt) + 1e-300)))
                if best is None or sn_ > best[0]:
                    best = (sn_, T)
            if best[0] > 1e-3:
                out.append((best[1], Y))
    return out


def setup(args):
    """For one surface of one wall: the scaffold and every copy crossing it, with its side at each point."""
    w, sn, D = args
    work, ex, lines = D
    S = surface(sn)
    V, E, T = mesh(sn)
    ins = inside(V)
    funcs = []                                             # (kind, source wall or line, surface, extra)
    meet = [e for e in work[w]["edges"] if e not in lines] + [c["by"] for c in work[w].get("cuts", [])]
    for Y in dict.fromkeys(meet):
        kind = "edge" if Y in work[w]["edges"] else "cut"
        for yn in ex["walls"].get(Y, {}).get("surface", []):
            if yn not in LIB:
                continue
            for gi in copies_crossing(LIB[yn], V, ins):
                funcs.append((kind, Y, Img(LIB[yn], gi), None))
    for L in work[w]["edges"]:
        if L in lines:
            for T_, P in line_copies(L, lines, S, V, ins):
                funcs.append(("line", L, T_, P))
    for sn2 in ex["walls"][w]["surface"]:
        if sn2 != sn and sn2 in LIB:
            funcs.append(("seam", sn2, LIB[sn2], None))
    sg = np.stack([fvals(f[2], V) > 0 for f in funcs], 1) if funcs else np.zeros((len(V), 0), bool)
    return w, sn, V, E, T, ins, funcs, sg


_EP = []


def exact_points():
    """Every image of every exact point in the half-cell, with its name."""
    if not _EP:
        X, names = [], []
        for p in json.load(open("exact_points.json"))["points"]:
            if not p["exact"]:
                continue
            Y = GG @ np.array(p["float"])
            s_ = Y.sum(1)
            ok = np.abs(s_) > 1e-12
            Y = Y[ok] / s_[ok, None]
            Y = np.unique(Y[(Y @ CELL.T >= -1e-9).all(1)].round(12), axis=0)
            X.append(Y)
            names += [p["name"]] * len(Y)
        _EP.extend([np.vstack(X), names])
    return _EP[0], _EP[1]


def sheet_key(S):
    """The same surface whatever its name: its values at fixed points, up to scale and sign."""
    v = fvals(S, KEY_PTS)
    v = v / (np.abs(v).max() + 1e-300)
    return tuple(np.round(v * np.sign(v[np.argmax(np.abs(v) > 1e-6)]), 7))


KEY_PTS = np.random.default_rng(1).dirichlet(np.ones(4), 24)


def crossing(V, E, f, S):
    """For the neighbour pairs E (rows) changing side of f: the exact point on S and f between them."""
    a, b = V[E[:, 0]], V[E[:, 1]]
    fa, fb = fvals(f, a), fvals(f, b)
    t = np.nan_to_num(fa / np.where(fa == fb, 1, fa - fb), nan=0.5)
    return onto([S, f], a + np.clip(t, 0, 1)[:, None] * (b - a))


class Wall:
    def __init__(self, w, sheets, bases=()):
        self.w, self.sheets, self.bases = w, sheets, list(bases)   # sn -> dict; the library surfaces it is copies of
        self.pending = Counter()
        self.inwall = {sn: sh["ins"].copy() for sn, sh in sheets.items()}   # (to start: everywhere)
        self.tree = {}

    def present(self, X):
        """For points X (betas): is this wall there? Strictly (STRICT, the default): wherever any copy of its surfaces
        is, so every copy of a wall it ends on stops it (a looser test, where the wall as grown so far is, let walls
        leak). Else carried by the group into the half-cell, onto its surfaces as grown."""
        X = np.atleast_2d(X)
        if STRICT:
            got = np.zeros(len(X), bool)
            for b in self.bases:
                for i, x in enumerate(X):
                    if not got[i]:
                        v = fvals(LIB[b], np.einsum("gij,j->gi", GINV, x))
                        got[i] = (np.abs(v) < 1e-7 * (np.abs(v).max() + 1e-300)).any()
            return got
        got = np.zeros(len(X), bool)
        for sn, sh in self.sheets.items():
            if sn not in self.tree:
                pts = sh["V"][self.inwall[sn]]
                self.tree[sn] = cKDTree(pts) if len(pts) else None
        for ni, P in carried(X):
            for sn in self.sheets:
                if self.tree[sn] is None:
                    continue
                on_ = off(surface(sn), P) < 1e-7
                if not on_.any():
                    continue
                d, _ = self.tree[sn].query(P[on_])
                got[ni[on_][d < 1.5 * H]] = True
        return got


def carried(X, strict=False, chunk=200):
    """Every image of the points X in the half-cell: yields (index of the point, image), a chunk at a time."""
    for c in range(0, len(X), chunk):
        Y = np.einsum("gij,nj->gni", GG, X[c:c + chunk])
        s_ = Y.sum(-1)
        ok = np.abs(s_) > 1e-12
        Y = Y / np.where(ok, s_, 1)[..., None]
        cin = ok & ((Y @ CELL.T > 1e-12) if strict else (Y @ CELL.T >= -1e-9)).all(-1)
        gi, ni = np.nonzero(cin)
        yield ni + c, Y[gi, ni]


def line_steps(V, Ei, P, S, per=False):
    """Which steps Ei (pairs of neighbouring scaffold points) the line's points P (lying on the surface S) pass between:
    for each stretch of the line, the steps near it whose ends lie on its two sides, in the surface's own tangent plane,
    and whose crossing falls within the stretch (no second surface is needed: the line's own exact points decide)."""
    out = np.zeros(len(Ei), bool)
    segs = [np.zeros(0, int) for _ in range(max(0, len(P) - 1))]
    if len(Ei) == 0 or len(P) < 2:
        return segs if per else out
    a_, b_ = V[Ei[:, 0]], V[Ei[:, 1]]
    tree = cKDTree((a_ + b_) / 2)
    G = grads(S, P)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        L = np.linalg.norm(b - a)
        if L < 1e-15:
            continue
        cand = tree.query_ball_point((a + b) / 2, L / 2 + 1.5 * H)
        if not cand:
            continue
        cand = np.array(cand)
        e1 = (b - a) / L
        e2 = np.linalg.svd(np.vstack([np.ones(4), G[i], e1]))[2][-1]
        yu, yv = (a_[cand] - a) @ e2, (b_[cand] - a) @ e2
        xu, xv = (a_[cand] - a) @ e1, (b_[cand] - a) @ e1
        cross = (yu > 0) != (yv > 0)
        t = yu / np.where(cross, yu - yv, 1)
        xc = xu + t * (xv - xu)
        hit = cross & (xc >= -1e-12) & (xc <= L + 1e-12)
        out[cand[hit]] = True
        segs[i] = cand[hit]
    return segs if per else out


def real_at(kind, src, P, X, walls):
    """Which crossings X of a copy are real: a line where its verified pieces run, a wall where it is there; and a
    crossing within two scaffold steps of a real one on the same copy (at a corner the scaffold's steps can leave a gap
    of a step or two between where one curve stops counting and the next begins; an opening in the data is far wider)."""
    if kind == "line":
        real = cKDTree(P).query(X)[0] < 1.5 * H
    elif kind in ("edge", "cut"):
        real = walls[src].present(X) if src in walls else np.ones(len(X), bool)
    else:
        return np.ones(len(X), bool)
    if real.any() and (~real).any():
        real = real.copy()
        real[~real] = cKDTree(X[real]).query(X[~real])[0] < 2 * H
    return real


def grow(W, walls, D, seeds, dbg=None):
    """Grow wall W from its seed, stopped by its real edges, given the others as they now are."""
    lines = D[2]
    comp_of, n_comp, links, seed_comps = {}, 0, [], []
    for sn, sh in W.sheets.items():
        V, E, ins, funcs, sg = sh["V"], sh["E"], sh["ins"], sh["funcs"], sh["sg"]
        S = surface(sn)
        Ei = E[ins[E[:, 0]] & ins[E[:, 1]]]
        blocked = np.zeros(len(Ei), bool)
        rim_src = []                                       # (the steps each edge stops, by its name)
        for k, (kind, src, f, P) in enumerate(funcs):
            if kind == "line":
                b = line_steps(V, Ei, P, S)
                blocked |= b
                rim_src.append((src, np.unique(Ei[b])))
                continue
            ch = sg[Ei[:, 0], k] != sg[Ei[:, 1], k]
            # (a cut divides nothing; nor does a seam, where two copies of the wall in the half-cell meet: each goes on
            # through it, and where one ends there, a listed line is its edge)
            if not ch.any() or kind in ("cut", "seam"):
                continue
            idx = np.flatnonzero(ch)
            X = crossing(V, Ei[idx], f, S)
            r_ = idx[real_at(kind, src, P, X, walls)]
            blocked[r_] = True
            rim_src.append((src, np.unique(Ei[r_])))
        sh["rim_src"] = rim_src
        A = Ei[~blocked]
        bm = (V[Ei[blocked, 0]] + V[Ei[blocked, 1]]) / 2
        sh["edge_tree"] = cKDTree(bm) if len(bm) else None
        sh["rimv"] = np.unique(Ei[blocked])
        g = coo_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(len(V), len(V)))
        nc, lab = connected_components(g, directed=False)
        comp_of[sn] = lab + n_comp
        n_comp += nc
        sh["lab"] = lab
    root = list(range(n_comp))
    votes = Counter()
    cant, rim = Counter(), Counter()
    inwall_now = W.inwall
    W.pending = Counter()

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    trees = {sn: cKDTree(sh["V"][sh["ins"]]) for sn, sh in W.sheets.items()}
    idxs = {sn: np.flatnonzero(sh["ins"]) for sn, sh in W.sheets.items()}
    for sn, sh in W.sheets.items():
        V, E, ins, funcs, sg = sh["V"], sh["E"], sh["ins"], sh["funcs"], sh["sg"]
        # (into the next cell: a point just past a face, not across a real edge, carried back)
        Eo = E[ins[E[:, 0]] != ins[E[:, 1]]]
        Eo = np.where(ins[Eo[:, 0]][:, None], Eo, Eo[:, ::-1])           # (inside, outside)
        if not len(Eo):
            continue
        # (not across a real edge: a line, many of which lie in the half-cell's faces, or a copy of a wall that is
        # there, or a seam)
        stop = np.zeros(len(Eo), bool)
        for k, (kind, src, f, P) in enumerate(funcs):
            if kind == "line":
                b = line_steps(V, Eo, P, surface(sn))
                stop |= b
                sh["rim_src"].append((src, np.unique(Eo[b, 0])))      # (a line in a face of the half-cell)
                continue
            if kind in ("cut", "seam"):
                continue
            ch = np.flatnonzero(sg[Eo[:, 0], k] != sg[Eo[:, 1], k])
            if not len(ch):
                continue
            X = crossing(V, Eo[ch], f, surface(sn))
            stop[ch[real_at(kind, src, P, X, walls)]] = True
        # (a wall it ends on lying in the face itself, as X23 does in b2 = 0: no copy of it changes sign between
        # points of the half-cell, so it is tested where the step crosses the face)
        if len(Eo):
            # (the exact point where the step crosses the face it leaves by)
            fi = np.argmin(V[Eo[:, 1]] @ CELL.T, axis=1)
            mid = (V[Eo[:, 0]] + V[Eo[:, 1]]) / 2
            for f_ in np.unique(fi):
                m_ = fi == f_
                mid[m_] = onto([surface(sn), osd.Plane(CELL[f_], "face")], mid[m_])
            for Y in {f[1] for f in funcs if f[0] == "edge"}:
                if Y in walls:
                    live = np.flatnonzero(~stop)
                    if len(live):
                        stop[live[walls[Y].present(mid[live])]] = True
        Eo = Eo[~stop]
        for ni, Q in carried(V[Eo[:, 1]], strict=True):
            # (the point past the face is the wall only if every copy of it in the half-cell lies on one of the wall's
            # surfaces as placed there: one copy off them (the fold's, say, on a surface the wall does not use) means
            # the wall stops at the face)
            onW = {sn2: off(surface(sn2), Q) < 1e-7 for sn2 in W.sheets}
            anyW = np.any(list(onW.values()), axis=0)
            bad = np.isin(ni, ni[~anyW])
            # (a part whose rim on a face cannot go on: steps past the face, clear of the wall's edges, with a copy off
            # its surfaces)
            et0 = sh.get("edge_tree")
            for a in np.unique(ni[~anyW]):
                if et0 is None or et0.query(V[Eo[a, 0]])[0] > 2 * H:
                    cant[int(comp_of[sn][Eo[a, 0]])] += 1
                    # (the copy of one of the wall's own surfaces it lands on: a sheet of the wall the list lacked)
                    if not inwall_now[sn][Eo[a, 0]]:
                        continue
                    q = Q[(ni == a) & ~anyW][0]
                    for bn in W.bases:
                        v = fvals(LIB[bn], np.einsum("gij,j->gi", GINV, q))
                        gs = np.flatnonzero(np.abs(v) < 1e-9 * (np.abs(v).max() + 1e-300))
                        gs = [g for g in gs if off(Img(LIB[bn], g), q[None])[0] < 1e-7]
                        if gs:
                            W.pending[f"{bn}@{min(gs)}"] += 1
                            break
            for a in np.unique(ni):
                if et0 is None or et0.query(V[Eo[a, 0]])[0] > 2 * H:
                    rim[int(comp_of[sn][Eo[a, 0]])] += 1
            for sn2 in W.sheets:
                on_ = onW[sn2] & ~bad
                if not on_.any():
                    continue
                d, j = trees[sn2].query(Q[on_])
                # (both ends clear of the wall's real edges: at a corner on a face, where the wall pinches to a point,
                # a step out lands on the far side of an edge)
                et1, et2 = sh.get("edge_tree"), W.sheets[sn2].get("edge_tree")
                far1 = et1.query(V[Eo[ni[on_], 0]])[0] > 2 * H if et1 is not None else np.ones(on_.sum(), bool)
                far2 = et2.query(Q[on_])[0] > 2 * H if et2 is not None else np.ones(on_.sum(), bool)
                for a, dd, jj, ok_ in zip(ni[on_], d, j, far1 & far2):
                    if dd < 1.5 * H and ok_:
                        votes[(int(comp_of[sn][Eo[a, 0]]), int(comp_of[sn2][idxs[sn2][jj]]))] += 1
    # (two parts joined across a face only where three steps or more agree: a real continuation has many; a step
    # landing next to where a curve meets the face can fall on the wrong side of it)
    for (i1, i2), n_ in votes.items():
        if n_ >= 3:
            root[find(i1)] = find(i2)
    # (the seed)
    sd = np.asarray(seeds[W.w], float)
    Y = np.einsum("gij,j->gi", GG, sd)
    s_ = Y.sum(1)
    Y = Y[np.abs(s_) > 1e-12] / s_[np.abs(s_) > 1e-12, None]
    Y = Y[(Y @ CELL.T >= -1e-12).all(1)]
    # (each seed image starts the wall at the nearest scaffold point it reaches without crossing a real edge: the step
    # from the seed to that point tested exactly as every other step is; the seed is often a symmetric point lying a
    # small fraction of a step from one of the wall's edges)
    seed_comps = set()
    for sn, sh in W.sheets.items():
        S = surface(sn)
        Vs = sh["V"]
        for y in Y[off(S, Y) < 1e-9]:
            cand = [i for i in trees[sn].query_ball_point(y, 3 * H)]
            cand = sorted((float(np.linalg.norm(Vs[idxs[sn][i]] - y)), int(idxs[sn][i])) for i in cand)
            # (a copy passing through the seed itself does not divide the wall there: the seed is a point of the wall's
            # own shape; the wall starts on every side of such copies, at the nearest point on each)
            through = {k for k, (kind, src, f, P) in enumerate(sh["funcs"])
                       if kind in ("edge", "line") and (cKDTree(P).query(y)[0] < 1e-9 if kind == "line"
                                                        else off(f, y[None])[0] < 1e-9)}
            sides_seen = set()
            for _, v in cand:
                side = tuple(sh["sg"][v, sorted(through)]) if through else ()
                if side in sides_seen:
                    continue
                pair = np.vstack([y, Vs[v]])
                step = np.array([[0, 1]])
                ok = True
                for k, (kind, src, f, P) in enumerate(sh["funcs"]):
                    if kind in ("cut", "seam") or k in through:
                        continue
                    if kind == "line":
                        if line_steps(pair, step, P, S).any():
                            ok = False
                            break
                        continue
                    fy, fv = fvals(f, pair)
                    if (fy > 0) == (fv > 0):
                        continue
                    if real_at(kind, src, P, crossing(pair, step, f, S), walls).any():
                        ok = False
                        break
                if ok:
                    seed_comps.add(find(comp_of[sn][v]))
                    sides_seen.add(side)
                    if not through:
                        break
    if dbg is not None:
        dbg.update(seed_comps=seed_comps, cant=cant, rim=rim, votes={k: v for k, v in votes.items()},
                   sizes={sn: np.bincount(comp_of[sn][sh["ins"]] - comp_of[sn].min()) for sn, sh in W.sheets.items()},
                   comp_of=comp_of, root=[find(i) for i in range(n_comp)])
    new = {}
    for sn, sh in W.sheets.items():
        r = np.array([find(c) for c in comp_of[sn]])
        new[sn] = sh["ins"] & np.isin(r, list(seed_comps))
    return new


def main(only=None):
    work = json.load(open("working_data.json"))["walls"]
    ex = json.load(open("exact_defs.json"))
    lines = json.load(open("exact_lines.json"))["lines"]
    seeds = {k: v["seed"] for k, v in json.load(open("working_base.json"))["shapes"].items() if v.get("seed")}
    D = (work, ex, lines)
    names = list(work)
    jobs = [(w, sn, D) for w in names for sn in ex["walls"][w]["surface"] if sn in LIB]
    sp = os.path.join(CACHE, f"setup_{N}s.pkl")
    if os.path.exists(sp):
        got = pickle.load(open(sp, "rb"))
    else:
        with Pool(4) as pool:
            pool.map(mesh, sorted({sn for _, sn, _ in jobs}))
            got = pool.map(setup, jobs)
        pickle.dump(got, open(sp, "wb"))
    walls = {}
    for w, sn, V, E, T, ins, funcs, sg in got:
        walls.setdefault(w, {})[sn] = {"V": V, "E": E, "T": T, "ins": ins, "funcs": funcs, "sg": sg}
        print(f"{w} on {sn}: {len(V)} points, {len(funcs)} copies crossing "
              f"({', '.join(sorted({f[1] for f in funcs}))})", flush=True)
    walls = {w: Wall(w, sh, [n for n in ex["walls"][w]["surface"] if n in LIB]) for w, sh in walls.items()}
    extra = []                                             # (sheets found as the walls grew)
    # (a wall is grown again only when a wall bounding it changed in the round before)
    deps = {w: {f[1] for sh in W.sheets.values() for f in sh["funcs"] if f[0] == "edge"} for w, W in walls.items()}
    changed, history = list(walls), []
    for it in range(20):
        new = {}
        for w, W in walls.items():
            if it and w not in changed and not deps[w] & set(changed):
                continue
            t0 = time.time()
            new[w] = grow(W, walls, D, seeds)
            print(f"  {w}: {sum(int(v.sum()) for v in new[w].values())} points ({time.time() - t0:.0f} s)", flush=True)
        changed = [w for w in new if any((new[w][sn] != walls[w].inwall[sn]).any() for sn in walls[w].sheets)]
        for w in new:
            walls[w].inwall, walls[w].tree = new[w], {}
        # (a wall going on into the next cell onto a copy of its surface it was not known to use: that sheet added,
        # where three steps or more land on it)
        for w in new:
            W = walls[w]
            keys = {sheet_key(surface(sn)) for sn in W.sheets}
            for nm, n_ in W.pending.items():
                if n_ < 3 or len(W.sheets) >= 16 or nm in W.sheets:
                    continue
                k = sheet_key(surface(nm))
                if k in keys:
                    continue
                keys.add(k)
                cp = os.path.join(CACHE, "sheet_" + hashlib.md5((w + "|" + nm).encode()).hexdigest()[:12] + f"_{N}.pkl")
                if os.path.exists(cp):
                    r = pickle.load(open(cp, "rb"))
                else:
                    r = setup((w, nm, D))
                    pickle.dump(r, open(cp, "wb"))
                W.sheets[nm] = {"V": r[2], "E": r[3], "T": r[4], "ins": r[5], "funcs": r[6], "sg": r[7]}
                W.inwall[nm] = np.zeros(len(r[2]), bool)
                extra.append(r)
                print(f"  {w}: goes on onto {nm} ({n_} steps; {len(r[6])} copies crossing)", flush=True)
                if w not in changed:
                    changed.append(w)
                deps[w] |= {f[1] for f in r[6] if f[0] == "edge"}
        state = tuple(sorted((w, int(sum(v.sum() for v in walls[w].inwall.values()))) for w in walls))
        print(f"round {it}: changed {changed}", flush=True)
        if not changed:
            break
        if state in history:
            print(f"round {it}: back to the state of round {history.index(state)}: these walls do not settle: {changed}",
                  flush=True)
            break
        history.append(state)
    out = {}
    for w, W in walls.items():
        out[w] = {sn: {"points": int(W.inwall[sn].sum()), "of": int(sh["ins"].sum())} for sn, sh in W.sheets.items()}
        print(w, out[w], flush=True)
    np.savez_compressed(os.path.join(CACHE, "walls.npz"),
                        **{f"{w}|{sn}": W.inwall[sn] for w, W in walls.items() for sn in W.sheets})
    json.dump(out, open("arrangement.json", "w"), indent=1)
    pickle.dump(extra, open(os.path.join(CACHE, f"extra_{N}s.pkl"), "wb"))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
