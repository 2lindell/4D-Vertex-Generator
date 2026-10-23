"""Every wall cut into sections along the lines on it, and each section's two sides named from the region database.

The lines on a wall piece (one of the page's meshes of the wall, original or copy) are
  - the curves where other walls meet it, away from its own edge (meetings_table.py: a wall ending on it, a wall
    crossing it, or a named line some other wall meets it along), each lying on both walls' exact surfaces;
  - the named lines of the page (traced lines, segments, symmetry rings) lying inside it, straight lines in a flat piece;
  - the images of all those on the wall's other pieces: where an element of the extended group (the group and the
    extra half-turn, with the fold into the half-cell) takes one piece of the wall onto another, it takes the lines too.
A line never stops part way across a wall, so each cut runs edge to edge (a meeting found only part way, where a mesh
stops short, is carried on along its exact curve to the piece's edge).

The piece is subdivided (a flat wall is drawn as a few large triangles) and cut along each line, only beside that line,
exactly on its surface; the sections are the triangles between the cuts. Each section is named by the region
database's voxels (region_grid.json) beside it: for each of a few hundred points spread over it, the first labelled voxel
going out on each side. A piece lying in a face of the cell or in the splitting mirror has the inside region's glued
double or mirror image beyond it, so it is named by its inside only. No classification.

    python cut_walls.py X3,X9,X10,X11,X14,X21,T2      (writes sections/<wall>.json.gz)
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
import region_walls as rw
from arrangement import DOMAIN, Mesh, carriers, project, surfaces
from meetings import mesh_edges, sample_segs
from meetings_table import main as meetings
from wall_sections import Grid, surface_of, weld

OUT = "sections"
FLIP = np.array([1, 1, -1.0])
SUB = 0.004            # (pieces subdivided to this edge length)
TUBE = 0.006           # (a line cuts only the triangles this close to it)
REACH = 0.3            # (at most this far a line is carried on along its exact curve to reach an edge or line)
FOLD = np.eye(4)[[1, 0, 3, 2]]


def nrm(B):
    return B / B.sum(axis=-1, keepdims=True)


class Image:
    """A surface carried by a group element g (acting on betas): zero at g x where the surface is zero at x."""

    def __init__(self, s, g):
        self.s, self.A, self.name = s, np.linalg.inv(g), f"{s.name} (carried)"

    def at(self, X):
        return self.s.f((np.c_[X, np.ones(len(X))] @ rc.M_INV.T) @ self.A.T)

    def grad(self, X, e=1e-7):
        return np.stack([(self.at(X + e * np.eye(3)[k]) - self.at(X - e * np.eye(3)[k])) / (2 * e) for k in range(3)], 1)


class PolyCut:
    """A line given by its points in order (a named line, or a line's image), as a cut: near the line, the signed
    distance across it within the piece (the line's direction crossed with the piece's normal)."""

    def __init__(self, name, L, normal, h=2e-4, gap=5e-3):
        L = np.asarray(L, float)
        Q, D = [], []
        for a, b in zip(L[:-1], L[1:]):
            n_ = np.linalg.norm(b - a)
            if n_ < 1e-12 or n_ > gap:
                continue
            k = max(2, int(np.ceil(n_ / h)) + 1)
            Q.append(a + np.linspace(0, 1, k)[:, None] * (b - a))
            D.append(np.tile((b - a) / n_, (k, 1)))
        self.Q, self.D = np.concatenate(Q), np.concatenate(D)
        self.tree, self.normal, self.name = cKDTree(self.Q), normal, name

    def _m(self, X):
        i = self.tree.query(X)[1]
        m = np.cross(self.D[i], self.normal(X))
        return i, m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-30)

    def at(self, X):
        X = np.atleast_2d(X)
        i, m = self._m(X)
        return np.einsum("ij,ij->i", X - self.Q[i], m)

    def grad(self, X):
        return self._m(np.atleast_2d(X))[1]


def subdivide(T, h=SUB):
    """Every triangle split n by n, the same n over the piece, so neighbours share their new vertices."""
    out = []
    n = max(1, int(np.ceil(np.linalg.norm(T - T[:, [1, 2, 0]], axis=2).max() / h)))
    for t in T:
        P = lambda i, j: t[0] + (i / n) * (t[1] - t[0]) + (j / n) * (t[2] - t[0])
        for i in range(n):
            for j in range(n - i):
                out.append([P(i, j), P(i + 1, j), P(i, j + 1)])
                if i + j < n - 1:
                    out.append([P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])
    return np.array(out)


def weld_close(T, tol=1e-6):
    """A triangle soup as vertices and index triangles, vertices closer than tol made one (a copy's carried mesh is
    off by rounding)."""
    P = T.reshape(-1, 3)
    parent = np.arange(len(P))
    for a, b in sorted(cKDTree(P).query_pairs(tol)):
        ra, rb = a, b
        while parent[ra] != ra:
            ra = parent[ra]
        while parent[rb] != rb:
            rb = parent[rb]
        parent[max(ra, rb)] = min(ra, rb)
    root = parent.copy()
    for k in range(len(P)):
        r = k
        while root[r] != r:
            r = root[r]
        root[k] = r
    u, inv = np.unique(root, return_inverse=True)
    return P[u], inv.reshape(-1, 3)


def creases(V, T):
    """The vertices where a piece folded into the half-cell meets its fold: on a face of the cell or the mirror the piece
    does not lie in. The page's view is split there (not the wall): the sides beyond are the mirror images, so the
    piece's orientation is not carried across, and the sections are split there too."""
    def dom(X):
        B = rc.betas(X)
        return np.c_[B, B[:, 2] - B[:, 3]]
    Dv, Dp = np.abs(dom(V)), np.abs(dom(T.reshape(-1, 3)))
    planes = [k for k in range(5) if Dp[:, k].max() > 1e-6]
    return {k: set(np.flatnonzero(Dv[:, k] < 1e-9)) for k in planes}


def is_crease(a, b, crease):
    return any(a in c and b in c for c in crease.values())


def orient(F, crease=None):
    """The triangles turned to one orientation across the mesh (each shared edge run opposite ways by its two),
    not across a crease."""
    crease = crease or {}
    F = np.array(F)
    by = defaultdict(list)
    for t, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            by[(min(e), max(e))].append(t)
    seen = np.zeros(len(F), bool)
    for s0 in range(len(F)):
        if seen[s0]:
            continue
        seen[s0] = True
        todo = [s0]
        while todo:
            t = todo.pop()
            a, b, c = F[t]
            for e in ((a, b), (b, c), (c, a)):
                if is_crease(e[0], e[1], crease):
                    continue
                for u in by[(min(e), max(e))]:
                    if seen[u]:
                        continue
                    seen[u] = True
                    x, y, z = F[u]
                    if (e[0], e[1]) in ((x, y), (y, z), (z, x)):      # (run the same way: turned over)
                        F[u] = [x, z, y]
                    todo.append(u)
    return F


def cut_near(m, s, si, pts, r=TUBE):
    """Mesh.cut, only for the triangles near the points of the line."""
    tree = cKDTree(pts)
    keep_T = m.T
    C = np.array([m.V[list(t)].mean(0) for t in keep_T])
    near = tree.query(C)[0] < r
    far = [t for t, k in zip(keep_T, near) if not k]
    m.T = [t for t, k in zip(keep_T, near) if k]
    n = m.cut(s, si)
    # (vertices the line already runs through, where it follows the mesh's own edges, are on it too)
    vs = sorted({v for t in m.T for v in t})
    if vs:
        X = m.V[vs]
        g = np.abs(s.at(X)) / (np.linalg.norm(s.grad(X), axis=1) + 1e-30)
        for v, gv, dd in zip(vs, g, tree.query(X)[0]):
            if gv < 1e-9 and dd < r:
                m.on[v].add(si)
    m.T = m.T + far
    return n


def plane_normal(T):
    """A flat piece's normal, fitted to all its vertices (one triangle may be a sliver)."""
    X = T.reshape(-1, 3)
    return np.linalg.svd(X - X.mean(0))[2][-1]


def far_ends(P):
    """The two points of a line farthest apart: its ends."""
    P = np.asarray(P)
    i = np.linalg.norm(P - P[0], axis=1).argmax()
    j = np.linalg.norm(P - P[i], axis=1).argmax()
    return P[i], P[j]


def carry_on(pts, sa, sb, edge_tree, inside, stop=None, tol=1.5e-3, over=4, where=None, beside=6e-3):
    """The line's points carried on along its exact curve from both ends until it reaches the piece's edge or another
    line on the piece (stop), and a few steps past, so the cut closes there."""
    pts = np.asarray(pts)
    out = [pts]
    met = lambda x: edge_tree.query(x)[0] < tol or (stop is not None and stop.query(x)[0] < tol)
    for e in far_ends(pts):
        dd = np.linalg.norm(pts - e, axis=1)
        near = pts[(dd > 1e-3) & (dd < 1e-2)]
        if not len(near):
            near = pts[np.argsort(dd)[1:8]]
        t = e - near.mean(0)
        t /= np.linalg.norm(t) + 1e-30
        x, run, ext, past = e.copy(), 0.0, [], 0
        while run < REACH and past < over:
            if met(x):
                past += 1
            y = project((x + 5e-4 * t)[None], [sa, sb])[0]
            if not past and where is not None and where.query(y)[0] > beside:
                break                                 # (past the meeting wall's own edge)
            if not np.isfinite(y).all() or (not past and not inside(y)):
                if os.environ.get("CUTDBG"):
                    print(f"       (carried {run:.4f} on: {'off the curve' if not np.isfinite(y).all() else 'left the piece'}"
                          f", step {np.linalg.norm(y - x):.2e}, edge {edge_tree.query(x)[0]:.4f})")
                break
            t = (y - x) / (np.linalg.norm(y - x) + 1e-30)
            run += np.linalg.norm(y - x)
            x = y
            ext.append(x)
        if ext:
            out.append(np.array(ext))
    return np.concatenate(out)


def extended():
    from t2_pieces import extended as ext
    return ext()


def main(walls):
    want = walls.split(",")
    d = rw.page_data()
    recs, pieces = meetings(want, raw=True)
    surfs = surfaces()
    carry = carriers(surfs)
    surfs = [s for s in surfs if s.name in carry] + DOMAIN
    grid = Grid()
    wall_sides = json.load(open("wall_sides.json"))
    G = extended()
    # (the page's named lines, sampled)
    named = {}
    for key in ("xlines", "segments", "rings"):
        for l in d.get(key, []):
            if isinstance(l, dict) and isinstance(l.get("pts"), list) and len(l["pts"]) > 1:
                named.setdefault(l["id"] + (" (copy)" if l.get("copy") else ""), np.array(l["pts"], float) * FLIP)
    from t2_pieces import DROP
    mine = [pid for pid, p in enumerate(pieces) if p[0] in want and not (p[1] == "tpatch" and p[2] in DROP)]
    S_of = {pid: surface_of(pieces[pid][3], surfs) for pid in mine}
    edge_tree = {pid: cKDTree(sample_segs(mesh_edges(pieces[pid][3]))) for pid in mine}

    def inside_piece(pid):
        T = pieces[pid][3]
        cen = cKDTree(T.mean(1))
        R = np.linalg.norm(T - T.mean(1)[:, None], axis=2).max()

        def f(x):
            for k in cen.query_ball_point(x, R + 1e-9):
                t = T[k]
                M = np.c_[t[1] - t[0], t[2] - t[0]]
                uv = np.linalg.lstsq(M, x - t[0], rcond=None)[0]
                if np.linalg.norm(M @ uv - (x - t[0])) < 5e-4 and uv.min() > -1e-3 and uv.sum() < 1 + 1e-3:
                    return True
            return False
        return f
    inside = {pid: inside_piece(pid) for pid in mine}
    def normal_of(pid):
        S, T = S_of[pid], pieces[pid][3]
        if S is not None:
            return lambda X: S.grad(X) / (np.linalg.norm(S.grad(X), axis=1, keepdims=True) + 1e-30)
        n = plane_normal(T)
        return lambda X: np.tile(n, (len(X), 1))
    # (1) the lines on each piece: (name, cutter, points on the line in the piece, the meeting wall and its points)
    lines = defaultdict(list)
    # (a) the page's named lines lying in the piece, away from its edge: whole, as they are defined
    used = defaultdict(dict)
    for pid in mine:
        T = pieces[pid][3]
        near = cKDTree(subdivide(T, 0.002).mean(1))
        for lid, L in named.items():
            Q = sample_segs(np.stack([L[:-1], L[1:]], 1))
            Q = Q[near.query(Q)[0] < 2e-3]
            if len(Q) < 10:
                continue
            Q = Q[np.array([inside[pid](q) for q in Q])]
            Q = Q[edge_tree[pid].query(Q)[0] > 1.5e-3] if len(Q) else Q
            if len(Q) < 10:                           # (less than 5e-3 of it inside: it passes through, or along the edge)
                continue
            used[pid][lid] = len(lines[pid])
            lines[pid].append([lid, PolyCut(lid, L, normal_of(pid), gap=np.inf), Q, None])
    # (b) the other walls meeting the piece: along a named line, that line (the wall noted on it, for the sides);
    # else their meeting curve, carried on along it exactly to the edge or another line, only where the wall is
    raw = []
    for r in recs:
        pa = r.get("_host")
        if pa not in mine or r["kind"] in ("A ends on B", "same surface"):
            continue
        sa, sb = r["_s"]
        P = r["_pts"]
        if np.median(edge_tree[pa].query(P)[0]) < 1.5e-3:
            continue                                  # (along the piece's own edge: its boundary, not a cut)
        B = r["B"].split(" (")[0]
        Cb = pieces[r["_other"]][3].mean(1)
        db = cKDTree(P).query(Cb)[0]
        wb = (B, Cb[(db > 2e-3) & (db < 1.2e-2)])
        if r["kind"] == "named line":
            ids = [x for x in r["line"].split(", ") if x in used[pa]]
            for x in ids:
                if lines[pa][used[pa][x]][3] is None:
                    lines[pa][used[pa][x]][3] = wb
            if ids:
                continue
        if sb is None:
            continue
        raw.append((pa, B, sa, sb, P, wb, cKDTree(pieces[r["_other"]][3].reshape(-1, 3))))
    for pa, B, sa, sb, P, wb, btree in raw:
        if any(np.median(cKDTree(x[2]).query(P)[0]) < 1e-3 for x in lines[pa]):
            continue                                  # (the same line, met by another wall too)
        if sa is not None:
            others = [x[2] for x in lines[pa]] + [x[4] for x in raw if x[0] == pa and x[4] is not P]
            if os.environ.get("CUTDBG"):
                print(f"     carrying on {B} on piece {pa}, {len(P)} pts:")
            P = carry_on(P, sa, sb, edge_tree[pa], inside[pa], cKDTree(np.concatenate(others)) if others else None,
                         where=btree)
        lines[pa].append([B, sb, P, wb])
    for pid in lines:
        lines[pid] = [tuple(x) for x in lines[pid]]
    # (2) the images of each piece's lines on the wall's other pieces
    by_wall = defaultdict(list)
    for pid in mine:
        by_wall[pieces[pid][0]].append(pid)
    for w, pids in by_wall.items():
        for p in pids:
            Sp = rc.betas(pieces[p][3][:, 0])
            Sp = Sp[np.linspace(0, len(Sp) - 1, min(len(Sp), 24)).astype(int)]
            for q in pids:
                if q == p or S_of[q] is None or not lines[p]:
                    continue
                for M in G:
                    hit = None
                    for F in (np.eye(4), FOLD):
                        g = F @ M
                        Y = Sp @ g.T
                        if (np.abs(Y.sum(1)) < 1e-12).any():
                            continue
                        X = rc.xyz(nrm(Y))
                        if np.abs(S_of[q].at(X)).max() / (np.linalg.norm(S_of[q].grad(X), axis=1).min() + 1e-30) < 1e-6:
                            hit = g
                            break
                    if hit is None:
                        continue
                    for nm, s, P, wb in lines[p]:
                        Y = rc.betas(P) @ hit.T
                        ok = np.abs(Y.sum(1)) > 1e-12
                        X = rc.xyz(nrm(Y[ok]))
                        X = X[[inside[q](x) for x in X]] if len(X) else X
                        if len(X) < 5 or any(np.median(cKDTree(P2).query(X)[0]) < 1e-3 for _, _, P2, _ in lines[q]):
                            continue
                        if wb is not None and len(wb[1]):
                            Yb = rc.betas(wb[1]) @ hit.T      # (the meeting wall carried along with its line)
                            okb = np.abs(Yb.sum(1)) > 1e-12
                            wb = (wb[0], rc.xyz(nrm(Yb[okb])))
                        if isinstance(s, PolyCut):
                            # (a line given by its points: its image, by the images of its points, in order)
                            Y = rc.betas(s.Q) @ hit.T
                            img = PolyCut(nm, rc.xyz(nrm(Y)), normal_of(q), gap=5 * 2e-4)
                        else:
                            img = Image(s, hit)
                        lines[q].append((nm + " (carried)", img, X, wb))
    # (3) cut, gather sections, name them
    os.makedirs(OUT, exist_ok=True)
    out = defaultdict(list)
    for pid in mine:
        w, kind, _, T = pieces[pid]
        Ts = subdivide(T)
        V, F = weld_close(Ts)
        crease = creases(V, T)
        F = orient(F, crease)
        m = Mesh(V, F, S_of[pid] if S_of[pid] is not None and S_of[pid].kind != "domain" else None)
        names = []
        for k, (nm, s, P, _) in enumerate(lines[pid]):
            Xs = T.mean(1)
            if np.median(np.abs(s.at(Xs)) / (np.linalg.norm(s.grad(Xs), axis=1) + 1e-30)) < 1e-6:
                names.append(nm)
                continue                              # (the cutter holds the piece itself: no line on it)
            nc = cut_near(m, s, k, P)
            names.append(nm)
            if os.environ.get("CUTDBG"):
                e0, e1 = edge_tree[pid].query(np.array(far_ends(P)))[0]
                print(f"     {nm}: {nc} triangles cut; {len(P)} pts, end-to-edge {e0:.4f} {e1:.4f}, "
                      f"|s| on pts {np.median(np.abs(s.at(P))):.2e}", flush=True)
        Ti = np.array(m.T)
        crease = creases(m.V, T)                      # (with the cuts' new vertices on a crease too)
        # (sections: triangles joined across every edge that is not on a cut)
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
        for (a, b), ts in by_edge.items():
            if m.on.get(a, set()) & m.on.get(b, set()) or is_crease(a, b, crease):
                continue                              # (both ends on one cut: the edge lies along it)
            for t in ts[1:]:
                parent[find(t)] = find(ts[0])
        comp = np.array([find(t) for t in range(len(Ti))])
        Tri = m.V[Ti]
        A = 0.5 * np.linalg.norm(np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0]), axis=1)
        # (a scrap of a mesh, a few triangles cut off by a gap where the page's mesh was clipped, with no cut on it,
        # belongs to the section nearest it)
        oncut = {c for t, c in zip(Ti, comp) if any(m.on.get(v_) for v_ in t)}
        size = defaultdict(float)
        for c, a_ in zip(comp, A):
            size[c] += a_
        big = [c for c in size if size[c] >= 0.02 * A.sum() or c in oncut]
        if big:
            bt = cKDTree(Tri[np.isin(comp, big)].reshape(-1, 3))
            bc = np.repeat(comp[np.isin(comp, big)], 3)
            for c in [c for c in size if c not in big]:
                k = np.flatnonzero(comp == c)
                comp[k] = bc[bt.query(Tri[k].reshape(-1, 3))[1][np.argmin(bt.query(Tri[k].reshape(-1, 3))[0])]]
        N = np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0])
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-300
        Bv = rc.betas(T.reshape(-1, 3))
        face = [k for k in range(4) if np.abs(Bv[:, k]).max() < 1e-6]
        into = np.eye(4)[face[0]] if face else (np.array([0, 0, 0.5, -0.5]) if np.abs(Bv[:, 2] - Bv[:, 3]).max() < 1e-6 else None)
        rng = np.random.default_rng(pid)
        vox, secs = {}, [c for c in np.unique(comp) if A[comp == c].sum() >= 1e-8]
        for c in secs:
            idx = np.flatnonzero(comp == c)
            a = A[idx]
            j = idx[rng.choice(len(idx), 300, p=a / a.sum())]
            u, v = rng.random(300), rng.random(300)
            fl = u + v > 1
            u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
            X = Tri[j, 0] + u[:, None] * (Tri[j, 1] - Tri[j, 0]) + v[:, None] * (Tri[j, 2] - Tri[j, 0])
            nn = N[j]                                 # (the mesh is oriented: one side throughout)
            dirs = [into] if into is not None else [1, -1]
            sides = []
            for dr in dirs:
                first = np.zeros(len(X), int)
                for k in (1.5, 2.5, 3.5, 4.5, 5.5):
                    if into is not None:
                        Y = rc.xyz(nrm(rc.betas(X) + k * grid.h * into))
                    else:
                        Y = X + dr * k * grid.h * nn
                    lab = grid.at(Y)
                    take = (first == 0) & (lab > 0)
                    first[take] = lab[take]
                if not (first > 0).any() and into is None:
                    from region_db import home
                    for k in (1.5, 2.5, 3.5, 4.5, 5.5):
                        Bh = home(rc.betas(X + dr * k * grid.h * nn))
                        ok = ~np.isnan(Bh).any(1)
                        lab = np.zeros(len(X), int)
                        lab[ok] = grid.at(rc.xyz(Bh[ok]))
                        take = (first == 0) & (lab > 0)
                        first[take] = lab[take]
                cnt = np.bincount(first[first > 0]) if (first > 0).any() else None
                sides.append(grid.names[int(cnt.argmax())] if cnt is not None else "unknown")
            if into is not None:
                sides.append("(beyond the half-cell)")
            vox[c] = sides
        # (the sides named once, from the database: the largest section by its voxels, and each section beside it
        # across a line from the one before: on the side the meeting wall stands, the region changes to the wall's
        # other side; on the other side it stays)
        across = defaultdict(set)
        for (a_, b_), ts in by_edge.items():
            for k in m.on.get(a_, set()) & m.on.get(b_, set()):
                cs = {comp[t] for t in ts}
                if len(cs) == 2:
                    c1, c2 = cs
                    across[c1].add((c2, k))
                    across[c2].add((c1, k))
        cen = cKDTree(Tri.mean(1))
        stand = {}
        for k, x in enumerate(lines[pid]):
            wb = x[3]
            if wb is None or not len(wb[1]) or wb[0] not in wall_sides or len(wall_sides[wb[0]]) != 2:
                continue
            if into is not None:
                stand[k] = (wb[0], [0])
                continue
            dd, ti = cen.query(wb[1])
            sg = np.sign(np.einsum("ij,ij->i", wb[1] - Tri[ti].mean(1), N[ti]))
            st = ([0] if (sg > 0).mean() > 0.1 else []) + ([1] if (sg < 0).mean() > 0.1 else [])
            stand[k] = (wb[0], st)
        name, how = {}, {}
        for c0 in sorted(secs, key=lambda c: -A[comp == c].sum()):
            if c0 in name:
                continue
            name[c0], how[c0] = list(vox[c0]), "voxels"
            todo = [c0]
            while todo:
                c = todo.pop()
                for c2, k in sorted(across[c]):
                    if c2 in name or c2 not in vox:
                        continue
                    if k not in stand:
                        name[c2], how[c2] = list(vox[c2]), "voxels"
                    else:
                        B, st = stand[k]
                        nw = list(name[c])
                        for sd in st:
                            pair = wall_sides[B]
                            nw[sd] = pair[1 - pair.index(nw[sd])] if nw[sd] in pair else vox[c2][sd]
                        name[c2], how[c2] = nw, f"across {B}"
                    todo.append(c2)
        for c in secs:
            idx = np.flatnonzero(comp == c)
            bounding = sorted({names[k] for t in Ti[idx] for v_ in t for k in m.on.get(v_, set()) if 0 <= k < len(names)})
            out[w].append({"piece": pid, "kind": kind, "sides": name[c], "named": how[c], "voxels": vox[c],
                           "lines": bounding, "mirror": False, "cut": [False, False],
                           "area": round(float(A[idx].sum()), 6), "tris": Tri[idx].reshape(-1, 9).round(7).tolist()})
        print(f"  {w} ({kind}, piece {pid}): {len(lines[pid])} lines ({', '.join(x[0] for x in lines[pid]) or 'none'}); "
              f"{len(set(comp))} sections: " + "; ".join(f"{f['sides'][0]}|{f['sides'][1]} {f['area']:.4f}" + (f" ({f['named']})" if f['named'] != "voxels" else "")
                                                        for f in out[w] if f["piece"] == pid), flush=True)
    for w, faces in out.items():
        json.dump(faces, gzip.open(os.path.join(OUT, f"{w}.json.gz"), "wt"), separators=(",", ":"))


if __name__ == "__main__":
    main(sys.argv[1])
