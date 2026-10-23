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
TUBE = 0.012           # (a line cuts only the triangles this close to it)
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

    def __init__(self, name, L, normal, h=2e-4, gap=5e-3, segs=None):
        L = np.asarray(L, float)
        Q, D = [], []
        self.segs = None if segs is None else np.asarray(segs, float)
        pairs = list(zip(L[:-1], L[1:])) if segs is None else [(a, b) for a, b in self.segs]
        for a, b in pairs:
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


def sample_tris(T, h, owner=False):
    """Points spread over the triangles about h apart (each triangle on its own grid): centres of its sub-triangles
    (with owner: also the triangle each point lies in)."""
    out, own = [], []
    L = np.linalg.norm(T - T[:, [1, 2, 0]], axis=2).max(1)
    ns = np.maximum(1, np.ceil(L / h)).astype(int)
    for n in np.unique(ns):
        Tn = T[ns == n]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        k = i + j < n
        u, v = (i[k] + 1 / 3) / n, (j[k] + 1 / 3) / n
        out.append((Tn[:, None, 0] + u[None, :, None] * (Tn[:, None, 1] - Tn[:, None, 0])
                    + v[None, :, None] * (Tn[:, None, 2] - Tn[:, None, 0])).reshape(-1, 3))
        own.append(np.repeat(np.flatnonzero(ns == n), len(u)))
    return (np.concatenate(out), np.concatenate(own)) if owner else np.concatenate(out)


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


class EdgeMesh(Mesh):
    """Mesh.cut with each new point found on its own edge: the cutter's zero bisected along the edge (each trial
    point put on the wall's surface), so a cut point never slides off the edge it splits."""

    def cut(self, s, si):
        g = s.at(self.V)
        scale = np.abs(g).max()
        if scale < 1e-14:
            return 0
        g[np.abs(g) < 1e-12 * scale] = 0.0
        pairs, cache, out, n_cut = [], {}, [], 0

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                cache[k] = len(self.V) + len(pairs)
                pairs.append((i, j))
                self.on[cache[k]] |= (self.on[i] & self.on[j]) | {si}
            return cache[k]
        for tri in self.T:
            sg = np.sign(g[list(tri)])
            if not ((sg > 0).any() and (sg < 0).any()):
                out.append(tri)
                continue
            n_cut += 1
            a, b, c = tri
            zero = 0 in sg
            for _ in range(3):
                sa, sb, sc = np.sign(g[[a, b, c]])
                if (sa == 0) if zero else (sa != sb and sa != sc):
                    break
                a, b, c = b, c, a
            if zero:
                p = mid(b, c)
                self.on[a].add(si)
                out += [(a, b, p), (a, p, c)]
            else:
                p, q = mid(a, b), mid(a, c)
                out += [(a, p, q), (p, b, c), (p, c, q)]
        if pairs:
            I, J = np.array(pairs).T
            lo, hi = np.zeros(len(I)), np.ones(len(I))
            gi = g[I]
            on_w = (lambda X: project(X, [self.surf])) if self.surf is not None else (lambda X: X)
            for _ in range(40):
                t = (lo + hi) / 2
                X = on_w(self.V[I] + t[:, None] * (self.V[J] - self.V[I]))
                same = np.sign(s.at(X)) == np.sign(gi)
                lo = np.where(same, t, lo)
                hi = np.where(same, hi, t)
            t = (lo + hi) / 2
            self.V = np.vstack([self.V, on_w(self.V[I] + t[:, None] * (self.V[J] - self.V[I]))])
        self.T = out
        return n_cut


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


def flat_cut(T, plines):
    """A flat piece cut exactly in its own plane: its outline and the lines on it noded together, the faces between
    them the sections. A line is the cutter's zero set in the plane near its points (a named line: its own points);
    a line ending inside the piece leaves no face behind. Returns the sections' triangles, the section of each, the
    sections each line separates (section, other, line) and the lines bounding each section."""
    import shapely
    from shapely.geometry import LineString, Polygon
    from shapely.ops import polygonize, unary_union
    from skimage.measure import find_contours
    X = T.reshape(-1, 3)
    o, n = X.mean(0), plane_normal(T)
    e1 = np.linalg.svd(X - o)[2][0]
    e2 = np.cross(n, e1)
    to2 = lambda P: np.c_[(np.asarray(P) - o) @ e1, (np.asarray(P) - o) @ e2]
    poly = unary_union([Polygon(to2(t)).buffer(1e-10) for t in T]).buffer(-1e-10)
    lo, hi = np.array(poly.bounds[:2]) - 0.003, np.array(poly.bounds[2:]) + 0.003

    def runs(Q, gap):
        out, cur = [], [Q[0]]
        for a, b in zip(Q[:-1], Q[1:]):
            if np.linalg.norm(b - a) > gap:
                if len(cur) > 1:
                    out.append(LineString(cur))
                cur = []
            cur.append(b)
        if len(cur) > 1:
            out.append(LineString(cur))
        return out
    geoms, whole = {}, {}
    near_piece = poly.buffer(0.002)
    for k, (nm, s, P, _) in enumerate(plines):
        Xs = T.mean(1)
        if np.median(np.abs(s.at(Xs)) / (np.linalg.norm(s.grad(Xs), axis=1) + 1e-30)) < 1e-6:
            continue                                  # (the cutter holds the piece itself: no line on it)
        if isinstance(s, PolyCut) and s.segs is not None:
            # (another wall's own edge lying in the piece: its segments, joined, each run carried on straight)
            from shapely.ops import linemerge
            sg = [LineString(to2(seg)) for seg in s.segs if np.abs((seg - o) @ n).max() < 2e-4]
            if not sg:
                continue
            u_ = unary_union(sg)
            m_ = linemerge(u_) if u_.geom_type == "MultiLineString" else u_
            ls = list(getattr(m_, "geoms", [m_]))
            for i_, r_ in enumerate(ls):
                C = np.array(r_.coords)
                d0 = C[0] - C[1]
                d1 = C[-1] - C[-2]
                d0, d1 = d0 / (np.linalg.norm(d0) + 1e-30), d1 / (np.linalg.norm(d1) + 1e-30)
                ext = np.linspace(0, 0.03, 61)[1:, None]
                whole[(k, i_)] = (np.vstack([(C[0] + ext * d0)[::-1], C, C[-1] + ext * d1]), 60, 60 + len(C))
        elif isinstance(s, PolyCut):
            Q = s.Q[np.abs((s.Q - o) @ n) < 2e-4]
            if len(Q) < 2:
                continue
            Q2 = to2(Q)
            keep = np.array([near_piece.contains(shapely.Point(q)) for q in Q2])
            ls = runs(Q2[keep], 3e-3) if keep.sum() > 1 else []
            # (a named line's runs carried on straight from their ends, to close a gap or reach the outline)
            for i_, r_ in enumerate(ls):
                C = np.array(r_.coords)
                if len(C) < 2:
                    continue
                d0 = C[0] - C[min(3, len(C) - 1)]
                d1 = C[-1] - C[max(-4, -len(C))]
                d0, d1 = d0 / (np.linalg.norm(d0) + 1e-30), d1 / (np.linalg.norm(d1) + 1e-30)
                ext = np.linspace(0, 0.03, 61)[1:, None]
                C2 = np.vstack([(C[0] + ext * d0)[::-1], C, C[-1] + ext * d1])
                whole[(k, i_)] = (C2, 60, 60 + len(C))
        else:
            # (the zero set of the cutter in the plane, on a grid over the piece; kept beside the line's points, the
            # rest kept to carry the line on with below)
            P2 = to2(P)
            h = 4e-4
            u = np.arange(lo[0], hi[0] + h, h)
            v = np.arange(lo[1], hi[1] + h, h)
            U, Vv = np.meshgrid(u, v, indexing="ij")
            G3 = o + U.reshape(-1, 1) * e1 + Vv.reshape(-1, 1) * e2
            F = s.at(G3).reshape(U.shape)
            tree = cKDTree(P2)
            ls = []
            for c in find_contours(F, 0.0):
                C2 = np.c_[lo[0] + c[:, 0] * h, lo[1] + c[:, 1] * h]
                keep = tree.query(C2)[0] < 2.5e-3
                for a_, b_ in _true_runs(keep):
                    if b_ - a_ > 1:
                        ls.append(LineString(C2[a_:b_]))
                        whole[(k, len(ls) - 1)] = (C2, a_, b_)
        if ls:
            geoms[k] = unary_union(ls).simplify(2e-5)
        if os.environ.get("CUTDBG"):
            g_ = geoms.get(k)
            print(f"     {nm}: {type(s).__name__}, {len(P)} pts, {len(ls)} runs, length {g_.length if g_ else 0:.4f}, "
                  f"meets the outline at {len(list(getattr(g_.intersection(poly.boundary), 'geoms', [0]))) if g_ else 0}"
                  f", ends off the outline {[round(poly.boundary.distance(shapely.Point(c)), 4) for r in (getattr(g_, 'geoms', [g_]) if g_ else []) for c in (r.coords[0], r.coords[-1])]}")
    # (each run carried on along its zero set, both ways, to where it first meets the outline or another line: a
    # cut never stops part way; twice, so two lines carried toward each other meet)
    def carried(k, C2, a_, b_, stop):
        pts = list(C2[a_:b_])
        for fwd in (True, False):
            path = C2[b_ - 1:] if fwd else C2[:a_ + 1][::-1]
            if len(path) < 2:
                continue
            ext = LineString(path)
            hit = ext.intersection(stop)
            if hit.is_empty:
                continue
            ds = [ext.project(p_) for p_ in getattr(hit, "geoms", [hit]) for p_ in
                  ([p_] if p_.geom_type == "Point" else [shapely.Point(p_.coords[0])])]
            d_ = min(ds) if ds else None             # (an end already touching: carried just through it)
            if d_ is None or d_ > 0.15:
                continue
            seg = shapely.ops.substring(ext, 0, d_ + 1e-6)
            add = list(np.array(seg.coords)[1:])
            pts = pts + add if fwd else add[::-1] + pts
        return LineString(pts)
    import shapely.ops
    for _ in range(2):
        for k in list(geoms):
            runs_k = [key for key in whole if key[0] == k]
            if not runs_k:
                continue
            parts = {key: LineString(whole[key][0][whole[key][1]:whole[key][2]]) for key in runs_k}
            done_k = []
            for key in runs_k:
                stop = unary_union([poly.boundary] + [x for j, x in geoms.items() if j != k]
                                   + [g for kk, g in parts.items() if kk != key])
                done_k.append(carried(k, *whole[key], stop))
            geoms[k] = unary_union(done_k).simplify(2e-5)
    if os.environ.get("CUTDBG"):
        for k, g_ in geoms.items():
            oth = unary_union([poly.boundary] + [x for j, x in geoms.items() if j != k])
            print(f"     line {plines[k][0]}: ends off the outline or another line "
                  f"{[round(oth.distance(shapely.Point(c)), 4) for r in getattr(g_, 'geoms', [g_]) for c in (r.coords[0], r.coords[-1])]}")
    noded = unary_union([poly.boundary] + list(geoms.values()))
    faces = [f for f in polygonize(noded) if f.area > 1e-12 and poly.contains(f.representative_point())]
    tris, comp = [], []
    for i, f in enumerate(faces):
        for tri in shapely.constrained_delaunay_triangles(f).geoms:
            c = np.array(tri.exterior.coords)[:3]
            t3 = o + c[:, :1] * e1 + c[:, 1:2] * e2
            if np.cross(t3[1] - t3[0], t3[2] - t3[0]) @ n < 0:
                t3 = t3[[0, 2, 1]]                    # (all one way round: the piece's normal)
            tris.append(t3)
            comp.append(i)
    across, bounding = defaultdict(set), defaultdict(set)
    for i, f in enumerate(faces):
        for k, g in geoms.items():
            if f.boundary.intersection(g.buffer(1e-7)).length > 1e-6:
                bounding[i].add(k)
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            sh = faces[i].boundary.intersection(faces[j].boundary)
            if sh.length <= 1e-6:
                continue
            m_ = sh.interpolate(0.5, normalized=True)
            for k, g in geoms.items():
                if g.distance(m_) < 1e-6:
                    across[i].add((j, k))
                    across[j].add((i, k))
    return np.array(tris).reshape(-1, 3, 3), np.array(comp, int), across, bounding


def _true_runs(mask):
    """(start, end) of each run of True."""
    out, a = [], None
    for i, x in enumerate(list(mask) + [False]):
        if x and a is None:
            a = i
        elif not x and a is not None:
            out.append((a, i))
            a = None
    return out


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


BEYOND_ = "(beyond the half-cell)"


def copies_as_main(out, G):
    """Each copy's sections named as the main piece's sections they are the images of."""
    # (a copy is the wall carried by an element of the extended group, regions with it: each of its sections takes
    # the sides of the main piece's section it is the image of, not its own voxel votes)
    GG = np.array([F @ M for M in G for F in (np.eye(4), FOLD)])
    for w, faces in out.items():
        is_copy = lambda f: f.get("copy", f["kind"] == "copy")
        main_f = [f for f in faces if not is_copy(f) and BEYOND_ not in f["sides"]]   # (not a face's view of it)
        if not main_f:
            continue
        T0 = np.concatenate([np.array(f["tris"]).reshape(-1, 3, 3) for f in main_f])
        P0 = T0.mean(1)
        lab0 = np.concatenate([[k] * len(f["tris"]) for k, f in enumerate(main_f)])
        tree0 = cKDTree(P0)
        reach = 0.75 * max(float(np.linalg.norm(np.array(f["tris"]).reshape(-1, 3, 3) - np.array(f["tris"]).reshape(-1, 3, 3).mean(1)[:, None], axis=2).max()) for f in main_f)
        for f in faces:
            if not is_copy(f) or BEYOND_ in f["sides"]:
                continue
            T_ = np.array(f["tris"]).reshape(-1, 3, 3)
            A_ = 0.5 * np.linalg.norm(np.cross(T_[:, 1] - T_[:, 0], T_[:, 2] - T_[:, 0]), axis=1)
            X_ = T_[np.argsort(-A_)[:min(len(T_), 12)]].mean(1)
            Y = np.einsum("gij,nj->gni", GG, rc.betas(X_))
            s_ = Y.sum(2, keepdims=True)
            ok = (np.abs(s_[..., 0]) > 1e-12).all(1)
            best = None
            for g in np.flatnonzero(ok):
                B = Y[g] / s_[g]
                if (B < -1e-9).any() or (B[:, 2] < B[:, 3] - 1e-9).any():
                    continue
                Xg = rc.xyz(B)
                dd, _ = tree0.query(Xg)
                if dd.max() > reach:
                    continue
                # (each image point inside a main triangle, exactly: on its plane and within it)
                labs_ = []
                for x in Xg:
                    hit = None
                    for t in tree0.query_ball_point(x, reach):
                        tri = T0[t]
                        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
                        nrm_ = np.cross(e1, e2)
                        if abs((x - tri[0]) @ nrm_) > 1e-6 * np.linalg.norm(nrm_) * max(np.linalg.norm(e1), 1e-9) * 10:
                            continue
                        uv = np.linalg.lstsq(np.c_[e1, e2], x - tri[0], rcond=None)[0]
                        if uv.min() > -1e-6 and uv.sum() < 1 + 1e-6:
                            hit = lab0[t]
                            break
                    labs_.append(hit)
                n_in = sum(h_ is not None for h_ in labs_)
                if n_in >= 0.9 * len(labs_) and (best is None or n_in > best[0]):
                    best = (n_in, np.array([h_ for h_ in labs_ if h_ is not None]))
            if best is None:
                continue
            k = np.bincount(best[1]).argmax()
            src = main_f[k]["sides"]
            if set(src) != set(f["sides"]):
                f["voxels_said"] = f["sides"]
                f["sides"], f["named"] = list(src), "as its main piece"
                print(f"  {w} copy section {f['area']:.4f}: {'|'.join(f['voxels_said'])} -> {'|'.join(src)} (as its main piece)")


def curved_cut(T, plines, S, names):
    """A curved piece cut along its lines on its own mesh: the sections as triangles joined across every edge not on
    a cut (a scrap of the mesh cut off by a gap joins the section nearest it)."""
    V, F = weld_close(T)
    crease = creases(V, T)
    F = orient(F, crease)
    m = EdgeMesh(V, F, S if S is not None and S.kind != "domain" else None)
    for k, (nm, s, P, _) in enumerate(plines):
        Xs = T.mean(1)
        if np.median(np.abs(s.at(Xs)) / (np.linalg.norm(s.grad(Xs), axis=1) + 1e-30)) < 1e-6:
            continue                              # (the cutter holds the piece itself: no line on it)
        cut_near(m, s, k, P)
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
    across = defaultdict(set)
    for (a_, b_), ts in by_edge.items():
        for k in m.on.get(a_, set()) & m.on.get(b_, set()):
            cs = {comp[t] for t in ts}
            if len(cs) == 2:
                c1, c2 = cs
                across[c1].add((c2, k))
                across[c2].add((c1, k))
    bounding_of = {c: sorted({names[k] for t in Ti[comp == c] for v_ in t for k in m.on.get(v_, set()) if 0 <= k < len(names)})
                   for c in set(comp)}
    return Tri, comp, A, across, bounding_of


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
        near = cKDTree(sample_tris(T, 0.002))
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
        P = P[edge_tree[pa].query(P)[0] >= 1.5e-3]   # (the part along the piece's own edge is its boundary, not a cut)
        if len(P) < 5 or np.linalg.norm(P.max(0) - P.min(0)) < 3e-3:
            continue
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
    # (c) another wall ending on the piece exactly along its own edge: that edge lies in the piece's surface but
    # the two meshes only touch there, so no crossing of triangles finds it; it is read off the walls' outlines
    # (to 1e-4: a curved wall's outline is exact, but the page keeps its points to 6 decimals)
    bbox = {pid: (pieces[pid][3].reshape(-1, 3).min(0) - 0.01, pieces[pid][3].reshape(-1, 3).max(0) + 0.01)
            for pid in range(len(pieces))}
    outl = {}
    for pa in mine:
        S = S_of[pa]
        if S is None:
            continue
        near_a = cKDTree(sample_tris(pieces[pa][3], 0.002))
        for pb in range(len(pieces)):
            if pb == pa:
                continue
            lo_, hi_ = bbox[pb]
            if (lo_ > bbox[pa][1]).any() or (hi_ < bbox[pa][0]).any():
                continue
            if pb not in outl:
                outl[pb] = mesh_edges(pieces[pb][3])
            E = outl[pb]
            if not len(E):
                continue
            X_ = E.reshape(-1, 3)
            r_ = (np.abs(S.at(X_)) / (np.linalg.norm(S.grad(X_), axis=1) + 1e-30)).reshape(-1, 2).max(1)
            M_ = E.mean(1)
            keep = (r_ < 1e-4) & (near_a.query(M_)[0] < 1.5e-3) & (edge_tree[pa].query(M_)[0] > 1.5e-3)
            if not keep.any() or np.linalg.norm(E[keep, 1] - E[keep, 0], axis=1).sum() < 3e-3:
                continue
            seg = E[keep]
            P = sample_segs(seg)
            if any(np.median(cKDTree(x[2]).query(P)[0]) < 1e-3 for x in lines[pa]):
                continue                              # (already a line on the piece)
            B = pieces[pb][0]
            Cb = pieces[pb][3].mean(1)
            db = cKDTree(P).query(Cb)[0]
            lines[pa].append([f"{B} (its edge)", PolyCut(f"{B} (its edge)", P[:2], normal_of(pa), gap=np.inf, segs=seg), P,
                              (B, Cb[(db > 2e-3) & (db < 1.2e-2)])])
            if os.environ.get("CUTDBG"):
                print(f"     {B}'s edge lies in piece {pa}: {keep.sum()} segments", flush=True)
    for pid in lines:
        lines[pid] = [tuple(x) for x in lines[pid]]
    # (2) the images of each piece's lines on the wall's other pieces
    by_wall = defaultdict(list)
    for pid in mine:
        by_wall[pieces[pid][0]].append(pid)
    GG = np.array([F @ M for M in G for F in (np.eye(4), FOLD)])
    dense = {}

    def dense_of(q):
        if q not in dense:
            dense[q] = cKDTree(sample_tris(pieces[q][3], 0.002))
        return dense[q]

    def images(Pb, gs):
        """Points (betas) carried by each element: (len(gs), n, 3) in xyz, NaN where they leave the chart."""
        Y = np.einsum("gij,nj->gni", GG[gs], Pb)
        s_ = Y.sum(2, keepdims=True)
        bad = np.abs(s_[..., 0]) < 1e-12
        X = rc.xyz((Y / np.where(np.abs(s_) < 1e-12, 1.0, s_)).reshape(-1, 4)).reshape(len(gs), -1, 3)
        X[bad] = np.nan
        return X
    for w, pids in by_wall.items():
        for p in pids:
            if not lines[p]:
                continue
            T_ = pieces[p][3]
            Sp = rc.betas(sample_tris(T_, 0.02))
            Sp = Sp[np.linspace(0, len(Sp) - 1, min(len(Sp), 40)).astype(int)]
            for q in pids:
                if q == p or S_of[q] is None:
                    continue
                # (the elements taking piece p onto piece q: its points onto q's surface, most of them onto q itself)
                X = images(Sp, np.arange(len(GG)))
                fin = np.isfinite(X).all((1, 2))
                gs = np.flatnonzero(fin)
                Xf = X[gs].reshape(-1, 3)
                f = (np.abs(S_of[q].at(Xf)) / (np.linalg.norm(S_of[q].grad(Xf), axis=1) + 1e-30)).reshape(len(gs), -1)
                gs = gs[f.max(1) < 1e-6]
                near = (dense_of(q).query(X[gs].reshape(-1, 3))[0] < 3e-3).reshape(len(gs), -1).mean(1) if len(gs) else []
                hits = [g for g, m_ in zip(gs, near) if m_ > 0.3]
                for g in hits:
                    hit = GG[g]
                    for nm, s, P, wb in lines[p]:
                        Xl = images(rc.betas(P), [g])[0]
                        Xl = Xl[np.isfinite(Xl).all(1)]
                        Xl = Xl[dense_of(q).query(Xl)[0] < 2e-3] if len(Xl) else Xl
                        if len(Xl) < 5 or any(np.median(cKDTree(P2).query(Xl)[0]) < 1e-3 for _, _, P2, _ in lines[q]):
                            continue
                        if wb is not None and len(wb[1]):
                            Xb = images(rc.betas(wb[1]), [g])[0]   # (the meeting wall carried along with its line)
                            wb = (wb[0], Xb[np.isfinite(Xb).all(1)])
                        if isinstance(s, PolyCut):
                            # (a line given by its points: its image, by the images of its points, in order)
                            Xq = images(rc.betas(s.Q), [g])[0]
                            img = PolyCut(nm, Xq[np.isfinite(Xq).all(1)], normal_of(q), gap=5 * 2e-4)
                        else:
                            img = Image(s, hit)
                        lines[q].append((nm + " (carried)", img, Xl, wb))
    # (3) cut, gather sections, name them
    os.makedirs(OUT, exist_ok=True)
    out = defaultdict(list)
    for pid in mine:
        w, kind, _, T = pieces[pid]
        flat = S_of[pid] is not None and S_of[pid].kind in ("plane", "domain")
        names = [x[0] for x in lines[pid]]
        if flat:
            # (a flat piece: cut exactly in its plane)
            Tri, comp, across, bnd = flat_cut(T, lines[pid])
            A = 0.5 * np.linalg.norm(np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0]), axis=1)
            bounding_of = {c: sorted({names[k] for k in bnd.get(c, ())}) for c in set(comp)}
        else:
            Tri, comp, A, across, bounding_of = curved_cut(T, lines[pid], S_of[pid], names)
        N = np.cross(Tri[:, 1] - Tri[:, 0], Tri[:, 2] - Tri[:, 0])
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-300
        Bv = rc.betas(T.reshape(-1, 3))
        face = [k for k in range(4) if np.abs(Bv[:, k]).max() < 1e-6]
        into = np.eye(4)[face[0]] if face else (np.array([0, 0, 0.5, -0.5]) if np.abs(Bv[:, 2] - Bv[:, 3]).max() < 1e-6 else None)
        rng = np.random.default_rng(pid)
        vox, secs = {}, [c for c in np.unique(comp) if A[comp == c].sum() >= 1e-5]   # (smaller: degenerate slivers along a line)
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
            bounding = bounding_of.get(c, [])
            out[w].append({"piece": pid, "kind": kind, "sides": name[c], "named": how[c], "voxels": vox[c],
                           "lines": bounding, "mirror": False, "cut": [False, False],
                           "area": round(float(A[idx].sum()), 6), "tris": Tri[idx].reshape(-1, 9).round(7).tolist()})
        print(f"  {w} ({kind}, piece {pid}): {len(lines[pid])} lines ({', '.join(x[0] for x in lines[pid]) or 'none'}); "
              f"{len(set(comp))} sections: " + "; ".join(f"{f['sides'][0]}|{f['sides'][1]} {f['area']:.4f}" + (f" ({f['named']})" if f['named'] != "voxels" else "")
                                                        for f in out[w] if f["piece"] == pid), flush=True)
    copies_as_main(out, G)
    for w, faces in out.items():
        json.dump(faces, gzip.open(os.path.join(OUT, f"{w}.json.gz"), "wt"), separators=(",", ":"))


if __name__ == "__main__":
    if sys.argv[1] == "--copies":                     # (the copies of walls already cut, named again from their mains)
        out = {w: json.load(gzip.open(os.path.join(OUT, f"{w}.json.gz"), "rt")) for w in sys.argv[2].split(",")}
        copies_as_main(out, extended())
        for w, faces in out.items():
            json.dump(faces, gzip.open(os.path.join(OUT, f"{w}.json.gz"), "wt"), separators=(",", ":"))
    else:
        main(sys.argv[1])
