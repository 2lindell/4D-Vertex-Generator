"""Shapes around a 2400-symmetry axis, each shown once, exploded (axis_view.json).

The axis: the great circle fixed by one of the extra half-turns (the 2400-element group's elements
outside the swirlprism group); here the one X38 runs along. The view is gnomonic at the point of the
axis nearest X38, the axis a straight line through it (the view's x).

Which shapes: layer by layer out from the axis. The first layer is X38, the region X68 on its sides,
X68's other wall X99, and the edges and corners bounding them (E1, C2a, B1; A1, C6); the next layer is
every shape the shape map (shape_map3d.json) puts round those edges and on those walls' sides.

Each shape is shown once, with only the extent every seed it holds needs once: of the copies of one
named shape, one is shown (the one touching most of the shapes already chosen, then the one nearest
the axis), and a shape the 2400-element group carries onto itself by an element other than the
identity is cut to a part no such element carries onto itself (X38, turned onto itself by the axis's
half-turn: its half on one side of the axis; C2a and B1, turned end for end by it: their halves).

The layout is an exploded drawing that keeps every shape's place against its neighbours: each shape
is moved from its place so that, for every shape A on the boundary of a shape B (a corner of an edge,
an edge of a wall or region, a wall of a region), B stands the same distance GAP from A in the
direction it lies from A there (an edge leaves its corner along itself; the walls and regions round
an edge stand in a ring round it, each in its own direction; a region stands off its wall towards its
volume). These conditions are met as nearly as they can be together (least squares), each shape also
held weakly to its true place, so the picture stays centred on the axis.

A region is drawn as its shell (the wall patches bounding it), a wall as its patches; each patch's
triangles are those of the wall pieces' meshes (domain_pieces, the coarse meshes) found between that
patch's two pieces of the domain.

    python -m four_d_vertex_generator.sphere_map.axis_view
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .frame import cell_vertices, data_dir, group, locate, point_from_beta, source_dir

AXIS_ELEMENT = 1200 + 368          # the half-turn whose axis X38 runs along (in the 2400-element group)
GAP = 0.15                         # each shape's distance from the shapes on its boundary, in the view
NEAR = 0.035                       # the neighbourhood of a boundary shape a direction is read in
HOLD = 0.05                        # how weakly each shape is held to its true place
MIN_STEP = np.radians(8)           # the least angle between neighbours round an edge
RING = 0.30                        # how far the walls and regions round an edge stand from it
MAX_ANGLE = 70                     # how far round the sphere from the view's centre a shape may reach
TOUCH = 6e-3                       # shapes this close (in the view) touch


# ---------------------------------------------------------------- geometry of the 3-sphere map

def _patch_triangles(D, doc, z):
    """Each wall patch's triangles (on R^4, in the half-cell's own copy): every wall piece's coarse
    triangles, each named by the patch between the two pieces of the domain on either side of its
    middle (for a piece in a face of the half-cell: the piece inside and the piece across the face,
    in the copy beyond it)."""
    from . import domain_pieces as dp
    from .chords import FACE_N, FACE_NAMES
    T, Ti, G = cell_vertices(), np.linalg.inv(cell_vertices()), group()
    vals = lambda X: np.sign(np.stack([sf.f(X) for sf in D.lib], 1)).astype(np.int8)
    face_patches = defaultdict(list)
    for p in D.patches:
        if p["face"] is not None:
            face_patches[p["face"]].append(p)
    out = defaultdict(list)
    for f in doc["faces"]:
        X = z["coarse_verts"][f["coarse_verts"][0]:f["coarse_verts"][1]]
        F = z["coarse_tris"][f["coarse_tris"][0]:f["coarse_tris"][1]]
        if not len(F):
            continue
        cen = X[F].mean(1)
        if f["sheet"].startswith("face"):
            name = f["sheet"][5:]
            k = FACE_NAMES.index(name)
            nin = FACE_N[k] - FACE_N[k].mean()
            nin = nin / np.linalg.norm(nin)
            cen = cen - (cen @ FACE_N[k])[:, None] * FACE_N[k] / (FACE_N[k] @ FACE_N[k])
            A = [D.keys.get(r.tobytes()) for r in vals(cen + 1e-7 * nin)]
            for t, a in enumerate(A):
                if a is None:
                    continue
                y = cen[t] - 1e-7 * nin
                kk, _ = locate(point_from_beta(y)[None], tol=1e-13)
                kk = int(kk[0])
                M = T @ G[kk] @ Ti
                M = M * np.sign((y @ M).sum())
                b = D.keys.get(vals((y @ M)[None])[0].tobytes())
                if b is None:
                    continue
                # (the patch with this piece inside and that one across, in the copy beyond)
                cands = [p for p in face_patches[name] if p["pieces"] == [a, b] and p["side_b_copy_shift"] == kk
                         or p["pieces"] == [b, a] and p["side_b_copy_shift"] == int(D.inv[kk])]
                if len(cands) == 1:
                    out[cands[0]["id"]].append(point_from_beta(X[F[t]]))
            continue
        s = D.lib[int(f["sheet"])]
        n = s.grad(cen)
        n = n - n.mean(1, keepdims=True)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        # (each middle moved onto the surface first: a flat triangle's middle is off a curved one)
        lifted = [dp._lift(s, y, nn) for y, nn in zip(cen, n)]
        cen = np.array([y if l_ is None else l_ for y, l_ in zip(cen, lifted)])
        key = int(f["sheet"])
        A = [D.keys.get(r.tobytes()) for r in vals(cen + 1e-7 * n)]
        B = [D.keys.get(r.tobytes()) for r in vals(cen - 1e-7 * n)]
        for t, (a, b) in enumerate(zip(A, B)):
            if a is None or b is None or a == b:
                continue
            p = D.pw.get((min(a, b), max(a, b), key)) or D.pw.get((min(a, b), max(a, b), None))
            if p is not None:
                out[p["id"]].append(point_from_beta(X[F[t]]))
    return {k: np.array(v) for k, v in out.items()}


def _same_curve(L, M, tol=1e-4):
    """Whether two sampled curves on R^4 are the same set (each sample within tol of the other's
    polyline: their samples need not fall at the same places or run the same way)."""
    def to_path(X, Y):
        a, b = Y[:-1], Y[1:]
        ab = b - a
        t = np.clip(np.einsum("nsk,sk->ns", X[:, None] - a[None], ab) / np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-30), 0, 1)
        return np.linalg.norm(X[:, None] - (a[None] + t[..., None] * ab[None]), axis=2).min(1)
    return max(to_path(L, M).max(), to_path(M, L).max()) < tol


class Map:
    """Placed shapes of the shape map, with their geometry on R^4."""

    def __init__(self):
        from . import domain_pieces as dp
        from .shape_map3d import multiplication_table
        root = data_dir() / "sphere_map"
        self.D = dp.Data()
        doc = json.loads((root / "domain_pieces.json").read_text())
        z = np.load(root / "domain_pieces.npz")
        self.tris = _patch_triangles(self.D, doc, z)
        self.G = group()
        self.M = multiplication_table()
        Q = np.load(source_dir() / "normalizer_q.npy")
        self.GQ = np.concatenate([self.G, self.G @ Q])
        self.sm = json.loads((root / "shape_map3d.json").read_text())
        self.atlas = json.loads((root / "atlas.json").read_text())
        self.patches = json.loads((root / "patches.json").read_text())["patches"]
        self.pieces = {p["id"]: p for p in json.loads((root / "pieces.json").read_text())["pieces"]}
        self.node = {k: {x["id"]: x for x in self.sm[k + "s"]} for k in ("wall", "line", "region", "point")}
        # (each piece's patches, as side a and as side b; the copy a patch lies in, given side b's)
        self.as_a, self.as_b = defaultdict(list), defaultdict(list)
        for p in self.patches:
            self.as_a[p["pieces"][0]].append(p)
            self.as_b[p["pieces"][1]].append(p)
        self.copy_of_b = {}

    def name(self, kind, node):
        return self.node[kind][node]["name"]

    def wall(self, node, at):
        """A wall node placed by at: its patches' triangles (n, 3, 4)."""
        out = []
        for c, h in self.node["wall"][node]["members"]:
            for k, p in self.atlas["faces"][c]["members"]:
                if p in self.tris:
                    g = self.G[at] @ self.G[h] @ self.G[k]
                    out.append(self.tris[p] @ g.T)
        return np.concatenate(out) if out else np.zeros((0, 3, 4))

    def region(self, c, at):
        """A region class placed by at: its shell's triangles (n, 3, 4), and its volume's middle."""
        S = {(int(self.M[at, k]), p) for k, p in self.atlas["regions"][c]["members"]}
        parts = set()
        for kk, piece in S:
            for p in self.as_a[piece]:
                if (int(self.M[kk, p["side_b_copy_shift"]]), p["pieces"][1]) not in S:
                    parts.add((kk, p["id"]))
            for p in self.as_b[piece]:
                s = p["side_b_copy_shift"]
                if (s, kk) not in self.copy_of_b:
                    self.copy_of_b[(s, kk)] = int(np.flatnonzero(self.M[:, s] == kk)[0])
                ka = self.copy_of_b[(s, kk)]
                if (ka, p["pieces"][0]) not in S:
                    parts.add((ka, p["id"]))
        T = [self.tris[p] @ self.G[k].T for k, p in sorted(parts) if p in self.tris]
        w = np.array([self.pieces[p]["volume_share"] for _, p in S])
        P = np.array([point_from_beta(np.array(self.pieces[p]["sample_beta"])) @ self.G[k].T for k, p in S])
        mid = (w[:, None] * P).sum(0)
        return np.concatenate(T), mid / np.linalg.norm(mid)

    def line(self, node, at):
        """A line node placed by at: its curve (n, 4), its pieces joined end to end."""
        segs = []
        for piece, h in self.node["line"][node]["members"]:
            g = self.G[at] @ self.G[h]
            for run in self.sm["line_geo"][piece]:
                L = np.array(run) @ g.T
                if not any(_same_curve(o, L) for o in segs):
                    segs.append(L)
        path = segs[0]
        for L in segs[1:]:                        # (joined where their ends meet)
            ends = [(np.linalg.norm(path[-1] - L[0]), 0), (np.linalg.norm(path[-1] - L[-1]), 1),
                    (np.linalg.norm(path[0] - L[-1]), 2), (np.linalg.norm(path[0] - L[0]), 3)]
            e = min(ends)[1]
            path = [np.vstack([path, L]), np.vstack([path, L[::-1]]), np.vstack([L, path]),
                    np.vstack([L[::-1], path])][e]
        return path

    def point(self, node, at):
        return (np.array(self.node["point"][node]["c"]) @ self.G[at].T)[None]


# ---------------------------------------------------------------- reduced extent

def _samples(sh, n=1500, rng=np.random.default_rng(0)):
    """Points spread over a shape (on R^4): over its triangles by area, along its curve, or itself."""
    if sh["kind"] in ("wall", "region"):
        T = sh["geo"]
        ar = np.linalg.norm(np.einsum("ni,nj->nij", T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]).reshape(len(T), -1), axis=1)
        i = rng.choice(len(T), n, p=ar / ar.sum())
        a, b = rng.random(n), rng.random(n)
        f = a + b > 1
        a[f], b[f] = 1 - a[f], 1 - b[f]
        X = T[i, 0] + a[:, None] * (T[i, 1] - T[i, 0]) + b[:, None] * (T[i, 2] - T[i, 0])
    elif sh["kind"] == "line":
        L = sh["geo"]
        ln = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
        t = np.sort(rng.random(n)) * ln[-1]
        X = np.stack([np.interp(t, ln, L[:, i]) for i in range(4)], 1)
    else:
        X = sh["geo"]
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _cut_tris(T, u):
    """Triangles cut to x . u >= 0."""
    out = []
    for tri in T:
        s = tri @ u
        if (s >= -1e-12).all():
            out.append(tri)
        elif (s > 0).any():
            pts = []
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                sa, sb = a @ u, b @ u
                if sa >= 0:
                    pts.append(a)
                if (sa >= 0) != (sb >= 0):
                    pts.append(a + sa / (sa - sb) * (b - a))
            for k in range(1, len(pts) - 1):
                out.append(np.array([pts[0], pts[k], pts[k + 1]]))
    return np.array(out).reshape(-1, 3, T.shape[-1])


def _cut_path(L, u):
    s = L @ u
    keep = []
    for i in range(len(L)):
        if s[i] >= 0:
            keep.append(L[i])
        if i + 1 < len(L) and (s[i] >= 0) != (s[i + 1] >= 0):
            keep.append(L[i] + s[i] / (s[i] - s[i + 1]) * (L[i + 1] - L[i]))
    return np.array(keep)


def reduce_extent(sh, GQ, prefer):
    """Cut a shape to a part no element of the 2400-element group carries onto itself: for each
    element (other than the identity) carrying the part onto itself and moving it, the part is cut by
    the hyperplane through that element's fixed 2-plane across the part's widest direction in its
    turned 2-plane, keeping the side towards `prefer`. Returns the elements that cut it."""
    if sh["kind"] == "point":
        return []
    cuts = []
    for _ in range(4):
        X = _samples(sh)
        tree = cKDTree(X)
        spacing = np.median(tree.query(_samples(sh, rng=np.random.default_rng(1)))[0])
        c = X.mean(0)
        rad = np.linalg.norm(X - c, axis=1).max()
        found = None
        order = [AXIS_ELEMENT] + [e for e in range(len(GQ)) if e != AXIS_ELEMENT]   # (the axis's own first)
        for e in order:
            g = GQ[e]
            if e == 0 or np.linalg.norm(c @ g.T - c) > 0.5 * rad + 1e-9:
                continue
            Y = X @ g.T
            if np.abs(Y - X).max() < 2e-4:
                continue                              # (holds it fixed point by point: the data are to 1e-5)
            d = tree.query(Y)[0]
            # (the data are to 1e-5: matches are allowed a little more than the samples' spacing)
            if np.median(d) < max(1.3 * spacing, 1e-4) and np.quantile(d, 0.95) < max(3 * spacing, 6e-4):
                found = (e, g)
                break
        if found is None:
            break
        e, g = found
        w, V = np.linalg.eigh((g + g.T) / 2)
        Pm = V[:, w < -0.5].T
        if len(Pm) != 2 or not np.allclose(g @ g, np.eye(4), atol=1e-8):
            sh.setdefault("warn", []).append(f"element {e} of order > 2 carries it onto itself")
            break
        u = Pm.T @ np.linalg.svd(X @ Pm.T, full_matrices=False)[2][0]
        u /= np.linalg.norm(u)
        if prefer @ u < 0:
            u = -u
        sh["geo"] = _cut_tris(sh["geo"], u) if sh["kind"] in ("wall", "region") else _cut_path(sh["geo"], u)
        cuts.append(e)
    return cuts


def _main_body(T, eps=0.03, keep=0.1):
    """A surface's triangles without its small detached bits: triangles cluster where their middles
    lie within eps of one another; clusters under `keep` of the area are dropped (a sheet's stray
    bits elsewhere in the cell, far from the shapes round here)."""
    if len(T) < 2:
        return T
    C = T.mean(1)
    par = np.arange(len(T))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i, j in cKDTree(C).query_pairs(eps):
        par[find(i)] = find(j)
    root = np.array([find(i) for i in range(len(T))])
    ar = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    tot = defaultdict(float)
    for r_, a_ in zip(root, ar):
        tot[r_] += a_
    big = {r_ for r_, a_ in tot.items() if a_ >= keep * ar.sum()}
    return T[np.isin(root, list(big))]


# ---------------------------------------------------------------- the view

def build() -> dict:
    m = Map()
    G, M, GQ = m.G, m.M, m.GQ
    H = GQ[AXIS_ELEMENT]
    w, V = np.linalg.eigh((H + H.T) / 2)
    A, P = V[:, w > 0.5].T, V[:, w < -0.5].T

    # the first layer, in the half-cell's frame: X38's sheet, the X68 on its first side, X99's sheet
    # bounding that X68, and their edges and corners (from X38's neighbours)
    x38 = next(x for x in m.sm["walls"] if x["name"] == "X38")
    x68 = m.atlas["faces"][x38["members"][0][0]]["sides"][0]["class"]
    x99 = next(x for x in m.sm["walls"] if x["name"] == "X99"
               and any(n["kind"] == "region" and n["node"] == x68 and n["at"] == 0 for n in x["nb"]))
    first = [("wall", x38["id"], 0), ("region", x68, 0), ("wall", x99["id"], 0)]
    first += [(n["kind"], n["node"], n["at"]) for n in x38["nb"] if n["kind"] in ("line", "point")]

    def nb_of(kind, node, at):
        """The shape map's neighbours of a placed shape, placed (their 'at' carried by its own)."""
        rec = m.node[kind][node]
        out = []
        for n in rec["nb"]:
            out.append((n["kind"], n["node"], int(M[at, n["at"]])))
        return out
    def rel(a):
        """A placed shape's neighbours in the shape map, placed: {key: relation}."""
        return {(k, nd, int(M[a["key"][2], at])): r for (k, nd, at, r) in
                ((n["kind"], n["node"], n["at"], n["rel"]) for n in m.node[a["kind"]][a["key"][1]]["nb"])}

    def geometry(kind, node, at):
        if kind == "wall":
            return m.wall(node, at), None
        if kind == "region":
            return m.region(node, at)
        if kind == "line":
            return m.line(node, at), None
        return m.point(node, at), None
    cache = {}

    def make(key):
        if key not in cache:
            geo, mid = geometry(*key)
            cache[key] = {"key": key, "kind": key[0], "id": m.name(key[0], key[1]), "geo": geo, "mid": mid}
        return cache[key]

    # the view: gnomonic at the axis's point nearest X38, the axis along x; y across the axis's turned
    # 2-plane in X38's widest direction there (the side of the axis whose halves are kept)
    X38 = make(first[0])["geo"].reshape(-1, 4)
    c = A.T @ (A @ X38.mean(0))
    c /= np.linalg.norm(c)
    e1 = A.T @ (A @ np.array([1.0, 0, 0, 0]))
    e1 = A[0] if np.linalg.norm(e1) < 1e-9 else e1
    e1 = e1 - (e1 @ c) * c
    e1 /= np.linalg.norm(e1)
    u = P.T @ np.linalg.svd(X38 @ P.T, full_matrices=False)[2][0]
    u /= np.linalg.norm(u)
    e3 = P.T @ (P @ np.ones(4))
    e3 = e3 - (e3 @ u) * u
    e3 = e3 if np.linalg.norm(e3) > 1e-9 else P[1]
    e3 /= np.linalg.norm(e3)
    chart = lambda X: np.stack([X @ e1, X @ u, X @ e3], -1) / (X @ c)[..., None]
    prefer = c + 0.3 * u                         # (the kept side: towards +u, at the view's centre)

    # one copy of each named shape, layer by layer out from the first: each next layer holds the shapes
    # the shape map puts beside the last layer's shapes whose names are not shown yet, each by the copy
    # it puts beside most of the shapes already shown (then the one nearest the axis); copies too far
    # round the sphere for the view (over MAX_ANGLE from its centre) are passed over
    rels = {}

    def rel_of(sh):
        if sh["key"] not in rels:
            rels[sh["key"]] = rel(sh)
        return rels[sh["key"]]

    def view_pts(sh):
        return chart(_samples(sh, 600))

    def in_view(sh):
        X = sh["geo"].reshape(-1, 4)
        return (X @ c).min() > np.cos(np.radians(MAX_ANGLE))
    chosen, layer_of, seen = {}, {}, defaultdict(set)
    for key in first:
        sh = make(key)
        seen[(sh["kind"], sh["id"])].add(key)
        if (sh["kind"], sh["id"]) not in chosen:
            chosen[(sh["kind"], sh["id"])] = sh
            layer_of[key] = 1
    frontier, layer = list(chosen.values()), 1
    while frontier:
        layer += 1
        by_name = defaultdict(list)
        for f in frontier:
            for key in nb_of(*f["key"]):
                nm = (key[0], m.name(key[0], key[1]))
                seen[nm].add(key)
                if nm not in chosen and key not in [x["key"] for x in by_name[nm]]:
                    by_name[nm].append(make(key))
        shown = list(chosen.values())
        new = []
        for nm, cands in by_name.items():
            cands = [x for x in cands if in_view(x)]
            if not cands:
                continue

            def score(sh):
                r_sh = rel_of(sh) if sh["kind"] != "point" else {}
                return (sum(sh["key"] in rel_of(f) or f["key"] in r_sh for f in shown if f["kind"] != "point"),
                        -np.linalg.norm(view_pts(sh)[:, 1:], axis=1).mean())
            best = max(cands, key=score) if len(cands) > 1 else cands[0]
            chosen[nm] = best
            layer_of[best["key"]] = layer
            new.append(best)
        frontier = new
    shapes = list(chosen.values())
    missing = sorted(f"{k} {n}" for k, n in {(k, x["name"]) for k in ("wall", "line", "region", "point")
                                               for x in m.sm[k + "s"]} if (k, n) not in chosen)

    # each cut to the extent its seeds need once
    for sh in shapes:
        sh["cuts"] = reduce_extent(sh, GQ, prefer)
        n_copies = len(seen[(sh["kind"], sh["id"])])
        sh["note"] = ("cut by the symmetries carrying it onto itself: " + ", ".join(
            "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in sh["cuts"])
            if sh["cuts"] else "whole: no symmetry carries it onto itself") + \
            (f"; one of {n_copies} copies met" if n_copies > 1 else "")

    # ---- the layout
    for sh in shapes:
        geo = sh["geo"]
        sh["v"] = chart(geo)
        if sh["kind"] in ("wall", "region"):
            sh["v"] = _main_body(sh["v"])
        X = sh["v"].reshape(-1, 3)
        sh["pts"] = (sh["v"].reshape(-1, 3) if sh["kind"] != "point" else X) if sh["kind"] in ("wall", "region", "point") else view_pts(sh)
    dim = {"point": 0, "line": 1, "wall": 2, "region": 3}
    idx = {id(sh): i for i, sh in enumerate(shapes)}
    # which shapes bound which (the shape map's own relations, for the copies shown): a corner ends an
    # edge, an edge has walls and regions round it, a wall has regions on its sides
    def bounds(a, b):
        if a["kind"] == "point":
            return b["kind"] == "line" and min(np.linalg.norm(b["v"][[0, -1]] - a["v"].reshape(3), axis=1)) < 2e-3
        if a["kind"] == "line":
            return b["kind"] in ("wall", "region") and rel(a).get(b["key"]) == "round"
        if a["kind"] == "wall":
            return b["kind"] == "region" and rel(a).get(b["key"]) == "side"
        return False
    rows = []                                   # (i, j, direction): shape j off shape i
    for a in shapes:
        ta = cKDTree(a["pts"])
        for b in shapes:
            if not bounds(a, b):
                continue
            d, j = ta.query(b["pts"])
            if a["kind"] == "wall":                  # (a region off its wall, towards its volume)
                Pa = a["pts"]
                n = np.linalg.svd(Pa - Pa.mean(0), full_matrices=False)[2][2]
                mid = chart(b["mid"]) if b["mid"] is not None else b["pts"].mean(0)
                v = n * np.sign(n @ (mid - Pa.mean(0)))
            elif a["kind"] == "point":               # (an edge leaves its corner along itself)
                v = (b["pts"][:len(b["pts"]) // 10 + 1].mean(0) if np.linalg.norm(b["v"][0] - a["v"].reshape(3))
                     < np.linalg.norm(b["v"][-1] - a["v"].reshape(3)) else b["pts"][-(len(b["pts"]) // 10 + 1):].mean(0)) \
                    - a["v"].reshape(3)
            else:                                    # (off an edge: across it, where b lies near it,
                L = a["pts"]                         # away from its ends, where every shape meets)
                near = (d < NEAR) & (j > 0.1 * len(L)) & (j < 0.9 * len(L))
                if not near.any():
                    near = d < NEAR
                Bn, jj = b["pts"][near], j[near]
                tng = L[np.minimum(jj + 3, len(L) - 1)] - L[np.maximum(jj - 3, 0)]
                tng /= np.maximum(np.linalg.norm(tng, axis=1, keepdims=True), 1e-12)
                off = Bn - L[jj]
                off -= np.einsum("nk,nk->n", off, tng)[:, None] * tng
                off /= np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)
                v = off.mean(0)
            if np.linalg.norm(v) < 1e-9:
                continue
            rows.append((idx[id(a)], idx[id(b)], v / np.linalg.norm(v)))   # (unit; a circle may widen it)
    # round each edge: its walls and regions keep the directions they truly lie in from it, in their
    # order round it and over the arc they truly span; only where they crowd are they opened out, to at
    # least MIN_STEP apart (about the arc's middle), and they stand RING out from it
    arcs = {}
    for i, a in enumerate(shapes):
        if a["kind"] != "line":
            continue
        ring = [r for r, (ii, j, v) in enumerate(rows) if ii == i]
        if len(ring) < 3:
            continue
        L = a["pts"]
        t = L[-1] - L[0]
        t /= np.linalg.norm(t)
        vs = np.array([rows[r][2] - (rows[r][2] @ t) * t for r in ring])
        b1 = vs[0] / np.linalg.norm(vs[0])
        b2 = np.cross(t, b1)
        ang = np.arctan2(vs @ b2, vs @ b1)
        order = np.argsort(ang)
        sa = ang[order]
        # (unwrapped from the widest empty gap round the edge: the arc they truly span)
        gaps = np.diff(np.r_[sa, sa[0] + 2 * np.pi])
        start = (int(np.argmax(gaps)) + 1) % len(sa)
        sa = np.r_[sa[start:], sa[:start] + 2 * np.pi]
        order = np.r_[order[start:], order[:start]]
        new = sa.copy()
        for _ in range(500):                      # (opened out where closer than MIN_STEP)
            moved = False
            for k in range(len(new) - 1):
                d = new[k + 1] - new[k]
                if d < MIN_STEP - 1e-9:
                    push = (MIN_STEP - d) / 2
                    new[k] -= push
                    new[k + 1] += push
                    moved = True
            if not moved:
                break
        new += sa.mean() - new.mean()             # (about the arc's middle)
        arcs[a["id"]] = {"true_span_deg": round(float(np.degrees(sa[-1] - sa[0])), 1),
                         "shown_span_deg": round(float(np.degrees(new[-1] - new[0])), 1)}
        for th, o in zip(new, order):
            rows[ring[o]] = (rows[ring[o]][0], rows[ring[o]][1], RING / GAP * (np.cos(th) * b1 + np.sin(th) * b2))
    n = len(shapes)
    Am = np.zeros((len(rows) + n, n))
    bm = np.zeros((len(rows) + n, 3))
    for r, (i, j, v) in enumerate(rows):
        Am[r, j], Am[r, i], bm[r] = 1, -1, GAP * v
    for i in range(n):
        Am[len(rows) + i, i] = HOLD
    shift = np.linalg.lstsq(Am, bm, rcond=None)[0]
    resid = [float(np.linalg.norm(shift[j] - shift[i] - GAP * v)) / float(np.linalg.norm(GAP * v)) for i, j, v in rows]

    out = []
    for sh, s in zip(shapes, shift):
        V3 = sh["v"] + s
        rec = {"id": sh["id"], "kind": sh["kind"], "note": sh["note"],
               "layer": layer_of[sh["key"]]}
        if sh["kind"] in ("wall", "region"):
            Vu, F3 = np.unique(V3.reshape(-1, 3).round(6), axis=0, return_inverse=True)
            rec.update(verts=Vu.round(5).tolist(), faces=F3.reshape(-1, 3).tolist())
        elif sh["kind"] == "line":
            rec["path"] = V3.round(5).tolist()
        else:
            rec["at"] = V3.reshape(3).round(5).tolist()
        if sh.get("warn"):
            rec["warn"] = sh["warn"]
        out.append(rec)
    L = np.linspace(-0.35, 0.35, 2)
    return {"_about": __doc__, "axis": [[float(t), 0.0, 0.0] for t in L], "shapes": out,
            "links": [[i, j] for i, j, _ in rows], "missing": missing,
            "fit": {"arcs": arcs, "gap": GAP, "relative_residual_median": float(np.median(resid)), "relative_residual_max": float(np.max(resid)),
                    "achieved": [[i, j, round(float(np.linalg.norm(shift[j] - shift[i])), 4)] for i, j, _ in rows]}}


def run() -> str:
    path = data_dir() / "sphere_map" / "axis_view.json"
    data = build()
    path.write_text(json.dumps(data, separators=(",", ":")))
    for s in data["shapes"]:
        size = f"{len(s['faces'])} triangles" if "faces" in s else f"{len(s['path'])} steps" if "path" in s else "point"
        print(f"{s['id']} ({s['kind']}, layer {s['layer']}): {size}; {s['note']}{'; ' + str(s['warn']) if s.get('warn') else ''}")
    print(f"{len(data['links'])} links; not shown: {data['missing']}")
    return str(path)


if __name__ == "__main__":
    print(run())
