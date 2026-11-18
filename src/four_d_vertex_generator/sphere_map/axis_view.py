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
EXPLODE = 1.5                      # the corners (and any shape with nothing shown on its boundary) spread
                                   # as the picture scaled by this about the view's centre
MIN_STEP = np.radians(8)           # the least angle between neighbours round an edge
MAX_STEP = np.radians(30)          # ... which may widen to this round an edge with few shapes
RING_ARC = 300                     # ... so that they spread over about this many degrees
JOIN_EDGES = ("E1", "X13")         # edges joined to B1's group, lying along the shapes they bound
EDGE_ON_WALL = {"B1": ("X38", "A1"), "C2a": ("X96", "C6"), "D2": ("F4", "A1")}   # edges along a wall, in line with a corner
END_CORNERS = {"C4": ("D2", "X4")}  # corners at the far ends of edges (in line with both, if two)
FAN_ON_EDGE = {"X13": ("X43", "X31", "X54")}     # layer 3: a small even fan round an edge
FAN3_GAP = 0.015                   # ... the gap between neighbours in it
FAN3_CLEAR = 0.03                  # ... measured away from the edge, where they all meet
RING_ON_EDGE = {"X4": ("X17",)}    # layer 3: walls lined up on an edge
LINE_AT_CORNER = {"X2": ("X4", "C4", "X43"), "X80": ("X13", "A1", "X43")}   # layer 3: edges in line with a corner
MIN_CORNER_GAP = 0.025             # the least gap between an edge's end and its corner
LAYER3_OUT = 0.04                  # ... and this much further, away from the group shapes they touch
RING3_GAP = 0.02                   # ... those shapes set this far out from their edge
JOIN_CORNERS = ("A1", "C6")        # corners set at the ends of the group's edges (in this order)
EDGE_GAP = 0.02                    # ... the edges set this far out from those shapes
CORNER_GAP = 0.012                 # ... and the corners this far beyond the edges' ends
STACK_ON = {"X49": ("X28", "X50", "X33")}   # shapes stacked on a region, one beyond the other
STACK_OFF = 0.02                   # ... each this far beyond the one before
ALIGN_ON = {"X4": ("X82", ("X22",))}  # along an edge: walls lined up on the first (moved and turned as it is)
ALIGN_OFF = 0.008                  # ... each set this far off its own face, away from the first
EDGE_OFF = 0.016                   # ... and the edge this far out from the first
CUT_ALONG = {"F3": ("T2", "F1")}   # cut shapes cut flat along where these two walls meet (see build)
COMPLETE = {"F3": ("X4",)}         # regions whose shown part gets its own walls (and these edges) beside it
COMPLETE_OFF = 0.03                # ... and a wall completing such a region's shell, a little more
WALL_OFF = 0.01                    # how far a wall attached to a region of such a group stands off it
HELD_RING = 0.06                   # how far the group round an edge with a turn of order 3 or more stands off it
RING = 0.20                        # how far the walls and regions round an edge stand from it
DEBUG_PICK = ()                    # (names whose copy picks are printed)
PATCH_NAMING = 5                   # (bumped when the way triangles are named changes: the cache is redone)
GRID = 1e-5                        # the page's meshes are thinned to this (in the view's units)
FILL = False                       # whether unnamed triangles take their neighbours' patch
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
    from collections import Counter
    faces_of = defaultdict(Counter)

    def fill(F, lab):
        """Triangles left unnamed take the name most of the triangles sharing a corner with them have
        (only where a sheet's triangles are coarse: with the fine meshes, none are filled)."""
        if not FILL:
            return lab
        by_v = defaultdict(list)
        for t, tri in enumerate(F):
            for v_ in tri:
                by_v[int(v_)].append(t)
        for _ in range(20):
            todo = [t for t in range(len(F)) if lab[t] is None]
            if not todo:
                break
            got = {}
            for t in todo:
                near = [lab[u] for v_ in F[t] for u in by_v[int(v_)] if lab[u]]
                if near:
                    got[t] = max(set(near), key=near.count)
            if not got:
                break
            lab.update(got)
        return lab
    for f in doc["faces"]:
        # (the fine meshes: a coarse triangle can be larger than a small patch, and is named by the
        # patch at its middle only)
        X = z["verts"][f["verts"][0]:f["verts"][1]]
        F = z["tris"][f["tris"][0]:f["tris"][1]]
        if not len(F):
            continue
        lab = {t: None for t in range(len(F))}         # (each triangle's patch, and the copy it is carried from)
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
                # (a patch has its side a inside its own copy: with this piece inside it is as it lies;
                # with the piece across inside, it lies in the copy beyond and is carried back from there)
                cands = [(p, -1) for p in face_patches[name] if p["pieces"] == [a, b] and p["side_b_copy_shift"] == kk] + \
                        [(p, kk) for p in face_patches[name] if p["pieces"] == [b, a] and p["side_b_copy_shift"] == int(D.inv[kk])]
                if len(cands) == 1:
                    lab[t] = (cands[0][0]["id"], cands[0][1])
                    continue
                # (or a patch kept on the face glued to this one, seen from the copy beyond: the piece
                # across is its inside; carried back into that patch's own copy)
                cands = [p for fn, ps in face_patches.items() if fn != name for p in ps
                         if p["pieces"] == [b, a] and p["side_b_copy_shift"] == int(D.inv[kk])]
                if len(cands) == 1:
                    lab[t] = (cands[0]["id"], kk)
            for t, l_ in fill(F, lab).items():
                if l_:
                    Y = point_from_beta(X[F[t]])
                    out[l_[0]].append(Y if l_[1] < 0 else Y @ G[l_[1]])
                    faces_of[l_[0]][(f["id"], l_[1])] += 1
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
            if a is not None and a == b:
                lab[t] = False                    # (the same piece on both sides: no wall here)
            if a is None or b is None or a == b:
                continue
            p = D.pw.get((min(a, b), max(a, b), key)) or D.pw.get((min(a, b), max(a, b), None))
            if p is not None:
                lab[t] = (p["id"], -1)
        for t, l_ in fill(F, lab).items():
            if l_:
                out[l_[0]].append(point_from_beta(X[F[t]]))
                faces_of[l_[0]][(f["id"], -1)] += 1
    # (and each patch's wall piece: the piece most of its triangles lie in, and the copy it is carried
    # from, -1 for none)
    return {k: np.array(v) for k, v in out.items()}, \
        {k: max(c.items(), key=lambda kv: kv[1])[0] for k, c in faces_of.items()}


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
        # (cached: naming every fine triangle takes a while)
        import pickle
        cache = root.parent.parent.parent / ".cache" / "axis_patch_tris.pkl"
        stamp = (round((root / "domain_pieces.npz").stat().st_mtime), round((root / "patches.json").stat().st_mtime),
                 PATCH_NAMING)
        got = None
        if cache.exists():
            try:
                got = pickle.loads(cache.read_bytes())
            except Exception:
                got = None
        if got and got[0] == stamp:
            self.tris, self.face_of = got[1]
        else:
            self.tris, self.face_of = _patch_triangles(self.D, doc, z)
            cache.parent.mkdir(exist_ok=True)
            cache.write_bytes(pickle.dumps((stamp, (self.tris, self.face_of))))
        # each wall piece's own (coarse) mesh, whole: shapes are drawn from whole pieces, whose edges
        # follow the curves where walls meet, not from triangles sorted one by one into patches
        self.piece_mesh = {}
        for f in doc["faces"]:
            X = z["coarse_verts"][f["coarse_verts"][0]:f["coarse_verts"][1]]
            F = z["coarse_tris"][f["coarse_tris"][0]:f["coarse_tris"][1]]
            if len(F):
                self.piece_mesh[f["id"]] = point_from_beta(X)[F]
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

    def from_pieces(self, parts):
        """The triangles of the wall pieces holding these patches (copy, patch), each piece whole and
        once, in its place."""
        seen, out, centres = set(), [], []
        for k, p in parts:
            if p not in self.face_of:
                continue
            fid, t = self.face_of[p]
            g = k if t < 0 else int(self.M[k, self.D.inv[t]])     # (carried from copy t, then to k)
            if (fid, g) not in seen and fid in self.piece_mesh:
                seen.add((fid, g))
                T = self.piece_mesh[fid] @ self.G[g].T
                # (a piece in a face of the half-cell is met from both faces glued there, and the two
                # land on one another: drawn once, or the two flicker through each other)
                c_ = T.reshape(-1, 4).mean(0)
                if any(np.linalg.norm(c_ - c2) < 3e-3 for c2 in centres):
                    continue
                centres.append(c_)
                out.append(T)
        return np.concatenate(out) if out else np.zeros((0, 3, 4))

    def wall(self, node, at):
        """A wall node placed by at: its wall pieces' triangles (n, 3, 4)."""
        parts = []
        for c, h in self.node["wall"][node]["members"]:
            for k, p in self.atlas["faces"][c]["members"]:
                parts.append((int(self.M[self.M[at, h], k]), p))
        return self.from_pieces(parts)

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
        T = [self.from_pieces(sorted(parts))]
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

def _samples(sh, n=1500, rng=None):
    """Points spread over a shape (on R^4): over its triangles by area, along its curve, or itself
    (the same points each time, unless another generator is given)."""
    rng = np.random.default_rng(0) if rng is None else rng
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


def _cut_field(T, f):
    """Triangles (n, 3, k) cut to f >= 0, f given at each corner (n, 3): the cut runs where f, taken
    as linear along each side, is 0."""
    out = []
    for tri, ft in zip(T, f):
        if (ft >= 0).all():
            out.append(tri)
        elif (ft > 0).any():
            pts = []
            for i in range(3):
                a, b, fa, fb = tri[i], tri[(i + 1) % 3], ft[i], ft[(i + 1) % 3]
                if fa >= 0:
                    pts.append(a)
                if (fa >= 0) != (fb >= 0):
                    pts.append(a + fa / (fa - fb) * (b - a))
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
        cut = (lambda v: _cut_tris(sh["geo"], v)) if sh["kind"] in ("wall", "region") else (lambda v: _cut_path(sh["geo"], v))
        if callable(prefer):
            # (the half scoring higher: the one touching more of the shapes shown beside it)
            halves = [cut(u), cut(-u)]
            score = [prefer({"kind": sh["kind"], "geo": h_}) if len(h_) else -1 for h_ in halves]
            sh["geo"] = halves[int(score[1] > score[0])]
        else:
            sh["geo"] = cut(u if prefer @ u >= 0 else -u)
        cuts.append(e)
    return cuts


def ib_edge(shapes, name):
    """The index of the line shown with this name, or None."""
    return next((i for i, x in enumerate(shapes) if x["kind"] == "line" and x["id"] == name), None)


def _turn(V, P0, t, ang):
    """Points turned by ang about the line through P0 along the unit vector t."""
    X = V.reshape(-1, 3) - P0
    c_, s_ = np.cos(ang), np.sin(ang)
    X = X * c_ + np.cross(t, X) * s_ + np.outer(X @ t, t) * (1 - c_)
    return (X + P0).reshape(V.shape)


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
        # (picked one at a time, each against every shape shown so far, this layer's too: the shape with
        # the copy beside most of them goes first, so its neighbours can then take the copies beside it)
        shown = [x for x in chosen.values() if x["kind"] != "point"]
        cands_of = {nm: [x for x in cs if in_view(x)] for nm, cs in by_name.items()}
        cands_of = {nm: cs for nm, cs in cands_of.items() if cs}
        far = {}

        def near_axis(sh):
            if sh["key"] not in far:
                far[sh["key"]] = -np.linalg.norm(view_pts(sh)[:, 1:], axis=1).mean()
            return far[sh["key"]]

        def score(sh):
            # (first: shown shapes it bounds or lies on, round an edge they both are round, so the
            # shapes round an edge follow on one from the next; then: shown shapes beside it at all;
            # then: nearness to the axis)
            r_sh = rel_of(sh) if sh["kind"] != "point" else {}
            beside = [f for f in shown if sh["key"] in rel_of(f) or f["key"] in r_sh]
            lines = [f for f in shown if f["kind"] == "line" and sh["key"] in rel_of(f)]
            follow = sum(1 for f in beside if f["kind"] in ("wall", "region") and f["kind"] != sh["kind"]
                         and any(f["key"] in rel_of(L) for L in lines))
            return (follow, len(beside), near_axis(sh))
        new = []
        while cands_of:
            best_nm, best = max(((nm, max(cs, key=score)) for nm, cs in cands_of.items()),
                                key=lambda t: score(t[1]))
            chosen[best_nm] = best
            layer_of[best["key"]] = layer
            if DEBUG_PICK and best_nm[1] in DEBUG_PICK:
                for x in cands_of[best_nm]:
                    r_sh = rel_of(x)
                    print("pick", best_nm, x["key"], score(x)[:2], [f["id"] for f in shown if x["key"] in rel_of(f) or f["key"] in r_sh],
                          "<-" if x is best else "")
            new.append(best)
            if best["kind"] != "point":
                shown.append(best)
            del cands_of[best_nm]
        frontier = new
    shapes = list(chosen.values())
    missing = sorted(f"{k} {n}" for k, n in {(k, x["name"]) for k in ("wall", "line", "region", "point")
                                               for x in m.sm[k + "s"]} if (k, n) not in chosen)

    # each cut to the extent its seeds need once (in the order chosen: each against those cut before)
    done = []
    for sh in shapes:
        # (of the two halves, the one touching more of the shapes the shape map puts beside it that
        # are shown and already cut (walls and regions count most, then edges, then the corner A1
        # shown), so the halves kept meet one another: X38, B1 and F3 meet at that A1, F3 at F1)
        a1 = chosen.get(("point", "A1"))
        c6 = chosen.get(("point", "C6"))
        near = [f for f in done if f["kind"] != "point" and (f["key"] in rel_of(sh) or sh["key"] in rel_of(f))]

        def touching(half, near=near):
            P = _samples(half, 2000)
            tr = cKDTree(P)
            sc = sum((2 if f["kind"] in ("wall", "region") else 1) * (tr.query(_samples(f, 800))[0].min() < 3e-3)
                     for f in near)
            Vh = np.asarray(half["geo"]).reshape(-1, 4)          # (measured to the half's own corners,
            if a1 is not None:                                   # as a corner is one point)
                sc += 100 * (np.linalg.norm(Vh - a1["geo"][0], axis=1).min() < 3e-3)   # (the A1 shown first,
            if c6 is not None:                                                         # then the C6 shown)
                sc += 50 * (np.linalg.norm(Vh - c6["geo"][0], axis=1).min() < 3e-3)
            return sc + 1e-3 * float(np.sign((P.mean(0) @ (prefer - c))))   # (then: the side of the axis kept)
        sh["cuts"] = reduce_extent(sh, GQ, touching)
        done.append(sh)
        n_copies = len(seen[(sh["kind"], sh["id"])])
        sh["note"] = ("cut by the symmetries carrying it onto itself: " + ", ".join(
            "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in sh["cuts"])
            if sh["cuts"] else "whole: no symmetry carries it onto itself") + \
            (f"; one of {n_copies} copies met" if n_copies > 1 else "")

    # regions whose shell is completed: each wall bounding the part of it shown, and the edges named,
    # are shown by the copy that bounds that part (F3: its walls T2 and X22 below F1, and its edge X4)
    completed = {}
    for reg, extra in COMPLETE.items():
        R = chosen.get(("region", reg))
        if R is None:
            continue
        tr = cKDTree(_samples(R, 3000))
        rr = rel_of(R)
        want = {(k[0], m.name(k[0], k[1])) for k in rr if k[0] == "wall"} | {("line", nm) for nm in extra}
        for kind, nm in sorted(want):
            cands = [k for k in rr if k[0] == kind and m.name(kind, k[1]) == nm]
            if not cands:
                continue

            def share(key):                       # (how much of the region's part it lies along)
                return (tr.query(_samples(make(key), 2000))[0] < 4e-3).mean()
            best = max(cands, key=share)
            old = chosen.get((kind, nm))
            # (a copy shown that already lies along the region's part stays: F1, beside X49 round B1)
            if old is not None and (old["key"] == best or old["key"] in cands and share(old["key"]) > 0):
                if kind == "line":
                    completed[nm] = reg
                continue
            new = dict(make(best))
            new["geo"] = np.array(new["geo"])
            new["cuts"] = reduce_extent(new, GQ, R["geo"].reshape(-1, 4).mean(0))
            new["note"] = (("cut by the symmetries carrying it onto itself: " + ", ".join(
                "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in new["cuts"]))
                if new["cuts"] else "whole: no symmetry carries it onto itself") + \
                f"; the copy bounding the {reg} shown"
            layer_of[best] = layer_of[old["key"]] if old is not None else layer_of[R["key"]]
            if old is not None:
                shapes[[id(x) for x in shapes].index(id(old))] = new
            else:
                shapes.append(new)
            chosen[(kind, nm)] = new
            completed[nm] = reg

    # shapes stacked on a region, one beyond the other (X49, then X28, X50, X33): each shown by the copy
    # the shape map puts beside the one before it (the one lying along it most, if more than one)
    stacked = {}
    for base, chain in STACK_ON.items():
        prev = next((x for x in chosen.values() if x["kind"] != "point" and x["id"] == base), None)
        for nm in chain:
            if prev is None:
                break
            rp = rel_of(prev)
            cands = [k for k in rp if k[0] in ("wall", "region") and m.name(k[0], k[1]) == nm]
            if not cands:
                break
            tr = cKDTree(_samples(prev, 3000))
            best = max(cands, key=lambda k_: (tr.query(_samples(make(k_), 1500))[0] < 4e-3).mean())
            kind = best[0]
            old = chosen.get((kind, nm))
            if old is None or old["key"] != best:
                new = dict(make(best))
                new["geo"] = np.array(new["geo"])
                new["cuts"] = reduce_extent(new, GQ, prev["geo"].reshape(-1, 4).mean(0))
                new["note"] = (("cut by the symmetries carrying it onto itself: " + ", ".join(
                    "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in new["cuts"]))
                    if new["cuts"] else "whole: no symmetry carries it onto itself") + f"; the copy beside the {prev['id']} shown"
                layer_of[best] = layer_of[old["key"]] if old is not None else layer_of[prev["key"]]
                if old is not None:
                    shapes[[id(x) for x in shapes].index(id(old))] = new
                else:
                    shapes.append(new)
                chosen[(kind, nm)] = new
                prev = new
            else:
                prev = old
            stacked[nm] = base

    # shapes set round an edge of the group (layer 3: round X13, X43, X54, X31; round X4, X17): each
    # shown by the copy the shape map puts round the edge shown
    for e_name, names in list(FAN_ON_EDGE.items()) + list(RING_ON_EDGE.items()):
        E = chosen.get(("line", e_name))
        if E is None:
            continue
        re_ = rel_of(E)
        for nm in names:
            cands = [k for k in re_ if k[0] in ("wall", "region") and m.name(k[0], k[1]) == nm]
            if not cands:
                continue
            kind = cands[0][0]
            old = chosen.get((kind, nm))
            if old is not None and old["key"] in cands:
                continue
            best = cands[0]
            new = dict(make(best))
            new["geo"] = np.array(new["geo"])
            new["cuts"] = reduce_extent(new, GQ, E["geo"].reshape(-1, 4).mean(0))
            new["note"] = (("cut by the symmetries carrying it onto itself: " + ", ".join(
                "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in new["cuts"]))
                if new["cuts"] else "whole: no symmetry carries it onto itself") + f"; the copy round the {e_name} shown"
            layer_of[best] = layer_of[old["key"]] if old is not None else layer_of[E["key"]]
            if old is not None:
                shapes[[id(x) for x in shapes].index(id(old))] = new
            else:
                shapes.append(new)
            chosen[(kind, nm)] = new

    # edges of layer 3 shown by the copy beside the shape named (X2 and X80 beside X43), through the
    # corner shown
    for l_name, (_by, c_name, beside) in LINE_AT_CORNER.items():
        B = next((x for x in chosen.values() if x["kind"] != "point" and x["id"] == beside), None)
        C = chosen.get(("point", c_name))
        if B is None or C is None:
            continue
        cands = [k for k in rel_of(B) if k[0] == "line" and m.name("line", k[1]) == l_name]
        cands = [k for k in cands if np.linalg.norm(make(k)["geo"] - C["geo"][0], axis=1).min() < 1e-4] or cands
        if not cands:
            continue
        old = chosen.get(("line", l_name))
        if old is not None and old["key"] in cands:
            continue
        new = dict(make(cands[0]))
        new["geo"] = np.array(new["geo"])
        new["cuts"] = reduce_extent(new, GQ, C["geo"][0])
        new["note"] = (("cut by the symmetries carrying it onto itself: " + ", ".join(
            "the axis's half-turn" if e == AXIS_ELEMENT else f"element {e}" for e in new["cuts"]))
            if new["cuts"] else "whole: no symmetry carries it onto itself") + f"; the copy beside the {beside} shown"
        layer_of[cands[0]] = layer_of[old["key"]] if old is not None else layer_of[B["key"]]
        if old is not None:
            shapes[[id(x) for x in shapes].index(id(old))] = new
        else:
            shapes.append(new)
        chosen[("line", l_name)] = new

    # a cut shape cut flat so its front edge lines up with the edge where two of its walls meet (F3:
    # with the top of T2, where it meets F1): a flat cut through its half-turn's fixed 2-plane is valid
    # at any angle round it (the half-turn carries it onto itself, so the part kept and its image make
    # up the shape once); the angle is set by that edge's direction across the turned 2-plane, and the
    # side kept is the one holding most of the first wall
    for nm, (w1, w2) in CUT_ALONG.items():
        R = chosen.get(("region", nm))
        W1, W2 = chosen.get(("wall", w1)), chosen.get(("wall", w2))
        if R is None or W1 is None or W2 is None or len(R.get("cuts", [])) != 1:
            continue
        g = GQ[R["cuts"][0]]
        w, V = np.linalg.eigh((g + g.T) / 2)
        Pm = V[:, w < -0.5].T
        P1 = _samples(W1, 8000)
        edge = P1[cKDTree(_samples(W2, 8000)).query(P1)[0] < 3e-3]
        if len(edge) < 10:
            continue
        along = np.linalg.svd(edge @ Pm.T, full_matrices=False)[2][0]
        u_cut = Pm.T @ along                               # (not u: the view's own y direction)
        u_cut /= np.linalg.norm(u_cut)
        if (P1 @ u_cut).mean() < 0:
            u_cut = -u_cut
        whole = m.region(*R["key"][1:])[0]
        R["geo"] = _cut_tris(whole, u_cut)
        R["note"] = R["note"].replace(f"element {R['cuts'][0]}",
                                      f"element {R['cuts'][0]} (cut flat along the top of {w1}, where it meets {w2})")

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
                if not near.any():                   # (or its nearest points, wherever they are)
                    near = d <= np.quantile(d, 0.05)
                Bn, jj = b["pts"][near], j[near]
                tng = L[np.minimum(jj + 3, len(L) - 1)] - L[np.maximum(jj - 3, 0)]
                tng /= np.maximum(np.linalg.norm(tng, axis=1, keepdims=True), 1e-12)
                off = Bn - L[jj]
                off -= np.einsum("nk,nk->n", off, tng)[:, None] * tng
                off /= np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)
                v = off.mean(0)
            if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
                continue
            rows.append((idx[id(a)], idx[id(b)], v / np.linalg.norm(v)))   # (unit; a circle may widen it)
    # round each edge: its walls and regions keep the directions they truly lie in from it, in their
    # order round it and over the arc they truly span; only where they crowd are they opened out, to at
    # least MIN_STEP apart (about the arc's middle), and they stand RING out from it
    arcs = {}
    turns = defaultdict(list)
    sector_of = {}
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
        # (the least step: wider round an edge with few shapes, up to MAX_STEP, so they spread over
        # about RING_ARC; never under MIN_STEP)
        step = float(np.clip(np.radians(RING_ARC) / len(sa), MIN_STEP, MAX_STEP))
        # (round an edge with a turn of order 3 or more about it, the shapes shown lie in one of the
        # sectors the turn repeats, and keep their true directions there: close together, as they lie)
        held_ring = m.node["line"][a["key"][1]]["stab"] >= 3
        if held_ring:
            step = 0.0
        new = sa.copy()
        for _ in range(500):                      # (opened out where closer than the step)
            moved = False
            for k in range(len(new) - 1):
                d = new[k + 1] - new[k]
                if d < step - 1e-9:
                    push = (step - d) / 2
                    new[k] -= push
                    new[k + 1] += push
                    moved = True
            if not moved:
                break
        # (held so the shapes of the earliest layer round it keep their true directions on average:
        # the others swing out from them)
        lay = np.array([layer_of[shapes[rows[ring[o]][1]]["key"]] for o in order])
        held = lay == lay.min()
        new += (sa[held] - new[held]).mean()
        if held_ring:
            # (round such an edge the shapes shown fill one sector of the turn: spread evenly over it in
            # the order they lie, the shapes of the earliest layer keeping their true directions, each
            # turned to its place there so it stands straight out from the edge)
            sector = 2 * np.pi / m.node["line"][a["key"][1]]["stab"]
            new = (sector / len(sa)) * np.arange(len(sa))
            new += (sa[held] - new[held]).mean()
        arcs[a["id"]] = {"true_span_deg": round(float(np.degrees(sa[-1] - sa[0])), 1),
                         "shown_span_deg": round(float(np.degrees(new[-1] - new[0])), 1),
                         "order": [[shapes[rows[ring[o]][1]]["id"], round(float(np.degrees(x)), 1), round(float(np.degrees(y)), 1)]
                                   for o, x, y in zip(order, sa, new)]}
        for th, o, th0 in zip(new, order, sa):
            rows[ring[o]] = (rows[ring[o]][0], rows[ring[o]][1], RING / GAP * (np.cos(th) * b1 + np.sin(th) * b2))
            if held_ring:                         # (placed off this edge alone, as one group round it)
                sector_of[rows[ring[o]][1]] = (i, HELD_RING * (np.cos(th) * b1 + np.sin(th) * b2))
            # (and the turn about the edge that would swing it from its true direction to this one)
            j_ = rows[ring[o]][1]
            if layer_of[a["key"]] <= layer_of[shapes[j_]["key"]]:      # (only edges of its layer or nearer)
                turns[j_].append((np.inf if held_ring else abs(th - th0), th - th0, L.mean(0), t, a["id"]))
    # the moves, from the lowest shapes up, each from the shapes on its boundary only (no shape's move
    # feeds back into those below it, and no chain is longer than corner, edge, wall, region):
    # a shape with nothing shown on its boundary (the corners, and any other) moves as the whole
    # picture scaled by EXPLODE about the view's centre would carry its middle, so corners spread out
    # beyond their edges' ends; any other shape takes the mean, over the shapes on its boundary, of
    # that shape's move plus its gap from it in the direction it stands off it
    n = len(shapes)
    shift = np.zeros((n, 3))
    parents = defaultdict(list)
    for i, j, v in rows:
        parents[j].append((i, v))
    for i in sorted(range(n), key=lambda i: dim[shapes[i]["kind"]]):
        # (only from boundary shapes of its own layer or nearer the axis: a shape is placed among
        # the shapes it was found beside, not pulled away by ones found from it later)
        own = [(a_, v) for a_, v in parents[i] if layer_of[shapes[a_]["key"]] <= layer_of[shapes[i]["key"]]]
        if i in sector_of:
            e_, off = sector_of[i]
            shift[i] = shift[e_] + off
        elif own:
            shift[i] = np.mean([shift[a_] + GAP * v for a_, v in own], 0)
        else:
            shift[i] = (EXPLODE - 1) * shapes[i]["pts"].mean(0)
    resid = [float(np.linalg.norm(shift[j] - shift[i] - GAP * v)) / float(np.linalg.norm(GAP * v)) for i, j, v in rows]

    # each shape's turn: about the edge round which it is opened out most, so the shapes round an edge
    # fan out from it rather than standing parallel, side by side
    turn_of = {}
    for i in range(n):
        if turns.get(i):
            _, ang, P0, t, about = max(turns[i], key=lambda x: x[0])
            turn_of[i] = (ang, P0, t, about)
    # a group fanned round an edge takes in the walls and regions beside two or more of its shapes that
    # are not round that edge (X100, which bounds X61 and meets F4 below it; then X58, beside F4 and
    # X100): each turned and moved by the mean of the shapes it lies beside there, so it sits between them
    attached = {}
    for e_ in sorted({e_ for e_, _ in sector_of.values()}):
        group = [j for j, (ee, _) in sector_of.items() if ee == e_]
        grew = True
        while grew:
            grew = False
            for j in range(n):
                sh = shapes[j]
                if j in group or j in sector_of or sh["kind"] not in ("wall", "region") or sh["id"] in completed:
                    continue
                r_j = rel_of(sh)
                mates = [g for g in group if shapes[g]["key"] in r_j or sh["key"] in rel_of(shapes[g])]
                if len(mates) < 2:
                    continue
                _, P0, t, about = turn_of[mates[0]]
                turn_of[j] = (float(np.mean([turn_of[g][0] for g in mates])), P0, t, about)
                shift[j] = np.mean([shift[g] for g in mates], 0)
                attached[sh["id"]] = [shapes[g]["id"] for g in mates]
                group.append(j)
                grew = True
        # then the walls bounding one of the group's regions and beside nothing else in it (X96 and
        # X41 against X58, X82 against X49): each moved and turned as that region is, so it stands up
        # against it where it truly bounds it, set WALL_OFF off its face, away from its volume
        regions_in = [g for g in group if shapes[g]["kind"] == "region"]
        for j in range(n):
            sh = shapes[j]
            if j in group or sh["kind"] not in ("wall", "line"):
                continue
            r_j = rel_of(sh)
            if sh["kind"] == "line":
                # (an edge completing a group region's shell lies on it: moved and turned as it is)
                on = [g for g in regions_in if completed.get(sh["id"]) == shapes[g]["id"]]
                if on:
                    turn_of[j] = turn_of[on[0]]
                    shift[j] = shift[on[0]]
                    attached[sh["id"]] = [shapes[on[0]]["id"]]
                    group.append(j)
                continue
            on = [g for g in regions_in if r_j.get(shapes[g]["key"]) == "side"]
            on = [g for g in on if completed.get(sh["id"], shapes[g]["id"]) == shapes[g]["id"]] or on
            if not on:
                continue
            g = on[0]
            ang, P0, t, about = turn_of[g]
            turn_of[j] = (ang, P0, t, about)
            away = sh["pts"].mean(0) - chart(shapes[g]["mid"])
            away = _turn(np.array([away]), np.zeros(3), t, ang)[0]
            off = COMPLETE_OFF if sh["id"] in completed else WALL_OFF
            shift[j] = shift[g] + off * away / max(np.linalg.norm(away), 1e-12)
            attached[sh["id"]] = [shapes[g]["id"]]
            group.append(j)
    # the stack on a region: each shape turned as the region is and set a step further out than the one
    # before, along the way the first lies from the region's middle, so they rest on top of it in order
    for base, chain in STACK_ON.items():
        ib = next((i for i, x in enumerate(shapes) if x["kind"] != "point" and x["id"] == base), None)
        if ib is None or ib not in turn_of:
            continue
        ang, P0, t, about = turn_of[ib]
        prev = ib
        for k_, nm in enumerate(chain):
            j = next((i for i, x in enumerate(shapes) if x["kind"] in ("wall", "region") and x["id"] == nm), None)
            if j is None:
                break
            if k_ == 0:
                mid = chart(shapes[ib]["mid"]) if shapes[ib].get("mid") is not None else shapes[ib]["pts"].mean(0)
                out_ = shapes[j]["pts"].mean(0) - mid
                out_ = _turn(np.array([out_]), np.zeros(3), t, ang)[0]
                out_ /= max(np.linalg.norm(out_), 1e-12)
            turn_of[j] = (ang, P0, t, about)
            shift[j] = shift[prev] + STACK_OFF * out_
            attached[nm] = [shapes[prev]["id"]]
            prev = j

    # walls lined up on one another along an edge (X22 on X82, along X4): each moved and turned as the
    # first is, and set ALIGN_OFF out from the edge in the direction it leaves it; the edge goes with
    # the first (X82 stays where it is, beside X49; X22 lines up just off it, under F3; X4 along X82)
    for e_name, (lead, walls) in ALIGN_ON.items():
        ie = next((i for i, x in enumerate(shapes) if x["kind"] == "line" and x["id"] == e_name), None)
        il = next((i for i, x in enumerate(shapes) if x["kind"] == "wall" and x["id"] == lead), None)
        if ie is None or il is None or il not in turn_of:
            continue
        ang, P0, t, about = turn_of[il]
        turn_of[ie] = turn_of[il]
        shift[ie] = shift[il]
        attached[e_name] = [lead]
        # (the edge set a little out from the leading wall, along the way it lies from its middle)
        away = shapes[ie]["pts"].mean(0) - shapes[il]["pts"].mean(0)
        away = _turn(np.array([away]), np.zeros(3), t, ang)[0]
        shift[ie] = shift[il] + EDGE_OFF * away / max(np.linalg.norm(away), 1e-12)
        for w_name in walls:
            j = next((i for i, x in enumerate(shapes) if x["kind"] == "wall" and x["id"] == w_name), None)
            if j is None:
                continue
            # (moved and turned exactly as the leading wall, so their edges on the edge and their
            # points line up; then set a small gap straight off its own face, away from the leading wall)
            P = shapes[j]["pts"]
            nrm = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][2]
            nrm *= np.sign(nrm @ (P.mean(0) - shapes[il]["pts"].mean(0))) or 1.0
            nrm = _turn(np.array([nrm]), np.zeros(3), t, ang)[0]
            turn_of[j] = (ang, P0, t, about)
            shift[j] = shift[il] + ALIGN_OFF * nrm
            attached[w_name] = [lead]
    # edges joined to B1's group (E1, X13): each turned by the mean of the turns of the group's walls
    # and regions it bounds and moved by the mean of their moves, then set EDGE_GAP further out from
    # them, so it lies along them, close but clear; then the corners (C6, A1) at the ends of the edges
    # in the group, set CORNER_GAP beyond the end of each and taking the mean over them
    in_group = set(sector_of) | {i for i, x in enumerate(shapes) if x["id"] in attached}
    joined_edges = [] if ib_edge(shapes, "B1") is None else [ib_edge(shapes, "B1")]
    for nm in JOIN_EDGES:
        i = ib_edge(shapes, nm)
        if i is None:
            continue
        r_i = rel_of(shapes[i])
        mates = [g for g in in_group if shapes[g]["kind"] in ("wall", "region") and g in turn_of
                 and (shapes[g]["key"] in r_i or shapes[i]["key"] in rel_of(shapes[g]))]
        if not mates:
            continue
        _, P0, t, about = turn_of[mates[0]]
        ang = float(np.mean([turn_of[g][0] for g in mates]))
        out_ = shapes[i]["pts"].mean(0) - np.mean([shapes[g]["pts"].mean(0) for g in mates], 0)
        out_ = _turn(np.array([out_]), np.zeros(3), t, ang)[0]
        turn_of[i] = (ang, P0, t, about)
        shift[i] = np.mean([shift[g] for g in mates], 0) + EDGE_GAP * out_ / max(np.linalg.norm(out_), 1e-12)
        attached[nm] = sorted({shapes[g]["id"] for g in mates})
        in_group.add(i)
        joined_edges.append(i)

    def placed(i, X):                              # (points of shape i, where shape i is drawn)
        if i in turn_of:
            ang, P0, t, _ = turn_of[i]
            X = _turn(X, P0, t, ang)
        return X + shift[i]
    # the corners at the joined edges' ends: one where two of them end (A1, at E1 and X13) at the
    # point in line with both (where their ends' lines, carried on, pass closest); one where one ends
    # (C6, at E1's other end) carried on beyond that end by as far as the first corner lies beyond it
    def end_line(ie, c0):
        L = shapes[ie]["v"]
        k = int(np.argmin(np.linalg.norm(L[[0, -1]] - c0, axis=1)))
        if np.linalg.norm(L[[0, -1]][k] - c0) > 2e-3:
            return None
        Lp = placed(ie, L)
        end, inner = (Lp[0], Lp[min(5, len(Lp) - 1)]) if k == 0 else (Lp[-1], Lp[-min(6, len(Lp))])
        d_ = end - inner
        return end, d_ / max(np.linalg.norm(d_), 1e-12)
    corner_gap = None
    warn_edges = []
    for nm in JOIN_CORNERS:
        ic = next((i for i, x in enumerate(shapes) if x["kind"] == "point" and x["id"] == nm), None)
        if ic is None:
            continue
        c0 = shapes[ic]["v"].reshape(3)
        ends = [(shapes[ie]["id"], e) for ie in joined_edges if shapes[ie]["id"] in JOIN_EDGES
                for e in [end_line(ie, c0)] if e is not None]
        if len(ends) >= 2:
            (n1, (p1, d1)), (n2, (p2, d2)) = ends[:2]
            # (the closest points of the lines p1 + a d1 and p2 + b d2)
            w0 = p1 - p2
            b_, d_, e_ = d1 @ d2, d1 @ w0, d2 @ w0
            den = 1 - b_ * b_
            a_ = (b_ * e_ - d_) / den if den > 1e-9 else 0.0
            c_ = (e_ - b_ * d_) / den if den > 1e-9 else 0.0
            spot = (p1 + a_ * d1 + p2 + c_ * d2) / 2
            corner_gap = float(np.linalg.norm(spot - p1))
        elif len(ends) == 1:
            n1, (p1, d1) = ends[0]
            spot = p1 + (corner_gap if corner_gap is not None else CORNER_GAP) * d1
        else:
            continue
        shift[ic] = spot - c0
        attached[nm] = [n_ for n_, _ in ends]
        in_group.add(ic)

    # edges lined up on a wall and in line with a corner (B1: on X38, with A1; C2a: on X96, with C6):
    # each moved and turned as its wall is, set EDGE_GAP out from it, so it lies along the wall's edge;
    # then slid only sideways (keeping its direction and its place along itself) so its line, carried
    # on past its end, runs through the corner: in line with it, its distance from it set by the wall
    for e_name, (w_name, c_name) in EDGE_ON_WALL.items():
        ie = ib_edge(shapes, e_name)
        iw = next((i for i, x in enumerate(shapes) if x["kind"] == "wall" and x["id"] == w_name), None)
        ic = next((i for i, x in enumerate(shapes) if x["kind"] == "point" and x["id"] == c_name), None)
        if ie is None or iw is None or iw not in turn_of:
            continue
        ang, P0, t, about = turn_of[iw]
        Le = shapes[ie]["pts"]
        dir_ = Le[-1] - Le[0]
        out_ = Le.mean(0) - shapes[iw]["pts"].mean(0)
        out_ -= (out_ @ dir_) / max(dir_ @ dir_, 1e-12) * dir_
        out_ = _turn(np.array([out_]), np.zeros(3), t, ang)[0]
        turn_of[ie] = (ang, P0, t, about)
        shift[ie] = shift[iw] + EDGE_GAP * out_ / max(np.linalg.norm(out_), 1e-12)
        attached.setdefault(e_name, [w_name])
        if ic is not None:
            c0 = shapes[ic]["v"].reshape(3)
            e = end_line(ie, c0)
            if e is not None:
                p_, d_ = e
                r_ = c0 + shift[ic] - p_
                shift[ie] = shift[ie] + (r_ - (r_ @ d_) * d_)
            else:
                warn_edges.append(f"{e_name} does not end at the {c_name} shown")

    # corners at the far ends of edges lined up on a wall (C4, beyond D2): carried on beyond that end
    # by as far as A1 lies beyond E1
    for c_name, e_names in END_CORNERS.items():
        ic = next((i for i, x in enumerate(shapes) if x["kind"] == "point" and x["id"] == c_name), None)
        if ic is None:
            continue
        c0 = shapes[ic]["v"].reshape(3)
        lines_ = []
        for e_name in e_names:
            ie = ib_edge(shapes, e_name)
            if ie is None or ie not in turn_of:
                continue
            e = end_line(ie, c0)
            if e is None:
                warn_edges.append(f"{e_name} does not end at the {c_name} shown")
                continue
            lines_.append((e_name, e))
        if len(lines_) >= 2:
            # (on the first edge's line, at the point of it nearest the second's line, but at least
            # MIN_CORNER_GAP beyond the first edge's end)
            (_, (p1, d1)), (_, (p2, d2)) = lines_[:2]
            w0 = p1 - p2
            b_, d_, e_ = d1 @ d2, d1 @ w0, d2 @ w0
            den = 1 - b_ * b_
            a_ = (b_ * e_ - d_) / den if den > 1e-9 else 0.0
            shift[ic] = p1 + max(a_, MIN_CORNER_GAP) * d1 - c0
        elif lines_:
            _, (p_, d_) = lines_[0]
            shift[ic] = p_ + (corner_gap if corner_gap is not None else CORNER_GAP) * d_ - c0
        else:
            continue
        attached[c_name] = [n_ for n_, _ in lines_]

    # layer 3, round edges of the group. Round X13 (FAN_ON_EDGE) the shapes form a small fan: in the
    # order they lie round it, the middle one (X31, the wall between X43 and X54) at its true angle and
    # the others turned about X13 to the same angle either side of it, so it is as far from each; all
    # turned as X13 is, set RING3_GAP out from it, and moved together LAYER3_OUT away from the group.
    # Round X4 (RING_ON_EDGE) a wall lines up on the edge: turned as X4 is, RING3_GAP out from it.
    layer3 = set()
    turned_about_edge = {}
    grp_now = set(sector_of) | {i for i, x in enumerate(shapes) if x["id"] in attached}

    def leave_dir(ie, j, V=None):               # (the way shape j leaves edge ie, across it)
        Le = shapes[ie]["pts"]
        P = shapes[j]["pts"] if V is None else V
        d, k = cKDTree(Le).query(P)
        near = d <= np.quantile(d, 0.1)
        tng = Le[np.minimum(k + 3, len(Le) - 1)] - Le[np.maximum(k - 3, 0)]
        tng /= np.maximum(np.linalg.norm(tng, axis=1, keepdims=True), 1e-12)
        off = P - Le[k]
        off -= np.einsum("nk,nk->n", off, tng)[:, None] * tng
        v = off[near].mean(0)
        return v / max(np.linalg.norm(v), 1e-12)
    for e_name, names in FAN_ON_EDGE.items():
        ie = ib_edge(shapes, e_name)
        if ie is None or ie not in turn_of:
            continue
        ang, P0, t, about = turn_of[ie]
        Le = shapes[ie]["pts"]
        ax_p, ax_t = Le.mean(0), (Le[-1] - Le[0]) / max(np.linalg.norm(Le[-1] - Le[0]), 1e-12)
        js = [next((i for i, x in enumerate(shapes) if x["kind"] in ("wall", "region") and x["id"] == nm), None)
              for nm in names]
        js = [j for j in js if j is not None]
        if not js:
            continue
        dirs = [leave_dir(ie, j) for j in js]
        b1_ = dirs[0] - (dirs[0] @ ax_t) * ax_t
        b1_ /= np.linalg.norm(b1_)
        b2_ = np.cross(ax_t, b1_)
        th = np.array([np.arctan2(v @ b2_, v @ b1_) for v in dirs])
        # (in the order listed: the middle one stays at its true angle; each other is turned about the
        # edge, away from it, until the nearest gap between their surfaces is FAN3_GAP, the same each
        # side; all are set out from the edge together, in the middle one's direction)
        mid = len(js) // 2

        def off_axis(P):                          # (points away from the edge: all meet at it)
            r_ = P - ax_p
            r_ -= np.outer(r_ @ ax_t, ax_t)
            return P[np.linalg.norm(r_, axis=1) > FAN3_CLEAR]
        Pm = off_axis(shapes[js[mid]]["pts"])
        tm = cKDTree(Pm)
        for k_, j in enumerate(js):
            if k_ == mid:
                continue
            side = np.sign(np.angle(np.exp(1j * (th[k_] - th[mid])))) or (1.0 if k_ > mid else -1.0)
            P = off_axis(shapes[j]["pts"])
            c_ang = lambda Q: np.arctan2((Q.mean(0) - ax_p) @ b2_, (Q.mean(0) - ax_p) @ b1_)
            th_m = c_ang(Pm)
            best_err, a_best = np.inf, 0.0
            for a_ in np.radians(np.arange(-40, 60.5, 0.5)):
                Q = _turn(P, ax_p, ax_t, side * a_)
                if side * np.angle(np.exp(1j * (c_ang(Q) - th_m))) <= 0:
                    continue                      # (it must stay on its own side of the middle one)
                err = abs(tm.query(Q)[0].min() - FAN3_GAP)
                if err < best_err:
                    best_err, a_best = err, a_
            for key_ in ("v", "pts"):
                shapes[j][key_] = _turn(shapes[j][key_], ax_p, ax_t, side * a_best)
        vm = np.cos(th[mid]) * b1_ + np.sin(th[mid]) * b2_
        vm = _turn(np.array([vm]), np.zeros(3), t, ang)[0]
        away = np.zeros(3)
        for j in js:
            turn_of[j] = (ang, P0, t, about)
            shift[j] = shift[ie] + RING3_GAP * vm
            attached[shapes[j]["id"]] = [e_name]
            layer3.add(j)
        # (then all moved together, straight away from the group shapes they lie on)
        for j in js:
            Pj = placed(j, shapes[j]["pts"])
            for g in grp_now:
                if shapes[g]["kind"] in ("wall", "region") and g not in js:
                    Pg = placed(g, shapes[g]["pts"])
                    if (cKDTree(Pg).query(Pj)[0] < 0.01).mean() >= 0.1:
                        away += Pj.mean(0) - Pg.mean(0)
        if np.linalg.norm(away) > 1e-12:
            for j in js:
                shift[j] = shift[j] + LAYER3_OUT * away / np.linalg.norm(away)
    for e_name, names in RING_ON_EDGE.items():
        ie = ib_edge(shapes, e_name)
        if ie is None or ie not in turn_of:
            continue
        ang, P0, t, about = turn_of[ie]
        for nm in names:
            j = next((i for i, x in enumerate(shapes) if x["kind"] in ("wall", "region") and x["id"] == nm), None)
            if j is None:
                continue
            # (moved and turned exactly as the edge is, so its own edge lies along it, and set
            # RING3_GAP out from it in the direction it truly leaves it)
            v = _turn(np.array([leave_dir(ie, j)]), np.zeros(3), t, ang)[0]
            turn_of[j] = (ang, P0, t, about)
            shift[j] = shift[ie] + RING3_GAP * v
            attached[nm] = [e_name]
            layer3.add(j)
    for l_name, (by, c_name, _beside) in LINE_AT_CORNER.items():
        il, ib_, ic = ib_edge(shapes, l_name), ib_edge(shapes, by), \
            next((i for i, x in enumerate(shapes) if x["kind"] == "point" and x["id"] == c_name), None)
        if il is None or ib_ is None or ic is None or ib_ not in turn_of:
            continue
        turn_of[il] = turn_of[ib_]
        shift[il] = shift[ib_].copy()
        c0 = shapes[ic]["v"].reshape(3)
        e = end_line(il, c0)
        if e is None:
            warn_edges.append(f"{l_name} does not end at the {c_name} shown")
        else:
            p_, d_ = e
            r_ = c0 + shift[ic] - p_
            along = r_ @ d_
            # (slid sideways into line with the corner; and back along itself if it reaches nearer
            # than MIN_CORNER_GAP to it)
            shift[il] = shift[il] + (r_ - along * d_) + min(0.0, along - MIN_CORNER_GAP) * d_
        attached[l_name] = [by, c_name]
        layer3.add(il)

    # the layers shown (the slider): first B1 and every shape touching it, then the shapes set with
    # those round B1 (the group's added shapes), then outward, each layer the shapes the shape map puts
    # beside the last (where shapes are placed is not changed by this)
    show_layer = {}
    ib1 = next((i for i, x in enumerate(shapes) if x["kind"] == "line" and x["id"] == "B1"), None)
    if ib1 is not None:
        r_b1 = rel_of(shapes[ib1])
        ends = shapes[ib1]["v"][[0, -1]]
        at_end = {i for i, x in enumerate(shapes) if x["kind"] == "point"
                  and np.linalg.norm(ends - x["v"].reshape(3), axis=1).min() < 2e-3}   # (its corners)
        first_ids = {ib1} | {i for i, x in enumerate(shapes) if x["key"] in r_b1} | set(sector_of) | at_end
        for i in first_ids:
            show_layer[i] = 1
        for i, x in enumerate(shapes):
            if i not in show_layer and x["id"] in attached and i not in layer3:
                show_layer[i] = 2
        for i in layer3:
            if i not in show_layer:
                show_layer[i] = 3
        lay_n = max(show_layer.values())
        frontier = [i for i in show_layer if show_layer[i] == lay_n]
        rels_all = [rel_of(x) if x["kind"] != "point" else {} for x in shapes]
        while len(show_layer) < len(shapes):
            lay_n += 1
            fk = {shapes[i]["key"] for i in frontier}
            nxt = [i for i, x in enumerate(shapes) if i not in show_layer and
                   (any(k in rels_all[i] for k in fk) or any(x["key"] in rels_all[f] for f in frontier))]
            if not nxt:                           # (any left over, by where they were found)
                nxt = [i for i in range(len(shapes)) if i not in show_layer]
            for i in nxt:
                show_layer[i] = lay_n
            frontier = nxt
    out = []
    for sh, s in zip(shapes, shift):
        V3 = sh["v"]
        rec_turn = None
        if idx[id(sh)] in turn_of:
            ang, P0, t, about = turn_of[idx[id(sh)]]
            rec_turn = (about, round(float(np.degrees(ang)), 1))
            V3 = _turn(V3, P0, t, ang)
        V3 = V3 + s
        rec = {"id": sh["id"], "kind": sh["kind"], "note": sh["note"],
               "layer": show_layer.get(idx[id(sh)], layer_of[sh["key"]]), "moved": np.round(s, 4).tolist()}
        if rec_turn:
            rec["turned"] = {"about": rec_turn[0], "degrees": rec_turn[1]}
        if sh["kind"] in ("wall", "region"):
            # (thinned for the page: corners within a GRID cell merged, triangles left flat dropped)
            Vu, F3 = np.unique(np.round(V3.reshape(-1, 3) / GRID), axis=0, return_inverse=True)
            F3 = F3.reshape(-1, 3)
            F3 = F3[(F3[:, 0] != F3[:, 1]) & (F3[:, 1] != F3[:, 2]) & (F3[:, 0] != F3[:, 2])]
            _, first = np.unique(np.sort(F3, 1), axis=0, return_index=True)
            F3 = F3[np.sort(first)]
            used = np.unique(F3)
            remap = -np.ones(len(Vu), int)
            remap[used] = np.arange(len(used))
            # (each merged corner at the mean of the corners it stands for, not its cell's middle,
            # so edges stay smooth)
            inv = np.unique(np.round(V3.reshape(-1, 3) / GRID), axis=0, return_inverse=True)[1].ravel()
            mean = np.zeros((len(Vu), 3))
            np.add.at(mean, inv, V3.reshape(-1, 3))
            mean /= np.bincount(inv, minlength=len(Vu))[:, None]
            rec.update(verts=mean[used].round(4).tolist(), faces=remap[F3].tolist())
        elif sh["kind"] == "line":
            rec["path"] = V3.round(5).tolist()
        else:
            rec["at"] = V3.reshape(3).round(5).tolist()
        if sh.get("warn"):
            rec["warn"] = sh["warn"]
        out.append(rec)
    # the axis, moved with X38 (turned and shifted as X38 is), so it still runs along X38's edge on it
    L = np.linspace(-0.35, 0.35, 2)
    axis3 = np.array([[t_, 0.0, 0.0] for t_ in L])
    i38 = next(i for i, sh in enumerate(shapes) if sh["id"] == "X38")
    if i38 in turn_of:
        ang, P0, t, _ = turn_of[i38]
        axis3 = _turn(axis3, P0, t, ang)
    axis3 = axis3 + shift[i38]
    return {"_about": __doc__, "axis": axis3.round(5).tolist(), "shapes": out,
            "links": [[i, j] for i, j, _ in rows], "missing": missing,
            "fit": {"turned_about_edge": turned_about_edge, "warnings": warn_edges, "attached": attached, "arcs": arcs, "gap": GAP, "relative_residual_median": float(np.median(resid)), "relative_residual_max": float(np.max(resid)),
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
