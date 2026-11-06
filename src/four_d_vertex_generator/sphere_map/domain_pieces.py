"""Wall pieces in the domain, built from their exact edges, each triangulated inside its own loop.

The edges come first, all from the database:

* named lines: the atlas's line pieces (`atlas.json`), cut at the exact named points and with
  their ends snapped there; each stretch resampled with every point solved onto its two exact
  surfaces;
* wall-meeting curves: `merged_counts.json`'s curves where one wall meets another (those not on
  a named line), traced exactly between their snapped end points;
* the half-cell's boundary: on each wall surface, the exact curve where it meets each face of
  the half-cell (traced on the surface and the face, its ends solved onto the next face).

Then, on every surface a wall lies on (a library surface, or a face of the half-cell for the
walls lying in one), these curves form a graph: their ends and every point where two of them
cross (solved exactly on the surface and both curves' second surfaces) are its vertices. Going
round each vertex in the surface's tangent plane gives its faces, each bounded by exact loops.

A face is a piece of a wall, or of no wall, with the same regions on each side all through: it
is named from the database's pieces at points inside it (`pieces.json` signs, `patches.json`,
`components.npz` region components). Each wall face is triangulated inside its loops: the loop
points are the only boundary vertices (none added on the rim), every inner vertex is solved onto
the exact surface.

Writes `domain_pieces.json` (faces, their loops' curves and what they are) and
`domain_pieces.npz` (curve points, triangles).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .chords import FACE_N, FACE_NAMES
from .frame import data_dir, group, locate, point_from_beta

STEP = 3e-3              # spacing of the exact curve points (betas)
TOL_ON = 1e-9            # a point is on a surface when |f| / |grad f| is below this
TOL_V = 1e-6             # curve ends closer than this are one vertex (they are exact points)
FACES = [0, 1, 3, 4]     # the half-cell's faces: b1 = 0, b2 = 0, b4 = 0, b3 = b4 (b3 >= b4 >= 0)


# ------------------------------------------------------------------------------- surfaces

class Plane:
    kind = "plane"

    def __init__(self, n, name):
        self.normal = np.asarray(n, float)
        self.name = name

    def f(self, X):
        return np.atleast_2d(X) @ self.normal

    def grad(self, X):
        return np.broadcast_to(self.normal, np.atleast_2d(X).shape).copy()


class Pulled:
    """Surface s of copy c, read in the domain's betas: s(beta . M) with M = T g_c T^-1, signed so
    the domain's betas map to betas of positive sum."""
    kind = "curved"

    def __init__(self, s, M, name):
        self.s, self.M, self.name = s, M, name

    def f(self, X):
        return self.s.f(np.atleast_2d(X) @ self.M)

    def grad(self, X):
        return self.s.grad(np.atleast_2d(X) @ self.M) @ self.M.T


def _fg(s):
    def fn(x):
        g = s.grad(x[None])[0]
        return float(s.f(x[None])[0]), g - g.mean()
    return fn


def solve(surfs, x, iters=40):
    """x moved onto every surface in surfs, keeping sum(beta) = 1 (Newton, least norm)."""
    fs = [_fg(s) for s in surfs]
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        v = np.array([f(x)[0] for f in fs])
        if np.abs(v).max() < 1e-16:
            break
        J = np.array([f(x)[1] for f in fs])
        if not (np.isfinite(J).all() and np.isfinite(v).all()):
            return x, False
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-v, 1.0 - x.sum()], rcond=None)[0]
        if not np.isfinite(x).all() or np.abs(x).max() > 1e3:
            return x, False
    ok = all(abs(f(x)[0]) <= 1e-12 * (1 + np.linalg.norm(f(x)[1])) for f in fs)
    return x, ok


def dist_to(s, X):
    """|f| / |grad f| at the points X: the distance to the surface, to first order."""
    X = np.atleast_2d(X)
    g = s.grad(X)
    g = g - g.mean(1, keepdims=True)
    return np.abs(s.f(X)) / (np.linalg.norm(g, axis=1) + 1e-300)


def inside(X, tol=1e-12):
    return (np.atleast_2d(X) @ FACE_N[FACES].T >= -tol).all(1)


# ------------------------------------------------------------------------------- curves

def resample(P, surfs, step=STEP):
    """A polyline of points on the surfaces, made dense: new points every <= step, each solved
    exactly onto the surfaces (the given points kept)."""
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        if L < 1e-14:
            continue
        m = int(np.ceil(L / step))
        for t in np.arange(1, m) / m:
            x0 = a + t * (b - a)
            x, ok = solve(surfs, x0)
            out.append(x if ok and np.linalg.norm(x - x0) < 0.25 * L else x0)
        out.append(b)
    return np.array(out)


def trace_on(s, t, others, step=STEP, n_seeds=1500, rng=None, start=()):
    """Every arc of the curve s = t = 0 inside the half-cell (its ends solved onto the face it
    leaves by): polylines in betas."""
    rng = rng or np.random.default_rng(0)
    seeds = []
    for x0 in rng.dirichlet(np.ones(4), n_seeds):
        x0[2], x0[3] = max(x0[2], x0[3]), min(x0[2], x0[3])
        x, ok = solve([s, t], x0)
        if ok and inside(x, 1e-12):
            seeds.append(x)
    for x0 in start:
        x, ok = solve([s, t], x0)
        if ok and inside(x, 1e-11):
            seeds.insert(0, x)
    seeds = np.array(seeds) if seeds else np.zeros((0, 4))
    done = np.zeros(len(seeds), bool)
    arcs = []
    fs, ft = _fg(s), _fg(t)
    for i in range(len(seeds)):
        if done[i]:
            continue
        halves, closed = [], False
        for sgn in (1, -1):
            pts, x, tp = [seeds[i]], seeds[i], None
            for _ in range(4000):
                tan = np.linalg.svd(np.vstack([np.ones(4), fs(x)[1], ft(x)[1]]))[2][-1]
                tan = sgn * tan if tp is None else (tan if tan @ tp > 0 else -tan)
                tp = tan
                y, ok = solve([s, t], x + step * tan / np.linalg.norm(tan))
                if len(pts) == 1 and not (ok and inside(y, 1e-13)):
                    # (from a corner or an edge of the half-cell, a full step may leave it at once:
                    # shorter first steps)
                    for frac in (0.3, 0.1, 0.03, 0.01):
                        y, ok = solve([s, t], x + frac * step * tan / np.linalg.norm(tan))
                        if ok and inside(y, 1e-13):
                            break
                if not ok:
                    break
                if not inside(y, 1e-13):
                    vals = y @ FACE_N[FACES].T
                    k = FACES[int(np.argmin(vals))]
                    z, ok = solve([s, t, others[k]], x)
                    if ok and inside(z, 1e-11):
                        pts.append(z)
                    break
                if len(pts) > 20 and np.linalg.norm(y - seeds[i]) < 1.5 * step:
                    closed = True
                    pts.append(seeds[i])
                    break
                pts.append(y)
                x = y
            halves.append(pts)
            if closed:
                break
        poly = np.array(halves[0][::-1] + (halves[1][1:] if len(halves) > 1 else []))
        poly = poly[np.isfinite(poly).all(1)]
        d = np.min(np.linalg.norm(seeds[:, None] - poly[None], axis=2), axis=1)
        done |= d < 3 * step
        arcs.append(poly)
    return arcs


# ------------------------------------------------------------------------------- the build

def _dedupe(P, tol=1e-12):
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > tol:
            keep.append(i)
    return P[keep]


class Data:
    def __init__(self):
        from .atlas_db import _library_surfaces, _piece_keys
        root = self.root = data_dir() / "sphere_map"
        self.lib = list(_library_surfaces(root))
        self.keys = _piece_keys(root)
        pieces = json.loads((root / "pieces.json").read_text())["pieces"]
        self.region = [p["region"] for p in pieces]
        self.patches = json.loads((root / "patches.json").read_text())["patches"]
        self.rcomp = np.load(root / "components.npz")["region_component"]
        self.pw = {}
        self.fw = {}
        for p in self.patches:
            if not p["wall"]:
                continue
            a, b = p["pieces"]
            if p["face"] is None:
                self.pw[(min(a, b), max(a, b), p["surface"])] = p
            else:
                self.fw[(a, b, p["side_b_copy_shift"])] = p
        self.face_planes = {k: Plane(FACE_N[k], f"face {FACE_NAMES[k]}") for k in range(len(FACE_N))}
        from .frame import multiplication_table
        self.mult = multiplication_table()
        self.inv = np.array([int(np.flatnonzero(self.mult[g] == 0)[0]) for g in range(len(self.mult))])

    def signs(self, X):
        V = np.stack([s.f(X) for s in self.lib], 1)
        G = np.stack([np.linalg.norm(s.grad(X) - s.grad(X).mean(1, keepdims=True), axis=1)
                      for s in self.lib], 1)
        return np.where(np.abs(V) / (G + 1e-300) < 1e-11, 0, np.sign(V)).astype(np.int8)

    def piece(self, x):
        return self.keys.get(self.signs(x[None])[0].tobytes())


def curves(D, log=print):
    """The named lines' stretches and the wall-meeting curves in the domain, exact and dense, and
    the library surfaces each lies on."""
    from .wall_meets import resolve
    atlas = json.loads((D.root / "atlas.json").read_text())
    mc = json.loads((D.root / "merged_counts.json").read_text())["wall_meeting_curves"]
    out = []
    for lp in atlas["line_pieces"]:
        for s in lp["segments"]:
            P = np.asarray(s["polyline_beta"], float)
            P = _dedupe(P / P.sum(1, keepdims=True))
            eqs = [resolve(e) for e in s["equations"]]
            out.append({"kind": "line", "name": lp["line_shape"], "piece": lp["id"],
                        "pts": resample(P, eqs), "eqs": eqs})
    for e in mc:
        if not e.get("found") or e["on_atlas_line"]:
            continue
        for P in e["domain_polylines"]:
            P = _dedupe(np.asarray(P, float))
            P = P / P.sum(1, keepdims=True)
            out.append({"kind": "meet", "name": e["walls"], "pts": P, "eqs": None})
    # the library surfaces each lies on
    for c in out:
        P = c["pts"]
        c["on"] = [i for i, s in enumerate(D.lib) if dist_to(s, P).max() < 1e-8]
        if c["eqs"] is None:
            c["eqs"] = [D.lib[i] for i in c["on"][:2]]
    log(f"curves: {sum(c['kind'] == 'line' for c in out)} named line stretches, "
        f"{sum(c['kind'] == 'meet' for c in out)} wall-meeting curves; "
        f"{sum(len(c['on']) < 2 for c in out)} on fewer than two library surfaces")
    return out


def sheets(D):
    """The surfaces walls lie on in the domain: library surfaces of interior patches, and the
    domain's faces holding face patches."""
    out = defaultdict(set)
    for p in D.patches:
        if p["wall"]:
            out[p["surface"] if p["face"] is None else "face:" + p["face"]].add(p["wall"])
    return out


def wall_surfaces(D):
    """Library surface -> the walls with a patch on it."""
    out = defaultdict(set)
    for p in D.patches:
        if p["wall"] and p["face"] is None:
            out[p["surface"]].add(p["wall"])
    return out


def gaps(D, key, faces, WS, done, s, normal):
    """Surfaces to trace on this sheet: where the samples of one wall face are named differently,
    walk on the surface from one to the other and find, by bisection, the point where the naming
    changes; the surfaces changing sign there (read in the copy across, for a face sheet's far
    side) are traced."""
    from .frame import cell_vertices
    T = cell_vertices()
    Ti = np.linalg.inv(T)
    g = group()
    need = {}

    def at(x0):
        n = normal(x0)
        n = n / np.linalg.norm(n)
        y = _lift(s, x0, n)
        if y is None:
            return None, None
        gg = normal(y)
        return y, name_point(D, key, y, gg / np.linalg.norm(gg))

    key_of = lambda nm: None if nm is None else (nm["wall"], tuple(nm["sides"]))
    for f in faces:
        tn = f.get("tri_names")
        if tn is None or f["agree"]:
            continue
        F, cen = f["F"], f["centres"]
        nb = defaultdict(list)
        for ti, t in enumerate(F):
            for a_, b_ in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                nb[(min(a_, b_), max(a_, b_))].append(ti)
        pairs = {(u, v) for ts in nb.values() if len(ts) == 2 for u, v in [ts]
                 if tn[u] is not None and tn[v] is not None and tn[u] != tn[v]
                 and tn[u][0] is not None and tn[v][0] is not None}
        chosen = []                        # (a spread of pairs per change: each 0.02 from the last)
        for u, v in sorted(pairs):
            tr_ = (tn[u], tn[v])
            if any(t2 == tr_ and np.linalg.norm(cen[u] - c2) < 0.02 for t2, c2 in chosen):
                continue
            chosen.append((tr_, cen[u]))
            if len(chosen) > 60:
                break
        for (tr_, _c) in chosen:
            u = next(a_ for a_, b_ in pairs if (tn[a_], tn[b_]) == tr_ and np.allclose(cen[a_], _c))
            v = next(b_ for a_, b_ in pairs if a_ == u and (tn[a_], tn[b_]) == tr_)
            ya, y = cen[u], cen[v]
            na = tn[u]
            lo, hi = 0.0, 1.0
            for _ in range(32):
                mid = (lo + hi) / 2
                _, nmid = at(ya + mid * (y - ya))
                if key_of(nmid) == na:
                    lo = mid
                else:
                    hi = mid
            pa, _ = at(ya + lo * (y - ya))
            pb, _ = at(ya + hi * (y - ya))
            if pa is None or pb is None:
                continue
            sa, sb = D.signs(pa[None])[0], D.signs(pb[None])[0]
            xm = (pa + pb) / 2
            for t in np.flatnonzero(sa != sb):
                if t != key:
                    k_ = ("lib", int(t))
                    need.setdefault(k_, [D.lib[int(t)], "/".join(sorted(WS.get(int(t), ()))), []])[2].append(xm)
            if isinstance(key, str):
                k = FACE_NAMES.index(key[5:])
                nin = FACE_N[k] - FACE_N[k].mean()
                kk, _ = locate(point_from_beta(pa - 1e-7 * nin)[None], tol=1e-13)
                kk = int(kk[0])
                M = T @ g[kk] @ Ti
                M = M * np.sign((pa @ M).sum())
                qa, qb = D.signs((pa @ M)[None])[0], D.signs((pb @ M)[None])[0]
                for t in np.flatnonzero(qa != qb):
                    k_ = ("copy", kk, int(t))
                    need.setdefault(k_, [Pulled(D.lib[int(t)], M, f"{D.lib[int(t)].name} (copy {kk})"),
                                         "/".join(sorted(WS.get(int(t), ()))), []])[2].append(xm)
    return need


def do_sheet(D, C, key, WS, log=print, rounds=3):
    """One sheet: its graph of exact curves, faces named; where a wall face's naming changes
    inside it, the surfaces responsible traced and the sheet built again."""
    s = D.face_planes[FACE_NAMES.index(key[5:])] if isinstance(key, str) else D.lib[key]
    extra, done = [], set()
    r = 0
    while r < rounds:
        r += 1
        G = sheet(D, C, key, log, extra)["graph"]
        F = faces_of(D, key, G, log=log)
        F = [f for f in F if not f.get("too_curved")]
        need = gaps(D, key, F, WS, done, s, G.normal)
        # (a point already on a traced arc needs nothing more)
        for k in list(need):
            pts = [x for x in need[k][2]
                   if not any(seg_dist(e["pts"], None, x) < 1e-6 for e in extra if e.get("key") == k)]
            if pts:
                need[k][2] = pts
            else:
                del need[k]
        if not need or r >= rounds:
            break
        for k, (t, walls, pts) in need.items():
            first = k not in done
            done.add(k)
            for a in trace_on(s, t, D.face_planes, n_seeds=400 if first else 0, start=pts):
                if any(seg_dist(e["pts"], None, a[len(a) // 2]) < 1e-6 for e in extra if e.get("key") == k):
                    continue
                extra.append({"kind": "cross", "name": walls, "pts": a, "eqs": [s, t], "key": k,
                              "surface": getattr(t, "name", str(k))})
        log(f"  sheet {key}: {len(need)} surfaces traced where a face's naming changed")
    return G, F


def build(log=print, only=None):
    D = Data()
    C = curves(D, log)
    WS = wall_surfaces(D)
    out = []
    for key in sorted(sheets(D), key=str):
        if only and key not in only:
            continue
        G, F = do_sheet(D, C, key, WS, log)
        out.append((key, G, F))
    return D, out


# ------------------------------------------------------------------------------- one sheet

def sheet(D, C, key, log=print, extra=()):
    """The graph of exact curves on one surface, its faces, what they are, their triangles."""
    if isinstance(key, str):
        k = FACE_NAMES.index(key[5:])
        s = D.face_planes[k]
        normal = lambda x: FACE_N[k] - FACE_N[k].mean()
        own = [c for c in C if np.abs(c["pts"] @ FACE_N[k]).max() < 1e-10]
        own += glued_images(D, C, k)
        bound = [D.face_planes[j] for j in FACES if j != k]
        arcs = []
        arcs = boundary_arcs(s, [j for j in FACES if j != k], D, own)
    else:
        s = D.lib[key]
        normal = lambda x: (lambda g: g - g.mean())(s.grad(x[None])[0])
        own = [c for c in C if key in c["on"]]
        arcs = []
        arcs = boundary_arcs(s, FACES, D, own)
    edges = [dict(c, eqs=[s, second(s, c, D)]) for c in own] + arcs + list(extra)
    G = Graph(s, normal)
    for e in edges:
        G.add_curve(e)
    G.split_crossings()
    G.faces()
    log(f"sheet {key}: {len(own)} curves of the database on it, {len(arcs)} boundary arcs, "
        f"{len(extra)} traced where its faces changed; "
        f"{len(G.V)} vertices, {len(G.E)} edges, {len(G.F)} faces")
    return {"key": key, "graph": G}


def glued_images(D, C, k):
    """The curves of the copies glued to the domain across face k, read in the domain: each
    database curve carried by every element gluing a copy there (both ways), its stretches lying
    in the face kept. (On the face the domain and the copy across share their points, and the
    database keeps each such curve once, in one of the two copies' betas.)"""
    from .frame import cell_vertices
    T = cell_vertices()
    Ti = np.linalg.inv(T)
    g = group()
    shifts = sorted({p["side_b_copy_shift"] for p in D.patches if p["face"] == FACE_NAMES[k]})
    out = []
    for c_ in shifts:
        for gi in (c_, int(D.inv[c_])):
            M = T @ g[gi] @ Ti                     # beta (domain) -> beta in copy gi's frame
            Minv = np.linalg.inv(M)               # a point of that frame, read in the domain
            for cv in C:
                Y = cv["pts"] @ Minv
                Y = Y / Y.sum(1, keepdims=True)
                on = (np.abs(Y @ FACE_N[k]) < 1e-10) & inside(Y, 1e-10)
                runs, cur = [], []
                for y, o in zip(Y, on):
                    if o:
                        cur.append(y)
                    elif cur:
                        runs.append(cur)
                        cur = []
                if cur:
                    runs.append(cur)
                for r in runs:
                    if len(r) < 2:
                        continue
                    r = np.array(r)
                    if any(len(o["pts"]) == len(r) and np.abs(o["pts"] - r).max() < 1e-9 for o in C):
                        continue                   # (its own place: already in the database)
                    out.append({"kind": cv["kind"], "name": cv["name"], "piece": cv.get("piece"),
                                "pts": r, "on": [], "glued": int(gi),
                                "eqs": [Pulled(e, M * np.sign(np.sum(r[0] @ M)), f"{getattr(e, 'name', '')} @ {gi}")
                                        for e in cv["eqs"]]})
    return out


def boundary_arcs(s, faces, D, own):
    """The surface's whole boundary in the half-cell: its curves on each face, traced from random
    starts and from every point where a curve of the database on it ends on that face; then closed
    up: an arc's end on another face starts that face's curve there, until no end is left open."""
    arcs = {j: [] for j in faces}
    on_face = lambda x, j: abs(x @ FACE_N[j]) < 1e-9
    starts = {j: [] for j in faces}
    for c in own:
        for x in (c["pts"][0], c["pts"][-1]):
            for j in faces:
                if on_face(x, j):
                    starts[j].append(x)
    for j in faces:
        arcs[j] = trace_on(s, D.face_planes[j], D.face_planes, start=starts[j])
    for _ in range(4):
        new = False
        for j in faces:
            for a in list(arcs[j]):
                for x in (a[0], a[-1]):
                    for j2 in faces:
                        if j2 == j or not on_face(x, j2):
                            continue
                        if any(seg_dist(b, None, x) < 1e-7 for b in arcs[j2]):
                            continue
                        got = trace_on(s, D.face_planes[j2], D.face_planes, n_seeds=0, start=[x])
                        got = [g for g in got if len(g) > 1]
                        if got:
                            arcs[j2] += got
                            new = True
        if not new:
            break
    # the half-cell's edges: where the surface meets each (exact points), or the stretch of it the
    # surface holds; an arc end left open near such a point is joined to it along the surface
    corners = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5]], float)
    nodes = {j: [] for j in faces}
    for a_i in range(4):
        for b_i in range(a_i + 1, 4):
            A, B = corners[a_i], corners[b_i]
            on = [j for j in FACES if abs(A @ FACE_N[j]) < 1e-12 and abs(B @ FACE_N[j]) < 1e-12]
            if len(on) < 2:
                continue
            tt = np.linspace(0, 1, 2001)
            X = A + tt[:, None] * (B - A)
            v = s.f(X)
            if dist_to(s, X[1:-1:50]).max() < 1e-9:
                # (the surface holds this edge: the stretches traced along it on either face are
                # dropped, and the edge itself is the boundary there, once)
                seg = X[::20]
                for j in on:
                    if j not in faces:
                        continue
                    kept = []
                    for a in arcs[j]:
                        off_edge = ~((np.abs(a @ FACE_N[on[0]]) < 1e-9) & (np.abs(a @ FACE_N[on[1]]) < 1e-9))
                        run, off = [], False
                        for x, o in zip(a, off_edge):
                            if o:
                                run.append(x)
                                off = True
                            else:
                                if off:
                                    run.append(x)          # (a stretch off the edge, ending on it)
                                    kept.append(np.array(run))
                                run, off = [x], False
                        if off and len(run) > 1:
                            kept.append(np.array(run))
                    arcs[j] = kept
                j0 = next((j for j in on if j in faces), None)
                if j0 is not None:
                    arcs[j0].append(seg)
                continue
            for k in np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:])):
                lo, hi = tt[k], tt[k + 1]
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if np.sign(s.f((A + mid * (B - A))[None])[0]) == np.sign(v[k]):
                        lo = mid
                    else:
                        hi = mid
                x = A + (lo + hi) / 2 * (B - A)
                for j in on:
                    if j in faces:
                        nodes[j].append(x)
    for j in faces:
        for a_i in range(len(arcs[j])):
            a = arcs[j][a_i]
            for k in (0, -1):
                x = a[k]
                if any(np.linalg.norm(x - y) < 1e-6 for y in nodes[j]):
                    continue
                touching = sum(1 for j2 in faces for b in arcs[j2]
                               if b is not a and seg_dist(b, None, x) < 1e-6)
                if touching:
                    continue
                # the nearest exact point to join: a point where the surface meets an edge of the
                # half-cell, or the nearest point of another arc on this face (solved onto it)
                cands = [(np.linalg.norm(y - x), y) for y in nodes[j]]
                for j2 in faces:
                    for b in arcs[j2]:
                        if b is a or len(b) < 2:
                            continue
                        d_ = np.linalg.norm(b - x, axis=1)
                        q = b[int(np.argmin(d_))]
                        if abs(q @ FACE_N[j]) < 1e-9:
                            y, ok = solve([s, D.face_planes[j], D.face_planes[j2]] if j2 != j else [s, D.face_planes[j]], q)
                            cands.append((np.linalg.norm((y if ok else q) - x), y if ok else q))
                if not cands:
                    continue
                dmin, y = min(cands, key=lambda c: c[0])
                if 1e-9 < dmin < 0.03:
                    arcs[j].append(resample(np.array([x, y]), [s, D.face_planes[j]], step=STEP / 3))
    out = []
    for j in faces:
        kept = []
        for a in sorted([a for a in arcs[j] if len(a) > 1], key=len, reverse=True):
            probe_ = a[[0, len(a) // 4, len(a) // 2, 3 * len(a) // 4, -1]]
            if any(all(seg_dist(b, None, x) < 2e-5 for x in probe_) for b in kept):
                continue                         # (the same arc traced again)
            kept.append(a)
        out += [{"kind": "face", "name": D.face_planes[j].name, "pts": a, "eqs": [s, D.face_planes[j]]}
                for a in kept]
    return out


def second(s, c, D):
    """The surface that, with s, cuts out curve c: of its equations and the library surfaces it
    lies on, the one most across s along it."""
    x = c["pts"][len(c["pts"]) // 2]
    gs = s.grad(x[None])[0]
    gs = gs - gs.mean()
    best, bc = None, 2.0
    for e in list(c["eqs"]) + [D.lib[i] for i in c["on"]]:
        g = e.grad(x[None])[0]
        g = g - g.mean()
        cs = abs(g @ gs) / (np.linalg.norm(g) * np.linalg.norm(gs) + 1e-300)
        if cs < bc:
            best, bc = e, cs
    return best


def _cross_along(s, ti, tj, a, b, iters=60):
    """Where the curve s = ti = 0 between its points a and b crosses the surface tj: bisection on
    tj along it, every point solved back onto the curve."""
    fa = float(tj.f(a[None])[0])
    fb = float(tj.f(b[None])[0])
    if np.sign(fa) == np.sign(fb):
        return a, False
    lo, hi = 0.0, 1.0
    x = a
    for _ in range(iters):
        mid = (lo + hi) / 2
        x, ok = solve([s, ti], a + mid * (b - a))
        if not ok:
            return x, False
        fm = float(tj.f(x[None])[0])
        if abs(fm) < 1e-15:
            break
        if np.sign(fm) == np.sign(fa):
            lo = mid
        else:
            hi = mid
    return x, True


def seg_dist(P, tree, x):
    """Distance from x to the polyline P (exact, to its segments near x)."""
    if tree is None:
        js = range(len(P) - 1)
    else:
        _, idx = tree.query(x, k=min(4, len(P)))
        js = {j for i in np.atleast_1d(idx) for j in (i - 1, i) if 0 <= j < len(P) - 1}
    best = np.linalg.norm(P[0] - x) if len(P) else np.inf
    for j in js:
        a, b = P[j], P[j + 1]
        L2 = (b - a) @ (b - a)
        t = 0.0 if L2 == 0 else float(np.clip((x - a) @ (b - a) / L2, 0, 1))
        best = min(best, float(np.linalg.norm(a + t * (b - a) - x)))
    return best


class Graph:
    """Curves on one surface, as a graph whose faces are found by going round each vertex."""

    def __init__(self, s, normal):
        self.s, self.normal = s, normal
        self.V = []                  # vertex points
        self.E = []                  # {"a", "b", "pts", "curve"}
        self.curves = []
        self.F = []

    def vertex(self, x):
        for i, v in enumerate(self.V):
            if np.linalg.norm(v - x) < TOL_V:
                return i
        self.V.append(np.asarray(x, float))
        return len(self.V) - 1

    def add_curve(self, c):
        self.curves.append(c)

    def split_crossings(self):
        """Every curve cut where another curve on the surface meets it: where an end of one lies
        on the other (that exact end), and where two cross (solved onto the surface and both
        curves). A curve lying along another (a stretch of the same exact curve, as a wall
        meeting along the half-cell's face) is not crossed by it: it is cut at its ends, and
        the stretch kept once, named by what the database says it is."""
        from scipy.spatial import cKDTree
        cs = self.curves
        # (exact duplicates dropped)
        keep = []
        for ci, c in enumerate(cs):
            if any(len(c["pts"]) == len(cs[k]["pts"]) and np.abs(c["pts"] - cs[k]["pts"]).max() < 1e-9
                   for k in keep):
                continue
            keep.append(ci)
        cs = self.curves = [cs[k] for k in keep]
        trees = [cKDTree(c["pts"]) for c in cs]
        near = lambda k, X: np.array([seg_dist(cs[k]["pts"], trees[k], x) for x in X])
        overlap = set()
        for a_ in range(len(cs)):
            for b_ in range(len(cs)):
                if a_ != b_:
                    X = cs[b_]["pts"][1:-1][::5]
                    # (the same exact curve, its points elsewhere along it: within the chords' sag)
                    if len(X) and (near(a_, X) < 2e-5).sum() >= max(2, len(X) // 4):
                        overlap |= {(a_, b_), (b_, a_)}
        cuts = defaultdict(list)
        for a_ in range(len(cs)):                         # ends lying on other curves
            for x in (cs[a_]["pts"][0], cs[a_]["pts"][-1]):
                for b_ in range(len(cs)):
                    if b_ != a_ and seg_dist(cs[b_]["pts"], trees[b_], x) < 3e-5:   # (within the chords' sag)
                        cuts[b_].append(x)
        segs, owner = [], []
        for ci, c in enumerate(cs):
            P = c["pts"]
            for j in range(len(P) - 1):
                m = max(1, int(np.ceil(np.linalg.norm(P[j + 1] - P[j]) / (2 * STEP))))
                for q in range(m):          # (a long step looked at in short pieces)
                    segs.append((P[j] + (P[j + 1] - P[j]) * q / m, P[j] + (P[j + 1] - P[j]) * (q + 1) / m))
                    owner.append(ci)
        mids = np.array([(a + b) / 2 for a, b in segs])
        half = np.array([np.linalg.norm(b - a) / 2 for a, b in segs])
        tree = cKDTree(mids)
        for i, j in tree.query_pairs(2 * half.max() + 1e-6):
            ci, cj = owner[i], owner[j]
            if ci == cj or (ci, cj) in overlap:
                continue
            a, b = segs[i]
            c_, d = segs[j]
            u, v, w = b - a, d - c_, a - c_
            A, B, Cc, Dd, E = u @ u, u @ v, v @ v, u @ w, v @ w
            den = A * Cc - B * B
            if den < 1e-6 * A * Cc:                      # (nearly parallel: not a crossing)
                continue
            t1 = np.clip((B * E - Cc * Dd) / den, 0, 1)
            t2 = np.clip((A * E - B * Dd) / den, 0, 1)
            if np.linalg.norm((a + t1 * u) - (c_ + t2 * v)) > 1e-4:
                continue
            x0 = (a + t1 * u + c_ + t2 * v) / 2
            x, ok = solve([self.s, cs[ci]["eqs"][1], cs[cj]["eqs"][1]], x0)
            tol_on = 1e-6
            if not ok or np.linalg.norm(x - x0) > 5e-3:
                tol_on = 3e-5
                # (the three surfaces meeting at a small angle: the crossing found along the first
                # curve instead, where the second's own surface changes sign, each point on the first)
                x, ok = _cross_along(self.s, cs[ci]["eqs"][1], cs[cj]["eqs"][1], a, b)
                if not ok:
                    continue
            # (on both curves: within their polylines' chord sag)
            if max(seg_dist(cs[ci]["pts"], trees[ci], x), seg_dist(cs[cj]["pts"], trees[cj], x)) > tol_on:
                continue
            cuts[ci].append(x)
            cuts[cj].append(x)
        self._overlap = overlap
        self._cuts = cuts
        for ci, c in enumerate(cs):
            P = c["pts"]
            pts = [P[0], P[-1]] + cuts.get(ci, [])
            # where each cut falls along the curve
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            at = []
            for x in pts:
                d = np.linalg.norm(P - x, axis=1)
                j = int(np.argmin(d))
                # refine onto the segment
                best = (d[j], cum[j], j, 0.0)
                for jj in (j - 1, j):
                    if 0 <= jj < len(P) - 1:
                        a, b = P[jj], P[jj + 1]
                        L2 = (b - a) @ (b - a)
                        t = 0 if L2 == 0 else float(np.clip((x - a) @ (b - a) / L2, 0, 1))
                        dd = np.linalg.norm(a + t * (b - a) - x)
                        if dd < best[0]:
                            best = (dd, cum[jj] + t * np.sqrt(L2), jj, t)
                at.append((best[1], x))
            at.sort(key=lambda z: z[0])
            stops = []
            for sarc, x in at:
                if stops and abs(sarc - stops[-1][0]) < 1e-9:
                    continue
                stops.append((sarc, x))
            for (s0, x0), (s1, x1) in zip(stops[:-1], stops[1:]):
                inner = P[(cum > s0 + 1e-9) & (cum < s1 - 1e-9)]
                pts_ = np.vstack([x0, inner, x1]) if len(inner) else np.vstack([x0, x1])
                if len(pts_) < 2 or s1 - s0 < 1e-9:
                    continue
                a, b = self.vertex(x0), self.vertex(x1)
                self.E.append({"a": a, "b": b, "pts": pts_, "curve": ci})
        # one edge per stretch: a stretch two curves share kept once, as the named line, else
        # the meeting curve, else the half-cell's boundary
        rank = {"line": 0, "meet": 1, "cross": 2, "face": 3, "aux": 4}
        best = {}
        for e in self.E:
            if e["a"] == e["b"] and len(e["pts"]) < 4:
                continue
            m = e["pts"][len(e["pts"]) // 2]
            key = (min(e["a"], e["b"]), max(e["a"], e["b"]))
            for k2, e2 in list(best.items()):
                if k2[0] == key and (seg_dist(e2["pts"], None, m) < 1e-6):
                    if rank[cs[e["curve"]]["kind"]] < rank[cs[e2["curve"]]["kind"]]:
                        best[k2] = e
                    break
            else:
                best[(key, len(best))] = e
        self.E = list(best.values())

    def split_cone_points(self):
        """A vertex where the surface comes to a point (its gradient 0: the tip of a cone) is two
        points of the surface, one on each of the cone's two sheets: the edges leaving it are
        shared between two vertices by the side of the cone's axis they leave on."""
        Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
        for v in range(len(self.V)):
            x = self.V[v]
            g = self.normal(x)
            if np.linalg.norm(g) > 1e-7:
                continue
            # sheets of the surface through the point (two smooth sheets crossing there, or a plane
            # the surface contains): the edges grouped by the surface's normal a short way along
            ends_n = []
            for ei, e in enumerate(self.E):
                for P, k in ((e["pts"], "a"), (e["pts"][::-1], "b")):
                    if e[k] != v:
                        continue
                    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                    y = P[min(int(np.searchsorted(cum, min(2e-3, cum[-1] / 3))), len(P) - 1)]
                    m = self.normal(y)
                    if np.linalg.norm(m) > 1e-12:
                        ends_n.append((ei, k, m / np.linalg.norm(m)))
            groups = []
            for ei, k, m in ends_n:
                for gr in groups:
                    if abs(gr[0][2] @ m) > 0.95:
                        gr.append((ei, k, m))
                        break
                else:
                    groups.append([(ei, k, m)])
            if len(groups) >= 2:
                big = [gr for gr in groups if len(gr) >= 2]
                for gr in groups:
                    if len(gr) < 2 and big:
                        best = max(big, key=lambda b: abs(b[0][2] @ gr[0][2]))
                        best += gr
                if len(big) >= 2:
                    for gr in big[1:]:
                        self.V.append(x.copy())
                        nv = len(self.V) - 1
                        for ei, k, _ in gr:
                            self.E[ei][k] = nv
                    self.cone_points = getattr(self, "cone_points", []) + [v]
                    continue
            h = 1e-5
            H = np.array([(self.normal(x + h * b) - self.normal(x - h * b)) @ Bz.T / (2 * h) for b in Bz])
            w, U = np.linalg.eigh((H + H.T) / 2)
            if np.abs(w).min() < 1e-6 * np.abs(w).max():
                continue                              # (not a cone: a degenerate point)
            sg = np.sign(w)
            odd = [i for i in range(3) if (sg == sg[i]).sum() == 1]
            if not odd:
                continue
            axis = U[:, odd[0]] @ Bz
            ends = []
            for ei, e in enumerate(self.E):
                if e["a"] == v:
                    ends.append((ei, "a", e["pts"][min(2, len(e["pts"]) - 1)]))
                if e["b"] == v:
                    ends.append((ei, "b", e["pts"][max(-3, -len(e["pts"]))]))
            neg = [(ei, k) for ei, k, y in ends if (y - x) @ axis < 0]
            if 2 <= len(neg) <= len(ends) - 2:       # (each sheet through the tip: two edges at least)
                self.V.append(x.copy())
                nv = len(self.V) - 1
                for ei, k in neg:
                    self.E[ei][k] = nv
                self.cone_points = getattr(self, "cone_points", []) + [v]

    def snap_open_ends(self, r=1e-3):
        """A boundary curve's end left loose (tracing stopped just short of a singular corner or
        crossing) joined to the nearest vertex within r: its end point replaced by that vertex."""
        from collections import Counter
        deg = Counter()
        for e in self.E:
            deg[e["a"]] += 1
            deg[e["b"]] += 1
        for v in [v for v in deg if deg[v] == 1]:
            x = self.V[v]
            cands = [(np.linalg.norm(self.V[u] - x), u) for u in deg if u != v]
            if not cands:
                continue
            d, u = min(cands)
            if d > r:
                continue
            for e in self.E:
                for k, idx in (("a", 0), ("b", -1)):
                    if e[k] == v:
                        e[k] = u
                        e["pts"] = e["pts"].copy()
                        e["pts"][idx] = self.V[u]
            deg[u] += 1
            deg[v] = 0

    def faces(self):
        """Faces: half-edges followed round each vertex in the surface's tangent plane (each face
        on the left of its half-edges, turning by the surface's normal)."""
        self.snap_open_ends()
        self.split_cone_points()
        out = defaultdict(list)            # vertex -> [(angle, half-edge)]
        H = []                             # half-edges: (edge, forward)
        for ei, e in enumerate(self.E):
            for fwd in (True, False):
                H.append((ei, fwd))
        hid = {h: i for i, h in enumerate(H)}
        vn = {}

        def vnormal(v):
            """The surface's normal at vertex v (at a singular point, where the gradient is 0:
            the mean of the normals a short way along its edges)."""
            if v not in vn:
                n = self.normal(self.V[v])
                if np.linalg.norm(n) < 1e-9:
                    ns = []
                    for e in self.E:
                        for P in ((e["pts"],) if e["a"] == v else ()) + ((e["pts"][::-1],) if e["b"] == v else ()):
                            y = P[min(len(P) - 1, 2)]
                            m = self.normal(y)
                            if np.linalg.norm(m) > 1e-12:
                                ns.append(m / np.linalg.norm(m))
                    n = np.mean(ns, axis=0) if ns else np.array([1.0, -1, 0, 0])
                    if np.linalg.norm(n) < 1e-6:
                        n = ns[0]
                vn[v] = n / np.linalg.norm(n)
            return vn[v]
        for i, (ei, fwd) in enumerate(H):
            e = self.E[ei]
            v = e["a"] if fwd else e["b"]
            P = e["pts"] if fwd else e["pts"][::-1]
            x = self.V[v]
            n = vnormal(v)
            basis = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
            e1, e2 = basis
            # orientation: (e1, e2, n) right-handed in the sum-zero space
            if np.linalg.det(np.vstack([np.ones(4) / 2, e1, e2, n])) < 0:
                e2 = -e2
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            d = min(2e-2, cum[-1] / 3)        # (far enough along to tell near-tangent curves apart)
            j = int(np.searchsorted(cum, d))
            y = P[min(j, len(P) - 1)] - x
            out[v].append((np.arctan2(y @ e2, y @ e1), i))
        nxt = {}
        for v, lst in out.items():
            lst.sort()
            order = [h for _, h in lst]
            for k, h in enumerate(order):
                # arriving along twin(h) into v, leave by the half-edge just clockwise of h
                ei, fwd = H[h]
                twin = hid[(ei, not fwd)]
                nxt[twin] = order[(k - 1) % len(order)]
        seen = np.zeros(len(H), bool)
        self.F = []
        for h0 in range(len(H)):
            if seen[h0]:
                continue
            cyc, h = [], h0
            while not seen[h]:
                seen[h] = True
                cyc.append(H[h])
                h = nxt.get(h)
                if h is None:
                    break
            self.F.append(cyc)

    def loop(self, cyc, gap=2e-5):
        """A face's loop: its edges' exact points in order (the graph's vertices always kept;
        a curve point closer than gap to the point before it, or to the edge's end, left out)."""
        pts = []
        for ei, fwd in cyc:
            P = self.E[ei]["pts"] if fwd else self.E[ei]["pts"][::-1]
            keep = [P[0]]
            for x in P[1:-1]:
                if np.linalg.norm(x - keep[-1]) > gap and np.linalg.norm(x - P[-1]) > gap:
                    keep.append(x)
            pts += keep
        return np.array(pts)


# ------------------------------------------------------------------------------- faces

def _plane_frame(P, normal):
    """A plane for a loop on the surface: its centre, the mean surface normal, two directions."""
    c = P.mean(0)
    ns = [normal(x) for x in P]
    ns = [m / np.linalg.norm(m) for m in ns if np.linalg.norm(m) > 1e-12]
    if ns:
        n = np.mean(ns, axis=0)
    else:
        Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
        n = np.linalg.svd((P - c) @ Bz.T, full_matrices=False)[2][-1] @ Bz
    n /= np.linalg.norm(n)
    e1, e2 = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
    if np.linalg.det(np.vstack([np.ones(4) / 2, e1, e2, n])) < 0:
        e2 = -e2
    return c, n, e1, e2


def _area(Q):
    return 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))


def _lift(s, x, n, iters=30):
    """x moved along n onto the surface s (Newton on t: s(x + t n) = 0)."""
    t = 0.0
    for _ in range(iters):
        y = x + t * n
        f = float(s.f(y[None])[0])
        if abs(f) < 1e-15:
            break
        d = float(s.grad(y[None])[0] @ n)
        if abs(d) < 1e-14:
            return None
        t -= f / d
    y = x + t * n
    return y if abs(float(s.f(y[None])[0])) < 1e-12 and abs(t) < 0.2 else None


def name_point(D, key, x, n):
    """What the surface is at x: (wall, region on the + side, on the - side, their components)."""
    eps = 1e-7
    if isinstance(key, str):
        k = FACE_NAMES.index(key[5:])
        nin = FACE_N[k] - FACE_N[k].mean()
        nin /= np.linalg.norm(nin)
        a = D.piece(x + eps * nin)
        kk, beyond = locate(point_from_beta(x - eps * nin)[None], tol=1e-13)
        kk = int(kk[0])
        b = D.piece(beyond[0])
        if a is None or b is None:
            return None
        p = D.fw.get((a, b, kk)) or D.fw.get((b, a, int(D.inv[kk])))
        return {"wall": p["wall"] if p else None, "patch": p["id"] if p else None,
                "sides": [D.region[a], D.region[b]],
                "components": [int(D.rcomp[0, a]), int(D.rcomp[kk, b])]}
    a, b = D.piece(x + eps * n), D.piece(x - eps * n)
    if a is None or b is None:
        return None
    p = D.pw.get((min(a, b), max(a, b), key)) if a != b else None
    return {"wall": p["wall"] if p else None, "patch": p["id"] if p else None,
            "sides": [D.region[a], D.region[b]],
            "components": [int(D.rcomp[0, a]), int(D.rcomp[0, b])]}


def name_points(D, key, Y, N):
    """name_point at many points at once: [(wall, sides) or None]."""
    eps = 1e-7
    vals = lambda X: np.sign(np.stack([sf.f(X) for sf in D.lib], 1)).astype(np.int8)
    pk = lambda S: [D.keys.get(r.tobytes()) for r in S]
    if isinstance(key, str):
        k = FACE_NAMES.index(key[5:])
        nin = FACE_N[k] - FACE_N[k].mean()
        nin /= np.linalg.norm(nin)
        A = pk(vals(Y + eps * nin))
        kk, beyond = locate(point_from_beta(Y - eps * nin), tol=1e-13)
        B = pk(vals(beyond))
        out = []
        for a, b, c in zip(A, B, kk):
            if a is None or b is None:
                out.append(None)
                continue
            p = D.fw.get((a, b, int(c))) or D.fw.get((b, a, int(D.inv[int(c)])))
            out.append((p["wall"] if p else None, (D.region[a], D.region[b])))
        return out
    A, B = pk(vals(Y + eps * N)), pk(vals(Y - eps * N))
    out = []
    for a, b in zip(A, B):
        if a is None or b is None:
            out.append(None)
            continue
        p = D.pw.get((min(a, b), max(a, b), key)) if a != b else None
        out.append((p["wall"] if p else None, (D.region[a], D.region[b])))
    return out


def faces_of(D, key, G, max_area=6e-5, log=print):
    """Each face of the graph: its loops (outer, holes), its triangles inside them (loop points the
    only boundary vertices, inner points on the surface), and what it is."""
    import triangle as tr
    s = G.s
    loops = []
    for cyc in G.F:
        P = G.loop(cyc)
        if len(P) < 3:
            continue
        c, n, e1, e2 = _plane_frame(P, G.normal)
        Q = np.c_[(P - c) @ e1, (P - c) @ e2]
        loops.append({"cyc": cyc, "P": P, "frame": (c, n, e1, e2), "area": _area(Q)})
    # components of the graph: each has one outer loop (negative area); a component lying inside a
    # face of another is a hole of that face
    parent = list(range(len(G.V)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in G.E:
        parent[find(e["a"])] = find(e["b"])
    comp = lambda L: find(G.E[L["cyc"][0][0]]["a"])
    outer = {}
    for L in loops:
        if L["area"] < 0 and (comp(L) not in outer or L["area"] < outer[comp(L)]["area"]):
            outer[comp(L)] = L
    inner = [L for L in loops if L is not outer.get(comp(L))]
    holes = defaultdict(list)
    for co, O in outer.items():
        if len(outer) == 1:
            break
        x = O["P"][0]
        best = None
        for L in inner:
            if comp(L) == co:
                continue
            c, n, e1, e2 = L["frame"]
            Q = np.c_[(L["P"] - c) @ e1, (L["P"] - c) @ e2]
            q = np.array([(x - c) @ e1, (x - c) @ e2])
            if _inside_poly(Q, q) and (best is None or L["area"] < best["area"]):
                best = L
        if best is not None:
            holes[id(best)].append(O)
    out = []
    # (a loop walking an edge both ways is split there: a part turning the loop's way is a face of
    # its own, a part turning the other way a hole in the part round it; a curve reaching into the
    # face and ending there is left out)
    expanded = []
    for L in inner:
        parts_c = [c_ for c_ in _split_bridges(L["cyc"]) if c_]
        if len(parts_c) == 1 and len(parts_c[0]) == len(L["cyc"]):
            expanded.append(L)
            continue
        c0, n0, e10, e20 = L["frame"]
        pos, neg = [], []
        for c_ in parts_c:
            P_ = G.loop(c_)
            if len(P_) < 3:
                continue
            a_ = _area(np.c_[(P_ - c0) @ e10, (P_ - c0) @ e20])
            (pos if a_ > 0 else neg).append({"cyc": c_, "P": P_, "frame": _plane_frame(P_, G.normal), "area": a_})
        for N_ in neg:
            q = N_["P"][0]
            host = None
            for Pp in pos:
                cf, nf, f1, f2 = Pp["frame"]
                if _inside_poly(np.c_[(Pp["P"] - cf) @ f1, (Pp["P"] - cf) @ f2], np.array([(q - cf) @ f1, (q - cf) @ f2])):
                    host = Pp
                    break
            if host is not None:
                holes[id(host)].append(N_)
        for Pp in pos:
            holes[id(Pp)] += [] if Pp is not pos[0] else holes[id(L)]
        expanded += pos
    inner = expanded
    for L in inner:
        rings = [L["P"]] + [H["P"] for H in holes[id(L)]]
        # (a loop walking an edge both ways: a curve reaching into the face and ending there is
        # left out; one joining the outer loop to an inner one splits off that inner loop, a hole)
        rings = [L["P"]] + [H["P"] for H in holes[id(L)]]
        # first, what it is: at points a short way inside its loop (from the middle of each
        # edge, across the edge in the surface); a face of no wall is not meshed
        probe = []
        for ei, fwd in L["cyc"][:12]:
            Pe = G.E[ei]["pts"] if fwd else G.E[ei]["pts"][::-1]
            if len(Pe) < 3:
                continue
            k = len(Pe) // 2
            m, tng = Pe[k], Pe[k + 1] - Pe[k - 1]
            nn = G.normal(m)
            if np.linalg.norm(nn) < 1e-12 or np.linalg.norm(tng) < 1e-14:
                continue
            nn = nn / np.linalg.norm(nn)
            Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
            if np.linalg.det(np.vstack([np.ones(4) / 2, Bz])) < 0:
                Bz = Bz[[1, 0, 2]]
            left = np.cross(nn @ Bz.T, tng @ Bz.T) @ Bz          # (n x t: to the left of the edge)
            left /= np.linalg.norm(left)
            for dlt in (1e-3, 3e-3):
                y = _lift(s, m + dlt * left, nn)
                if y is not None:
                    gy = G.normal(y)
                    nm = name_point(D, key, y, gy / np.linalg.norm(gy))
                    if nm:
                        probe.append(nm)
        if probe and not any(x["wall"] for x in probe):
            continue
        # the plane the face is laid flat on: the first of these on which its loops do not cross
        # themselves and no triangle, put back on the surface, turns over: the mean normal, the
        # normals at points round the loop, the loop's least spread direction
        parts = triangulate_face(s, rings, G.normal, max_area, tr)
        if parts is None and len(rings) == 1:
            # (laying it flat failed even cut: triangulated on the surface itself)
            res = _ear_clip_on_surface(s, rings[0], G.normal)
            parts = [res] if res is not None else None
        if parts is None:
            log(f"  sheet {key}: a face of {len(L['P'])} loop points could not be laid flat, even cut")
            out.append({"sheet": key, "too_curved": True, "P": L["P"], "frame": L["frame"]})
            continue
        Xs, Fs, off_ = [], [], 0
        for Xp, Fp in parts:
            Xs.append(Xp)
            Fs.append(Fp + off_)
            off_ += len(Xp)
        X, F = np.vstack(Xs), np.vstack(Fs)
        bad = 0
        c, n, e1, e2 = L["frame"]
        flips = 0
        nrm = np.linalg.norm(_tri_normals(X, F), axis=1)
        # what it is: at the middle of its largest triangles, and at triangles all over it
        ar = np.abs(nrm)
        order = list(np.argsort(-ar)[:3])
        order += list(np.random.default_rng(0).choice(len(F), min(len(F), 30), replace=False))
        names, at = [], []
        for t_i in order:
            y = _lift(s, X[F[t_i]].mean(0), n)
            if y is None:
                continue
            g = G.normal(y)
            nm = name_point(D, key, y, g / np.linalg.norm(g))
            names.append(nm)
            at.append(y)
        named = [x for x in names if x]
        kinds = {(x["wall"], tuple(x["sides"])) for x in named}
        edges = []
        for ei, fwd in L["cyc"]:
            e = G.E[ei]
            cv = G.curves[e["curve"]]
            edges.append({"edge": ei, "kind": cv["kind"], "name": cv["name"],
                          "piece": cv.get("piece")})
        for H in holes[id(L)]:
            for ei, fwd in H["cyc"]:
                cv = G.curves[G.E[ei]["curve"]]
                edges.append({"edge": ei, "kind": cv["kind"], "name": cv["name"], "hole": True})
        tri_names, cen = None, None
        if any(x["wall"] for x in named):
            cen = X[F].mean(1)
            Nn = np.array([G.normal(y) for y in cen])
            Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
            # (centres moved onto the surface along the normal)
            lifted = [_lift(s, y, nn) for y, nn in zip(cen, Nn)]
            cen = np.array([y if l_ is None else l_ for y, l_ in zip(cen, lifted)])
            tri_names = name_points(D, key, cen, Nn)
            # (a triangle at the rim where no patch is found, its sides the same, is not a change)
            kinds = {x for x in tri_names if x is not None and x[0] is not None}
        out.append({"sheet": key, "rings": rings, "what": named[0] if named else None, "agree": len(kinds) <= 1,
                    "tri_names": tri_names, "centres": cen, "F": F,
                    "samples": [(y, nm) for y, nm in zip(at, names) if nm],
                    "names": named, "edges": edges, "verts": X, "tris": F, "flips": flips,
                    "lift_failed": bad, "area2d": L["area"]})
    return out


def _normals_to_try(P, normal):
    ns = []
    m = [normal(x) for x in P[:: max(1, len(P) // 60)]]
    m = [v / np.linalg.norm(v) for v in m if np.linalg.norm(v) > 1e-12]
    if m:
        ns.append(np.mean(m, axis=0))
    ns += m[:: max(1, len(m) // 8)]
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    Z = (P - P.mean(0)) @ Bz.T
    ns.append(np.linalg.svd(Z, full_matrices=False)[2][-1] @ Bz)
    # (and the coordinate directions: a face folded over its own normals may lie flat along one)
    E4 = np.eye(4)
    for i in range(4):
        ns.append(E4[i] - 0.25)
        for j in range(i + 1, 4):
            ns.append(E4[i] - E4[j])
    out = []
    for v in ns:
        v = v - v.mean()
        if np.linalg.norm(v) > 1e-9:
            out.append(v / np.linalg.norm(v))
    return out


def _triangulate(s, rings, c, n, normal, max_area, tr):
    """The face laid flat on the plane through c across n, triangulated inside its loops (the
    loop points the only rim vertices), inner vertices moved along n onto the surface. None if
    its loops cross themselves there or a triangle turns over once on the surface."""
    e1, e2 = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
    if np.linalg.det(np.vstack([np.ones(4) / 2, e1, e2, n])) < 0:
        e2 = -e2
    flat = [np.c_[(R - c) @ e1, (R - c) @ e2] for R in rings]
    if _area(flat[0]) <= 0 or sum(_crossings(Q) for Q in flat):
        return None
    verts, segs, hole_pts, off = [], [], [], 0
    for r_i, Q in enumerate(flat):
        verts.append(Q)
        m = len(Q)
        segs += [[off + i, off + (i + 1) % m] for i in range(m)]
        off += m
        if r_i:
            t = tr.triangulate({"vertices": Q, "segments": [[i, (i + 1) % m] for i in range(m)]}, "p")
            hole_pts.append(Q[t["triangles"][0]].mean(0))
    Vall = np.vstack(verts)
    uniq, inv_ = np.unique(np.round(Vall, 13), axis=0, return_inverse=True)
    inv_ = inv_.ravel()
    rep = np.zeros(len(uniq), int)
    for i in range(len(Vall) - 1, -1, -1):
        rep[inv_[i]] = i
    segs = np.array(sorted({(min(int(inv_[a]), int(inv_[b])), max(int(inv_[a]), int(inv_[b])))
                            for a, b in segs if inv_[a] != inv_[b]}), dtype=np.int32)
    A = {"vertices": np.ascontiguousarray(Vall[rep], dtype=np.float64), "segments": np.ascontiguousarray(segs)}
    if hole_pts:
        A["holes"] = np.ascontiguousarray(hole_pts, dtype=np.float64)
    T = tr.triangulate(A, f"pYq20a{max_area:.3g}")
    if "triangles" not in T or not len(T["triangles"]):
        return None
    res = _lift_all(s, T, rep, rings, c, n, e1, e2, normal)
    if res is not None:
        return res
    # (inner points that cannot all be put on the surface, next to a singular point: the loop's own
    # points only, then inner edges split onto the surface where they can be)
    T = tr.triangulate(A, "pY")
    if "triangles" not in T or not len(T["triangles"]) or len(T["vertices"]) != len(rep):
        return None
    X0 = np.vstack(rings)[rep]
    ring_ids = []
    off = 0
    rim = set()
    seg = A["segments"]
    for a_, b_ in seg:
        rim.add(frozenset((int(a_), int(b_))))
    X, F = _refine_on_surface(s, [x for x in X0], [list(t) for t in T["triangles"]], rim, normal)
    X, F = np.array(X), np.array(F)
    if _turned(_tri_normals(X, F), np.array([normal(x) for x in X[F].mean(1)])):
        return None
    return X, F, 0, (c, n, e1, e2)


def _lift_all(s, T, rep, rings, c, n, e1, e2, normal):
    """The triangulation's inner points moved along n onto the surface; None if one cannot be, or
    a triangle then faces against the surface."""
    nb = len(rep)
    X = np.empty((len(T["vertices"]), 4))
    X[:nb] = np.vstack(rings)[rep]
    bad = 0
    for i in range(nb, len(X)):
        u, v = T["vertices"][i]
        y = _lift(s, c + u * e1 + v * e2, n)
        if y is None:
            return None
        X[i] = y
    F = T["triangles"]
    # (each triangle, on the surface, facing the way the surface does)
    cr = _tri_normals(X, F)
    sn = np.array([normal(x) for x in X[F].mean(1)])
    if _turned(cr, sn):
        return None
    return X, F, bad, (c, n, e1, e2)


def _inward_cut(s, t, P, i, xs_pts, normal, step=STEP, max_len=3.0):
    """From the loop's crossing i with the plane t, the exact curve s = t = 0 traced into the face
    (to the left of the loop there) up to the first other crossing it reaches: (path, j) or None."""
    fs, ft = _fg(s), _fg(t)
    n = len(P)
    x0 = xs_pts[i]
    k = list(xs_pts).index(x0) if False else None
    nn = normal(x0)
    if np.linalg.norm(nn) < 1e-12:
        return None
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if np.linalg.det(np.vstack([np.ones(4) / 2, Bz])) < 0:
        Bz = Bz[[1, 0, 2]]
    a_idx = xs_idx_of[id(xs_pts)][i]
    tng = P[(a_idx + 1) % n] - P[a_idx]
    left = np.cross(nn @ Bz.T, tng @ Bz.T) @ Bz
    tan = np.linalg.svd(np.vstack([np.ones(4), fs(x0)[1], ft(x0)[1]]))[2][-1]
    if tan @ left < 0:
        tan = -tan
    pts, x, tp = [x0], x0, tan
    others = [(k2, y) for k2, y in enumerate(xs_pts) if k2 != i]
    for _ in range(int(max_len / step)):
        tan = np.linalg.svd(np.vstack([np.ones(4), fs(x)[1], ft(x)[1]]))[2][-1]
        tan = tan if tan @ tp > 0 else -tan
        tp = tan
        y, ok = solve([s, t], x + step * tan / np.linalg.norm(tan))
        if not ok or not inside(y, 1e-9):
            return None
        for k2, z in others:
            if np.linalg.norm(y - z) < 1.5 * step:
                return np.array(pts + [z]), k2
        pts.append(y)
        x = y
    return None


xs_idx_of = {}


def _refine_on_surface(s, X, F, rim, normal, h=0.02, rounds=8):
    """Inner edges longer than h split at their middles, each solved onto the surface (a middle
    that cannot be: left); rim edges never split."""
    for _ in range(rounds):
        edges = {}
        for t_i, t in enumerate(F):
            for k in range(3):
                e = frozenset((t[k], t[(k + 1) % 3]))
                edges.setdefault(e, []).append(t_i)
        long_ = [e for e, ts in edges.items() if e not in rim and len(ts) == 2
                 and np.linalg.norm(X[min(e)] - X[max(e)]) > h]
        if not long_:
            break
        taken, newF, split_at = set(), [], {}
        for e in sorted(long_, key=lambda e: -np.linalg.norm(X[min(e)] - X[max(e)])):
            t1, t2 = edges[e]
            if t1 in taken or t2 in taken:
                continue
            u, v = tuple(e)
            m = (X[u] + X[v]) / 2
            # (moved onto the surface across the two triangles' own mean normal, no further than half
            # the edge; kept only if the four new triangles all face the way the surface does)
            tn = _tri_normals(np.array(X), np.array([F[t1], F[t2]])).sum(0)
            if np.linalg.norm(tn) < 1e-14:
                continue
            y = _lift(s, m, tn / np.linalg.norm(tn))
            if y is None or np.linalg.norm(y - m) > 0.5 * np.linalg.norm(X[u] - X[v]):
                continue
            newt = []
            for t_ in (F[t1], F[t2]):
                k_ = next(k for k in range(3) if frozenset((t_[k], t_[(k + 1) % 3])) == e)
                a_, b_, c_ = t_[k_], t_[(k_ + 1) % 3], t_[(k_ + 2) % 3]
                newt += [[a_, -1, c_], [-1, b_, c_]]
            Xt = np.array(X + [y])
            Ft = np.array([[len(X) if q == -1 else q for q in t_] for t_ in newt])
            sn = np.array([normal(x) for x in Xt[Ft].mean(1)])
            if _turned(_tri_normals(Xt, Ft), sn):
                continue
            X.append(y)
            split_at[e] = len(X) - 1
            taken |= {t1, t2}
        for t_i, t in enumerate(F):
            if t_i not in taken:
                newF.append(t)
                continue
            for k in range(3):
                e = frozenset((t[k], t[(k + 1) % 3]))
                if e in split_at:
                    m_ = split_at[e]
                    a_, b_, c_ = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                    newF += [[a_, m_, c_], [m_, b_, c_]]
                    break
        F = newF
    return X, F


def _ear_clip_on_surface(s, P, normal, h=0.02, rounds=8):
    """A loop on the surface triangulated without laying it flat: ears clipped one by one, each
    checked in the surface's tangent plane at its tip (turning the way the surface does, no other
    loop point inside); then inner edges longer than h split at their middles, each solved onto the
    surface (rim edges never split). None if a triangle ends up facing against the surface."""
    from scipy.spatial import cKDTree
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if np.linalg.det(np.vstack([np.ones(4) / 2, Bz])) < 0:
        Bz = Bz[[1, 0, 2]]
    X3 = P @ Bz.T
    n = len(P)
    if n < 3:
        return None
    N3 = np.array([normal(x) @ Bz.T for x in P])
    N3 /= np.linalg.norm(N3, axis=1, keepdims=True) + 1e-300
    tree = cKDTree(X3)
    alive = list(range(n))
    nxt = {alive[i]: alive[(i + 1) % n] for i in range(n)}
    prv = {alive[i]: alive[i - 1] for i in range(n)}
    active = set(alive)
    tris = []

    def ear_ok(i):
        a, b, c = prv[i], i, nxt[i]
        t = np.cross(X3[b] - X3[a], X3[c] - X3[b])
        if t @ N3[b] <= 0:
            return None
        nb = N3[b]
        e1 = X3[c] - X3[b]
        e1 -= (e1 @ nb) * nb
        if np.linalg.norm(e1) < 1e-14:
            return None
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nb, e1)
        pr = lambda y: np.array([(y - X3[b]) @ e1, (y - X3[b]) @ e2])
        A, B_, Cc = pr(X3[a]), pr(X3[b]), pr(X3[c])
        r = max(np.linalg.norm(X3[a] - X3[b]), np.linalg.norm(X3[c] - X3[b]), np.linalg.norm(X3[a] - X3[c]))
        for j in tree.query_ball_point(X3[b], r * 1.01):
            if j in (a, b, c) or j not in active:
                continue
            q = pr(X3[j])
            d1 = (B_[0] - A[0]) * (q[1] - A[1]) - (B_[1] - A[1]) * (q[0] - A[0])
            d2 = (Cc[0] - B_[0]) * (q[1] - B_[1]) - (Cc[1] - B_[1]) * (q[0] - B_[0])
            d3 = (A[0] - Cc[0]) * (q[1] - Cc[1]) - (A[1] - Cc[1]) * (q[0] - Cc[0])
            if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
                return None
        # quality: the smallest angle
        def ang(u, v):
            return np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300), -1, 1))
        return min(ang(X3[a] - X3[b], X3[c] - X3[b]), ang(X3[b] - X3[a], X3[c] - X3[a]),
                   ang(X3[a] - X3[c], X3[b] - X3[c]))
    stall = 0
    i = alive[0]
    while len(active) > 3:
        q = ear_ok(i)
        if q is not None:
            a, c = prv[i], nxt[i]
            tris.append([a, i, c])
            active.discard(i)
            nxt[a], prv[c] = c, a
            stall = 0
            i = c
        else:
            i = nxt[i]
            stall += 1
            if stall > len(active) + 2:
                return None
    rest = list(active)
    a = rest[0]
    tris.append([prv[nxt[a]], nxt[a], nxt[nxt[a]]] if False else [a, nxt[a], nxt[nxt[a]]])
    X, F = _refine_on_surface(s, [p for p in P], [list(t) for t in tris],
                              {frozenset((k, (k + 1) % n)) for k in range(n)}, normal, h, rounds)
    X = np.array(X)
    F = np.array(F)
    cr = _tri_normals(X, F)
    sn = np.array([normal(x) for x in X[F].mean(1)])
    if _turned(cr, sn):
        return None
    return X, F


def triangulate_face(s, rings, normal, max_area, tr, depth=0):
    """A face triangulated inside its exact loops: laid flat on a plane where that works; else cut
    in two along the exact curve where the surface meets a plane across it (from where that plane
    crosses its loop, into the face, to the next crossing; the cut shared by both halves, so they
    meet exactly), each half done the same way. Returns a list of (X, F) or None."""
    P = rings[0]
    for n_try in _normals_to_try(P, normal):
        res = _triangulate(s, rings, P.mean(0), n_try, normal, max_area, tr)
        if res is not None:
            return [res[:2]]
    if depth >= 6 or len(rings) > 1:
        return None
    n = len(P)
    for i0, i1 in ((0, n // 2), (n // 4, 3 * n // 4)):
        a_, b_ = P[i0], P[i1]
        d = (b_ - a_) - (b_ - a_).mean()
        cpt = (a_ + b_) / 2
        nrm = d - (d @ cpt) * np.ones(4)
        t = Plane(nrm, "helper cut")
        sv = P @ nrm
        idx = [int(k) for k in np.flatnonzero(np.sign(sv) != np.sign(np.roll(sv, -1)))]
        if len(idx) < 2:
            continue
        pts = []
        for k in idx:
            x, ok = solve([s, t], (P[k] + P[(k + 1) % n]) / 2)
            pts.append(x if ok else (P[k] + P[(k + 1) % n]) / 2)
        xs_idx_of[id(pts)] = idx
        res = _inward_cut(s, t, P, 0, pts, normal)
        xs_idx_of.pop(id(pts), None)
        if res is None:
            continue
        cut, jj = res
        i, j = idx[0], idx[jj]
        xi, xj = pts[0], pts[jj]
        inner = list(cut[1:-1])
        L1 = np.array([xi] + list(P[i + 1:j + 1]) + [xj] + inner[::-1])
        L2 = np.array([xj] + list(P[j + 1:]) + list(P[:i + 1]) + [xi] + inner)
        out = []
        for Lh in (L1, L2):
            part = triangulate_face(s, [Lh], normal, max_area, tr, depth + 1)
            if part is None:
                return None
            out += part
        return out
    return None


def _turned(cr, sn):
    """Whether a triangle faces against the surface (where the surface's normal is too small to
    tell, next to a singular point, it is not counted)."""
    m = np.linalg.norm(sn, axis=1)
    ok = m > 0.05 * np.median(m)
    return bool(((cr * sn).sum(1)[ok] <= 0).any())


def _tri_normals(X, F):
    """Normals (in the sum-zero 3-space, as 4-vectors) of triangles of points with sum 1."""
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    if np.linalg.det(np.vstack([np.ones(4) / 2, Bz])) < 0:
        Bz = Bz[[1, 0, 2]]
    a, b = (X[F[:, 1]] - X[F[:, 0]]) @ Bz.T, (X[F[:, 2]] - X[F[:, 0]]) @ Bz.T
    return np.cross(a, b) @ Bz


def _split_bridges(cyc):
    """A face's cycle of half-edges with any edge walked both ways removed: the cycle split there
    into the parts on either side (an empty part: the edge was a spur)."""
    out, todo = [], [list(cyc)]
    while todo:
        c = todo.pop()
        pos = {}
        hit = None
        for k, (ei, fwd) in enumerate(c):
            if ei in pos:
                hit = (pos[ei], k)
                break
            pos[ei] = k
        if hit is None:
            out.append(c)
            continue
        i, j = hit
        a, b = c[j + 1:] + c[:i], c[i + 1:j]
        for part in (a, b):
            if part:
                todo.append(part)
    return out or [[]]


def _crossings(Q):
    """How many pairs of a closed polyline's segments cross (not counting shared ends)."""
    from scipy.spatial import cKDTree
    n = len(Q)
    A, B = Q, np.roll(Q, -1, axis=0)
    mids = (A + B) / 2
    L = np.linalg.norm(B - A, axis=1).max()
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    cnt = 0
    for i, j in cKDTree(mids).query_pairs(L + 1e-12):
        if abs(i - j) <= 1 or abs(i - j) == n - 1:
            continue
        a, b, c_, e = A[i], B[i], A[j], B[j]
        if len({tuple(np.round(x, 12)) for x in (a, b, c_, e)}) < 4:
            continue
        if d(a, b, c_) * d(a, b, e) < 0 and d(c_, e, a) * d(c_, e, b) < 0:
            cnt += 1
    return cnt


def _inside_poly(Q, q):
    x, y = q
    inside_ = False
    for (x1, y1), (x2, y2) in zip(Q, np.roll(Q, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside_ = not inside_
    return inside_


# ------------------------------------------------------------------------------- output

def coarse(D, key, rings, gap=0.02, max_area=1.2e-3):
    """A coarser triangulation of a face: its loops' points kept every gap or so (their ends and
    sharp turns always), larger triangles inside, inner points on the surface."""
    import triangle as tr
    s = D.face_planes[FACE_NAMES.index(key[5:])] if isinstance(key, str) else D.lib[key]
    if isinstance(key, str):
        k = FACE_NAMES.index(key[5:])
        normal = lambda x: FACE_N[k] - FACE_N[k].mean()
    else:
        normal = lambda x: (lambda g: g - g.mean())(s.grad(x[None])[0])
    thin = []
    for R in rings:
        keep = [0]
        for i in range(1, len(R)):
            if np.linalg.norm(R[i] - R[keep[-1]]) >= gap:
                keep.append(i)
            elif i + 1 < len(R):
                a, b = R[i] - R[i - 1], R[i + 1] - R[i]
                if a @ b < 0.9 * np.linalg.norm(a) * np.linalg.norm(b):
                    keep.append(i)
        if len(keep) < 3:
            keep = list(range(len(R)))
        thin.append(R[keep])
    parts = triangulate_face(s, thin, normal, max_area, tr)
    if parts is None:
        return None
    Xs, Fs, off = [], [], 0
    for X, F in parts:
        Xs.append(X)
        Fs.append(F + off)
        off += len(X)
    return np.vstack(Xs), np.vstack(Fs)


def _sheet_job(key):
    import pickle
    cache = data_dir() / "sphere_map" / ".cache" / "domain_sheets" / f"{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    D = Data()
    C = pickle.loads((D.root / ".cache" / "domain_curves.pkl").read_bytes())
    try:
        G, F = do_sheet(D, C, key, wall_surfaces(D), log=lambda *a: None)
    except Exception as ex:                                  # noqa: BLE001
        import traceback
        return key, None, traceback.format_exc()
    G.s = G.normal = None                      # (sent back: points and names only)
    G.curves = [{k: v for k, v in c.items() if k != "eqs"} for c in G.curves]
    for f in F:
        f["samples"] = [(y, nm) for y, nm in f["samples"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps((key, G, F)))
    return key, G, F


def export(log=print, workers=4):
    """Build every sheet and write domain_pieces.json / .npz."""
    import pickle
    from collections import Counter
    from multiprocessing import Pool
    D = Data()
    C = curves(D, log)
    (D.root / ".cache").mkdir(exist_ok=True)
    (D.root / ".cache" / "domain_curves.pkl").write_bytes(pickle.dumps(C))
    keys = sorted(sheets(D), key=str)
    results, failed = [], []
    with Pool(workers) as pool:
        for key, G, F in pool.imap_unordered(_sheet_job, keys):
            if G is None:
                log(f"sheet {key}: FAILED\n{F}")
                failed.append(key)
                continue
            results.append((key, G, F))
            log(f"sheet {key}: {len(F)} faces, {sum(1 for f in F if f['what'] and f['what']['wall'])} "
                f"of walls ({len(results)}/{len(keys)})")
    results.sort(key=lambda r: str(r[0]))
    faces, edges, V, T, E = [], [], [], [], []
    CV, CT, cv_n, ct_n = [], [], 0, 0
    nv = 0
    stats = Counter()
    for key, G, F in results:
        e_off = len(edges)
        for e in G.E:
            cv = G.curves[e["curve"]]
            edges.append({"sheet": str(key), "kind": cv["kind"], "name": cv["name"],
                          "line_piece": cv.get("piece"), "glued": cv.get("glued"),
                          "surface": cv.get("surface"), "n": len(e["pts"])})
            E.append(e["pts"])
        for f in F:
            tn = [x for x in (f.get("tri_names") or []) if x is not None and x[0] is not None]
            if not tn:
                stats["faces of no wall"] += 1
                continue
            (wall, sides), n_ = Counter(tn).most_common(1)[0]
            stats["wall faces"] += 1
            stats["named the same all through"] += f["agree"]
            comp = None
            for y, nm in f["samples"]:
                if nm["wall"] == wall and tuple(nm["sides"]) == tuple(sides):
                    comp = nm["components"]
                    break
            faces.append({"id": len(faces), "sheet": str(key), "wall": wall, "sides": list(sides),
                          "components": comp, "agree": bool(f["agree"]),
                          "share_of_triangles_named_so": round(n_ / max(1, len(f["tri_names"])), 4),
                          "edges": [e_off + x["edge"] for x in f["edges"]],
                          "hole_edges": [e_off + x["edge"] for x in f["edges"] if x.get("hole")],
                          "verts": [nv, nv + len(f["verts"])], "tris": [sum(len(t) for t in T), sum(len(t) for t in T) + len(f["tris"])],
                          "turned_triangles": f["flips"], "lift_failed": f["lift_failed"]})
            V.append(f["verts"])
            T.append(f["tris"] + 0)
            nv += len(f["verts"])
            # (a coarser mesh of the same face, for views holding many images of the half-cell:
            # its loops' own points, every few of them, and bigger triangles inside)
            cr = coarse(D, key, f["rings"])
            if cr is None:
                stats["coarse mesh: the fine one used"] += 1
                cr = (f["verts"], f["tris"])
            faces[-1]["coarse_verts"] = [cv_n, cv_n + len(cr[0])]
            faces[-1]["coarse_tris"] = [ct_n, ct_n + len(cr[1])]
            CV.append(cr[0])
            CT.append(cr[1])
            cv_n += len(cr[0])
            ct_n += len(cr[1])
    lens = np.array([len(x) for x in E])
    np.savez_compressed(D.root / "domain_pieces.npz", verts=np.concatenate(V), tris=np.concatenate(T).astype(np.int32),
                        edge_pts=np.concatenate(E), edge_len=lens.astype(np.int32),
                        coarse_verts=np.concatenate(CV), coarse_tris=np.concatenate(CT).astype(np.int32))
    doc = {"_about": __doc__ + "\nnpz: verts (betas), tris (indices into each face's own verts range), "
           "edge_pts / edge_len (each edge's exact points, edges in order); coarse_verts / coarse_tris "
           "(each face's coarser mesh: its loop points every 0.02 or so, larger triangles).",
           "faces": faces, "edges": edges, "stats": dict(stats), "sheets_failed": [str(k) for k in failed]}
    (D.root / "domain_pieces.json").write_text(json.dumps(doc, indent=1))
    log(f"{len(faces)} wall faces, {sum(f['tris'][1] - f['tris'][0] for f in faces)} triangles, "
        f"{len(edges)} edges; {dict(stats)}")
    return doc


if __name__ == "__main__":
    export()
