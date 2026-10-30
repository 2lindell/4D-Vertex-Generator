"""Each wall's faces in the half-cell from its exact edges, and the edges that bound them (wall_faces.json).

For each wall, from the working data (working_data.json, its "exact" section) alone:
  * its edge stretches: every copy in the half-cell of each of its edge lines, split only at its own corners, kept from
    corner to corner (or to a face of the half-cell); and its edges along other walls, the exact curves of
    exact_curves.py;
  * the copies of its surfaces each stretch lies on, found exactly (every image of each of its library surfaces, the
    stretch tested at points inside it): not taken from a list;
  * on each such copy, its stretches and the copy's own rim on the half-cell's faces laid out flat (a plane in its own
    coordinates, a curved surface along its mean normal), and the faces they enclose;
  * the wall's faces: those holding an image of its seed;
  * its edges along another wall: where every copy of that wall's surface crosses each copy of its own, from corner to
    corner, where that wall is there (in one of its own faces, computed first);
  * its drawn edges: the stretches with one of its faces on one side only; its cuts (dashed): the stretches of the
    edges other walls end along on it that run through its faces, the wall on both sides. A stretch with its faces on neither side
    is a copy of the line where the wall is not, and is not drawn.
Checked: every listed edge drawn somewhere, and every corner inside the half-cell met by two drawn edges.

    python wall_faces.py [walls]
"""
from __future__ import annotations

import json
import sys

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

import arrangement as ar
import core as osd
import lines_page as lp
from domain_page import line_in_cell
from exact_page import in_cell

OUT = "wall_faces.json"
E1_, E2_, E3_, M_ = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2], np.array([0, 0, .5, .5])
FACES = {"b4=0": (np.eye(4)[3], (E1_, E2_, E3_)), "b3=b4": (np.array([0, 0, 1, -1.0]), (E1_, E2_, M_)),
         "b1=0": (np.eye(4)[0], (E2_, E3_, M_)), "b2=0": (np.eye(4)[1], (E1_, E3_, M_))}


def dist(S, X):
    X = np.atleast_2d(X)
    return np.abs(ar.fvals(S, X)) / (np.linalg.norm(ar.grads(S, X), axis=1) + 1e-300)


def inner(P):
    return P[sorted({len(P) // 4, len(P) // 2, (3 * len(P)) // 4})]


def copies_of(bases, P):
    """The copies (library name, symmetry) of the wall's surfaces the stretch P lies on."""
    out = []
    for n in bases:
        ok = np.ones(len(ar.GINV), bool)
        for x in inner(P):
            ok &= dist(ar.LIB[n], np.einsum("gij,j->gi", ar.GINV, x)) < 1e-7
        seen = set()
        for g in np.flatnonzero(ok):                       # (one per distinct surface: many symmetries give the same)
            k = ar.sheet_key(ar.Img(ar.LIB[n], int(g)))
            if k not in seen:
                seen.add(k)
                out.append((n, int(g)))
    return out


def face_curves(S, n=160):
    """Where the surface S meets the half-cell's faces (its rim in the half-cell), each point solved onto S and the face;
    a surface that is itself one of the faces: that face's triangle."""
    out = []
    for name, (nf, (A, B, C)) in FACES.items():
        if dist(S, np.array([A, B, C, (A + B + C) / 3])).max() < 1e-9:
            out += [np.array([A, B]), np.array([B, C]), np.array([C, A])]
            continue
        I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        ok = I + J <= n
        idx = -np.ones((n + 1, n + 1), int)
        idx[ok] = np.arange(ok.sum())
        P = A + (I[ok, None] / n) * (B - A) + (J[ok, None] / n) * (C - A)
        F = ar.fvals(S, P)
        F[F == 0] = 1e-300
        adj = {}
        for i in range(n):
            for j in range(n - i):
                tl = [(idx[i, j], idx[i + 1, j], idx[i, j + 1])]
                if i + j + 1 < n:
                    tl.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
                for t in tl:
                    ks = [tuple(sorted((t[a], t[b]))) for a, b in ((0, 1), (1, 2), (2, 0))
                          if (F[t[a]] > 0) != (F[t[b]] > 0)]
                    if len(ks) == 2:
                        adj.setdefault(ks[0], []).append(ks[1])
                        adj.setdefault(ks[1], []).append(ks[0])
        seen = set()
        fp = osd.Plane(nf, name)
        pt = lambda k: P[k[0]] + (P[k[1]] - P[k[0]]) * F[k[0]] / (F[k[0]] - F[k[1]])
        for k0 in adj:
            if k0 in seen:
                continue
            start, prev = k0, None
            while True:
                nx = [k for k in adj[start] if k != prev]
                if len(adj[start]) < 2 or not nx or nx[0] == k0:
                    break
                prev, start = start, nx[0]
            chain, prev, cur = [start], None, start
            seen.add(start)
            while True:
                nx = [k for k in adj[cur] if k != prev and k not in seen]
                if not nx:
                    if len(adj[cur]) == 2 and start in adj[cur] and len(chain) > 2:
                        chain.append(start)
                    break
                prev, cur = cur, nx[0]
                seen.add(cur)
                chain.append(cur)
            Q = ar.onto([S, fp], np.array([pt(k) for k in chain]))
            if len(Q) >= 2:
                out.append(Q)
    return out


def crossing_curves(sn, S, names):
    """Where every copy of the surfaces `names` crosses the surface S (named sn) in the half-cell: traced across S's
    scaffold (arrangement.mesh) and every point solved exactly onto both surfaces."""
    from arrangement_draw import chains
    ar.SURF.setdefault(sn, S)
    V, E, T = ar.mesh(sn)
    ins = ar.inside(V)
    own = ar.sheet_key(S)
    out, n0 = [], 0
    for yn in names:
        for g in ar.copies_crossing(ar.LIB[yn], V, ins):
            f = ar.Img(ar.LIB[yn], g)
            if ar.sheet_key(f) == own:
                continue
            sg = ar.fvals(f, V) > 0
            ch = sg[T] != sg[T[:, [1, 2, 0]]]
            rows = np.flatnonzero(ch.sum(1) == 2)
            if not len(rows):
                continue
            segs = []
            for r in rows:
                e = [tuple(sorted((T[r, a], T[r, (a + 1) % 3]))) for a in range(3) if ch[r, a]]
                segs.append((e[0], e[1]))
            for c in chains(segs):
                keys = np.array(c)
                X = ar.crossing(V, keys, f, S)
                inside = (X @ ar.CELL.T >= -1e-9).all(1)
                cur = []
                for x, k in zip(X, inside):
                    if k:
                        cur.append(x)
                    elif cur:
                        if len(cur) >= 2:
                            out.append(np.array(cur))
                        cur = []
                if len(cur) >= 2:
                    out.append(np.array(cur))
                # (each end taken exactly to where the curve really ends: the exact point on both surfaces there, or the
                # face of the half-cell; the scaffold's last point is a step short of it)
            for i, P in enumerate(out[n0:], n0):
                out[i] = finish(P, S, f)
            n0 = len(out)
    # (one of each: copies of different library surfaces can be the same surface)
    uniq = []
    for P in out:
        m = P[len(P) // 2]
        if not any(np.linalg.norm(Q - m, axis=1).min() < 1e-7 for Q in uniq):
            uniq.append(P)
    return uniq


def finish(P, S, f):
    EP, EN = ar.exact_points()
    ends = []
    for x in (P[0], P[-1]):
        d = np.linalg.norm(EP - x, axis=1)
        near = [i for i in np.argsort(d)[:8] if d[i] < 2.5 * ar.H
                and dist(S, EP[i][None])[0] < 1e-9 and dist(f, EP[i][None])[0] < 1e-9]
        if near:
            ends.append(EP[near[0]])
            continue
        vals = x @ ar.CELL.T
        k = int(np.argmin(vals))
        if vals[k] < 2.5 * ar.H:
            y = ar.onto([S, f, osd.Plane(ar.CELL[k], "face")], x[None])[0]
            if np.linalg.norm(y - x) < 3 * ar.H and abs(y @ ar.CELL[k]) < 1e-12:
                ends.append(y)
                continue
        ends.append(None)
    Q = P
    if ends[0] is not None and np.linalg.norm(ends[0] - P[0]) > 1e-15:
        Q = np.vstack([ends[0][None], Q])
    if ends[1] is not None and np.linalg.norm(ends[1] - P[-1]) > 1e-15:
        Q = np.vstack([Q, ends[1][None]])
    return Q


def wall(w, work, curves, seeds, dbg=None, others=None):
    wd = work["walls"][w]
    XL = work["exact"]["lines"]
    bases = [n for n in work["exact"]["wall_surfaces"][w] if n in ar.LIB]
    corners = set(wd["points"])
    # (its edge stretches)
    st = []
    for L in wd["edges"]:
        if L in XL:
            for c in line_in_cell(XL[L]["points"]):
                for P, a, b in lp.split_at_points(np.asarray(c, float), corners):
                    if not a == b == "face" and {a, b} <= corners | {"face"}:
                        st.append((L, P, a, b))

    # (the copies of its surfaces they lie on)
    sheets = {}
    on_sheet = []
    for i, (L, P, a, b) in enumerate(st):
        for n, g in copies_of(bases, P):
            S = ar.Img(ar.LIB[n], g)
            k = ar.sheet_key(S)
            sheets.setdefault(k, {"S": S, "name": f"{n}@{g}", "st": []})["st"].append(i)
    # (its seed's images in the half-cell)
    sd = np.asarray(seeds[w], float)
    Y = ar.GG @ sd
    s_ = Y.sum(1)
    Y = Y[np.abs(s_) > 1e-12] / s_[np.abs(s_) > 1e-12, None]
    Y = Y[(Y @ ar.CELL.T >= -1e-12).all(1)]
    import shapely
    from scipy.spatial import cKDTree

    face_cache = {}

    def in_face(Y, names, fi):
        """Does some copy of Y's surfaces coincide with the half-cell's face fi?"""
        if (Y, fi) not in face_cache:
            tri = {0: (E2_, E3_, M_), 1: (E1_, E3_, M_), 2: (E1_, E2_, E3_), 3: (E1_, E2_, M_)}[fi]
            pts = np.array([(tri[0] + tri[1] + tri[2]) / 3, (2 * tri[0] + tri[1] + tri[2]) / 4,
                            (tri[0] + 2 * tri[1] + tri[2]) / 4])
            ok = False
            for n in names:
                good = np.ones(len(ar.GINV), bool)
                for x in pts:
                    good &= dist(ar.LIB[n], np.einsum("gij,j->gi", ar.GINV, x)) < 1e-9
                ok |= bool(good.any())
            face_cache[(Y, fi)] = ok
        return face_cache[(Y, fi)]

    loaded = {}

    def present(Y, Q):
        """Is the wall Y there, along the stretch Q (points inside it)? Each point carried into the half-cell by the
        group lands inside one of Y's own faces (as computed for Y); Y not computed yet: taken as there (strictly)."""
        if others is None or Y not in others or "faces" not in others[Y]:
            return True
        if Y not in loaded:
            from shapely.geometry import Polygon
            loaded[Y] = [(ar.surface(e["name"]), np.array(e["c0"]), np.array(e["U"]),
                          unary_union([Polygon(p_[0], p_[1:]) for p_ in e["polys"]]).buffer(1e-9))
                         for e in others[Y]["faces"]]
        for x in inner(Q):
            hit = False
            for _, Qi in ar.carried(x[None]):
                for q in Qi:
                    for S_, c0_, U_, poly in loaded[Y]:
                        if dist(S_, q[None])[0] < 1e-7 and poly.contains(Point((q - c0_) @ U_.T)):
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if not hit:
                return False
        return True

    def through_seed(Q):
        """A stretch passing through an image of the wall's own seed: the seed is a point of the wall's own shape, so
        the wall lies on both sides there; not an edge of it."""
        a, b = Q[:-1], Q[1:]
        d = b - a
        for y in Y:
            t = np.clip(((y - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0, 1)
            if np.linalg.norm(a + t[:, None] * d - y, axis=1).min() < 1e-9:
                return True
        return False
    edge_walls_of = {Y: [n for n in work["exact"]["wall_surfaces"].get(Y, []) if n in ar.LIB]
                     for Y in wd["edges"] if Y not in XL}

    def build(k):
        sh = sheets[k]
        S = sh["S"]
        if "st" not in sh:
            sh["st"] = [i for i, (L, P, a, b) in enumerate(st) if L in XL and dist(S, inner(P)).max() < 1e-7]
        # (its edges along other walls on this copy: every copy of the other wall's surface crossing it, the crossing
        # solved exactly, cut at its own corners, kept from corner to corner)
        for Y, names in edge_walls_of.items():
            for P in crossing_curves(sh["name"], S, names):
                for Q, a, b in lp.split_at_points(P, corners, tol=3e-4):
                    # (corner to corner; or from a face of the half-cell to a corner inside it: a corner on the faces,
                    # as A1, a vertex of the half-cell, is reached from the faces by every copy through it)
                    inner_corner = lambda x: min(x[0], x[1], x[3], x[2] - x[3]) > 1e-9
                    face_end = "face" in (a, b)
                    other = Q[-1] if a == "face" else Q[0]
                    if not a == b == "face" and {a, b} <= corners | {"face"} and a != "?" and b != "?" \
                            and (not face_end or inner_corner(other)) and not through_seed(Q) and present(Y, Q):
                        st.append((Y, Q, a, b))
                        sh["st"].append(len(st) - 1)
        sh["rim"] = face_curves(S)
        # (a wall it ends on lying in a face of the half-cell, as X23 in b2 = 0: its edge there runs along this copy's
        # rim on that face, kept from corner to corner like any other)
        for R in sh["rim"]:
            mid = R[len(R) // 2]
            fi = int(np.argmin(np.abs(mid @ ar.CELL.T)))
            for Y, names in edge_walls_of.items():
                if not in_face(Y, names, fi):
                    continue
                for Q, a, b in lp.split_at_points(R, corners, tol=3e-4):
                    inner_corner = lambda x: min(x[0], x[1], x[3], x[2] - x[3]) > 1e-9
                    other = Q[-1] if a == "face" else Q[0]
                    if a != "?" and b != "?" and {a, b} <= corners | {"face"} and not a == b == "face" \
                            and ("face" not in (a, b) or inner_corner(other)) and not through_seed(Q) \
                            and present(Y, Q):
                        st.append((Y, Q, a, b))
                        sh["st"].append(len(st) - 1)
        sh["wall"] = set()
        pts = [R for R in sh["rim"]] + [st[i][1] for i in sh["st"]]
        if not pts:
            sh["faces"] = []
            return
        pts = np.vstack(pts)
        c0 = pts.mean(0)
        U = np.linalg.svd(pts - c0, full_matrices=False)[2][:2]
        sh["ch"] = lambda X, c0=c0, U=U: (np.atleast_2d(X) - c0) @ U.T
        sh["c0"], sh["U"] = c0, U
        geo = [LineString(sh["ch"](st[i][1])) for i in sh["st"]] + [LineString(sh["ch"](R)) for R in sh["rim"]]
        allg = unary_union(geo)
        sh["faces"] = list(polygonize(unary_union([shapely.snap(g, allg, 1e-6) for g in geo])))

    def face_at(sh, y):
        q = Point(sh["ch"](y)[0])
        return [j for j, f in enumerate(sh["faces"]) if f.buffer(1e-12).contains(q)]
    # (and every copy holding an image of its seed, though none of its stretches lies on it)
    for y in Y:
        for n in bases:
            v = dist(ar.LIB[n], np.einsum("gij,j->gi", ar.GINV, y))
            for gg in np.flatnonzero(v < 1e-9)[:1]:
                S2 = ar.Img(ar.LIB[n], int(gg))
                sheets.setdefault(ar.sheet_key(S2), {"S": S2, "name": f"{n}@{gg}"})
    for k in list(sheets):
        build(k)
    # (its faces: those holding an image of its seed)
    queue = []
    for k, sh in sheets.items():
        for y in Y:
            if sh["faces"] and dist(sh["S"], y[None])[0] < 1e-9:
                for j in face_at(sh, y):
                    if j not in sh["wall"]:
                        sh["wall"].add(j)
                        queue.append((k, j))
    # (and those it goes on to past a face of the half-cell where none of its edges is: the point just past the face
    # carried back into the half-cell by the group, onto the copy of its surface it lands on)
    # (strictly: no going on where the crossing lies on any copy of one of its edge lines, or of the surface of a wall
    # it ends on)
    line_pts = [np.asarray(c, float) for L in wd["edges"] if L in XL for c in line_in_cell(XL[L]["points"])]
    tree, tree_n = [None], [-1]
    edge_walls = [n for Y in wd["edges"] if Y not in XL for n in work["exact"]["wall_surfaces"].get(Y, []) if n in ar.LIB]

    def on_edge_wall(m, S):
        """m on a copy of the surface of a wall it ends on, other than its own surface there (walls can share planes:
        F1 lies on copies of T2's)."""
        # (only a copy lying in the face itself, as X23 does in b2 = 0: it crosses no copy of the surface, so it is
        # not among the traced edges; the edges that cross are, corner to corner)
        X = np.einsum("gij,j->gi", ar.GINV, m)
        nf = ar.CELL[int(np.argmin(np.abs(m @ ar.CELL.T)))]
        face = ar.sheet_key(osd.Plane(nf, "face"))
        for n in edge_walls:
            for g in np.flatnonzero(dist(ar.LIB[n], X) < 1e-7):
                if ar.sheet_key(ar.Img(ar.LIB[n], int(g))) == face:
                    return True
        return False
    while queue:
        k, j = queue.pop()
        sh = sheets[k]
        S, f = sh["S"], sh["faces"][j]
        for R in sh["rim"]:
            C2 = sh["ch"](R)
            mids3, mids2 = (R[:-1] + R[1:]) / 2, (C2[:-1] + C2[1:]) / 2
            on = np.array([f.boundary.distance(Point(m)) < 1e-9 for m in mids2], bool)
            for m in mids3[on][::4]:
                if len(st) != tree_n[0]:
                    tree_n[0] = len(st)
                    tree[0] = cKDTree(np.vstack(line_pts + [P for _, P, _, _ in st]))
                if tree[0].query(m)[0] < 1e-6:
                    continue                               # (an edge of it lies in the face there)
                vals = m @ ar.CELL.T
                nf = ar.CELL[int(np.argmin(np.abs(vals)))]
                g = ar.grads(S, m[None])[0]
                d = -(nf - nf.mean())
                d = d - g * (d @ g) / (g @ g + 1e-300)
                d = d - d.mean()
                if np.linalg.norm(d) < 1e-12:
                    continue
                p = ar.onto([S], (m + 1e-5 * d / np.linalg.norm(d))[None])
                for _, Q in ar.carried(p, strict=True):
                    for q in Q:
                        for n in bases:
                            v = dist(ar.LIB[n], np.einsum("gij,j->gi", ar.GINV, q))
                            for gg in np.flatnonzero(v < 1e-7)[:1]:
                                S2 = ar.Img(ar.LIB[n], int(gg))
                                k2 = ar.sheet_key(S2)
                                if k2 not in sheets:
                                    sheets[k2] = {"S": S2, "name": f"{n}@{gg}"}
                                    build(k2)
                                sh2 = sheets[k2]
                                if not sh2["faces"]:
                                    continue
                                for j2 in face_at(sh2, q):
                                    if j2 not in sh2["wall"]:
                                        sh2["wall"].add(j2)
                                        queue.append((k2, j2))
    drawn = set()
    report = {"sheets": len(sheets), "sheets_with_wall": sum(1 for sh in sheets.values() if sh["wall"])}
    for k, sh in sheets.items():
        wf = [sh["faces"][j] for j in sh["wall"]]
        if not wf:
            continue
        for i in sh["st"]:
            P = sh["ch"](st[i][1])
            m = len(P) // 2
            a, b = P[max(0, m - 1)], P[min(len(P) - 1, m + 1)]
            t = b - a
            nrm = np.array([-t[1], t[0]]) / (np.linalg.norm(t) + 1e-300)
            mid = P[m] if len(P) > 2 else (P[0] + P[1]) / 2
            side = [any(f.contains(Point(mid + s * 1e-5 * nrm)) for f in wf) for s in (1, -1)]
            if side[0] != side[1]:
                drawn.add(i)
    if dbg is not None:
        dbg.update(st=st, sheets=sheets)
    # (checks)
    got = {st[i][0] for i in drawn}
    deg = {}
    for i in drawn:
        L, P, a, b = st[i]
        for x in (P[0], P[-1]):
            if min(x[0], x[1], x[3], x[2] - x[3]) < 1e-9:
                continue
            kk = tuple(np.round(x, 6))
            deg[kk] = deg.get(kk, 0) + 1
    report["edges_not_drawn"] = sorted(set(wd["edges"]) - got)
    report["corners_met_once"] = sorted({str(list(k)) for k, n in deg.items() if n < 2})
    out = [{"along": st[i][0], "from": st[i][2], "to": st[i][3], "points": np.round(st[i][1], 12).tolist()}
           for i in sorted(drawn)]
    # (its cuts: where a wall that ends on it does so, the stretches of that wall's edge on it that run through its
    # faces, the wall on both sides)
    cuts = []
    for c in wd.get("cuts", []):
        Yc = c["by"]
        for run in curves.get(Yc, {}).get(w, []):
            for P in in_cell(run):
                for k, sh in sheets.items():
                    if not sh["wall"] or dist(sh["S"], inner(P)).max() > 1e-7:
                        continue
                    wfu = unary_union([sh["faces"][j] for j in sh["wall"]])
                    C2 = sh["ch"](P)
                    keep = [wfu.contains(Point((C2[i] + C2[i + 1]) / 2)) for i in range(len(P) - 1)]
                    cur = []
                    for i, kp in enumerate(keep):
                        if kp:
                            cur = cur or [P[i]]
                            cur.append(P[i + 1])
                        elif cur:
                            cuts.append({"by": Yc, "points": np.round(np.array(cur), 12).tolist()})
                            cur = []
                    if len(cur) >= 2:
                        cuts.append({"by": Yc, "points": np.round(np.array(cur), 12).tolist()})
    report["cuts_drawn"] = len(cuts)
    report["faces"] = [{"name": sh["name"], "c0": sh["c0"].tolist(), "U": sh["U"].tolist(),
                        "polys": [[list(sh["faces"][j].exterior.coords)] + [list(r.coords) for r in sh["faces"][j].interiors]
                                  for j in sorted(sh["wall"])]}
                       for sh in sheets.values() if sh.get("wall")]
    return out, report, cuts


def main(only=None):
    work = json.load(open("working_data.json"))
    curves = json.load(open("exact_curves.json"))
    seeds = {k: v["seed"] for k, v in json.load(open("working_base.json"))["shapes"].items() if v.get("seed")}
    try:
        res = json.load(open(OUT)) if only else {}
    except FileNotFoundError:
        res = {}
    # (a wall after the walls it ends on, so whether they are there is known from their own faces)
    XL = work["exact"]["lines"]
    todo = [w for w in work["walls"] if not only or w in only]
    deps = {w: {e for e in work["walls"][w]["edges"] if e not in XL} for w in todo}
    order = []
    while todo:
        ready = [w for w in todo if not (deps[w] & set(todo))] or todo[:1]
        order += ready
        todo = [w for w in todo if w not in ready]
    for w in order:
        edges, rep, cuts = wall(w, work, curves, seeds, others=res)
        faces = rep.pop("faces")
        res[w] = {"edges": edges, "report": rep, "cuts": cuts, "faces": faces}
        print(f"{w}: {len(edges)} edges drawn; sheets {rep['sheets']} ({rep['sheets_with_wall']} with its faces); "
              f"not drawn {rep['edges_not_drawn']}; corners met once {len(rep['corners_met_once'])}", flush=True)
        json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
