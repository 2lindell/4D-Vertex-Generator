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
  * its drawn edges: the stretches with one of its faces on one side only. A stretch with its faces on neither side
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


def wall(w, work, curves, seeds, dbg=None):
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
        else:
            for run in curves.get(w, {}).get(L, []):
                # (cut at its own corners too, like a line: a traced curve runs on past where the wall ends)
                st += [(L, Q, a, b) for P in in_cell(run) for Q, a, b in lp.split_at_points(P, corners)]
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

    def build(k):
        sh = sheets[k]
        S = sh["S"]
        if "st" not in sh:
            sh["st"] = [i for i, (L, P, a, b) in enumerate(st) if dist(S, inner(P)).max() < 1e-7]
        sh["rim"] = face_curves(S)
        sh["wall"] = set()
        pts = [R for R in sh["rim"]] + [st[i][1] for i in sh["st"]]
        if not pts:
            sh["faces"] = []
            return
        pts = np.vstack(pts)
        c0 = pts.mean(0)
        U = np.linalg.svd(pts - c0, full_matrices=False)[2][:2]
        sh["ch"] = lambda X, c0=c0, U=U: (np.atleast_2d(X) - c0) @ U.T
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
    allst = cKDTree(np.vstack(line_pts + [P for _, P, _, _ in st])) if line_pts or st else None
    edge_walls = [n for Y in wd["edges"] if Y not in XL for n in work["exact"]["wall_surfaces"].get(Y, []) if n in ar.LIB]

    def on_edge_wall(m, S):
        """m on a copy of the surface of a wall it ends on, other than its own surface there (walls can share planes:
        F1 lies on copies of T2's)."""
        X = np.einsum("gij,j->gi", ar.GINV, m)
        own = ar.sheet_key(S)
        for n in edge_walls:
            for g in np.flatnonzero(dist(ar.LIB[n], X) < 1e-7):
                if ar.sheet_key(ar.Img(ar.LIB[n], int(g))) != own:
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
                if (allst is not None and allst.query(m)[0] < 1e-6) or on_edge_wall(m, S):
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
    return out, report


def main(only=None):
    work = json.load(open("working_data.json"))
    curves = json.load(open("exact_curves.json"))
    seeds = {k: v["seed"] for k, v in json.load(open("working_base.json"))["shapes"].items() if v.get("seed")}
    try:
        res = json.load(open(OUT)) if only else {}
    except FileNotFoundError:
        res = {}
    for w in work["walls"]:
        if only and w not in only:
            continue
        edges, rep = wall(w, work, curves, seeds)
        res[w] = {"edges": edges, "report": rep}
        print(f"{w}: {len(edges)} edges drawn; sheets {rep['sheets']} ({rep['sheets_with_wall']} with its faces); "
              f"not drawn {rep['edges_not_drawn']}; corners met once {len(rep['corners_met_once'])}", flush=True)
        json.dump(res, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
