"""Each wall's faces in the half-cell, from its exact curves alone (exact_faces.json): what the viewer draws as its
edges and cuts.

On each of the wall's surfaces as placed in the half-cell, every exact curve lying on it is laid out: the pieces of its
lines (exact_lines.json) and its edges along other walls (exact_curves.json), the cuts other walls make on it (their
own edges on it), where its other planes meet it (seams), and where the half-cell's faces meet it. The faces these
enclose are found exactly (in a flat chart of the surface: a plane's own coordinates, or a curved surface seen along its
mean normal, checked to be one sheet over the chart). The wall is the faces holding its seed (working_base.json), and
every face reached from one of them across a cut, or across a face of the half-cell (the wall going on into the next
cell: the point just past it carried back into the half-cell by the group, onto another of the wall's surfaces), never
across an edge or a seam. Then each stretch of curve is:
  * an edge of the wall where the wall lies on one side only;
  * a cut where the wall lies on both sides;
  * not drawn where the wall lies on neither side (a copy of the curve where the wall is not).
No mesh and no classification of points is used. The working data is not changed: this is for the viewer.

    python exact_faces.py [walls]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

import core as osd
from core import FOLD, extended
from domain_page import clip_polyline, line_in_cell
from exact_page import in_cell
from pieces import grad, project

OUT = "exact_faces.json"
E_ = extended()
GG = np.concatenate([E_, E_ @ FOLD])
planes_, curved_ = osd.library()
LIB = {s.name: s for s in list(planes_) + list(curved_)}
B = np.linalg.svd(np.vstack([np.ones(4), np.zeros((3, 4))]))[2][1:]       # (an orthonormal basis of sum-zero)
E1, E2, E3, M = np.eye(4)[0], np.eye(4)[1], np.eye(4)[2], np.array([0, 0, .5, .5])
FACES = {"face b4=0": (np.eye(4)[3], (E1, E2, E3)), "mirror b3=b4": (np.array([0, 0, 1, -1.0]), (E1, E2, M)),
         "face b1=0": (np.eye(4)[0], (E2, E3, M)), "face b2=0": (np.eye(4)[1], (E1, E3, M))}
HALF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])
SNAP = 1e-7


def f_of(S, X):
    return np.atleast_1d(S.f(np.atleast_2d(X))).ravel()


def off(S, X):
    """Distance of the points X (betas) from the surface S, to first order."""
    X = np.atleast_2d(X)
    return np.abs(f_of(S, X)) / np.array([np.linalg.norm(grad(S, x)) + 1e-300 for x in X])


def on(S, P, tol=1e-7):
    P = np.asarray(P, float)
    k = sorted({0, len(P) // 4, len(P) // 2, 3 * len(P) // 4, len(P) - 1})
    return bool((off(S, P[k]) < tol).all())


def face_curves(S, n=200):
    """Where S meets each face of the half-cell: marching triangles on the face, each point solved onto S and the face."""
    out = []
    for name, (nf, (A, Bv, C)) in FACES.items():
        I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        ok = I + J <= n
        idx = -np.ones((n + 1, n + 1), int)
        idx[ok] = np.arange(ok.sum())
        P = A + (I[ok, None] / n) * (Bv - A) + (J[ok, None] / n) * (C - A)
        F = f_of(S, P)
        F[F == 0] = 1e-300
        tris = []
        for i in range(n):
            for j in range(n - i):
                tris.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
                if i + j + 1 < n:
                    tris.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
        adj = {}
        for t in tris:
            ks = [tuple(sorted((t[a], t[b]))) for a, b in ((0, 1), (1, 2), (2, 0)) if (F[t[a]] > 0) != (F[t[b]] > 0)]
            if len(ks) == 2:
                adj.setdefault(ks[0], []).append(ks[1])
                adj.setdefault(ks[1], []).append(ks[0])
        seen = set()
        fp = osd.Plane(nf, name)
        pt = lambda k: P[k[0]] + (P[k[1]] - P[k[0]]) * F[k[0]] / (F[k[0]] - F[k[1]])
        for k0 in adj:
            if k0 in seen:
                continue
            # (walk to one end of the chain, then along it)
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
                        chain.append(start)                # (a closed loop)
                    break
                prev, cur = cur, nx[0]
                seen.add(cur)
                chain.append(cur)
            Q = np.array([project([S, fp], pt(k)) for k in chain])
            if len(Q) >= 2:
                out.append((name, Q))
    return out


def seam(S, S2):
    """Where the planes S and S2 meet, in the half-cell."""
    a, b = np.linalg.svd(np.vstack([S.n, S2.n]))[2][2:]
    t = np.linspace(0, np.pi, 4001)
    X = np.cos(t)[:, None] * a + np.sin(t)[:, None] * b
    s = X.sum(1)
    runs, cur = [], []
    for x, sm in zip(X, s):
        if abs(sm) > 1e-6:
            cur.append(x / sm)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [np.array(r) for R in runs if len(R) >= 2 for r in clip_polyline(np.array(R)) if len(r) >= 2]


class Chart:
    def __init__(self, S, pts):
        if isinstance(S, osd.Plane):
            N = B @ (S.n - S.n.mean())
        else:
            G = np.array([B @ grad(S, p) for p in pts])
            G /= np.linalg.norm(G, axis=1)[:, None]
            G *= np.sign(G @ G[0])[:, None]
            N = G.mean(0)
            self.worst = float((G @ (N / np.linalg.norm(N))).min())
        N = N / np.linalg.norm(N)
        U = np.linalg.svd(N[None])[2][1:]
        self.U = U @ B

    def __call__(self, X):
        return np.atleast_2d(X) @ self.U.T


def wall_faces(w, D, debug=False):
    work, ex, lines, curves, seeds = D
    wd = work[w]
    own = [n for n in ex["walls"][w]["surface"] if n in LIB]
    sheets = {}
    for sn in own:
        S = LIB[sn]
        cur = []                                           # (kind, name, betas)
        for e in wd["edges"]:
            if e in lines:
                cur += [("edge", e, np.asarray(P, float)) for P in line_in_cell(lines[e]["points"])]
            else:
                cur += [("edge", e, P) for run in curves.get(w, {}).get(e, []) for P in in_cell(run)]
        for c in wd.get("cuts", []):
            cur += [("cut", c["by"], P) for run in curves.get(c["by"], {}).get(w, []) for P in in_cell(run)]
        cur = [c for c in cur if len(c[2]) >= 2 and on(S, c[2])]
        for sn2 in own:
            if sn2 != sn and isinstance(S, osd.Plane):
                cur += [("seam", sn2, P) for P in seam(S, LIB[sn2])]
        cur += [("face", nm, P) for nm, P in face_curves(S)]
        ch = Chart(S, np.vstack([c[2] for c in cur]))
        geo = [LineString(ch(c[2])) for c in cur]
        # (ends of curves snapped onto the curves they end on, then every crossing made a node)
        allg = unary_union(geo)
        geo = [shapely.snap(g, allg, SNAP) for g in geo]
        noded = unary_union(geo)
        polys = list(polygonize(noded))
        segs = list(getattr(noded, "geoms", [noded]))
        tree = shapely.STRtree(geo)
        lab = []
        for sg in segs:
            m = sg.interpolate(0.5, normalized=True)
            hit = [i for i in tree.query(m.buffer(1e-6)) if geo[i].distance(m) < 1e-6]
            lab.append(hit)
        ptree = shapely.STRtree(polys)
        sides = []
        for sg in segs:
            m = sg.interpolate(0.5, normalized=True)
            sides.append([int(i) for i in ptree.query(m.buffer(1e-7)) if polys[i].boundary.distance(m) < 1e-7])
        sheets[sn] = {"S": S, "chart": ch, "cur": cur, "geo": geo, "polys": polys, "segs": segs, "lab": lab,
                      "sides": sides, "ptree": ptree, "worst": getattr(ch, "worst", 1.0)}
    # (the faces holding the seed)
    wall = {sn: set() for sn in sheets}
    sd = np.asarray(seeds[w], float)
    Y = np.einsum("gij,j->gi", GG, sd)
    s_ = Y.sum(1)
    Y = Y[np.abs(s_) > 1e-12] / s_[np.abs(s_) > 1e-12, None]
    Y = Y[(Y @ HALF.T > -1e-12).all(1)]
    for sn, sh in sheets.items():
        for y in Y[off(sh["S"], Y) < 1e-9]:
            p = Point(sh["chart"](y)[0])
            for i in sh["ptree"].query(p.buffer(1e-9)):
                if sh["polys"][i].distance(p) < 1e-9:
                    wall[sn].add(int(i))
    # (spread across cuts on a sheet, and across the half-cell's faces into the next cell, carried back)
    changed = True
    while changed:
        changed = False
        for sn, sh in sheets.items():
            for k, sg in enumerate(sh["segs"]):
                kinds = {sh["cur"][i][0] for i in sh["lab"][k]}
                sd_ = sh["sides"][k]
                if kinds == {"cut"} and len(sd_) == 2 and (sd_[0] in wall[sn]) != (sd_[1] in wall[sn]):
                    wall[sn] |= set(sd_)
                    changed = True
                if kinds == {"face"} and len(sd_) == 1 and sd_[0] in wall[sn] and not sh.get("done_%d" % k):
                    sh["done_%d" % k] = True
                    for sn2, i2 in across(sh, k, sheets):
                        if i2 not in wall[sn2]:
                            wall[sn2].add(i2)
                            changed = True
    if debug:
        return sheets, wall
    # (each stretch: an edge where the wall is on one side, a cut where on both)
    solid, dashed, notes = [], [], []
    for sn, sh in sheets.items():
        if sh["worst"] < 0.2:
            notes.append(f"{sn}: the chart folds (worst {sh['worst']:.2f})")
        for k, sg in enumerate(sh["segs"]):
            n_in = sum(i in wall[sn] for i in sh["sides"][k])
            for i in sh["lab"][k]:
                kind, nm, P = sh["cur"][i]
                if kind not in ("edge", "cut"):
                    continue
                X = lift(sg, sh, P)
                if kind == "edge" and n_in == 1:
                    solid.append((nm, X))
                elif kind == "cut" and n_in == 2:
                    dashed.append((nm, X))
    area = {sn: float(sum(sh["polys"][i].area for i in wall[sn])) for sn, sh in sheets.items()}
    return {"solid": [[nm, X.round(12).tolist()] for nm, X in solid],
            "dashed": [[nm, X.round(12).tolist()] for nm, X in dashed],
            "faces": {sn: [len(wall[sn]), len(sh["polys"])] for sn, sh in sheets.items()}, "area": area, "notes": notes}


def lift(sg, sh, P):
    """A stretch of noded curve in the chart back onto the curve's own points (betas)."""
    Q = sh["chart"](P)
    C = np.asarray(sg.coords)
    out = []
    for c in C:
        d = np.linalg.norm(Q - c, axis=1)
        k = int(d.argmin())
        if d[k] < 1e-9:
            out.append(P[k])
            continue
        # (a node between two of the curve's points)
        a, b = Q[:-1], Q[1:]
        t = np.clip(((c - a) * (b - a)).sum(1) / np.maximum(((b - a) ** 2).sum(1), 1e-300), 0, 1)
        j = int(np.linalg.norm(a + t[:, None] * (b - a) - c, axis=1).argmin())
        out.append(P[j] + t[j] * (P[j + 1] - P[j]))
    return np.array(out)


def across(sh, k, sheets, eps=1e-4):
    """The wall's faces just across a face of the half-cell from stretch k: the point just past it, carried back into
    the half-cell by each group element, located on the wall's sheet it lands on."""
    S = sh["S"]
    sg = sh["segs"][k]
    m2 = np.asarray(sg.interpolate(0.5, normalized=True).coords[0])
    i = sh["lab"][k][0]
    _, nm, P = sh["cur"][i]
    m = lift(LineString([m2, m2]), sh, P)[0]
    nf = FACES[nm][0]
    # (the direction on S across the face, away from the half-cell)
    g = grad(S, m)
    d = -(nf - nf.mean())
    d = d - g * (d @ g) / (g @ g)
    d = d - d.mean()
    out = []
    p = project([S], m + eps * d / np.linalg.norm(d))
    if (HALF @ p > 0).all():
        return out
    Y = np.einsum("gij,j->gi", GG, p)
    s_ = Y.sum(1)
    ok = np.abs(s_) > 1e-12
    Y = Y[ok] / s_[ok, None]
    Y = Y[(Y @ HALF.T > 1e-9).all(1)]
    for sn2, sh2 in sheets.items():
        for y in Y[off(sh2["S"], Y) < 1e-8]:
            q = Point(sh2["chart"](y)[0])
            for j in sh2["ptree"].query(q.buffer(1e-9)):
                if sh2["polys"][j].distance(q) < 1e-9:
                    out.append((sn2, int(j)))
    return out


def load():
    work = json.load(open("working_data.json"))["walls"]
    ex = json.load(open("exact_defs.json"))
    lines = json.load(open("exact_lines.json"))["lines"]
    curves = json.load(open("exact_curves.json"))
    seeds = {k: v["seed"] for k, v in json.load(open("working_base.json"))["shapes"].items() if v.get("seed")}
    return work, ex, lines, curves, seeds


def one(w):
    try:
        return w, wall_faces(w, load())
    except Exception as e:                                 # (reported, not hidden)
        return w, {"error": repr(e)}


def main(only=None):
    walls = [w for w in json.load(open("working_data.json"))["walls"] if not only or w in only]
    out = {}
    try:
        if only:
            out = json.load(open(OUT))
    except FileNotFoundError:
        pass
    with Pool() as pool:
        for w, r in pool.imap_unordered(one, walls):
            out[w] = r
            if "error" in r:
                print(w, "ERROR", r["error"], flush=True)
            else:
                print(w, "faces", r["faces"], "solid", len(r["solid"]), "dashed", len(r["dashed"]),
                      r["notes"] or "", flush=True)
            json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
