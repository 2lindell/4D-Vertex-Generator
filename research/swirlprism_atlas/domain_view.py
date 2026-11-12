"""The Symmetry Domain page's walls, lines and regions, from the 3-sphere map's wall pieces.

Input: the sphere map's `domain_pieces.json` / `.npz` (branch claude/sharp-rubin-c0puiw, module
`sphere_map.domain_pieces`). There every wall surface in the half-cell is cut by its exact edges
only: the named lines (stretches cut and snapped at the exact named points) and the curves where
another wall meets it, with the half-cell's own faces; each face of that graph is one wall piece
in the half-cell, named from the map's pieces, and triangulated inside its own exact loop (the loop
points the only rim vertices, every inner vertex on the exact surface).

This page keeps everything else it shows (frame, mirrors, axes, samples, seeds, key). Its three
views are each a projective chart of the 3-sphere; each chart is read off the page itself:

  - the half-cell: fitted to the page's own seed points (each listed with its point on the sphere);
  - the dodecahedron round V1 (and its tenth, the chambers): the gnomonic chart at V1, turned to
    match the page's samples there; the walls carried there by every element of the group and cut
    exactly at the dodecahedron's faces (the tenth: at its own faces, as the page draws them).

Wall pieces are drawn as they are; a piece's edges on named lines solid, where another wall meets
it dashed, nothing along the half-cell's faces. The named lines are their exact stretches. A
region shows the wall pieces with it on one side and their edges.

    PYTHONPATH=../../src python domain_view.py <sphere_map data dir> ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

T = np.load(Path(__file__).with_name("cell_vertices.npy"))
RINGS = ("C2a", "C2b", "C3", "D1", "D2", "D3")
DODECA_LINE_TOL = 2e-3   # (curves in the dodecahedron views kept where they turn by more than this)
SLIVER = 1e-3          # a wall copy reaching no deeper than this into a dodecahedron view is left out
CLIP_TOL = 1e-9         # a point within this of a view's face counts as inside (pieces lying in the face)
SEL_ONLY = {"T1", "E1", "E2"}


def to_sphere(B):
    X = np.atleast_2d(B) @ T
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def blocks(html):
    out = {}
    for m in re.finditer(r'<script id="([^"]+)" type="application/json">', html):
        s = m.end()
        out[m.group(1)] = (s, html.index("</script>", s))
    return out


def seed_of(hover):
    m = re.search(r"seed ([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+), ([-0-9.e]+)", hover)
    if not m:
        return None
    x = np.array([float(v) for v in m.groups()])
    return x / np.linalg.norm(x)


# ---------------------------------------------------------------- charts

class Chart:
    """u = (X M) / (X . p) for points X on the sphere (X . p > 0)."""

    def __init__(self, M, p):
        self.M, self.p = M, p

    def __call__(self, X):
        X = np.atleast_2d(X)
        return (X @ self.M) / (X @ self.p)[:, None]


def fit_cell_chart(DC):
    """The half-cell view's chart, from the page's seed points (point on the sphere, place drawn)."""
    X, Q = [], []
    for u in DC["uniform"] + DC["special"]:
        x = seed_of(u.get("hover", ""))
        if x is not None:
            X.append(x)
            Q.append(u["q"])
    X, Q = np.array(X), np.array(Q)
    rows = []
    for x, q in zip(X, Q):
        for k in range(3):
            r = np.zeros(16)
            r[4 * k:4 * k + 4] = x
            r[12:] = -q[k] * x
            rows.append(r)
    sol = np.linalg.svd(np.array(rows))[2][-1]
    M, p = sol[:12].reshape(3, 4).T, sol[12:]
    if (X @ p).mean() < 0:
        M, p = -M, -p
    ch = Chart(M, p)
    err = np.abs(ch(X) - Q).max()
    return ch, err


def fit_v1_chart(DC, V1, G):
    """The dodecahedron views' chart: gnomonic at V1 = beta (1, 0, 0, 0), turned (any linear map of
    the chart) so the page's samples there are images of the half-cell's samples."""
    c = to_sphere([[1, 0, 0, 0]])[0]
    B0 = np.linalg.svd(c[None])[2][1:]
    w0 = lambda X: (np.atleast_2d(X) @ B0.T) / (np.atleast_2d(X) @ c)[:, None]
    near = dodeca_planes(c, G)
    src = {k: [seed_of(p[3]) for p in v] for k, v in DC["samples"].items()}
    pairs = []
    for k, v in V1["samples"].items():
        for p in v:
            x = src[k][p[3]]
            if x is None:
                continue
            Y = x @ G.transpose(0, 2, 1)                       # every image g x
            Y = Y[(Y @ near.T >= -1e-9).all(1) & (Y @ c > 0)]
            if len(Y):
                pairs.append((np.array(p[:3]), w0(Y)))
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[:400]]
    best = (-1, None)
    u1, W1 = pairs[0]
    for i2 in range(1, 6):
        for i3 in range(6, 12):
            for a in W1[:1]:
                for b in pairs[i2][1]:
                    for c3 in pairs[i3][1]:
                        A = np.vstack([a, b, c3])
                        if abs(np.linalg.det(A)) < 1e-6:
                            continue
                        R = np.linalg.solve(A, np.vstack([u1, pairs[i2][0], pairs[i3][0]])).T
                        good = sum(np.linalg.norm(W @ R.T - u, axis=1).min() < 2e-3 for u, W in pairs)
                        if good > best[0]:
                            best = (good, R)
    R = best[1]
    ch = Chart((R @ B0).T, c)
    return ch, best[0] / len(pairs), near


def dodeca_planes(c, G):
    """The 12 faces of the dodecahedron round c: X . (c - n) >= 0 for its 12 nearest images n."""
    orbit = np.unique(np.round(G @ c, 9), axis=0)
    near = orbit[np.argsort(-(orbit @ c))[1:13]]
    return c - near


def chamber_planes(CH, chart, G):
    """The tenth's faces (as the page draws them), as half-spaces X . h >= 0 on the sphere."""
    from scipy.spatial import ConvexHull
    P = np.unique(np.round(np.array([p for f in CH["dodecaFaces"] for p in f]), 5), axis=0)
    H = ConvexHull(P)
    out = []
    Bm, c = chart.M, chart.p
    for a0, a1, a2, b in H.equations:                # inside: a . u + b <= 0
        a = np.array([a0, a1, a2])
        out.append(-(Bm @ a + b * c))
    # (the page draws the tenth to 3 places: each face snapped to the exact plane it is, an H4 mirror
    # (its normal a vertex of the 600-cell: an image of V1) or a face of the dodecahedron round V1)
    orbit = np.unique(np.round(G @ c, 12), axis=0)
    D_ = dodeca_planes(c, G)
    exact = [v / np.linalg.norm(v) for v in orbit] + [d / np.linalg.norm(d) for d in D_]
    # (and the planes through V1 and two of the dodecahedron's own points, the chambers' faces: its
    # face centres, corners and edge midpoints)
    near = c - D_
    centres = [(c + n) / np.linalg.norm(c + n) for n in near]
    from itertools import combinations
    corners = []
    for i, j, k in combinations(range(len(D_)), 3):
        x = np.linalg.svd(np.vstack([D_[i], D_[j], D_[k]]))[2][-1]
        x = x if x @ c > 0 else -x
        if (D_ @ x >= -1e-9).all() and not any(np.linalg.norm(x - y) < 1e-9 for y in corners):
            corners.append(x)
    mids = []
    for a, b in combinations(corners, 2):
        m = (a + b) / np.linalg.norm(a + b)
        if sum(abs(d @ m) < 1e-9 for d in D_) >= 2:
            mids.append(m)
    pts = centres + corners + mids
    for a, b in combinations(pts, 2):
        nrm = np.linalg.svd(np.vstack([c, a, b]))[2][-1]
        exact.append(nrm / np.linalg.norm(nrm))
    snapped = []
    for h in out:
        h = h / np.linalg.norm(h)
        best = max(exact, key=lambda e: abs(e @ h))
        snapped.append(best * np.sign(best @ h) if abs(best @ h) > 0.999 else h)
    return np.unique(np.round(np.array(snapped), 12), axis=0)


# ---------------------------------------------------------------- cutting on the sphere

def clip_polygon(P, H):
    """A convex polygon of points on R^4 (straight in every gnomonic chart) cut by X . h >= 0."""
    for h in H:
        if not P:
            return []
        out = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            fa, fb = a @ h + CLIP_TOL, b @ h + CLIP_TOL      # (a piece lying in the face: kept)
            if fa >= 0:
                out.append(a)
            if (fa >= 0) != (fb >= 0):
                out.append(a + fa / (fa - fb) * (b - a))
        P = out
    return P


def clip_tris(X, F, H):
    """Triangles (vertices X on R^4, faces F) cut by the half-spaces: a list of triangles (n,3,4)."""
    S = X @ H.T
    allin = (S[F] >= -CLIP_TOL).all((1, 2))
    out = [X[F[allin]]]
    for t in np.flatnonzero(~allin & ~(S[F] < -CLIP_TOL).all(1).any(1)):
        P = clip_polygon([X[i] for i in F[t]], H)
        for k in range(1, len(P) - 1):
            out.append(np.array([[P[0], P[k], P[k + 1]]]))
    return np.concatenate(out) if out else np.zeros((0, 3, 4))


def clip_line(P, H):
    """A polyline on R^4 cut by the half-spaces: its runs inside."""
    runs, cur = [], []
    for a, b in zip(P[:-1], P[1:]):
        lo, hi = 0.0, 1.0
        for h in H:
            fa, fb = a @ h + CLIP_TOL, b @ h + CLIP_TOL
            if fa < 0 and fb < 0:
                lo, hi = 1, 0
                break
            if fa < 0:
                lo = max(lo, fa / (fa - fb))
            elif fb < 0:
                hi = min(hi, fa / (fa - fb))
        if hi <= lo:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = []
            continue
        p, q = a + lo * (b - a), a + hi * (b - a)
        if not cur or np.linalg.norm(cur[-1] - p) > 1e-12:
            if len(cur) > 1:
                runs.append(np.array(cur))
            cur = [p]
        cur.append(q)
    if len(cur) > 1:
        runs.append(np.array(cur))
    return runs


def _surface_near(Q):
    """For a clipped set of triangles (n, 3, 4): a tree of its corners and centres, and a reach of
    about one and a half of its triangles."""
    from scipy.spatial import cKDTree
    if not len(Q):
        return None
    reach = 1.5 * np.median(np.linalg.norm(Q[:, 1] - Q[:, 0], axis=1))
    return cKDTree(np.r_[Q.reshape(-1, 4), Q.mean(1)]), reach


def _runs_beside(R, near, planes):
    """The stretches of a clipped polyline that border the surface: a point lying on a face of the
    view is kept only where the surface is beside it (an edge of a copy lying flat on a face of the
    view, with its surface going outward, is not this view's); points off the faces are kept."""
    if near is None:
        return []
    tree, reach = near
    on = (np.abs(R @ planes.T) < 1e-7).any(1)
    ok = ~on
    if on.any():
        ok[on] = tree.query(R[on])[0] < reach
    out, cur = [], []
    for x, k in zip(R, ok):
        if k:
            cur.append(x)
        elif cur:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        out.append(np.array(cur))
    return out

def orient(F):
    """Faces turned so neighbours agree (each shared edge run opposite ways in the two), so the
    surface is shaded as one sheet."""
    from collections import deque
    F = [list(f) for f in F]
    by = defaultdict(list)
    for t, f in enumerate(F):
        for k in range(3):
            by[frozenset((f[k], f[(k + 1) % 3]))].append(t)
    seen = [False] * len(F)
    for t0 in range(len(F)):
        if seen[t0]:
            continue
        seen[t0] = True
        q = deque([t0])
        while q:
            t = q.popleft()
            f = F[t]
            for k in range(3):
                a, b = f[k], f[(k + 1) % 3]
                for u in by[frozenset((a, b))]:
                    if u == t or seen[u]:
                        continue
                    g = F[u]
                    same = any(g[j] == a and g[(j + 1) % 3] == b for j in range(3))
                    if same:
                        F[u] = [g[0], g[2], g[1]]
                    seen[u] = True
                    q.append(u)
    return F


def mesh(Q, nd):
    """Triangles (n,3,3) as shared vertices and faces, rounded to nd places, turned consistently."""
    Q = np.round(Q, nd).reshape(-1, 3)
    V, inv = np.unique(Q, axis=0, return_inverse=True)
    F = inv.ravel().reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V.tolist(), orient(F.tolist())


def rounded(o, n):
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rounded(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, n) for v in o]
    return o


def thin(P, tol):
    """Points of a polyline kept where it turns by more than tol (ends kept)."""
    P = np.asarray(P, float)
    if len(P) <= 2:
        return P
    keep = [0]
    for i in range(1, len(P) - 1):
        a, c = P[keep[-1]], P[i + 1]
        d = c - a
        L = np.linalg.norm(d)
        x = P[keep[-1] + 1:i + 1] - a
        if L < 1e-12 or np.max(np.linalg.norm(x - np.outer(x @ d / L ** 2, d), axis=1)) > tol:
            keep.append(i)
    keep.append(len(P) - 1)
    return P[keep]


# ---------------------------------------------------------------- the page

def add_point_copies(DC, cell, G):
    """The page keeps each seed once; a seed on a face of the half-cell (the splitting mirror and the
    faces glued to other copies) also lies there as its copies under the group, on the other side of
    where the face is glued. Those copies are added (once: a copy already there is not added again),
    each with its own position and seed."""
    Tinv = np.linalg.inv(T)
    H = np.c_[Tinv, Tinv[:, 2] - Tinv[:, 3]].T
    H /= np.linalg.norm(H, axis=1, keepdims=True)

    def copies(x):
        if (x @ H.T).min() > 1e-9:                     # (inside: no other copy in the half-cell)
            return []
        Y = np.einsum("gij,j->gi", G, x)
        Y = Y[((Y @ H.T) >= -1e-9).all(1) & (np.abs(Y - x).max(1) > 1e-7)]
        return list({tuple(np.round(y, 9)): y for y in Y}.values())

    def relabel(hover, y):
        b = y @ Tinv
        b = np.clip(b / b.max(), 0, None)
        bt = ", ".join("0" if v < 5e-10 else "1" if abs(v - 1) < 5e-10 else f"{v:.4f}" for v in b)
        hover = re.sub(r"β ∝ \([^)]*\)", f"β ∝ ({bt})", hover)
        return re.sub(r"seed [-0-9.e]+, [-0-9.e]+, [-0-9.e]+, [-0-9.e]+",
                      "seed " + ", ".join(repr(float(v)) for v in y), hover)
    added = 0
    for nm, pts in DC["samples"].items():
        have = {tuple(np.round(p[:3], 3)) for p in pts}
        for p in list(pts):
            x = seed_of(p[3])
            if x is None:
                continue
            for y in copies(x):
                q = [round(float(v), 4) for v in cell(y[None])[0]]
                if tuple(np.round(q, 3)) in have:
                    continue
                have.add(tuple(np.round(q, 3)))
                pts.append(q + [relabel(p[3], y)])
                added += 1
    have = {(u["id"], u.get("copy"), tuple(np.round(u["q"], 3))) for u in DC["uniform"]}
    for u in list(DC["uniform"]):
        x = seed_of(u.get("hover", ""))
        if x is None:
            continue
        for y in copies(x):
            q = [round(float(v), 4) for v in cell(y[None])[0]]
            k = (u["id"], u.get("copy"), tuple(np.round(q, 3)))
            if k in have:
                continue
            have.add(k)
            DC["uniform"].append({**u, "q": q, "hover": relabel(u["hover"], y)})
            added += 1
    print(f"seeds on the half-cell's faces: {added} copies added on the other side of their face", flush=True)


COLOR_OF = {"region": "--x", "wall": "--xw", "line": "--xl", "point": "--xp"}


def add_new_shapes(DC, src_dir):
    """Shapes of the database the page's key does not list yet (found after the key was made): a
    row each, from the shape's record (working_data.json): its counts, and its seed, a point well
    inside it. Its boundaries are filled in from the 3-sphere map with every other shape's."""
    wd = json.loads((Path(src_dir).parent / "source" / "working_data.json").read_text())["shapes"]
    have = {t["id"] for t in DC["types"]}

    def order(k):
        m = re.match(r"([A-Z]+)(\d+)", k)
        return (m.group(1), int(m.group(2))) if m else (k, 0)
    for k in sorted(set(wd) - have, key=order):
        v = wd[k]
        if v.get("dim") not in COLOR_OF or v.get("seed") is None:
            continue
        b = np.asarray(v["seed"], float)
        x = to_sphere(b)[0]
        bt = "(" + ", ".join("1" if i == 0 else f"{t:.6f}" for i, t in enumerate(b / b[0])) + ")"
        seed = ", ".join(repr(float(t)) for t in x)
        DC["types"].append({"id": k, "name": v.get("name", "Not in the wiki list"), "listed": bool(v.get("listed")),
                            "dim": v["dim"], "color": COLOR_OF[v["dim"]], "symbol": "circle", "samples": 0,
                            "vertices": v.get("vertices", ""), "cells": v.get("cells", ""),
                            "faces": v.get("faces", ""), "edges": v.get("edges", ""), "locus": None,
                            "bounds": [], "example": f"{seed}  β ∝ {bt}", "seed": seed, "betatext": bt,
                            "partial": []})
        print(f"key: {k} ({v['dim']}) added from the database", flush=True)


def add_wiki_list(DC, src_dir):
    """Each shape's entry in the 5-swirldoic list (working_data.json, wiki_list) for the key: the
    list's name and 5sd codes, its counts and edge valences and edge ratio; a shape split from a
    listed one, with its counts and valences but another vertex figure, says so."""
    wd = json.loads((Path(src_dir).parent / "source" / "working_data.json").read_text())["shapes"]
    for t in DC["types"]:
        v = wd.get(t["id"], {})
        t.pop("wiki", None)
        t.pop("wikiNote", None)
        if "wiki_list" in v:
            e = v["wiki_list"]
            t["wiki"] = {"name": e["name"], "codes": e["codes"], "counts": e["counts"],
                         "valences": e["edge_valences"], "ratio": e["edge_ratio"]}
        elif v.get("split_from") and "wiki_list" in wd.get(v["split_from"], {}):
            e = wd[v["split_from"]]["wiki_list"]
            code = (e["codes"] or [e["name"]])[0]
            t["wikiNote"] = (f"not in the list: the counts and edge valences of {v['split_from']} "
                             f"({code}: {e['counts']}, {e['edge_valences']}), another vertex figure")


def main(src_dir, out_path):
    src_dir = Path(src_dir)
    doc = json.loads((src_dir / "domain_pieces.json").read_text())
    z = np.load(src_dir / "domain_pieces.npz")
    html = Path(out_path).read_text()
    pos = blocks(html)
    DC = json.loads(html[slice(*pos["atlas-data"])])
    V1 = json.loads(html[slice(*pos["atlas-v1"])])
    CH = json.loads(html[slice(*pos["atlas-chambers"])])
    add_new_shapes(DC, src_dir)
    add_wiki_list(DC, src_dir)
    names = {t["id"]: t["name"] for t in DC["types"]}
    by = {t["id"]: t for t in DC["types"]}
    G = np.stack(group_elements(named_symmetry("h4_swirlprism")))

    # the key's boundaries, from the 3-sphere map's own neighbours (shape_map3d.json, as the shape map
    # shows them): a region's walls, lines and corner points; a wall's edge lines and corner points,
    # and apart the regions it separates; a line's end points and the lines it ends on
    sm = json.loads((src_dir / "shape_map3d.json").read_text())
    name_of = {k: {x["id"]: x["name"] for x in sm[k]} for k in ("regions", "walls", "lines", "points")}
    plural = {"region": "regions", "wall": "walls", "line": "lines", "point": "points"}
    bnd, sep = defaultdict(set), defaultdict(set)
    want = {"regions": {("wall", "boundary"), ("line", "boundary"), ("point", "corner")},
            "walls": {("line", "boundary"), ("point", "corner")},
            "lines": {("point", "end"), ("line", "ends on")}}
    for k, rels in want.items():
        for x in sm[k]:
            for n in x["nb"]:
                nm_ = name_of[plural[n["kind"]]].get(n["node"])
                if nm_ is None:
                    continue
                if (n["kind"], n["rel"]) in rels:
                    bnd[x["name"]].add(nm_)
                elif k == "walls" and n["rel"] == "side":
                    sep[x["name"]].add(nm_)
    for t in DC["types"]:
        if t["id"] in bnd or t["dim"] in ("region", "wall", "line"):
            t["bounds"] = sorted(bnd.get(t["id"], ()))
            t["partial"] = []
            if t["dim"] == "wall":
                t["separates"] = sorted(sep.get(t["id"], ()))

    cell, err = fit_cell_chart(DC)
    v1c, share, dplanes = fit_v1_chart(DC, V1, G)
    chplanes = chamber_planes(CH, v1c, G)
    print(f"charts: half-cell fitted to the page's seeds within {err:.1e}; dodecahedron: "
          f"{share:.0%} of the page's samples there matched", flush=True)

    add_point_copies(DC, cell, G)

    verts, tris = z["verts"], z["tris"]
    ep, el = z["edge_pts"], z["edge_len"]
    eoff = np.r_[0, np.cumsum(el)]
    edge_pts = lambda i: ep[eoff[i]:eoff[i + 1]]
    E = doc["edges"]

    # every face: its triangles on the sphere, its drawn edges
    faces = []
    for f in doc["faces"]:
        Xv = to_sphere(verts[f["verts"][0]:f["verts"][1]])
        F = tris[f["tris"][0]:f["tris"][1]]
        drawn = []
        for i in f["edges"]:
            if E[i]["kind"] in ("line", "meet", "cross"):
                drawn.append((E[i]["kind"], E[i]["name"], to_sphere(edge_pts(i)), i))
        a, b = f["sides"]
        lines = sorted({n for k, n, _, _ in drawn if k == "line"})
        meets = sorted({n for k, n, _, _ in drawn if k != "line"})
        hv = (f"{f['wall']} · {names.get(f['wall'], f['wall'])}<br>wall piece between {a} and {b}"
              + (f"<br>edges on {', '.join(lines)}" if lines else "")
              + (f"<br>met by {', '.join(meets)}" if meets else ""))
        rec = {"f": f, "X": Xv, "F": F, "drawn": drawn, "hover": hv}
        if "coarse_verts" in f:                    # (the coarser mesh, for the many-image views)
            rec["Xc"] = to_sphere(z["coarse_verts"][f["coarse_verts"][0]:f["coarse_verts"][1]])
            rec["Fc"] = z["coarse_tris"][f["coarse_tris"][0]:f["coarse_tris"][1]]
        faces.append(rec)

    # an edge two pieces of one wall share is where the wall is cut (the wall goes on across it); an
    # edge only one piece of the wall has is where the wall ends. Each drawn once.
    ecount = defaultdict(int)
    for fc in faces:
        for i in {d[3] for d in fc["drawn"]}:
            ecount[(fc["f"]["wall"], i)] += 1

    # solid only along a named line (a shape of its own); along a curve where walls meet, which is no
    # shape, dashed, whether the wall goes on past it or ends there
    def dashed_edge(w, kind, i):
        return ecount[(w, i)] >= 2 or kind != "line"

    # (a face whose mesh is far denser than the rest, its rim's short steps crowding it: shown with
    # its coarser mesh in the half-cell too)
    Bz = np.linalg.svd(np.ones((1, 4)))[2][1:]
    dens = []
    for fc in faces:
        Xb = verts[fc["f"]["verts"][0]:fc["f"]["verts"][1]] @ Bz.T
        F_ = fc["F"]
        a = np.linalg.norm(np.cross(Xb[F_[:, 1]] - Xb[F_[:, 0]], Xb[F_[:, 2]] - Xb[F_[:, 0]]), axis=1).sum() / 2
        dens.append(len(F_) / max(a, 1e-12))
    med = float(np.median(dens))
    for fc, d_ in zip(faces, dens):
        fc["dense"] = d_ > 1.6 * med and "Xc" in fc

    def outlines(fc, draw, done):
        out = []
        w = fc["f"]["wall"]
        for kind, nm, P, i in fc["drawn"]:
            if (w, i) in done:
                continue
            done.add((w, i))
            cut = dashed_edge(w, kind, i)
            for Q in draw(P):
                if len(Q) < 2:
                    continue
                what = (f"on {nm}" if kind == "line" else f"where {nm} meets it")
                hv = (f"{w} · cut {what}, between its pieces" if ecount[(w, i)] >= 2 else
                      f"{w} · ends {what}" if kind != "line" else f"{w} · edge of the wall, {what}")
                out.append({"id": w, "outline": True, "copy": cut, "pts": Q, "hover": hv})
        return out

    # the named lines: every stretch once
    seen, line_runs = set(), []
    for i, e in enumerate(E):
        if e["kind"] != "line":
            continue
        P = edge_pts(i)
        key = (e["name"], tuple(np.round(np.sort(np.r_[P[0], P[-1], P[len(P) // 2]]), 6)))
        if key in seen:
            continue
        seen.add(key)
        line_runs.append((e["name"], to_sphere(P)))

    # the half-cell's faces (beta_i >= 0 and beta_3 >= beta_4) as half-spaces on R^4
    Tinv = np.linalg.inv(T)
    cell_planes = np.c_[Tinv, Tinv[:, 2] - Tinv[:, 3]].T
    cell_planes /= np.linalg.norm(cell_planes, axis=1, keepdims=True)

    def line_copies(P, planes, images):
        """The runs of a line's copies (under images) that lie in the view (planes): the database
        keeps a line lying in a face of the domain once, on one side of where the face is glued to
        itself, so its other half is a copy"""
        if planes is None:
            return [P @ g.T for g in images]
        Y = np.einsum("gij,nj->gni", np.asarray(images), P)
        hit = ((Y @ planes.T) >= -CLIP_TOL).all(-1).any(-1)
        return [R for k in np.flatnonzero(hit) for R in clip_line(Y[k], planes)]

    def build_view(chart, planes, nd, line_tol, images, inside_only=False, line_planes=None, line_images=None):
        xwalls, xlines, segments, rings, main_ = [], [], [], [], []
        shells = defaultdict(lambda: {"tris": [], "rim": [], "walls": set()})
        drawn_once, rim_done = {}, set()
        for fc in faces:
            parts = []
            X0, F0 = (fc["Xc"], fc["Fc"]) if (len(images) > 1 or fc.get("dense")) and "Xc" in fc else (fc["X"], fc["F"])
            for g in images:
                Xg = X0 @ g.T
                if planes is not None:
                    if not (Xg @ planes.T >= -CLIP_TOL).all(1).any():
                        continue
                    Q = clip_tris(Xg, F0, planes)
                    if len(Q):
                        u, v = Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]
                        area = 0.5 * np.sqrt(np.clip((u * u).sum(1) * (v * v).sum(1) - (u * v).sum(1) ** 2, 0, None))
                        Q = Q[area > 1e-12]
                        if area.sum() < 1e-9:
                            continue
                        # (a copy reaching into the view by a hair only, not lying in a face of it:
                        # nothing of it would show but its edge along the face)
                        in_face = (np.abs(Xg @ planes.T) < 1e-6).all(0).any()     # (the whole copy)
                        if not in_face and (Q.reshape(-1, 4) @ planes.T).min(1).max() < SLIVER:
                            continue
                else:
                    Q = Xg[F0]
                if len(Q):
                    parts.append((g, Q))
            if not parts:
                continue
            Q3 = chart(np.concatenate([q for _, q in parts]).reshape(-1, 4)).reshape(-1, 3, 3)
            v_, f_ = mesh(Q3, nd)
            xwalls.append({"id": fc["f"]["wall"], "hover": fc["hover"], "copy": False, "verts": v_, "faces": f_})
            # (a piece lying in a face of the half-cell has only its inside side in this view's domain;
            # in the dodecahedron views, a piece lying in a face of the view likewise: only the region
            # on the view's side of it)
            base_sides = fc["f"]["sides"][:1] if inside_only and fc["f"]["sheet"].startswith("face:") else fc["f"]["sides"]
            for g, Qg in parts:
                sides = base_sides
                if planes is not None and "probe" in fc["f"]:
                    Xg = X0 @ g.T
                    on = [h for h in planes if np.abs(Xg @ h).max() < 1e-9 * (1 + np.linalg.norm(h))]
                    if on:
                        x0 = np.asarray(fc["f"]["probe"])
                        d0 = np.asarray(fc["f"]["side0_dir"])
                        step = (to_sphere(x0 + 1e-6 * d0) - to_sphere(x0))[0] @ g.T
                        sides = fc["f"]["sides"][:1] if step @ on[0] > 0 else fc["f"]["sides"][1:]
                if planes is None:
                    draw = (lambda P, g=g: [chart(P @ g.T)])
                else:
                    # (only where the line borders this copy's own surface in the view: an edge of a
                    # copy lying flat on a face of the view, with its surface going outward, is not
                    # this view's)
                    near = _surface_near(Qg)
                    draw = (lambda P, g=g, near=near: [chart(R) for R0 in clip_line(P @ g.T, planes)
                                                       for R in _runs_beside(R0, near, planes)])
                done = drawn_once.setdefault(id(g), set())
                for o in outlines(fc, draw, done):
                    xlines.append({**o, "pts": thin(o["pts"], line_tol).tolist()})
                # (a region's edges: every drawn edge of its pieces once, dashed where the wall goes
                # on across it, as the walls draw them)
                w_ = fc["f"]["wall"]
                for kind, nm, P, i in fc["drawn"]:
                    for r in sides:
                        if (r, w_, i, id(g)) in rim_done:
                            continue
                        rim_done.add((r, w_, i, id(g)))
                        for Q in draw(P):
                            if len(Q) > 1:
                                shells[r]["rim"].append({"pts": thin(Q, line_tol).tolist(),
                                                         "dash": dashed_edge(w_, kind, i)})
                Qc = chart(Qg.reshape(-1, 4)).reshape(-1, 3, 3)
                for r in set(sides):
                    shells[r]["tris"].append(Qc)
                    shells[r]["walls"].add(fc["f"]["wall"])
        lp = planes if line_images is None else line_planes
        li = images if line_images is None else line_images
        line_seen = set()
        for nm, P in line_runs:
            for R in line_copies(P, lp, li):
                if line_images is not None:
                    # (the half-cell's copies: none only touching it, and a stretch reached from two
                    # copies, either way round, once)
                    if np.linalg.norm(np.diff(R, axis=0), axis=1).sum() < 1e-4:
                        continue
                    ends = sorted(tuple(np.round(x, 5)) for x in (R[0], R[-1]))
                    key = (nm, *ends, tuple(np.round(R.mean(0), 4)))
                    if key in line_seen:
                        continue
                    line_seen.add(key)
                Q = thin(chart(R), line_tol / 4).tolist()
                if nm in RINGS:
                    rings.append({"id": nm, "pts": Q})
                elif nm == "B1":
                    main_.append({"id": nm, "pts": Q})
                else:
                    hv = f"{nm} · {names.get(nm, nm)}<br>named line (each point on its two exact surfaces)"
                    if nm in SEL_ONLY or by.get(nm, {}).get("listed"):
                        segments.append({"id": nm, "copy": False, "selOnly": nm in SEL_ONLY, "pts": Q, "hover": hv})
                    else:
                        xlines.append({"id": nm, "copy": False, "pts": Q, "hover": hv})
        return {"xwalls": xwalls, "xlines": xlines, "segments": segments, "rings": rings, "main": main_}, shells

    ident = [np.eye(4)]
    cv, cshell = build_view(cell, None, 4, 1e-3, ident, inside_only=True, line_planes=cell_planes, line_images=list(G))
    vv, vshell = build_view(v1c, dplanes, 3, DODECA_LINE_TOL, list(G))
    chv, chshell = build_view(v1c, chplanes, 3, DODECA_LINE_TOL, list(G))

    region_files = {}
    for vk, view, shells in (("cell", DC, cshell), ("v1", V1, vshell), ("chambers", CH, chshell)):
        view.update({k: v for k, v in (cv if vk == "cell" else vv if vk == "v1" else chv).items()})
        view["regionShells"], view["regionEdges"] = {}, {}
        ids = []
        for r, sh in shells.items():
            if by.get(r, {}).get("dim") != "region":
                continue
            v_, f_ = mesh(np.concatenate(sh["tris"]), 4 if vk == "cell" else 3)
            # (its edges beside its walls, fetched with them when it is selected)
            region_files[f"{vk}/{r}.json"] = {"verts": v_, "faces": f_, "walls": sorted(sh["walls"]),
                                              "edges": sh["rim"]}
            ids.append(r)
        view["regionWallsIds"] = sorted(ids)
        view["tpatches"] = []

    dump = lambda o, n: json.dumps(rounded(o, n), separators=(",", ":"))
    rdir = Path(out_path).parent / "symmetry_domain_regions"
    shutil.rmtree(rdir, ignore_errors=True)
    for f, m in region_files.items():
        (rdir / f).parent.mkdir(parents=True, exist_ok=True)
        n_ = 4 if f.startswith("cell/") else 3
        (rdir / f).write_text(json.dumps({k: rounded(v, 4 if k == "edges" else n_) for k, v in m.items()},
                                         separators=(",", ":")))
    # (the dodecahedron views: meshes to 3 places; their curves to 4, so they stay smooth)
    LINE_KEYS = ("xlines", "segments", "rings", "main")
    dump_dv = lambda V: json.dumps({k: rounded(v, 4 if k in LINE_KEYS else 3) for k, v in V.items()},
                                   separators=(",", ":"))
    new = {"atlas-data": dump(DC, 4), "atlas-v1": dump_dv(V1), "atlas-chambers": dump_dv(CH)}
    for name in sorted(new, key=lambda n: -pos[n][0]):
        s, e = pos[name]
        html = html[:s] + new[name] + html[e:]
    # what is not drawn yet: walls with a patch of the map on a surface where no piece was built
    patches = json.loads((src_dir / "patches.json").read_text())["patches"]
    have = defaultdict(set)
    for f in doc["faces"]:
        have[f["sheet"]].add(f["wall"])
    whole, part = set(), defaultdict(int)
    built = {f["wall"] for f in doc["faces"]}
    for p in patches:
        if not p["wall"]:
            continue
        sheet = str(p["surface"]) if p["face"] is None else "face:" + p["face"]
        if p["wall"] not in built:
            whole.add(p["wall"])
        elif p["wall"] not in have[sheet]:
            part[(p["wall"], sheet)] += 1
    partial = sorted({w for w, _ in part})
    note = ("Walls are drawn from the 3-sphere map's wall pieces, each bounded by its exact edges. "
            + (f"Not drawn yet: {', '.join(sorted(whole))}" if whole else "")
            + (("; " if whole else "Not drawn yet: ") + f"part of {', '.join(partial)}" if partial else "")
            + ("." if whole or partial else ""))
    trimmed = sorted({f["wall"] for f in doc["faces"] if f.get("trimmed")})
    if trimmed:
        note += (f" Part of {', '.join(trimmed)} ends where the database has no curve yet; there it is cut "
                 "to its own triangles, so that edge is not exact.")
    tag = f'<p id="dv-note" class="lede">{note}</p>'
    html = re.sub(r'<p id="dv-note"[^>]*>.*?</p>', "", html)
    anchor = "map from several fundamental domains."
    i = html.find(anchor)
    if i >= 0:
        j = html.find("</p>", i) + 4
        html = html[:j] + "\n      " + tag + html[j:]
    print(note, flush=True)
    Path(out_path).write_text(html)
    write_shape_map(src_dir, Path(out_path).parent)
    n = lambda D: sum(len(w["faces"]) for w in D["xwalls"])
    print(f"{out_path}: {len(html) / 1e6:.1f} MB; wall pieces {len(DC['xwalls'])}; triangles: half-cell {n(DC)}, "
          f"dodecahedron {n(V1)}, tenth {n(CH)}; regions {len(DC['regionWallsIds'])}", flush=True)


def write_shape_map(src_dir, out_dir):
    """symmetry_domain_shape_map.html beside the page: the 3-sphere map's shape map (shape_map3d.html,
    built in the main repository), framed in the page's "Shape map" section. Its own heading and
    introduction are left out (the section explains it), it takes the page's fonts and theme, and its
    stage has a fixed height so the page can size the frame to it."""
    html = (Path(src_dir) / "shape_map3d.html").read_text()
    html = re.sub(r"<header>.*?</header>", "", html, count=1, flags=re.S)
    html = html.replace("family=Chivo:wght@400;600;700&family=Chivo+Mono:wght@400;500",
                        "family=Spline+Sans:wght@400;500;600&family=Spline+Sans+Mono:wght@400;500")
    extra = """<style>
:root { --font-text: "Spline Sans", system-ui, sans-serif; --font-data: "Spline Sans Mono", ui-monospace, monospace; }
html, body { height: auto; background: transparent; }
.wrap { padding: 0; max-width: none; }
.stage { height: 600px; min-height: 0; }
.detail { max-height: 600px; }
</style>
<script>
// the page's theme, followed as it changes
(function () {
  let host = null;
  try { host = window.parent !== window ? window.parent.document.documentElement : null; } catch (e) {}
  if (!host) return;
  const sync = () => { const t = host.getAttribute("data-theme");
    if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); };
  sync();
  new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ["data-theme"] });
})();
</script>
"""
    i = html.rfind("</body>")
    html = html[:i] + extra + html[i:]
    out = Path(out_dir) / "symmetry_domain_shape_map.html"
    out.write_text(html)
    print(f"{out}: {len(html) / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
