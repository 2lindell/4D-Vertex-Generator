"""The Symmetry Domain page (cell_atlas2_template.html) fed from the working data alone.

The page and its features stay as they are; what feeds it is rebuilt:
  - projection: the half-cell view is the gnomonic projection of the 3-sphere (great circles and great spheres are
    straight lines and flat planes, so the half-cell's edges and faces are straight and flat), centred on the pole of
    the half-cell's four A1 corners (betas (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,1/2,1/2)), so all four lie at the same
    distance from the centre (radius 1 in the view);
  - walls: each wall's pieces (wall_parts.json: built by pieces.py out to the wall's exact edges and cuts, named from
    the working data), each drawn as a mesh of its own, with every image of it under the symmetry group that reaches
    into the half-cell, cut exactly at the half-cell's faces (the part outside discarded, no line drawn along the cut);
    copies are drawn exactly like the pieces. A wall's real edges are solid lines; the invisible lines in the middle of
    a wall (where a cut divides two of its pieces) dashed;
  - lines: the named lines' verified pieces (exact_defs.json), every image of them in the half-cell;
  - lists: a wall's boundary is its edges and points and it separates its regions; a region's boundary is its walls
    (partial where only part of the wall faces it), its lines and points; a line's boundary is what its ends lie on
    (all from working_data.json); each shape's example seed (working_data.json); the golden samples, uniform seeds
    and exact points named by the classifier;
  - a selected region shows the wall pieces with it on one side (the same triangles the walls draw) and their rims.
The page's overlays (the frame, H4 mirrors, extra-symmetry axes, the 2400 fundamental domain, the regular cells) are
computed from the group alone (overlays.py); the golden samples, uniform seeds and exact points are named by the
classifier; the rings are the exact cross-ring and main-ring lines. The dodecahedron views map the same content (the
half-cell's, under the group: the fold is one of its elements).

    python domain_page.py ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial import cKDTree

import chamber_domain
import dodeca_view as dv
import overlays
from core import CANON, CHART, beta_text, extended, home, in_half, label, seed_text
from cellframe import T as TCELL
from sphere_complex import GG, PROJ

HALF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])
CORNERS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5]])
G_CELL = 0.015          # (mesh coarsening in betas for the half-cell view; boundary vertices on a 4x finer grid)
G_DODECA = 0.1          # (... for the dodecahedron views, which hold about 20 half-cells)
G_DODECA_RIM = 0.012    # (... and for their rims, small at that scale; the edges are drawn from the exact lines)
LINE_TOL = 1e-3         # (polylines kept where they turn by more than this, in view coordinates)


# ---------------------------------------------------------------- projection

def frame():
    """The pole of the four A1 corners on the 3-sphere and an orthonormal frame of the view round it."""
    A = CORNERS @ TCELL
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    n = np.linalg.solve(A, np.ones(4))
    p = n / np.linalg.norm(n)
    U = np.linalg.svd(p[None, :])[2][1:]
    X = A / (A @ p)[:, None]
    return p, U, float(np.linalg.norm(X @ U.T, axis=1)[0])


PV, UV, RV = frame()


def _gnom(B):
    X = np.atleast_2d(np.asarray(B, float)) @ TCELL
    return (X / (X @ PV)[:, None]) @ UV.T / RV


def turn():
    """The view turned so the splitting mirror b3 = b4 (the plane through the middle of the tetrahedron) is the top face,
    level, with the corner V1 = b (1, 0, 0, 0) at the back (+y, away from the default camera); FLIP then mirrors it."""
    F = _gnom([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5]])
    z = np.cross(F[1] - F[0], F[2] - F[0])
    z /= np.linalg.norm(z)
    inner = _gnom([[.3, .3, .3, .1]])[0]                      # (a point with b3 > b4: inside, below the top face)
    if (inner - F[0]) @ z > 0:
        z = -z
    c = _gnom(CORNERS).mean(0)
    y = _gnom([[1, 0, 0, 0]])[0] - c
    y -= (y @ z) * z
    y /= np.linalg.norm(y)
    return np.vstack([np.cross(y, z), y, z]), c


ROT, CEN = turn()


def flip():
    """The picture mirrored across the splitting mirror (the half-cell drawn as the half b4 >= b3, as on the earlier
    page) and turned about the vertical so the corner M34 = b (0, 0, 1/2, 1/2) is at the back (+y); the top face stays
    the level top."""
    m = (_gnom([[0, 0, .5, .5]])[0] - CEN) @ ROT.T
    m = np.array([-m[0], m[1]])                                   # (after the mirror x -> -x)
    a = np.arctan2(m[0], m[1])                                    # (turn taking it to +y)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return Rz @ np.diag([-1.0, 1, 1])


FLIP = flip()


def view(B):
    """Betas (rows) to the half-cell view (centred on the corners' centroid, turned by ROT, mirrored and turned by
    FLIP)."""
    return (_gnom(B) - CEN) @ ROT.T @ FLIP.T


def chart(B):
    """Betas (rows) to the cell chart the dodecahedron views start from (cell_atlas.P, before the picture's flip)."""
    B = np.atleast_2d(np.asarray(B, float))
    return (B / B.sum(1, keepdims=True)) @ CHART


# ---------------------------------------------------------------- the half-cell and the group

def carry(V, g):
    """Points (betas) carried by g, normalised to sum 1 (nan where the sum is ~0: far outside)."""
    Y = V @ g.T
    s = Y.sum(1)
    ok = np.abs(s) > 1e-9
    Y[ok] /= s[ok, None]
    Y[~ok] = np.nan
    return Y


def inside(Y, tol=1e-9):
    return np.isfinite(Y).all(1) & ((Y @ HALF.T) > -tol).all(1)


def images(Vs):
    """The group elements (one per projective map) carrying some of the points Vs (betas) into the half-cell."""
    return [gi for gi in PROJ if inside(carry(Vs, GG[gi]), 1e-7).any()]


def clip_poly(P):
    """A convex polygon (betas, sum 1) cut to the half-cell (exact: the faces are linear in betas)."""
    for n in HALF:
        if not P:
            return []
        out = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            fa, fb = a @ n, b @ n
            if fa >= 0:
                out.append(a)
            if (fa >= 0) != (fb >= 0):
                out.append(a + fa / (fa - fb) * (b - a))
        P = out
    return P


def clip_tris(T):
    """Triangles (n,3,4) cut to the half-cell, with the index of the triangle each piece came from."""
    F = np.einsum("tvk,fk->tvf", T, HALF)
    allin = (F >= 0).all((1, 2))
    out, src = [T[allin]], [np.flatnonzero(allin)]
    for i in np.flatnonzero(~allin & (F.max(1) >= 0).all(1)):
        P = clip_poly(list(T[i]))
        for k in range(1, len(P) - 1):
            out.append(np.array([[P[0], P[k], P[k + 1]]]))
            src.append([i])
    C, S = np.concatenate(out), np.concatenate(src).astype(int)
    u, v = C[:, 1] - C[:, 0], C[:, 2] - C[:, 0]          # (slivers of no area, left along a face, dropped)
    keep = (u * u).sum(1) * (v * v).sum(1) - ((u * v).sum(1)) ** 2 > 1e-16
    return C[keep], S[keep]


def clip_polyline(P, tol=1e-9):
    """A polyline (betas) cut to the half-cell: its pieces inside (a piece lying in a face, to round-off, kept)."""
    runs, cur = [], []
    for a, b in zip(P[:-1], P[1:]):
        lo, hi = 0.0, 1.0
        for n in HALF:
            fa, fb = a @ n, b @ n
            if fa < -tol and fb < -tol:
                lo, hi = 1, 0
                break
            if fa < -tol:
                lo = max(lo, fa / (fa - fb))
            elif fb < -tol:
                hi = min(hi, fa / (fa - fb))
        if hi - lo <= 1e-12:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        p, q = a + lo * (b - a), a + hi * (b - a)
        if cur and np.linalg.norm(cur[-1] - p) < 1e-12:
            cur.append(q)
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = [p, q]
        if hi < 1:
            runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return runs


# ---------------------------------------------------------------- meshes

def coarsen(T, G, tags, GB=None):
    """Vertex merging on a grid in betas (boundary vertices kept exactly, so edges stay in place; or, for the small
    dodecahedron views, merged on the much finer grid GB): (vertices, faces, tag per face)."""
    Vu, inv = np.unique(T.reshape(-1, 4).round(12), axis=0, return_inverse=True)
    F = inv.ravel().reshape(-1, 3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    e, c = np.unique(E, axis=0, return_counts=True)
    bnd = np.zeros(len(Vu), bool)
    bnd[e[c == 1].ravel()] = True
    # (a boundary vertex is kept exactly where it is, so the wall's edges stay on its lines)
    key = np.round(Vu[:, :3] / G).astype(np.int64)
    key[bnd] = (np.round(Vu[bnd, :3] / GB).astype(np.int64) if GB
                else np.c_[np.flatnonzero(bnd), np.zeros((bnd.sum(), 2), np.int64)])
    key = np.c_[key, bnd]
    _, first, cl = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V, I = Vu[first], cl.ravel()[F]
    good = (I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 0] != I[:, 2])
    I, tg = I[good], [t for t, g in zip(tags, good) if g]
    _, k = np.unique(np.sort(I, 1), axis=0, return_index=True)
    k = np.sort(k)
    return V, I[k], [tg[i] for i in k]


def chains(segs, nd=9):
    """Segments (pairs of points) joined end to end into polylines."""
    key = lambda p: tuple(np.round(p, nd) + 0.0)
    uniq = {}
    for a, b in segs:                                   # (a segment listed twice, or of no length, once or not at all)
        ka, kb = key(a), key(b)
        if ka != kb:
            uniq.setdefault(tuple(sorted((ka, kb))), (a, b))
    segs = list(uniq.values())
    adj = {}
    for n, (a, b) in enumerate(segs):
        adj.setdefault(key(a), []).append(n)
        adj.setdefault(key(b), []).append(n)
    used, out = set(), []
    order = sorted(range(len(segs)), key=lambda n: min(len(adj[key(segs[n][0])]), len(adj[key(segs[n][1])])))
    for n in order:
        if n in used:
            continue
        used.add(n)
        line = [segs[n][0], segs[n][1]]
        for end in (1, 0):
            while True:
                at = key(line[-1] if end else line[0])
                m = next((m for m in adj.get(at, []) if m not in used), None)
                if m is None:
                    break
                used.add(m)
                a, b = segs[m]
                nxt = b if key(a) == at else a
                line.append(nxt) if end else line.insert(0, nxt)
        out.append(line)
    return out


def on_face(a, b, tol=1e-7):
    """Does the segment a-b lie in a face of the half-cell?"""
    return any(abs(a @ n) < tol and abs(b @ n) < tol for n in HALF)


def piece_rims(pc, edges):
    """A piece's boundary segments by kind: its wall's real edges (solid), its cuts (dashed, with the cutting wall),
    and its whole rim but the half-cell's faces and seams (the region's edges there)."""
    solid, dashed, rim = [], [], []
    for (a, na), (b, nb) in pc["boundary"]:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if na in edges:
            solid.append((a, b))
        elif na.startswith("cut by"):
            dashed.append((na[7:], a, b))
        if not na.startswith(("face", "mirror", "seam", "?")):
            rim.append((a, b))
    return solid, dashed, rim


def carry_segs(S_, g):
    """Segments [(a, b)] carried by g and cut to the half-cell."""
    if not S_:
        return []
    Sg = np.stack([carry(np.array([a for a, _ in S_]), g), carry(np.array([b for _, b in S_]), g)], 1)
    return [(p, q) for a, b in Sg[np.isfinite(Sg).all((1, 2))] for r in clip_polyline(np.array([a, b]))
            for p, q in zip(r[:-1], r[1:])]


def wall_in_cell(w, pieces, edges, G):
    """A wall's pieces (wall_parts.json) in the half-cell: each piece and every image of it under the group reaching into
    the half-cell, cut at the faces; a triangle lying on one already drawn for this wall (an image mapping the wall onto
    itself, or onto a copy already made) left out. Returns per piece: {between, tris (betas), solid, dashed, rim}."""
    out, tree, kept = [], None, np.zeros((0, 4))
    for pc in pieces:
        T = np.array(pc["tris"], float)
        if not len(T):
            continue
        V, F, _ = coarsen(T, G, [None] * len(T))
        solid, dashed, rim = piece_rims(pc, edges)
        rng = np.random.default_rng(0)
        Bs = np.array([a for a, _ in rim + solid]) if rim + solid else np.zeros((0, 4))
        Vs = V[rng.choice(len(V), min(500, len(V)), replace=False)]
        cand = images(np.vstack([Vs, Bs[:: max(1, len(Bs) // 200)]]) if len(Bs) else Vs)
        ident = [g for g in cand if np.allclose(GG[g], GG[g][0, 0] * np.eye(4))]
        rec = {"between": pc["between"], "tris": [], "solid": [], "dashed": [], "rim": []}
        for gi in ident + [g for g in cand if g not in ident]:
            Tg = carry(V, GG[gi])[F]
            ok = np.flatnonzero(np.isfinite(Tg).all((1, 2)))
            C, _ = clip_tris(Tg[ok])
            if not len(C):
                continue
            cen = C.mean(1)
            new = tree.query(cen)[0] > 0.5 * G if tree is not None else np.ones(len(C), bool)
            if not new.any():
                continue
            rec["tris"].append(C[new])
            kept = np.vstack([kept, cen[new]])
            tree = cKDTree(kept)
            g = GG[gi]
            rec["solid"] += carry_segs(solid, g)
            rec["rim"] += carry_segs(rim, g)
            for y in {y for y, _, _ in dashed}:
                rec["dashed"] += [(y, p, q) for p, q in carry_segs([(a, b) for y_, a, b in dashed if y_ == y], g)]
        if rec["tris"]:
            rec["tris"] = np.concatenate(rec["tris"])
            out.append(rec)
    return out


def wall_in_cell_chart(pieces, edges, G):
    """A wall's pieces in the half-cell, for the dodecahedron views (the group carries them over the whole 3-sphere: the
    fold is one of its elements): per piece {between, tris (betas), solid, dashed, rim}."""
    out = []
    for pc in pieces:
        T = np.array(pc["tris"], float)
        if not len(T):
            continue
        V, F, _ = coarsen(T, G, [None] * len(T), G_DODECA_RIM)
        solid, dashed, rim = piece_rims(pc, edges)
        out.append({"between": pc["between"], "tris": V[F], "solid": solid, "rim": rim, "dashed": dashed})
    return out


def line_in_cell(pieces):
    """A named line's verified pieces: every image in the half-cell, cut at the faces, as polylines (betas)."""
    out, seen = [], set()
    for P in pieces:
        P = np.asarray(P, float)
        if len(P) < 2:
            continue
        for gi in images(P):
            Y = carry(P, GG[gi])
            Y = Y[np.isfinite(Y).all(1)]
            for r in clip_polyline(Y):
                r = np.array(r)
                k = tuple(np.round(np.sort(r.mean(0)), 5))
                if k not in seen:
                    seen.add(k)
                    out.append(r)
    return out


def mesh(Q, nd):
    """Triangles (n,3,3) as shared vertices and index triples, rounded to nd places."""
    at, verts, faces = {}, [], []
    for t in Q:
        f = []
        for p in t:
            k = tuple(round(float(c), nd) for c in p)
            if k not in at:
                at[k] = len(verts)
                verts.append(list(k))
            f.append(at[k])
        if len(set(f)) == 3:
            faces.append(f)
    return verts, faces


def thin_line(P, tol):
    """A polyline's points kept where it turns by more than tol from the straight run (ends kept)."""
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

def rounded(o, n):
    """Every float in the page data rounded to n places (the only rounding: everything is computed at full precision)."""
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, dict):
        return {k: rounded(v, n) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, n) for v in o]
    return o


RINGS = ("C2a", "C2b", "C3", "D1", "D2", "D3")   # (the cross ring's stretches; B1 is the main ring)
DIM_TOKEN = {"region": "--x", "wall": "--xw", "line": "--xl", "point": "--xp"}


def order_key(tid):
    """The wiki's order for its shapes (core.CANON), then the others by letter and number."""
    ids = [c[0] for c in CANON]
    if tid in ids:
        return (0, ids.index(tid), 0)
    return (1, tid[0], int("".join(ch for ch in tid[1:] if ch.isdigit()) or 0))


def _label(b):
    return label(b)[0]


def page_samples():
    """The golden seeds (golden_seeds.json: seeds with golden weights), each named by the classifier."""
    from multiprocessing import Pool
    B = [home(s["beta"]) for s in json.load(open("golden_seeds.json"))["seeds"]]
    with Pool(4) as pool:
        names = pool.map(_label, B, chunksize=16)
    out = {}
    for b, n in zip(B, names):
        out.setdefault(n, []).append((b, f"β ∝ {beta_text(b)}<br>seed {seed_text(b)}"))
    return out


def page_uniform(names_of):
    """The uniform H4 seeds (the 15 sets of the cell's corners), each in the half-cell with its copy under the extra
    half-turn Q, named by the classifier."""
    from itertools import combinations
    E = extended()
    out = []
    for k in range(1, 5):
        for sub in combinations(range(4), k):
            b0 = np.zeros(4)
            b0[list(sub)] = 1.0
            n = _label(b0)
            seen = []
            for gi, g in enumerate(E):
                y = g @ b0
                if abs(y.sum()) < 1e-12:
                    continue
                y = home(y)
                if not in_half(y, 1e-9) or any(np.allclose(y, s_, atol=1e-9) for s_, _ in seen):
                    continue
                seen.append((y, gi >= len(E) // 2))
            for y, copy in seen:
                out.append({"id": n, "b": y, "copy": copy,
                            "hover": f"{n} · {names_of.get(n, n)}<br>uniform seed, β ∝ {beta_text(b0)}"
                                     + ("<br>(its copy under the extra half-turn)" if copy else "") + f"<br>seed {seed_text(y)}"})
    return out


def page_special(names_of):
    """The exact points that are not golden: the icosafold points on the cross ring (cross_ring_points.json)."""
    out = []
    for c in json.load(open("cross_ring_points.json"))["points"]:
        if c["name"] == "icosafold point":
            b = np.asarray(c["beta"], float)
            n = _label(b)
            out.append({"id": n, "b": b, "kind": "icosafold",
                        "hover": f"{n} · exact icosafold point (not golden)<br>{names_of.get(n, n)}<br>seed {seed_text(b)}"})
    return out


def page_types(work, counts):
    """The key: every shape of the working data, with its seed, its counts and its boundary lists."""
    types = []
    for tid in sorted(work["shapes"], key=order_key):
        s = work["shapes"][tid]
        t = {"id": tid, "name": s["name"], "listed": bool(s.get("listed")), "dim": s["dim"],
             "color": f"--ty-{tid}" if s.get("listed") else DIM_TOKEN[s["dim"]],
             "symbol": {"1200": "circle", "600": "diamond"}.get(str(s.get("vertices")), "square"),
             "samples": counts.get(tid, 0), "vertices": s.get("vertices"), "cells": s.get("cells"),
             "faces": s.get("faces"), "edges": s.get("edges"), "locus": None, "bounds": []}
        if s.get("seed"):
            b = home(s["seed"])
            t.update({"example": f"{seed_text(b)}  β ∝ {beta_text(b)}", "seed": seed_text(b), "betatext": beta_text(b)})
        if s.get("evenness"):
            t["evenness"] = s["evenness"]
        if tid in work["walls"]:
            v = work["walls"][tid]
            t.update({"bounds": v["edges"] + [p for p in v["boundary_points"] if p not in v["edges"]],
                      "separates": v["regions"], "where": "Wall between " + ", ".join(v["regions"])})
        elif tid in work["regions"]:
            v = work["regions"][tid]
            ext, ls = v.get("wall_extent", {}), v.get("lines", {})
            t.update({"bounds": v["walls"] + list(ls) + v.get("points", []),
                      "partial": [w for w in v["walls"] if ext.get(w) == "partial"] + [L for L, h in ls.items() if h == "partial"],
                      "where": "Region bounded by " + ", ".join(v["walls"])})
        elif tid in work.get("lines", {}):
            ends = list(dict.fromkeys(e["on"] for e in work["lines"][tid]["ends"] if e.get("on")))
            t.update({"bounds": ends, "where": "Line" + (" ending on " + ", ".join(ends) if ends else "")})
        else:
            t["where"] = s["dim"].capitalize()
        types.append(t)
    return types


def wall_hover(w, t, v):
    s = v.get("sphere", {})
    pts = v.get("boundary_points", v["points"])
    return (f"{w} · {t['name']}<br>wall between {' and '.join(v['regions'])}<br>edges: {', '.join(v['edges'])}"
            + (f"<br>points: {', '.join(pts)}" if pts else "")
            + (f"<br>on the 3-sphere: {s['patch_images']} piece image{'s' if s['patch_images'] != 1 else ''} round its "
               f"stabiliser of {s['stabiliser']}" if s else ""))


def main(out_path):
    DC = {}
    work = json.load(open("working_data.json"))
    P = json.load(open("wall_parts.json"))           # (each wall's pieces: wall_parts.py, from pieces.py)
    ex = json.load(open("exact_defs.json"))["lines"]
    names_of = {k: v["name"] for k, v in work["shapes"].items()}
    # (the overlays, from the group alone: overlays.py)
    edges_b, split_b, mirrors_b = overlays.frame()
    axes_b = overlays.axes()
    fdom_b, fcentre_b = overlays.fundamental_domain()
    regular_b, regular_pts = overlays.regular(names_of)
    seg9 = lambda a, b: np.array([a + (b - a) * k / 8 for k in range(9)])
    DC["edges"], DC["edgesOnRing"] = [view(e).tolist() for e in edges_b], []
    DC["split"] = view(split_b).tolist()
    DC["mirrors"] = [view(m).tolist() for m in mirrors_b]
    DC["qaxes"] = [{"pts": view(seg9(a["a"], a["b"])).tolist(), "order": a["order"], "hover": a["hover"]} for a in axes_b]
    DC["fdomain"] = [{"kind": f["kind"], "pts": view(f["pts"]).tolist(), "axis": view(f["axis"]).tolist() if f["axis"] else [],
                      "hover": f["hover"]} for f in fdom_b]
    DC["fcentre"] = view([fcentre_b])[0].tolist()
    DC["regular"] = [{"kind": r["kind"], "copy": r["copy"], "pts": view(r["pts"]).tolist(), "hover": r["hover"]} for r in regular_b]
    # (the key, the golden samples, the uniform seeds, the exact points and the rings: from the working data, the
    # classifier and the exact lines alone)
    samples = page_samples()
    DC["types"] = page_types(work, {k: len(v) for k, v in samples.items()})
    by = {t["id"]: t for t in DC["types"]}
    DC["totalSamples"] = sum(len(v) for v in samples.values())
    DC["samples"] = {k: [view([b])[0].tolist() + [h] for b, h in v] for k, v in samples.items()}
    uniform, special = page_uniform(names_of), page_special(names_of) + regular_pts
    DC["uniform"] = [{"id": u["id"], "q": view([u["b"]])[0].tolist(), "hover": u["hover"], "copy": u["copy"]} for u in uniform]
    DC["special"] = [{**{a: u[a] for a in u if a != "b"}, "q": view([u["b"]])[0].tolist()} for u in special]
    rings = {L: line_in_cell(ex[L]["points"]) for L in RINGS + ("B1",) if L in ex}
    DC["rings"] = [{"id": L, "pts": view(r).tolist()} for L in RINGS for r in rings.get(L, [])]
    DC["main"] = [{"id": "B1", "pts": view(r).tolist()} for r in rings.get("B1", [])]
    dv.split_edges_on_rings(DC)
    # (the shapes in each surface: the walls with a piece on it, from their pieces)
    surf = {}
    for w, v in P.items():
        for p in v["pieces"]:
            surf.setdefault(p["surface"], set()).add(w)
    DC["mirrorWalls"] = [{"name": f"the plane n · β = 0, n = {n[6:]}" if n.startswith("plane") else f"the surface {n}",
                          "ids": sorted(ws, key=order_key)} for n, ws in sorted(surf.items()) if len(ws) > 1]
    seg_ids = {"T1": True, "E1": True, "E2": True}             # (True: drawn only while its row is selected)
    ring_ids = set(RINGS) | {"B1"}

    cell = {"xwalls": [], "xlines": [], "segments": [], "regionShells": {}, "regionEdges": {}}
    src = {"xwalls": [], "xlines": [], "segments": []}           # (the cell chart, for the dodecahedron views)
    shell_cell, shell_src = {}, {}          # (region: its pieces' triangles and rims; the pieces themselves)
    for w, v in work["walls"].items():
        if w not in P:
            continue
        t, edges = by[w], set(v["edges"])
        hv = wall_hover(w, t, v)
        n_tri = 0
        for pc in wall_in_cell(w, P[w]["pieces"], edges, G_CELL):
            Q = view(pc["tris"].reshape(-1, 4)).reshape(-1, 3, 3)
            verts, faces = mesh(Q, 4)
            n_tri += len(faces)
            pair = pc["between"]
            phv = hv + (f"<br>this piece: between {' and '.join(pair)}" if pair else "<br>this piece: sides not settled")
            cell["xwalls"].append({"id": w, "hover": phv, "copy": False, "verts": verts, "faces": faces})
            for line in chains(pc["solid"]):
                cell["xlines"].append({"id": w, "outline": True, "copy": False, "pts": thin_line(view(line), LINE_TOL).tolist(),
                                       "hover": f"{w} · edge of the wall"})
            for y in {y for y, _, _ in pc["dashed"]}:
                for line in chains([(a, b) for y_, a, b in pc["dashed"] if y_ == y]):
                    cell["xlines"].append({"id": w, "outline": True, "copy": True, "pts": thin_line(view(line), LINE_TOL).tolist(),
                                           "hover": f"{w} · invisible line: where {y} meets it, between its pieces"})
            for r in pair or ():
                shell_cell.setdefault(r, {"tris": [], "rim": []})
                shell_cell[r]["tris"].append(Q)
                shell_cell[r]["rim"] += pc["rim"]
        print(f"{w}: {len(P[w]['pieces'])} pieces, {n_tri} triangles in the half-cell", flush=True)
        for pc in wall_in_cell_chart(P[w]["pieces"], edges, G_DODECA):
            pair = pc["between"]
            phv = hv + (f"<br>this piece: between {' and '.join(pair)}" if pair else "")
            src["xwalls"].append({"id": w, "hover": phv, "copy": False,
                                  "tris": chart(pc["tris"].reshape(-1, 4)).reshape(-1, 3, 3).tolist()})
            for line in chains(pc["solid"]):
                src["xlines"].append({"id": w, "outline": True, "copy": False, "pts": thin_line(chart(line), LINE_TOL / 3).tolist(),
                                      "hover": f"{w} · edge of the wall"})
            for y in {y for y, _, _ in pc["dashed"]}:
                for line in chains([(a, b) for y_, a, b in pc["dashed"] if y_ == y]):
                    src["xlines"].append({"id": w, "outline": True, "copy": True, "pts": thin_line(chart(line), LINE_TOL / 3).tolist(),
                                          "hover": f"{w} · invisible line: where {y} meets it, between its pieces"})
            for r in pair or ():
                shell_src.setdefault(r, {"tris": [], "rim": []})
                shell_src[r]["tris"].append(pc["tris"])
                shell_src[r]["rim"] += pc["rim"]
    for L, v in ex.items():
        if L not in by or L in ring_ids and L not in seg_ids:
            continue
        ends = [e["on"] for e in work["lines"].get(L, {}).get("ends", []) if e.get("on")]
        hv = f"{L} · {by[L]['name']}<br>exact line" + (f"<br>ends on {', '.join(dict.fromkeys(ends))}" if ends else "")
        key = "segments" if L in seg_ids or by[L]["listed"] else "xlines"
        extra = {"selOnly": seg_ids.get(L, False)} if key == "segments" else {}
        for r in line_in_cell(v["points"]):
            cell[key].append({"id": L, "copy": False, **extra, "pts": thin_line(view(r), LINE_TOL / 4).tolist(), "hover": hv})
        for Pc in v["points"]:
            Pc = np.asarray(Pc, float)
            if len(Pc) > 1:
                src[key].append({"id": L, "copy": False, **extra, "pts": thin_line(chart(Pc), LINE_TOL / 12).tolist(), "hover": hv})
    # a region's wall pieces (the sides of each piece) and their edges; the pieces in a file of their own beside the
    # page (symmetry_domain_regions/cell/<id>.json), fetched when the region is selected
    cell_files = {}
    for r, sh in shell_cell.items():
        if by.get(r, {}).get("dim") != "region":
            continue
        verts, faces = mesh(np.concatenate(sh["tris"]), 4)            # (the very triangles its walls' pieces draw)
        cell_files[r] = {"verts": verts, "faces": faces, "walls": work["regions"].get(r, {}).get("walls", [])}
        cell["regionEdges"][r] = [thin_line(view(np.array(c)), LINE_TOL).tolist() for c in chains(sh["rim"], 7)]
    for r, sh in shell_src.items():
        if by.get(r, {}).get("dim") != "region":
            continue
        src["xwalls"].append({"id": f"__region_{r}", "hover": "",
                              "tris": chart(np.concatenate(sh["tris"]).reshape(-1, 4)).reshape(-1, 3, 3).tolist()})

    # the dodecahedron views: the same walls and lines, every image under the group, cut to each view
    empty = {"samples": {}, "uniform": [], "special": [], "rings": [], "main": [], "qaxes": [], "regular": [],
             "tpatches": [], "mirrors": [], "fdomain": [], "fcentre": [0, 0, 0]}
    src = {**empty, **src}
    src["rings"] = [{"id": L, "pts": chart(r).tolist()} for L in RINGS for r in rings.get(L, [])]
    src["main"] = [{"id": "B1", "pts": chart(r).tolist()} for r in rings.get("B1", [])]
    src["samples"] = {k: [chart([b])[0].tolist() for b, _ in v] for k, v in samples.items()}
    src["uniform"] = [{**{a: u[a] for a in u if a != "b"}, "q": chart([u["b"]])[0].tolist()} for u in uniform]
    src["special"] = [{**{a: u[a] for a in u if a != "b"}, "q": chart([u["b"]])[0].tolist()} for u in special]
    src["qaxes"] = [{"pts": chart(seg9(a["a"], a["b"])).tolist(), "order": a["order"], "hover": a["hover"]} for a in axes_b]
    src["regular"] = [{"kind": r["kind"], "copy": r["copy"], "pts": chart(r["pts"]).tolist(), "hover": r["hover"]} for r in regular_b]
    src["mirrors"] = [chart(m).tolist() for m in mirrors_b]
    src["fdomain"] = [{"kind": f["kind"], "pts": chart(f["pts"]).tolist(), "axis": [], "hover": f["hover"]} for f in fdom_b]
    src["fcentre"] = chart([fcentre_b])[0].tolist()
    dv.use_centre((1, 0, 0, 0))
    v1 = dv.view(src)
    ch = chamber_domain.view(v1)
    dv.use_centre((0, 0, 1, 1))
    V1 = {k: v for k, v in v1.items() if k not in ("xwalls", "xlines", "segments")}
    CH = {k: v for k, v in ch.items() if k not in ("xwalls", "xlines", "segments")}
    # (a region's wall pieces in these views go in files of their own beside the page, fetched when it is selected:
    # symmetry_domain_regions/<view>/<id>.json; the page's own shells are emptied so the half-cell's are not shown)
    region_files = {}
    for vk, vw, new in (("v1", V1, v1), ("chambers", CH, ch)):
        dv.tidy_lines(new)
        for k_ in ("rings", "main", "qaxes", "regular"):
            vw[k_] = new[k_]
        dv.split_edges_on_rings(vw)
        vw["regionShells"], vw["regionEdges"] = {}, {}
        for w in [w for w in new["xwalls"] if w["id"].startswith("__region_")]:
            verts, faces = mesh(w["tris"], 3)
            r = w["id"][9:]
            region_files[f"{vk}/{r}.json"] = {"verts": verts, "faces": faces, "walls": work["regions"].get(r, {}).get("walls", [])}
        vw["regionWallsIds"] = sorted(f.split("/")[1][:-5] for f in region_files if f.startswith(vk + "/"))
        for s in [s for s in new["xlines"] if s["id"].startswith("__region_")]:
            vw["regionEdges"].setdefault(s["id"][9:], []).append(s["pts"])
        vw["xwalls"] = []
        for w in new["xwalls"]:
            if not w["id"].startswith("__region_"):
                verts, faces = mesh(w["tris"], 3)
                vw["xwalls"].append({**{a: b for a, b in w.items() if a != "tris"}, "verts": verts, "faces": faces})
        # (scraps are left out: a piece shorter than 0.015, where an image only grazes the view)
        length = lambda P: float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())
        vw["xlines"] = [{**s, "pts": thin_line(s["pts"], 2 * LINE_TOL).tolist()} for s in new["xlines"]
                        if not s["id"].startswith("__region_") and length(s["pts"]) >= 0.015]
        vw["segments"] = [{**s, "pts": thin_line(s["pts"], LINE_TOL / 4).tolist()} for s in new["segments"]]
        vw["regionEdges"] = {}          # (as before: these views show a region's wall pieces, without their edges)
        vw["tpatches"] = []
    for r, m in cell_files.items():
        region_files[f"cell/{r}.json"] = m
    DC.update(cell)
    DC["regionShells"], DC["regionWallsIds"] = {}, sorted(cell_files)
    DC["tpatches"] = []
    DC["camera"] = {"eye": {"x": 0.3, "y": -1.3, "z": 0.45}, "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}}
    DC["bounds"] = [view(CORNERS).min(0).tolist(), view(CORNERS).max(0).tolist()]     # (the box round the four corners)

    template = open("cell_atlas2_template.html").read()
    dump = lambda o: json.dumps(rounded(o, 4), separators=(",", ":"))
    dump3 = lambda o: json.dumps(rounded(o, 3), separators=(",", ":"))      # (the dodecahedron views: 3 places)
    import shutil
    from pathlib import Path
    rdir = Path(out_path).parent / "symmetry_domain_regions"
    shutil.rmtree(rdir, ignore_errors=True)
    for f, m in region_files.items():
        (rdir / f).parent.mkdir(parents=True, exist_ok=True)
        (rdir / f).write_text(dump(m) if f.startswith("cell/") else dump3(m))
    from place_colors import assign, css
    listed = [t["id"] for t in DC["types"] if t["listed"]]
    type_css = css(assign(listed, {t["id"]: t["dim"] for t in DC["types"]}))
    html = (template.replace("__TYPE_CSS__", type_css).replace("__DATA__", dump(DC)).replace("__DODECA__", "{}")
            .replace("__PIECE__", "{}").replace("__V1__", dump3(V1)).replace("__CHAMBERS__", dump3(CH))
            .replace("__COHESIVE__", "{}"))
    open(out_path, "w").write(html)
    n = lambda D: sum(len(w["faces"]) for w in D["xwalls"])
    print(f"{out_path}: {len(html) / 1e6:.1f} MB; triangles: cell {n(DC)}, dodecahedron {n(V1)}, chambers {n(CH)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
