"""Each wall's pieces as meshes of their own (wall_parts.json): every patch of pieces.json split exactly along its cuts,
each part named from the working data alone.

pieces.py meshes a wall's patch out to its edges (each boundary point solved exactly onto the wall's surface and the
edge's) and keeps the cuts across it (the curves where other walls meet it) as traced curves. Here the patch is split
along them: a triangle a cut passes through is divided at the exact points where the cut crosses its sides (each solved
onto the wall's surface and the cutting wall's surface, the same point for the two triangles sharing the side).

The parts are then named from the working data, as a puzzle (no point is classified):
  * a part bounded by lines is the working data's piece of the wall with those lines;
  * across a cut by Y, the neighbouring part is the same piece, or the piece with the cut's side swapped (the working
    data's cut lists the two regions it changes between);
  * across a seam (where two of a flat wall's planes meet), the neighbouring part is the same piece.
A traced cut follows the two surfaces past where the cutting wall really is; where it has the same piece on both sides
it divides nothing and is dropped, the parts joined. What is left of each cut is the wall's invisible line there.

Each piece keeps its triangles, its boundary (the patch's edge segments along it, labelled with their edges, and its
cut segments, labelled "cut by Y") and the two regions it lies between. Regions use these pieces directly.

    python wall_parts.py [X14,X41,...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import optimal_seeds as osd
from pieces import dens, fval, nrm, project
from sphere_complex import GG, PROJ

OUT = "wall_parts.json"
SIDES = ((0, 1), (1, 2), (2, 0))


def split(V, F, S, sY, run_tree, h):
    """The mesh (vertices V, triangles F) split along the curve where the surface sY meets the wall's surface S, near
    the traced cut (run_tree): new vertices at the exact crossings; returns (V, F, the cut's sides as vertex pairs)."""
    V = np.asarray(V, float)
    near = run_tree.query(V)[0] < 3 * h
    f = np.full(len(V), np.nan)
    f[near] = [fval(sY, v) for v in V[near]]
    V = list(V)
    at = {}

    def cross(u, v):
        k = (min(u, v), max(u, v))
        if k not in at:
            a, b = V[u], V[v]
            x = nrm(project([S, sY], a + f[u] / (f[u] - f[v]) * (b - a)))
            at[k] = len(V) if run_tree.query(x)[0] < 1.5 * h else -1
            if at[k] >= 0:
                V.append(x)
        return at[k]

    def crossed(u, v):
        return u < len(f) and v < len(f) and np.isfinite(f[u]) and np.isfinite(f[v]) and f[u] * f[v] < 0 \
            and cross(u, v) >= 0

    out, cut = [], set()
    for t in F:
        e = [(t[i], t[j]) for i, j in SIDES]
        c = [crossed(u, v) for u, v in e]
        if sum(c) != 2:
            out.append(list(t))                            # (not crossed, or the cut ends inside it: kept whole)
            continue
        i = c.index(False)                                 # (the side not crossed: its two corners together)
        a, b = e[i]
        o = t[(i + 2) % 3]                                 # (the corner alone on its side of the cut)
        p, q = cross(b, o), cross(o, a)
        out += [[o, p, q], [a, b, p], [a, p, q]]
        cut.add((min(p, q), max(p, q)))
    return np.array(V), np.array(out), cut


class Image:
    """A library surface carried by a group element g: f(x) = surface(g^-1 x)."""

    def __init__(self, s, g):
        self.s, self.ginv, self.name = s, np.linalg.inv(g), f"{s.name}@image"

    def f(self, X):
        Y = np.atleast_2d(X) @ self.ginv.T
        return self.s.f(Y / Y.sum(1, keepdims=True))


def cut_curves(w, S, Pall, work_w, LIB, tol=1e-7):
    """The cuts across the wall's surface S, exact and only as far as each cutting wall Y really goes: Y's own edge
    on the wall (the boundary segments of Y's patches labelled w, each point solved onto both surfaces by pieces.py),
    carried by every group element that lays it on S in the half-cell; with the image of Y's surface there, for the
    side test. A cut whose wall does not end on this one (the two cross, both going on) has no such segments: then the
    curve traced by pieces.py is used."""
    out = {}
    for c in work_w.get("cuts", []):
        Y = c["by"]
        for pt in Pall.get(Y, {}).get("patches", []):
            segs = [(np.asarray(a, float), np.asarray(b, float)) for (a, na), (b, nb) in pt["boundary"] if na == w]
            if not segs or pt["surface"] not in LIB:
                continue
            A = np.array([a for a, _ in segs])
            B = np.array([b for _, b in segs])
            for gi in PROJ:
                g = GG[gi]
                Ya, Yb = A @ g.T, B @ g.T
                sa, sb = Ya.sum(1), Yb.sum(1)
                if (np.abs(sa) < 1e-9).any() or (np.abs(sb) < 1e-9).any():
                    continue
                Ya, Yb = Ya / sa[:, None], Yb / sb[:, None]
                ok = (Ya > -tol).all(1) & (Ya[:, 2] >= Ya[:, 3] - tol) & (Yb > -tol).all(1) & (Yb[:, 2] >= Yb[:, 3] - tol)
                if not ok.any():
                    continue
                on = np.array([abs(fval(S, y)) < 1e-6 for y in Ya[ok]])
                if not on.all():
                    continue
                pts = np.vstack([dens(np.array([a, b]), 0.002) for a, b in zip(Ya[ok], Yb[ok])])
                out.setdefault(Y, []).append((pts, Image(LIB[pt["surface"]], g)))
    return out


def patch_parts(pt, eqw, LIB, cuts):
    """A patch split along its cuts: (vertices, triangles, part of each triangle, cut sides {vertex pair: Y})."""
    h = float(pt.get("h", 0.008))
    T = np.array(pt["tris"], float)
    V, inv = np.unique(T.reshape(-1, 4).round(11), axis=0, return_inverse=True)
    F = inv.ravel().reshape(-1, 3)
    S = LIB[pt["surface"]]
    curves = []
    for Y, lst in cuts.items():
        curves += [(Y, pts, sY) for pts, sY in lst]
    for Y, P in pt.get("cuts", []):                    # (the traced curve, for a cut with no edge segments)
        if Y not in cuts and eqw.get("cuts", {}).get(Y, {}).get("surface") in LIB:
            curves.append((Y, np.asarray(P, float), LIB[eqw["cuts"][Y]["surface"]]))
    cut_edges = {}
    for Y, pts, sY in curves:
        V, F, cut = split(V, F, S, sY, cKDTree(pts), h)
        cut_edges.update({k: Y for k in cut})
    E = np.concatenate([np.sort(F[:, list(s)], 1) for s in SIDES])
    tri = np.tile(np.arange(len(F)), 3)
    o = np.lexsort((E[:, 1], E[:, 0]))
    E, tri = E[o], tri[o]
    same = (E[1:] == E[:-1]).all(1)
    a, b = tri[:-1][same], tri[1:][same]
    keep = np.array([(int(x), int(y)) not in cut_edges for x, y in E[:-1][same]], bool)
    G = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(len(F), len(F)))
    _, comp = connected_components(G, directed=False)
    return V, F, comp, cut_edges


def wall_pieces(w, rec, eqw, work_w, lines, LIB, Pall):
    """The wall's pieces, named by the working data: [{patch, surface, tris, boundary, between}], and notes."""
    known = [tuple(sorted(p["between"])) for p in work_w.get("pieces", [])]
    plines = {tuple(sorted(p["between"])): set(p["lines"]) for p in work_w.get("pieces", [])}
    if not known and len(work_w.get("regions", [])) == 2:
        known = [tuple(sorted(work_w["regions"]))]
        plines[known[0]] = set()
    cut_sides = {c["by"]: tuple(c["between"]) for c in work_w.get("cuts", []) if len(c.get("between", [])) == 2}
    parts = []                                             # (patch, V, F, triangle indices, edge segments, cut sides)
    links = []                                             # (part i, part j, "seam" or the cut's wall)
    for k, pt in enumerate(rec["patches"]):
        if not pt["tris"]:
            continue
        V, F, comp, cut_edges = patch_parts(pt, eqw, LIB, cut_curves(w, LIB[pt["surface"]], Pall, work_w, LIB))
        base = len(parts)
        ids = {c: base + n for n, c in enumerate(np.unique(comp))}
        C = V[F].mean(1)
        tree = cKDTree(C)
        segs = {i: [] for i in ids.values()}
        for (x, nx), (y, ny) in pt["boundary"]:
            m = (np.asarray(x, float) + np.asarray(y, float)) / 2
            segs[ids[comp[tree.query(m)[1]]]].append([[x, nx], [y, ny]])
        for c, i in ids.items():
            parts.append({"patch": k, "surface": pt["surface"], "V": V, "F": F, "idx": np.flatnonzero(comp == c),
                          "segs": segs[i], "cuts": []})
        by_edge = {}
        for t_i, t in enumerate(F):
            for i_, j_ in SIDES:
                e = (min(t[i_], t[j_]), max(t[i_], t[j_]))
                if e in cut_edges:
                    by_edge.setdefault(e, []).append(ids[comp[t_i]])
        for e, ps in by_edge.items():
            for p_ in ps:
                others = [q for q in ps if q != p_]
                parts[p_]["cuts"].append((e, cut_edges[e], others[0] if others else None))
            if len(ps) == 2 and ps[0] != ps[1]:
                links.append((ps[0], ps[1], cut_edges[e]))
    # (seams: parts of different patches with seam segments lying together)
    seam_pts = [(i, (np.asarray(x, float) + np.asarray(y, float)) / 2) for i, p in enumerate(parts)
                for (x, nx), (y, ny) in p["segs"] if nx.startswith("seam")]
    if seam_pts:
        tree = cKDTree(np.array([m for _, m in seam_pts]))
        for a_, b_ in tree.query_pairs(0.004):
            i, j = seam_pts[a_][0], seam_pts[b_][0]
            if parts[i]["patch"] != parts[j]["patch"]:
                links.append((i, j, "seam"))
    links = list({(min(i, j), max(i, j), y) for i, j, y in links})
    # (the puzzle: each part's possible pieces, narrowed by its lines and by its neighbours until nothing changes)
    dom = []
    for p in parts:
        Lp = {n for (_, n), _ in p["segs"] if n in lines} | {n for _, (_, n) in p["segs"] if n in lines}
        cand = [x for x in known if Lp <= plines[x]] if Lp else list(known)
        dom.append(set(cand))

    def swap(x, y):
        s = cut_sides.get(y)
        if not s or y == "seam":
            return None
        a, b = s
        if a in x and b not in x:
            return tuple(sorted((set(x) - {a}) | {b}))
        if b in x and a not in x:
            return tuple(sorted((set(x) - {b}) | {a}))
        return None

    def fits(x, y_, how):
        return y_ == x or (how != "seam" and swap(x, how) == y_)
    changed = True
    while changed:
        changed = False
        for i, j, how in links:
            for a_, b_ in ((i, j), (j, i)):
                keep = {x for x in dom[a_] if any(fits(x, y_, how) for y_ in dom[b_])}
                if keep != dom[a_]:
                    dom[a_], changed = keep, True
    name = [next(iter(d)) if len(d) == 1 else None for d in dom]
    # (parts with the same piece on both sides of a cut, or of a seam within one patch, are one piece)
    root = list(range(len(parts)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for i, j, how in links:
        if name[i] and name[i] == name[j] and parts[i]["patch"] == parts[j]["patch"]:
            root[find(i)] = find(j)
    groups = {}
    for i in range(len(parts)):
        groups.setdefault(find(i), []).append(i)
    out, notes = [], []
    for r, mem in groups.items():
        p0 = parts[mem[0]]
        V, F = p0["V"], p0["F"]
        idx = np.concatenate([parts[i]["idx"] for i in mem])
        inner = set(mem)
        segs = [s for i in mem for s in parts[i]["segs"]]
        for i in mem:
            for e, y, other in parts[i]["cuts"]:
                if other is not None and other not in inner:      # (the cut divides two pieces here: kept)
                    segs.append([[V[e[0]].round(10).tolist(), f"cut by {y}"], [V[e[1]].round(10).tolist(), f"cut by {y}"]])
        segs = [s for s in {json.dumps(s): s for s in segs}.values()]
        nm = name[r]
        if nm is None:
            notes.append(f"a piece of {len(idx)} triangles on {p0['surface']} not settled (could be "
                         f"{', '.join('|'.join(x) for x in sorted(dom[r])) or 'none of the working data pieces'})")
        out.append({"patch": p0["patch"], "surface": p0["surface"], "tris": V[F[idx]].round(10).tolist(),
                    "boundary": segs, "between": list(nm) if nm else None})
    return out, notes


def main(only=None):
    P = json.load(open("pieces.json"))
    eq = json.load(open("wall_equations.json"))["walls"]
    work = json.load(open("working_data.json"))["walls"]
    lines = set(json.load(open("exact_defs.json"))["lines"])
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    out = json.load(open(OUT)) if only and os.path.exists(OUT) else {}
    for w, rec in P.items():
        if only and w not in only:
            continue
        pcs, notes = wall_pieces(w, rec, eq.get(w, {}), work.get(w, {}), lines, LIB, P)
        out[w] = {"pieces": pcs, "notes": notes}
        got = Counter("|".join(p["between"]) if p["between"] else "?" for p in pcs)
        print(f"{w}: {len(pcs)} pieces {dict(got)}" + (f"; {len(notes)} not settled" if notes else ""), flush=True)
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
