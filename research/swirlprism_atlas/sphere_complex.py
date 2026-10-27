"""The closed 3-sphere assembled from the half-cell pieces, nothing grown (sphere_complex.json).

The half-cell pieces (pieces.json) are taken as they are. Every symmetry of the group carries each of them somewhere on
the 3-sphere; a wall on the sphere is a set of these images, joined where one continues another across a face of the
half-cell (their face arcs lie on one curve) or where two of them are the same part of the wall (the half-cell is
larger than a fundamental domain, so some parts of a wall lie in it twice). Each wall's images fall into such joined
sets, all copies of one another: one of them is the whole wall, the others its copies.

For each wall: the patches and symmetries making up its whole piece (the set holding the first patch, as it is), and
the symmetries carrying that whole piece onto itself.

    python sphere_complex.py [walls]
"""
from __future__ import annotations

import json
import sys
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

import optimal_seeds as osd
from pieces import fval, grad, nrm
from t2_pieces import FOLD, extended

G = extended()
GG = np.concatenate([G, G @ FOLD])
FACES = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1.0]])
FACE_LABELS = ("face ", "mirror ")


def key(g):
    """A projective map's key: g and -g are one map of the betas."""
    v = g.ravel()
    v = v * np.sign(v[np.flatnonzero(np.abs(v) > 1e-9)[0]])
    return tuple(np.round(v, 6))


INDEX = {}
for i, g in enumerate(GG):
    INDEX.setdefault(key(g), i)
PROJ = sorted(set(INDEX.values()))                       # (one element per projective map)


def mul(i, j):
    return INDEX[key(GG[i] @ GG[j])]


def surf_dist(s, X):
    X = np.atleast_2d(X)
    if isinstance(s, osd.Plane):
        return np.abs(X @ s.n) / np.linalg.norm(s.n - s.n.mean())
    return np.array([abs(fval(s, x)) / (np.linalg.norm(grad(s, x)) + 1e-300) for x in X])


def carry(X, gi):
    Y = X @ GG[gi].T
    s = Y.sum(1)
    ok = np.abs(s) > 1e-9
    Y[ok] = Y[ok] / s[ok, None]
    Y[~ok] = np.nan
    return Y


def in_half(Y, tol=1e-9):
    return (Y > -tol).all(1) & (Y[:, 2] >= Y[:, 3] - tol)


def on_face(Y, tol=1e-9):
    return (np.abs(Y @ FACES.T) < tol).any(1)


def lines_on(X, S, LIB, lines, tol=2e-3):
    """For each point X (on the surface S): the named line it lies on (any image of its verified pieces), or None."""
    from pieces import line_pieces_on
    out = [None] * len(X)
    for L, v in lines.items():
        ps = line_pieces_on(v, S, GG, LIB)
        if not ps:
            continue
        d = cKDTree(np.vstack([p for p, _ in ps])).query(X)[0]
        for i in np.flatnonzero(d < tol):
            out[i] = out[i] or L
    return out


def patch_data(pt, LIB, h, rng, lines, through):
    T = np.array(pt["tris"], float)
    V = np.unique(T.reshape(-1, 4).round(12), axis=0)
    face = np.array([x for sg in pt["boundary"] for x, n in sg if n.startswith(FACE_LABELS)], float).reshape(-1, 4)
    # (a face arc along a named line is where a different copy meets the wall, an edge, unless the wall is verified
    # to go on through that line)
    on = lines_on(face, LIB[pt["surface"]], LIB, lines) if len(face) else []
    at_lines = sorted({L for L in on if L and L not in through})
    face = face[[L is None or L in through for L in on]] if len(face) else face
    seam = np.array([x for sg in pt["boundary"] for x, n in sg if n.startswith("seam ")], float).reshape(-1, 4)
    seam_planes = sorted({n[5:] for sg in pt["boundary"] for x, n in sg if n.startswith("seam ")})
    inner = V[rng.choice(len(V), min(300, len(V)), replace=False)]
    bound = np.array([x for sg in pt["boundary"] for x, n in sg], float).reshape(-1, 4)
    return {"S": LIB[pt["surface"]], "V": V, "tree": cKDTree(V), "face": face,
            "face_tree": cKDTree(face) if len(face) else None, "at_lines": at_lines,
            "seam": seam, "seam_planes": seam_planes, "seam_tree": cKDTree(seam) if len(seam) else None,
            "bound_tree": cKDTree(bound) if len(bound) else None, "inner": inner, "h": pt.get("h", 0.008)}


def relations(D, LIB):
    """(p, q, g): the image of patch q by g joins patch p, across a face (face arcs on one curve) or as the same part
    (inner points of the image on p)."""
    out = []
    for p, dp in enumerate(D):
        for q, dq in enumerate(D):
            for gi in PROJ:
                if p == q and gi == PROJ[0]:
                    continue
                how = None
                if len(dq["face"]) and dp["face_tree"] is not None:
                    Y = carry(dq["face"], gi)
                    m = np.isfinite(Y).all(1)
                    Y = Y[m]
                    if len(Y):
                        m = in_half(Y) & on_face(Y)
                        Y = Y[m]
                    if len(Y) >= 3:
                        close = dp["face_tree"].query(Y)[0] < dp["h"]
                        if close.sum() >= 3 and surf_dist(dp["S"], Y[close][:5]).max() < 1e-8:
                            how = "across a face"
                if how is None and len(dq["seam"]) and dp["seam_tree"] is not None and p != q:
                    Y = carry(dq["seam"], gi)
                    Y = Y[np.isfinite(Y).all(1)]
                    if len(Y):
                        Y = Y[in_half(Y)]
                    if len(Y) >= 3:
                        close = dp["seam_tree"].query(Y)[0] < dp["h"]
                        Z = carry(dq["inner"][:8], gi)          # (the image lies on the plane p's seam names)
                        Z = Z[np.isfinite(Z).all(1)]
                        # (a seam on both sides: p's own seam points met as well, not a touch at a point)
                        back = cKDTree(Y).query(dp["seam"])[0] < dp["h"] if len(Y) else np.zeros(0, bool)
                        if close.sum() >= 10 and back.sum() >= 10 and surf_dist(dp["S"], Y[close][:5]).max() < 1e-8 and len(Z) and any(
                                surf_dist(LIB[n], Z).max() < 1e-8 for n in dp["seam_planes"] if n in LIB):
                            how = "at a seam"
                if how is None:
                    # (points the symmetry moves: near its fixed set every point lands near itself, which says
                    # nothing about the image covering the same part)
                    Y = carry(dq["inner"], gi)
                    m = np.isfinite(Y).all(1) & (np.linalg.norm(np.nan_to_num(Y) - dq["inner"], axis=1) > 2 * dp["h"])
                    Y = Y[m]
                    if len(Y):
                        Y = Y[in_half(Y)]
                    if len(Y) and dp["bound_tree"] is not None:   # (well inside p: not touching it along an edge)
                        Y = Y[dp["bound_tree"].query(Y)[0] > 2 * dp["h"]]
                    if len(Y) >= 5:
                        close = dp["tree"].query(Y)[0] < dp["h"] / 2
                        if close.sum() >= 5 and surf_dist(dp["S"], Y[close][:5]).max() < 1e-8:
                            how = "same part"
                if how:
                    out.append((p, q, gi, how))
    return out


def component(rel, n_patches, start=(0, None)):
    """The joined set holding patch 0 as it is: (patch, symmetry) nodes reached through the relations."""
    e0 = PROJ[0]
    start = (start[0], e0)
    seen = {start}
    queue = deque([start])
    while queue:
        p, g = queue.popleft()
        for pp, q, h, _ in rel:                       # (q carried by h joins pp; so q by g·h joins pp by g)
            if pp == p:
                nxt = (q, mul(g, h))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
            if q == p:                                 # (and the other way: pp by g·h^-1 joins q by g)
                hi = INDEX[key(np.linalg.inv(GG[h]))]
                nxt = (pp, mul(g, hi))
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if len(seen) > 5000:
            break
    return seen


def main(only=None):
    P = json.load(open("pieces.json"))
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    rng = np.random.default_rng(0)
    work = json.load(open("working_data.json"))
    lines = json.load(open("exact_defs.json"))["lines"]
    out = {}
    for w, rec in P.items():
        if only and w not in only:
            continue
        through = set(work["walls"][w]["through"])
        D = [patch_data(pt, LIB, 0.008, rng, lines, through) for pt in rec["patches"]]
        rel = relations(D, LIB)
        comp = component(rel, len(D))
        stab = sorted(g for p, g in comp if p == 0)
        per_patch = {k: sum(1 for p, _ in comp if p == k) for k in range(len(D))}
        out[w] = {"patches": [pt["surface"] for pt in rec["patches"]],
                  "joins": [{"patch": p, "image_of": q, "by": g, "how": how} for p, q, g, how in rel],
                  "whole_piece": sorted([list(x) for x in comp]),
                  "stabiliser": stab,
                  "face_lines_not_edges": sorted({L for d in D for L in d["at_lines"]} - set(work["walls"][w]["edges"]))}
        hows = {}
        for _, _, _, how in rel:
            hows[how] = hows.get(how, 0) + 1
        print(f"{w}: {len(D)} patches; joins {hows}; whole piece = {len(comp)} patch images "
              f"(per patch {per_patch}); stabiliser {len(stab)}"
              + (f"; meets a copy along {out[w]['face_lines_not_edges']} on a face (not edges in the working data)"
                 if out[w]["face_lines_not_edges"] else ""), flush=True)
    json.dump(out, open("sphere_complex.json", "w"), indent=1)


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
