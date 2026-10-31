"""Each wall's meshes, filled exactly inside its faces (wall_meshes.json), from wall_faces.json.

Each face of a wall (a polygon in the flat layout of one copy of the wall's surface, bounded by its exact edges and the
half-cell's faces) is triangulated inside its own outline (a constrained triangulation, `triangle`: no point added on
the outline, so the mesh meets the exact edges and nothing else), and every point is put on the exact surface: the point
of the surface whose layout coordinates are the point's (solved by Newton steps on the surface's equation, the layout's
two coordinates and the betas' sum), so the outline's points land back on the exact edges they came from.

    python face_meshes.py [walls]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import triangle
from shapely.geometry import Polygon

import arrangement as ar

OUT = os.environ.get("MESHES_OUT", "wall_meshes.json")
MAX_AREA = 2e-4           # (in the layout's own units, about a 0.02 triangle side)
STEP = 0.01               # (the outline's points kept about this far apart)


def lift(S, c0, U, uv, steps=30):
    """The points of S whose layout coordinates (U (x - c0)) are uv, betas summing to 1."""
    x = c0[None] + uv @ U
    x = x / x.sum(1, keepdims=True)
    for _ in range(steps):
        F = np.column_stack([ar.fvals(S, x), (x - c0) @ U.T - uv, x.sum(1) - 1])
        if np.abs(F).max() < 1e-15:
            break
        G = np.stack([(ar.fvals(S, x + 1e-7 * e) - ar.fvals(S, x - 1e-7 * e)) / 2e-7 for e in np.eye(4)], 1)
        J = np.stack([G, np.broadcast_to(U[0], x.shape), np.broadcast_to(U[1], x.shape), np.ones_like(x)], 1)
        x = x - np.linalg.solve(J, F[..., None])[..., 0]
    return x


def clean(rings):
    """The polygon's rings with points closer than 1e-9 merged, and the polygon made valid (the outline's pieces were
    joined within 1e-6, which can leave a pinch or a sliver): [outer, holes...], or [] where it has no area."""
    from shapely.validation import make_valid
    P = Polygon(rings[0], rings[1:])
    if not P.is_valid:
        P = make_valid(P)
        P = max((g for g in getattr(P, "geoms", [P]) if g.geom_type == "Polygon"), key=lambda g: g.area, default=None)
    if P is None or P.area < 1e-9:
        return []
    out = []
    for R in [P.exterior] + list(P.interiors):
        R = np.asarray(R.coords)[:-1]
        keep = [0] + [i for i in range(1, len(R)) if np.linalg.norm(R[i] - R[i - 1]) > 1e-9]
        R = R[keep]
        # (thinned to about one point per STEP along it, every corner kept: each kept point is still a point of the
        # exact edge; the outline is traced far finer than the view needs)
        if len(R) > 8:
            d_in = R - np.roll(R, 1, 0)
            d_out = np.roll(R, -1, 0) - R
            cosang = (d_in * d_out).sum(1) / (np.linalg.norm(d_in, axis=1) * np.linalg.norm(d_out, axis=1) + 1e-300)
            sel, run = [], STEP
            for i in range(len(R)):
                run += np.linalg.norm(d_in[i])
                if cosang[i] < 0.995 or run >= STEP:
                    sel.append(i)
                    run = 0.0
            R = R[sel]
        if len(R) >= 3:
            out.append(R)
    return out


def mesh_polygon(rings):
    """A polygon (outer ring, holes) triangulated inside its outline, no point added on it: (points, triangles)."""
    pts, segs, holes = [], [], []
    rings = clean(rings)
    if not rings:
        return np.zeros((0, 2)), np.zeros((0, 3), int)
    for k, ring in enumerate(rings):
        R = np.asarray(ring, float)
        if np.linalg.norm(R[0] - R[-1]) < 1e-15:
            R = R[:-1]
        n0 = len(pts)
        pts += R.tolist()
        segs += [[n0 + i, n0 + (i + 1) % len(R)] for i in range(len(R))]
        if k:
            holes.append(list(Polygon(R).representative_point().coords)[0])
    A = {"vertices": np.array(pts), "segments": np.array(segs)}
    if holes:
        A["holes"] = np.array(holes)
    T = triangle.triangulate(A, f"pYa{MAX_AREA}")
    return T["vertices"], T["triangles"]


def main(only=None):
    WF = json.load(open(os.environ.get("FACES_IN", "wall_faces.json")))
    try:
        out = json.load(open(OUT)) if only else {}
    except FileNotFoundError:
        out = {}
    for w, r in WF.items():
        if only and w not in only:
            continue
        pieces, worst, dropped = [], 0.0, 0
        for e in r.get("faces", []):
            S = ar.surface(e["name"])
            c0, U = np.array(e["c0"]), np.array(e["U"])
            for rings in e["polys"]:
                try:
                    uv, tri = mesh_polygon(rings)
                except Exception as ex:                    # (reported, not hidden)
                    print(f"  {w} on {e['name']}: a face could not be triangulated ({ex})", flush=True)
                    continue
                if not len(tri):
                    continue
                X = lift(S, c0, U, uv)
                offs = np.abs(ar.fvals(S, X)) / (np.linalg.norm(ar.grads(S, X), axis=1) + 1e-300)
                bad = ~(offs < 1e-9) | ~np.isfinite(X).all(1)
                if bad.any():
                    # (where the layout folds, the exact inverse can fail: the point put on the surface by the least
                    # change instead; a triangle with a point still off it, or carried far from its neighbours, dropped)
                    Y = ar.onto([S], np.nan_to_num(c0[None] + uv[bad] @ U))
                    X[bad] = Y
                    offs = np.abs(ar.fvals(S, X)) / (np.linalg.norm(ar.grads(S, X), axis=1) + 1e-300)
                ok = (offs < 1e-9) & np.isfinite(X).all(1)
                span = np.linalg.norm(X[tri] - X[tri][:, [1, 2, 0]], axis=2).max(1)
                keep = ok[tri].all(1) & (span < 0.1)
                dropped += int((~keep).sum())
                tri = tri[keep]
                worst = max(worst, float(offs[ok].max()) if ok.any() else 0.0)
                pieces.append({"surface": e["name"], "verts": X.round(12).tolist(), "tris": tri.tolist()})
        out[w] = pieces
        print(f"{w}: {len(pieces)} faces, {sum(len(p['tris']) for p in pieces)} triangles; "
              f"farthest point off its surface {worst:.1e}; triangles dropped (off it) {dropped}", flush=True)
        json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
