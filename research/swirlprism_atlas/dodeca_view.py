"""The atlas seen from the dodecahedron around M34 instead of the half-cell.

M34 = beta (0, 0, 1, 1) is a vertex of a 600-cell (its orbit under the swirlprism group has 120 points). The points
of the 3-sphere nearer to M34 than to the rest of that orbit form a regular dodecahedron (a cell of the dual
120-cell); 120 of them tile the sphere, and the 10 group elements that fix M34 turn it onto itself, so it holds
ten copies of every seed. This module takes the cell view's geometry (points, polylines, polygons in the cell's
gnomonic chart), maps it back to seeds, applies every group element and keeps what falls inside the
dodecahedron, clipped to its 12 faces. The picture is the gnomonic chart at M34 (great circles stay straight),
turned so the axis of M34's 5-fold rotation is vertical.
"""
from __future__ import annotations

import numpy as np
from cell_atlas import P
from cellframe import T

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

E = np.stack(group_elements(named_symmetry("h4_swirlprism")))
_A = np.vstack([P.T, np.ones(4)])                    # cell chart: q = (beta / sum) @ P


def _seed(q):
    """Seed direction (unit 4-vector) of a point given in the cell chart (it may lie outside the cell)."""
    b = np.linalg.solve(_A, np.append(np.asarray(q[:3], float), 1.0))
    x = b @ T
    return x / np.linalg.norm(x)


def _frame():
    c = np.array([0, 0, 1, 1.0]) @ T
    c /= np.linalg.norm(c)
    orbit = np.unique(np.round(E @ c, 9), axis=0)
    near = orbit[np.argsort(-(orbit @ c))[1:13]]     # its 12 neighbours: one dodecahedron face each
    stab = [g for g in E if np.allclose(g @ c, c, atol=1e-9)]
    perp = np.linalg.svd(c[None])[2][1:]               # basis of c's orthogonal 3-space
    axis = None
    for g in stab:                                   # an element of order 5: its axis in c's 3-space
        R = perp @ g @ perp.T
        if np.allclose(np.linalg.matrix_power(R, 5), np.eye(3), atol=1e-8) and not np.allclose(R, np.eye(3)):
            w, v = np.linalg.eig(R)
            axis = np.real(v[:, np.argmin(np.abs(w - 1))])
            break
    z = axis / np.linalg.norm(axis)
    x = np.linalg.svd(z[None])[2][1]
    y = np.cross(z, x)
    B = np.vstack([x, y, z]) @ perp                  # chart basis (3 x 4)
    d = c - near                                     # inside: X . d_k >= 0
    return c, B, d


C, B, D = _frame()
_HA = (B @ D.T).T                                    # in chart coordinates u: X ~ c + B^T u, so a_k . u + b_k >= 0
_HB = D @ C


def _chart(X):
    X = np.atleast_2d(X)
    return (X @ B.T) / (X @ C)[:, None]


def _inside(U, tol=1e-9):
    return np.all(U @ _HA.T + _HB >= -tol, axis=1)


def _images(Q):
    """Seeds of points Q (cell chart), mapped by every group element: array (|E|, len(Q), 4)."""
    X = np.array([_seed(q) for q in Q])
    return np.einsum("gij,nj->gni", E, X)


def _survivors(pts):
    """(chart points) of the images of a point list that can reach inside the dodecahedron."""
    Y = _images(pts)                                  # (g, n, 4)
    den = Y @ C                                       # (g, n)
    ok = np.all(den > 0.3, axis=1)
    U = (Y @ B.T) / np.where(den > 0.3, den, 1)[..., None]
    f = U @ _HA.T + _HB                               # (g, n, 12)
    ok &= ~np.any(np.all(f < -1e-9, axis=1), axis=1)  # not wholly beyond one face
    return [U[g] for g in np.flatnonzero(ok)]


def points(Q):
    """Images of each point inside the dodecahedron: list (per point) of chart positions."""
    if not len(Q):
        return []
    Y = _images(Q)
    out = []
    for n in range(len(Q)):
        Yn = Y[:, n]
        Yn = Yn[Yn @ C > 0.5]
        U = _chart(Yn)
        U = U[_inside(U)]
        keep = []
        for u in U:
            if not any(np.linalg.norm(u - w) < 1e-7 for w in keep):
                keep.append(u)
        out.append([np.round(u, 4).tolist() for u in keep])
    return out


def _clip_segment(a, b):
    t0, t1 = 0.0, 1.0
    d = b - a
    for h, k in zip(_HA, _HB):
        fa, fd = h @ a + k, h @ d
        if abs(fd) < 1e-15:
            if fa < 0:
                return None
            continue
        t = -fa / fd
        if fd > 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return a + t0 * d, b - (1 - t1) * d


def polylines(lines):
    """Every image of every polyline (cell chart), clipped to the dodecahedron: list per input of pieces."""
    out = []
    for pts in lines:
        if len(pts) < 2:
            out.append([])
            continue
        pieces = []
        for U in _survivors(pts):
            cur = []
            for a, b in zip(U[:-1], U[1:]):
                s = _clip_segment(a, b)
                if s is None:
                    if len(cur) > 1:
                        pieces.append(cur)
                    cur = []
                    continue
                p, q = s
                if cur and np.linalg.norm(cur[-1] - p) < 1e-9:
                    cur.append(q)
                else:
                    if len(cur) > 1:
                        pieces.append(cur)
                    cur = [p, q]
                if np.linalg.norm(q - b) > 1e-9:     # left the dodecahedron inside this segment
                    pieces.append(cur)
                    cur = []
            if len(cur) > 1:
                pieces.append(cur)
        out.append([[np.round(u, 4).tolist() for u in pc] for pc in pieces if len(pc) > 1])
    return out


def _clip_polygon(U):
    poly = list(U)
    for h, k in zip(_HA, _HB):
        new = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            fa, fb = h @ a + k, h @ b + k
            if fa >= -1e-12:
                new.append(a)
            if (fa >= -1e-12) != (fb >= -1e-12):
                new.append(a + (fa / (fa - fb)) * (b - a))
        poly = new
        if len(poly) < 3:
            return None
    return poly


def polygons(polys):
    """Every image of every polygon (cell chart), clipped to the dodecahedron: list per input of polygons."""
    out = []
    for pts in polys:
        if len(pts) < 3:
            out.append([])
            continue
        got = []
        for U in _survivors(pts):
            pc = _clip_polygon(U)
            if pc is not None:
                got.append([np.round(u, 4).tolist() for u in pc])
        out.append(got)
    return out


def dodecahedron_edges():
    """The 30 edges of the dodecahedron in the chart."""
    from itertools import combinations

    from scipy.spatial import HalfspaceIntersection
    hs = HalfspaceIntersection(np.hstack([-_HA, -_HB[:, None]]), np.zeros(3))
    V = []
    for v in hs.intersections:
        if not any(np.linalg.norm(v - w) < 1e-9 for w in V):
            V.append(v)
    V = np.array(V)
    on = [set(np.flatnonzero(np.abs(V @ h + k) < 1e-9)) for h, k in zip(_HA, _HB)]
    edges = []
    for i, j in combinations(range(len(V)), 2):
        if sum(1 for s in on if i in s and j in s) >= 2:
            edges.append([np.round(V[i], 6).tolist(), np.round(V[j], 6).tolist()])
    faces = []
    for s in on:
        idx = sorted(s)
        P3 = V[idx]
        c = P3.mean(axis=0)
        n = np.cross(P3[1] - P3[0], P3[2] - P3[0])
        e1 = (P3[0] - c) / np.linalg.norm(P3[0] - c)
        e2 = np.cross(n / np.linalg.norm(n), e1)
        ang = np.arctan2((P3 - c) @ e2, (P3 - c) @ e1)
        faces.append([np.round(V[idx[k]], 6).tolist() for k in np.argsort(ang)])
    return edges, faces


def view(data):
    """The geometry of the atlas data (cell chart, before the picture is flipped) as seen in the dodecahedron."""
    out = {}
    # points
    samples = {}
    for k, pts in data["samples"].items():
        imgs = points([p[:3] for p in pts])
        # each image refers back to its sample (index into the cell view's list) for the hover text
        samples[k] = [u + [i] for i, us in enumerate(imgs) for u in us]
    out["samples"] = samples
    for key in ("uniform", "special"):
        imgs = points([u["q"] for u in data[key]])
        out[key] = [{**u, "q": q} for u, qs in zip(data[key], imgs) for q in qs]
    # polylines
    for key in ("rings", "main", "segments", "qaxes", "regular", "xlines"):
        items = data[key]
        pieces = polylines([r["pts"] for r in items])
        out[key] = [{**r, "pts": pc} for r, pcs in zip(items, pieces) for pc in pcs]
    # polygons
    pieces = polygons([r["pts"][:-1] if len(r["pts"]) > 3 and r["pts"][0] == r["pts"][-1] else r["pts"]
                       for r in data["tpatches"]])
    out["tpatches"] = [{**r, "pts": pc + [pc[0]]} for r, pcs in zip(data["tpatches"], pieces) for pc in pcs]
    xw = []
    for w in data["xwalls"]:
        tris = [t for pcs in polygons(w["tris"]) for pc in pcs
                for t in ([pc[0], pc[i], pc[i + 1]] for i in range(1, len(pc) - 1))]
        if tris:
            xw.append({**w, "tris": tris})
    out["xwalls"] = xw
    out["mirrors"] = [pc for pcs in polygons(data["mirrors"]) for pc in pcs]
    fd = []
    for f in data["fdomain"]:
        for pc in [pc for pcs in polygons([f["pts"]]) for pc in pcs]:
            fd.append({**f, "pts": pc, "axis": []})
    out["fdomain"] = fd
    centre = points([data["fcentre"]])[0]
    out["fcentre"] = centre[0] if centre else [0, 0, 0]
    edges, faces = dodecahedron_edges()
    out["edges"] = edges
    out["split"] = []
    out["dodecaFaces"] = faces
    return out


def clip_images(X, H, elements=E, hemi=None):
    """Pieces (in seed space) of every image g.arc, g in elements, inside the cone {x: x . h >= 0 for h in H}.

    X samples the arc (rows, consecutive points joined by straight chords, which are the arc's pieces of great
    circles in any gnomonic chart); hemi, if given, is a direction every kept point must face (x . hemi > 0)."""
    H = np.asarray(H, float)
    Y = np.einsum("gij,nj->gni", elements, np.asarray(X, float))      # (g, n, 4)
    f = Y @ H.T                                                         # (g, n, k)
    ok = ~np.any(np.all(f < -1e-12, axis=1), axis=1)
    if hemi is not None:
        ok &= np.all(Y @ hemi > 0, axis=1)
    pieces = []
    for g in np.flatnonzero(ok):
        cur = []
        for a, b in zip(Y[g, :-1], Y[g, 1:]):
            t0, t1 = 0.0, 1.0
            for h in H:
                fa, fd = a @ h, (b - a) @ h
                if abs(fd) < 1e-15:
                    if fa < -1e-12:
                        t0, t1 = 1.0, 0.0
                    continue
                t = -fa / fd
                if fd > 0:
                    t0 = max(t0, t)
                else:
                    t1 = min(t1, t)
            if t0 > t1 - 1e-12:
                if len(cur) > 1:
                    pieces.append(cur)
                cur = []
                continue
            p, q = a + t0 * (b - a), a + t1 * (b - a)
            if cur and np.linalg.norm(cur[-1] / np.linalg.norm(cur[-1]) - p / np.linalg.norm(p)) < 1e-9:
                cur.append(q)
            else:
                if len(cur) > 1:
                    pieces.append(cur)
                cur = [p, q]
            if t1 < 1 - 1e-12:
                pieces.append(cur)
                cur = []
        if len(cur) > 1:
            pieces.append(cur)
    return [np.array(pc) for pc in pieces]


# ---- one domain: a tenth of the dodecahedron -------------------------------------------------------------
# The ten elements fixing M34 act on the chart as the dihedral group D5: the vertical axis (the main ring) is
# 5-fold, and five horizontal half-turn axes are cross rings. A tenth is the upper half (z >= 0) cut to the
# 72-degree wedge between two half-turn axes, so its edges at M34 lie on the main ring and on two cross rings.

def _wedge():
    S = [g for g in E if np.allclose(g @ C, C, atol=1e-9)]
    axes = []
    for g in S:
        R = B @ g @ B.T
        if np.isclose(np.trace(R), -1, atol=1e-8):                     # a half-turn
            w, v = np.linalg.eig(R)
            a = np.real(v[:, np.argmin(np.abs(w - 1))])
            axes.append(np.arctan2(a[1], a[0]) % np.pi)
    axes = sorted(axes)
    t0 = axes[0]
    t1 = t0 + 2 * np.pi / 5
    n0 = np.array([-np.sin(t0), np.cos(t0), 0.0])                     # inward normals of the two vertical planes
    n1 = np.array([np.sin(t1), -np.cos(t1), 0.0])
    return np.array([[0, 0, 1.0], n0, n1]), (t0, t1)


WEDGE, WEDGE_ANGLES = _wedge()


def _in_wedge(u, tol=2e-4):           # coordinates are rounded to 4 decimals
    return bool(np.all(WEDGE @ np.asarray(u, float)[:3] >= -tol))


def _clip_line(pts):
    pieces, cur = [], []
    P = [np.asarray(p, float)[:3] for p in pts]
    for a, b in zip(P[:-1], P[1:]):
        t0, t1, d = 0.0, 1.0, b - a
        for h in WEDGE:
            fa, fd = h @ a, h @ d
            if abs(fd) < 1e-15:
                if fa < -1e-9:
                    t0, t1 = 1.0, 0.0
                continue
            t = -fa / fd
            if fd > 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
        if t0 > t1 - 1e-12:
            if len(cur) > 1:
                pieces.append(cur)
            cur = []
            continue
        p, q = a + t0 * d, a + t1 * d
        if cur and np.linalg.norm(np.array(cur[-1]) - p) < 1e-9:
            cur.append(q.tolist())
        else:
            if len(cur) > 1:
                pieces.append(cur)
            cur = [p.tolist(), q.tolist()]
        if t1 < 1 - 1e-12:
            pieces.append(cur)
            cur = []
    if len(cur) > 1:
        pieces.append(cur)
    return [[np.round(x, 4).tolist() for x in pc] for pc in pieces]


def _clip_poly(pts):
    poly = [np.asarray(p, float)[:3] for p in pts]
    for h in WEDGE:
        new = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            fa, fb = h @ a, h @ b
            if fa >= -1e-12:
                new.append(a)
            if (fa >= -1e-12) != (fb >= -1e-12):
                new.append(a + (fa / (fa - fb)) * (b - a))
        poly = new
        if len(poly) < 3:
            return None
    return [np.round(x, 4).tolist() for x in poly]


def piece_view(dd):
    """The dodecahedron view cut down to one domain (a tenth)."""
    out = {}
    out["samples"] = {k: [p for p in v if _in_wedge(p)] for k, v in dd["samples"].items()}
    for key in ("uniform", "special"):
        out[key] = [u for u in dd[key] if _in_wedge(u["q"])]
    for key in ("rings", "main", "segments", "qaxes", "regular", "xlines"):
        out[key] = [{**r, "pts": pc} for r in dd[key] for pc in _clip_line(r["pts"])]
    out["tpatches"] = []
    for r in dd["tpatches"]:
        pc = _clip_poly(r["pts"][:-1])
        if pc:
            out["tpatches"].append({**r, "pts": pc + [pc[0]]})
    out["xwalls"] = []
    for w in dd["xwalls"]:
        tris = []
        for t in w["tris"]:
            pc = _clip_poly(t)
            if pc:
                tris += [[pc[0], pc[i], pc[i + 1]] for i in range(1, len(pc) - 1)]
        if tris:
            out["xwalls"].append({**w, "tris": tris})
    out["mirrors"] = [pc for pc in (_clip_poly(m) for m in dd["mirrors"]) if pc]
    out["fdomain"] = [{**f, "pts": pc} for f in dd["fdomain"] for pc in [_clip_poly(f["pts"])] if pc]
    out["fcentre"] = dd["fcentre"]
    # the piece itself: the dodecahedron's faces cut by the wedge, and the wedge's own three faces
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    H = np.vstack([np.hstack([-_HA, -_HB[:, None]]), np.hstack([-WEDGE, np.zeros((3, 1))])])
    inner = np.array([np.cos(np.mean(WEDGE_ANGLES)), np.sin(np.mean(WEDGE_ANGLES)), 1.0]) * 0.05
    V = HalfspaceIntersection(H, inner).intersections
    hull = ConvexHull(V)
    edges, faces = set(), {}
    for simplex, eq in zip(hull.simplices, hull.equations):
        key = tuple(np.round(eq, 6))
        faces.setdefault(key, set()).update(simplex.tolist())
    face_polys = []
    for key, idx in faces.items():
        idx = sorted(idx)
        P3 = V[idx]
        c = P3.mean(axis=0)
        n = np.array(key[:3])
        e1 = (P3[0] - c) / np.linalg.norm(P3[0] - c)
        e2 = np.cross(n, e1)
        order = [idx[k] for k in np.argsort(np.arctan2((P3 - c) @ e2, (P3 - c) @ e1))]
        face_polys.append([np.round(V[k], 4).tolist() for k in order])
        for i in range(len(order)):
            edges.add(tuple(sorted((order[i], order[(i + 1) % len(order)]))))
    out["edges"] = [[np.round(V[i], 4).tolist(), np.round(V[j], 4).tolist()] for i, j in edges]
    out["dodecaFaces"] = face_polys
    out["split"] = []
    lo, hi = V.min(axis=0), V.max(axis=0)
    out["bounds"] = [lo.tolist(), hi.tolist()]
    return out
