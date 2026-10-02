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
