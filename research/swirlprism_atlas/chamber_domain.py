"""A fundamental domain made of whole H4 chambers, inside the dodecahedron around the 600-cell vertex V1.

That dodecahedron's faces are H4 mirrors, and the mirrors through V1 cut it into 120 chambers: each joins V1 to
one face centre, one edge midpoint and one vertex of the dodecahedron (its barycentric subdivision). The ten
elements of the swirlprism group fixing V1 move these chambers freely, in 12 orbits of 10, so one chamber from
each orbit is a fundamental domain (14400 / 1200 = 12 chambers). The choice is one convex solid (the union of its
chambers equals their convex hull; ties broken by the most faces shared between chambers) and then, as for the
other domains, the one with the most chamber edges on significant rings (cross rings, the main ring, the purple 2400 axes
and the green order-3 girdle), found exactly: an edge is on one when a group element fixes both its ends. Call with the dodecahedron module centred on V1 (dodeca_view.use_centre).
"""
from __future__ import annotations

from itertools import combinations

import dodeca_view as V
import numpy as np


def chambers():
    """The 120 chambers (4 chart points each) and their faces."""
    edges, faces = V.dodecahedron_edges()
    out = []
    for f in faces:
        F = np.array(f)
        fc = F.mean(axis=0)
        for i in range(len(F)):
            a, b = F[i], F[(i + 1) % len(F)]
            em = (a + b) / 2
            for v in (a, b):
                out.append(np.array([np.zeros(3), fc, em, v]))
    return out


def _key(p):
    return tuple(np.round(p, 4))


def orbits(ch):
    """Orbit index of each chamber under the ten rotations fixing the centre."""
    S = [g for g in V.E if np.allclose(g @ V.C, V.C, atol=1e-9)]
    R = [V.B @ g @ V.B.T for g in S]
    cent = np.array([c.mean(axis=0) for c in ch])                     # chambers matched by their centroids
    orb = [-1] * len(ch)
    n = 0
    for k, c in enumerate(ch):
        if orb[k] >= 0:
            continue
        for r in R:
            j = int(np.argmin(np.linalg.norm(cent - r @ cent[k], axis=1)))
            orb[j] = n
        n += 1
    return orb, n


_FIXERS = None


def _fixers():
    """Elements whose fixed great circles are the significant rings: the swirlprism group (cross rings and the
    main ring), its coset under the extra half-turn (purple axes) and the extra order-3 elements of the
    3600-element group (green girdle)."""
    global _FIXERS
    if _FIXERS is None:
        from normalizer import halfturn
        from supergroups import supergroup
        G = [g for g in V.E if not np.allclose(g, np.eye(4))]
        coset = list(V.E @ halfturn())
        known = {tuple(np.round(g, 7).ravel()) for g in V.E}
        g3 = [g for g in supergroup(3) if tuple(np.round(g, 7).ravel()) not in known
              and np.allclose(np.linalg.matrix_power(g, 3), np.eye(4), atol=1e-7)]
        _FIXERS = np.stack(G + coset + g3)
    return _FIXERS


def _seed_of(u):
    x = V.C + V.B.T @ np.asarray(u, float)
    return x / np.linalg.norm(x)


def _on_ring(a, b, segs=None, tol=1e-5):        # chart points carry ~1e-6 rounding
    """Does the chamber edge a-b (chart points) lie on a significant ring? Exactly when some element of the
    fixers fixes both ends (then it fixes the whole great circle through them)."""
    F = _fixers()
    x, y = _seed_of(a), _seed_of(b)
    return bool(np.any((np.linalg.norm(F @ x - x, axis=1) < tol) & (np.linalg.norm(F @ y - y, axis=1) < tol)))


def _ring_segments(view_data):
    return None


def choose(view_data, beam=300):
    ch = chambers()
    orb, n = orbits(ch)
    segs = _ring_segments(view_data)
    edge_keys = [[frozenset((_key(c[i]), _key(c[j]))) for i, j in combinations(range(4), 2)] for c in ch]
    ring_edge = {}
    for c, keys in zip(ch, edge_keys):
        for (i, j), e in zip(combinations(range(4), 2), keys):
            if e not in ring_edge:
                ring_edge[e] = _on_ring(c[i], c[j], segs)
    tri = [[frozenset(_key(c[i]) for i in t) for t in combinations(range(4), 3)] for c in ch]
    owner = {}
    for k, ts in enumerate(tri):
        for t in ts:
            owner.setdefault(t, []).append(k)
    nbrs = [set(j for t in ts for j in owner[t] if j != k) for k, ts in enumerate(tri)]

    vol = [abs(np.linalg.det(c[1:] - c[0])) / 6 for c in ch]

    def convex(sel):
        """Is the union of the chambers a convex polytope? (its hull has no more volume than they do)"""
        from scipy.spatial import ConvexHull
        pts = np.array([p for k in sel for p in ch[k]])
        return ConvexHull(pts).volume <= sum(vol[k] for k in sel) * (1 + 1e-6)

    def score(sel):
        """Cohesion first (faces shared between chambers), then chamber edges on significant rings."""
        es = {e for k in sel for e in edge_keys[k]}
        faces = [t for k in sel for t in tri[k]]
        inner = sum(1 for t in set(faces) if faces.count(t) == 2)
        return (inner, sum(1 for e in es if ring_edge[e]))

    states = [frozenset([k]) for k in range(len(ch)) if orb[k] == 0]
    for _ in range(n - 1):
        nxt = {}
        for sel in states:
            used = {orb[k] for k in sel}
            for k in set().union(*(nbrs[j] for j in sel)):
                if orb[k] not in used:
                    s2 = sel | {k}
                    nxt.setdefault(s2, score(s2))
        states = [s for s, _ in sorted(nxt.items(), key=lambda kv: kv[1], reverse=True)[:beam]]
    # the final choice: convex if any candidate is, then the most ring edges, then the most shared faces
    choose.candidates = [(bool(convex(s)), *score(s)) for s in states]
    # among convex candidates every one is a single solid piece, so the ring rule decides
    best = max(states, key=lambda s: (convex(s), score(s)[1], score(s)[0]))
    inner, rings_on = score(best)
    choose.report = {"convex": bool(convex(best)), "sharedFaces": inner, "edgesOnRings": rings_on,
                     "convexCandidates": sum(1 for s in states if convex(s)), "candidates": len(states)}
    return [ch[k] for k in sorted(best)], rings_on, ring_edge, edge_keys


def _halfspaces(c):
    """Inward planes (n, k): n . u + k >= 0 for the tetrahedron c."""
    H = []
    for t in combinations(range(4), 3):
        a, b, d = c[list(t)]
        n = np.cross(b - a, d - a)
        o = [i for i in range(4) if i not in t][0]
        k = -n @ a
        if n @ c[o] + k < 0:
            n, k = -n, -k
        s = np.linalg.norm(n)
        H.append((n / s, k / s))
    return H


def _clip_line(pts, H):
    pieces, cur = [], []
    P = [np.asarray(p, float)[:3] for p in pts]
    for a, b in zip(P[:-1], P[1:]):
        t0, t1, d = 0.0, 1.0, b - a
        for n, k in H:
            fa, fd = n @ a + k, n @ d
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
    return [[np.round(x, 4).tolist() for x in pc] for pc in pieces]


def _clip_poly(pts, H):
    poly = [np.asarray(p, float)[:3] for p in pts]
    for n, k in H:
        new = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            fa, fb = n @ a + k, n @ b + k
            if fa >= -1e-12:
                new.append(a)
            if (fa >= -1e-12) != (fb >= -1e-12):
                new.append(a + (fa / (fa - fb)) * (b - a))
        poly = new
        if len(poly) < 3:
            return None
    return [np.round(x, 4).tolist() for x in poly]


def view(vd):
    """The V1 dodecahedron view (vd) cut down to the chosen 12 chambers."""
    sel, rings_on, ring_edge, _ = choose(vd)
    Hs = [_halfspaces(c) for c in sel]

    def inside(u, tol=2e-4):
        u = np.asarray(u, float)[:3]
        return any(all(n @ u + k >= -tol for n, k in H) for H in Hs)
    out = {}
    out["samples"] = {k: [p for p in v if inside(p)] for k, v in vd["samples"].items()}
    for key in ("uniform", "special"):
        out[key] = [u for u in vd[key] if inside(u["q"])]
    for key in ("rings", "main", "segments", "qaxes", "regular", "xlines"):
        out[key] = [{**r, "pts": pc} for r in vd[key] for H in Hs for pc in _clip_line(r["pts"], H)]
    out["tpatches"] = []
    for r in vd["tpatches"]:
        for H in Hs:
            pc = _clip_poly(r["pts"][:-1], H)
            if pc:
                out["tpatches"].append({**r, "pts": pc + [pc[0]]})
    out["xwalls"] = []
    for w in vd["xwalls"]:
        tris = []
        for t in w["tris"]:
            for H in Hs:
                pc = _clip_poly(t, H)
                if pc:
                    tris += [[pc[0], pc[i], pc[i + 1]] for i in range(1, len(pc) - 1)]
        if tris:
            out["xwalls"].append({**w, "tris": tris})
    out["mirrors"] = [pc for m in vd["mirrors"] for H in Hs for pc in [_clip_poly(m, H)] if pc]
    out["fdomain"] = [{**f, "pts": pc} for f in vd["fdomain"] for H in Hs for pc in [_clip_poly(f["pts"], H)] if pc]
    out["fcentre"] = vd["fcentre"]
    # the domain's own outline: the chamber faces on its boundary, and all chamber edges (faint)
    tris = [frozenset(_key(c[i]) for i in t) for c in sel for t in combinations(range(4), 3)]
    counts = {t: tris.count(t) for t in set(tris)}
    faces, edges = [], set()
    for c in sel:
        for t in combinations(range(4), 3):
            if counts[frozenset(_key(c[i]) for i in t)] == 1:
                faces.append([np.round(c[i], 4).tolist() for i in t])
        for i, j in combinations(range(4), 2):
            edges.add(tuple(sorted((_key(c[i]), _key(c[j])))))
    out["edges"] = [[list(a), list(b)] for a, b in edges]
    out["dodecaFaces"] = faces
    out["split"] = []
    allp = np.array([p for c in sel for p in c])
    out["bounds"] = [allp.min(axis=0).tolist(), allp.max(axis=0).tolist()]
    out["chamberInfo"] = {"chambers": 12, **choose.report}
    return out
