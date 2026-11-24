"""A closed exact map of a curved A4 group (kidecaic, iodecaic), built the way the swirlprism Symmetry Domain was:
exact curves, exact points, the faces they cut each surface into, each face classified from exact points inside
it, then checks that the whole map closes. Nothing is meshed here and nothing is patched: what fails a check is
reported.

  1. surfaces: those of a4_exact (the surfaces carrying the group's walls, their swap images, the four mirrors).
  2. curves: for every pair of surfaces, their whole common curve in the chamber. Seeds are found systematically:
     every edge of a fine grid on the first surface where the second changes sign, put on both by Newton's method;
     each seed not already on a traced curve is traced both ways (a4_exact.trace). A stretch of a curve already
     traced from another pair (a curve on three or more surfaces) is kept once.
  3. points: where a third surface crosses a curve, solved on all three; closed curves no surface crosses are kept
     (cut at two points of their own).
  4. segments: the curves cut at their points, each labelled with every surface it lies on; then a4_faces'
     clean_graph (points merged, the swap's axis rebuilt as one chain) and loose-end closing.
  5. faces: on each surface the segments' graph walked with the face on its left (a4_faces.faces_on).
  6. classification: each face probed at points just inside it, a step in from every segment of its outline; each
     probe classified exactly (hull tolerance 1e-7), and a step off the surface both ways for the regions beside it.
  7. checks: (a) no loose ends; (b) every face on one side of every other surface (else a curve is missing inside
     it); (c) every probe of a face the same shape (else a surface is missing); (d) every sign pattern a fine grid
     finds on a surface is some face's (else a face is missing); (e) the swap image of every face is a face of the
     same shape.

    python a4_closed.py iodecaic      (writes a4_closed_<group>.pkl and a4_closed_<group>.json, the report)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_exact as E  # noqa: E402
import a4_faces as A  # noqa: E402
import a4_pieces as P  # noqa: E402
import a4_snub as SN  # noqa: E402

BZ = P.BZ
REL = lambda S_, x: abs(S_.f(x)) / (np.linalg.norm(S_.g(x)) + 1e-300)


class Near:
    """Points along curves, hashed in cells, to ask whether a point lies on a curve already traced."""

    def __init__(self, tol=3e-4, step=2e-4):
        self.tol, self.step, self.cells = tol, step, defaultdict(list)

    def add(self, Q):
        for x in A.dense(Q, self.step):
            self.cells[tuple(np.floor(x[:3] / self.tol).astype(int))].append(x)

    def near(self, x):
        c = np.floor(x[:3] / self.tol).astype(int)
        for d in np.ndindex(3, 3, 3):
            for y in self.cells.get(tuple(c + np.array(d) - 1), ()):
                if np.linalg.norm(y - x) < self.tol:
                    return True
        return False


def grid_of(S_, k):
    V, T = E.surface_mesh(S_, k)
    V = np.asarray(V, float)
    edges = sorted({(min(a, b), max(a, b)) for t in T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
    return V, np.asarray(T, int), np.asarray(edges, int)


def curves_of(surfs, grids, log):
    """Every pair's common curve, from every grid crossing (2)."""
    mirrors = surfs[:4]
    seen = Near()
    curves = []
    t0 = time.time()
    for a in range(len(surfs)):
        V, _, Ed = grids[a]
        if not len(Ed):
            continue
        for b in range(len(surfs)):
            if b == a or (b < a and len(grids[b][2])):      # (each pair once, from the surface with a grid)
                continue
            fb = np.array([surfs[b].f(x) for x in V])
            # (a surface exactly zero at a grid vertex, as a chamber edge or a plane through lattice points is,
            #  counts as below zero there, so the edges from it still show the crossing)
            pos = fb > 0
            cr = Ed[pos[Ed[:, 0]] != pos[Ed[:, 1]]]
            if not len(cr):
                continue
            fa, fb_ = fb[cr[:, 0]], fb[cr[:, 1]]
            X0 = V[cr[:, 0]] + (fa / (fa - fb_ + 1e-300))[:, None] * (V[cr[:, 1]] - V[cr[:, 0]])
            for x in X0:
                y, ok = E.newton(x, [surfs[a], surfs[b]])
                if not ok or not E.inside(y, 1e-12) or seen.near(y):
                    continue
                if E.crossing_angle(y, surfs[a], surfs[b]) < E.SING:
                    continue
                pts, closed = E.trace(y, surfs[a], surfs[b], mirrors)
                if len(pts) < 2:
                    continue
                Q = np.array(pts)
                # (the stretches not already traced: a curve on three or more surfaces is found from several pairs)
                old = np.array([seen.near(q) for q in Q])
                if old.all():
                    continue
                if not old.any():
                    runs = [(0, len(Q) - 1)]
                else:
                    idx = np.flatnonzero(~old)
                    cut = np.flatnonzero(np.diff(idx) > 1)
                    runs = [(r[0], r[-1]) for r in np.split(idx, cut + 1)]
                    closed = False
                for i0, i1 in runs:
                    sub = Q[max(i0 - 1, 0): i1 + 2]
                    if len(sub) >= 2:
                        curves.append({"pair": (a, b), "pts": sub, "closed": closed and len(runs) == 1})
                        seen.add(sub)
        log(f"    surface {a}: {len(curves)} curves so far, {time.time() - t0:.0f}s")
    return curves


def segments_of(surfs, curves, log):
    """The points where third surfaces cross the curves, and the segments between them (3, 4)."""
    points = []
    tree = {"t": None, "n": 0}

    def pid(x):
        x = np.asarray(x, float)
        for i in range(len(points) - 1, -1, -1):
            if np.linalg.norm(points[i] - x) < 1e-9:
                return i
        points.append(x)
        return len(points) - 1

    segs = []
    for c in curves:
        (a, b), Q = c["pair"], np.asarray(c["pts"], float)
        cuts = []
        on = {a, b}
        for m, C in enumerate(surfs):
            if m in (a, b):
                continue
            fv = np.array([C.f(p) for p in Q])
            sc = np.array([np.linalg.norm(C.g(p)) + 1e-300 for p in Q])
            r = np.abs(fv / sc)
            if np.mean(r < 1e-9) > 0.9 and r.max() < 1e-4:
                on.add(m)
                continue
            for i in range(len(Q) - 1):
                if fv[i] * fv[i + 1] < 0 and max(r[i], r[i + 1]) > 1e-7:
                    t = fv[i] / (fv[i] - fv[i + 1])
                    z, ok = E.newton(Q[i] + t * (Q[i + 1] - Q[i]), [surfs[a], surfs[b], C])
                    if ok and E.inside(z, 1e-10):
                        cuts.append((i + t, pid(z)))
            for i in (0, len(Q) - 1):
                if r[i] < 1e-10:
                    cuts.append((float(i), pid(Q[i])))
        if not c["closed"]:
            cuts += [(0.0, pid(Q[0])), (len(Q) - 1.0, pid(Q[-1]))]
        cuts = sorted(set(cuts))
        if c["closed"]:
            if len(cuts) < 2:                    # (a closed curve no surface crosses: two points of its own)
                n = len(Q) - 1
                cuts = sorted(set(cuts) | {(0.0, pid(Q[0])), (float(n // 2), pid(Q[n // 2]))})
            cuts = cuts + [(cuts[0][0] + len(Q) - 1, cuts[0][1])]
            Q = np.vstack([Q[:-1], Q[:-1], Q[:1]])           # (the loop twice, to run past its start)
        for (t0, p0), (t1, p1) in zip(cuts, cuts[1:]):
            if p0 == p1 and t1 - t0 < 2:
                continue
            i0, i1 = int(np.floor(t0)) + 1, int(np.ceil(t1))
            poly = np.vstack([points[p0]] + [Q[i] for i in range(i0, min(i1, len(Q)))] + [points[p1]])
            if np.linalg.norm(np.diff(poly, axis=0), axis=1).sum() < 1e-9:
                continue
            segs.append({"pair": (a, b), "ends": (p0, p1), "pts": poly, "on": set(on), "key": None, "sig": None})
    log(f"  {len(points)} points, {len(segs)} segments")
    return points, segs


def probes_of(S, k, loop, cyc, segs, eps=1e-4):
    """Points just inside a face: a step to the left of its outline, a few along every segment of it. (Much nearer a
    chamber edge than 1e-4 the hull test sees vertices all but coincide and misnames the shape.)"""
    out = []
    for i, fw in cyc:
        Q = np.asarray(segs[i]["pts"], float)
        Q = Q if fw else Q[::-1]
        for f in (0.25, 0.5, 0.75):
            m = min(max(int(f * (len(Q) - 1)), 0), len(Q) - 2)
            x = Q[m] + (Q[m + 1] - Q[m]) * (f * (len(Q) - 1) - m) if len(Q) > 2 else Q[0] + f * (Q[1] - Q[0])
            d = Q[m + 1] - Q[m]
            n = P.normal(S, x) @ BZ.T
            side = np.cross(n, d @ BZ.T) @ BZ
            nn = np.linalg.norm(side)
            if nn < 1e-14:
                continue
            y = P.lift_normal(S, x + eps * side / nn) if k >= 4 else x + eps * side / nn
            if y is not None and E.inside(y, 1e-12):
                out.append(y)
    return out


def build(group, log=print):
    sub = SN.SUB[group]
    X0 = pickle.load(open(os.path.join(HERE, f"a4_exact_{group}.pkl"), "rb"))
    names = X0["surfaces"]
    surfs = [P.surf_of(n) for n in names]
    keyname = X0.get("keyname", {})
    log(f"{group}: {len(surfs)} surfaces, grid {E.GRID}")
    grids = {k: grid_of(S_, k) for k, S_ in enumerate(surfs)}
    curves = curves_of(surfs, grids, log)
    log(f"  {len(curves)} curves ({sum(c['closed'] for c in curves)} closed)")
    points, segs = segments_of(surfs, curves, log)
    X = {"surfaces": names, "points": points, "segments": segs, "faces": []}
    A.clean_graph(X, surfs, log)
    A.complete_graph(X, surfs, log)               # (no pieces: only its loose-end closing runs)
    segs, Pts = X["segments"], X["points"]
    report = {"group": group, "surfaces": len(surfs), "points": len(Pts), "segments": len(segs)}
    # (a) loose ends
    loose = []
    for k in range(len(surfs)):
        deg = Counter()
        for s_ in segs:
            if k in s_["on"]:
                deg[s_["ends"][0]] += 1
                deg[s_["ends"][1]] += 1
        loose += [(k, p) for p, d in deg.items() if d == 1]
    report["loose ends"] = len(loose)
    # 5, 6. faces, probed and classified
    faces = []
    for k, S in enumerate(surfs):
        for loops, hc in A.faces_on(k, S, segs, Pts):
            pr = probes_of(S, k, loops[0], hc[0], segs)
            if pr:
                faces.append({"surface": k, "loop": loops[0], "cycle": hc[0], "probes": pr})
    allp = [p for F in faces for p in F["probes"]]
    labs = SN.classify_many(allp, sub, tol=1e-7)
    i = 0
    for F in faces:
        F["keys"] = [labs[i + j][1] for j in range(len(F["probes"]))]
        i += len(F["probes"])
        F["key"] = Counter(F["keys"]).most_common(1)[0][0]
    log(f"  {len(faces)} faces")
    # (b) every face on one side of every other surface; (c) every probe the same shape
    cut_through, mixed = [], []
    for fi, F in enumerate(faces):
        k = F["surface"]
        pr = np.array(F["probes"])
        bad = []
        for m, C in enumerate(surfs):
            if m == k:
                continue
            r = np.array([C.f(x) / (np.linalg.norm(C.g(x)) + 1e-300) for x in pr])
            s = np.sign(r[np.abs(r) > 1e-9])
            if len(s) and (s > 0).any() and (s < 0).any():
                bad.append(m)
        F["cut by"] = bad
        if bad:
            cut_through.append(fi)
        if len(set(F["keys"])) > 1:
            mixed.append(fi)
    # (d) every sign pattern of a fine grid on a surface is some face's
    def pattern(k, pr):
        v = {}
        for m, C in enumerate(surfs):
            if m == k:
                continue
            r = np.array([C.f(x) / (np.linalg.norm(C.g(x)) + 1e-300) for x in pr])
            s = np.sign(r[np.abs(r) > 1e-9])
            if len(s) and (s == s[0]).all():
                v[m] = int(s[0])
        return v
    for F in faces:
        F["pattern"] = pattern(F["surface"], np.array(F["probes"]))
    unmatched = []
    from scipy.spatial import cKDTree
    for k, S in enumerate(surfs):
        V = grids[k][0]
        mine = [F["pattern"] for F in faces if F["surface"] == k]
        own = [A.dense(s_["pts"], 2e-3) for s_ in segs if k in s_["on"]]
        on_curve = cKDTree(np.vstack(own)) if own else None
        for x in V[:: max(1, len(V) // 400)]:
            x = P.lift_normal(S, x) if k >= 4 else x
            if x is None or not E.inside(x, 1e-12):
                continue
            if on_curve is not None and on_curve.query(x)[0] < 3e-3:     # (on a curve of the map, not in a face)
                continue
            v = {}
            for m, C in enumerate(surfs):
                if m == k:
                    continue
                r = C.f(x) / (np.linalg.norm(C.g(x)) + 1e-300)
                if abs(r) > 2e-3:
                    v[m] = int(np.sign(r))
            if not any(all(pat.get(m, s_) == s_ for m, s_ in v.items()) for pat in mine):
                unmatched.append((k, x))
    report["grid points in no face (face missing)"] = len(unmatched)
    report["grid points in no face, by surface"] = dict(Counter(k for k, _ in unmatched))
    # (e) the swap image of every face: on the swap surface, the same shape
    swap_of = {}
    for k, S in enumerate(surfs):
        for m, T in enumerate(surfs):
            if E.same_surface(T.expr, E.rev(S.expr)):
                swap_of[k] = m
    sw_pr = [np.array(F["probes"][0])[::-1] for F in faces]
    sw_labs = SN.classify_many(sw_pr, sub, tol=1e-7)
    bad_swap = []
    for fi, (F, (_, key2)) in enumerate(zip(faces, sw_labs)):
        m = swap_of.get(F["surface"])
        pat = pattern(m, [sw_pr[fi]]) if m is not None else None
        there = m is not None and any(G["surface"] == m and all(G["pattern"].get(q, s_) == s_ for q, s_ in pat.items())
                                      for G in faces)
        if key2 != F["key"] or not there:
            bad_swap.append(fi)
    report["faces whose swap image is not a face of the same shape"] = len(bad_swap)
    # (sides: the region a step off the face each way, at its first probe)
    side_pts = []
    for F in faces:
        x = np.array(F["probes"][0])
        n = P.normal(surfs[F["surface"]], x)
        side_pts += [x + 1e-4 * n, x - 1e-4 * n]
    side_labs = SN.classify_many(side_pts, sub)
    for i, F in enumerate(faces):
        F["sides"] = [side_labs[2 * i][1] if E.inside(side_pts[2 * i], 0) else None,
                      side_labs[2 * i + 1][1] if E.inside(side_pts[2 * i + 1], 0) else None]
    report["faces"] = len(faces)
    report["faces cut through by a surface (curve missing)"] = len(cut_through)
    report["faces whose probes disagree on the shape"] = len(mixed)
    report["faces by shape"] = dict(Counter(keyname.get(F["key"], "other") for F in faces))
    pickle.dump({"surfaces": names, "points": Pts, "segments": segs, "faces": faces, "keyname": keyname,
                 "report": report}, open(os.path.join(HERE, f"a4_closed_{group}.pkl"), "wb"))
    json.dump(report, open(os.path.join(HERE, f"a4_closed_{group}.json"), "w"), indent=1, default=str)
    log(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    for g in sys.argv[1:]:
        build(g)
