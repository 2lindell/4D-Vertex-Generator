"""The wall meshes filled out to their corners (pieces.json, in place).

Where two of a piece's boundary stretches meet, the mesh triangle there is cut by both, and the boundary cuts across the
corner instead of reaching it. Here each such corner is found exactly (the point on the wall's surface and both
stretches' surfaces) and the triangle out to it added, for the patch and for each of its parts. A stretch's surface at a
boundary point is a surface through that point and its neighbour along the same stretch: a library surface or a face of
the half-cell for an edge, an image of the cutting wall's surface under the group for a cut.

    python fill_corners.py [X14,X41,...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import core as osd
from core import FOLD, extended
from pieces import HALF, Image, fval, grad, in_half, nrm, project

K = lambda x: tuple(np.round(np.asarray(x, float), 9))


def on(s, x, tol=1e-8):
    return abs(fval(s, x)) / (np.linalg.norm(grad(s, x)) + 1e-300) < tol


def fill(S, tris, segs, cand, h, GG, invs, LIB, ex):
    """Corner triangles added to tris, boundary segments split at the corners; the number added."""
    nb = {}
    for (a, na), (b, nb_) in segs:
        nb.setdefault((K(a), na), []).append(np.asarray(b, float))
        nb.setdefault((K(b), nb_), []).append(np.asarray(a, float))

    def surf(x, n, other):
        nxt = [y for y in nb.get((K(x), n), []) if np.linalg.norm(y - other) > 1e-12]
        pts = [x] + nxt[:1]
        if n.startswith("cut by"):
            Y = n[7:]
            for nm in ex["walls"].get(Y, {}).get("surface", []):
                if nm not in LIB:
                    continue
                for g, gi in zip(GG, invs):
                    s_ = Image(LIB[nm], g)
                    if all(on(s_, p) for p in pts):
                        return s_
            return None
        for s_ in cand:
            if s_ is not S and all(on(s_, p) for p in pts):
                return s_
        return None
    out, added = [], 0
    for (x1, n1), (x2, n2) in segs:
        x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
        if n1 == n2 or "?" in (n1, n2):
            out.append([(x1, n1), (x2, n2)])
            continue
        s1, s2 = surf(x1, n1, x2), surf(x2, n2, x1)
        c = None if s1 is None or s2 is None else nrm(project([S, s1, s2], (x1 + x2) / 2))
        if (c is None or max(abs(fval(s_, c)) for s_ in (S, s1, s2)) > 1e-10 or not in_half(c, 1e-9)
                or max(np.linalg.norm(c - x1), np.linalg.norm(c - x2)) > 4 * h):
            out.append([(x1, n1), (x2, n2)])
            continue
        tris.append([x1.tolist(), x2.tolist(), c.tolist()])
        out += [[(x1, n1), (c, n1)], [(c, n2), (x2, n2)]]
        added += 1
    segs[:] = [[[np.asarray(a, float).round(10).tolist(), na], [np.asarray(b, float).round(10).tolist(), nb_]]
               for (a, na), (b, nb_) in out]
    return added


def fill_ends(tris, segs, ends, h):
    """Where two of the wall's lines meet at a point the mesh stops short of (at a vertex of the cell a surface can
    be singular, and the mesh does not reach it): the point is known exactly (the lines' end), so the rim between the two
    lines there (the segments in the angle between them) is joined to it by triangles. The number of points reached."""
    if not segs or not tris:
        return 0
    T = np.array(tris, float).reshape(-1, 4)
    done = 0
    for e, rays in ends:
        gap = np.linalg.norm(T - e, axis=1).min()
        if gap < 1e-7 or gap > 0.08:
            continue
        S_ = [(np.asarray(a, float), na, np.asarray(b, float), nb) for (a, na), (b, nb) in segs]
        R = gap + 3 * h
        # (where only one line ends there, the other side of the angle is the rest of the rim: a face, a seam, another
        # wall's edge; its direction from the point is that of its farthest rim point within reach)
        rays = list(rays)
        far = {}
        for a, na, b, nb in S_:
            for x, n in ((a, na), (b, nb)):
                dd = np.linalg.norm(x - e)
                if dd < R and dd > far.get(n, (0, None))[0]:
                    far[n] = (dd, x)
        for n, (dd, x) in far.items():
            d = (x - e) / dd
            if all(abs(d @ r) < 0.9999 for r in rays):
                rays.append(d)
        best = None
        for i in range(len(rays)):
            for j in range(i + 1, len(rays)):
                dA, dB = rays[i], rays[j]
                if abs(dA @ dB) > 0.9999:
                    continue
                M = np.vstack([dA, dB]).T
                cap = []
                for k, (a, na, b, nb) in enumerate(S_):
                    m = (a + b) / 2
                    if np.linalg.norm(m - e) > R:
                        continue
                    co, res = np.linalg.lstsq(M, m - e, rcond=None)[:2]
                    if co.min() > 1e-9:
                        cap.append(k)
                if cap and (best is None or len(cap) > len(best)):
                    best = cap
        if not best:
            # (the two stretches meeting at a very small angle there (tangent): the rim segment across from one to the
            # other, nearest the point)
            mixed = [k for k, (a, na, b, nb) in enumerate(S_) if na != nb and "?" not in (na, nb)
                     and np.linalg.norm((a + b) / 2 - e) < R]
            if mixed:
                best = [min(mixed, key=lambda k: np.linalg.norm((S_[k][0] + S_[k][2]) / 2 - e))]
        if not best:
            # (no rim across the angle, the rim's two stretches stopping short apart: their loose ends joined to it)
            deg = {}
            for a, na, b, nb in S_:
                for x, n in ((a, na), (b, nb)):
                    deg.setdefault(K(x), [0, x, n])[0] += 1
            loose = [(x, n) for c, x, n in deg.values() if c == 1 and np.linalg.norm(x - e) < R]
            if len(loose) == 2 and loose[0][1] != loose[1][1]:
                (x1, n1), (x2, n2) = loose
                tris.append([x1.tolist(), x2.tolist(), e.tolist()])
                segs += [[[x1.round(10).tolist(), n1], [e.round(10).tolist(), n1]],
                         [[x2.round(10).tolist(), n2], [e.round(10).tolist(), n2]]]
                T = np.array(tris, float).reshape(-1, 4)
                done += 1
            continue
        cnt = {}
        for k in best:
            a, na, b, nb = S_[k]
            for x, n in ((a, na), (b, nb)):
                cnt.setdefault(K(x), [0, x, n])[0] += 1
            tris.append([a.tolist(), b.tolist(), e.tolist()])
        free = [(x, n) for c, x, n in cnt.values() if c == 1]
        keep = [s for k, s in enumerate(segs) if k not in set(best)]
        keep += [[[x.round(10).tolist(), n], [e.round(10).tolist(), n]] for x, n in free]
        segs[:] = keep
        T = np.array(tris, float).reshape(-1, 4)
        done += 1
    return done


def line_ends(w, S, work, ex, GG, LIB):
    """The ends of the wall's lines on the surface S, each with the directions the lines leave it."""
    from pieces import line_pieces_on
    pts = []
    for L in work[w].get("edges", []):
        if L not in ex["lines"]:
            continue
        for Q, _ in line_pieces_on(ex["lines"][L], S, GG, LIB):
            if np.linalg.norm(Q[-1] - Q[0]) < 1e-6:
                continue
            for e, Qs in ((Q[0], Q), (Q[-1], Q[::-1])):
                d = next((q - e for q in Qs[1:] if np.linalg.norm(q - e) > 1e-9), None)
                if d is None:
                    continue
                pts.append((e, d / np.linalg.norm(d)))
    out = []
    for e, d in pts:
        for o in out:
            if np.linalg.norm(o[0] - e) < 1e-7:
                if all(abs(d @ r) < 0.9999 for r in o[1]):
                    o[1].append(d)
                break
        else:
            out.append((e, [d]))
    return out


def main(only=None):
    P = json.load(open("pieces.json"))
    ex = json.load(open("exact_defs.json"))
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    cand = list(LIB.values()) + [osd.Plane(n, name) for name, n in HALF]
    E_ = extended()
    GG = np.concatenate([E_, E_ @ FOLD])
    invs = np.linalg.inv(GG)
    work = json.load(open("working_data.json"))["walls"]
    for w, rec in P.items():
        if only and w not in only:
            continue
        tot = tend = 0
        for pt in rec["patches"]:
            S = LIB[pt["surface"]]
            h = pt.get("h", 0.008)
            # (the lines' own ends first, known exactly; then the other corners, solved)
            ends = line_ends(w, S, work, ex, GG, LIB)
            tend += fill_ends(pt["tris"], pt["boundary"], ends, h)
            for p in pt.get("parts", []):
                fill_ends(p["tris"], p["boundary"], ends, h)
            tot += fill(S, pt["tris"], pt["boundary"], cand, h, GG, invs, LIB, ex)
            for p in pt.get("parts", []):
                fill(S, p["tris"], p["boundary"], cand, h, GG, invs, LIB, ex)
        print(f"{w}: {tot} corners filled, {tend} line ends reached", flush=True)
    json.dump(P, open("pieces.json", "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
