"""Each wall's piece, built from its equations alone (pieces.json).

A wall's piece is bounded by stretches, each exact:
  * an edge on a line L: every piece of L lying on the wall's surface (its verified points, exact_defs.json), each with
    its own second surface (one through that piece of L that crosses the wall);
  * an edge on, or a cut by, another wall Y: the curve where the wall's surface meets Y's surface (wall_equations.json),
    traced exactly from one of its corners to the next (Newton onto both surfaces at every step);
  * the faces of the half-cell and its mirror, where the piece is cut for storage only (the views cut it themselves).
The wall's surface is meshed by marching tetrahedra over the half-cell; from a point known to be on the wall (its seed,
carried onto its own surface by the group) the mesh is filled out, a mesh edge being crossed unless one of the
stretches cuts it there (its second surface changes sign along the mesh edge, at a point on the stretch). Where the
piece ends, the boundary point is solved exactly onto the wall's surface and that stretch's second surface (or the
face). Each boundary point is labelled with its edge. Cuts are stretches too, but they divide the piece rather than end
it: the cut curves are kept with the piece, for the white lines.

The piece is checked: points of it should classify as the wall.

    python pieces.py [X14,X41,...] [h]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, deque

import numpy as np
from scipy.spatial import cKDTree

import optimal_seeds as osd
from probe import _one
from t2_pieces import extended
from xsurface import CUBE, TETS, U, C0

FOLD = np.eye(4)[[1, 0, 3, 2]]
OUT = os.environ.get("PIECES_OUT", "pieces.json")
HALF = [("face b1=0", np.eye(4)[0]), ("face b2=0", np.eye(4)[1]), ("face b3=0", np.eye(4)[2]),
        ("face b4=0", np.eye(4)[3]), ("mirror b3=b4", np.array([0, 0, 1, -1.0]))]


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def fval(s, x):
    return float(np.atleast_1d(s.f(np.atleast_2d(x)).ravel())[0])


def grad(s, x):
    if isinstance(s, osd.Plane):
        g = s.n.astype(float)
    else:
        e = 1e-7
        g = np.array([(fval(s, x + e * u) - fval(s, x - e * u)) / (2 * e) for u in np.eye(4)])
    return g - g.mean()


def project(surfs, x, steps=30):
    """x moved onto all the surfaces (sum(b) = 1 kept), by Newton steps (least change)."""
    x = np.asarray(x, float).copy()
    for _ in range(steps):
        F = np.array([fval(s, x) for s in surfs])
        if np.abs(F).max() < 1e-14:
            break
        J = np.array([grad(s, x) for s in surfs])
        x = x - J.T @ np.linalg.lstsq(J @ J.T, F, rcond=None)[0]
    return x


def in_half(x, tol=1e-12):
    return (x > -tol).all() and x[2] >= x[3] - tol


def trace(A, B, p0, targets, step=2e-3, max_len=2.0):
    """The curve where A and B meet, from p0 (each way), until it reaches another of the targets (within 3e-3) or
    leaves the half-cell: the runs that end on a target."""
    out = []
    for sgn in (1, -1):
        x, pts, prev = np.asarray(p0, float), [np.asarray(p0, float)], None
        for _ in range(int(max_len / step)):
            M = np.vstack([grad(A, x), grad(B, x), np.ones(4)])
            t = np.linalg.svd(M)[2][-1]
            if prev is None:
                t = sgn * t
            elif t @ prev < 0:
                t = -t
            prev = t
            x = nrm(project([A, B], x + step * t / np.linalg.norm(t)))
            if not in_half(x, 1e-9):                         # (out of the half-cell: the run ends on its face)
                if len(pts) > 2:
                    out.append(np.array(pts))
                break
            pts.append(x)
            d = [np.linalg.norm(x - q) for q in targets]
            hit = [k for k, dd in enumerate(d) if dd < 3e-3 and np.linalg.norm(q := targets[k]) and np.linalg.norm(q - p0) > 1e-6]
            if hit:
                pts[-1] = targets[hit[0]]
                out.append(np.array(pts))
                break
    return out


class Stretch:
    """A boundary stretch: its points (None: the whole curve where the wall meets surf), its second surface."""

    def __init__(self, name, P, surf, kind):
        self.name, self.surf, self.kind = name, surf, kind
        self.P = None if P is None else np.asarray(P, float)
        self.tree = None if P is None else cKDTree(self.P)


def dens(P, h):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        k = max(1, int(np.linalg.norm(b - a) / h))
        out += [a + (b - a) * t for t in np.linspace(0, 1, k + 1)[1:]]
    return np.array(out)


def build(w, S, stretches, ref, h=0.008, max_cells=120000):
    """The piece: (triangles, boundary segments [(point, label), (point, label)], cut curves, capped)."""
    node = lambda n: C0 + U.T @ (h * np.asarray(n, float))
    Fn = {}

    def F(n):
        if n not in Fn:
            Fn[n] = fval(S, node(n))
        return Fn[n]
    verts = {}

    def vert(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in verts:
            fa, fb = F(k[0]), F(k[1])
            verts[k] = nrm(project([S], node(k[0]) + fa / (fa - fb) * (node(k[1]) - node(k[0]))))
        return k

    def cell_tris(c):
        ns = [tuple(int(v) for v in np.add(c, d)) for d in CUBE]
        vals = [F(n) for n in ns]
        out = []
        for tet in TETS:
            q = [ns[i] for i in tet]
            fs = [vals[i] for i in tet]
            pos = [i for i in range(4) if fs[i] > 0]
            neg = [i for i in range(4) if fs[i] <= 0]
            if not pos or not neg:
                continue
            if len(pos) == 1 or len(neg) == 1:
                one, rest = (pos, neg) if len(pos) == 1 else (neg, pos)
                out.append(tuple(vert(q[one[0]], q[r]) for r in rest))
            else:
                a, b_ = pos
                c_, d = neg
                k = [vert(q[a], q[c_]), vert(q[a], q[d]), vert(q[b_], q[d]), vert(q[b_], q[c_])]
                out += [(k[0], k[1], k[2]), (k[0], k[2], k[3])]
        return out
    bars = [s for s in stretches if s.kind != "cut"]

    def blocker(u, v):
        """The stretch cutting the mesh edge u-v, if any (its second surface changes sign along it, at a point on
        the stretch), and the exact point where it does."""
        a, b = verts[u], verts[v]
        for s in bars:
            if s.tree is not None and s.tree.query((a + b) / 2)[0] > 2.5 * h:
                continue
            fa, fb = fval(s.surf, a), fval(s.surf, b)
            if fa == 0 or np.sign(fa) == np.sign(fb):
                continue
            x = nrm(project([S, s.surf], a + fa / (fa - fb) * (b - a)))
            if s.tree is None or s.tree.query(x)[0] < 1.2 * h:
                return s, x
        return None
    # (the mesh: grown from the reference point across the cells the piece reaches)
    start = tuple(int(v) for v in np.floor(U @ (ref - C0) / h))
    cells, tris = set(), {}
    queue = deque([start])
    reached, seen_v = set(), set()
    adj = {}
    ref_key = None
    while queue and len(cells) < max_cells:
        c = queue.popleft()
        if c in cells:
            continue
        cells.add(c)
        tris[c] = cell_tris(c)
        for t in tris[c]:
            for a in range(3):
                u, v = t[a], t[(a + 1) % 3]
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
        if ref_key is None:
            ks = [k for t in tris[c] for k in t]
            if ks:
                ref_key = min(ks, key=lambda k: np.linalg.norm(verts[k] - ref))
                reached.add(ref_key)
        # (spread over the vertices known so far; reached vertices of the new cell looked at again, for its edges)
        for t in tris[c]:
            for k in t:
                if k in reached:
                    seen_v.discard(k)
        grow = deque(k for k in reached if k not in seen_v)
        while grow:
            k = grow.popleft()
            if k in seen_v:
                continue
            seen_v.add(k)
            for m in adj.get(k, ()):
                if m in reached or not in_half(verts[m]) or blocker(k, m):
                    continue
                reached.add(m)
                grow.append(m)
        if any(k in reached for t in tris[c] for k in t):
            for d in [(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1)]:
                n = tuple(np.add(c, d))
                if n not in cells:
                    queue.append(n)
    # (the triangles: all corners reached kept; a mesh edge from a reached corner to one not reached cut exactly)
    cross = {}

    def boundary(ki, ko):
        if (ki, ko) not in cross:
            bl = blocker(ki, ko)
            if bl:
                s, x = bl
                cross[(ki, ko)] = (x, s.name)
            else:
                a, b = verts[ki], verts[ko]
                for name, n in HALF:
                    fa, fb = a @ n, b @ n
                    if fa >= 0 > fb:
                        pl = osd.Plane(n, name)
                        cross[(ki, ko)] = (nrm(project([S, pl], a + fa / (fa - fb) * (b - a))), name)
                        break
                else:
                    cross[(ki, ko)] = ((a + b) / 2, "?")
        return cross[(ki, ko)]
    out_tris, segs = [], []
    for c in cells:
        for t in tris[c]:
            ins = [k in reached for k in t]
            if not any(ins):
                continue
            if all(ins):
                out_tris.append([verts[k] for k in t])
                continue
            poly, ends = [], []
            for a in range(3):
                u, v = t[a], t[(a + 1) % 3]
                if u in reached:
                    poly.append(verts[u])
                if (u in reached) != (v in reached):
                    x, name = boundary(u, v) if u in reached else boundary(v, u)
                    poly.append(x)
                    ends.append((x, name))
            for k in range(1, len(poly) - 1):
                out_tris.append([poly[0], poly[k], poly[k + 1]])
            if len(ends) == 2:
                segs.append(ends)
    # (zero-area slivers left by the cuts are dropped)
    area = lambda t: np.linalg.norm(np.cross(*(U @ (np.asarray(t[1:]) - t[0]).T).T))
    out_tris = [t for t in out_tris if area(t) > 1e-14]
    return out_tris, segs, len(cells) >= max_cells




def ends_here(w, S, run, h):
    """The wall ends along the run: at most of the points tested inside it, one side (just off it along S) is the wall
    and the other not (sides past the half-cell's faces tell nothing)."""
    if len(run) < 5:
        return False
    hits = tried = 0
    for i in np.linspace(1, len(run) - 2, 7).astype(int):
        p, t = run[i], run[i + 1] - run[i - 1]
        T_ = np.linalg.svd(np.vstack([grad(S, p), np.ones(4), t]))[2][-1]
        ys = [nrm(project([S], p + sg * 3 * h * T_)) for sg in (1, -1)]
        if not all(in_half(y) for y in ys):
            continue
        tried += 1
        hits += (_one(ys[0])[0] == w) != (_one(ys[1])[0] == w)
    return tried >= 3 and hits >= 0.6 * tried


def line_pieces_on(L, S, GG, LIB):
    """The pieces of a line (its verified pieces carried by every group element) lying on the surface S in the
    half-cell, each with the library surfaces through it."""
    out, seen = [], set()
    for P in L["points"]:
        P = np.asarray(P, float)
        if len(P) < 2:
            continue
        if len(P) == 2:
            P = np.vstack([P[0], nrm(P[0] + P[1]), P[1]])
        Y = np.einsum("gij,nj->gni", GG, P)
        sm = Y.sum(-1, keepdims=True)
        good = (np.abs(sm) > 1e-9).all((1, 2))
        Y = Y[good] / sm[good]
        ok = (Y > -3e-6).all((1, 2)) & (Y[..., 2] >= Y[..., 3] - 3e-6).all(1)   # (6 decimals kept)
        for Q in Y[ok]:
            key = tuple(np.round(np.sort(Q[[0, -1]], axis=0).ravel(), 6))
            if key in seen:
                continue
            gs = np.array([np.linalg.norm(grad(S, q)) for q in Q]) + 1e-300
            if (np.abs([fval(S, q) for q in Q]) / gs).max() > 3e-6:
                continue
            seen.add(key)
            probe = Q[np.linspace(0, len(Q) - 1, 3).astype(int)]
            Sn = [n for n, s_ in LIB.items() if all(abs(fval(s_, q)) / (np.linalg.norm(grad(s_, q)) + 1e-300) < 3e-6 for q in probe)]
            out.append((Q, Sn))
    return out


def patch(w, Sname, own, work, eq, ex, seeds, LIB, GG, h):
    """One patch of the wall: the part of the surface Sname bounded by its stretches (and its other planes)."""
    S = LIB[Sname]
    refs = []
    if w in seeds:
        b0 = nrm(seeds[w])
        for g in GG:
            y = g @ b0
            if abs(y.sum()) > 1e-12:
                y = y / y.sum()
                if in_half(y) and abs(fval(S, y)) < 1e-10 and not any(np.allclose(y, r) for r in refs):
                    refs.append(y)
    if not refs:
        # (no image of the seed on this surface: a point of the wall just off one of its edge lines lying on it)
        for e in work["walls"][w]["edges"]:
            if e not in ex["lines"]:
                continue
            for P, _ in line_pieces_on(ex["lines"][e], S, GG, LIB):
                p = P[len(P) // 2]
                t = P[min(len(P) - 1, len(P) // 2 + 1)] - P[max(0, len(P) // 2 - 1)]
                T_ = np.linalg.svd(np.vstack([grad(S, p), np.ones(4), t]))[2][-1]
                for r in (3 * h, 6 * h):
                    for sg in (1, -1):
                        y = nrm(project([S], p + sg * r * T_))
                        if in_half(y) and _one(y)[0] == w:
                            refs.append(y)
            if refs:
                break
    if not refs:
        print(f"{w} on {Sname}: no point of the wall on this surface", flush=True)
        return None
    stretches = []
    on_S = {e: line_pieces_on(ex["lines"][e], S, GG, LIB) for e in work["walls"][w]["edges"] if e in ex["lines"]}
    for e in work["walls"][w]["edges"]:
        if e in ex["lines"]:
            for P, Sn in on_S[e]:
                cand = []
                for n in Sn:
                    if n == Sname or n not in LIB:
                        continue
                    Q = P[1:-1] if len(P) > 2 else P              # (inside the piece: at an end point, a vertex of
                    Q = Q[::max(1, len(Q) // 5)]                  # the cell, surfaces can be tangent)
                    ga = np.array([grad(LIB[n], p) for p in Q])
                    gb = np.array([grad(S, p) for p in Q])
                    c = np.abs((ga * gb).sum(1)) / (np.linalg.norm(ga, axis=1) * np.linalg.norm(gb, axis=1) + 1e-300)
                    if c.max() < 0.999:
                        cand.append((float(c.max()), n))
                # (a surface crossing the wall clearly all along the piece, a plane first (it cannot be tangent along
                # it); else the one crossing most steeply)
                clear = sorted((not n.startswith("plane"), c, n) for c, n in cand if c < 0.95)
                cand = [n for _, _, n in clear] or [n for _, n in sorted(cand)]
                if cand:
                    stretches.append(Stretch(e, dens(P, h / 3), LIB[cand[0]], "edge"))
        else:
            r = eq[w]["edges"].get(e, {})
            # (each of the other wall's own surfaces bounds it too, all along where they meet: a stretch the
            # corners did not reach is still closed)
            traced = {rr["surface"] for rr in ([r] if r.get("surface") else []) + r.get("more", [])}
            for n in ex["walls"].get(e, {}).get("surface", []):
                if n in LIB and n not in traced and n != Sname:
                    stretches.append(Stretch(e, None, LIB[n], "edge"))
            for rr in ([r] if r.get("surface") else []) + r.get("more", []):
                if rr["surface"] not in LIB:
                    continue
                C = [np.asarray(c, float) for c in rr.get("corners", [])]
                for c in C:
                    for run in trace(S, LIB[rr["surface"]], c, C):
                        stretches.append(Stretch(e, run, LIB[rr["surface"]], "edge"))
            # (a copy of the other wall may meet this one at the same corners: every other image of its surfaces
            # through a corner, traced, where the wall does end along it (its two sides not both the wall))
            bases = {n.split(":")[0] for n in ex["walls"].get(e, {}).get("surface", [])}
            used = {rr["surface"] for rr in ([r] if r.get("surface") else []) + r.get("more", [])}
            C = [np.asarray(c, float) for rr in ([r] if r.get("surface") else []) + r.get("more", [])
                 for c in rr.get("corners", [])]
            C = [c for c in C if (np.abs(c) < 1e-6).sum() < 2]
            for n, sY in LIB.items():
                if n in used or n == Sname or n.split(":")[0] not in bases:
                    continue
                for c in C:
                    if abs(fval(sY, c)) / (np.linalg.norm(grad(sY, c)) + 1e-300) > 1e-6:
                        continue
                    for run in trace(S, sY, nrm(project([S, sY], c)), C):
                        if ends_here(w, S, run, h):
                            stretches.append(Stretch(e, run, sY, "edge"))
    # (an end of one of the wall's lines inside the half-cell, where none of its other lines is: a wall edge meets it
    # there, perhaps a copy of that wall (an image of its surface through the end), whose curve bounds the patch too)
    allP = [(e, P) for e, ps in on_S.items() for P, _ in ps if np.linalg.norm(P[-1] - P[0]) > 1e-6]   # (not a lone point)
    same = lambda P, Q: min(np.linalg.norm(P[0] - Q[0]) + np.linalg.norm(P[-1] - Q[-1]),
                            np.linalg.norm(P[0] - Q[-1]) + np.linalg.norm(P[-1] - Q[0])) < 2e-4   # (the same piece twice)
    walls_e = [e for e in work["walls"][w]["edges"] if e not in ex["lines"]]
    for e, P in allP:
        for p in (P[0], P[-1]):
            if p.min() < 1e-6 or p[2] - p[3] < 1e-6 or any(
                    Q is not P and not same(P, Q) and np.min(np.linalg.norm(Q - p, axis=1)) < 1e-4 for _, Q in allP):
                continue
            for Y in walls_e:
                bases = {n.split(":")[0] for n in ex["walls"].get(Y, {}).get("surface", [])}
                for n, sY in LIB.items():
                    if n == Sname or n.split(":")[0] not in bases or isinstance(sY, osd.Plane) != (n.startswith("plane")):
                        continue
                    if abs(fval(sY, p)) / (np.linalg.norm(grad(sY, p)) + 1e-300) > 1e-4:
                        continue
                    if any(s_.surf is sY and s_.tree is not None and s_.tree.query(p)[0] < 2 * h for s_ in stretches):
                        continue
                    c = nrm(project([S, sY], p))
                    for run in trace(S, sY, c, [c]):
                        if ends_here(w, S, run, h):
                            stretches.append(Stretch(Y, run, sY, "edge"))
    # (a flat wall's other planes: where its patches meet, a seam that bounds this patch)
    for n in own:
        if n != Sname:
            stretches.append(Stretch(f"seam {n}", None, LIB[n], "seam"))
    cuts = []
    for Y, r in eq[w]["cuts"].items():
        if r.get("surface") in LIB:
            C = [np.asarray(c, float) for c in r.get("corners", [])]
            for c in C:
                for run in trace(S, LIB[r["surface"]], c, C):
                    cuts.append(Stretch(Y, run, LIB[r["surface"]], "cut"))
    # (the reference point: of the seed's images on the wall's own surface, the one farthest from every stretch)
    far = lambda y: min([s.tree.query(y)[0] for s in stretches if s.tree is not None]
                        + [min(y.min(), (y[2] - y[3]) / 2)] or [1.0])
    ref = max(refs, key=far)
    if far(ref) < 2 * h:
        # (the seed lies on an edge: a point of the wall's surface a little off it, still the wall)
        T_ = np.linalg.svd(np.vstack([grad(S, ref), np.ones(4)]))[2][2:]
        tries = []
        for r in (3 * h, 6 * h, 12 * h):
            for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                y = nrm(project([S], ref + r * (np.cos(ang) * T_[0] + np.sin(ang) * T_[1])))
                if in_half(y) and _one(y)[0] == w:
                    tries.append(y)
        if tries:
            ref = max(tries, key=far)
    tris, segs, capped = build(w, S, stretches, ref, h)
    T = np.array(tris)
    lab = Counter(n for sg in segs for _, n in sg)
    rng = np.random.default_rng(0)
    pts = [nrm(project([S], p)) for p in (T.mean(1)[rng.choice(len(T), min(80, len(T)), replace=False)] if len(T) else [])]
    cls = Counter(_one(p)[0][:8] for p in pts)
    rec = {"surface": Sname, "ref": ref.round(12).tolist(), "h": h,
              "stretches": [[s.name, s.surf.name, None if s.P is None else len(s.P)] for s in stretches],
              "tris": T.round(10).tolist(),
              "boundary": [[[x.round(10).tolist(), n] for x, n in sg] for sg in segs],
              "cuts": [[s.name, s.P.round(10).tolist()] for s in cuts],
              "check": {"share_classified_as_wall": round(cls.get(w, 0) / max(1, len(pts)), 3),
                        "classified": dict(cls.most_common(5))},
              "bounded_by": dict(lab), "capped": capped}
    print(f"{w} on {Sname}: {len(T)} triangles; {len(stretches)} stretches; bounded by {dict(lab)}; "
          f"points classified {dict(cls.most_common(4))}{' (capped)' if capped else ''}", flush=True)
    return rec

def main(only=None, h=0.008):
    work = json.load(open("working_data.json"))
    eq = json.load(open("wall_equations.json"))["walls"]
    ex = json.load(open("exact_defs.json"))
    seeds = {k: v["beta"] for k, v in json.load(open("optimal_seeds.json")).items()}
    for e in json.load(open("extra_samples.json")):            # (a wall's own example seed, where it has one)
        k = e.get("kind", "").split(" ")[0]
        if k.startswith(("X", "F", "T")) and k not in seeds:
            seeds[k] = e["beta"]
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    G = extended()
    GG = np.concatenate([G, G @ FOLD])
    out = json.load(open(OUT)) if only and os.path.exists(OUT) else {}
    for w in work["walls"]:
        if only and w not in only:
            continue
        own = ex["walls"][w]["surface"]
        patches = []
        for Sname in own:                                  # (a curved wall: its surface; a flat wall: each plane)
            rec = patch(w, Sname, own, work, eq, ex, seeds, LIB, GG, h)
            if rec:
                patches.append(rec)
        out[w] = {"patches": patches}
        json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None,
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.008)
