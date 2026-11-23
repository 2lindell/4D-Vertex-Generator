"""Exact pieces of a curved A4 map (kidecaic, iodecaic): points, lines, then wall pieces bounded by them
(a4_exact_<group>.json), for a4_snub.py to draw. The order follows the swirlprism Symmetry Domain plan: exact points,
exact lines between them, then each wall's pieces as the faces those lines cut its surface into, and only then meshes.

Everything is on the whole A4 chamber (beta >= 0, sum 1). The surfaces are those of the atlas's map
(decaic_atlas/snub_map_<sub>.json, found on the half beta_1 >= beta_4) that carry a wall of the group, each with its
image under the swap (beta read backwards), and the four mirrors beta_k = 0:

  1. lines: every pair of surfaces' common curve in the chamber, traced on both surfaces (each step put back on both by
     Newton's method, to 1e-15), from every seed Newton finds on it;
  2. points: where a third surface crosses a traced curve, solved on all three (a mirror crossing ends the curve);
  3. segments: each curve cut at its points; each classified at its middle (hull tolerance 1e-7): a line shape of the
     group, or an edge where walls meet;
  4. pieces: on each surface the segments cut it into faces (found by walking each segment with the face on its left,
     turning at each point to the next segment clockwise); each face classified at a point just inside it: a wall
     piece where it is a wall shape; then its regions on each side (a step off the surface both ways), so each region's
     extent is the pieces round it;
  5. meshes: each piece triangulated inside its own outline (in the tangent plane at its middle, lifted onto the
     surface), so its edges are the traced lines.

    python a4_exact.py iodecaic
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEC = os.path.join(ROOT, "decaic_atlas")
sys.path.insert(0, HERE)
import a4_map as M  # noqa: E402
import a4_snub as S  # noqa: E402

B = sp.symbols("b1 b2 b3 b4")
ONE = np.ones(4)


class Surf:
    def __init__(self, expr, name):
        self.expr = sp.expand(expr)
        self.name = name
        self._f = sp.lambdify(B, self.expr, "numpy")
        self._g = [sp.lambdify(B, sp.diff(self.expr, v), "numpy") for v in B]

    def f(self, x):
        return float(self._f(*x))

    def g(self, x):
        return np.array([float(gi(*x)) for gi in self._g])


def rev(e):
    return e.subs({B[0]: B[3], B[3]: B[0], B[1]: B[2], B[2]: B[1]}, simultaneous=True)


def same_surface(a, b):
    r = sp.cancel(a / b)
    return r.is_number


def newton(x, Ss, it=60):
    x = np.asarray(x, float).copy()
    for _ in range(it):
        r = np.array([s.f(x) for s in Ss] + [x.sum() - 1])
        J = np.array([s.g(x) for s in Ss] + [ONE])
        st = np.linalg.lstsq(J, r, rcond=None)[0]
        x -= st
        if np.abs(st).max() < 1e-15:
            break
    ok = max(abs(s.f(x)) / (np.linalg.norm(s.g(x)) + 1e-300) for s in Ss) < 1e-11 and abs(x.sum() - 1) < 1e-12
    return x, ok


def tangent(x, A, Bs):
    J = np.array([A.g(x), Bs.g(x), ONE])
    t = np.linalg.svd(J)[2][-1]
    return t / np.linalg.norm(t)


def inside(x, eps=1e-12):
    return x.min() >= -eps


def trace(x0, A, Bs, mirrors, h=2e-3, nmax=4000):
    """The curve A = B = 0 from x0 both ways, inside the chamber; an end on a mirror is solved there."""
    halves = []
    for sgn in (1, -1):
        x, t = x0.copy(), tangent(x0, A, Bs) * sgn
        pts, closed = [], False
        hh = h
        for n in range(nmax):
            tn = tangent(x, A, Bs)
            tn = tn if tn @ t > 0 else -tn
            y, ok = newton(x + hh * tn, [A, Bs])
            if not ok or np.linalg.norm(y - x) > 3 * hh:
                if hh < 1e-6:
                    break
                hh /= 2
                continue
            if not inside(y):
                k = int(np.argmin(y))           # (out through a mirror: the end, solved on it)
                z, ok2 = newton(x + (y - x) * (x[k] / (x[k] - y[k])), [A, Bs, mirrors[k]])
                if ok2 and inside(z, 1e-10):
                    pts.append(np.clip(z, 0, None) if abs(z[k]) < 1e-13 else z)
                break
            pts.append(y)
            x, t = y, tn
            hh = min(h, hh * 1.5)
            if n > 5 and np.linalg.norm(y - x0) < 0.6 * h and sgn == 1:
                closed = True
                break
        halves.append(pts)
        if closed:
            return [x0] + halves[0] + [x0], True
    return halves[1][::-1] + [x0] + halves[0], False


GRID = int(os.environ.get("A4_FACE_GRID", "40"))


def _lattice(n):
    P = [(i, j, k, n - i - j - k) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1 - i - j)]
    return np.array(P, float) / n, {p[:3]: idx for idx, p in enumerate(P)}


def surface_mesh(Sf, si):
    """A fine triangle mesh of the surface in the chamber: a mirror's grid triangles, or marching tetrahedra on a
    curved surface (each vertex put on it)."""
    L, index = _lattice(GRID)
    tets = [[index[v] for v in t] for t in S.kuhn_tets(GRID)]
    if si < 4:
        tris = set()
        for t in tets:
            on = [v for v in t if L[v][si] == 0]
            if len(on) == 3:
                tris.add(tuple(sorted(on)))
        ids = sorted({v for t in tris for v in t})
        ren = {v: n for n, v in enumerate(ids)}
        return [L[v] for v in ids], [[ren[v] for v in t] for t in sorted(tris)]
    f = np.array([Sf.f(x) for x in L])
    V, T, key = [], [], {}

    def vert(a, b):
        kk = (min(a, b), max(a, b))
        if kk not in key:
            t = f[a] / (f[a] - f[b])
            x = L[a] + t * (L[b] - L[a])
            y, ok = newton(x, [Sf])
            V.append(y if ok and np.linalg.norm(y - x) < 2.0 / GRID else x)
            key[kk] = len(V) - 1
        return key[kk]
    for t in tets:
        fv = f[t]
        pos = [v for v in t if f[v] > 0]
        neg = [v for v in t if f[v] <= 0]
        if not pos or not neg:
            continue
        if len(pos) == 1 or len(neg) == 1:
            lone, rest = (pos[0], neg) if len(pos) == 1 else (neg[0], pos)
            T.append([vert(lone, r) for r in rest])
        else:
            a1, a2 = pos
            b1, b2 = neg
            q = [vert(a1, b1), vert(a1, b2), vert(a2, b2), vert(a2, b1)]
            T += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    return V, T


def build(group):
    sub = S.SUB[group]
    shapes = S.own_shapes(group)
    lab = S.classify_many([s["seed"] for s in shapes], sub)
    for s, (sig, key) in zip(shapes, lab):
        s["sig"], s["key"] = sig, key
    keyname = {s["key"]: s["id"] for s in shapes}
    SM = json.load(open(os.path.join(DEC, f"snub_map_{sub}.json")))
    # the surfaces that carry a wall: points spread on each (and its swap image) classified at hull tolerance 1e-7
    rng = np.random.default_rng(5)
    mirrors = [Surf(B[k], f"b{k + 1}") for k in range(4)]
    cand = []
    for text in SM["surfaces"]:
        e = sp.sympify(text)
        cand.append(Surf(e, text))
        er = rev(e)
        if not same_surface(e, er):
            cand.append(Surf(er, "swap of " + text))
    samples, owner = [], []
    for i, s in enumerate(cand):
        got = 0
        for x in rng.dirichlet([1, 1, 1, 1], 400):
            y, ok = newton(x, [s])
            if ok and inside(y, 0) and y.min() > 1e-6:
                samples.append(y)
                owner.append(i)
                got += 1
                if got >= 30:
                    break
    labs = S.classify_many(samples, sub, tol=1e-7)
    walls_k = {s["key"] for s in shapes if s["kind"] == "wall"}
    carry = defaultdict(Counter)
    for i, (_, k) in zip(owner, labs):
        if k in walls_k:
            carry[i][keyname[k]] += 1
    for s_ in shapes:                            # (and every surface through a wall's own seed, or its swap image)
        if s_["kind"] != "wall":
            continue
        b0 = np.asarray(s_["seed"], float) / np.sum(s_["seed"])
        for i, c_ in enumerate(cand):
            for bb in (b0, b0[::-1]):
                if abs(c_.f(bb)) / (np.linalg.norm(c_.g(bb)) + 1e-300) < 1e-7:
                    carry[i][s_["id"] + " (seed)"] += 1
    R = [cand[i] for i in sorted(carry)]
    print(group, "surfaces carrying walls:", [(cand[i].name[:40], dict(carry[i])) for i in sorted(carry)], flush=True)
    def make(R):
        surfs = mirrors + R
        # 1. lines: every pair's common curve
        curves = []
        for a in range(len(surfs)):
            for b in range(a + 1, len(surfs)):
                if a < 4 and b < 4:                 # (two mirrors: a chamber edge, a straight line)
                    pass
                A, Bs = surfs[a], surfs[b]
                found = []
                for x in rng.dirichlet([1, 1, 1, 1], 250):
                    y, ok = newton(x, [A, Bs])
                    if not ok or not inside(y, 1e-12):
                        continue
                    if any(min(np.linalg.norm(np.asarray(c["pts"]) - y, axis=1)) < 5e-3 for c in found):
                        continue
                    pts, closed = trace(y, A, Bs, mirrors)
                    if len(pts) >= 2:
                        found.append({"pair": (a, b), "pts": [np.asarray(p) for p in pts], "closed": closed})
                curves += found
        # (one curve can be found from many pairs: the swap's axis lies on every swap-odd surface; keep one copy, the
        #  others' probes lying on it)
        def seg_dist(q, P):
            P = np.asarray(P); A_, B_ = P[:-1], P[1:]; d = B_ - A_
            t = np.clip(np.einsum("ij,ij->i", q - A_, d) / (np.einsum("ij,ij->i", d, d) + 1e-300), 0, 1)
            return np.linalg.norm(A_ + t[:, None] * d - q, axis=1).min()
        kept = []
        for c in curves:
            P = c["pts"]
            probe = [P[len(P) // 4], P[len(P) // 2], P[(3 * len(P)) // 4]]
            if any(len(k["pts"]) > 1 and all(seg_dist(q, k["pts"]) < 1e-6 for q in probe) for k in kept):
                continue
            kept.append(c)
        curves = kept
        print(" ", len(curves), "curves", flush=True)
        # 2. points: a third surface crossing a curve
        points = []

        def add_point(x):
            for i, p in enumerate(points):
                if np.linalg.norm(p - x) < 1e-9:
                    return i
            points.append(x)
            return len(points) - 1
        for c in curves:
            a, b = c["pair"]
            P = c["pts"]
            cuts = []                                # (index along the polyline, point id)
            for k, C in enumerate(surfs):
                if k in (a, b):
                    continue
                fv = np.array([C.f(p) for p in P])
                sc = np.array([np.linalg.norm(C.g(p)) + 1e-300 for p in P])
                if np.abs(fv / sc).max() < 1e-9:         # (the curve lies on C too: no crossing)
                    continue
                for i in range(len(P) - 1):
                    if abs(fv[i]) < 1e-12 * (1 + np.linalg.norm(C.g(P[i]))) and i in (0,):
                        cuts.append((0.0, add_point(P[0])))
                    if fv[i] * fv[i + 1] < 0 and max(abs(fv[i]) / sc[i], abs(fv[i + 1]) / sc[i + 1]) > 1e-12:
                        t = fv[i] / (fv[i] - fv[i + 1])
                        z, ok = newton(P[i] + t * (P[i + 1] - P[i]), [surfs[a], surfs[b], C])
                        if ok and inside(z, 1e-10):
                            cuts.append((i + t, add_point(z)))
                if abs(fv[-1]) < 1e-12 * (1 + np.linalg.norm(C.g(P[-1]))):
                    cuts.append((len(P) - 1.0, add_point(P[-1])))
            c["cuts"] = sorted(set(cuts))
        print(" ", len(points), "points", flush=True)
        # 3. segments
        segs = []
        for c in curves:
            P = c["pts"]
            cuts = c["cuts"]
            if c["closed"] and not cuts:
                continue
            if not cuts:
                continue
            for (t0, p0), (t1, p1) in zip(cuts, cuts[1:]):
                if p0 == p1 and t1 - t0 < 1:
                    continue
                i0, i1 = int(np.floor(t0)) + 1, int(np.ceil(t1))
                poly = [points[p0]] + [P[i] for i in range(i0, min(i1, len(P)))] + [points[p1]]
                if len(poly) < 2 or sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)) < 1e-9:
                    continue
                segs.append({"pair": c["pair"], "ends": (p0, p1), "pts": poly})
        # (a segment may lie on more surfaces than its pair: every surface zero at three points along it; copies of one
        #  curve found from different pairs are merged)
        merged = []
        for s_ in segs:
            P = s_["pts"]
            probe = [P[len(P) // 4], P[len(P) // 2], P[(3 * len(P)) // 4]]
            s_["on"] = {k for k, C in enumerate(surfs)
                        if all(abs(C.f(q)) / (np.linalg.norm(C.g(q)) + 1e-300) < 1e-9 for q in probe)}
            mid = P[len(P) // 2]
            twin = next((m for m in merged if set(m["ends"]) == set(s_["ends"])
                         and all(abs(surfs[k].f(mid)) / (np.linalg.norm(surfs[k].g(mid)) + 1e-300) < 1e-9
                                 for k in m["on"])), None)
            if twin:
                twin["on"] |= s_["on"]
            else:
                merged.append(s_)
        segs = merged
        mids = []
        for s in segs:
            L = np.cumsum([0] + list(np.linalg.norm(np.diff(s["pts"], axis=0), axis=1)))
            i = int(np.searchsorted(L, L[-1] / 2))
            i = min(max(i, 1), len(s["pts"]) - 1)
            mids.append(newton(s["pts"][i - 1] * 0.5 + s["pts"][i] * 0.5, [surfs[s["pair"][0]], surfs[s["pair"][1]]])[0])
        labs = S.classify_many(mids, sub, tol=1e-7)
        for s, m, (sig, k) in zip(segs, mids, labs):
            s["key"], s["sig"], s["mid"] = k, sig, m
        print(" ", len(segs), "segments", Counter(keyname.get(s["key"], "other") for s in segs), flush=True)
        # 4. pieces: each surface meshed finely in the chamber (a mirror: the grid's triangles on it; a curved surface:
        # marching tetrahedra, each vertex put on the surface), cut along every other surface (each cut vertex solved
        # on both: a point of their exact common curve), and split into its connected parts on one side of all the
        # others: the faces, each bounded by the traced curves
        keep = []
        for si, Sf in enumerate(surfs):
            others = [k for k in range(4, len(surfs)) if k != si]
            V, T = surface_mesh(Sf, si)
            if not T:
                continue
            key_of = {}
            V = list(V)
            for k in others:                    # (cut along surface k)
                C = surfs[k]
                vals = [C.f(v) for v in V]
                newT = []
                for tri in T:
                    sg = [np.sign(vals[a]) if abs(vals[a]) > 1e-14 else 0 for a in tri]
                    if not (1 in sg and -1 in sg):
                        newT.append(tri)
                        continue
                    def cut(a, b):
                        kk = (min(a, b), max(a, b), k)
                        if kk not in key_of:
                            t = vals[a] / (vals[a] - vals[b])
                            z, ok = newton(V[a] + t * (V[b] - V[a]), [Sf, C])
                            if not ok or np.linalg.norm(z - (V[a] + t * (V[b] - V[a]))) > 0.05:
                                z = V[a] + t * (V[b] - V[a])
                            V.append(z)
                            vals.append(0.0)
                            key_of[kk] = len(V) - 1
                        return key_of[kk]
                    a, b, c = tri
                    # (the lone vertex on its side, then the other two)
                    for lone, p, q in ((a, b, c), (b, c, a), (c, a, b)):
                        if sg[tri.index(lone)] != 0 and sg[tri.index(p)] * sg[tri.index(lone)] <= 0 \
                                and sg[tri.index(q)] * sg[tri.index(lone)] <= 0:
                            break
                    sp_, sq_ = sg[tri.index(p)], sg[tri.index(q)]
                    if sp_ == 0 and sq_ != 0:          # (one corner on the cut: two triangles)
                        m = cut(lone, q)
                        newT += [[lone, p, m], [p, q, m]]
                    elif sq_ == 0 and sp_ != 0:
                        m = cut(lone, p)
                        newT += [[lone, m, q], [m, p, q]]
                    else:
                        m1, m2 = cut(lone, p), cut(lone, q)
                        newT += [[lone, m1, m2], [m1, p, q], [m1, q, m2]]
                T = newT
            Varr = np.array(V)
            # (each triangle's side of every other surface, at its centre put on the surface)
            sides_of = []
            cents = []
            for tri in T:
                c = newton(Varr[tri].mean(0), [Sf])[0]
                cents.append(c)
                sides_of.append(tuple(int(np.sign(surfs[k].f(c))) for k in others))
            parent = list(range(len(T)))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            edge_tri = {}
            for ti, tri in enumerate(T):
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    e = (min(a, b), max(a, b))
                    if e in edge_tri and sides_of[edge_tri[e]] == sides_of[ti]:
                        parent[find(ti)] = find(edge_tri[e])
                    edge_tri.setdefault(e, ti)
            comps = defaultdict(list)           # (parts on the same side of every other surface are one piece: the
            for ti in range(len(T)):            #  grid can break a thin sheet hugging a mirror into separate caps)
                comps[sides_of[ti]].append(ti)
            for comp in comps.values():
                area = sum(np.linalg.norm(np.cross(*(Varr[T[t][1:]] - Varr[T[t][0]])[:, :3])) for t in comp) / 2
                if area < 1e-7:
                    continue
                # (its inner point: the triangle centre farthest from the other surfaces)
                depth = lambda t: min([abs(surfs[k].f(cents[t])) / (np.linalg.norm(surfs[k].g(cents[t])) + 1e-300)
                                       for k in others + [kk for kk in range(4) if kk != si]] or [1.0])
                best = max(comp, key=depth)
                ids = sorted({v for t in comp for v in T[t]})
                ren = {v: n for n, v in enumerate(ids)}
                n = Sf.g(cents[best])
                F = {"surface": si, "inner": np.clip(cents[best], 0, None), "normal": (n - n.mean()) / np.linalg.norm(n - n.mean()),
                     "probes": [cents[t] for t in comp[:: max(1, len(comp) // 15)]],
                     "mesh": {"verts": [Varr[v] for v in ids], "faces": [[ren[v] for v in T[t]] for t in comp]}}
                keep.append(F)
        # (a piece's mesh stops up to a grid step short of an exact point on its outline, a corner of the chamber
        #  among them: that gap is filled directly, a fan of triangles from the point to the piece's outline edges
        #  near it)
        r_fill = 1.6 / GRID
        for F in keep:
            Sf = surfs[F["surface"]]
            if F["surface"] < 4:
                continue
            V = F["mesh"]["verts"]
            T = F["mesh"]["faces"]
            ec = Counter()
            for t in T:
                for a_, b_ in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                    ec[(min(a_, b_), max(a_, b_))] += 1
            bd = [e for e, c_ in ec.items() if c_ == 1]
            Varr = np.array(V)
            for p in points:
                if abs(Sf.f(p)) / (np.linalg.norm(Sf.g(p)) + 1e-300) > 1e-9:
                    continue
                if np.linalg.norm(Varr - p, axis=1).min() < 1e-9:
                    continue
                near = [e for e in bd if np.linalg.norm(Varr[e[0]] - p) < r_fill and np.linalg.norm(Varr[e[1]] - p) < r_fill]
                if not near:
                    continue
                V.append(np.asarray(p, float))
                pi = len(V) - 1
                T += [[pi, a_, b_] for a_, b_ in near]
                F.setdefault("filled", []).append(pi)
        labs = S.classify_many([F["inner"] for F in keep], sub, tol=1e-7)
        for F, (sig, k) in zip(keep, labs):
            F["key"], F["sig"] = k, sig
        side_pts = [F["inner"] + e * F["normal"] for F in keep for e in (1e-5, -1e-5)]
        side_labs = S.classify_many(side_pts, sub)
        for i, F in enumerate(keep):
            F["sides"] = [side_labs[2 * i][1] if inside(side_pts[2 * i], 0) else None,
                          side_labs[2 * i + 1][1] if inside(side_pts[2 * i + 1], 0) else None]
        print(" ", len(keep), "faces:", Counter(keyname.get(F["key"], "other") for F in keep), flush=True)
        faces = keep
        make.all_faces = faces
        return surfs, points, segs, keep

    # closure: every face checked at many points (its mesh's inner vertices); where they disagree, the surface through
    # the change is added and the map rebuilt
    for rnd in range(4):
        surfs, points, segs, keep = make(R)
        probes, owner = [], []
        for fi, F in enumerate(keep):
            for q in F["probes"]:
                if inside(q, 0) and q.min() > 1e-7:
                    probes.append(q)
                    owner.append(fi)
        labs = S.classify_many(probes, sub, tol=1e-7)
        bad = defaultdict(list)
        for fi, q, (_, kk) in zip(owner, probes, labs):
            if kk != keep[fi]["key"]:
                bad[fi].append(q)
        if not bad:
            print("  closure: every face is one polychoron at all its probes", flush=True)
            break
        add = set()
        for fi, qs in bad.items():
            F = keep[fi]
            Sf = surfs[F["surface"]]
            a0 = F["inner"]
            for q in qs[:4]:                     # (the change between the face's inner point and a disagreeing probe)
                lo, hi = 0.0, 1.0
                k0 = F["key"]
                for _ in range(30):
                    m = (lo + hi) / 2
                    x = newton(a0 + m * (q - a0), [Sf])[0]
                    if M.classify([x], sub, 1e-7)[0][1] == k0:
                        lo = m
                    else:
                        hi = m
                x = newton(a0 + lo * (q - a0), [Sf])[0]
                for ci, c_ in enumerate(cand):
                    if c_ not in R and abs(c_.f(x)) / (np.linalg.norm(c_.g(x)) + 1e-300) < 1e-6:
                        add.add(ci)
            print(f"  closure: face of {keyname.get(F['key'], F['key'])} on {Sf.name[:30]} disagrees at {len(qs)} probes",
                  flush=True)
        if not add:
            print("  closure: no surface of the map through the changes found", flush=True)
            break
        R = R + [cand[ci] for ci in sorted(add)]
        print("  closure: added", [cand[ci].name[:40] for ci in sorted(add)], flush=True)
    out = {"surfaces": [s.name for s in surfs], "points": points, "segments": segs, "faces": keep,
           "all_faces": [{k: v for k, v in F.items() if k != "mesh"} for F in make.all_faces],
           "keyname": keyname, "shapes": shapes}
    return out


if __name__ == "__main__":
    import pickle
    for g in sys.argv[1:] or ["iodecaic"]:
        res = build(g)
        pickle.dump(res, open(os.path.join(HERE, f"a4_exact_{g}.pkl"), "wb"))
