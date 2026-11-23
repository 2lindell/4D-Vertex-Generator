"""Exact pieces for the curved A4 maps (kidecaic, iodecaic), from a4_exact_<group>.pkl, as the swirlprism atlas makes
its Symmetry Domain's walls (swirlprism_atlas/DOMAIN_PLAN.md, sphere_map.domain_pieces): every piece triangulated
inside its own exact loop, the loop's points its only rim vertices, every inner vertex put on the exact surface.

  1. Lines that stop inside the chamber are carried on to the chamber's boundary along their own curve (traced on the
     pair of its surfaces that cross best there, the pair a4_exact traced having turned tangent), and ended at the
     exact point: on an edge of the chamber that both surfaces contain, where their gradients across it are parallel
     (sawbosid's and gawbosid's ends, beta = (0, 1/2 +- sqrt21/14, 0, 1/2 -+ sqrt21/14)); else on the mirror.
  2. A piece's loop: the exact segments on its surface along its marching mesh's rim (whose cut vertices a4_exact put on
     the exact curves), chained at their exact points; more than one loop: the largest outer, the others holes.
  3. The loop laid flat across the surface's mean normal and triangulated inside (Shewchuk's triangle: no points added
     on the rim, angles over 20 degrees, a largest area), the inner vertices moved along the normal onto the surface;
     other normals tried where a loop crosses itself flat or a triangle turns over. A piece whose loop does not close
     keeps its marching mesh (listed).

    python a4_pieces.py kidecaic iodecaic        # a4_pieces_<group>.pkl, read by a4_snub.py
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import sympy as sp
import triangle as tr
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_exact as E  # noqa: E402

BZ = np.linalg.svd(np.ones((1, 4)))[2][1:]          # (an orthonormal basis of sum(beta) = 0: the chart's 3-space)
MAX_AREA = float(os.environ.get("A4_PIECE_AREA", "1.5e-4"))


def surf_of(name):
    if name in ("b1", "b2", "b3", "b4"):
        return E.Surf(E.B[int(name[1]) - 1], name)
    if name.startswith("swap of "):
        return E.Surf(E.rev(sp.sympify(name[8:])), name)
    return E.Surf(sp.sympify(name), name)


def normal(S, x):
    g = S.g(x)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-14 else g


# ------------------------------------------------------------------------------------------------ 1. lines to their ends
def edge_end(A, Bs, i, j, x):
    """The point on the chamber edge beta_i = beta_j = 0 (both surfaces containing it) where the curve A = B = 0 meets
    it: their gradients across the edge parallel there. The root nearest x, or None."""
    b, p = E.B, sp.symbols("p")
    sub = {b[i]: 0, b[j]: 0}
    if sp.expand(A.expr.subs(sub)) != 0 or sp.expand(Bs.expr.subs(sub)) != 0:
        return None
    k, l_ = [m for m in range(4) if m not in (i, j)]
    det = sp.expand((sp.diff(A.expr, b[i]) * sp.diff(Bs.expr, b[j]) - sp.diff(A.expr, b[j]) * sp.diff(Bs.expr, b[i]))
                    .subs(sub).subs({b[k]: p, b[l_]: 1 - p}))
    if det == 0:
        return None
    best = None
    for r in sp.Poly(det, p).real_roots():
        v = float(r)
        if 1e-9 < v < 1 - 1e-9:
            y = np.zeros(4)
            y[k], y[l_] = v, 1 - v
            if best is None or np.linalg.norm(y - x) < np.linalg.norm(best - x):
                best = y
    return best


def extend_lines(X, surfs, log):
    """Segments of line shapes ending inside the chamber, at a point no other segment of the map meets, carried on
    to the chamber's boundary."""
    kn = X["keyname"]
    lines = {k for k, s in ((s["key"], s) for s in X["shapes"]) if s.get("kind") == "line"}
    deg = Counter(e for s in X["segments"] for e in s["ends"])
    mir = surfs[:4]
    added = 0
    for s in list(X["segments"]):
        if s["key"] not in lines:
            continue
        for end in s["ends"]:
            x = np.asarray(X["points"][end], float)
            if deg[end] != 1 or x.min() < 1e-9:
                continue
            Q = np.asarray(s["pts"], float)
            if np.abs(Q[:, 0] - Q[:, 3]).max() < 1e-6 and np.abs(Q[:, 1] - Q[:, 2]).max() < 1e-6:
                continue                                 # (the swap's axis: drawn whole, corner to corner)
            Q = Q[::-1] if s["ends"][0] == end else Q        # (Q ends at the dangling point)
            on = [surfs[k] for k in s["on"]]
            t = Q[-1] - Q[-min(10, len(Q))]
            t /= np.linalg.norm(t)
            pts, y = [x], x
            for _ in range(20000):
                pair = max(((a, b) for a in on for b in on if a is not b), key=lambda ab: E.crossing_angle(y, *ab))
                tn = E.tangent(y, *pair)
                tn = tn if tn @ t > 0 else -tn
                z, ok = E.newton(y + 1e-4 * tn, list(pair))
                if not ok:
                    break
                if z.min() < 0:
                    k = int(np.argmin(z))
                    w, ok2 = E.newton(y + (z - y) * (y[k] / (y[k] - z[k])), list(pair) + [mir[k]])
                    w = np.clip(w, 0, None)
                    # (on a chamber edge both surfaces contain: its exact point)
                    for j in range(4):
                        if j != k and w[j] < 1e-4:
                            e = next((edge_end(a, b, min(j, k), max(j, k), w) for a in on for b in on
                                      if a is not b and edge_end(a, b, min(j, k), max(j, k), w) is not None), None)
                            if e is not None and np.linalg.norm(e - w) < 1e-3:
                                w = e
                    pts.append(w)
                    break
                t, y = tn, z
                pts.append(y)
            if len(pts) < 2 or pts[-1].min() > 1e-12:
                log(f"  line {kn.get(s['key'], s['key'])} from {np.round(x, 4)}: not carried to the boundary")
                continue
            X["points"].append(pts[-1])
            pi = len(X["points"]) - 1
            P = pts[:: max(1, len(pts) // 200)] + [pts[-1]]
            X["segments"].append({"pair": s["pair"], "ends": (end, pi), "pts": P, "on": set(s["on"]),
                                  "key": s["key"], "extended": True})
            deg[end] += 1
            added += 1
            log(f"  line {kn.get(s['key'], s['key'])}: carried from {np.round(x, 4)} to {np.round(pts[-1], 8)} "
                f"({len(pts)} steps)")
    return added


# ------------------------------------------------------------------------------------------------ 2. loops of the pieces
def rim_of(F):
    # (without a4_exact's gap fills: fans from an exact point, which the exact corners put back here replace)
    fl = set(F.get("filled", []))
    T = [t for t in F["mesh"]["faces"] if not fl.intersection(t)]
    ec = Counter(tuple(sorted(e)) for t in T for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    return [e for e, c in ec.items() if c == 1]


def loops_of(F, X, by_surf, h):
    """The piece's exact loops: segments on its surface whose points all lie along its mesh's rim, chained."""
    V = np.asarray(F["mesh"]["verts"], float)
    rim = rim_of(F)
    if not rim:
        return None
    rv = sorted({a for e in rim for a in e})
    # (the rim, densified, so a segment along it is near it at every sample)
    R = [V[a] + u * (V[b] - V[a]) for a, b in rim for u in np.linspace(0, 1, 6)]
    tree = cKDTree(np.array(R))
    use = []
    for si in by_surf[F["surface"]]:
        s = X["segments"][si]
        Q = np.asarray(s["pts"], float)
        inner = Q[1:-1] if len(Q) > 2 else Q
        d = tree.query(inner)[0]
        if len(d) and np.mean(d < h) > 0.85 and np.median(d) < 0.5 * h:
            use.append(si)
    if not use:
        return None
    nb = defaultdict(list)
    for si in use:
        a, b = X["segments"][si]["ends"]
        nb[a].append(si)
        nb[b].append(si)
    if any(len(v) != 2 for v in nb.values()):
        return None
    loops, left = [], set(use)
    while left:
        si = left.pop()
        s = X["segments"][si]
        start, cur = s["ends"]
        pts = list(np.asarray(s["pts"], float))
        while cur != start:
            nx = [k for k in nb[cur] if k in left]
            if not nx:
                return None
            si = nx[0]
            left.discard(si)
            s = X["segments"][si]
            Q = np.asarray(s["pts"], float)
            if s["ends"][0] == cur:
                pts += list(Q[1:])
                cur = s["ends"][1]
            else:
                pts += list(Q[::-1][1:])
                cur = s["ends"][0]
        loops.append(np.array(pts[:-1]))
    return loops


def rim_loops(F, S, P, h):
    """The piece's loops from its marching mesh's rim (whose cut vertices a4_exact solved onto the exact curves), each
    exact point of the map on its surface and near the rim put into it at the rim edge nearest it: the corners the
    marching mesh cut off. (Where its segments do not chain: curves traced from several pairs, overlapping.)"""
    V = np.asarray(F["mesh"]["verts"], float)
    rim = rim_of(F)
    if not rim:
        return None
    # (each loop walked round the triangles at each of its vertices, so a vertex where the rim touches itself (a
    #  pinch) is passed on the right side; a loop through a vertex twice is split there into two)
    fl = set(F.get("filled", []))
    T = [tuple(t) for t in F["mesh"]["faces"] if not fl.intersection(t)]
    tri_of = defaultdict(list)
    for ti, t in enumerate(T):
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            tri_of[frozenset(e)].append(ti)
    rimset = {frozenset(e) for e in rim}
    left = set(rimset)
    loops = []
    while left:
        e0 = left.pop()
        a, b = tuple(e0)
        t = tri_of[e0][0]
        L = [a]
        cur_e, prev = e0, a
        for _ in range(len(rim) + 5):
            # (from the rim edge prev-b, turn round b through its triangles to the next rim edge)
            ti = tri_of[cur_e][0] if len(tri_of[cur_e]) == 1 else t
            other = prev
            for _ in range(200):
                c = next(v for v in T[ti] if v not in (b, other))
                e = frozenset((b, c))
                if e in rimset:
                    break
                ti = next(x for x in tri_of[e] if x != ti)
                other = c
            else:
                return None
            if e == e0:
                break
            left.discard(e)
            L.append(b)
            prev, b, cur_e, t = b, c, e, ti
        else:
            return None
        # (split at a repeated vertex)
        stack, out = [L], []
        while stack:
            M_ = stack.pop()
            seen = {}
            for i, v in enumerate(M_):
                if v in seen:
                    j = seen[v]
                    stack += [M_[j:i], M_[:j] + M_[i:]]
                    break
                seen[v] = i
            else:
                out.append(M_)
        loops += [[V[i] for i in M_] for M_ in out if len(M_) >= 3]
    if not loops:
        return None
    on = [p for p in P if abs(S.f(p)) / (np.linalg.norm(S.g(p)) + 1e-300) < 1e-9]
    keep = []
    for L in loops:
        L = list(L)
        fixed = set()
        for p in on:
            Q = np.array(L)
            A_, B_ = Q, np.roll(Q, -1, 0)
            d = B_ - A_
            u = np.clip(np.einsum("ij,ij->i", p - A_, d) / (np.einsum("ij,ij->i", d, d) + 1e-300), 0, 1)
            dist = np.linalg.norm(A_ + u[:, None] * d - p, axis=1)
            i = int(np.argmin(dist))
            if dist[i] > h:
                continue
            if np.linalg.norm(Q - p, axis=1).min() < 1e-9:
                fixed.add(int(np.argmin(np.linalg.norm(Q - p, axis=1))))
                continue
            L.insert(i + 1, np.asarray(p, float))
            fixed = {j + (j > i) for j in fixed} | {i + 1}
        keep.append((np.array(L), fixed))
    return keep


# --------------------------------------------------------------------------------------------- 3. the loops triangulated
def thin(P, h, fixed=()):
    """A loop's points kept at least h apart along it (its exact points always: every original corner of its
    segments, where it turns)."""
    out = [P[0]]
    for i in range(1, len(P)):
        turn = False
        if 0 < i < len(P) - 1:
            a, b = P[i] - P[i - 1], P[i + 1] - P[i]
            turn = a @ b < 0.9 * np.linalg.norm(a) * np.linalg.norm(b)
        if np.linalg.norm(P[i] - out[-1]) >= h or turn or i in fixed:
            out.append(P[i])
    return np.array(out)


def crossings(Q):
    n = len(Q)
    c = 0
    for i in range(n):
        a, b = Q[i], Q[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p, q = Q[j], Q[(j + 1) % n]
            d = lambda u, v, w: (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
            if d(a, b, p) * d(a, b, q) < 0 and d(p, q, a) * d(p, q, b) < 0:
                c += 1
    return c


def area2(Q):
    return 0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))


def lift(S, x, n):
    t = 0.0
    for _ in range(40):
        y = x + t * n
        f = S.f(y)
        if abs(f) < 1e-15:
            break
        d = S.g(y) @ n
        if abs(d) < 1e-14:
            return None
        t -= f / d
    y = x + t * n
    return y if abs(S.f(y)) / (np.linalg.norm(S.g(y)) + 1e-300) < 1e-11 and abs(t) < 0.2 else None


def triangulate(S, loops, planar):
    """Triangles inside the loops (largest the outer), laid flat across a normal and lifted onto S."""
    # (no point twice: triangle fails on repeated vertices)
    clean = []
    for L in loops:
        L = np.asarray(L, float)
        keep = [0] + [i for i in range(1, len(L)) if np.linalg.norm(L[i] - L[i - 1]) > 1e-10]
        L = L[keep]
        if len(L) > 1 and np.linalg.norm(L[-1] - L[0]) <= 1e-10:
            L = L[:-1]
        if len(L) >= 3:
            clean.append(L)
    loops = clean
    if not loops:
        return None
    P = np.vstack(loops)
    if len(np.unique(np.round(P, 11), axis=0)) != len(P):
        return None
    tries = [np.mean([normal(S, x) for x in P[:: max(1, len(P) // 40)]], 0)]
    tries += [normal(S, x) for x in P[:: max(1, len(P) // 6)]]
    Z = (P - P.mean(0)) @ BZ.T
    tries.append(np.linalg.svd(Z, full_matrices=False)[2][-1] @ BZ)
    for n in tries:
        n = n - n.mean()
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        e = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
        c = P.mean(0)
        flat = [np.c_[(L - c) @ e[0], (L - c) @ e[1]] for L in loops]
        if any(crossings(Q) for Q in flat):
            continue
        order = np.argsort([-abs(area2(Q)) for Q in flat])
        loops_o = [loops[i] for i in order]
        flat = [flat[i] for i in order]
        verts, segs, holes, off = [], [], [], 0
        for r, Q in enumerate(flat):
            m = len(Q)
            verts.append(Q)
            segs += [[off + i, off + (i + 1) % m] for i in range(m)]
            off += m
            if r:
                t0 = tr.triangulate({"vertices": Q, "segments": [[i, (i + 1) % m] for i in range(m)]}, "p")
                holes.append(Q[t0["triangles"][0]].mean(0))
        A = {"vertices": np.ascontiguousarray(np.vstack(verts), dtype=np.float64),
             "segments": np.ascontiguousarray(np.array(segs, np.int32))}
        if len(np.unique(np.round(A["vertices"], 12), axis=0)) != len(A["vertices"]):
            continue
        if holes:
            A["holes"] = np.ascontiguousarray(np.array(holes), dtype=np.float64)
        # (a flat piece needs no inner points: its loop's own)
        for opts in ((f"pYq20a{MAX_AREA:.3g}", f"pYq25a{MAX_AREA * 4:.3g}") if not planar else ()) + ("pY",):
            try:
                T = tr.triangulate(A, opts)
            except Exception:
                continue
            if "triangles" not in T or not len(T["triangles"]):
                continue
            nb = off
            Xv = np.empty((len(T["vertices"]), 4))
            Xv[:nb] = np.vstack(loops_o)
            ok = True
            for i in range(nb, len(Xv)):
                u, v = T["vertices"][i]
                y = c + u * e[0] + v * e[1]
                y = y if planar else lift(S, y, n)
                if y is None:
                    ok = False
                    break
                Xv[i] = y
            if not ok:
                continue
            Fa = np.asarray(T["triangles"])
            # (every triangle facing the way the surface does at it)
            cr = np.array([np.cross(*(((Xv[t[1]] - Xv[t[0]]) @ BZ.T, (Xv[t[2]] - Xv[t[0]]) @ BZ.T))) for t in Fa])
            sn = np.array([normal(S, Xv[t].mean(0)) @ BZ.T for t in Fa])
            dots = np.einsum("ij,ij->i", cr, sn)
            if (dots > 0).all() or (dots < 0).all():
                return Xv, Fa.tolist()
    return None


def groups(S, loops):
    """The loops as outer loops each with the holes inside it (one group when there is one loop), by a flat view
    across the mean normal."""
    if len(loops) == 1:
        return [loops]
    P = np.vstack(loops)
    n = np.mean([normal(S, x) for x in P[:: max(1, len(P) // 40)]], 0)
    n = n - n.mean()
    n /= np.linalg.norm(n) + 1e-300
    e = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
    flat = [np.c_[(np.asarray(L) - P.mean(0)) @ e[0], (np.asarray(L) - P.mean(0)) @ e[1]] for L in loops]

    def inside(q, Q):
        x, y = q
        c = False
        for i in range(len(Q)):
            (x1, y1), (x2, y2) = Q[i], Q[i - 1]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1:
                c = not c
        return c
    depth = [sum(inside(flat[i][0], flat[j]) for j in range(len(loops)) if j != i) for i in range(len(loops))]
    out = []
    for i in range(len(loops)):
        if depth[i] % 2 == 0:
            holes = [loops[j] for j in range(len(loops)) if depth[j] == depth[i] + 1 and inside(flat[j][0], flat[i])]
            out.append([loops[i]] + holes)
    return out


def rework(group, log=print):
    X = pickle.load(open(os.path.join(HERE, f"a4_exact_{group}.pkl"), "rb"))
    surfs = [surf_of(n) for n in X["surfaces"]]
    log(f"{group}: {len(X['faces'])} pieces, {len(X['segments'])} segments")
    extend_lines(X, surfs, log)
    by_surf = defaultdict(list)
    for si, s in enumerate(X["segments"]):
        for k in s["on"]:
            by_surf[k].append(si)
    h = 1.6 / E.GRID if hasattr(E, "GRID") else 0.035
    kept, done, how = Counter(), 0, Counter()
    Pts = [np.asarray(p, float) for p in X["points"]]
    for F in X["faces"]:
        S = surfs[F["surface"]]
        L = loops_of(F, X, by_surf, h)
        if L is not None:
            L = [thin(Q, 0.25 * np.sqrt(MAX_AREA)) for Q in L]
            how["exact segments"] += 1
        else:
            R_ = rim_loops(F, S, Pts, h)
            if R_ is None:
                kept["rim not a loop"] += 1
                continue
            L = [thin(Q, 0.5 * np.sqrt(MAX_AREA), fx) for Q, fx in R_]
            how["mesh rim, exact corners"] += 1
        planar = F["surface"] < 4 or S.expr.as_poly().total_degree() == 1
        res = None
        for G in groups(S, L):
            r_ = triangulate(S, G, planar)
            if r_ is None:
                res = None
                break
            if res is None:
                res = (list(r_[0]), list(r_[1]))
            else:
                o = len(res[0])
                res[0].extend(r_[0])
                res[1].extend([[a + o, b + o, c + o] for a, b, c in r_[1]])
        if res is None:
            kept["not triangulated"] += 1
            continue
        V, T = res
        F["mesh"] = {"verts": [np.asarray(v) for v in V], "faces": T}
        F["filled"] = []
        F["exact_loop"] = True
        done += 1
    log(f"  {done} pieces meshed inside their loops ({dict(how)}); kept their marching mesh: {dict(kept)}")
    pickle.dump(X, open(os.path.join(HERE, f"a4_pieces_{group}.pkl"), "wb"))
    return X


if __name__ == "__main__":
    for g in sys.argv[1:] or ["iodecaic", "kidecaic"]:
        rework(g)
