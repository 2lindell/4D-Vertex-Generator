"""Wall pieces from the exact curve graph, as the swirlprism atlas builds them (sphere_map/domain_pieces.py): on each
surface the exact segments of a4_exact (the curves where it meets the other surfaces and the mirrors, cut at the exact
points) form a graph; going round each point in the surface's tangent plane gives its faces, each bounded by exact
loops, so every corner is an exact point and every edge an exact curve. Each face is triangulated inside its loops
(laid flat across the surface's normal, quality triangles, every inner vertex solved onto the surface).

A face takes the place of the marching piece it lies in (a4_exact's pieces carry the classification: shape, sides);
a face in no wall piece is dropped. A piece whose faces do not cover it (area off by more than AREA_TOL) keeps its
marching mesh.

    python a4_faces.py iodecaic          (reads a4_pieces_<group>.pkl, writes it back)
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_pieces as P  # noqa: E402
import a4_exact as E  # noqa: E402

BZ = P.BZ
P.MAX_AREA = float(os.environ.get("A4_FACE_AREA", "6e-4"))   # (about the marching grid's triangles)
AREA_TOL = 0.08


def frame_at(S, x):
    """An orthonormal frame (e1, e2) of the surface's tangent plane at x (in the 3-space sum(beta) = 0), turning the
    way its gradient does."""
    n = P.normal(S, x) @ BZ.T
    n /= np.linalg.norm(n)
    a = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def leave_dir(Q, d=0.02):
    """The direction a polyline leaves its first point: the point a little way along, from the start."""
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    d = min(d, cum[-1] / 3)
    i = int(np.searchsorted(cum, d))
    i = min(max(i, 1), len(Q) - 1)
    return Q[i] - Q[0]


def thin_seg(Q):
    """A curve's points about half a triangle's side apart, both its ends (exact points) kept."""
    step = 0.5 * np.sqrt(P.MAX_AREA)
    keep = [0]
    for i in range(1, len(Q) - 1):
        if np.linalg.norm(Q[i] - Q[keep[-1]]) >= step and np.linalg.norm(Q[-1] - Q[i]) >= 0.5 * step:
            keep.append(i)
    keep.append(len(Q) - 1)
    return Q[keep]


def faces_on(k, S, segs, Pts):
    """The faces of the curve graph on surface k: (loops [outer, holes...], half-edge cycles) each."""
    own = [i for i, s in enumerate(segs) if k in s["on"]]
    if not own:
        return []
    # (one copy of each curve: two segments with the same ends running together are one)
    uniq = []
    for i in own:
        s = segs[i]
        Q = np.asarray(s["pts"], float)
        dup = False
        for j in uniq:
            t = segs[j]
            if set(t["ends"]) == set(s["ends"]):
                R = np.asarray(t["pts"], float)
                if np.min(np.linalg.norm(R - Q[len(Q) // 2], axis=1)) < 1e-7:
                    dup = True
                    break
        if not dup:
            uniq.append(i)
    out = defaultdict(list)
    for i in uniq:
        Q = np.asarray(segs[i]["pts"], float)
        a, b = segs[i]["ends"]
        out[a].append((i, True, Q))
        out[b].append((i, False, Q[::-1]))
    ang = {}
    for v, hs in out.items():
        x = np.asarray(Pts[v], float)
        e1, e2 = frame_at(S, x)
        for i, fw, Q in hs:
            d = leave_dir(Q) @ BZ.T
            ang[(i, fw)] = np.arctan2(d @ e2, d @ e1)
        hs.sort(key=lambda h: ang[(h[0], h[1])])
    pos = {v: {(h[0], h[1]): j for j, h in enumerate(hs)} for v, hs in out.items()}

    def head(i, fw):
        a, b = segs[i]["ends"]
        return b if fw else a
    seen, cycles, hcyc = set(), [], []
    for v, hs in out.items():
        for h in hs:
            he = (h[0], h[1])
            if he in seen:
                continue
            cyc, cur = [], he
            for _ in range(10 * len(uniq) + 10):
                if cur in seen:
                    break
                seen.add(cur)
                cyc.append(cur)
                w = head(*cur)
                j = pos[w][(cur[0], not cur[1])]
                nxt = out[w][(j - 1) % len(out[w])]
                cur = (nxt[0], nxt[1])
            if cur != he:
                continue
            pts = []
            for i, fw in cyc:
                Q = thin_seg(np.asarray(segs[i]["pts"], float))
                Q = Q if fw else Q[::-1]
                pts.extend(Q[:-1])
            cycles.append(np.array(pts))
            hcyc.append(cyc)

    # (the graph on each surface is connected, so every cycle is one face, on its left; the cycle round the outside
    #  has the outside of the chamber on its left)
    res = []
    for L, cyc in zip(cycles, hcyc):
        mir = [sorted(set(segs[i]["on"]) & ({0, 1, 2, 3} - {k})) for i, _ in cyc]
        if not all(mir):                         # (a curve inside the chamber in it: an inside face)
            res.append(([L], [cyc]))
            continue
        keep = True                              # (all on the chamber's faces: inside, or the cycle round the outside)
        for (i, fw), ms in zip(cyc, mir):
            if len(ms) != 1:
                continue
            Q = np.asarray(segs[i]["pts"], float)
            Q = Q if fw else Q[::-1]
            m = len(Q) // 2
            x, d = Q[m], Q[min(m + 1, len(Q) - 1)] - Q[max(m - 1, 0)]
            n = P.normal(S, x) @ BZ.T
            side = np.cross(n, d @ BZ.T) @ BZ
            side /= np.linalg.norm(side) + 1e-300
            y = P.lift_normal(S, x + 1e-4 * side)  # (the walk keeps the face on its left)
            keep = y is not None and y[ms[0]] > 0
            break
        if keep:
            res.append(([L], [cyc]))
    out = []                                     # (a slit walked both ways joins two faces: each loop its own face)
    for r in res:
        loops, cyc = unbridge(segs, r[1][0])
        out += [([L], [c]) for L, c in zip(loops, cyc)]
    return out


def unbridge(segs, cyc):
    """A cycle that walks a curve both ways (a bridge: the face on both sides of it) split there into the loops it
    joins: (loops, [cycles])."""
    stack, done = [list(cyc)], []
    while stack:
        c = stack.pop()
        pos = {}
        for a, (i, fw) in enumerate(c):
            if i in pos:
                b = pos[i]
                inner, outer = c[b + 1:a], c[:b] + c[a + 1:]
                stack += [x for x in (inner, outer) if x]
                break
            pos[i] = a
        else:
            done.append(c)
    loops = []
    for c in done:
        pts = []
        for i, fw in c:
            Q = thin_seg(np.asarray(segs[i]["pts"], float))
            Q = Q if fw else Q[::-1]
            pts.extend(Q[:-1])
        if len(pts) >= 3:
            loops.append(np.array(pts))
    return loops, done


def cut(k, S, segs, Pts, surfs, hcycles, step=3e-3):
    """A face too curved to lay flat, cut across by the curve where its surface meets a plane (traced exactly, its
    ends solved on the face's edges): the face's edges split there, the cut added as curves of the graph."""
    L = np.vstack([np.asarray(segs[i]["pts"], float) for cyc in hcycles for i, _ in cyc])
    c0 = L.mean(0)
    axes = np.linalg.svd(L - c0, full_matrices=False)[2][:2]
    for w in axes:                               # (planes across the face's longest and next longest ways, through
        w = w - w.mean()                         #  its middle or a little to either side)
        w /= np.linalg.norm(w)
        v = L @ w
        for c in (np.median(v), np.quantile(v, 0.4), np.quantile(v, 0.6), np.quantile(v, 0.3), np.quantile(v, 0.7)):
            n0, p0 = len(segs), len(Pts)
            saved = [dict(x) for x in segs]
            if _cut_with(k, S, segs, Pts, surfs, hcycles, w, c, step):
                return True
            segs[:] = saved                      # (nothing traced: the edges left as they were)
            del Pts[p0:]
    return False


def _cut_with(k, S, segs, Pts, surfs, hcycles, w, c, step):
    if True:
        pass
    plane = E.Surf(sum((float(w[i]) - c) * E.B[i] for i in range(4)), "cut")
    # crossings of the face's edges with the plane
    cross = []                                   # (segment, index along it, point, direction along the walk)
    for cyc in hcycles:
        for i, fw in cyc:
            Q = np.asarray(segs[i]["pts"], float)
            v = Q @ w - c
            for j in np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0):
                x0 = Q[j] + (Q[j + 1] - Q[j]) * v[j] / (v[j] - v[j + 1])
                other = [surfs[m] for m in sorted(segs[i]["on"]) if m != k][:1]
                x, ok = E.newton(x0, [S] + other + [plane])
                if not ok or np.linalg.norm(x - x0) > 1e-3:
                    x, ok = E.newton(x0, [S, plane])
                if not ok:
                    continue
                d = (Q[j + 1] - Q[j]) * (1 if fw else -1)
                cross.append((i, j, x, d))
    uniq = []
    for cr in cross:                             # (an edge walked both ways crosses once)
        if not any(np.linalg.norm(cr[2] - u[2]) < 1e-9 for u in uniq):
            uniq.append(cr)
    cross = uniq
    if len(cross) < 2:
        return False
    # (split the edges at the crossings: new exact points)
    ids = []
    for i, j, x, d in cross:
        Pts.append(x)
        ids.append(len(Pts) - 1)
    by_seg = defaultdict(list)
    for (i, j, x, d), pid in zip(cross, ids):
        by_seg[i].append((j, x, pid))
    for i, cs in by_seg.items():
        s = segs[i]
        Q = list(np.asarray(s["pts"], float))
        cs.sort(key=lambda t: t[0])
        parts, start, a = [], 0, s["ends"][0]
        for j, x, pid in cs:
            parts.append((a, pid, Q[start:j + 1] + [x]))
            Q[j] = Q[j]
            start = j + 1
            a = pid
            Q_ = Q
        parts.append((a, s["ends"][1], [cs[-1][1]] + Q[start:]))
        for n_, (a_, b_, pts) in enumerate(parts):
            seg = dict(s, ends=(a_, b_), pts=np.array(pts))
            if n_ == 0:
                segs[i] = seg
            else:
                segs.append(seg)
    # (trace inward from each crossing to the next it reaches)
    used = set()
    pos = np.array([x for _, _, x, _ in cross])
    for q, (i, j, x, d) in enumerate(cross):
        if q in used:
            continue
        t0 = E.tangent(x, S, plane)
        end = None
        for sgn in (1, -1):                  # (the way into the face: the way that stays in the chamber and reaches
            t, pts, y = sgn * t0, [x], x.copy()          # another crossing)
            for _ in range(4000):
                z, ok = E.newton(y + step * t, [S, plane])
                if not ok or z.min() < -1e-9:
                    break
                t2 = E.tangent(z, S, plane)
                t = t2 if t2 @ t > 0 else -t2
                y = z
                pts.append(y)
                dd = np.linalg.norm(pos - y, axis=1)
                dd[q] = np.inf
                r = int(np.argmin(dd))
                if dd[r] < 3 * step and len(pts) > 2:
                    end = r
                    break
            if end is not None:
                break
        if end is None or end in used:
            continue
        pts[-1] = pos[end]
        used |= {q, end}
        segs.append({"ends": (ids[q], ids[end]), "pts": np.array(pts), "on": {k}, "key": None, "sig": None})
    return bool(used)


def inside(q, Q):
    x, y = q
    c = False
    for i in range(len(Q)):
        (x1, y1), (x2, y2) = Q[i], Q[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1:
            c = not c
    return c


def mesh_area(V, T):
    V = np.asarray(V, float)
    T = np.asarray(T, int)
    if not len(T):
        return 0.0
    e1 = (V[T[:, 1]] - V[T[:, 0]]) @ BZ.T
    e2 = (V[T[:, 2]] - V[T[:, 0]]) @ BZ.T
    return float(0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum())


def disk_layout(F, S):
    """The marching piece laid on the unit disk (Tutte: its rim round the circle by length, each inner vertex the
    mean of its neighbours): (V, T, uv), or None when the piece is not a disk."""
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    G = {"surface": F["surface"], "mesh": F["mesh"], "filled": F.get("filled", [])}
    P.weld(G)
    R = P.rim_loops(G, S, [], 1.0, ids=True)
    if R is None or len(R) != 1:
        return None
    loop = [int(v) for v in R[0][0]]
    V = np.asarray(G["mesh"]["verts"], float)
    fl = set(G.get("filled", []))
    T = np.array([t for t in G["mesh"]["faces"] if not fl.intersection(t)], int)
    seg = [np.linalg.norm(V[loop[(i + 1) % len(loop)]] - V[loop[i]]) for i in range(len(loop))]
    ang = np.concatenate([[0], np.cumsum(seg)[:-1]]) / sum(seg) * 2 * np.pi
    bnd = {v: (np.cos(a), np.sin(a)) for v, a in zip(loop, ang)}
    used = sorted({int(a) for t in T for a in t})
    inner = [v for v in used if v not in bnd]
    nb = defaultdict(set)
    for t in T:
        for a in t:
            nb[int(a)].update(int(b) for b in t if b != a)
    kx = {v: j for j, v in enumerate(inner)}
    uv = np.zeros((len(V), 2))
    if inner:
        A = lil_matrix((len(inner), len(inner)))
        rhs = np.zeros((len(inner), 2))
        for v in inner:
            A[kx[v], kx[v]] = len(nb[v])
            for w in nb[v]:
                if w in kx:
                    A[kx[v], kx[w]] -= 1
                elif w in bnd:
                    rhs[kx[v]] += bnd[w]
        sol = spsolve(A.tocsr(), rhs).reshape(-1, 2)
        for v, j in kx.items():
            uv[v] = sol[j]
    for v, q in bnd.items():
        uv[v] = q
    return V, T, uv


def onto_mesh(V, T, x):
    """The point of the mesh nearest x: (triangle, barycentric coordinates)."""
    best = None
    X3 = V @ BZ.T
    y = x @ BZ.T
    for j in range(len(T)):
        a, b, c = X3[T[j]]
        # (closest point on a triangle: clamped barycentric, refined over its edges)
        e0, e1, d = b - a, c - a, y - a
        M = np.array([[e0 @ e0, e0 @ e1], [e0 @ e1, e1 @ e1]])
        try:
            s_, t_ = np.linalg.solve(M, [e0 @ d, e1 @ d])
        except np.linalg.LinAlgError:
            continue
        lam = np.array([1 - s_ - t_, s_, t_])
        if lam.min() < 0:
            lam = np.clip(lam, 0, None)
            lam /= lam.sum()
        dist = np.linalg.norm(lam @ X3[T[j]] - y)
        if best is None or dist < best[0]:
            best = (dist, j, lam)
    return best[1], best[2]


def tutte_exact(F, S, loop, planar):
    """A face too curved to lay flat, on its marching piece laid on a disk: its exact loop put on the disk through
    the nearest points of the piece, triangulated there (refined by area on the surface), the inner vertices carried
    back through the piece and solved onto the surface; the loop's own points kept exact."""
    D = disk_layout(F, S)
    if D is None:
        return None
    V, T, uv = D
    L = P.thin(np.asarray(loop, float), 0.5 * np.sqrt(P.MAX_AREA))
    cen = cKDTree((V[T].mean(1)) @ BZ.T)
    Lu = []
    for x in L:
        js = cen.query(x @ BZ.T, k=min(16, len(T)))[1]
        js = np.atleast_1d(js)
        j, lam = onto_mesh(V, T[js], x)
        Lu.append(lam @ uv[T[js][j]])
    Lu = np.array(Lu)
    keep = [0] + [i for i in range(1, len(Lu)) if np.linalg.norm(Lu[i] - Lu[i - 1]) > 1e-9]
    L, Lu = L[keep], Lu[keep]
    if len(L) < 3 or P.crossings(Lu):
        return None
    if P.area2(Lu) < 0:
        L, Lu = L[::-1], Lu[::-1]
    m = len(Lu)
    a_, b_, c_ = uv[T][:, 0], uv[T][:, 1], uv[T][:, 2]
    det = (b_[:, 1] - c_[:, 1]) * (a_[:, 0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (a_[:, 1] - c_[:, 1])
    ok = np.abs(det) > 1e-300
    det = np.where(ok, det, 1.0)

    def to3(q):
        l1 = ((b_[:, 1] - c_[:, 1]) * (q[0] - c_[:, 0]) + (c_[:, 0] - b_[:, 0]) * (q[1] - c_[:, 1])) / det
        l2 = ((c_[:, 1] - a_[:, 1]) * (q[0] - c_[:, 0]) + (a_[:, 0] - c_[:, 0]) * (q[1] - c_[:, 1])) / det
        l3 = 1 - l1 - l2
        worst = np.where(ok, np.minimum(np.minimum(l1, l2), l3), -np.inf)
        j = int(np.argmax(worst))
        if worst[j] < -0.02:
            return None
        lam = np.clip([l1[j], l2[j], l3[j]], 0, None)
        lam /= lam.sum()
        y = lam @ V[T[j]]
        return y if planar else P.lift_normal(S, y)
    A3 = mesh_area(V, T)
    amax = P.MAX_AREA * abs(P.area2(Lu)) / max(A3, 1e-12)
    try:
        Tr = P.tr.triangulate({"vertices": np.ascontiguousarray(Lu, dtype=np.float64),
                               "segments": np.array([[i, (i + 1) % m] for i in range(m)], np.int32)},
                              f"pYq20a{amax:.3g}")
    except Exception:
        return None
    if "triangles" not in Tr or len(Tr["vertices"]) < m:
        return None
    for rnd in range(5):
        Xv = list(L)
        for q in Tr["vertices"][m:]:
            y = to3(q)
            if y is None:
                return None
            Xv.append(y)
        Xv = np.array(Xv)
        Ft = np.asarray(Tr["triangles"])
        e1 = (Xv[Ft[:, 1]] - Xv[Ft[:, 0]]) @ BZ.T
        e2 = (Xv[Ft[:, 2]] - Xv[Ft[:, 0]]) @ BZ.T
        a3 = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        if a3.max() <= 1.6 * P.MAX_AREA or rnd == 4:
            break
        U = Tr["vertices"]
        ad = 0.5 * np.abs((U[Ft[:, 1], 0] - U[Ft[:, 0], 0]) * (U[Ft[:, 2], 1] - U[Ft[:, 0], 1])
                          - (U[Ft[:, 1], 1] - U[Ft[:, 0], 1]) * (U[Ft[:, 2], 0] - U[Ft[:, 0], 0]))
        lim = np.where(a3 > P.MAX_AREA, ad * P.MAX_AREA / np.maximum(a3, 1e-300), -1.0)
        try:
            Tr = P.tr.triangulate({"vertices": U, "triangles": Ft, "segments": Tr["segments"],
                                   "triangle_max_area": lim}, "rpYq20a")
        except Exception:
            return None
    return Xv, np.asarray(Tr["triangles"])


def rework(group, log=print):
    path = os.path.join(HERE, f"a4_pieces_{group}.pkl")
    X = pickle.load(open(path, "rb"))
    surfs = [P.surf_of(n) for n in X["surfaces"]]
    X["points"] = [np.asarray(p, float) for p in X["points"]]
    Pts = X["points"]                            # (cuts add exact points to it)
    by_surf = defaultdict(list)
    for fi, F in enumerate(X["faces"]):
        by_surf[F["surface"]].append(fi)
    new = defaultdict(list)
    stats = Counter()
    for k, fis in sorted(by_surf.items()):
        S = surfs[k]
        planar = k < 4 or S.expr.as_poly().total_degree() == 1
        faces = faces_on(k, S, X["segments"], Pts)
        meshes = {}
        for rnd in range(int(os.environ.get("A4_CUT_ROUNDS", "1"))):   # (cuts: off unless asked)
            bad = []
            for q, (loops, hc) in enumerate(faces):
                r_ = P.triangulate(S, loops, planar)
                if r_ is None:
                    bad.append(hc)
                else:
                    meshes[q] = r_
            if not bad or rnd == int(os.environ.get("A4_CUT_ROUNDS", "1")) - 1:
                break
            did = [cut(k, S, X["segments"], Pts, surfs, hc) for hc in bad]
            stats["cuts"] += sum(did)
            if not any(did):
                break
            faces, meshes = faces_on(k, S, X["segments"], Pts), {}
        # (the marching pieces on this surface, to say which piece each face is)
        cent, owner = [], []
        for fi in fis:
            F = X["faces"][fi]
            V = np.asarray(F["mesh"]["verts"], float)
            T = np.asarray([t for t in F["mesh"]["faces"] if not set(F.get("filled", [])) & set(t)], int)
            if len(T):
                cent.append(V[T].mean(1))
                owner += [fi] * len(T)
        if not cent:
            continue
        tree = cKDTree(np.vstack(cent))
        owner = np.array(owner)
        for q, (loops, hc) in enumerate(faces):
            r_ = meshes.get(q)
            if r_ is None and len(loops) == 1:
                # (too curved to lay flat: on the marching piece its loop runs round, laid on a disk)
                Lp = loops[0][:: max(1, len(loops[0]) // 300)]
                best = None
                for fi in fis:
                    Vm = np.asarray(X["faces"][fi]["mesh"]["verts"], float)
                    if not len(Vm):
                        continue
                    dm = cKDTree(Vm).query(Lp)[0].mean()
                    if best is None or dm < best[0]:
                        best = (dm, fi)
                if best is not None and best[0] < 1.6 / 40:
                    tu = tutte_exact(X["faces"][best[1]], S, loops[0], planar)
                    if tu is not None:
                        new[best[1]].append(tu)
                        stats["faces laid on their piece's disk"] += 1
                        continue
            if r_ is None:
                stats["face not triangulated"] += 1
                continue
            V, T = np.asarray(r_[0]), np.asarray(r_[1])
            C = V[T].mean(1)
            d, j = tree.query(C)
            near = d < 0.5 * 1.6 / 40
            if near.mean() < 0.5:
                stats["face in no wall piece"] += 1
                continue
            fi = Counter(owner[j[near]]).most_common(1)[0][0]
            new[fi].append((V, T))
            stats["faces"] += 1
    # (a piece the walk gave no face: its graph walked again on the curves along its own rim only)
    for fi, F in enumerate(X["faces"]):
        if fi in new:
            continue
        k = F["surface"]
        S = surfs[k]
        planar = k < 4 or S.expr.as_poly().total_degree() == 1
        rim = P.rim_of(dict(F))
        if not rim:
            continue
        Vm = np.asarray(P.weld(G := {"mesh": F["mesh"], "filled": F.get("filled", [])}) or G["mesh"]["verts"], float)
        R = [Vm[a] + u * (Vm[b] - Vm[a]) for a, b in P.rim_of(G) for u in np.linspace(0, 1, 6)]
        tree = cKDTree(np.array(R))
        h = 1.6 / 40
        sub = []
        for s_ in X["segments"]:
            if k not in s_["on"]:
                continue
            Q = np.asarray(s_["pts"], float)
            d = tree.query(Q[1:-1] if len(Q) > 2 else Q)[0]
            if len(d) and np.mean(d < h) > 0.85:
                sub.append(s_)
        if not sub:
            continue
        sub_ids = [X["segments"].index(s_) for s_ in sub]
        view = [dict(s_, on=(s_["on"] if i in sub_ids else set(s_["on"]) - {k})) for i, s_ in enumerate(X["segments"])]
        old_a = mesh_area(G["mesh"]["verts"], [t for t in G["mesh"]["faces"] if not set(G.get("filled", [])) & set(t)])
        for loops, hc in faces_on(k, S, view, Pts):
            Lp = loops[0][:: max(1, len(loops[0]) // 300)]
            if cKDTree(Vm).query(Lp)[0].mean() > 0.25 * h:
                continue
            r_ = P.triangulate(S, loops, planar) or tutte_exact(F, S, loops[0], planar)
            if r_ is None:
                continue
            if abs(mesh_area(*r_) - old_a) < AREA_TOL * old_a:
                new[fi].append((np.asarray(r_[0]), np.asarray(r_[1])))
                stats["faces from the piece's own rim curves"] += 1
                break
    for fi, F in enumerate(X["faces"]):
        parts = new.get(fi)
        old_a = mesh_area(F["mesh"]["verts"], [t for t in F["mesh"]["faces"]
                                               if not set(F.get("filled", [])) & set(t)])
        if not parts:
            stats["piece kept (no exact face)"] += 1
            continue
        Vs, Ts = [], []
        for V, T in parts:
            o = len(Vs)
            Vs.extend(V)
            Ts.extend((T + o).tolist())
        a = mesh_area(Vs, Ts)
        if abs(a - old_a) > AREA_TOL * max(old_a, 1e-9):
            stats["piece kept (area off)"] += 1
            log(f"  piece {fi} ({F['key'][:6]}): exact faces {a:.5f}, marching {old_a:.5f}: kept")
            continue
        F["mesh"] = {"verts": list(Vs), "faces": Ts}
        F["filled"] = []
        F["exact_loop"] = True
        stats["piece from exact faces"] += 1
    log(f"{group}: {dict(stats)}")
    pickle.dump(X, open(path, "wb"))


if __name__ == "__main__":
    for g in sys.argv[1:] or ["iodecaic"]:
        rework(g)
