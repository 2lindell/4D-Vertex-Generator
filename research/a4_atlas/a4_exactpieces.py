"""The viewer's pieces from the exact map (a4_exactmap) and its walls as whole pieces (a4_walls ... exact), in the
form a4_snub.build reads (a4_pieces_<group>_exact.pkl):

  * each wall piece off the mirrors: every face of it triangulated inside its own exact loop (a4_faces.tri_face: no
    point added on the rim, inner points put on the surface), the faces' meshes joined (a face's rim points are its
    arcs' own, so neighbouring faces meet on the same points); its shape the piece's; the regions each side of it
    read a step (1e-4) off the surface both ways at its deepest point;
  * each face on a mirror: meshed the same way, with the region on the chamber's side (for the regions' shells);
  * every arc: its shape read at its middle (a line shape, or an edge where walls meet);
  * the shapes, the exact points and the surfaces as a4_exact gave them.

    python a4_exactpieces.py kidecaic
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_faces as A  # noqa: E402
import a4_pieces as P  # noqa: E402
import a4_snub as SN  # noqa: E402

X = SURFS = None


ARC = {}
TOL = float(os.environ.get("A4_ARC_TOL", "1e-5"))   # (how near the kept chain stays to the traced one)


def arc_pts(i):
    """Arc i's points for meshing, the same for every face it bounds: its traced chain thinned (Douglas-Peucker) so
    that the chain kept is within 1e-6 of the traced one and no two kept points are over 6e-3 apart. Where two arcs
    run close (near a point where many meet) the points they need are kept; neighbouring faces share them exactly."""
    if i not in ARC:
        Q = np.asarray(X["segments"][i]["pts"], float)
        keep = np.zeros(len(Q), bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(Q) - 1)]
        while stack:
            a, b = stack.pop()
            if b - a < 2:
                continue
            d = Q[a + 1:b] - Q[a]
            u = Q[b] - Q[a]
            uu = u @ u + 1e-300
            t = np.clip(d @ u / uu, 0, 1)
            dev = np.linalg.norm(d - t[:, None] * u, axis=1)
            k = int(np.argmax(dev))
            if dev[k] > TOL or np.linalg.norm(u) > 6e-3:
                m = a + 1 + k if dev[k] > TOL else (a + b) // 2
                keep[m] = True
                stack += [(a, m), (m, b)]
        ARC[i] = Q[keep]
    return ARC[i]


def face_loop(F):
    pts = []
    for i, fw in F["cycle"]:
        Q = arc_pts(i)
        pts.extend((Q if fw else Q[::-1])[:-1])
    L = [pts[0]]
    for q in pts[1:]:
        if np.linalg.norm(q - L[-1]) > 1e-12:
            L.append(q)
    if np.linalg.norm(L[-1] - L[0]) <= 1e-12:
        L.pop()
    return np.array(L)


def mesh_face(fi):
    F = X["faces"][fi]
    k = F["surface"]
    S = SURFS[k]
    planar = k < 4 or S.expr.as_poly().total_degree() == 1
    r = A.tri_face(S, [face_loop(F)], planar)
    if r is None:
        r = graph_mesh(S, cusps_cut(S, F), planar)
    return fi, (None if r is None else {"verts": np.asarray(r[0], float), "faces": np.asarray(r[1], int)})


def graph_mesh(S, L, planar):
    """A face that is a graph over the plane across its loop's mean normal (no two of its points on one line along
    that normal): its loop laid flat and triangulated there (no point added on the rim), each inner point moved onto
    the surface along that one normal. Laid flat, the triangles cannot overlap, and lifted along one direction they
    stay so. Used where the surface's own normal is no guide: at a singular point of the surface, a corner of the
    face, where it turns every way. Checked: every triangle faces the same way along the normal used."""
    import triangle as tr
    n = len(L)
    nm = np.mean([P.normal(S, x) for x in L[:: max(1, n // 60)]], 0)
    nm = nm - nm.mean()
    nm /= np.linalg.norm(nm) + 1e-300
    e = np.linalg.svd(np.vstack([np.ones(4) / 2, nm]))[2][2:]
    c = L.mean(0)
    Q = np.c_[(L - c) @ e[0], (L - c) @ e[1]]
    # (triangle fails, or crashes, on two equal points: none closer than 1e-11 laid flat)
    keep = [0]
    for i in range(1, n):
        if np.linalg.norm(Q[i] - Q[keep[-1]]) > 1e-11:
            keep.append(i)
    if np.linalg.norm(Q[keep[-1]] - Q[keep[0]]) <= 1e-11:
        keep.pop()
    L, Q, n = L[keep], Q[keep], len(keep)
    if n < 3 or len(np.unique(np.round(Q, 11), axis=0)) != n or P.crossings(Q):
        return None
    try:
        T = tr.triangulate({"vertices": np.ascontiguousarray(Q), "segments": [[i, (i + 1) % n] for i in range(n)]},
                           f"pYq20a{P.MAX_AREA:.3g}")
    except Exception:  # noqa: BLE001
        return None
    V = np.empty((len(T["vertices"]), 4))
    V[:n] = L
    for i in range(n, len(V)):
        u, v = T["vertices"][i]
        y = c + u * e[0] + v * e[1]
        y = y if planar else P.lift(S, y, nm)
        if y is None:
            return None
        V[i] = y
    Fa = np.asarray(T["triangles"])
    cr = np.cross((V[Fa[:, 1]] - V[Fa[:, 0]]) @ A.BZ.T, (V[Fa[:, 2]] - V[Fa[:, 0]]) @ A.BZ.T)
    d = cr @ (nm @ A.BZ.T)
    if not ((d > 0).all() or (d < 0).all()):
        return None
    return V, Fa.tolist()


def split_mesh(S, L, planar, depth=3):
    """A loop too curved for one flat layout: meshed whole if it can be, else cut in two along a chord between two of
    its points about half its length apart (the chord's points put on the surface, shared by both halves) and each
    half meshed so, down to depth cuts."""
    r = A.tri_face(S, [L], planar)
    if r is not None or depth == 0:
        return r
    seg = np.linalg.norm(np.diff(np.vstack([L, L[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = len(L)
    # (the chord must lie inside the face: seen flat across the loop's mean normal, it crosses no edge of the loop and
    #  its middle is inside; a straight chord across a face that is not convex can leave it)
    nm = np.mean([P.normal(S, x) for x in L[:: max(1, n // 40)]], 0)
    nm = nm - nm.mean()
    nm /= np.linalg.norm(nm) + 1e-300
    e = np.linalg.svd(np.vstack([np.ones(4) / 2, nm]))[2][2:]
    F2 = np.c_[L @ e[0], L @ e[1]]

    def inside_chord(i, j):
        a, b = F2[i], F2[j]
        for k in range(n):
            if k in (i, j) or (k + 1) % n in (i, j):
                continue
            p, q = F2[k], F2[(k + 1) % n]
            d1 = np.cross(b - a, p - a)
            d2 = np.cross(b - a, q - a)
            d3 = np.cross(q - p, a - p)
            d4 = np.cross(q - p, b - p)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return False
        m = (a + b) / 2
        ang = np.arctan2(F2[:, 1] - m[1], F2[:, 0] - m[0])
        w = np.sum(np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))) / (2 * np.pi)
        return abs(round(w)) == 1
    for shift in np.arange(0, 0.5, 1 / 32):
        for half in (0.5, 0.4, 0.6, 0.3, 0.7):
            i = int(np.searchsorted(cum, shift * cum[-1])) % n
            j = int(np.searchsorted(cum, ((shift + half) % 1) * cum[-1])) % n
            if abs(i - j) >= 3 and inside_chord(i, j):
                break
        else:
            continue
        a, b = L[i], L[j]
        m = max(3, int(np.linalg.norm(b - a) / 4e-3))
        ch = []
        for t in np.linspace(0, 1, m + 1)[1:-1]:
            y = a + t * (b - a)
            y = y if planar else P.lift_normal(S, y)
            if y is None:
                break
            ch.append(y)
        else:
            i, j = min(i, j), max(i, j)
            ch = np.array(ch)
            if np.linalg.norm(ch[0] - L[i]) > np.linalg.norm(ch[0] - L[j]):
                ch = ch[::-1]
            h1 = np.vstack([L[i:j + 1], ch[::-1]])
            h2 = np.vstack([L[j:], L[:i + 1], ch])
            r1 = split_mesh(S, h1, planar, depth - 1)
            r2 = split_mesh(S, h2, planar, depth - 1)
            if r1 is not None and r2 is not None:
                V = np.vstack([r1[0], r2[0]])
                T = np.vstack([np.asarray(r1[1]), np.asarray(r2[1]) + len(r1[0])])
                return V, T.tolist()
    return None


def cusps_cut(S, F):
    """The face's outline with its cusps cut: where the two arcs into a corner leave it tangent, their traced chains
    (each exact to 1e-12, but sampled at different points) weave across each other within a little of the corner;
    both are cut back to 1.5 times the farthest crossing from the corner and joined there. Only the mesh loses that
    sliver (its area of the order of the cube of that distance)."""
    chains = []
    for i, fw in F["cycle"]:
        Q = arc_pts(i)
        chains.append(Q if fw else Q[::-1])
    allp = np.vstack(chains)
    n = np.mean([P.normal(S, x) for x in allp[:: max(1, len(allp) // 40)]], 0)
    n = n - n.mean()
    n /= np.linalg.norm(n) + 1e-300
    e = np.linalg.svd(np.vstack([np.ones(4) / 2, n]))[2][2:]
    flat = lambda Q: np.c_[Q @ e[0], Q @ e[1]]  # noqa: E731

    def crossings(A_, B_):
        out = []
        a, b = flat(A_), flat(B_)
        for m in range(len(a) - 1):
            p1, p2 = a[m], a[m + 1]
            for k in range(len(b) - 1):
                q1, q2 = b[k], b[k + 1]
                d = (p2 - p1)[0] * (q2 - q1)[1] - (p2 - p1)[1] * (q2 - q1)[0]
                if abs(d) < 1e-30:
                    continue
                w = q1 - p1
                t = (w[0] * (q2 - q1)[1] - w[1] * (q2 - q1)[0]) / d
                u = (w[0] * (p2 - p1)[1] - w[1] * (p2 - p1)[0]) / d
                if 1e-12 < t < 1 - 1e-12 and 1e-12 < u < 1 - 1e-12:
                    out.append(A_[m] + t * (A_[m + 1] - A_[m]))
        return out
    k = len(chains)
    for j in range(k):
        cin, cout = chains[j], chains[(j + 1) % k]
        v = cout[0]
        near = lambda Q: Q[np.linalg.norm(Q - v, axis=1) < 0.05]  # noqa: E731
        xs = crossings(near(cin[::-1])[::-1], near(cout))
        if not xs:
            continue
        r = 1.5 * max(np.linalg.norm(x - v) for x in xs)
        chains[j] = cin[np.linalg.norm(cin - v, axis=1) >= r]
        chains[(j + 1) % k] = cout[np.linalg.norm(cout - v, axis=1) >= r]
    L = []
    for Q in chains:
        for q in Q:
            if not L or np.linalg.norm(q - L[-1]) > 1e-9:
                L.append(q)
    return np.array(L)


def main(group):
    global X, SURFS
    X = pickle.load(open(os.path.join(HERE, f"a4_exactmap_{group}.pkl"), "rb"))
    pieces = pickle.load(open(os.path.join(HERE, f"a4_walls_exact_{group}.pkl"), "rb"))
    old = pickle.load(open(os.path.join(HERE, f"a4_exact_{group}.pkl"), "rb"))
    SURFS = [P.surf_of(n) for n in X["surfaces"]]
    sub = SN.SUB[group]
    t0 = time.time()
    want = [p for p in pieces if p["wall"]]
    todo = sorted({fi for p in want for fi in p["faces"]})
    meshes = {}
    with Pool(4) as pool:
        for fi, m in pool.imap_unordered(mesh_face, todo, chunksize=8):
            meshes[fi] = m
    failed = [fi for fi in todo if meshes.get(fi) is None]
    print(f"  {len(todo)} faces meshed, {len(failed)} failed ({time.time() - t0:.0f}s)", flush=True)
    # (the regions each side of every wall piece, a step off its surface at its deepest point; a mirror piece's
    #  region on the chamber's side)
    probes, owner = [], []
    for pi, p in enumerate(want):
        x = np.asarray(p["deep"], float)
        k = p["surface"]
        if k < 4:
            y = x.copy()
            y[k] += 1e-4
            probes.append(y / y.sum())
            owner.append((pi, 0))
            continue
        n = P.normal(SURFS[k], x)
        for s_, side in ((1, 0), (-1, 1)):
            y = np.clip(x + s_ * 1e-4 * n, 0, None)
            probes.append(y / y.sum())
            owner.append((pi, side))
    labs = SN.classify_many(probes, sub, tol=1e-7)
    sides = {}
    for (pi, side), (_, kk) in zip(owner, labs):
        sides.setdefault(pi, [None, None])[side] = kk
    faces = []
    for pi, p in enumerate(want):
        V, T, off = [], [], 0
        for fi in p["faces"]:
            m = meshes.get(fi)
            if m is None:
                continue
            V.append(m["verts"])
            T.append(m["faces"] + off)
            off += len(m["verts"])
        if not V:
            continue
        faces.append({"surface": p["surface"], "key": p["key"], "sides": sides.get(pi, [None, None]),
                      "inner": np.asarray(p["deep"], float), "exact_loop": True,
                      "mesh": {"verts": [list(map(float, v)) for v in np.vstack(V)],
                               "faces": np.vstack(T).tolist()}})
    # (every arc's shape, at its middle)
    segs = X["segments"]
    mids = [np.asarray(s["pts"], float)[len(s["pts"]) // 2] for s in segs]
    for s, (_, kk) in zip(segs, SN.classify_many(mids, sub, tol=1e-7)):
        s["key"] = kk
        s["pts"] = [np.asarray(q, float) for q in s["pts"]]
    # (the edges drawn: where walls hit each other, the edges of the wall pieces off the mirrors, and on the mirrors
    #  the curves between two of their pieces)
    edge = {i for p in pieces if p["wall"] and p["surface"] >= 4 for i, _ in p["edges"]}
    side_m = {}
    for pi, p in enumerate(pieces):
        if p["surface"] < 4:
            for i, _ in p["edges"]:
                side_m.setdefault((p["surface"], i), set()).add(pi)
    edge |= {i for (k, i), ps in side_m.items() if len(ps) > 1}
    out = {"shapes": old["shapes"], "surfaces": X["surfaces"], "points": X["points"],
           "segments": [{"key": s["key"], "pts": s["pts"], "ends": s["ends"], "on": s["on"], "edge": i in edge}
                        for i, s in enumerate(segs)],
           "faces": faces}
    pickle.dump(out, open(os.path.join(HERE, f"a4_pieces_{group}_exact.pkl"), "wb"))
    print(f"{len(faces)} pieces meshed ({sum(f['surface'] >= 4 for f in faces)} walls off the mirrors); "
          f"faces that failed to mesh: {failed[:20]}")


if __name__ == "__main__":
    for g in sys.argv[1:]:
        main(g)
