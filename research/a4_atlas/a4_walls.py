"""The walls of a curved A4 map as whole pieces, found the way the swirlprism Symmetry Domain found its walls
(swirlprism_atlas/DOMAIN_PLAN.md, "Approach change"): no probe votes.

On every surface the closed map (a4_closed, faces re-walked by a4_merge) cuts it into faces along every curve where it
meets another surface. Only some of those curves are edges: a curve is where walls hit each other only where the other
surface is itself a wall there. So, solved for all surfaces together until nothing changes:

  1. on each surface, a piece is the faces joined across every curve that is not an edge: an edge is a segment of the
     surface lying on another surface whose face beside that segment is wall, or on a mirror (the chamber's boundary);
  2. each piece is read once, at its deepest point (the point of its faces farthest from its own edges), exactly: it is
     wall where that reads a wall shape (or a line or point: a wall's degenerate reading), else the surface is only
     passing through a region there;
  3. which faces are wall changes which curves are edges: back to 1.

Checks, all on whole pieces:
  (a) every wall's edges close: no edge point of odd degree on any surface;
  (b) an edge of a wall on k along surface m lies on a wall of m too (its rim, or a cut across it): the walls hit
      each other there;
  (c) the swap carries every wall piece onto a wall piece of the same shape;
  (d) every edge exact: on all its surfaces to 1e-7.

    python a4_walls.py kidecaic        (reads a4_rewalked_<group>.pkl; writes a4_walls_<group>.json/.pkl)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import a4_pieces as P  # noqa: E402
import a4_snub as SN  # noqa: E402

REL = lambda S_, x: abs(S_.f(x)) / (np.linalg.norm(S_.g(x)) + 1e-300)  # noqa: E731


SEEDS = []          # (wall name, its exact key, its example seed put exactly on its surface), filled by dims_of


def dims_of(group):
    """Each shape key's kind (region, wall, line, point) and name, from the group's map. The map's example seeds are
    printed to 4 decimals, off a line by up to 1e-4, and the mirrors' shapes carry the map's own keys, not the exact
    reading's. So each wall, line and point example is put exactly onto its shape's set first (the mirrors its zero
    coordinates name, then the surfaces nearest it, until it has its dimension: a wall on one surface, a line on two,
    a point on three) and read there; that key is the shape's too (a wall's name kept where a line reads alike)."""
    import a4_exact as E
    import a4_pieces as P_
    m = json.load(open(os.path.join(HERE, f"a4_map_{group}.json")))
    dim = {t["key"]: t["dim"] for t in m["types"] if t.get("key")}
    name = {t["key"]: t["id"] for t in m["types"] if t.get("key")}
    T = {t["id"]: t for t in m["types"]}
    names = pickle.load(open(os.path.join(HERE, f"a4_closed_{group}.pkl"), "rb"))["surfaces"]
    surfs = [P_.surf_of(n) for n in names]
    codim = {"wall": 1, "line": 2, "point": 3}
    ex, pts = [], []
    for e in m["examples"]:
        d = T[e["id"]]["dim"]
        if d not in codim:
            continue
        x = np.array([float(v) for v in e["beta"].strip("()").split(",")])
        x = x / x.sum()
        on = [k for k in range(4) if x[k] == 0]
        rest = sorted((REL(S, x), k) for k, S in enumerate(surfs) if k >= 4)
        on += [k for r, k in rest if r < 1e-3][:max(0, codim[d] - len(on))]
        y, ok = E.newton(x, [surfs[k] for k in on]) if on else (x, True)
        if ok and np.linalg.norm(y - x) < 1e-3 and y.min() > -1e-12:
            ex.append(e)
            pts.append(np.clip(y, 0, None))
    rank = {"point": 0, "line": 1, "wall": 2}
    SEEDS.clear()
    for e, y, (_, k) in zip(ex, pts, SN.classify_many(pts, SN.SUB[group], tol=1e-7)):
        d = T[e["id"]]["dim"]
        if d == "wall":
            SEEDS.append((e["id"], k, y))
        if k is not None and (k not in dim or (dim[k] in rank and rank[d] > rank[dim[k]])):
            dim[k], name[k] = d, e["id"]
    return dim, name


def face_of(X, surfs, k, x):
    """The face of surface k holding the point x of it: its outline winds once round x, seen in the surface's tangent
    plane at x (of faces whose outline winds round x, the nearest)."""
    import a4_faces as A
    n = P.normal(surfs[k], x) @ A.BZ.T
    e1 = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(n, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    best = None
    for fi, F in enumerate(X["faces"]):
        if F["surface"] != k:
            continue
        L = (np.asarray(F["loop"], float) - x) @ A.BZ.T
        u, v = L @ e1, L @ e2
        ang = np.arctan2(v, u)
        w = np.sum(np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))) / (2 * np.pi)
        if abs(round(w)) == 1:
            d = float(np.min(np.hypot(u, v)))
            if best is None or d > best[0]:
                best = (d, fi)
    return best[1] if best else None


def seg_dist(x, Q):
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return float(np.linalg.norm(x - Q[0]))
    A_, D_ = Q[:-1], Q[1:] - Q[:-1]
    u = np.clip(np.einsum("ij,ij->i", x - A_, D_) / (np.einsum("ij,ij->i", D_, D_) + 1e-300), 0, 1)
    return float(np.min(np.linalg.norm(A_ + u[:, None] * D_ - x, axis=1)))


def solve(X, surfs, group, log=print, rounds=12, forced=None):
    segs, faces = X["segments"], X["faces"]
    dim, name = dims_of(group)
    kn = X.get("keyname", {})
    nm = lambda k: name.get(k) or kn.get(k) or (str(k)[:6] if k else "none")  # noqa: E731

    # (the faces of each surface on each side of each segment)
    beside = defaultdict(list)                      # (surface, segment) -> faces
    for fi, F in enumerate(faces):
        for i, fw in F["cycle"]:
            beside[(F["surface"], i)].append(fi)
    seg_len = [float(np.linalg.norm(np.diff(np.asarray(s["pts"], float), axis=0), axis=1).sum()) for s in segs]
    cache = {}                                      # (point) -> key

    def read(points):
        todo = [p for p in points if tuple(np.round(p, 12)) not in cache]
        if todo:
            for p, (_, k) in zip(todo, SN.classify_many(todo, SN.SUB[group], tol=1e-7)):
                cache[tuple(np.round(p, 12))] = k
        return [cache[tuple(np.round(p, 12))] for p in points]

    # (start: every face of every surface counted as wall, so every curve is an edge and the pieces are the faces)
    wall = [True] * len(faces)
    forced = forced or {}
    prev = None
    for rnd in range(rounds):
        def is_edge(k, i):
            on = segs[i]["on"]
            if any(m < 4 for m in on) and k >= 4:
                return True                         # (the chamber's boundary)
            return any(m != k and any(wall[fj] for fj in beside.get((m, i), ())) for m in on)
        par = list(range(len(faces)))

        def root(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for (k, i), fs in beside.items():
            if len(fs) == 2 and not is_edge(k, i):
                a, b = root(fs[0]), root(fs[1])
                if a != b:
                    par[max(a, b)] = min(a, b)
        groups = defaultdict(list)
        for fi in range(len(faces)):
            groups[root(fi)].append(fi)
        pieces = []
        for r, members in groups.items():
            k = faces[r]["surface"]
            edges = [(i, fw) for fi in members for i, fw in faces[fi]["cycle"]
                     if not (len(beside[(k, i)]) == 2 and root(beside[(k, i)][0]) == root(beside[(k, i)][1]))]
            # (its deepest point: of its faces' points, the farthest from its own edges)
            cand = [np.asarray(x, float) for fi in members for x in list(faces[fi]["probes"]) +
                    list(faces[fi].get("cut probes", []))]
            E_ = [np.asarray(segs[i]["pts"], float) for i, _ in edges]
            best, bd = None, -1.0
            for x in cand:
                d = min(seg_dist(x, Q) for Q in E_) if E_ else 1.0
                if d > bd:
                    best, bd = x, d
            pieces.append({"surface": k, "faces": members, "edges": edges, "deep": best, "depth": bd})
        keys = read([p["deep"] for p in pieces])
        for p, kk in zip(pieces, keys):
            fk = [forced[fi] for fi in p["faces"] if fi in forced]
            p["key"] = fk[0] if fk else kk
            p["inserted"] = bool(fk) and fk[0] != kk
            p["wall"] = (dim.get(p["key"]) == "wall") if p["surface"] >= 4 else True
        new = [False] * len(faces)
        for p in pieces:
            for fi in p["faces"]:
                new[fi] = p["wall"]
        n_wall = sum(p["wall"] for p in pieces if p["surface"] >= 4)
        log(f"  round {rnd}: {len(pieces)} pieces, {n_wall} wall pieces off the mirrors; "
            f"{sum(a != b for a, b in zip(new, wall))} faces changed")
        state = tuple(new)
        if state == prev or new == wall:
            wall = new
            break
        prev = tuple(wall)
        wall = new

    # ---------------------------------------------------------------- checks, on the final pieces
    walls = [p for p in pieces if p["wall"] and p["surface"] >= 4]
    # (a) closing
    loose = []
    for p in walls:
        deg = Counter()
        for i, _ in p["edges"]:
            a, b = segs[i]["ends"]
            if a == b:
                continue
            deg[a] += 1
            deg[b] += 1
        odd = [v for v, d in deg.items() if d % 2]
        if odd:
            loose.append((p["surface"], nm(p["key"]), len(odd)))
    # (b) each edge along another surface lies on a wall there
    unmet = []
    for p in walls:
        for i, _ in p["edges"]:
            on = segs[i]["on"]
            if any(m < 4 for m in on):
                continue
            if not any(m != p["surface"] and any(wall[fj] for fj in beside.get((m, i), ())) for m in on):
                unmet.append((p["surface"], i))
    # (c) the swap, piece by piece: every segment carried to the segment it reverses onto, every face to the face
    #  with those segments, so each wall piece's faces must make exactly one piece there, of the same shape
    from scipy.spatial import cKDTree
    Pts = X["points"]
    live = sorted({e for t in segs for e in t["ends"]})
    ptree = cKDTree(np.array([Pts[v] for v in live]))
    pt_sw = {}
    for v in live:
        d, j = ptree.query(np.asarray(Pts[v], float)[::-1])
        if d < 1e-6:
            pt_sw[v] = live[j]
    by_ends = defaultdict(list)
    for j, t in enumerate(segs):
        by_ends[frozenset(t["ends"])].append(j)
    seg_sw = {}
    for i, t in enumerate(segs):
        a, b = t["ends"]
        if a not in pt_sw or b not in pt_sw:
            continue
        Q = np.asarray(t["pts"], float)[:, ::-1]
        # (the nearest arc with the swapped ends: curves can run within 1e-5 of each other between the same two
        #  points, while an arc's swap image, traced from the swapped level point, lies on it to 1e-15)
        cands = [(max(seg_dist(Q[m], segs[j]["pts"]) for m in (len(Q) // 4, len(Q) // 2, 3 * len(Q) // 4)), j)
                 for j in by_ends.get(frozenset((pt_sw[a], pt_sw[b])), ())]
        if cands and min(cands)[0] < 1e-6:
            seg_sw[i] = min(cands)[1]
    surf_sw = {}
    for k in range(len(surfs)):
        xs = [np.asarray(t["pts"], float)[len(t["pts"]) // 3] for t in segs if k in t["on"]][:12]
        for m in range(len(surfs)):
            if xs and all(REL(surfs[m], x[::-1]) < 1e-8 for x in xs):
                surf_sw[k] = m
                break
    face_at = {(F["surface"], frozenset(i for i, _ in F["cycle"])): fi for fi, F in enumerate(faces)}
    piece_of = {fi: pi for pi, p in enumerate(pieces) for fi in p["faces"]}
    bad_swap = []
    for p in walls:
        m = surf_sw.get(p["surface"])
        img = []
        for fi in p["faces"]:
            cyc = [seg_sw.get(i) for i, _ in faces[fi]["cycle"]]
            img.append(face_at.get((m, frozenset(cyc))) if None not in cyc and m is not None else None)
        qs = {piece_of[f] for f in img if f is not None}
        q = pieces[next(iter(qs))] if len(qs) == 1 else None
        if None in img or len(qs) != 1 or q["key"] != p["key"] or len(q["faces"]) != len(p["faces"]):
            bad_swap.append((p["surface"], nm(p["key"]), len(p["faces"]),
                             [(pieces[x]["surface"], nm(pieces[x]["key"]), len(pieces[x]["faces"])) for x in qs],
                             sum(f is None for f in img)))
    # (d) exact edges
    edge_segs = {i for p in walls for i, _ in p["edges"]}
    off = [i for i in edge_segs
           if max(REL(surfs[m], x) for x in np.asarray(segs[i]["pts"], float) for m in segs[i]["on"]) > 1e-7]
    # (wall shapes found, and the group's walls not found)
    found = Counter(nm(p["key"]) for p in pieces if dim.get(p["key"]) == "wall")
    listed = sorted({name[k] for k, d in dim.items() if d == "wall"})
    report = {
        "pieces": len(pieces),
        "wall pieces (off the mirrors)": len(walls),
        "wall pieces read at a point under 1e-3 from their edges": sum(p["depth"] < 1e-3 for p in walls),
        "wall pieces whose edges do not close": len(loose),
        "edge segments with no wall on their other surface": len(unmet),
        "wall pieces whose swap image reads another shape": len(bad_swap),
        "edge segments off their surfaces by more than 1e-7": len(off),
        "map points with no swap image within 1e-6": len(live) - len(pt_sw),
        "segments with no swap image": len(segs) - len(seg_sw),
        "pieces off the mirrors reading a line or point shape": sum(dim.get(p["key"]) in ("line", "point")
                                                                    for p in pieces if p["surface"] >= 4),
        "mirror pieces not reading a wall shape": sum(dim.get(p["key"]) != "wall" for p in pieces
                                                      if p["surface"] < 4),
        "wall pieces by shape (mirrors too)": dict(sorted(found.items())),
        "the group's walls with no piece": [w for w in listed if w not in found],
        "not closing": loose[:40],
        "swap": bad_swap[:40],
    }
    return pieces, report


if __name__ == "__main__":
    for g in [a for a in sys.argv[1:] if a != "exact"]:
        src = "exactmap" if "exact" in sys.argv[2:] else "rewalked"
        X = pickle.load(open(os.path.join(HERE, f"a4_{src}_{g}.pkl"), "rb"))
        surfs = [P.surf_of(n) for n in X["surfaces"]]
        pieces, report = solve(X, surfs, g)
        # (walls the known data puts there: each wall's example seed and its swap image lie on a piece of that wall;
        #  where the solve left the piece out or read it otherwise, its face is held as that wall and all solved again)
        forced, inserted = {}, []
        piece_of = {fi: p for p in pieces for fi in p["faces"]}
        for wname, key, y in SEEDS:
            for z in (y, y[::-1]):
                for k, S in enumerate(surfs):
                    if REL(S, z) > 1e-9:
                        continue
                    fi = face_of(X, surfs, k, z)
                    if fi is None:
                        inserted.append((wname, k, "no face holds the seed"))
                        continue
                    p = piece_of[fi]
                    if p["key"] != key:
                        forced[fi] = key
                        inserted.append((wname, k, fi, f"read {p['key']}"))
        if forced:
            pieces, report = solve(X, surfs, g, forced=forced)
        report["walls inserted from their seeds"] = inserted
        for p in pieces:
            p["deep"] = [float(v) for v in p["deep"]]
        tag = "_exact" if src == "exactmap" else ""
        pickle.dump(pieces, open(os.path.join(HERE, f"a4_walls{tag}_{g}.pkl"), "wb"))
        json.dump(report, open(os.path.join(HERE, f"a4_walls{tag}_{g}.json"), "w"), indent=1, default=str)
        print(json.dumps(report, indent=1, default=str))
