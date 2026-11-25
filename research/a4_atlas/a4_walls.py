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


def dims_of(group):
    """Each shape key's kind (region, wall, line, point) from the group's map."""
    m = json.load(open(os.path.join(HERE, f"a4_map_{group}.json")))
    return {t["key"]: t["dim"] for t in m["types"] if t.get("key")}, {t["key"]: t["id"] for t in m["types"]
                                                                       if t.get("key")}


def seg_dist(x, Q):
    Q = np.asarray(Q, float)
    if len(Q) == 1:
        return float(np.linalg.norm(x - Q[0]))
    A_, D_ = Q[:-1], Q[1:] - Q[:-1]
    u = np.clip(np.einsum("ij,ij->i", x - A_, D_) / (np.einsum("ij,ij->i", D_, D_) + 1e-300), 0, 1)
    return float(np.min(np.linalg.norm(A_ + u[:, None] * D_ - x, axis=1)))


def solve(X, surfs, group, log=print, rounds=12):
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
            p["key"] = kk
            p["wall"] = dim.get(kk) in ("wall", "line", "point") if p["surface"] >= 4 else True
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
    # (c) the swap: each wall piece's deepest point reversed reads the same shape
    sw = read([np.asarray(p["deep"], float)[::-1] for p in walls])
    bad_swap = [(p["surface"], nm(p["key"]), nm(s)) for p, s in zip(walls, sw) if s != p["key"]]
    # (d) exact edges
    edge_segs = {i for p in walls for i, _ in p["edges"]}
    off = [i for i in edge_segs
           if max(REL(surfs[m], x) for x in np.asarray(segs[i]["pts"], float) for m in segs[i]["on"]) > 1e-7]
    # (wall shapes found, and the group's walls not found)
    found = Counter(nm(p["key"]) for p in walls)
    listed = sorted(name[k] for k, d in dim.items() if d == "wall")
    report = {
        "pieces": len(pieces),
        "wall pieces (off the mirrors)": len(walls),
        "wall pieces read at a point under 1e-3 from their edges": sum(p["depth"] < 1e-3 for p in walls),
        "wall pieces whose edges do not close": len(loose),
        "edge segments with no wall on their other surface": len(unmet),
        "wall pieces whose swap image reads another shape": len(bad_swap),
        "edge segments off their surfaces by more than 1e-7": len(off),
        "wall pieces by shape": dict(sorted(found.items())),
        "the group's walls with no piece": [w for w in listed if w not in found],
        "not closing": loose[:40],
        "swap": bad_swap[:40],
    }
    return pieces, report


if __name__ == "__main__":
    for g in sys.argv[1:]:
        X = pickle.load(open(os.path.join(HERE, f"a4_rewalked_{g}.pkl"), "rb"))
        surfs = [P.surf_of(n) for n in X["surfaces"]]
        pieces, report = solve(X, surfs, g)
        for p in pieces:
            p["deep"] = [float(v) for v in p["deep"]]
        pickle.dump(pieces, open(os.path.join(HERE, f"a4_walls_{g}.pkl"), "wb"))
        json.dump(report, open(os.path.join(HERE, f"a4_walls_{g}.json"), "w"), indent=1, default=str)
        print(json.dumps(report, indent=1, default=str))
