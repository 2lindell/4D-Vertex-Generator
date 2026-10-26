"""The regions beside each wall's pieces, and from them each region's walls, lines and points (piece_regions.json).

For each patch of each wall (pieces.json): points spread over it (triangle centres, moved onto the wall's surface) are
stepped just off it both ways and classified; the pair of regions there is that part's two sides (a cut divides a
patch into parts with different sides). The patch's boundary, labelled stretch by stretch with its edge, gives the
lines; where the label changes along it is a corner, classified (hull tolerance relaxed) to name the point.

For each region: the walls with a part beside it (full where all of the wall faces it on that side, partial where only
some parts do), the lines bounding those parts, and the points at their corners.

    python piece_regions.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import optimal_seeds as osd
import pieces as pc
from edge_relabel import lab_edge
from probe import _one

STEP = 3e-4
N = 240
LIB = None
JOBS = []


def lib():
    global LIB
    if LIB is None:
        planes, curved = osd.library()
        LIB = {s.name: s for s in list(planes) + list(curved)}
    return LIB


def sides(i):
    """The two regions beside the point (stepped off along the surface's normal), and the shape at it."""
    x, sname = JOBS[i]
    S = lib()[sname]
    g = pc.grad(S, x)
    g = g / (np.linalg.norm(g) + 1e-300)
    out = []
    for sg in (1, -1):
        y = x + sg * STEP * g
        h = osd.home(y / y.sum()) if (y > -1e-12).all() else None
        out.append(_one(h)[0] if h is not None else "OUT")
    return out


def corner(i):
    return lab_edge(osd.home(pc.nrm(JOBS[i])))


def main():
    work = json.load(open("working_data.json"))
    P = json.load(open("pieces.json"))
    dims = {w: "wall" for w in work["walls"]}
    regions = set(work["regions"])
    jobs, where = [], []
    corners, cwhere = [], []
    for w, rec in P.items():
        for k, pt in enumerate(rec.get("patches", [rec])):
            T = np.array(pt["tris"], float)
            if not len(T):
                continue
            S = lib()[pt["surface"]]
            rng = np.random.default_rng(0)
            idx = rng.choice(len(T), min(N, len(T)), replace=False)
            for t in T[idx]:
                x = pc.nrm(pc.project([S], t.mean(0)))
                jobs.append((x, pt["surface"]))
                where.append((w, k))
            # (corners: where the boundary label changes between neighbouring boundary segments)
            segs = pt.get("boundary", [])
            ends = [(np.array(a[0]), a[1]) for sg in segs for a in sg]
            if ends:
                E = np.array([e for e, _ in ends])
                L = [n for _, n in ends]
                tr = cKDTree(E)
                for i, j in tr.query_pairs(2e-3):
                    if L[i] != L[j]:
                        corners.append(pc.nrm((E[i] + E[j]) / 2))
                        cwhere.append((w, k, tuple(sorted((L[i], L[j])))))
    JOBS[:] = jobs
    with Pool(3) as pool:
        S2 = pool.map(sides, range(len(jobs)), chunksize=16)
    JOBS[:] = corners
    with Pool(3) as pool:
        CN = pool.map(corner, range(len(corners)), chunksize=16)
    # (each patch: its parts by region pair, with their sample points)
    parts = defaultdict(lambda: defaultdict(list))
    for (w, k), (a, b), x in zip(where, S2, [j[0] for j in jobs]):
        a = a if a in regions else None
        b = b if b in regions else None
        if a and b and a != b:
            parts[(w, k)][tuple(sorted((a, b)))].append(x)
    cpts = defaultdict(list)
    for (w, k, labs), name, x in zip(cwhere, CN, corners):
        cpts[(w, k)].append((name, labs, x))
    walls_out = {}
    reg = defaultdict(lambda: {"walls": {}, "lines": set(), "points": set()})
    for w, rec in P.items():
        ps = rec.get("patches", [rec])
        sides_all = Counter()
        for k, pt in enumerate(ps):
            for pair, xs in parts.get((w, k), {}).items():
                sides_all[pair] += len(xs)
        tot = sum(sides_all.values())
        walls_out[w] = {"sides": {f"{a}|{b}": round(n / max(1, tot), 3) for (a, b), n in sides_all.most_common()}}
        # (a region beside the wall: full if it faces all of the wall on its side, else partial)
        for (a, b), n in sides_all.items():
            for r in (a, b):
                share = sum(m for pair, m in sides_all.items() if r in pair) / max(1, tot)
                reg[r]["walls"][w] = "full" if share > 0.97 else "partial"
        # (lines and points: the boundary of the parts facing each region)
        for k, pt in enumerate(ps):
            segs = pt.get("boundary", [])
            if not segs:
                continue
            B = np.array([a[0] for sg in segs for a in sg], float)
            labs = [a[1] for sg in segs for a in sg]
            tb = cKDTree(B)
            for pair, xs in parts.get((w, k), {}).items():
                X = np.array(xs)
                near = set()
                for x in X:
                    for i in tb.query_ball_point(x, 0.05):
                        near.add(labs[i])
                ls = {n for n in near if n in work["walls"][w]["edges"] or n in [c["by"] for c in work["walls"][w]["cuts"]]}
                pts = set()
                for name, lb, x in cpts.get((w, k), []):
                    if np.min(np.linalg.norm(X - x, axis=1)) < 0.08 and not name.startswith(("new", "ERR", "OUT")):
                        pts.add(name)
                for r in pair:
                    reg[r]["lines"] |= {n for n in ls if n not in work["walls"]}
                    reg[r]["points"] |= pts
    out = {"_about": __doc__.strip().split("\n\n")[0], "walls": walls_out,
           "regions": {r: {"walls": dict(sorted(v["walls"].items())), "lines": sorted(v["lines"]),
                           "points": sorted(v["points"])} for r, v in sorted(reg.items())}}
    json.dump(out, open("piece_regions.json", "w"), indent=1)
    print(f"{len(walls_out)} walls; {len(out['regions'])} regions")
    for r, v in out["regions"].items():
        print(f"  {r}: walls {v['walls']}; lines {v['lines']}; points {v['points']}")


if __name__ == "__main__":
    main()
