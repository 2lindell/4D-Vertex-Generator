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
import os
import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import optimal_seeds as osd
import pieces as pc
from edge_relabel import lab_edge
from probe import _one

STEP = 1e-5                   # (off the wall: small enough not to pass a thin region, X68 is 1.6e-4 across)
N = 300
EDGE_OFF = 0.01
MINOR = 0.02
CUT_OFF = 0.004
CACHE = os.environ.get("PR_CACHE", "piece_regions_cache.pkl")
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
        got = set()
        for st in (STEP, STEP / 3):                        # (two distances off: they must agree)
            y = x + sg * st * g
            h = osd.home(y / y.sum()) if (y > -1e-12).all() else None
            got.add(_one(h)[0] if h is not None else "OUT")
        out.append(got.pop() if len(got) == 1 else "UNSURE")
    return out


def own_label(i):
    x = JOBS[i]
    return _one(osd.home(pc.nrm(x)))[0]


def corner(i):
    return lab_edge(osd.home(pc.nrm(JOBS[i])))


def named_points():
    """The named points, exactly: each line's ends that are points (working data) and the cross-ring points."""
    work = json.load(open("working_data.json"))
    out = [(p["name"], np.array(p["beta"], float)) for p in json.load(open("cross_ring_points.json"))["points"]
           if p["name"] != "icosafold point"]
    for v in work.get("lines", {}).values():
        for e in v.get("ends", []):
            if e.get("status") == "point" and e.get("at"):
                x = np.array(e["at"], float)
                if not any(np.linalg.norm(x - y) < 1e-7 for _, y in out):
                    out.append((e["on"], x))
    return out


NAMED_POINTS = []


def main():
    NAMED_POINTS[:] = named_points()
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
            segs = pt.get("boundary", [])
            Bt = cKDTree(np.array([a[0] for sg in segs for a in sg], float)) if segs else None
            for t in T[idx]:
                x = pc.nrm(pc.project([S], t.mean(0)))
                if Bt is not None and Bt.query(x)[0] < EDGE_OFF:   # (too near the piece's edge to tell its sides)
                    continue
                jobs.append((x, pt["surface"]))
                where.append((w, k))
            # (beside each cut, both ways: the parts it divides, however narrow)
            Tc = cKDTree(T.mean(1))
            for name, Pc in pt.get("cuts", []):
                Pc = np.asarray(Pc, float)
                if len(Pc) < 3:
                    continue
                for i in np.linspace(1, len(Pc) - 2, 8).astype(int):
                    p, t = Pc[i], Pc[i + 1] - Pc[i - 1]
                    Tn = np.linalg.svd(np.vstack([pc.grad(S, p), np.ones(4), t]))[2][-1]
                    for sg in (1, -1):
                        x = pc.nrm(pc.project([S], p + sg * CUT_OFF * Tn))
                        if pc.in_half(x) and Tc.query(x)[0] < 2 * pt.get("h", 0.008):   # (on the piece only)
                            jobs.append((x, pt["surface"]))
                            where.append((w, k, "cut"))
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
    # (side checks cached by point, so a rerun only does the new ones)
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    keys = [(sn, tuple(np.round(x, 12))) for x, sn in jobs]
    todo = [i for i, kk in enumerate(keys) if kk not in cache]
    JOBS[:] = jobs
    with Pool(4) as pool:
        for i, r in zip(todo, pool.map(sides, todo, chunksize=16)):
            cache[keys[i]] = r
    pickle.dump(cache, open(CACHE, "wb"))
    S2 = [cache[kk] for kk in keys]
    # (a sample beside a cut counts only where it is on the wall itself, not on another wall on the same surface)
    ci = [i for i, wh in enumerate(where) if len(wh) > 2]
    ck = [("self", kk) for kk in (keys[i] for i in ci)]
    todo = [n for n, kk in enumerate(ck) if kk not in cache]
    JOBS[:] = [jobs[i][0] for i in ci]
    with Pool(4) as pool:
        for n, r in zip(todo, pool.map(own_label, todo, chunksize=16)):
            cache[ck[n]] = r
    pickle.dump(cache, open(CACHE, "wb"))
    for i, kk in zip(ci, ck):
        if cache[kk] != where[i][0]:
            S2[i] = ["UNSURE", "UNSURE"]
    JOBS[:] = corners
    with Pool(4) as pool:
        CN = pool.map(corner, range(len(corners)), chunksize=16)
    # (each patch: its parts by region pair, with their sample points)
    parts = defaultdict(lambda: defaultdict(list))
    cut_pairs = defaultdict(set)
    for wh, (a, b), x in zip(where, S2, [j[0] for j in jobs]):
        w, k = wh[:2]
        a = a if a in regions else None
        b = b if b in regions else None
        if a and b and (a != b or w in work.get("same_both_sides", {})):   # (a wall with one region both sides)
            parts[(w, k)][tuple(sorted((a, b)))].append(x)
            if len(wh) > 2:
                cut_pairs[w].add(tuple(sorted((a, b))))
    cpts = defaultdict(list)
    for (w, k, labs), name, x in zip(cwhere, CN, corners):
        cpts[(w, k)].append((name, labs, x))
    walls_out = {}
    reg = defaultdict(lambda: {"walls": {}, "lines": set(), "points": set(), "at": []})
    for w, rec in P.items():
        ps = rec.get("patches", [rec])
        sides_all = Counter()
        for k, pt in enumerate(ps):
            for pair, xs in parts.get((w, k), {}).items():
                sides_all[pair] += len(xs)
        tot = sum(sides_all.values())
        # (a pair of sides seen at under MINOR of the wall's points is listed for review, not counted)
        minor = {pair for pair, n in sides_all.items() if n < MINOR * tot and pair not in cut_pairs[w]}
        walls_out[w] = {"sides": {f"{a}|{b}": round(n / max(1, tot), 3) for (a, b), n in sides_all.most_common()
                                  if (a, b) not in minor},
                        "minor_sides": {f"{a}|{b}": round(n / max(1, tot), 3) for (a, b), n in sides_all.most_common()
                                        if (a, b) in minor}}
        for pair in minor:
            del sides_all[pair]
            for k in range(len(ps)):
                parts.get((w, k), {}).pop(pair, None)
        tot = sum(sides_all.values())
        # (a region beside the wall: full if it faces all of the wall on its side, else partial)
        for (a, b), n in sides_all.items():
            for r in (a, b):
                share = sum(m for pair, m in sides_all.items() if r in pair) / max(1, tot)
                reg[r]["walls"][w] = "full" if share > 0.97 else "partial"
        # (lines and points: each boundary point of the piece, and each corner, belongs to the parts of the nearest
        # sample points (their sides))
        for k, pt in enumerate(ps):
            segs = pt.get("boundary", [])
            prt = parts.get((w, k), {})
            if not segs or not prt:
                continue
            X = np.array([x for xs in prt.values() for x in xs])
            Pr = [pair for pair, xs in prt.items() for _ in xs]
            tx = cKDTree(X)
            edges = set(work["walls"][w]["edges"]) | {c["by"] for c in work["walls"][w]["cuts"]}
            for sg in segs:
                for x, n in sg:
                    if n not in edges or n in work["walls"]:
                        continue
                    d, i = tx.query(np.asarray(x, float))
                    if d < 0.04:
                        for r in Pr[i]:
                            reg[r]["lines"].add(n)
            # (points: the named points, exact (the lines' ends and the cross-ring points), lying on this wall's surface
            # at its piece's boundary, belong to the parts beside them there)
            S = lib()[pt["surface"]]
            Bp = np.array([x for sg in segs for x, _ in sg], float)
            tb = cKDTree(Bp)
            for name, x in NAMED_POINTS:
                if tb.query(x)[0] > 0.012 or abs(pc.fval(S, x)) / (np.linalg.norm(pc.grad(S, x)) + 1e-300) > 1e-8:
                    continue
                d, i = tx.query(x)
                if d < 0.04:
                    for r in Pr[i]:
                        reg[r]["points"].add(name)
                        reg[r]["at"].append(x)
    # (a line that only touches a region at one of its points, without bounding it along its length: partial)
    # (by position: a point name has several places in the half-cell, and a line touches the region only at the one
    # where it ends)
    ends = {L_: [np.array(e["at"], float) for e in v.get("ends", []) if e.get("at")]
            for L_, v in work.get("lines", {}).items()}
    for r, v in reg.items():
        v["lines_partial"] = {L_ for L_, xs in ends.items() if L_ not in v["lines"] and any(
            np.linalg.norm(x - y) < 1e-6 for x in xs for y in v["at"])}
    out = {"_about": __doc__.strip().split("\n\n")[0], "walls": walls_out,
           "regions": {r: {"walls": dict(sorted(v["walls"].items())),
                           "lines": {**{L_: "full" for L_ in sorted(v["lines"])},
                                     **{L_: "partial (touches at a point)" for L_ in sorted(v["lines_partial"])}},
                           "points": sorted(v["points"])} for r, v in sorted(reg.items())}}
    json.dump(out, open("piece_regions.json", "w"), indent=1)
    print(f"{len(walls_out)} walls; {len(out['regions'])} regions")
    for r, v in out["regions"].items():
        print(f"  {r}: walls {v['walls']}; lines {v['lines']}; points {v['points']}")


if __name__ == "__main__":
    main()
