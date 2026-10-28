"""The regions and their walls as one model, from the exact data alone (no cutting of meshes): region_model.json.

For each wall (its main pieces: the curved walls' meshes on their exact surfaces, the flat walls' exact patches):
  1. every exact surface of the library (the half-cell's planes under the extended group, every curved wall's surface
     and its images) is evaluated at points spread over the wall; one that changes sign over it crosses the wall
     there, along the curve where it is zero;
  2. the wall's points with the same sign against every crossing surface, and joined to each other, are one face:
     a piece of the wall bounded by exact curves (each named by its surface) and the wall's own edges;
  3. each face is named by classifying points just off it on each side (two points, each side, as a check: a face
     named two ways is crossed by a surface the library lacks);
  4. neighbouring faces with the same regions on both sides are one section; the curves left between sections are
     the lines where a region changes on the wall.
Each region's walls are the walls with a section beside it (in part, where other sections of that wall are not).

    python region_model.py [X94,F1,...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import optimal_seeds as osd
from probe import _one

PHI = (1 + 5 ** 0.5) / 2
H = 0.004                    # (spacing of the points spread over a wall)
OFF = (1.5e-4, 4e-4)         # (steps off a face to name the regions either side)
ZERO = 1e-12


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def flat_points(C, h=H):
    """Points spread over a flat polygon (its corners in betas, in order)."""
    import shapely
    from shapely.geometry import Polygon
    o = C.mean(0)
    U = np.linalg.svd(C - o)[2][:2]
    P2 = Polygon((C - o) @ U.T).buffer(0)
    lo, hi = np.array(P2.bounds[:2]), np.array(P2.bounds[2:])
    g = np.mgrid[lo[0]:hi[0]:h, lo[1]:hi[1]:h].reshape(2, -1).T
    inside = shapely.contains_xy(P2, g[:, 0], g[:, 1])
    pts = nrm(o + g[inside] @ U)
    return pts[(pts > -1e-12).all(1)]


def curved_points(tris, h=H):
    """Points spread over a curved wall's mesh (its vertices and triangle centres, thinned to the spacing)."""
    T = nrm(np.array(tris, float))
    P = np.vstack([T.reshape(-1, 4), T.mean(1)])
    P = np.unique(np.round(P, 9), axis=0)
    keep, tree_pts = [], []
    order = np.random.default_rng(0).permutation(len(P))
    taken = np.zeros(len(P), bool)
    tree = cKDTree(P)
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(P[i], 0.6 * h)] = True
    return P[keep]


def walls():
    """(wall, list of (piece name, points, own surface name or plane normal))."""
    out = defaultdict(list)
    X = json.load(open("xsurfaces.json"))
    for tid, sf in X.items():
        out[tid].append(("mesh", curved_points(sf["tris"]), tid))
    d = json.load(open("fexact_patches.json"))
    for p in (d if isinstance(d, list) else d.values()):
        if not isinstance(p, dict) or "corners" not in p:
            continue
        C = nrm(json.loads(p["corners"]) if isinstance(p["corners"], str) else p["corners"])
        if p["target"] == "T2" and len(C) == 3 and abs(np.asarray(p["normal"], float) @ [0, 1, -1, 0]) > 1.4:
            continue                                  # (T2's former region B: the wall X96)
        out[p["target"]].append((f"patch in {p['wall']}", flat_points(C), np.asarray(p["normal"], float)))
    tri = nrm([[0, 1, 1, 0], [0, 1, 1, 1], [1, PHI ** 2, PHI ** 2, 0]])
    out["X96"].append(("triangle in b2=b3", flat_points(tri), np.array([0, 1, -1, 0.0])))
    return out


LIB = None
JOBS = []


def name_side(i):
    x, n = JOBS[i]
    out = []
    for sg in (1, -1):
        labs = []
        for off in OFF:
            b = osd.home(x + sg * off * n)
            labs.append(_one(b)[0] if np.isfinite(b).all() and (b > -1e-12).all() else "OUT")
        out.append(labs)
    return out


def main(only=None):
    global LIB
    planes, curved = osd.library()
    faces_ = [osd.Plane(np.eye(4)[k], f"face b{k + 1}=0") for k in range(4)] + [osd.Plane([0, 0, 1, -1.0], "mirror b3=b4")]
    surfs = faces_ + list(planes) + list(curved)
    W = walls()
    model, jobs, where = {}, [], []
    for w, pieces in W.items():
        if only and w not in only:
            continue
        for pname, P, own in pieces:
            if len(P) < 3:
                continue
            # (1) the surfaces changing sign over the piece (not one it lies on)
            V = np.stack([s.f(P) for s in surfs], 1)
            scale = np.abs(V).max(0) + 1e-300
            S = np.sign(np.where(np.abs(V) < ZERO * scale, 0, V)).astype(np.int8)
            crossing = [k for k in range(len(surfs)) if (S[:, k] > 0).any() and (S[:, k] < 0).any()]
            # (2) faces: neighbouring points with the same signs against every crossing surface
            tree = cKDTree(P)
            pairs = np.array(sorted(tree.query_pairs(1.6 * H)))
            if not len(pairs):
                continue
            sig = S[:, crossing]
            same = (sig[pairs[:, 0]] == sig[pairs[:, 1]]).all(1)
            pp = pairs[same]
            G = coo_matrix((np.ones(len(pp)), (pp[:, 0], pp[:, 1])), shape=(len(P), len(P)))
            nf, lab = connected_components(G, directed=False)
            # (3) each face named: two of its points farthest inside it, stepped off both ways along the normal
            if isinstance(own, str):
                s_own = next((c for c in curved if c.name == own), None) or min(
                    curved, key=lambda c: float(np.abs(c.f(P[::9])).max() / (np.abs(c.f(P[::9] + 1e-3)).max() + 1e-30)))
                normal = lambda x: (lambda g: g / np.linalg.norm(g))(s_own.grad(x) - s_own.grad(x).mean())
            else:
                nn = own - own.mean()
                nn = nn / np.linalg.norm(nn)
                normal = lambda x: nn
            faces = []
            for f in range(nf):
                idx = np.flatnonzero(lab == f)
                if len(idx) < 3:
                    continue
                other = np.flatnonzero(lab != f)
                depth = cKDTree(P[other]).query(P[idx])[0] if len(other) else np.ones(len(idx))
                pick = idx[np.argsort(-depth)[:2]]
                fi = {"points": int(len(idx)), "crossing": [surfs[crossing[k]].name for k in range(len(crossing))
                                                            if len(set(sig[idx, k])) == 1 and sig[idx[0], k] != 0][:0],
                      "signs": sig[idx[0]].tolist(), "centre": P[pick[0]].tolist()}
                for x in P[pick]:
                    jobs.append((x, normal(x)))
                    where.append((w, pname, len(faces)))
                faces.append(fi)
            model.setdefault(w, []).append({"piece": pname, "crossing": [surfs[k].name for k in crossing],
                                            "faces": faces, "pairs_between": None})
            # (which faces touch which, and across which surfaces: for the lines between sections)
            touch = defaultdict(set)
            for a, b in pairs[~same]:
                fa, fb = lab[a], lab[b]
                if fa != fb:
                    diff = [crossing[k] for k in np.flatnonzero(sig[a] != sig[b])]
                    touch[(min(fa, fb), max(fa, fb))] |= {surfs[k].name for k in diff}
            keep = {f: i for i, f in enumerate([f for f in range(nf) if (lab == f).sum() >= 3])}
            model[w][-1]["touch"] = [[keep[a], keep[b], sorted(v)[:4]] for (a, b), v in touch.items() if a in keep and b in keep]
            print(f"{w} {pname}: {len(P)} points, {len(crossing)} surfaces cross it, {len(faces)} faces", flush=True)
    JOBS[:] = jobs
    with Pool(3) as pool:
        named = pool.map(name_side, range(len(jobs)), chunksize=8)
    for (w, pname, fi), sides in zip(where, named):
        piece = next(p for p in model[w] if p["piece"] == pname)
        piece["faces"][fi].setdefault("named", []).append(sides)
    # (4) each face's regions: the first region met stepping off each side; a face named two ways is flagged
    regions = {t for t in json.load(open("region_defs.json")) if not t.startswith("_")}
    for w, pieces in model.items():
        for pc in pieces:
            for f in pc["faces"]:
                sides = []
                for s_ in range(2):
                    got = [next((x for x in nm[s_] if x in regions), None) for nm in f.get("named", [])]
                    sides.append(got)
                f["sides"] = [max(set(g), key=g.count) if g else None for g in sides]
                f["agree"] = all(len(set(g)) == 1 for g in sides)
    json.dump(model, open("region_model.json", "w"), indent=1)
    # (the regions' walls, from the faces)
    reg = defaultdict(lambda: defaultdict(int))
    for w, pieces in model.items():
        for pc in pieces:
            for f in pc["faces"]:
                a, b = f["sides"]
                if a and b and a != b:
                    reg[a][w] += f["points"]
                    reg[b][w] += f["points"]
    defs = json.load(open("region_defs.json"))
    print()
    for r in sorted(reg, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0)):
        old = set(defs.get(r, {}).get("walls", [])) if isinstance(defs.get(r), dict) else set()
        new = set(reg[r])
        print(f"{r}: walls {sorted(new)}" + (f"   (database: +{sorted(new - old)} -{sorted(old - new)})" if new != old else "   (as in the database)"))
    bad = [(w, pc["piece"], i, f["named"]) for w, pcs in model.items() for pc in pcs for i, f in enumerate(pc["faces"]) if not f["agree"]]
    print(f"\nfaces named two ways (a surface missing from the library?): {len(bad)}")
    for b in bad[:20]:
        print("  ", b[0], b[1], "face", b[2], b[3])


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None)
