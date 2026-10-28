"""The mixed sections mixed_lines.py could not settle with one surface.

  * Two cuts on one section: the boundary points are matched surface by surface: the best surface takes the points it
    passes through (to 1e-4), the next best is sought among the rest, and so on; each surface through at least three
    boundary points is a cut (added to forced_lines.json).
  * No surface through the points: the wall the two regions meet across is looked for off the section: from a
    boundary point, a step 1e-3 off the section on the mixed side, then bisected between the two regions parallel to
    the section; the points so found lie on that wall, and are classified with the hull tolerance relaxed
    (edge_relabel.lab_edge) to name it, and matched to the library too.

    python mixed_more.py "T2:10:0,T2:10:1,X20:2:0,..."      (wall:section:side, written to mixed_more.json)
"""
from __future__ import annotations

import gzip
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import mixed_lines as ml
import optimal_seeds as osd
import region_cells as rc
from cut_walls import sample_tris
from edge_relabel import lab_edge
from probe import _one
from region_db import home


def label(x):
    b = home(rc.betas(x[None]))[0]
    return _one(b)[0] if np.isfinite(b).all() else "OUT"


def across(args):
    """From x (on the section, normal n, mixed side sg), 1e-3 off it, the point between region a (toward x0) and
    the other region (toward x1), found by bisection: a point on the wall between them."""
    x0, x1, n, sg, a = args
    off = 1e-3 * sg * n
    p0, p1 = x0 + off, x1 + off
    if label(p0) != a:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(30):
        m = (lo + hi) / 2
        if label(p0 + m * (p1 - p0)) == a:
            lo = m
        else:
            hi = m
    return p0 + (lo + hi) / 2 * (p1 - p0)


def match(B, planes, curved, own):
    """Surfaces through the points, greedily: [(surface name, number of points, median off)]."""
    left = np.arange(len(B))
    out = []
    while len(left) >= 3:
        best = None
        for s in list(planes) + list(curved):
            if s in own:
                continue
            r = np.array([abs(s.f(b)[0]) / (np.linalg.norm(s.grad(b)) + 1e-30) for b in B[left]])
            k = (r < 1e-4).sum()
            if best is None or k > best[1] or (k == best[1] and np.median(r) < best[2]):
                best = (s, k, float(np.median(r)), r)
        if best is None or best[1] < 3:
            break
        out.append((best[0], int(best[1]), float(np.median(best[3][best[3] < 1e-4]))))
        left = left[best[3] >= 1e-4]
    return out, len(left)


def main(spec):
    osd.LIB = osd.library()
    planes, curved = osd.LIB
    forced = json.load(open(ml.OUT))
    results = {}
    with Pool(3) as pool:
        for item in spec.split(","):
            w, k, sd = item.split(":")
            k, sd = int(k), int(sd)
            f = json.load(gzip.open(f"sections/{w}.json.gz", "rt"))[k]
            T = np.array(f["tris"]).reshape(-1, 3, 3)
            X, owner = sample_tris(T, max(0.0015, np.sqrt(f["area"]) / 15), owner=True)
            N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            N = (N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-300))[owner]
            if len(X) > 300:
                pick = np.random.default_rng(0).choice(len(X), 300, replace=False)
                X, N = X[pick], N[pick]
            sg = 1 if sd == 0 else -1
            labs = np.array(pool.map(ml.name_at, [(x, n, sg) for x, n in zip(X, N)]))
            cnt = {l: int((labs == l).sum()) for l in set(labs)}
            regions = [l for l in cnt if l.startswith(("X", "F")) and l not in (w,)]
            tree = cKDTree(X)
            h = 2.5 * np.median(tree.query(X, 2)[0][:, 1])
            pairs = [(i, j) for i, j in tree.query_pairs(h) if labs[i] != labs[j] and labs[i] in regions and labs[j] in regions]
            rng = np.random.default_rng(0)
            pairs = [pairs[i] for i in rng.choice(len(pairs), min(len(pairs), 24), replace=False)] if pairs else []
            if not pairs:
                results[item] = {"labels": cnt, "note": "no neighbouring points named differently"}
                print(item, results[item], flush=True)
                continue
            P = np.array(pool.map(ml.bisect, [(X[i], X[j], N[i], N[j], sg, labs[i]) for i, j in pairs]))
            B = rc.betas(P)
            B = B / B.sum(1, keepdims=True)
            Bs = rc.betas(X[:12])
            Bs = Bs / Bs.sum(1, keepdims=True)
            own = [s for s in list(planes) + list(curved)
                   if np.median([abs(s.f(b)[0]) / (np.linalg.norm(s.grad(b)) + 1e-30) for b in Bs]) < 3e-4]
            cuts, left = match(B, planes, curved, own)
            # (the wall across: off the section, between the two regions)
            Q = [q for q in pool.map(across, [(X[i], X[j], N[i], sg, labs[i]) for i, j in pairs[:12]]) if q is not None]
            names = pool.map(lab_edge, [osd.home(rc.betas(q[None])[0]) for q in Q]) if Q else []
            QB = rc.betas(np.array(Q)) if Q else np.zeros((0, 4))
            QB = QB / QB.sum(1, keepdims=True) if len(QB) else QB
            wall_match, _ = match(QB, planes, curved, own) if len(QB) >= 3 else ([], 0)
            results[item] = {
                "labels": cnt, "pairs": len(pairs),
                "cuts": [{"surface": s.name, "points": n_, "median_off": m_} for s, n_, m_ in cuts],
                "unmatched_points": int(left),
                "wall_across": {"names": {x: names.count(x) for x in set(names)},
                                "surfaces": [{"surface": s.name, "points": n_, "median_off": m_} for s, n_, m_ in wall_match],
                                "points": QB.round(8).tolist()}}
            print(item, json.dumps({k_: v_ for k_, v_ in results[item].items() if k_ != "wall_across"}),
                  "| across:", results[item]["wall_across"]["names"], [x["surface"] for x in results[item]["wall_across"]["surfaces"]],
                  flush=True)
            # (every cut through three or more boundary points, kept for cut_walls)
            for s, n_, m_ in cuts:
                Pk = P[np.array([abs(s.f(b)[0]) / (np.linalg.norm(s.grad(b)) + 1e-30) for b in B]) < 1e-4]
                e = {"section": k, "side": sd, "regions": regions, "surface": s.name,
                     "normal": s.n.tolist() if isinstance(s, osd.Plane) else None, "points": Pk.round(8).tolist(),
                     "median_off": m_, "by": "mixed_more"}
                if not any(x["section"] == k and x["surface"] == s.name for x in forced.get(w, [])):
                    forced.setdefault(w, []).append(e)
            json.dump(results, open("mixed_more.json", "w"), indent=1)
    json.dump(forced, open(ml.OUT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
