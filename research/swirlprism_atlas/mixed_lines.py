"""The cuts a mixed section is missing. A section the classifier found mixed (one region on part of a side, another
on the rest) is crossed by a line no meeting found: where another wall's surface passes through it. That line is
found from the classification itself: points spread over the section are named by the region a small step off the
mixed side; between two neighbours named differently, the change is bisected along the section; and the boundary
points are matched to the exact surface library (optimal_seeds.library: every plane of the half-cell under the
extended group, every curved wall's surface and its images). The surface through all of them, to 1e-6, is the cut.

    python mixed_lines.py          (reads region_check.json and sections/; writes forced_lines.json)

cut_walls.py adds each as a line on the wall's pieces it crosses ("<surface> (region boundary)").
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
import region_cells as rc
from cut_walls import sample_tris
from probe import _one
from region_db import home

STEP = 2e-5                 # (off the section, on the mixed side: close, so the boundary is where it crosses the wall)
OUT = "forced_lines.json"


def name_at(args):
    x, n, sg = args
    b = home(rc.betas((x + sg * STEP * n)[None]))[0]
    return _one(b)[0] if np.isfinite(b).all() else "OUT"


def bisect(args):
    """The point between x0 (named a) and x1 (named otherwise) on the section where the name changes."""
    x0, x1, n0, n1, sg, a = args
    lo, hi = 0.0, 1.0
    for _ in range(22):
        m = (lo + hi) / 2
        x = x0 + m * (x1 - x0)
        nn = n0 + m * (n1 - n0)
        if name_at((x, nn / np.linalg.norm(nn), sg)) == a:
            lo = m
        else:
            hi = m
    return x0 + (lo + hi) / 2 * (x1 - x0)


def main():
    check = json.load(open("region_check.json"))
    osd.LIB = osd.library()
    planes, curved = osd.LIB
    out = defaultdict(list)
    todo = [m for m in check["mixed"] if m.get("area", 0) > 1e-4]
    with Pool(4) as pool:
        for m in todo:
            w, k, sd = m["wall"], m["section"], m["side"]
            f = json.load(gzip.open(f"sections/{w}.json.gz", "rt"))[k]
            T = np.array(f["tris"]).reshape(-1, 3, 3)
            X, owner = sample_tris(T, max(0.0015, np.sqrt(f["area"]) / 15), owner=True)
            N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-300)
            N = N[owner]
            sg = 1 if sd == 0 else -1
            labs = pool.map(name_at, [(x, n, sg) for x, n in zip(X, N)])
            labs = np.array(labs)
            regions = [l for l in set(labs) if l in m["labels"]]
            if len(regions) < 2:
                continue
            # (neighbouring points named differently: the change bisected between them)
            from scipy.spatial import cKDTree
            tree = cKDTree(X)
            h = 2.5 * np.median(tree.query(X, 2)[0][:, 1])
            pairs = [(i, j) for i, j in tree.query_pairs(h) if labs[i] != labs[j] and labs[i] in regions
                     and labs[j] in regions]
            rng = np.random.default_rng(0)
            pairs = [pairs[i] for i in rng.choice(len(pairs), min(len(pairs), 24), replace=False)] if pairs else []
            if not pairs:
                continue
            P = np.array(pool.map(bisect, [(X[i], X[j], N[i], N[j], sg, labs[i]) for i, j in pairs]))
            B = rc.betas(P)
            B = B / B.sum(1, keepdims=True)
            # (the library surface through the boundary points)
            best = []
            for s in list(planes) + list(curved):
                r = np.array([abs(s.f(b)[0]) / (np.linalg.norm(s.grad(b)) + 1e-30) for b in B])
                best.append((np.median(r), (r < 1e-4).mean(), s))
            best.sort(key=lambda x: x[0])
            r_med, share, s = best[0]
            print(f"{w} section {k} side {sd} ({'/'.join(regions)}): {len(P)} boundary points; best surface {s.name} "
                  f"(median off {r_med:.1e}, {share:.0%} within 1e-4); next {best[1][2].name} ({best[1][0]:.1e})",
                  flush=True)
            if r_med < 1e-4 and share > 0.6:
                n_ = s.n.tolist() if isinstance(s, osd.Plane) else None
                out[w].append({"section": k, "side": sd, "regions": regions, "surface": s.name, "normal": n_,
                               "points": P.round(8).tolist(), "median_off": float(r_med)})
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
