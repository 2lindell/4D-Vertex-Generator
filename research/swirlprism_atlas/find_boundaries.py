"""Where a wall section's side changes region with no line on the section to cut it (region_check.json "mixed"):
the boundary itself located and identified. Points a little off the section on that side, over all of it, are
classified; between neighbouring points of different regions the boundary is bisected; the boundary points are then
tested on every known surface (the walls' surfaces and their images, the cell's faces and mirror). A surface they all
lie on is the wall the cutter missed there; none is a wall not yet known (a new shape).

    python find_boundaries.py         (writes boundary_points.json; prints a line per mixed side)
"""
from __future__ import annotations

import gzip
import json
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import region_cells as rc
from cut_walls import sample_tris
from region_db import home

OFF = 8e-4


def lab(b):
    return rc._lab_job(b) if not np.isnan(b).any() else "OUT"


def labs(X):
    with Pool(4) as p:
        return p.map(lab, [home(rc.betas(x[None]))[0] for x in X], chunksize=16)


def main():
    c = json.load(open("region_check.json"))
    from arrangement import DOMAIN, carriers, surfaces
    S = surfaces()
    keep = carriers(S)
    S = [s for s in S if s.name in keep] + DOMAIN
    regions = set(c["regions"])
    import os
    out = json.load(open("boundary_points.json")) if os.path.exists("boundary_points.json") else []
    done = {(r["wall"], r["section"], r["side"]) for r in out}
    for x in c["mixed"]:
        if x["area"] < 3e-4 or len(x.get("between", [])) < 2 or (x["wall"], x["section"], x["side"]) in done:
            continue
        f = json.load(gzip.open(f"sections/{x['wall']}.json.gz", "rt"))[x["section"]]
        T = np.array(f["tris"]).reshape(-1, 3, 3)
        X, own = sample_tris(T, 0.004, owner=True)
        N = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-300)
        sg = 1 if x["side"] == 0 else -1
        Y = X + sg * OFF * N[own]
        L = np.array(labs(Y))
        a, b = x["between"]
        pairs = [(i, j) for i, j in cKDTree(Y).query_pairs(0.0065) if {L[i], L[j]} == {a, b}]
        pairs = pairs[:24]
        p_ = np.array([Y[i] for i, j in pairs]).reshape(-1, 3)
        q_ = np.array([Y[j] for i, j in pairs]).reshape(-1, 3)
        lp = [L[i] for i, j in pairs]
        for _ in range(10):                           # (all the pairs bisected together)
            m_ = (p_ + q_) / 2
            same = np.array(labs(m_)) == np.array(lp) if len(m_) else np.zeros(0, bool)
            p_ = np.where(same[:, None], m_, p_)
            q_ = np.where(same[:, None], q_, m_)
        bpts = (p_ + q_) / 2
        fits = []
        if len(bpts):
            for s in S:
                r = np.abs(s.at(bpts)) / (np.linalg.norm(s.grad(bpts), axis=1) + 1e-30)
                fits.append((float(np.median(r)), s.name))
            fits.sort()
        rec = {"wall": x["wall"], "section": x["section"], "side": x["side"], "between": [a, b],
               "counts": {k: int((L == k).sum()) for k in set(L)}, "boundary_points": bpts.tolist(),
               "best_surfaces": fits[:3]}
        out.append(rec)
        best = fits[0] if fits else None
        json.dump(out, open("boundary_points.json", "w"), indent=1)
        print(f"{x['wall']} s{x['section']} side {x['side']}: {a}|{b}, {len(bpts)} boundary points; "
              + (f"nearest surface {best[1]} (median {best[0]:.1e})" if best else "no boundary found")
              + ("  -> a known surface" if best and best[0] < 2e-4 else "  -> NO known surface" if best else ""),
              flush=True)
    json.dump(out, open("boundary_points.json", "w"), indent=1)


if __name__ == "__main__":
    main()
