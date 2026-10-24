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
    carry_map = keep
    import region_walls as rw
    d = rw.page_data()
    mesh_tree = {}
    for w_, k_, i_, T_ in rw.pieces_of(d, {w["id"] for w in d["xwalls"]} | {"T2"}):
        mesh_tree.setdefault(w_, []).append(T_)
    mesh_tree = {w_: cKDTree(sample_tris(np.concatenate(Ts), 0.002)) for w_, Ts in mesh_tree.items()}
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
        # (beside the section's own edge: a wall leaving that edge at a shallow angle, or a bent wall's next
        # patch, crossed by the stepped points; not a boundary on the section)
        from collections import Counter as C_
        key_ = lambda p: tuple(np.round(p, 7))
        cnt = C_()
        for t in T:
            for a_, b_ in ((0, 1), (1, 2), (2, 0)):
                cnt[frozenset((key_(t[a_]), key_(t[b_])))] += 1
        E = [np.array(list(e)) for e, k_ in cnt.items() if k_ == 1 and len(e) == 2]
        edge_d = float(np.median(cKDTree(np.concatenate([p + np.linspace(0, 1, 50)[:, None] * (q - p) for p, q in E]))
                                 .query(bpts)[0])) if len(bpts) and E else None
        # (the walls the surface carries, and how far the page's nearest mesh of each is from the boundary)
        carried_by, near = [], {}
        if fits and fits[0][0] < 2e-4:
            carried_by = sorted(carry_map.get(fits[0][1], []))
            for w_ in carried_by:
                if w_ in mesh_tree:
                    near[w_] = round(float(np.median(mesh_tree[w_].query(bpts)[0])), 4)
        rec = {"wall": x["wall"], "section": x["section"], "side": x["side"], "between": [a, b],
               "edge_distance": edge_d, "surface_carries": carried_by, "nearest_mesh": near,
               "counts": {k: int((L == k).sum()) for k in set(L)}, "boundary_points": bpts.tolist(),
               "best_surfaces": fits[:3]}
        out.append(rec)
        best = fits[0] if fits else None
        json.dump(out, open("boundary_points.json", "w"), indent=1)
        print(f"{x['wall']} s{x['section']} side {x['side']}: {a}|{b}, {len(bpts)} boundary points"
              + (f", {edge_d:.4f} from the section's edge" if edge_d is not None else "") + "; "
              + (f"on {best[1]} (median {best[0]:.1e}), carrying {carried_by}, page meshes at {near}" if best and best[0] < 2e-4
                 else f"NO known surface (nearest {best[1]}, {best[0]:.1e})" if best else "no boundary found"),
              flush=True)
    json.dump(out, open("boundary_points.json", "w"), indent=1)


if __name__ == "__main__":
    main()
