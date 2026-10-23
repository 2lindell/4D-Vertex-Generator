"""An example seed for a wall shape with the most even edges: the point of the wall (on its exact surface, still
classified as the shape) whose polychoron has the lowest ratio of longest to shortest edge. Points spread over the
wall's mesh are tried, the best few refined by a simplex search in the wall's surface.

    python best_seed.py X99          (prints the seed; writes it as the shape's sample in extra_samples.json)
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import region_cells as rc
import xsurface
from cellframe import seed_from_beta
from classify import ACTION
from edge_snap import Poly, newton
from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull
from probe import _one

TID = None
W = None


def ratio(b):
    """Longest over shortest hull edge of the polychoron from beta b (inf where it is not the shape)."""
    b = np.asarray(b, float)
    if (b < -1e-12).any():
        return np.inf
    b = b / b.sum()
    if _one(b)[0] != TID:
        return np.inf
    V = generate_vertices_from_seed(seed_from_beta(b), ACTION, tol=1e-9)
    faces, _ = compute_convex_hull(V, tol=1e-9)
    E = np.array([tuple(e) for e in hull_edges(faces)])
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(L.max() / L.min())


def _init(tid):
    global TID, W
    TID = tid
    s = xsurface.surface(tid)
    W = Poly(s["mons"], s["coeffs"], tid)


def refine(b0):
    U = np.linalg.svd(np.vstack([np.ones(4), W.grad(b0[None])[0]]))[2][2:]
    on = lambda y: newton([W], b0 + y @ U)
    res = minimize(lambda y: ratio(on(y)), np.zeros(2), method="Nelder-Mead",
                   options={"initial_simplex": np.array([[0, 0], [0.004, 0], [0, 0.004]]), "xatol": 1e-6,
                            "fatol": 1e-7, "maxiter": 200})
    b = on(res.x)
    return float(res.fun), (b / b.sum()).tolist()


def main(tid, n=600, top=6):
    _init(tid)
    sf = json.load(open("xsurfaces.json"))[tid]
    T = np.array(sf["tris"], float)
    T = T / T.sum(2, keepdims=True)
    V = np.unique(np.round(T.reshape(-1, 4), 12), axis=0)
    rng = np.random.default_rng(0)
    cand = V[rng.choice(len(V), min(n, len(V)), replace=False)]
    with Pool(4, initializer=_init, initargs=(tid,)) as pool:
        r = pool.map(ratio, list(cand))
        order = np.argsort(r)[:top]
        print("best of the spread points:", [round(r[i], 4) for i in order], flush=True)
        fine = pool.map(refine, [cand[i] for i in order])
    best = min(fine)
    b = np.array(best[1])
    print(f"{tid}: edge ratio {best[0]:.6f} at beta {b.round(8).tolist()}", flush=True)
    lab, sig = _one(b)
    assert lab == tid, lab
    e = json.load(open("extra_samples.json"))
    e = [x for x in e if not (x.get("sig") == sig or x.get("kind", "").startswith(f"{tid} "))]
    entry = {"beta": b.tolist(), "golden": False, "sig": sig, "hits": 1,
             "kind": f"{tid} example: the wall's most even-edged polychoron (longest/shortest edge {best[0]:.6f}, "
                     f"best_seed.py)"}
    raw = open("extra_samples.json").read()
    e.insert(0, entry)                      # (the first sample of a shape is its example seed)
    json.dump(e, open("extra_samples.json", "w"), indent=1 if raw.startswith("[\n") else None)


if __name__ == "__main__":
    main(sys.argv[1])
