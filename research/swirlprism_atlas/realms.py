"""Realm walls: where the hull of the orbit stops changing smoothly.

Two seeds are corealmic when one can be moved to the other without the hull's combinatorics changing: every
cell keeps its vertices and the cells keep their neighbours, so the polytope only deforms. A realm ends where a
vertex reaches the hyperplane of a cell it is not on (five vertices become co-hyperplanar, so cells merge or
split), or on a ring, where vertices collide.

Inside a realm the cells are fixed sets of group elements (vertex k is g_k . seed), so the distance from a vertex
w to the hyperplane of a cell A is an exact, smooth function of the seed, with no hull to compute:
    m(A, w)(beta) = offset of A's hyperplane - its normal . g_w seed.
The wall is where the smallest such distance reaches zero. Step "points" takes golden samples, computes the hull
once, and moves each sample with Newton steps onto the nearest few walls of its own realm (checking that the
realm holds all the way there), then records the shape inside, on and beyond the wall.

    python realms.py points [max_samples]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from cellframe import seed_from_beta

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull

G = np.stack(group_elements(named_symmetry("h4_swirlprism")))
BASIS = np.linalg.svd(np.ones((1, 4)))[2][1:]          # directions that keep sum(beta)


def vertices(beta):
    return G @ seed_from_beta(beta)


def realm(beta):
    """The cells of the hull at beta as sets of group-element indices (None if vertices coincide)."""
    v = vertices(beta)
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    np.fill_diagonal(d, 1)
    if d.min() < 1e-6:
        return None
    faces, cells = compute_convex_hull(v, tol=1e-9)
    return [np.array(sorted({x for f in c for x in faces[f]})) for c in cells]


def plane(v, cell):
    P = v[cell]
    c = P.mean(axis=0)
    n = np.linalg.svd(P - c)[2][3]
    if n @ c < 0:
        n = -n
    return n, n @ c


def margins(beta, cells):
    """For each cell: distances (sorted) of the other vertices from its hyperplane; negative = outside."""
    v = vertices(beta)
    out = []
    for cell in cells:
        n, o = plane(v, cell)
        d = o - v @ n
        d[cell] = np.inf
        out.append(d)
    return out


def _classes(beta, cells):
    """One representative cell per symmetry class (cells with equal size and equal sorted distance profile)."""
    v = vertices(beta)
    reps, keys = [], set()
    for k, cell in enumerate(cells):
        n, o = plane(v, cell)
        d = np.sort(o - np.delete(v, cell, axis=0) @ n)[:6]
        key = (len(cell), tuple(np.round(d, 8)))
        if key not in keys:
            keys.add(key)
            reps.append(k)
    return reps


def _m(beta, cell, w):
    v = vertices(beta)
    n, o = plane(v, cell)
    return o - v[w] @ n


def _newton(beta, cell, w, steps=30):
    b = np.array(beta, float)
    for _ in range(steps):
        f = _m(b, cell, w)
        if abs(f) < 1e-14:
            return b
        g = np.array([(_m(b + 1e-7 * e, cell, w) - f) / 1e-7 for e in BASIS]) @ BASIS
        if g @ g < 1e-20:
            return None
        b = b - f * g / (g @ g)
        if b.min() < -1e-9:
            return None
    return b if abs(_m(b, cell, w)) < 1e-12 else None


def _sample(job):
    from probe import _one
    beta, own = job
    beta = np.asarray(beta, float) / sum(beta)
    cells = realm(beta)
    if cells is None:
        return []
    reps = _classes(beta, cells)
    rep_cells = [cells[k] for k in reps]           # symmetric cells have equal distances for every seed
    m0 = dict(zip(reps, margins(beta, rep_cells)))
    found = []
    for k in reps:
        order = np.argsort(m0[k])
        tried = set()
        for w in order[:12]:                                # the nearest few vertices to this cell's hyperplane
            key = round(float(m0[k][w]), 9)
            if key in tried or len(tried) >= 2:
                continue
            tried.add(key)
            b = _newton(beta, cells[k], int(w))
            if b is None or np.linalg.norm(b - beta) > 0.25:
                continue
            # the realm must hold all the way: no other vertex crosses a cell hyperplane first
            ok = True
            for t in np.linspace(0.1, 0.995, 10):
                p = beta + t * (b - beta)
                if min(m.min() for m in margins(p, rep_cells)) < -1e-12:
                    ok = False
                    break
            if not ok:
                continue
            # which side is which: step a little past the wall
            if any(abs(_m(b, f["_cell"], f["_w"])) < 1e-9 for f in found):
                continue                                    # on a wall already found (from another cell or vertex)
            found.append({"from": beta.tolist(), "wall": b, "inside": own, "cell_size": int(len(cells[k])),
                          "dist": float(np.linalg.norm(b - beta)), "_cell": cells[k], "_w": int(w)})
    for f in found:                                         # what lies on the wall and just beyond it
        b = f["wall"]
        past = b + 1e-4 * (b - beta) / np.linalg.norm(b - beta)
        f["on"] = _one(b)[0]
        f["beyond"] = _one(past)[0] if past.min() >= 0 else "outside"
        f["wall"] = b.tolist()
        f.pop("_cell")
        f.pop("_w")
    return found


def points(max_samples=None):
    from cell_atlas2 import identify, load_refs
    refs = load_refs()[0]
    xids = json.load(open("atlas_xids.json"))
    regions = {t for t, r in json.load(open("xloci_step1.json")).items() if r["kept"] == r["of"]}
    jobs = []
    for s in json.load(open("golden_22.json")):
        cid, xkey, _ = identify(s["sig"], refs)
        t = cid or xids.get(xkey)
        b = np.array(s["beta"], float)
        if t in regions and b.min() > 1e-9:
            jobs.append((b.tolist(), t))
    if max_samples:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in rng.choice(len(jobs), min(max_samples, len(jobs)), replace=False)]
    print(len(jobs), "region samples", flush=True)
    out = []
    with Pool(4) as pool:
        for k, res in enumerate(pool.imap_unordered(_sample, jobs, chunksize=2)):
            out += res
            if k % 50 == 0:
                print(k, "samples,", len(out), "wall points", flush=True)
                json.dump(out, open("realm_points.json", "w"))
    json.dump(out, open("realm_points.json", "w"))
    from collections import Counter
    for (a, o, c), n in Counter((r["inside"], r["on"], r["beyond"]) for r in out).most_common():
        print(f"{a:5s} | on {o:5s} | beyond {c:5s} : {n}")


def _components(B, eps):
    """Connected pieces: points closer than eps are joined."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(B).query_pairs(eps, output_type="ndarray")
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(B), len(B)))
    return connected_components(A, directed=False)[1]


def _monomials(B, deg):
    from itertools import combinations_with_replacement
    return np.stack([np.prod(B[:, list(c)], axis=1) for c in combinations_with_replacement(range(4), deg)], axis=1)


def fit_surface(B):
    """Lowest degree homogeneous surface through the points (beta normalised): (degree, coefficients, residual)."""
    B = B / B.sum(axis=1, keepdims=True)
    for deg in (1, 2, 3, 4):
        M = _monomials(B, deg)
        if len(B) < M.shape[1] + 3:
            return None
        _, s, vt = np.linalg.svd(M, full_matrices=False)
        if s[-1] / s[0] < 1e-9:
            return deg, vt[-1], float(s[-1] / s[0])
    return None


def walls(eps=0.06):
    """Group the wall points by (the two shapes, the shape on the wall), split each group into connected pieces,
    and fit each piece with the lowest-degree surface through it."""
    from collections import defaultdict
    P = json.load(open("realm_points.json"))
    groups = defaultdict(list)
    short = lambda t: t if not t.startswith("new:") else "new:" + t[4:].split(" | val")[0]
    for r in P:
        if r["beyond"] == "outside":
            continue
        key = (tuple(sorted([short(r["inside"]), short(r["beyond"])])), short(r["on"]))
        groups[key].append(r["wall"])
    out = []
    for (pair, on), pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        B = np.array(pts)
        lab = _components(B, eps)
        for c in np.unique(lab):
            Q = B[lab == c]
            f = fit_surface(Q) if len(Q) >= 6 else None
            out.append({"between": list(pair), "on": on, "points": Q.tolist(),
                        "degree": f and f[0], "coef": f and f[1].tolist(), "residual": f and f[2]})
    json.dump(out, open("realm_walls.json", "w"))
    from collections import Counter
    print(len(out), "pieces;", Counter(w["degree"] for w in out if len(w["points"]) >= 6))
    for w in out[:60]:
        print(len(w["points"]), w["between"], "on", w["on"][:28], "degree", w["degree"])


if __name__ == "__main__":
    if sys.argv[1] == "points":
        points(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if sys.argv[1] == "walls":
        walls()
