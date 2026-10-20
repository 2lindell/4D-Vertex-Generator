"""Check each wall surface of region_forms_oct.json by the classifier, and find what lies on it.

From each survey sample towards a neighbour of another piece, the first of its piece's conditions to fail names the
wall crossed (region_forms.py); here the crossing is solved exactly on that surface (bisection on its factor), and the
seeds a short step either side along the surface's normal, and on the surface itself, are classified. A surface is a
wall where the two sides are different shapes; what is on it is the wall's own polychoron.

    SWIRL_GROUP=oct python walls_verify.py          (writes walls_verified_oct.json)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

import domain as D
from region_forms import B_SYM, evaluate

STEP = 2e-4                                  # (in beta, summing to 1)


def label(beta):
    b = np.asarray(beta, float)
    if b.min() < 0:
        return "outside"
    try:
        return D.signature(D.classify(D.seed_from_chart(D.chart_from_seed(b @ D.CELL))))
    except Exception:
        return "ERR"


def main(per_surface=12):
    d = json.load(open(f"region_forms_{D.KEY}.json"))
    P = d["pieces"]
    names = list(P)
    conds = {p: np.array(v["conditions"]) for p, v in P.items()}
    ids = json.load(open(f"ids_{D.KEY}.json"))
    s = json.load(open(f"survey_{D.KEY}.json"))
    U = np.array(s["points"])
    Bs = D.beta_from_seed(D.seed_from_chart(U))
    piece = np.array([next((p for p in names if (evaluate(conds[p].T, b[None]) > 0).all()), None) for b in Bs], dtype=object)
    surfaces = [w["surface"] for w in d["walls"]]
    fns = {f: sp.lambdify(B_SYM, sp.sympify(f), "numpy") for f in surfaces}
    grads = {f: sp.lambdify(B_SYM, [sp.diff(sp.sympify(f), v) for v in B_SYM], "numpy") for f in surfaces}
    pairs = cKDTree(U).query_pairs(1.01 * s["h"], output_type="ndarray")
    pairs = pairs[[piece[a] is not None and piece[b] is not None and P[piece[a]]["type"] != P[piece[b]]["type"] for a, b in pairs]]
    t = np.linspace(0, 1, 801)[1:]
    hits = defaultdict(list)                  # surface -> [(point on it, normal, own piece, far piece)]
    for a, b in np.vstack([pairs, pairs[:, ::-1]]):
        pa = piece[a]
        path = (1 - t)[:, None] * Bs[a] + t[:, None] * Bs[b]
        V = evaluate(conds[pa].T, path)
        bad = (V <= 0).any(1)
        if not bad.any():
            continue
        i = int(np.argmax(bad))
        lo, hi = (Bs[a] if i == 0 else path[i - 1]), path[i]
        for f in surfaces:
            fn = fns[f]
            if np.sign(fn(*lo)) == np.sign(fn(*hi)):
                continue
            x0, x1 = lo.copy(), hi.copy()
            for _ in range(50):                 # (the crossing, on the surface itself)
                m = (x0 + x1) / 2
                if np.sign(fn(*m)) == np.sign(fn(*x0)):
                    x0 = m
                else:
                    x1 = m
            m = (x0 + x1) / 2
            g = np.array(grads[f](*m), float)
            g -= g.mean()                       # (within the plane of weights summing to 1)
            g /= np.linalg.norm(g)
            if np.sign(fn(*(m + 1e-6 * g))) != np.sign(fn(*Bs[b])):
                g = -g                          # (pointing to the far side)
            hits[f].append((m / m.sum(), g, pa, piece[b]))
    jobs, keys = [], []
    rng = np.random.default_rng(0)
    for f, hs in hits.items():
        for k in rng.permutation(len(hs))[:per_surface]:
            m, g, pa, pb = hs[k]
            for side, x in (("near", m - STEP * g), ("on", m), ("far", m + STEP * g)):
                jobs.append(x)
                keys.append((f, int(k), side, pa, pb))
    print(f"{len(hits)} surfaces crossed; classifying {len(jobs)} seeds", flush=True)
    with Pool(4) as pool:
        labs = pool.map(label, jobs, chunksize=4)
    inv = {v: k for k, v in ids.items()}
    name = lambda sg: ids.get(sg, "on-wall" if sg not in ("ERR", "outside", "none") else sg)
    res = defaultdict(lambda: defaultdict(dict))
    for (f, k, side, pa, pb), l, x in zip(keys, labs, jobs):
        res[f][k][side] = l
        res[f][k]["pieces"] = (pa, pb)
        if side == "on":
            res[f][k]["beta"] = np.round(x, 7).tolist()
    out = []
    for f in surfaces:
        r = res.get(f, {})
        rows = list(r.values())
        differ = [x for x in rows if x.get("near") != x.get("far")]
        pairs_ = Counter("|".join(sorted([name(x["near"]), name(x["far"])])) for x in differ)
        expect = Counter("|".join(sorted([P[x["pieces"][0]]["type"], P[x["pieces"][1]]["type"]])) for x in rows)
        on = Counter(x["on"] for x in differ)
        verdict = "wall" if rows and len(differ) >= max(1, 0.5 * len(rows)) else ("not a wall" if rows else "not crossed")
        out.append({"surface": f, "verdict": verdict, "checked": len(rows), "sides_differ": len(differ),
                    "separates": dict(pairs_), "expected": dict(expect), "crossings": len(hits.get(f, [])),
                    "on_wall": [{"sig": sg, "count": n, "known": ids.get(sg)} for sg, n in on.most_common()],
                    "points": [x["beta"] for x in differ][:6]})
        print(f"{verdict:12s} {len(differ):2d}/{len(rows):2d}  {', '.join(pairs_)[:50]:50s} {f[:60]}", flush=True)
    json.dump({"group": D.NAME, "domain": D.DOMAIN, "walls": out}, open(f"walls_verified_{D.KEY}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
