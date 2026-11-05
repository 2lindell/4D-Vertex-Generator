"""Is every kidecaic / iodecaic region closed off by recorded walls? (snub_closure_<sub>.json)

As tetraswirl_atlas/region_closure.py, over the curved-wall map. For each region type (wall_shapes_<sub>.json's regions,
and any the dense edge survey added), starting points are found in it (a Halton survey of the domain, spread out),
and from each a walk goes out along a fixed, evenly spread set of directions in steps of 0.01 until the polytope
changes or the domain ends; the change is bisected to ~1e-11 and there:
  * the domain's edge: a mirror (beta_k = 0, where the orbit is the decaic one) or the fold b1 = b4;
  * otherwise the surface through the point (the map's surfaces; if none, the forms of both sides' polytopes), the
    polytope a step beyond, and the polytope on the surface itself (the point put exactly on it, wall tolerance 1e-7).
An end is closed when the polytope beyond is a known region and the one on the surface is a known wall shape (or a known
region on one side: a change with no shape of its own). Otherwise: a region missing (unknown beyond), or a wall shape
missing (unknown on the surface), or a surface missing (none through the point).

    DECAIC_SUB=ki python snub_closure.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import sympy as sp
from scipy.stats import qmc

import group as G
from snub_map import Surface, forms_job

STEP, MAXLEN = 0.01, 1.5


def directions(n):
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1) @ U


def label(b, tol=1e-9):
    if isinstance(b, tuple) and len(b) == 2 and not np.isscalar(b[0]):
        b, tol = b
    try:
        return G.label(b, tol)[1]
    except Exception:
        return None


def walk(item):
    """From s along d: (the end point, the key inside, the key beyond, or 'domain')."""
    s, d, k0 = item
    s, d = np.asarray(s), np.asarray(d)
    prev = s
    t = STEP
    while t < MAXLEN:
        p = s + t * d
        if not G.in_domain(p[None])[0]:
            lo, hi = prev, p
            for _ in range(50):
                m = (lo + hi) / 2
                if G.in_domain(m[None])[0] and label(m) == k0:
                    lo = m
                else:
                    hi = m
            x = (lo + hi) / 2
            if not G.in_domain(hi[None])[0] and label(lo) == k0:
                return x.tolist(), "domain", None
            # (a change just before the edge)
            p = hi
        k = label(p)
        if k != k0:
            lo, hi, kb = prev, p, k
            for _ in range(36):
                m = (lo + hi) / 2
                km = label(m)
                if km == k0:
                    lo = m
                else:
                    hi = m
                    if km is not None:
                        kb = km
            x = (lo + hi) / 2
            beyond = x + 1e-6 * d
            kb2 = label(beyond)
            return x.tolist(), "change", kb2 if kb2 is not None else kb, beyond.tolist()
        prev = p
        t += STEP
    return s.tolist(), "not ended", None


def wall_key(item):
    surf_text, x = item
    S = Surface(surf_text)
    y = S.project(np.asarray(x))
    return y.tolist(), label(y, 1e-7)


def main():
    sub = G.SUB
    W = json.load(open(G.fname("wall_shapes")))["shapes"]
    M = json.load(open(G.fname("snub_map")))
    surf_text = set(M["surfaces"])
    regions = [(w["seed_beta"], w["signature"]) for w in W if w["kind"] == "region"]
    walls = [(w["seed_beta"], w["signature"]) for w in W if w["kind"] != "region"]
    try:
        es = json.load(open(G.fname("edge_survey")))["zones"]
        for z in es.values():
            for t in z["types"]:
                if t["known_region"] is None:
                    regions.append((t["example"], t["signature"]))
    except FileNotFoundError:
        pass
    pool = Pool(4)
    rkeys = pool.map(label, [r[0] for r in regions])
    # (one key per region type; extra survey examples of a known key dropped)
    reg = {}
    for (b, sg), k in zip(regions, rkeys):
        reg.setdefault(k, {"signature": sg, "example": b})
    # (wall shapes found by an earlier run of this check count as recorded)
    try:
        prev = json.load(open(G.fname("snub_closure")))["new_wall_shapes"]
    except (FileNotFoundError, KeyError):
        prev = {}
    walls += [(w["seed"], w["signature"]) for w in prev.values()]
    wkeys = set(pool.map(label, [(tuple(b), 1e-7) for b, _ in walls], chunksize=1)) if walls else set()
    print(f"{len(reg)} region types, {len(wkeys)} wall shapes known", flush=True)
    # (starting points: a Halton survey, labelled; up to 3 spread points per region, plus its example)
    X = qmc.Halton(4, seed=7).random(12000)
    X = -np.log(1 - X * 0.999999)
    X = X / X.sum(1, keepdims=True)
    X = X[X[:, 0] > X[:, 3]][:5000]
    K = pool.map(label, list(map(tuple, X)), chunksize=32)
    starts = defaultdict(list)
    for k, v in reg.items():
        starts[k].append(np.asarray(v["example"]) / np.sum(v["example"]))
    for x, k in zip(X, K):
        if k in reg:
            starts[k].append(x)
    for k in starts:
        P = np.array(starts[k])
        chosen = [P[0]]
        while len(chosen) < min(4, len(P)):
            dd = np.min([np.linalg.norm(P - q, axis=1) for q in chosen], axis=0)
            chosen.append(P[int(dd.argmax())])
        starts[k] = chosen
    D = directions(24)
    items = [(tuple(s), tuple(d), k) for k in reg for s in starts[k] for d in D]
    print(f"{len(items)} walks", flush=True)
    E = pool.map(walk, items, chunksize=4)
    # (the changes: surface, beyond, wall shape)
    surfs = [Surface(t) for t in sorted(surf_text)]
    ch = [(i, e) for i, e in enumerate(E) if e[1] == "change"]
    nosurf = []
    sid = {}
    for i, e in ch:
        x = np.asarray(e[0])
        d = np.array([s.dist(x) for s in surfs])
        if x.min() < 1e-6 or abs(x[0] - x[3]) < 1e-6:
            continue                                    # (the domain's edge)
        if d.min() < 1e-6:
            sid[i] = surfs[int(d.argmin())].text
        else:
            nosurf.append(i)
    if nosurf:
        more = pool.map(forms_job, [tuple(E[i][0]) for i in nosurf] + [tuple(E[i][3]) for i in nosurf], chunksize=1)
        new = set().union(*more) - surf_text
        surf_text |= new
        surfs = [Surface(t) for t in sorted(surf_text)]
        for i in nosurf:
            x = np.asarray(E[i][0])
            d = np.array([s.dist(x) for s in surfs])
            if d.min() < 1e-6:
                sid[i] = surfs[int(d.argmin())].text
        print(f"{len(nosurf)} changes on no map surface: {len(new)} surfaces added from their forms", flush=True)
    WK = pool.map(wall_key, [(sid[i], E[i][0]) for i, _ in ch if i in sid], chunksize=4)
    wk = dict(zip([i for i, _ in ch if i in sid], WK))
    pool.close()
    res = defaultdict(Counter)
    newwalls = {}
    detail = defaultdict(list)
    for i, e in enumerate(E):
        k0 = items[i][2]
        if e[1] == "domain":
            x = np.asarray(e[0])
            edge = "fold b1 = b4" if abs(x[0] - x[3]) < 1e-9 else f"mirror b{int(np.argmin(x)) + 1} = 0"
            res[k0][("domain edge", edge)] += 1
            continue
        if e[1] == "not ended":
            res[k0][("not ended", "")] += 1
            continue
        kb = e[2]
        x = np.asarray(e[0])
        if x.min() < 1e-6 or abs(x[0] - x[3]) < 1e-6:
            edge = "fold b1 = b4" if abs(x[0] - x[3]) < 1e-6 else f"mirror b{int(np.argmin(x)) + 1} = 0"
            res[k0][("domain edge", edge)] += 1
            continue
        if i not in sid:
            verdict = "surface missing"
        elif kb not in reg:
            verdict = "region missing beyond" if kb is not None else "unclassifiable beyond"
        else:
            y, kw = wk[i]
            if kw in wkeys:
                verdict = "recorded wall"
            elif kw in reg:
                verdict = "change with no wall shape of its own"
            else:
                verdict = "wall shape missing"
                newwalls.setdefault(kw, {"surface": sid[i], "sides": sorted({k0, kb}), "seed": y,
                                         "signature": G.label(y, 1e-7)[0]})
        res[k0][(verdict, sid.get(i, "")[:60])] += 1
        if verdict not in ("recorded wall", "change with no wall shape of its own"):
            detail[k0].append({"verdict": verdict, "at": e[0], "surface": sid.get(i), "beyond_key": kb})
    out = {"_about": __doc__.strip().split("\n\n")[0], "regions": {}, "new_wall_shapes": newwalls,
           "ends": [{"region": items[i][2], "end": list(e[:2]), "beyond": e[2] if len(e) > 2 else None,
                     "surface": sid.get(i), "wall": wk[i][1] if i in wk else None} for i, e in enumerate(E)]}
    for k, v in reg.items():
        bad = {f"{a} | {b}": n for (a, b), n in res[k].items() if a not in ("recorded wall", "domain edge",
                                                                           "change with no wall shape of its own")}
        out["regions"][k] = {"signature": v["signature"], "ends": {f"{a} | {b}": n for (a, b), n in res[k].items()},
                             "open": bad, "examples_open": detail[k][:5]}
        cells = v["signature"].split("|")[0]
        print(f"{k} {cells[:60]:60} " + ("CLOSED" if not bad else f"OPEN {bad}"), flush=True)
    for kw, w in newwalls.items():
        print(f"NEW WALL SHAPE {kw} between {w['sides']} on {w['surface'][:70]}  {w['signature'][:90]}", flush=True)
    json.dump(out, open(G.fname("snub_closure"), "w"), indent=1)


if __name__ == "__main__":
    main()
