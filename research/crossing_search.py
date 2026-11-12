"""Shapes where a sheet's interior walls cross: lines on two walls at once, points on three (crossing_search/).

The atlas maps take lines and points only on the chamber's own strata (its edges, corners, fold lines); a shape that
lives where two curved (or plane) walls of the map cross inside the chamber was never sampled -- oxitic_line_01, on
b2 = b3 and a quadric, was found by the community that way. For every pair of the sheet's wall surfaces (snub_map's
"surfaces"), points of their crossing curve inside the domain (betas positive, off the chamber's boundary; the fold
kept where the group has one) are found by alternating projections from random starts, and the polytope there is
classified (wall tolerance). A key not among the sheet's known polytopes (map regions, wall shapes, strata, the
family's merged shape lists) is a candidate: stepped 1e-4 along the curve (both ways) it is a line if it stays, else
an isolated point on the curve. Triples of surfaces are searched the same way for points.

    python crossing_search.py contic ox oxb tox icp icpb ki io | decaic ki io | hyic ix0 ... ix9    (one atlas per run)
    python crossing_search.py summary
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crossing_search")
ATLAS = {"contic": ("contic_atlas", "CONTIC_SUB"), "decaic": ("decaic_atlas", "DECAIC_SUB"),
         "hyic": ("hyic_atlas", "HYIC_SUB")}
FAMILY = {"contic": [("toxitic_atlas/toxitic_shapes.json", ("toxitic", "oxitic"))],
          "hyic": [("hyic_atlas/ixitic_shapes.json", ("shapes",))], "decaic": []}

CODE = r'''
import json, os, sys, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, %(atlas)r)
import group as G
from snub_map import Surface
A = json.loads(sys.stdin.read())
rng = np.random.default_rng(11)
M = json.load(open(G.fname("snub_map")))
TEXT = M["surfaces"]
S = [Surface(t) for t in TEXT]
FOLD = bool(getattr(G, "FOLD", False))
STARTS, TRIPLE_STARTS = A["starts"], A["triple_starts"]


def lab(b, tol=1e-7):
    try:
        return G.label(np.asarray(b, float), tol)[1]
    except Exception:
        return None


def inside(x):
    return x.min() > 1e-6 and (not FOLD or x[0] > x[3] + 1e-6)


def onto(x, idx, n=80):
    for _ in range(n):
        for i in idx:
            x = S[i].project(x)
            x = x / x.sum()
    return x if max(S[i].dist(x) for i in idx) < 1e-10 else None


def starts(n):
    X = rng.dirichlet([1, 1, 1, 1], n)
    if FOLD:
        X[:, [0, 3]] = np.sort(X[:, [0, 3]], 1)[:, ::-1]
    return X


def find(idx):
    pts = []
    for x in starts(STARTS if len(idx) == 2 else TRIPLE_STARTS):
        y = onto(x, idx)
        if y is not None and inside(y) and all(np.abs(y - p).max() > 1e-4 for p in pts):
            pts.append(y)
    return [(list(idx), p.tolist()) for p in pts]


def known():
    K = set()
    items = [(t["example"], 1e-9 if t["grid_regions"] else 1e-7) for t in M["types"]]
    try:
        items += [(w["seed_beta"], 1e-7) for w in json.load(open(G.fname("wall_shapes")))["shapes"]]
    except FileNotFoundError:
        pass
    for f in A["strata"]:
        try:
            K |= {t["key"] for t in json.load(open(f))["types"]}
        except FileNotFoundError:
            pass
    return K, items


def classify(item):
    idx, p = item
    return idx, p, lab(p)


def along(item):
    idx, p, k = item
    x = np.asarray(p)
    C = np.array([np.ones(4)] + [S[i].grad(x) for i in idx])
    t = np.linalg.svd(C)[2][-1]
    kept = []
    for e in (1e-4, -1e-4):
        y = onto(x + e * t, idx)
        kept.append(y is not None and lab(y) == k)
    # (off the curve within each wall: a line is lost both ways on both walls; a shape that stays on one wall fills it)
    walls = []
    for a, b in ((idx[0], idx[1]), (idx[1], idx[0])):
        g = S[b].grad(x)
        ga = S[a].grad(x)
        d = g - (g @ ga) / (ga @ ga) * ga
        d -= d.mean()
        d /= np.linalg.norm(d)
        stay = []
        for e in (1e-4, -1e-4):
            y = onto(x + e * d, [a])
            stay.append(y is not None and lab(y) == k)
        if all(stay):
            walls.append(TEXT[a])
    return idx, p, k, kept, walls


if __name__ == "__main__":
    pool = Pool(int(os.environ.get("NPROC", "3")))
    K, items = known()
    K |= set(A["family_keys"])
    K |= {k for k in pool.starmap(lab, items) if k}
    n = len(S)
    pairs = list(itertools.combinations(range(n), 2))
    triples = list(itertools.combinations(range(n), 3)) if A["triples"] else []
    found = [x for r in pool.map(find, pairs + triples, chunksize=8) for x in r]
    C = pool.map(classify, found, chunksize=8)
    cand = [c for c in C if c[2] and c[2] not in K]
    V = pool.map(along, [c for c in cand if len(c[0]) == 2], chunksize=1)
    out = {"sheet": G.SUB, "surfaces": n, "pairs": len(pairs), "triples": len(triples), "crossing_points": len(found),
           "known_keys": len(K), "keys_seen": dict(__import__("collections").Counter(str(c[2]) for c in C)),
           "candidates": {}}
    for idx, p, k, kept, walls in V:
        kind = ("wall" if walls else "line") if all(kept) else "point"
        c = out["candidates"].setdefault(k, {"kind": kind, "seed": p, "surfaces": walls or [TEXT[i] for i in idx],
                                             "hits": 0})
        c["hits"] += 1
    for idx, p, k in cand:
        if len(idx) == 3:
            c = out["candidates"].setdefault(k, {"kind": "point", "seed": p, "surfaces": [TEXT[i] for i in idx],
                                                 "hits": 0})
            c["hits"] += 1
    for k, c in out["candidates"].items():
        sg = G.label(np.asarray(c["seed"]), 1e-7)[0]
        c["signature"] = sg
    print(json.dumps(out))
'''


def run(atlas, subs):
    adir, envvar = ATLAS[atlas]
    adir = os.path.join(HERE, adir)
    os.makedirs(OUT, exist_ok=True)
    fam = []
    for f, groups in FAMILY[atlas]:
        d = json.load(open(os.path.join(HERE, f)))
        for g in groups:
            fam += [s["key"] for s in d[g]]
    for sub in subs:
        dst = os.path.join(OUT, f"{atlas}_{sub}.json")
        if os.path.exists(dst):
            print(sub, "done already", flush=True)
            continue
        strata = [os.path.join(HERE, "toxitic_atlas", f"strata_{sub}.json"), os.path.join(adir, f"strata_{sub}.json")]
        arg = {"family_keys": fam, "strata": strata, "starts": 6, "triple_starts": 3,
               "triples": os.environ.get("TRIPLES", "1") == "1"}
        p = subprocess.run([sys.executable, "-c", CODE % {"atlas": adir}], input=json.dumps(arg), capture_output=True,
                           text=True, cwd=adir, env={**os.environ, envvar: sub})
        if p.returncode:
            print(sub, "FAILED", p.stderr[-1500:], flush=True)
            continue
        o = json.loads(p.stdout.strip().splitlines()[-1])
        json.dump(o, open(dst, "w"), indent=1)
        print(f"{atlas} {sub}: {o['surfaces']} surfaces, {o['crossing_points']} crossing points, "
              f"{len(o['candidates'])} new: " + "; ".join(f"{c['kind']} {c['signature'].split('|')[0][:60]}"
                                                       for c in o["candidates"].values()), flush=True)


def summary():
    L = ["# Crossing search: shapes where interior walls cross", "", __doc__.strip().split("\n\n")[1], "",
         "| atlas | sheet | surfaces | crossing points | new key | kind | seed | on | polytope |", "|---|---|---|---|---|---|---|---|---|"]
    for f in sorted(os.listdir(OUT)):
        if not f.endswith(".json"):
            continue
        o = json.load(open(os.path.join(OUT, f)))
        atlas = f.split("_")[0]
        if not o["candidates"]:
            L.append(f"| {atlas} | {o['sheet']} | {o['surfaces']} | {o['crossing_points']} | none |  |  |  |  |")
        for k, c in o["candidates"].items():
            L.append(f"| {atlas} | {o['sheet']} | {o['surfaces']} | {o['crossing_points']} | {k} | {c['kind']} | "
                     f"{[round(v, 6) for v in c['seed']]} | {' ; '.join(c['surfaces'])[:120]} | {c['signature'].split('|')[0]} |")
    open(os.path.join(OUT, "README.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(sys.argv[1], sys.argv[2:])
