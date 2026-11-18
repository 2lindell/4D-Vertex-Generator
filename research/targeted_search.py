"""Targeted crossing search: new walls, lines and points next to a sheet's known walls (crossing_search/targeted_*).

crossing_search.py tries every pair and triple of a sheet's surfaces from random starts anywhere in the domain and
labels every crossing point it finds; most of them lie inside one region, where nothing new can be. This search uses
the map instead. The map's walls (snub_map "walls" / "wall_points": a surface between two region types, with points
on it) are walked on their own surface:

  1. cells:  points sampled on the surface (projected from all over the domain and from near the wall's points) are
             grouped by their side of every other surface; each group is one cell of the surface, and one point per
             cell is labelled (a cell has one shape).
  2. walls:  starting from the cells holding the map's wall points, neighbouring cells (one other surface's side
             changed) are labelled in turn, going on while the cell is a wall (its key is not a region's) -- a wall
             piece whose key is unknown is a new wall.
  3. lines:  where two neighbouring cells of a walk meet, the crossing curve with the surface between them is
             labelled (one point, found by Newton's method on both surfaces) -- an unknown key is a new line.
  4. points: in each wall cell, the crossing points of two of its bounding surfaces are labelled the same way.

Known keys are those crossing_search.py uses (map types, wall shapes, strata, the family's merged shape lists).

    python targeted_search.py hyic ixo0 ixo1 ...        (writes crossing_search/targeted_<atlas>_<sheet>.json)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crossing_search import ATLAS, FAMILY, OUT  # noqa: E402

CODE = r'''
import json, os, sys, time, itertools, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, %(atlas)r)
import group as G
from snub_map import Surface
A = json.loads(sys.stdin.read())
M = json.load(open(G.fname("snub_map")))
TEXT = M["surfaces"]
S = [Surface(t) for t in TEXT]
NS = len(S)
FOLD = bool(getattr(G, "FOLD", False))
ZERO = 1e-9


def vec(fn, X):
    return np.broadcast_to(np.asarray(fn(*X.T), float), (len(X),))


def vals(i, X):
    return vec(S[i].f, X)


def grads(i, X):
    g = np.stack([vec(gi, X) for gi in S[i].g], 1)
    return g - g.mean(1, keepdims=True)


def inside(X):
    ok = X.min(1) > 1e-6
    if FOLD:
        ok &= X[:, 0] > X[:, 3] + 1e-6
    return ok


def project1(i, X, n=40):
    """Points X (rows) onto surface i, by Newton steps along its gradient (vectorised)."""
    X = X.copy()
    for _ in range(n):
        f = vals(i, X)
        g = grads(i, X)
        X -= (f / np.maximum((g * g).sum(1), 1e-300))[:, None] * g
        X /= X.sum(1, keepdims=True)
    f = np.abs(vals(i, X)) / (np.linalg.norm(grads(i, X), axis=1) + 1e-300)
    return X[(f < 1e-11) & inside(X) & np.isfinite(X).all(1)]


def newton(x, idx, n=60):
    """One point onto several surfaces at once (least-norm Newton steps, staying on sum = 1)."""
    x = np.array(x, float)
    for _ in range(n):
        F = np.array([S[i].val(x) for i in idx])
        J = np.array([S[i].grad(x) for i in idx])
        step = np.linalg.lstsq(J, F, rcond=None)[0]
        step -= step.mean()
        x = x - step
        x /= x.sum()
        if np.abs(step).max() < 1e-15:
            break
    if not np.isfinite(x).all() or max(S[i].dist(x) for i in idx) > 1e-10 or not inside(x[None])[0]:
        return None
    return x


def signs(X, on):
    """Each point's side of every surface (0 on it); `on` are the surfaces the points are on."""
    out = np.zeros((len(X), NS), np.int8)
    for t in range(NS):
        if t in on:
            continue
        d = vals(t, X) / (np.linalg.norm(grads(t, X), axis=1) + 1e-300)
        out[:, t] = np.where(np.abs(d) < ZERO, 0, np.sign(d)).astype(np.int8)
    return out


def lab(b, tol=1e-7):
    try:
        return G.label(np.asarray(b, float), tol)[1]
    except Exception:
        return None


def known():
    K = set(A["family_keys"])
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
    region_items = [(t["example"], 1e-9) for t in M["types"] if t["grid_regions"]]
    return K, items, region_items


def walk(s):
    """Cells of surface s next to its known wall pieces: their keys, the lines where they meet, their corners."""
    t0 = time.time()
    rng = np.random.default_rng(1000 + s)
    seeds = np.array([w["point"] for w in M["wall_points"] if w["surface"] == TEXT[s]], float).reshape(-1, 4)
    parts = [rng.dirichlet([1, 1, 1, 1], A["global"])]
    for x in seeds:
        for r in (1e-3, 1e-2, 5e-2, 2e-1):
            Y = np.abs(x + r * rng.standard_normal((A["local"], 4)))
            parts.append(Y / Y.sum(1, keepdims=True))
    X = project1(s, np.vstack(parts))
    Xs = project1(s, seeds) if len(seeds) else np.zeros((0, 4))
    if not len(X):
        return {"surface": s, "cells": 0, "labelled": 0, "found": [], "seconds": time.time() - t0}
    sg = signs(X, {s})
    cells = collections.defaultdict(list)
    for k, row in enumerate(map(bytes, sg)):
        cells[row].append(k)
    # a cell's point: the sample nearest the cell's mean (it is inside the cell)
    rep = {}
    for c, ks in cells.items():
        P = X[ks]
        rep[c] = P[np.argmin(((P - P.mean(0)) ** 2).sum(1))]
    # neighbouring cells: sample points of the two cells that are near neighbours on the surface (several surfaces
    # can pass through one crossing here, so a step between cells can change more than one side)
    from scipy.spatial import cKDTree
    cid = np.empty(len(X), object)
    for c, ks in cells.items():
        for k in ks:
            cid[k] = c
    tree = cKDTree(X)
    dist, nb = tree.query(X, k=min(10, len(X)))
    pair = {}
    for i in range(len(X)):
        for d, j in zip(dist[i][1:], nb[i][1:]):
            if cid[j] != cid[i]:
                tag = (min(cid[i], cid[j]), max(cid[i], cid[j]))
                if tag not in pair or d < pair[tag][0]:
                    pair[tag] = (d, i, j)
    key = {c: lab(rep[c]) for c in cells}          # every cell of the surface: one label each
    found = [{"kind": "wall", "key": key[c], "seed": rep[c].tolist(), "surfaces": [s]} for c in cells]
    walls = [c for c in cells if key[c] is not None and key[c] not in REGION]
    # lines: where two neighbouring cells meet (one of them a wall), on the surface(s) between them
    lines = {}
    bound = collections.defaultdict(set)
    for (c, d), (_, i, j) in pair.items():
        diff = [int(t) for t in np.nonzero(sg[i] != sg[j])[0]]
        bound[c].update(diff)
        bound[d].update(diff)
        if c not in walls and d not in walls:
            continue
        for t in diff:
            y = newton((X[i] + X[j]) / 2, [s, t])
            if y is None:
                continue
            row = bytes(signs(y[None], {s, t})[0])
            if row not in lines:
                lines[row] = y
                found.append({"kind": "line", "key": lab(y), "seed": y.tolist(), "surfaces": [s, t]})
            break
    # points: two bounding surfaces of a wall cell crossing on s
    for c in walls:
        pts = []
        for t, u in itertools.combinations(sorted(bound[c]), 2):
            for x0 in X[cells[c]][:: max(1, len(cells[c]) // 4)][:4]:
                y = newton(x0, [s, t, u])
                if y is not None and all(np.abs(y - p).max() > 1e-6 for p in pts):
                    pts.append(y)
                    found.append({"kind": "point", "key": lab(y), "seed": y.tolist(), "surfaces": [s, t, u]})
    return {"surface": s, "cells": len(cells), "wall_cells": len(walls), "lines": len(lines),
            "labelled": len(found), "found": found, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    pool = Pool(int(os.environ.get("NPROC", "4")))
    K, items, region_items = known()
    K |= {k for k in pool.starmap(lab, items) if k}
    REGION = {k for k in pool.starmap(lab, region_items) if k}
    wall_surfaces = list(range(NS))   # every surface: a wall piece the map never saw can be on any of them
    pool.close()
    pool = Pool(int(os.environ.get("NPROC", "4")))   # (the workers see REGION)
    R = pool.map(walk, wall_surfaces, chunksize=1)
    out = {"sheet": G.SUB, "surfaces": NS, "wall_surfaces": len(wall_surfaces), "known_keys": len(K),
           "labelled": sum(r["labelled"] for r in R),
           "walks": [{k: v for k, v in r.items() if k != "found"} for r in R], "candidates": {}}
    for r in R:
        for f in r["found"]:
            k = f["key"]
            if not k or k in K:
                continue
            c = out["candidates"].setdefault(k, {"kind": f["kind"], "seed": f["seed"],
                                                 "surfaces": [TEXT[i] for i in f["surfaces"]], "hits": 0})
            c["hits"] += 1
            if f["kind"] == "wall":
                c["kind"] = "wall"
    for k, c in out["candidates"].items():
        c["signature"] = G.label(np.asarray(c["seed"]), 1e-7)[0]
    print(json.dumps(out))
'''


def run(atlas, subs):
    adir, envvar = ATLAS[atlas]
    adir = os.path.join(HERE, adir)
    fam = []
    for f, groups in FAMILY[atlas]:
        if os.path.exists(os.path.join(HERE, f)):
            d = json.load(open(os.path.join(HERE, f)))
            for g in groups:
                fam += [s["key"] for s in d[g]]
    for sub in subs:
        dst = os.path.join(OUT, f"targeted_{atlas}_{sub}.json")
        strata = [os.path.join(HERE, "toxitic_atlas", f"strata_{sub}.json"), os.path.join(adir, f"strata_{sub}.json")]
        arg = {"family_keys": fam, "strata": strata, "global": int(os.environ.get("GLOBAL", "20000")),
               "local": int(os.environ.get("LOCAL", "300"))}
        p = subprocess.run([sys.executable, "-c", CODE % {"atlas": adir}], input=json.dumps(arg), capture_output=True,
                           text=True, cwd=adir, env={**os.environ, envvar: sub})
        if p.returncode:
            print(sub, "FAILED", p.stderr[-2500:], flush=True)
            continue
        o = json.loads(p.stdout.strip().splitlines()[-1])
        json.dump(o, open(dst, "w"), indent=1)
        secs = sum(w["seconds"] for w in o["walks"])
        print(f"{atlas} {sub}: {o['wall_surfaces']} wall surfaces walked, {o['labelled']} labels, {secs:.0f} s of walks, "
              f"{len(o['candidates'])} new: " + "; ".join(f"{k} {c['kind']}" for k, c in o["candidates"].items()),
              flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
