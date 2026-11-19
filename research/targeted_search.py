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


# every surface as one table of monomial coefficients: one matrix product gives all their values and gradients
import sympy as _sp
_B = _sp.symbols("b1 b2 b3 b4")
_terms = [{m: float(c) for m, c in _sp.Poly(_sp.sympify(t), *_B).terms()} for t in TEXT]
_EXP = sorted(set().union(*_terms))
E = np.array(_EXP, float)                       # monomials (m x 4)
C = np.zeros((NS, len(_EXP)))
for _i, _tm in enumerate(_terms):
    for _m, _c in _tm.items():
        C[_i, _EXP.index(_m)] = _c
_ED = []
for _v in range(4):
    _e = E.copy()
    _e[:, _v] = np.maximum(_e[:, _v] - 1, 0)
    _ED.append(_e)


def mono(X):
    return np.prod(X[:, None, :] ** E[None], axis=2)


def dmono(X):
    """d(monomial)/d(b_v): (4, n, m)."""
    return np.stack([E[:, v] * np.prod(X[:, None, :] ** _ED[v][None], axis=2) for v in range(4)])


def _proj(g):                                   # (along beta summing to 1)
    return g - g.mean(-1, keepdims=True)


def vals(i, X):
    return mono(X) @ C[i]


def grads(i, X):
    return _proj((dmono(X) @ C[i]).T)


def inside(X):
    ok = X.min(1) > 1e-6
    if FOLD:
        ok &= X[:, 0] > X[:, 3] + 1e-6
    return ok


def project1(i, X, n=40):
    """Points X (rows) onto surface i, by Newton steps along its gradient (vectorised)."""
    X = np.array(X, float)
    for _ in range(n):
        f = vals(i, X)
        g = grads(i, X)
        X -= (f / np.maximum((g * g).sum(1), 1e-300))[:, None] * g
        X /= X.sum(1, keepdims=True)
    f = np.abs(vals(i, X)) / (np.linalg.norm(grads(i, X), axis=1) + 1e-300)
    return X[(f < 1e-11) & inside(X) & np.isfinite(X).all(1)]


def newton_pairs(X, s, T, n=50):
    """Rows X[r] onto surfaces s and T[r] at once (least-norm Newton steps, on sum = 1). Rows that do not converge,
    or land outside the domain, come back as nan."""
    X = np.array(X, float)
    Cs, Ct = C[s], C[T]
    live = np.ones(len(X), bool)
    for it in range(n):
        Xl = X[live]
        if not len(Xl):
            break
        mo, dm = mono(Xl), dmono(Xl)
        F = np.stack([mo @ Cs, (mo * Ct[live]).sum(1)], 1)
        J = np.stack([_proj((dm @ Cs).T), _proj(np.einsum("vnm,nm->nv", dm, Ct[live]))], 1)
        step = np.einsum("nij,nj->ni", np.linalg.pinv(J), F)
        step = _proj(step)
        Xl = Xl - step
        Xl /= Xl.sum(1, keepdims=True)
        X[live] = Xl
        big = np.abs(step).max(1)
        bad = ~np.isfinite(Xl).all(1) | (np.abs(Xl).max(1) > 10) | ((it > 12) & (big > 1e-3))
        X[np.nonzero(live)[0][bad]] = np.nan
        live[np.nonzero(live)[0][bad | (big < 1e-15)]] = False
    ok = np.isfinite(X).all(1)
    Y = np.where(ok[:, None], X, 0.25)
    mo, dm = mono(Y), dmono(Y)
    ds = np.abs(mo @ Cs) / (np.linalg.norm(_proj((dm @ Cs).T), axis=1) + 1e-300)
    dt = np.abs((mo * Ct).sum(1)) / (np.linalg.norm(_proj(np.einsum("vnm,nm->nv", dm, Ct)), axis=1) + 1e-300)
    ok &= (ds < 1e-10) & (dt < 1e-10) & inside(Y)
    X[~ok] = np.nan
    return X


def signs(X, on):
    """Each point's side of every surface (0 on it); `on` are the surfaces the points are on."""
    X = np.asarray(X, float)
    V = mono(X) @ C.T
    G = np.linalg.norm(_proj(np.moveaxis(dmono(X) @ C.T, 0, -1)), axis=-1)
    D = V / (G + 1e-300)
    out = np.where(np.abs(D) < ZERO, 0, np.sign(D)).astype(np.int8)
    out[:, sorted(on)] = 0
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


def walk(job):
    """Wall cells of surface s, walked out from seed points on it: each cell's key, the lines on its edges, the cells
    across them; the walk goes on while the cell is a wall (its key is not a known region's)."""
    s, seeds, visited = job
    t0 = time.time()
    visited = set(bytes.fromhex(v) for v in visited)
    found, lines_out = [], []
    queue = []
    for x in project1(s, np.array(seeds, float).reshape(-1, 4)):
        queue.append(x)
    labelled = nlines = ncells = 0
    seen_lines = set()
    while queue:
        x = queue.pop(0)
        row = bytes(signs(x[None], {s})[0])
        if row in visited:
            continue
        visited.add(row)
        ncells += 1
        k = lab(x)
        labelled += 1
        found.append({"kind": "wall", "key": k, "seed": x.tolist(), "surfaces": [s]})
        if k is None or k in REGION:
            continue                            # not a wall here: the walk stops
        # the cell's points: a few, spread over it
        rng = np.random.default_rng(len(visited))
        pts = [x]
        for r in (1e-3, 1e-2, 5e-2):
            Y = project1(s, np.abs(x + r * rng.standard_normal((40, 4))) / 1.0)
            if len(Y):
                Y /= Y.sum(1, keepdims=True)
                Y = project1(s, Y)
                ok = [y for y, rr in zip(Y, signs(Y, {s})) if bytes(rr) == row]
                pts += ok[:2]
        me = np.frombuffer(row, np.int8)
        # its edges: Newton onto s and every other surface t at once, from each of the cell's points; per t, the first
        # landing point on this cell's edge
        T = np.array([t for t in range(NS) if t != s and me[t] != 0])
        if not len(T):
            continue
        Xs = np.repeat(np.array(pts), len(T), 0)
        Ts = np.tile(T, len(pts))
        Y = newton_pairs(Xs, s, Ts)
        good = np.isfinite(Y).all(1)
        RY = signs(np.where(good[:, None], Y, 0.25), {s})
        RY[np.arange(len(Ts)), Ts] = 0
        mism = ((RY != 0) & (me != 0)[None] & (RY != me[None])).any(1)
        good &= ~mism
        first = {}
        for r in np.nonzero(good)[0]:
            first.setdefault(int(Ts[r]), r)
        for t, r in sorted(first.items()):
            y = Y[r]
            lrow = bytes(RY[r])
            if lrow not in seen_lines:
                seen_lines.add(lrow)
                nlines += 1
                kl = lab(y)
                labelled += 1
                found.append({"kind": "line", "key": kl, "seed": y.tolist(), "surfaces": [s, t]})
                lines_out.append([t, y.tolist()])
            # the cell across t, on s
            gs, gt = grads(s, y[None])[0], grads(t, y[None])[0]
            d = gt - (gt @ gs) / (gs @ gs) * gs
            d -= d.mean()
            if np.linalg.norm(d) < 1e-12:
                continue
            d /= np.linalg.norm(d)
            d = d if me[t] < 0 else -d          # towards t's other side
            z = project1(s, (y + 1e-4 * d)[None])
            if len(z):
                queue.append(z[0])
    return {"surface": s, "cells": ncells, "lines": nlines, "labelled": labelled, "found": found,
            "lines_out": lines_out, "visited": [v.hex() for v in visited], "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    pool = Pool(int(os.environ.get("NPROC", "4")))
    K, items, region_items = known()
    K |= {k for k in pool.starmap(lab, items) if k}
    REGION = {k for k in pool.starmap(lab, region_items) if k}
    for f in A["family_files"]:
        try:
            REGION |= {x["key"] for x in json.load(open(f))["shapes"] if x.get("kind") == "region"}
        except FileNotFoundError:
            pass
    pool.close()
    pool = Pool(int(os.environ.get("NPROC", "4")))   # (forked after REGION, so the walks see it)
    seeds = collections.defaultdict(list)
    for w in M["wall_points"]:
        if w["surface"] in TEXT:
            seeds[TEXT.index(w["surface"])].append(w["point"])
    try:
        for w in json.load(open(G.fname("wall_shapes")))["shapes"]:
            for t in w.get("surfaces", []):
                if t in TEXT:
                    seeds[TEXT.index(t)].append(w["seed_beta"])
    except FileNotFoundError:
        pass
    visited = collections.defaultdict(list)
    R = []
    cache = A.get("cache")                      # each finished walk is saved there: a restart picks up from it
    for rnd in range(40):                       # later rounds: surfaces reached across a line from another's walk
        jobs = [(s, seeds[s], visited[s]) for s in sorted(seeds) if seeds[s]]
        if not jobs:
            break
        print(json.dumps({"round": rnd, "surfaces": len(jobs)}), file=sys.stderr, flush=True)
        path = os.path.join(cache, f"round{rnd:02d}.jsonl") if cache else None
        done = {}
        if path and os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                    done[r["surface"]] = r
                except ValueError:
                    pass                        # (a line cut off by the restart)
        todo = [j for j in jobs if j[0] not in done]
        for r in pool.imap_unordered(walk, todo):
            done[r["surface"]] = r
            if path:
                with open(path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            print(json.dumps({k: v for k, v in r.items() if k not in ("found", "lines_out", "visited")}),
                  file=sys.stderr, flush=True)
        seeds = collections.defaultdict(list)
        for s_ in sorted(done):                 # (in surface order, so a resumed run seeds the same way)
            r = done[s_]
            R.append(r)
            visited[r["surface"]] = r["visited"]
            for t, y in r["lines_out"]:
                seeds[t].append(y)
        # (a seed on a cell its surface has walked already is skipped inside walk)
    wall_surfaces = sorted({r["surface"] for r in R})
    out = {"sheet": G.SUB, "surfaces": NS, "wall_surfaces": len(wall_surfaces), "known_keys": len(K),
           "labelled": sum(r["labelled"] for r in R),
           "walks": [{k: v for k, v in r.items() if k not in ("found", "lines_out", "visited")} for r in R], "candidates": {}}
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
        cache = os.path.join(HERE, ".crossing_cache", f"targeted_{atlas}_{sub}")
        os.makedirs(cache, exist_ok=True)
        arg = {"cache": cache, "family_keys": fam, "family_files": [os.path.join(HERE, f) for f, _ in FAMILY[atlas]], "strata": strata, "global": int(os.environ.get("GLOBAL", "20000")),
               "local": int(os.environ.get("LOCAL", "300"))}
        with open(dst.replace(".json", ".log"), "w") as log:
            p = subprocess.run([sys.executable, "-c", CODE % {"atlas": adir}], input=json.dumps(arg),
                               stdout=subprocess.PIPE, stderr=log, text=True, cwd=adir, env={**os.environ, envvar: sub})
        if p.returncode:
            print(sub, "FAILED (see the .log)", flush=True)
            continue
        o = json.loads(p.stdout.strip().splitlines()[-1])
        json.dump(o, open(dst, "w"), indent=1)
        __import__("shutil").rmtree(cache, ignore_errors=True)
        secs = sum(w["seconds"] for w in o["walks"])
        print(f"{atlas} {sub}: {o['wall_surfaces']} wall surfaces walked, {o['labelled']} labels, {secs:.0f} s of walks, "
              f"{len(o['candidates'])} new: " + "; ".join(f"{k} {c['kind']}" for k, c in o["candidates"].items()),
              flush=True)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2:])
