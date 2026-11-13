"""The ixoic wall points that ixoic_walls.py left without a surface: find their walls' equations.

ixoic_walls.py takes the forms of each region's own polytope (two neighbouring cells becoming one 3-flat). A wall can
also be where a 3-flat forms that is not a cell on either side (a flip: a set of vertices passes through coplanarity
and is cut differently on the two sides). At such a wall point the polytope itself (hull at a loose tolerance, the
point being only a bisected one) has that flat as a cell with five or more vertices; for every such cell, four of its
vertices spanning it and each further vertex give a quartic (forms.form) that vanishes there. Nonzero ones vanishing at
the point (relative to their gradient, at the point's precision) are factored, and each factor is checked: the point
moved onto it (one Newton projection) and stepped 1e-5 to each side must give two different polytopes (not
necessarily the map's recorded sides: many such points were bisected across regions the grid missed, found later by the
survey, and lie on those regions' walls). Confirmed
surfaces are added to the sheet's map (surfaces, wall points' "surface"), and the near misses -- points within 1e-4 of an
existing surface that pass the same two-sided check -- are assigned too. Points are searched in rounds, and each round's new surfaces are tried on the points still
waiting first (most walls carry several points). Report: missing_walls_<sheet>.json.

    python missing_walls.py [ixo0 ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import json, os, sys, time
import numpy as np, sympy as sp
from multiprocessing import Pool
sys.path.insert(0, %(here)r)
import group as G
import forms as F
from snub_map import Surface, mixed
B = F.B
path = G.fname("snub_map")
M = json.load(open(path))
t0 = time.time()
TYPEKEY = {}


def key(b, tol=1e-9):
    # (some regions' polytopes have large flat cells -- 12-vertex icosahedra -- whose hull at 1e-9 trips the
    #  ambiguity check on float noise (~1e-13 vertex-plane gaps); at 1e-8 they classify, as known regions)
    for t in (tol, 1e-8):
        try:
            k = G.label(np.asarray(b, float), t)[1]
        except Exception:
            k = None
        if k:
            return k
    return None


def two_sided(s, p, sides):
    """p moved onto surface s, then stepped off both ways: the wall's two regions?"""
    q = s.project(np.asarray(p, float))
    q = q / q.sum()
    if np.abs(q - p).max() > 1e-3:
        return False
    n = s.grad(q)
    n = n - n.mean()
    n /= np.linalg.norm(n)
    ks = {key(q + e * n) for e in (1e-5, -1e-5)}
    # (a real wall: two different polytopes either side -- not necessarily the map's recorded sides: many of these
    #  points were bisected across regions the grid missed, which the survey found, and lie on those regions' walls)
    return len(ks) == 2 and None not in ks


RB = np.random.default_rng(3).dirichlet([1, 1, 1, 1], 3)


def numeric_alive(lab5):
    """Is the 5-vertex determinant not identically zero? (numerically, at three random seeds)"""
    for b in RB:
        P = np.array([G.UNIT @ G.ELEMENTS[g] @ G.X @ b for g in lab5])
        D = P[1:] - P[0]
        if abs(np.linalg.det(D)) > 1e-9 * max(1.0, np.abs(D).max() ** 4):
            return True
    return False


def flat_forms(p):
    """Forms of the polytope's flats with five or more vertices (cells at a loose tolerance) at a wall point."""
    out = set()
    for tol in (1e-6, 1e-5):
        try:
            V, Vn, idx, faces, cells = G.polytope(np.asarray(p, float), tol)
        except Exception:
            continue
        zero = {v: 0 for v, b in zip(B, p) if abs(b) < 1e-12}
        for c in cells:
            vs = sorted({v for f in c for v in faces[f]})
            if len(vs) < 5 or 0 not in vs:      # (one vertex's cells suffice: the group carries the rest)
                continue
            base = [vs[0]]
            for v in vs[1:]:
                if np.linalg.matrix_rank(np.vstack([Vn[u] - Vn[base[0]] for u in base[1:]] + [Vn[v] - Vn[base[0]]]), 1e-9) == len(base):
                    base.append(v)
                if len(base) == 4:
                    break
            if len(base) < 4:
                continue
            for v in [u for u in vs if u not in base][:4]:
                lab5 = [int(idx[i]) for i in base + [v]]
                if not numeric_alive(lab5):
                    continue                    # (flat for every beta: a cell on both sides, no wall)
                fx = F.form(lab5)
                if zero:
                    fx = sp.expand(fx.subs(zero))
                if fx != 0:
                    out.add(str(fx))
    return sorted(out)


def near(text, p, tol=1e-5):
    s = Surface(text)
    return s.dist(np.asarray(p, float)) < tol


def job(w):
    p = np.asarray(w["point"], float)
    found = []
    try:
        cand = [f for f in flat_forms(p) if near(f, p, 1e-4)]
    except Exception as e:
        return w, [], repr(e)[:200]
    for f in cand:
        try:
            facs = F._factor(f, 20)
        except Exception:
            continue
        for fac in facs:
            t = str(sp.expand(fac))
            if mixed(t) and near(t, p, 1e-4) and two_sided(Surface(t), p, w["sides"]):
                found.append(t)
    return w, sorted(set(found)), None


if __name__ == "__main__":
    for t in M["types"]:
        TYPEKEY[t["type"]] = key(t["example"]) or key(t["example"], 1e-7)
    pool = Pool(int(os.environ.get("NPROC", "3")))
    S = [Surface(t) for t in M["surfaces"]]
    todo = [w for w in M["wall_points"] if w["surface"].startswith("none")]
    print(f"{G.SUB}: {len(todo)} of {len(M['wall_points'])} wall points without a surface; {time.time() - t0:.0f}s", flush=True)
    rep = {"sheet": G.SUB, "points": len(todo), "near_misses": 0, "new_surfaces": [], "unresolved": []}
    # near misses first: within 1e-4 of a known surface and two-sided there
    rest = []
    for w in todo:
        p = np.asarray(w["point"], float)
        d = [s.dist(p) for s in S]
        i = int(np.argmin(d)) if d else -1
        if i >= 0 and d[i] < 1e-4 and two_sided(S[i], p, w["sides"]):
            w["surface"] = M["surfaces"][i]
            rep["near_misses"] += 1
        else:
            rest.append(w)
    print(f"  {rep['near_misses']} near misses assigned; {len(rest)} left; {time.time() - t0:.0f}s", flush=True)
    new = []
    NP = int(os.environ.get("NPROC", "3"))
    skipped = 0
    while rest:
        # (in rounds: each round's new surfaces are tried on the points still waiting -- a cheap distance and
        #  two-sided check -- before any of them gets the full search; many points lie on one wall)
        batch, rest = rest[:2 * NP], rest[2 * NP:]
        found_now = []
        for w, found, err in pool.imap_unordered(job, batch, chunksize=1):
            mine = next(x for x in M["wall_points"] if x["point"] == w["point"])
            if found:
                for t in found:
                    if t not in M["surfaces"] and t not in new:
                        new.append(t)
                        found_now.append(t)
                mine["surface"] = found[0]
            else:
                rep["unresolved"].append({"point": w["point"], "sides": w["sides"], "error": err})
        if found_now:
            NS = [(t, Surface(t)) for t in found_now]
            keep = []
            for w in rest:
                p = np.asarray(w["point"], float)
                hit = next((t for t, s in NS if s.dist(p) < 1e-4 and two_sided(s, p, w["sides"])), None)
                if hit:
                    next(x for x in M["wall_points"] if x["point"] == w["point"])["surface"] = hit
                    skipped += 1
                else:
                    keep.append(w)
            rest = keep
        print(f"  {len(new)} new surfaces, {skipped} points placed on them without a search, {len(rest)} waiting; "
              f"{time.time() - t0:.0f}s", flush=True)
    # points still without one may lie on a surface found at another point
    NS = [Surface(t) for t in new]
    for u in list(rep["unresolved"]):
        p = np.asarray(u["point"], float)
        for t, s in zip(new, NS):
            if s.dist(p) < 1e-4 and two_sided(s, p, u["sides"]):
                next(x for x in M["wall_points"] if x["point"] == u["point"])["surface"] = t
                rep["unresolved"].remove(u)
                break
    M["surfaces"] = M["surfaces"] + new
    rep["new_surfaces"] = new
    from collections import Counter
    summ = Counter((w["surface"], tuple(w["sides"]), w["wall_type"]) for w in M["wall_points"])
    M["walls"] = [{"surface": s, "sides": list(sd), "wall_type": wt, "crossings": n} for (s, sd, wt), n in summ.most_common()]
    M["surfaces_note"] += ("; missing walls added by missing_walls.py (flats of the wall polytope that are cells on "
                           "neither side; near misses checked two-sided)")
    json.dump(M, open(path, "w"), indent=1)
    json.dump(rep, open(G.fname("missing_walls"), "w"), indent=1)
    cov = sum(not w["surface"].startswith("none") for w in M["wall_points"])
    print(f"  {len(new)} new surfaces; {cov} of {len(M['wall_points'])} wall points on one now; "
          f"{len(rep['unresolved'])} unresolved; {time.time() - t0:.0f}s", flush=True)
'''


def main():
    subs = sys.argv[1:] or [f"ixo{i}" for i in range(10)]
    for sub in subs:
        if os.path.exists(os.path.join(HERE, f"missing_walls_{sub}.json")):
            print(sub, "done already", flush=True)
            continue
        p = subprocess.run([sys.executable, "-c", CODE % {"here": HERE}], capture_output=True, text=True, cwd=HERE,
                           env={**os.environ, "HYIC_SUB": sub})
        print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(sub, "FAILED", p.stderr[-1500:], flush=True)


if __name__ == "__main__":
    main()
