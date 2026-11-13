"""Systematic search: is every ixitic (or ixoic) region closed off by recorded walls? (closure_<sheet>.json)

On one sheet (HYIC_SUB = ix0..ix9, or ixo0..ixo9 for ixoic), as decaic_atlas/snub_closure.py (no fold here): for every region type of the
sheet (the map's and the dense survey's), up to four spread starting points (its example and Halton points labelled
with it), and from each a walk along 24 evenly spread directions in steps of 0.04 until the polytope changes or a mirror
is reached (a seam to another sheet: the walk stops). A change is bisected to ~1e-11, put exactly on the surface through
it (the map's surfaces, or the forms of both sides when none), and the polytope there (wall tolerance) and just beyond
classified. A key not among the group's merged shapes (ixitic_shapes.json / ixoic_shapes.json) is new: a missed region (beyond) or wall
shape (on the surface).

    HYIC_SUB=ix3 python closure.py      (ixoic: HYIC_SUB=ixo3, after python ixitic_shapes.py ixo)

Lite (the default; CLOSURE_LITE=0 for the full search): two starts and 12 of the directions, a quarter of the walks.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import group as G  # noqa: E402
from snub_map import Surface, forms_job  # noqa: E402

STEP, MAXLEN = 0.04, 1.5
# (lite: two starts per region and every other of the 24 directions -- a quarter of the walks. The step only finds
#  that the walk has left its region; the bisection then finds the first wall, so a coarser step loses nothing but
#  regions thinner than the step lying wholly between two samples, which the point just beyond the wall still sees
#  when it is the next one)
LITE = os.environ.get("CLOSURE_LITE", "1") == "1"


def directions(n):
    U = np.linalg.svd(np.ones((1, 4)))[2][1:]
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    r = np.sqrt(1 - z * z)
    a = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1) @ U


def label(b, tol=1e-9):
    try:
        return G.label(np.asarray(b, float), tol)[1]
    except Exception:
        return None


def walk(item):
    s, d, k0 = item
    s, d = np.asarray(s), np.asarray(d)
    prev, t = s, STEP
    while t < MAXLEN:
        p = s + t * d
        if p.min() < 0:
            return None                                   # (a mirror: a seam to another sheet)
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
                    kb = km if km is not None else kb
            x = (lo + hi) / 2
            return x.tolist(), kb, (x + 1e-6 * d).tolist()
        prev, t = p, t + STEP
    return None


def wall_job(item):
    text, x = item
    y = Surface(text).project(np.asarray(x)) if text else np.asarray(x)
    return y.tolist(), label(y, 1e-7)


def confirm(new_regions, known, pool=None):
    out = {}
    for k, v in new_regions.items():
        kb = label(tuple(v["at"]))
        if kb and kb not in known:
            out[kb] = {**v, "bisection_key": k} if kb != k else v
    return out


def recheck():
    """Re-check a finished sheet's new regions (closure_<sheet>.json) by the point beyond the wall."""
    f = os.path.join(HERE, f"closure_{G.SUB}.json")
    out = json.load(open(f))
    known = {s["key"] for s in json.load(open(os.path.join(HERE, f"{G.NAME}_shapes.json")))["shapes"]}
    before = out["new_regions"]
    out["new_regions"] = confirm(before, known)
    if len(out["new_regions"]) != len(before):
        out["rechecked"] = (f"{len(before)} new regions from the bisection's last label; {len(out['new_regions'])} "
                            f"confirmed by the point just beyond the wall")
    json.dump(out, open(f, "w"), indent=1)
    print(f"{G.SUB}: {len(before)} -> {len(out['new_regions'])} new regions", flush=True)


def main():
    M = json.load(open(os.path.join(HERE, G.fname("snub_map"))))
    regs = [t["example"] for t in M["types"] if t["grid_regions"]]
    try:
        regs += [t["example"] for t in json.load(open(os.path.join(HERE, G.fname("survey"))))["new"].values()]
    except FileNotFoundError:
        pass
    known = {s["key"] for s in json.load(open(os.path.join(HERE, f"{G.NAME}_shapes.json")))["shapes"]}
    surfs = [Surface(t) for t in M["surfaces"]]
    pool = Pool(4)
    keys = pool.map(label, regs)
    reg = {}
    for b, k in zip(regs, keys):
        if k:
            reg.setdefault(k, np.asarray(b) / np.sum(b))
    X = qmc.Halton(4, seed=5).random(3000)
    X = -np.log(1 - X * 0.999999)
    X = X / X.sum(1, keepdims=True)
    # (checkpoint: the start labels and each batch of walks, so a restart resumes)
    ck = os.path.join(HERE, f"closure_{G.SUB}.partial.json")
    C = json.load(open(ck)) if os.path.exists(ck) else {}
    K = C.get("K", [])
    for i in range(len(K), len(X), 200):
        K += pool.map(label, list(map(tuple, X[i:i + 200])), chunksize=16)
        C["K"] = K
        json.dump(C, open(ck, "w"))
    starts = defaultdict(list)
    for k, b in reg.items():
        starts[k].append(b)
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
    full = [((tuple(s), tuple(d), k), si, di) for k in reg for si, s in enumerate(starts[k]) for di, d in enumerate(D)]
    E = C.get("E", [])
    if "I" in C:                                        # (the walk list as first fixed, so a resume indexes the same)
        items = [(tuple(s), tuple(d), k) for s, d, k in C["I"]]
    else:                                               # (walks already done kept; the rest thinned when lite)
        items = [it for it, _, _ in full[:len(E)]] + [it for it, si, di in full[len(E):]
                                                      if not LITE or (si < 2 and di % 2 == 0)]
        C["I"] = items
    if E:
        print(f"resumed after {len(E)} of {len(items)} walks", flush=True)
    for i in range(len(E), len(items), 64):
        E += pool.map(walk, items[i:i + 64], chunksize=4)
        C["E"] = E
        json.dump(C, open(ck, "w"))
    ch = [(i, e) for i, e in enumerate(E) if e is not None]
    sid = {}
    miss = []
    for i, (x, kb, beyond) in ch:
        dd = [s.dist(np.asarray(x)) for s in surfs]
        if dd and min(dd) < 1e-6:
            sid[i] = surfs[int(np.argmin(dd))].text
        else:
            miss.append(i)
    if miss:
        more = pool.map(forms_job, [tuple(E[i][0]) for i in miss] + [tuple(E[i][2]) for i in miss], chunksize=1)
        new = set().union(*more) - {s.text for s in surfs}
        surfs += [Surface(t) for t in sorted(new)]
        for i in miss:
            dd = [s.dist(np.asarray(E[i][0])) for s in surfs]
            if min(dd) < 1e-6:
                sid[i] = surfs[int(np.argmin(dd))].text
    W = pool.map(wall_job, [(sid.get(i, ""), E[i][0]) for i, _ in ch], chunksize=4)
    pool.close()
    new_regions, new_walls, nosurf = {}, {}, 0
    for (i, (x, kb, beyond)), (y, kw) in zip(ch, W):
        if i not in sid:
            nosurf += 1
        if kb and kb not in known:
            new_regions.setdefault(kb, {"at": beyond, "from": items[i][2], "count": 0})["count"] += 1
        if kw and kw not in known and kw not in reg:
            new_walls.setdefault(kw, {"at": y, "surface": sid.get(i), "sides": [items[i][2], kb], "count": 0})["count"] += 1
    # (the bisection's last label is within 2^-36 of the wall, where a wall polytope can classify as a stray key: a new
    #  region counts only if the point just beyond the wall (1e-6 along the walk) is itself new)
    new_regions = confirm(new_regions, known, pool=None)
    out = {"_about": __doc__.strip().split("\n\n")[0], "sheet": G.SUB, "regions": len(reg), "walks": len(items),
           "changes": len(ch), "on_no_surface": nosurf, "new_regions": new_regions, "new_wall_shapes": new_walls}
    json.dump(out, open(os.path.join(HERE, f"closure_{G.SUB}.json"), "w"), indent=1)
    os.remove(ck)
    print(f"{G.SUB}: {len(reg)} regions, {len(items)} walks, {len(ch)} changes ({nosurf} on no surface); "
          f"{len(new_regions)} new regions, {len(new_walls)} new wall shapes", flush=True)


if __name__ == "__main__":
    recheck() if len(sys.argv) > 1 and sys.argv[1] == "recheck" else main()
