"""Central seeds of the ixitic shapes, found from the shapes' own extent (replaces the central seeds of ixitic_offs.py).

The earlier centre finder measured depth to the chamber's mirrors and walls; under ixitic the mirrors are seams, not
boundaries, so a shape on a mirror edge or face had nothing to measure (depth 1.0, the seed returned) and several
others stopped at their seed. Here the boundary is where the polytope itself changes: from a point x of the shape's
locus (its zero betas kept zero, its wall surfaces kept, betas summing to 1; negative betas allowed -- the seed is then
just a point beyond a seam, and its ixitic orbit the same polychoron as the reflected seed's in the next sheet), each
direction d of the locus is followed until the arrangement key changes (steps of 0.01, then bisection), giving the
chord r(d) + r(-d) through x. x moves to the mean of the chords' midpoints, x + sum (r(d) - r(-d))/2 d / n, until it
stops: for a line the segment's midpoint, for a face or region the point that every chord through it is (on average)
bisected at. depth = the shortest r(d). The central .off file is rewritten from the new seed.

Also for oxitic (its mirrors are seams too; the fold, kept by the swap in oxitic, is crossed like a seam: beyond it
lies the same polychoron): python recentre.py --oxitic [name ...] updates toxitic_atlas/opt_central_ox|oxb.json.

    python recentre.py [--oxitic] [name ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE = r'''
import json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, %(here)r)
import group as G
from snub_map import Surface  # (the atlas's own: hyic_atlas or contic_atlas)
from four_d_vertex_generator.off import to_4off
from four_d_vertex_generator.local_view import hull_edges

STEP, MAXLEN = 0.01, 1.0


def directions(k):
    if k == 1:
        return np.array([[1.0], [-1.0]])
    if k == 2:
        a = np.arange(12) * np.pi / 6
        return np.stack([np.cos(a), np.sin(a)], 1)
    n = 26
    i = np.arange(n // 2) + 0.5
    z = 1 - 2 * i / (n // 2)
    r = np.sqrt(1 - z * z)
    t = np.pi * (1 + 5 ** 0.5) * i
    D = np.stack([r * np.cos(t), r * np.sin(t), z], 1)
    return np.concatenate([D, -D])                  # (symmetric pairs: d and -d)


def job(row):
    b0 = np.asarray(row["seed"], float)
    b0 = b0 / b0.sum()
    zero = [i for i in range(4) if abs(b0[i]) < 1e-12]
    surfs = [Surface(t) for t in row["locus"]]
    tol = row["tol"]
    key = row["key"]

    def proj(x):
        x = np.array(x, float)
        for _ in range(8 if len(surfs) < 2 else 60):     # (two walls: alternate until on both)
            x[zero] = 0
            x = x / x.sum()
            for s in surfs:
                x = s.project(x)
        x[zero] = 0
        return x / x.sum()

    def lab(x):
        try:
            return G.label(x, tol)[1]
        except Exception:
            return None

    def basis(x):
        C = [np.ones(4)] + [np.eye(4)[z] for z in zero] + [s.grad(x) for s in surfs]
        _, S, Vt = np.linalg.svd(np.array(C))
        rank = int((S > 1e-9).sum())
        return Vt[rank:]                             # (orthonormal rows spanning the locus's tangent space)

    def chamber(x, d):
        """How far x + t d stays in the sheet's own chamber (its nonzero betas positive)."""
        ts = [-x[i] / d[i] for i in range(4) if i not in zero and d[i] < -1e-15]
        return min([t for t in ts if t > 0] + [MAXLEN])

    def reach(x, d):
        r = change(x, d)
        # (a shape that runs on past the cap across the seams has no centre of its own there: the chord stops at
        #  the sheet's chamber instead -- the seams then bound it, as in one sheet)
        return r if r < MAXLEN else max(chamber(x, d), 1e-6)

    def change(x, d):
        lo, t = 0.0, STEP / 2                        # (steps doubling: 0.005, 0.01, 0.02, ... then bisection)
        while t <= MAXLEN:
            p = proj(x + t * d)
            if lab(p) != key:
                hi = t
                for _ in range(20):
                    m = (lo + hi) / 2
                    if lab(proj(x + m * d)) == key:
                        lo = m
                    else:
                        hi = m
                return (lo + hi) / 2
            lo, t = t, 2 * t
        return MAXLEN

    ck = os.path.join(%(here)r, "recentre_ckpt", row["name"] + ".json")
    x = proj(b0)
    if lab(x) != key:
        return {"name": row["name"], "status": "the seed does not classify as the shape at its tolerance: kept",
                "centre": list(b0), "depth": None}
    k = 3 - len(zero) - len(surfs)
    hist = []
    it0 = 0
    if os.path.exists(ck):                           # (resume after a container restart)
        c = json.load(open(ck))
        x, it0 = np.asarray(c["x"], float), c["it"]
    for it in range(it0, 12):
        U = basis(x)
        D = directions(k) @ U
        R = np.array([reach(x, d) for d in D])
        half = len(D) // 2
        if k == 1:
            move = (R[0] - R[1]) / 2 * D[0]
        else:
            move = sum((R[i] - R[i + half]) / 2 * D[i] for i in range(half)) / half
        hist.append(float(R.min()))
        y = proj(x + move)
        if lab(y) != key:
            y = proj(x + move / 2)
            if lab(y) != key:
                break
        step = float(np.abs(y - x).max())
        x = y
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        json.dump({"x": list(map(float, x)), "it": 12 if step < 1e-4 else it + 1}, open(ck, "w"))
        if step < 1e-4:
            break
    if it0 >= 12:                                    # (converged before a restart: only the last chords to redo)
        it = it0 - 1
        D = directions(k) @ basis(x)
        R = np.array([reach(x, d) for d in D])
    unbounded = any(change(x, d) >= MAXLEN for d in D)
    return {"name": row["name"], "centre": list(map(float, x)), "depth": float(R.min()), "iterations": it + 1,
            "status": ("centre of the shape's extent v2 (chords to where its polytope changes, seams crossed"
                       + ("; where it runs on past the cap, to the sheet's chamber)" if unbounded else ")"))}


rows = json.loads(sys.stdin.read())


def run(row):
    return row, job(row)


pool = Pool(int(os.environ.get("NPROC", "3")))
for row, o in pool.imap_unordered(run, rows):
    b = np.asarray(o["centre"], float)
    V, Vn, idx, faces, cells = G.polytope(b, row["tol"])
    k = G.label(b, row["tol"])[1]
    b0 = np.asarray(row["seed"], float)
    V0, _, _, _, cells0 = G.polytope(b0 / b0.sum(), row["tol"])
    o["ok"] = k == row["key"] and (len(V), len(cells)) == (len(V0), len(cells0))
    if o["ok"]:
        open(os.path.join(%(root)r, row["central_off"]), "w").write(to_4off(Vn, faces, cells))
        E = [np.linalg.norm(Vn[a] - Vn[c]) for a, c in (tuple(e) for e in hull_edges(faces))]
        o["central_R"] = float(max(E) / min(E))
    print(json.dumps(o), flush=True)
'''


def main():
    args = sys.argv[1:]
    ox = "--oxitic" in args
    only = set(a for a in args if not a.startswith("--"))
    if ox:
        atlas, envvar = os.path.join(ROOT, "contic_atlas"), "CONTIC_SUB"
        files = [os.path.join(ROOT, "toxitic_atlas", f"opt_central_{s}.json") for s in ("ox", "oxb")]
    else:
        atlas, envvar = HERE, "HYIC_SUB"
        files = [os.path.join(HERE, "opt_central_ixitic.json")]
    for oc in files:
        run(oc, atlas, envvar, only)


def run(oc, atlas, envvar, only):
    O = json.load(open(oc))
    by = {}
    for r in O["shapes"]:
        if r["kind"] == "point" or (only and r["name"] not in only):
            continue
        if not only and r.get("central_status", "").startswith("centre of the shape's extent v2"):
            continue                                    # (done in an earlier run)
        by.setdefault(r["sheet"], []).append({
            "name": r["name"], "key": r["key"], "seed": r["seed"], "locus": r.get("locus") or [],
            "tol": 1e-7 if r["kind"] == "wall" else 1e-9, "central_off": r["central_off"]})
    rows = {r["name"]: r for r in O["shapes"]}
    for sub, items in sorted(by.items()):
        p = subprocess.Popen([sys.executable, "-c", CODE % {"here": atlas, "root": ROOT}], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True, cwd=atlas, env={**os.environ, envvar: sub})
        p.stdin.write(json.dumps(items))
        p.stdin.close()
        for line in p.stdout:
            o = json.loads(line)
            r = rows[o["name"]]
            if o.get("ok"):
                r.update(central=o["centre"], depth=o["depth"], central_status=o["status"], central_R=o["central_R"])
                r.pop("central_note", None)
            else:
                r["central_status"] = r.get("central_status", "") + "; the extent finder's centre was another polytope"
            json.dump(O, open(oc, "w"), indent=1)       # (saved after each shape: a rerun skips it)
            print(f"{sub} {o['name']:24} depth {o['depth'] if o['depth'] is None else round(o['depth'], 5)}  "
                  f"{'written' if o.get('ok') else 'NOT written'}  {[round(v, 5) for v in o['centre']]}  "
                  f"R {o.get('central_R', 0):.4f}", flush=True)
        json.dump(O, open(oc, "w"), indent=1)


if __name__ == "__main__":
    main()
