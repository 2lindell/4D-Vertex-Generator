"""Exact wall equations for the ixoic sheets, fitted after the maps (updates snub_map_ixo<i>.json and wall_shapes).

Ixoic is not inside H4: its elements have entries (a + b sqrt2 + c sqrt5 + d sqrt10)/4, which the map's exact forms
(over Q(sqrt5)) could not take, so the ixoic sheets were mapped numerically and their walls known only as bisected
points. group.exact now writes ixoic's elements over Q(sqrt2, sqrt5); here, for every region and wall type of a
sheet's map, its forms (forms.region_forms: one quartic per face class and side) are evaluated at the sheet's wall
points, and only the forms vanishing at one (|f| / |grad f| < 1e-7) are factored, over Q(sqrt2, sqrt5) (over Q after
60 s); the factors vanishing at wall points are the sheet's wall surfaces. Each wall point is assigned its surface,
the map's wall summary and surfaces list rewritten, and the coverage reported.

    python ixoic_walls.py [ixo0 ...]
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
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, %(here)r)
import group as G
import forms as F
from snub_map import Surface, mixed
B = F.B
path = G.fname("snub_map")
M = json.load(open(path))
P = np.array([w["point"] for w in M["wall_points"]], float)
t0 = time.time()


def forms_of(b):
    try:
        return [str(f) for f in F.region_forms(np.asarray(b, float))]
    except Exception:
        return []


def numeric(text):
    """(monomial exponents, float coefficients) of the form: a fast evaluator."""
    p = sp.Poly(sp.sympify(text), *B)
    return np.array(p.monoms(), float), np.array([float(c) for c in p.coeffs()])


def vanishing(text):
    E, c = numeric(text)
    vals = (np.prod(P[:, None, :] ** E[None], axis=2) * c).sum(1)
    scale = np.abs(c).sum() * np.abs(P).max(1) ** E.sum(1).max()
    return text, int((np.abs(vals) < 1e-11 * scale).sum())


def signature(text):
    """The form up to scale and sign, for merging duplicates."""
    E, c = numeric(text)
    c = c / c[np.argmax(np.abs(c))]
    return tuple(sorted(zip(map(tuple, E.astype(int)), np.round(c, 9))))


def factors(text):
    try:
        facs = F._factor(text, 20)
    except Exception:
        return []
    return [str(sp.expand(f)) for f in facs if mixed(str(f))]


if __name__ == "__main__":
    pool = Pool(int(os.environ.get("NPROC", "4")))
    ex = [t["example"] for t in M["types"]]
    fs = sorted({f for r in pool.map(forms_of, ex, chunksize=1) for f in r})
    sig = {}
    for f, g in zip(fs, pool.map(signature, fs, chunksize=8)):
        sig.setdefault(g, f)
    fs = sorted(sig.values())
    print(f"{G.SUB}: {len(ex)} types, {len(fs)} forms, {len(P)} wall points; {time.time() - t0:.0f}s", flush=True)
    hit = [t for t, n in pool.map(vanishing, fs, chunksize=4) if n]
    print(f"  {len(hit)} forms vanish at a wall point; {time.time() - t0:.0f}s", flush=True)
    facs = sorted({f for r in pool.map(factors, hit, chunksize=1) for f in r})
    surfs = [s for s, n in pool.map(vanishing, facs, chunksize=4) if n]
    S = [Surface(t) for t in surfs]
    covered = 0
    for w in M["wall_points"]:
        x = np.asarray(w["point"], float)
        d = [s.dist(x) for s in S]
        if d and min(d) < 1e-7:
            w["surface"] = surfs[int(np.argmin(d))]
            covered += 1
        else:
            w["surface"] = f"none (nearest {min(d) if d else 1:.0e})"
    summ = Counter((w["surface"], tuple(w["sides"]), w["wall_type"]) for w in M["wall_points"])
    M["walls"] = [{"surface": s, "sides": list(sd), "wall_type": wt, "crossings": n} for (s, sd, wt), n in summ.most_common()]
    M["surfaces"] = surfs
    M["surfaces_note"] = "fitted after the map by ixoic_walls.py (exact over Q(sqrt2, sqrt5))"
    json.dump(M, open(path, "w"), indent=1)
    try:
        wp = G.fname("wall_shapes")
        W = json.load(open(wp))
        for w in W["shapes"]:
            x = np.asarray(w["seed_beta"], float)
            w["surfaces"] = {t: float(s.dist(x)) for t, s in zip(surfs, S) if s.dist(x) < 1e-6}
        json.dump(W, open(wp, "w"), indent=1)
    except FileNotFoundError:
        pass
    print(f"  {len(surfs)} wall surfaces; {covered} of {len(P)} wall points on one; {time.time() - t0:.0f}s", flush=True)
'''


def main():
    subs = sys.argv[1:] or [f"ixo{i}" for i in range(10)]
    for sub in subs:
        M = json.load(open(os.path.join(HERE, f"snub_map_{sub}.json")))
        if M.get("surfaces_note"):
            print(sub, "done already", flush=True)
            continue
        p = subprocess.run([sys.executable, "-c", CODE % {"here": HERE}], capture_output=True, text=True, cwd=HERE,
                           env={**os.environ, "HYIC_SUB": sub})
        print(p.stdout.strip(), flush=True)
        if p.returncode:
            print(sub, "FAILED", p.stderr[-1500:], flush=True)


if __name__ == "__main__":
    main()
