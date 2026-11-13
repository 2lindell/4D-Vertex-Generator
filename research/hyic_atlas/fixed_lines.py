"""Ixoic shapes on the fixed lines of its half-turns (fixed_lines.json).

Ixoic has no reflections: its involutions are half-turns (and the central inversion). A seed x = X beta fixed by a
half-turn g gives an orbit of 1440 vertices, not 2880; the fixed betas, (g - 1) X beta = 0, are a 2-plane, so a line in
the sheet's chamber (betas positive, summing to 1). Lines on the chamber's own edges were in the strata; the others run
through the chamber's interior, where the map's grid and survey (random points) never land on them. Each line crossing
a sheet's chamber is sampled (N points), each key change bisected, and every polytope met is recorded with its kind
(line; a stretch of one key) and where two lines cross or a line meets the chamber's boundary, the point there.

    python fixed_lines.py [N]       (all ten ixoic sheets)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import json, sys, itertools
import numpy as np
sys.path.insert(0, %(here)r)
import group as G
N = %(n)d
E = G.ELEMENTS


def lab(b, tol=1e-9):
    try:
        s, k = G.label(np.asarray(b, float), tol)
        return (k, s) if k else (None, None)
    except Exception:
        return (None, None)


def segment(g):
    A = (g - np.eye(4)) @ G.X
    _, S, Vt = np.linalg.svd(A)
    if (S > 1e-9).sum() != 2:
        return None
    u, v = Vt[2:]
    if abs(u.sum()) < 1e-12:
        u, v = v, u
    p = u / u.sum()
    d = v - v.sum() * p
    d /= np.linalg.norm(d)
    lo, hi = -1e9, 1e9
    for k in range(4):
        if abs(d[k]) < 1e-14:
            if p[k] < -1e-12:
                return None
            continue
        t = -p[k] / d[k]
        lo, hi = (max(lo, t), hi) if d[k] > 0 else (lo, min(hi, t))
    return (p + lo * d, p + hi * d) if hi - lo > 1e-9 else None


segs = {}
for g in E:
    if np.allclose(g @ g, np.eye(4)) and not np.allclose(g, np.eye(4)) and not np.allclose(g, -np.eye(4)):
        s = segment(g)
        if s:
            k = tuple(sorted([tuple(np.round(s[0], 7)), tuple(np.round(s[1], 7))]))
            segs.setdefault(k, s)
out = {"sheet": G.SUB, "segments": [], "shapes": {}}


def add(b, kind, seg):
    k, s = lab(b)
    if not k:
        k, s = lab(b, 1e-7)
    if not k:
        return None
    r = out["shapes"].setdefault(k, {"signature": s, "kind": kind, "seed": list(map(float, b)), "segments": []})
    if kind == "line" and r["kind"] == "point":
        r.update(kind="line", seed=list(map(float, b)))
    if seg not in r["segments"]:
        r["segments"].append(seg)
    return k


for si, (a, b) in enumerate(segs.values()):
    ts = np.linspace(0, 1, N + 1)[1:-1]
    keys = [lab(a + t * (b - a))[0] for t in ts]
    out["segments"].append({"from": list(map(float, a)), "to": list(map(float, b)), "keys": len(set(keys))})
    for t, k in zip(ts, keys):
        if k:
            add(a + t * (b - a), "line", si)
    for t0, t1, k0, k1 in zip(ts, ts[1:], keys, keys[1:]):
        if k0 != k1:                                    # (a change: the point between, bisected)
            lo, hi = t0, t1
            for _ in range(40):
                m = (lo + hi) / 2
                (lo, hi) = (m, hi) if lab(a + m * (b - a))[0] == k0 else (lo, m)
            add(a + (lo + hi) / 2 * (b - a), "point", si)
    add(a, "point", si)                                 # (its ends, on the chamber's faces: seams to other sheets)
    add(b, "point", si)
S = list(segs.values())
for (i, (a, b)), (j, (c, d)) in itertools.combinations(enumerate(S), 2):
    M = np.array([b - a, -(d - c)]).T
    st, res, *_ = np.linalg.lstsq(M, c - a, rcond=None)
    if 0 < st[0] < 1 and 0 < st[1] < 1 and np.abs(M @ st - (c - a)).max() < 1e-9:
        add(a + st[0] * (b - a), "point", i)
print(json.dumps(out))
'''


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    res = {}
    for i in range(10):
        sub = f"ixo{i}"
        p = subprocess.run([sys.executable, "-c", CODE % {"here": HERE, "n": n}], capture_output=True, text=True,
                           cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        if p.returncode:
            print(sub, "FAILED", p.stderr[-800:], flush=True)
            continue
        o = json.loads(p.stdout.strip().splitlines()[-1])
        res[sub] = o
        print(f"{sub}: {len(o['segments'])} lines, {len(o['shapes'])} polytopes", flush=True)
        json.dump(res, open(os.path.join(HERE, "fixed_lines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
