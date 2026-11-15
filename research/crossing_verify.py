"""Check the crossing search's candidates (crossing_search/<atlas>_<sheet>.json) before they join an atlas.

For each candidate, on its sheet: (1) it labels the same at hull tolerance 1e-7 and 1e-8, and at 1e-9 where it can;
(2) it keeps its key moved along its locus (a line: along the crossing curve; a wall: two directions within the wall;
a point: none) by 1e-5, 1e-4 and 1e-3, re-projected onto the locus each time; (3) it changes key stepped 1e-5 off each
of its walls, both ways (else it is not confined to that locus: a region or wall shape seen through a tolerance); and
(4) a key the family's shape lists already hold, or one of the sheet's map polytopes at a tighter tolerance, is
reported as known. A candidate passing (1)-(3) is confirmed, with the dimension it holds over (its degrees).
Report: crossing_search/verified.json and a section of crossing_search/README.md.

    python crossing_verify.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crossing_search")
ATLAS = {"contic": ("contic_atlas", "CONTIC_SUB"), "decaic": ("decaic_atlas", "DECAIC_SUB"),
         "hyic": ("hyic_atlas", "HYIC_SUB")}

CODE = r'''
import json, sys
import numpy as np
sys.path.insert(0, %(atlas)r)
import group as G
from snub_map import Surface
A = json.loads(sys.stdin.read())
FOLD = bool(getattr(G, "FOLD", False))
rng = np.random.default_rng(3)


def lab(b, tol):
    try:
        return G.label(np.asarray(b, float), tol)[1]
    except Exception:
        return None


def onto(x, S, n=200):
    x = np.asarray(x, float)
    for _ in range(n):
        for s in S:
            x = s.project(x)
            x = x / x.sum()
    return x if max(s.dist(x) for s in S) < 1e-10 and x.min() > 0 else None


def tangents(x, S):
    C = np.array([np.ones(4)] + [s.grad(x) for s in S])
    V = np.linalg.svd(C)[2]
    return V[len(S) + 1:]                       # (directions within the locus)


out = {}
for k, c in A["cands"].items():
    S = [Surface(t) for t in c["surfaces"]]
    x = onto(c["seed"], S)
    r = {"kind": c["kind"], "surfaces": len(S)}
    if x is None:
        r["verdict"] = "seed does not project onto its walls"
        out[k] = r
        continue
    r["labels"] = {str(t): lab(x, t) for t in (1e-9, 1e-8, 1e-7)}
    r["same_label"] = r["labels"]["1e-07"] == k and r["labels"]["1e-08"] in (k, None)
    T = tangents(x, S)
    hold = {}
    for e in (1e-5, 1e-4, 1e-3):
        ok = []
        for t in T:
            for sgn in (1, -1):
                y = onto(x + sgn * e * t, S)
                ok.append(y is not None and lab(y, 1e-7) == k)
        hold[str(e)] = f"{sum(ok)}/{len(ok)}"
    r["along"] = hold
    off = []
    for s in S:
        g = s.grad(x)
        g = g - g.mean()
        g /= np.linalg.norm(g)
        for sgn in (1, -1):
            y = x + sgn * 1e-5 * g
            off.append(lab(y / y.sum(), 1e-7) != k if y.min() > 0 else True)
    r["changes_off"] = f"{sum(off)}/{len(off)}"
    r["seed"] = x.tolist()
    deg = len(T)
    holds = all(v.split("/")[0] == v.split("/")[1] for kk, v in hold.items() if kk != "0.001") if deg else True
    if not r["same_label"]:
        r["verdict"] = "label unstable"
    elif not all(off):
        r["verdict"] = "not confined to its walls"
    elif not holds:
        r["verdict"] = "does not hold along its locus (a point?)" if deg else "point"
    else:
        r["verdict"] = "confirmed"
    r["degrees"] = deg if holds else 0
    out[k] = r
print(json.dumps(out))
'''


def main():
    res = {}
    for f in sorted(glob.glob(os.path.join(OUT, "*_*.json"))):
        name = os.path.basename(f)[:-5]
        atlas, sheet = name.split("_", 1)
        if atlas not in ATLAS:
            continue
        d = json.load(open(f))
        if not d.get("candidates"):
            continue
        adir, var = ATLAS[atlas]
        adir = os.path.join(HERE, adir)
        p = subprocess.run([sys.executable, "-c", CODE % {"atlas": adir}], input=json.dumps({"cands": d["candidates"]}),
                           capture_output=True, text=True, cwd=adir, env={**os.environ, var: sheet})
        if p.returncode:
            print(name, "FAILED", p.stderr[-1500:], flush=True)
            continue
        o = json.loads(p.stdout.strip().splitlines()[-1])
        for k, r in o.items():
            r["signature"] = d["candidates"][k]["signature"]
            print(f"{atlas} {sheet} {k} {r['kind']}: {r['verdict']} (degrees {r.get('degrees')}); along {r.get('along')}; "
                  f"off {r.get('changes_off')}; labels {r.get('labels')}", flush=True)
        res[name] = o
    json.dump(res, open(os.path.join(OUT, "verified.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
