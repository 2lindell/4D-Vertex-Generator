"""Grow a finished closure check's new regions (closure_<sheet>.json) by sampling around them.

New regions come in clusters (thin slivers and the small regions between them), and a walk crossing one rarely
crosses its neighbours. Around each new region's point, N points at each of the radii 1e-4 and 1e-3 are labelled; a
key in neither the atlas nor any closure file's new regions is a new region too (with "from": "near <key>"), and is
sampled around in turn, until no more turn up. Points that fail to label at 1e-9 and 1e-8 are skipped.

    python closure_grow.py ixo0 [ixo2 ...]      (N = 48; CLOSURE_GROW_N to change)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, %(here)r)
import group as G
N = %(n)d
f = os.path.join(%(here)r, f"closure_{G.SUB}.json")
out = json.load(open(f))
known = {s["key"] for s in json.load(open(os.path.join(%(here)r, f"{G.NAME}_shapes.json")))["shapes"]}
for g in glob.glob(os.path.join(%(here)r, "closure_*.json")):
    if not g.endswith(".partial.json") and g != f:
        known |= set(json.load(open(g)).get("new_regions", {}))


def lab(p):
    for t in (1e-9, 1e-8):
        try:
            s, k = G.label(np.asarray(p, float), t)
            if k:
                return s, k
        except Exception:
            pass
    return None, None


rng = np.random.default_rng(7)
todo = list(out["new_regions"])
done = set()
added = 0
with Pool(int(os.environ.get("NPROC", "4"))) as P:
    while todo:
        k = todo.pop(0)
        if k in done:
            continue
        done.add(k)
        a = np.asarray(out["new_regions"][k]["at"], float)
        pts = []
        for r in (1e-4, 1e-3):
            for _ in range(N):
                d = rng.normal(size=4)
                d -= d.mean()
                p = a + r * d / np.linalg.norm(d)
                if (p > 0).all():
                    pts.append(p)
        for p, (s, kk) in zip(pts, P.map(lab, pts, chunksize=4)):
            if kk and kk not in known and kk not in out["new_regions"]:
                out["new_regions"][kk] = {"at": p.tolist(), "from": f"near {k}", "count": 1, "signature": s}
                todo.append(kk)
                added += 1
        print(f"{G.SUB}: around {k}: {len(out['new_regions'])} new regions", flush=True)
out["grown"] = f"{added} more new regions by sampling around the new regions ({N} points at 1e-4 and 1e-3 each)"
json.dump(out, open(f, "w"), indent=1)
print(f"{G.SUB}: {added} more new regions", flush=True)
'''


def main():
    n = int(os.environ.get("CLOSURE_GROW_N", "48"))
    for sub in sys.argv[1:]:
        p = subprocess.run([sys.executable, "-c", CODE % {"here": HERE, "n": n}], text=True, cwd=HERE,
                           env={**os.environ, "HYIC_SUB": sub})
        if p.returncode:
            print(sub, "FAILED", flush=True)


if __name__ == "__main__":
    main()
