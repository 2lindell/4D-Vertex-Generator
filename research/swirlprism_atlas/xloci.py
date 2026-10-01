"""Which atlas types fill 3D regions of the cell and which only live on walls, lines or points.

Step 1 (python xloci.py step1): for one sample of each type (the one deepest inside the cell), nudge the
seed by EPS in random directions, both ways, and record which nudges keep the type. A type that
survives nearly every nudge fills a region; one that survives few or none is lower-dimensional
(transitional) and is located further in step 2.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from cell_atlas2 import identify, load_refs
from probe import _one

EPS = 1e-5
N_DIRS = 16


def samples_by_type():
    refs = load_refs()[0]
    xids = json.load(open("atlas_xids.json"))
    out = defaultdict(list)
    for s in json.load(open("golden_22.json")):
        cid, xkey, _ = identify(s["sig"], refs)
        tid = cid or xids.get(xkey)
        if tid:
            b = np.array(s["beta"], float)
            out[tid].append(b / b.sum())
    return out


def directions(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((N_DIRS, 4))
    d -= d.mean(axis=1, keepdims=True)       # stay in the barycentric simplex (sum fixed)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _label(b):
    return _one(b)[0]


def step1():
    S = samples_by_type()
    jobs, meta = [], []
    D = directions()
    for tid, bs in sorted(S.items()):
        b = max(bs, key=lambda x: x.min())    # the sample deepest inside the cell
        meta.append((tid, b))
        jobs += [b + s * EPS * d for d in D for s in (1, -1)]
    with Pool(4) as pool:
        labels = pool.map(_label, jobs, chunksize=8)
    out, k = {}, 0
    for tid, b in meta:
        got = labels[k:k + 2 * N_DIRS]
        k += 2 * N_DIRS
        out[tid] = {"beta": b.tolist(), "kept": sum(g == tid for g in got), "of": len(got),
                    "neighbours": sorted({g for g in got if g != tid})}
    json.dump(out, open("xloci_step1.json", "w"), indent=1)
    for tid, r in out.items():
        print(f"{tid:5s} kept {r['kept']:2d}/{r['of']}  neighbours {r['neighbours'][:6]}")


if __name__ == "__main__":
    {"step1": step1}[sys.argv[1]]()
