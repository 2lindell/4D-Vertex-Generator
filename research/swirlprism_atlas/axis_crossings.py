"""Points where an extra-symmetry axis (the purple 2400 axes and the green order-3 girdle) crosses a transitional
class: walk along each axis segment, bisect every change of shape to the crossing, and classify the crossing
point itself. A transitional class there (one that differs from the shapes on either side) is a wall, line or
point that the axis passes through. Writes axis_crossings.json.

    python axis_crossings.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
from normalizer import coset_axes
from probe import _one
from supergroups import girdle_families, girdle_segments

N = 400


def _bisect(job, steps=50):
    a, b, la = job
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        m = (lo + hi) / 2
        if _one(a + m * (b - a))[0] == la:
            lo = m
        else:
            hi = m
    t = (lo + hi) / 2
    p = a + t * (b - a)
    return p.tolist(), _one(p)[0], _one(a + lo * (b - a))[0], _one(a + hi * (b - a))[0]


def main():
    fam = girdle_families()
    segs = [(f"purple axis ({fam(a, b)}-family)", a, b) for a, b in coset_axes()]
    segs += [("green order-3 girdle", a, b) for a, b in girdle_segments()]
    jobs, meta = [], []
    for k, (name, a, b) in enumerate(segs):
        for i in range(N + 1):
            jobs.append(a + (b - a) * i / N)
            meta.append((k, i))
    with Pool(4) as pool:
        labels = [x[0] for x in pool.map(_one, jobs, chunksize=8)]
        changes = []
        for j in range(len(jobs) - 1):
            if meta[j][0] == meta[j + 1][0] and labels[j] != labels[j + 1]:
                changes.append((meta[j][0], jobs[j], jobs[j + 1], labels[j]))
        res = pool.map(_bisect, [(p, q, la) for _, p, q, la in changes])
    out = []
    for (k, _, _, _), (p, on, below, above) in zip(changes, res):
        name = segs[k][0]
        out.append({"axis": name, "beta": p, "on": on, "below": below, "above": above})
        flag = "transitional" if on not in (below, above) else ""
        print(f"{name:32s} {below:6s} | {on:8s} | {above:6s} at β {np.round(p, 6).tolist()} {flag}", flush=True)
    json.dump(out, open("axis_crossings.json", "w"), indent=1)


if __name__ == "__main__":
    main()
