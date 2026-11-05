"""Random search for thin regions.   python dense.py N SEED OUT.json"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

import domain as D
from survey import _work


def main(n: int, seed: int, out: str):
    rng = np.random.default_rng(seed)
    V = D.domain_vertices()
    lo, hi = V.min(0), V.max(0)
    pts = []
    while len(pts) < n:
        X = rng.uniform(lo, hi, size=(4 * n, 3))
        pts.extend(X[D.in_domain(X)].tolist())
    pts = np.array(pts[:n])
    with Pool(2) as pool:
        sigs = pool.map(_work, [tuple(p) for p in pts], chunksize=64)
    json.dump({"points": np.round(pts, 9).tolist(), "sigs": sigs}, open(out, "w"))
    for s, k in Counter(sigs).most_common():
        print(k, s)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
