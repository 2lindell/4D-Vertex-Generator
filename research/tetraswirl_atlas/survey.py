"""Classify a grid over the fundamental domain.   python survey.py SPACING OUT.json"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import domain as D


def _work(u):
    info = D.classify(D.seed_from_chart(np.array(u)))
    return D.signature(info)


def grid(h: float) -> np.ndarray:
    V = D.domain_vertices()
    lo, hi = V.min(0), V.max(0)
    # offset the grid by an irrational-ish fraction so no sample sits on a wall or on the axis
    off = np.array([0.3137, 0.2711, 0.4129]) * h
    axes = [np.arange(l - off_i, hh + h, h) + 0 for l, hh, off_i in zip(lo, hi, off)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return X[D.in_domain(X)]


def main(h: float, out: str):
    pts = grid(h)
    print(len(pts), "points", flush=True)
    with Pool(2) as pool:
        sigs = pool.map(_work, [tuple(p) for p in pts], chunksize=64)
    json.dump({"h": h, "points": np.round(pts, 9).tolist(), "sigs": sigs}, open(out, "w"))
    from collections import Counter

    for s, n in Counter(sigs).most_common():
        print(n, s)


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
