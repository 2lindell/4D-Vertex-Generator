"""More classified seeds, spread evenly through the half-cell, for tracing how each region's pieces connect
(cohesive_domain.py). Writes cohesive_samples.json as [beta, label] pairs, saving as it goes.

    python cohesive_samples.py [n]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from cell_atlas import to_upper
from probe import _one


def seeds(n):
    rng = np.random.default_rng(23)
    return [to_upper(b) for b in rng.dirichlet(np.ones(4), n)]


def _job(b):
    return [list(map(float, b)), _one(b)[0]]


def main(n=6000):
    import os
    out = json.load(open("cohesive_samples.json")) if os.path.exists("cohesive_samples.json") else []
    with Pool(4) as pool:                       # run with OMP_NUM_THREADS=1: one thread per worker
        for k, r in enumerate(pool.imap(_job, seeds(n)[len(out):], chunksize=8), start=len(out)):
            out.append(r)
            if k % 200 == 199:
                json.dump(out, open("cohesive_samples.json", "w"))
    json.dump(out, open("cohesive_samples.json", "w"))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
