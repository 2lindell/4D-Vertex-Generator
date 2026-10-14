"""Classify seeds the way the cell atlas labels them (canonical id, X id, or raw counts)."""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
from cell_atlas2 import identify, load_refs
from cellframe import seed_from_beta
from classify import classify, signature

REFS = load_refs()[0]
XIDS = json.load(open("atlas_xids.json")) if os.path.exists("atlas_xids.json") else {}


def label_of_sig(sig: str) -> str:
    cid, xkey, _ = identify(sig, REFS)
    if cid:
        return cid
    return XIDS.get(xkey, "new:" + sig)


def _one(beta) -> tuple[str, str]:
    try:
        info = classify(seed_from_beta(np.asarray(beta, float)))
        sig = signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items())
    except Exception as exc:  # degenerate hulls
        return "ERR", repr(exc)
    return label_of_sig(sig), sig


def labels(betas, procs: int = 4) -> list[tuple[str, str]]:
    with Pool(procs) as pool:
        return pool.map(_one, [list(map(float, b)) for b in betas])


def boundary(b0, b1, lab0: str | None = None, lab1: str | None = None, steps: int = 40) -> tuple[float, str, str]:
    """Bisect the segment b0 -> b1 (barycentric) for where the label changes; returns (t, label below, label above)."""
    b0, b1 = np.asarray(b0, float), np.asarray(b1, float)
    lab0 = lab0 or _one(b0)[0]
    lab1 = lab1 or _one(b1)[0]
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        mid = (lo + hi) / 2
        if _one(b0 + mid * (b1 - b0))[0] == lab0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, lab0, _one(b0 + hi * (b1 - b0))[0]
