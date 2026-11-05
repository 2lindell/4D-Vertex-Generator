"""Walls: bisect label changes between neighbouring grid samples and name the coplanarity that causes each.

A wall is a surface where five orbit points lie in one hyperplane (the hull's combinatorics change there). The five are
named by group elements, up to translating the whole set by the group.

    python walls.py survey_006.json walls_raw.json
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

import domain as D

E = D.ELEMENTS
N = len(E)
# multiplication table: MUL[i][j] = index of E[i] @ E[j]
_key = {tuple(np.round(m, 5).ravel() + 0.0): i for i, m in enumerate(E)}


def _idx(m: np.ndarray) -> int:
    k = tuple(np.round(m, 5).ravel() + 0.0)
    return _key[k]


MUL = np.array([[_idx(E[i] @ E[j]) for j in range(N)] for i in range(N)])
IDENT = _idx(np.eye(4))


def canonical(subset: tuple[int, ...]) -> tuple[int, ...]:
    """Smallest of the 24 left translates h.S of a subset S of group elements."""
    return min(tuple(sorted(MUL[h][list(subset)])) for h in range(N))


def det5(seed: np.ndarray, subset: tuple[int, ...]) -> float:
    pts = E[list(subset)] @ seed
    return float(np.linalg.det(np.hstack([np.ones((5, 1)), pts])))


_COMBOS = None


def near_zero_subsets(seed: np.ndarray, tol: float) -> set[tuple[int, ...]]:
    """Canonical 5-subsets whose points are coplanar at the seed (vectorised)."""
    global _COMBOS
    others = [i for i in range(N) if i != IDENT]
    if _COMBOS is None:
        _COMBOS = np.array(list(itertools.combinations(others, 4)), dtype=np.int32)
    pts = E @ seed
    idx = np.concatenate([np.full((len(_COMBOS), 1), IDENT, np.int32), _COMBOS], axis=1)
    M = np.concatenate([np.ones((len(idx), 5, 1)), pts[idx]], axis=2)
    d = np.abs(np.linalg.det(M))
    return {canonical(tuple(int(x) for x in idx[i])) for i in np.flatnonzero(d < tol)}


def bisect(a: np.ndarray, b: np.ndarray, la: str, steps: int = 45) -> np.ndarray:
    for _ in range(steps):
        m = (a + b) / 2
        if D.signature(D.classify(D.seed_from_chart(m))) == la:
            a = m
        else:
            b = m
    return (a + b) / 2


def main(survey: str, out: str):
    d = json.load(open(survey))
    P = np.array(d["points"])
    h = d["h"]
    sigs = d["sigs"]
    lookup = {}
    for i, p in enumerate(P):
        lookup[tuple(np.round((p - P.min(0)) / h).astype(int))] = i
    walls = []
    for key, i in lookup.items():
        for axis in range(3):
            k2 = list(key)
            k2[axis] += 1
            j = lookup.get(tuple(k2))
            if j is None or sigs[i] == sigs[j]:
                continue
            q = bisect(P[i], P[j], sigs[i])
            walls.append({"pair": [sigs[i], sigs[j]], "u": q.tolist()})
    json.dump(walls, open(out, "w"))
    print(len(walls), "boundary crossings")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
