"""The faces of the chamber where a subgroup's orbits differ from the tessic ones (face_survey<tag>.json).

On a face beta_k = 0 whose mirror is in the subgroup (tessipyritic: b4 = 0, the sign flip), or where the subgroup is too
small to reach all of a tessic orbit (kidemitessic, index 4), a seed's orbit is only part of the tessic polytope's
vertices: the face carries shapes of its own (sadi on b4 = 0). Each such face is surveyed on a two-dimensional grid
(shifted off the lattice), every point classified, the polytopes found counted (with their cells and edge valences) and
their grid regions joined.

    TESSIC_SUB=pyr python face_survey.py k [N]          (k: the beta that is 0, 1..4)
"""
import json
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

import group as G


def job(b):
    return G.label(b)


def main():
    k = int(sys.argv[1]) - 1
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    pts, P = [], []
    shift = np.array([np.sqrt(2) - 1.4, np.sqrt(3) - 1.7, np.sqrt(5) - 2.2]) * 0.37
    for i in range(1, N):
        for j in range(1, N - i):
            l = N - i - j
            if l < 1:
                continue
            q = np.array([i, j, l], float) + shift
            b = np.insert(q / q.sum(), k, 0.0)
            pts.append((i, j))
            P.append(b)
    with Pool(4) as pool:
        R = pool.map(job, P, chunksize=16)
    types = {}
    for (sg, key), b in zip(R, P):
        if key is None:
            continue
        t = types.setdefault(key, {"signature": sg, "count": 0, "example": b.tolist()})
        t["count"] += 1
    index = {p: n for n, p in enumerate(pts)}
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for n, (i, j) in enumerate(pts):
        for di, dj in ((1, 0), (0, 1), (1, -1)):
            m = index.get((i + di, j + dj))
            if m is not None and R[n][1] == R[m][1]:
                parent[find(n)] = find(m)
    pieces = Counter((R[n][1], find(n)) for n in range(len(pts)))
    out = []
    for key, t in types.items():
        sg = t["signature"]
        e = Counter()
        for n, v in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
            e[int(v)] += int(n)
        cells = sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))
        out.append({"key": key, "vertices": int(sg.split(":")[0]), "cells": cells, "edges_by_valence": dict(sorted(e.items())),
                    "points": t["count"], "pieces": sum(1 for (kk, _) in pieces if kk == key), "example": t["example"],
                    "signature": sg})
        print(f"{int(sg.split(':')[0]):4}v {cells:4}c  edges {dict(sorted(e.items()))}  {t['count']:4} points in "
              f"{out[-1]['pieces']} piece(s)  e.g. {np.round(t['example'], 4).tolist()}", flush=True)
    json.dump({"face": f"b{k + 1} = 0", "N": N, "types": out}, open(G.fname(f"face_survey_b{k + 1}"), "w"), indent=1)


if __name__ == "__main__":
    main()
