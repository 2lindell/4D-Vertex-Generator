"""The shapes of the decaic domain: each face of the exact arrangement classified (shapes.json).

Each face's point (arrangement.json) is classified (group.classify: vertex, cell, face and edge classes and the
arrangement key), and each cell class is described by its faces' sizes. Faces with one key are one polytope; faces with
different keys but the same counts are compared by their incidences (group.same_polytope), since one polytope can have
its cells grouped differently. A shape's degrees of freedom are the largest dimension of the faces it fills (3 a
region, 2 a wall, 1 a line, 0 a point).

    python shapes.py
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

import group as G
from arrangement import Arrangement

OUT = G.fname("shapes")
KIND = {3: "region", 2: "wall", 1: "line", 0: "point"}


def describe(beta):
    """The polychoron's classes, with each cell class's face sizes, and its arrangement key."""
    info = G.classify(beta, 1e-9)
    if info is None:
        return None
    V5, V, faces, cells = G.polytope(beta, 1e-9)
    from scipy.spatial import cKDTree
    tree = cKDTree(V5)
    perms = []
    for g in G.GENS:
        perms.append(tree.query(V5 @ g.T)[1])
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    cls = G._classes(cell_sets, perms)
    cdesc = Counter()
    for cl in cls:
        c = cells[cl[0]]
        sizes = Counter(len(faces[f]) for f in c)
        prof = "+".join(f"{n}x{k}" for k, n in sorted(sizes.items()))
        cdesc[(len(cell_sets[cl[0]]), prof)] += len(cl)
    return {"vertices": info["vertices"], "key": info["key"], "signature": G.signature(info),
            "cells": sorted([[n, v, p] for (v, p), n in cdesc.items()], key=lambda r: (-r[1], r[2])),
            "n_cells": len(cells), "n_faces": sum(n for n, _ in info["faces"]), "n_edges": sum(n for n, _ in info["edges"])}


def job(point):
    b = [float(Fraction(v)) for v in point]
    try:
        return describe(b)
    except Exception as e:
        return {"error": str(e)}


def cells_text(d):
    return " + ".join(f"{n}x{v}v({p})" for n, v, p in d["cells"])


def main():
    global swapped
    t0 = time.time()
    A = json.load(open(G.fname("arrangement")))
    faces = A["faces"]
    # (the swap reverses beta: on the fold b1 = b4 a face and its image are the same seeds)
    arr = Arrangement([tuple(p) for p in A["planes"]])

    def swapped(f):
        return list(arr.signs(tuple(Fraction(v) for v in f["point"])[::-1]))
    with Pool(4) as pool:
        R = pool.map(job, [f["point"] for f in faces], chunksize=4)
    bykey = {}
    for f, d in zip(faces, R):
        f["shape"] = d
        if d and "key" in d:
            bykey.setdefault(d["key"], []).append(f)
    # (keys whose polytopes are the same, joined)
    types = []
    for k, fs in bykey.items():
        d = fs[0]["shape"]
        b = [float(Fraction(v)) for v in fs[0]["point"]]
        for t in types:
            e = t["shape"]
            if (e["vertices"], e["n_cells"], e["n_faces"], e["n_edges"]) == (d["vertices"], d["n_cells"], d["n_faces"], d["n_edges"]) \
                    and G.same_polytope(b, t["beta"]):
                t["keys"].append(k)
                t["faces"] += fs
                break
        else:
            types.append({"keys": [k], "faces": list(fs), "shape": d, "beta": b})
    out = []
    for t in types:
        dims = Counter(f["dim"] for f in t["faces"])
        top = max(dims)
        # (its pieces: faces of the top dimension joined through a face of the same shape one dimension down)
        tops = [f for f in t["faces"] if f["dim"] == top]
        lows = [f for f in t["faces"] if f["dim"] == top - 1]
        parent = list(range(len(tops)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def around(sg, faces_):
            return [i for i, f in enumerate(faces_) if all(a == 0 or a == b for a, b in zip(sg, f["signs"]))]
        for lo in lows:
            hit = around(lo["signs"], tops)
            if lo["signs"][4] == 0:             # (on the fold: the swap image is the same seed)
                hit += around(swapped(lo), tops)
            for i in hit[1:]:
                parent[find(i)] = find(hit[0])
        index = {tuple(f["signs"]): i for i, f in enumerate(tops)}
        for i, f in enumerate(tops):
            if f["signs"][4] == 0 and tuple(swapped(f)) in index:
                parent[find(index[tuple(swapped(f))])] = find(i)
        pieces = len({find(i) for i in range(len(tops))})
        best = next(f for f in t["faces"] if f["dim"] == top)
        out.append({"vertices": t["shape"]["vertices"], "degrees": top, "kind": KIND[top],
                    "cells": t["shape"]["cells"], "cells_text": cells_text(t["shape"]),
                    "n_cells": t["shape"]["n_cells"], "n_faces": t["shape"]["n_faces"], "n_edges": t["shape"]["n_edges"],
                    "keys": t["keys"], "pieces": pieces, "faces_by_dim": {str(k): v for k, v in sorted(dims.items())},
                    "example": best["point"], "example_float": [round(float(Fraction(v)), 6) for v in best["point"]],
                    "example_signs": best["signs"],
                    "signature": t["shape"]["signature"]})
    out.sort(key=lambda r: (-r["degrees"], r["vertices"], r["cells_text"]))
    bad = [f for f in faces if not f["shape"] or "key" not in f["shape"]]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out, "unclassified": len(bad),
               "seconds": round(time.time() - t0, 1)}, open(OUT, "w"), indent=1)
    for r in out:
        print(f"{r['kind']:6} {r['vertices']:4}v {r['n_cells']:3}c  {r['cells_text']}   faces {r['faces_by_dim']} pieces {r['pieces']}  "
              f"e.g. {r['example_float']}")
    print(f"{len(out)} shapes; {len(bad)} faces unclassified; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
