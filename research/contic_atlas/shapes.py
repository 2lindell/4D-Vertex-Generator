"""The shapes of the contic domain: each face of the exact arrangement classified (shapes.json); as
decaic_atlas/shapes.py. A shape's degrees of freedom are the largest dimension of the faces it fills.

    python shapes.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from multiprocessing import Pool

import group as G

OUT = "shapes.json"
KIND = {3: "region", 2: "wall", 1: "line", 0: "point"}


def describe(beta):
    info = G.classify(beta, 1e-9)
    if info is None:
        return None
    V, idx, faces, cells, perms, cc, fc = info["_hull"]
    cell_sets = [frozenset(v for f in c for v in faces[f]) for c in cells]
    cdesc = Counter()
    for cl in cc:
        c = cells[cl[0]]
        sizes = Counter(len(faces[f]) for f in c)
        prof = "+".join(f"{n}x{k}" for k, n in sorted(sizes.items()))
        cdesc[(len(cell_sets[cl[0]]), prof)] += len(cl)
    return {"vertices": info["vertices"], "key": info["key"], "signature": G.signature(info),
            "cells": sorted([[n, v, p] for (v, p), n in cdesc.items()], key=lambda r: (-r[1], r[2])),
            "n_cells": len(cells), "n_faces": len(faces), "n_edges": sum(n for n, _ in info["edges"])}


def job(b):
    try:
        return describe(b)
    except Exception as e:
        return {"error": repr(e)}


def cells_text(d):
    return " + ".join(f"{n}x{v}v({p})" for n, v, p in d["cells"])


def main():
    t0 = time.time()
    A = json.load(open("arrangement.json"))
    faces = A["faces"]
    with Pool(4) as pool:
        R = pool.map(job, [f["point_float"] for f in faces], chunksize=2)
    bykey = {}
    for f, d in zip(faces, R):
        f["shape"] = d
        if d and "key" in d:
            bykey.setdefault(d["key"], []).append(f)
    types = []
    for k, fs in bykey.items():
        d = fs[0]["shape"]
        b = fs[0]["point_float"]
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
        best = next(f for f in t["faces"] if f["dim"] == top)
        out.append({"vertices": t["shape"]["vertices"], "degrees": top, "kind": KIND[top],
                    "cells": t["shape"]["cells"], "cells_text": cells_text(t["shape"]),
                    "n_cells": t["shape"]["n_cells"], "n_faces": t["shape"]["n_faces"], "n_edges": t["shape"]["n_edges"],
                    "keys": t["keys"], "faces_by_dim": {str(k): v for k, v in sorted(dims.items())},
                    "example": best["point"], "example_float": [round(v, 6) for v in best["point_float"]],
                    "example_signs": best["signs"], "signature": t["shape"]["signature"]})
    out.sort(key=lambda r: (-r["degrees"], r["vertices"], r["cells_text"]))
    bad = [f for f in faces if not f["shape"] or "key" not in f["shape"]]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out, "unclassified": len(bad),
               "unclassified_points": [f["point_float"] for f in bad], "seconds": round(time.time() - t0, 1)},
              open(OUT, "w"), indent=1)
    for r in out:
        print(f"{r['kind']:6} {r['vertices']:5}v {r['n_cells']:4}c  {r['cells_text']}   faces {r['faces_by_dim']}  "
              f"e.g. {r['example_float']}")
    print(f"{len(out)} shapes; {len(bad)} faces unclassified; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
