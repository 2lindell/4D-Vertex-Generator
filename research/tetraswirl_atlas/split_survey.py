"""Does each shape of the octahedral swirl atlas stand for one polytope? (split_survey_oct.json)

The atlas names a shape by its signature (the sizes of its cell, face and edge classes under the group); two different
polytopes can share one. Each point here is also given its arrangement key (domain.classify: how the cells, faces and
edges fit together at a vertex); where one signature comes with more than one key, a representative of each key is
compared with the others by their incidences (vertices, faces and cells, any relabelling: make_atlas_oct._same_polytope),
since the same polytope can also have its cells grouped differently. Only polytopes that are not the same are a split.

The points, all from what is already found (nothing random):
  * the regions: points of a grid in the betas, each piece's from its exact sign conditions (region_forms_oct.json);
  * the walls: where the walls out of each region piece ended (region_closure_oct.json), put exactly on the surface;
  * the rings: points along each ring piece of the atlas (its lines S1..S4: make_atlas_oct.RINGS and the scanned pieces).

    SWIRL_GROUP=oct python split_survey.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import atlas_model as am
import domain as D
import region_closure as rc
from make_atlas_oct import RINGS, _same_polytope

OUT = f"split_survey_{D.KEY}.json"


def job(item):
    b, tol = item
    x = np.asarray(b, float) @ D.CELL
    info = D.classify(x / np.linalg.norm(x), hull_tol=tol)
    return (D.signature(info), info["vfig"]) if info else ("none", None)


def onto(item):
    surf, q = item
    x, ok = am.project(np.array([q], float), [am.Surf(surf)])
    return (x[0] / x[0].sum()) if ok[0] else None


def main():
    items = []
    G = rc.grid(60)
    gp = am.piece_at(G)
    for pi in range(len(am.PIECES)):
        P = G[gp == pi]
        for b in P[:: max(1, len(P) // 30)][:30]:
            items.append((b, 1e-9, f"region {am.PIECES[pi]}"))
    todo = []
    for p, v in json.load(open(f"region_closure_{D.KEY}.json"))["pieces"].items():
        for e in v["ends"]:
            if e["beyond_type"] and e["beyond_type"] != v["type"] and not e["surface"].startswith("none"):
                P = e.get("points", [])
                todo += [(e["surface"], q) for q in P[:: max(1, len(P) // 25)][:25]]
    with Pool() as pool:
        W = pool.map(onto, todo, chunksize=8)
    items += [(b, 1e-7, f"wall {s}") for (s, _), b in zip(todo, W) if b is not None]
    for name, kind, nv, a, b in RINGS:
        for t in np.linspace(0.02, 0.98, 49):
            items.append(((1 - t) * np.array(a, float) + t * np.array(b, float), 1e-7, f"ring {name}"))
    print(f"{len(items)} points", flush=True)
    with Pool() as pool:
        R = pool.map(job, [(b, tol) for b, tol, _ in items], chunksize=16)
    by = defaultdict(lambda: defaultdict(list))
    for (b, tol, src), (sg, k) in zip(items, R):
        if k:
            by[sg][k].append((np.asarray(b).round(10).tolist(), src))
    out = {"_about": __doc__.strip().split("\n\n")[0], "signatures": {}}
    for sg, keys in by.items():
        reps = {k: v[0][0] for k, v in keys.items()}
        # (keys whose polytopes are the same, joined: what is left are different polytopes)
        groups = []
        for k, b in reps.items():
            for g in groups:
                if _same_polytope(b, reps[g[0]]):
                    g.append(k)
                    break
            else:
                groups.append([k])
        out["signatures"][sg] = {"keys": {k: {"count": len(v), "sources": sorted({s for _, s in v})[:12], "example": v[0][0]}
                                          for k, v in keys.items()},
                                 "polytopes": groups}
        if len(groups) > 1:
            print(f"SPLIT {sg[:90]}", flush=True)
            for g in groups:
                n = sum(len(keys[k]) for k in g)
                print(f"   {n} points; e.g. {np.round(keys[g[0]][0][0], 5).tolist()} from {keys[g[0]][0][1]}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(out['signatures'])} signatures; {sum(len(v['polytopes']) > 1 for v in out['signatures'].values())} cover "
          f"more than one polytope", flush=True)


if __name__ == "__main__":
    main()
