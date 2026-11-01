"""The wall shapes of the kidecaic / iodecaic maps, reclassified cleanly (wall_shapes_<sub>.json).

snub_map.py registered some polytope types at the far end of a bisection (a point ~1e-11 from the wall, classified
with the strict hull tolerance), which is not a sound place to classify. Here every wall point of the map (bisected,
then put exactly on its surface by Newton's method) is classified with the wall tolerance (1e-7), and the polytopes
found are joined by their incidences (group.same_polytope) and compared with the map's region types (grid points, far
from any wall). A wall shape is one that is none of the regions.

    DECAIC_SUB=ki python wall_shapes.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

import group as G


def job(b):
    info = G.classify(b, 1e-7)
    return (G.signature(info), info["key"]) if info else ("none", None)


def edges(sg):
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += int(n)
    return dict(sorted(e.items()))


def main():
    d = json.load(open(G.fname("snub_map")))
    regions = [t for t in d["types"] if t["grid_regions"] or t["fold_points"]]
    pts = [w for w in d["wall_points"] if not w["surface"].startswith("none")]
    with Pool(4) as pool:
        R = pool.map(job, [w["point"] for w in pts], chunksize=4)
    types = []                                          # ({signature, beta, region (map type id or None), points})
    for t in regions:
        types.append({"signature": t["signature"], "beta": t["example"], "region": t["type"],
                      "fold": not t["grid_regions"], "walls": Counter(), "points": []})
    keyto = {}
    for w, (sg, k) in zip(pts, R):
        if k is None:
            continue
        if k not in keyto:
            nv = sg.split(":")[0]
            for i, t in enumerate(types):
                if t["signature"].split(":")[0] == nv and G.same_polytope(w["point"], t["beta"]):
                    keyto[k] = i
                    break
            else:
                types.append({"signature": sg, "beta": w["point"], "region": None, "fold": False, "walls": Counter(),
                              "points": []})
                keyto[k] = len(types) - 1
        t = types[keyto[k]]
        t["walls"][w["surface"]] += 1
        t["points"].append(w["point"])
    out = []
    for i, t in enumerate(types):
        kind = "region" if t["region"] is not None and not t["fold"] else ("wall (fold)" if t["fold"] else "wall")
        if kind == "region" and not t["walls"]:
            pass
        out.append({"kind": kind, "map_type": t["region"], "vertices": int(t["signature"].split(":")[0]),
                    "cells": sum(int(a) for a in re.findall(r"(\d+)x\[", t["signature"].split("|")[0])),
                    "edges_by_valence": edges(t["signature"]), "surfaces": dict(t["walls"]),
                    "seed_beta": [float(v) for v in t["beta"]], "wall_points": len(t["points"]), "signature": t["signature"]})
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": out}, open(G.fname("wall_shapes"), "w"), indent=1)
    for o in out:
        print(f"{o['kind']:11} {o['vertices']}v {o['cells']}c edges {o['edges_by_valence']}  wall points {o['wall_points']} "
              f"on {list(o['surfaces'])[:2]}")


if __name__ == "__main__":
    main()
