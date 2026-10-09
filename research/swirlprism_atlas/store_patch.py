"""Store an exactly built patch (corners given as golden betas) in fexact_patches.json, with its edge planes and its
check against the traced grid; edge shapes are labelled afterwards (fexact.py split_straight_edges / label_edges).

    from store_patch import store
    store("X20", "n(0,1,-1,0)", [[1, phi**2, phi**2, 0], ...], built="...")
"""
from __future__ import annotations

import json

import numpy as np

from fexact import NORMALS, check_patch, load_walls


def store(target, wall, corners, built, key=None, curved=None):
    load_walls()
    C = np.array(corners, float)
    C = C / C.sum(axis=1, keepdims=True)
    n = np.asarray(NORMALS[wall], float)
    edges = []
    for k in range(len(C)):
        a, b = C[k], C[(k + 1) % len(C)]
        m = np.linalg.svd(np.vstack([a, b, n]))[2][-1]
        edges.append([(m / m[np.argmax(np.abs(m))]).tolist(), "", ""])
    allp = json.load(open("fexact_patches.json"))
    allp[key or f"{target} {wall}"] = {"target": target, "wall": wall, "normal": NORMALS[wall], "corners": C.tolist(),
                                       "edges": edges, "curved": curved or [],
                                       "check": list(check_patch(target, wall, C.tolist(), margin=0.01 / 3)),
                                       "built": built}
    json.dump(allp, open("fexact_patches.json", "w"), indent=1)
    return allp[key or f"{target} {wall}"]["check"]
