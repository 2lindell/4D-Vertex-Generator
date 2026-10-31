"""Relabel the edge points of edge_shapes.json with the hull tolerance relaxed to 1e-7 outright. The bisected edge
points lie within ~1e-12 of the edge but not on it, so the strict classifier (and _lab_loose, which only relaxes when
the strict one fails) sees the shape beside the edge; at 1e-7 the near-coincident facets merge and the line shape
living on the edge (X81 along X42's edge, say) is seen.

    python edge_relabel.py [X42 ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np


def lab_edge(b):
    from classify import classify, signature
    from probe import label_of_sig, seed_from_beta
    b = np.asarray(b, float)
    try:
        info = classify(seed_from_beta(b / b.sum()), hull_tol=1e-7)
    except Exception:
        return "ERR"
    return label_of_sig(signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items()))


def main(tids=None):
    E = json.load(open("edge_shapes.json"))
    jobs = [(k, i, p["beta"]) for k, pts in E["points"].items() for i, p in enumerate(pts)
            if p.get("beta") and (not tids or k.split("|")[0] in tids)]
    with Pool(4) as pool:
        labs = pool.map(lab_edge, [b for _, _, b in jobs], chunksize=2)
    for (k, i, _), l in zip(jobs, labs):
        E["points"][k][i]["on"] = l
    for k, pts in E["points"].items():
        if not tids or k.split("|")[0] in tids:
            E["summary"][k] = Counter(p["on"] for p in pts if p.get("on")).most_common()
            print(k, E["summary"][k], flush=True)
    json.dump(E, open("edge_shapes.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
