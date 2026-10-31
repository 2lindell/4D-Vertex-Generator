"""For edges of curved walls with no shape found on them yet: five points along each, bisected across the edge on the
wall's surface (edge_shapes._job) onto the edge itself, classified with the hull tolerance relaxed (edge_relabel).
Added to edge_lines.json as [shape past, shape on the edge, count].

    python edge_fill_missing.py X10:X12 X51:X74 ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from edge_relabel import lab_edge
from edge_shapes import _job


def _one(args):
    tid, past, x = args
    out = _job((tid, past, x))
    m = out[4]
    return tid, past, (lab_edge(m) if m is not None else None)


def main(specs):
    X = json.load(open("xsurfaces.json"))
    jobs = []
    for spec in specs:
        tid, past = spec.split(":", 1)
        P = [q["beta"] for l in X[tid]["outline"] for q in l if q["beyond"] == past]
        for f in (0.15, 0.35, 0.5, 0.65, 0.85):
            jobs.append((tid, past, P[int(f * (len(P) - 1))]))
    with Pool(4) as pool:
        res = pool.map(_one, jobs)
    EL = json.load(open("edge_lines.json"))
    got = {}
    for tid, past, lab in res:
        if lab:
            got.setdefault((tid, past), Counter())[lab] += 1
    for (tid, past), c in got.items():
        print(tid, past, c.most_common())
        for lab, n in c.items():
            EL.setdefault(tid, []).append([past, lab, n])
    json.dump(EL, open("edge_lines.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
