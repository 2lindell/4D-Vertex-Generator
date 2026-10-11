"""The line shape along every edge of every curved wall: each run of a wall's outline (one shape past it, or a face
of the cell) is now exact, so its middle point lies on the edge; classified with the hull tolerance relaxed, it gives
the line shape there (a D or C line along a face or mirror, X18 along a seam, ...). Written to edge_lines.json as
{wall: [[shape past, line shape], ...]}.

    python edge_lines.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

from edge_relabel import lab_edge


def main():
    X = json.load(open("xsurfaces.json"))
    jobs = []
    for tid, sf in X.items():
        for l in sf["outline"]:
            runs, cur = [], []
            for q in l:
                if cur and q["beyond"] != cur[-1]["beyond"]:
                    runs.append(cur)
                    cur = []
                cur.append(q)
            if cur:
                runs.append(cur)
            for r in runs:
                if len(r) >= 3:
                    jobs.append((tid, r[0]["beyond"], r[len(r) // 2]["beta"]))
    with Pool(4) as pool:
        labs = pool.map(lab_edge, [b for _, _, b in jobs])
    out = {}
    for (tid, past, _), lab in zip(jobs, labs):
        pair = [past, lab]
        if pair not in out.setdefault(tid, []):
            out[tid].append(pair)
    json.dump(out, open("edge_lines.json", "w"), indent=1)
    for tid, v in out.items():
        print(tid, sorted({l for _, l in v if not l.startswith("new:")}))


if __name__ == "__main__":
    main()
