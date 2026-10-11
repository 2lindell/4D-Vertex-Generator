"""The shapes at the exact corners of the curved walls: each corner edge_snap.py solved (where the wall meets the
surfaces of its two edges there) is classified with the hull tolerance relaxed (it is exact to ~1e-15, the shape
there is a point or line shape), and written to corner_points.json as {wall: [{beta, shape, edges: [past A, past B]}]}.

    python corner_points.py dir_of_edge_snap_outputs
"""
from __future__ import annotations

import glob
import json
import os
import sys
from multiprocessing import Pool

from edge_relabel import lab_edge


def main(d):
    jobs = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        for tid, sf in json.load(open(f)).items():
            for beta, la, lb in sf.get("corner_points", []):
                jobs.append((tid, beta, la, lb))
    with Pool(4) as pool:
        labs = pool.map(lab_edge, [b for _, b, _, _ in jobs])
    out = {}
    for (tid, beta, la, lb), lab in zip(jobs, labs):
        out.setdefault(tid, []).append({"beta": beta, "shape": lab, "edges": [la, lb]})
        print(tid, lab[:40], [round(x / max(beta), 5) for x in beta], flush=True)
    json.dump(out, open("corner_points.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
