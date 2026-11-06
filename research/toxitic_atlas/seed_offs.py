"""Seed .off files of the toxitic, oxitic and icopyritic shapes (new_shapes/off/<group>/): each shape's polychoron at
the seed the map found it at (toxitic_shapes.json; icopyritic: optimize_central.ICOPYRITIC), built with its own sheet's
group. Unnamed shapes are numbered as in optimize_central.py.

    python seed_offs.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTIC = os.path.join(ROOT, "contic_atlas")
CODE = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, %r)
import group as G
from four_d_vertex_generator.off import to_4off
for s in json.loads(sys.stdin.read()):
    b = np.asarray(s["seed"], float); b = b / b.sum()
    V, Vn, idx, faces, cells = G.polytope(b, s["hull_tol"])
    d = os.path.join(%r, "new_shapes", "off", s["group"]); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{s['group']}_{s['kind']}_{s['name']}_{len(V)}v_{len(cells)}c_seed.off")
    open(f, "w").write(to_4off(Vn, faces, cells))
    print(os.path.basename(f), flush=True)
'''


def main():
    sys.path.insert(0, HERE)
    import optimize_central as OC
    for sub in ("tox", "ox", "oxb", "icp", "icpb"):
        S = OC.shapes_for(sub)
        r = subprocess.run([sys.executable, "-c", CODE % (CONTIC, ROOT)], input=json.dumps(S), capture_output=True,
                           text=True, cwd=CONTIC, env={**os.environ, "CONTIC_SUB": sub})
        print(sub, len(S), r.stdout.count("\n"), r.stderr[-300:], flush=True)


if __name__ == "__main__":
    main()
