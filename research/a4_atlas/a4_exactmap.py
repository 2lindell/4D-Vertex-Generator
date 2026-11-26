"""Faces of the exact map: a4_arcs' arcs walked on every surface (a4_faces.faces_on), each face given points just
inside it (a4_closed.probes_of) for a4_walls to read whole pieces at. Nothing is classified here.

    python a4_exactmap.py kidecaic        (a4_exactmap_<group>.pkl; then python a4_walls.py kidecaic exact)
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_closed as Cl  # noqa: E402
import a4_exact as E  # noqa: E402
import a4_faces as A  # noqa: E402
import a4_pieces as P  # noqa: E402

X = SURFS = None


def walk(k):
    S = SURFS[k]
    segs, Pts = X["segments"], X["points"]
    out = []
    for loops, hc in A.faces_on(k, S, segs, Pts):
        cp = Cl.probes_of(S, k, loops[0], hc[0], segs, frac=0.02)
        pr = Cl.probes_of(S, k, loops[0], hc[0], segs, minlen=2e-3) or cp
        out.append({"surface": k, "loop": loops[0], "cycle": hc[0], "probes": pr or [], "cut probes": cp or pr or []})
    return k, out


def main(group):
    global X, SURFS
    X = pickle.load(open(os.path.join(HERE, f"a4_arcs_{group}.pkl"), "rb"))
    SURFS = [P.surf_of(n) for n in X["surfaces"]]
    A.SURFS = SURFS
    E.SING = 1e-8
    t0 = time.time()
    faces = []
    with Pool(4) as pool:
        for k, out in pool.imap_unordered(walk, range(len(SURFS))):
            faces += out
            print(f"  surface {k}: {len(out)} faces ({time.time() - t0:.0f}s)", flush=True)
    faces.sort(key=lambda F: F["surface"])
    X["faces"] = faces
    pickle.dump(X, open(os.path.join(HERE, f"a4_exactmap_{group}.pkl"), "wb"))
    print(f"{len(faces)} faces on {len(SURFS)} surfaces")


if __name__ == "__main__":
    for g in sys.argv[1:]:
        main(g)
