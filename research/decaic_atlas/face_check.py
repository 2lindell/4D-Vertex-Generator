"""Is each face's polychoron the same all over the face? (face_check.json)

For every face of dimension 1-3 of the arrangement, five more points inside it (positive random weights on the face's
vertices, exact rationals) are classified; each must be the same polytope as the face's own point (same arrangement
key, or, where the key differs, the same incidences: group.same_polytope).

    python face_check.py
"""
from __future__ import annotations

import json
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

import group as G
from arrangement import Arrangement


def job(item):
    p0, pts = item
    k0 = G.label(p0)[1]
    bad = []
    for q in pts:
        k = G.label(q)[1]
        if k != k0 and not G.same_polytope(p0, q):
            bad.append(q)
    return bad


def main():
    A = json.load(open(G.fname("arrangement")))
    arr = Arrangement([tuple(p) for p in A["planes"]]).build()
    rng = np.random.default_rng(1)
    items, meta = [], []
    for f in A["faces"]:
        if f["dim"] == 0:
            continue
        s = tuple(f["signs"])
        vs = [x for x, t in arr.vertices if all(a == 0 or a == b for a, b in zip(t, s))]
        pts = []
        while len(pts) < 5:
            w = [Fraction(int(v)) for v in rng.integers(1, 50, len(vs))]
            q = tuple(sum(wi * x[i] for wi, x in zip(w, vs)) / sum(w) for i in range(4))
            if arr.signs(q) == s:
                pts.append([float(v) for v in q])
        items.append(([float(Fraction(v)) for v in f["point"]], pts))
        meta.append(f)
    with Pool(4) as pool:
        R = pool.map(job, items, chunksize=4)
    bad = [(m["dim"], m["point"], b) for m, b in zip(meta, R) if b]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "faces_checked": len(items),
               "points_checked": 5 * len(items), "faces_not_constant": bad}, open(G.fname("face_check"), "w"), indent=1)
    print(f"{len(items)} faces, {5 * len(items)} points: {len(bad)} faces not constant")
    for b in bad[:10]:
        print(b)


if __name__ == "__main__":
    main()
