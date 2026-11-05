"""Is each face's polychoron the same all over the face? (face_check.json; as decaic_atlas/face_check.py)

Five more exact points inside every face of dimension 1-3 (positive random weights on its vertices) are classified and
compared with the face's own point (same key, or the same incidences).

    python face_check.py
"""
from __future__ import annotations

import json
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

import group as G
from arrangement import Arrangement
from qsqrt2 import K


def job(item):
    p0, pts = item
    k0 = G.label(p0)[1]
    return [q for q in pts if G.label(q)[1] != k0 and not G.same_polytope(p0, q)]


def kparse(v):
    return K(Fraction(v[0]), Fraction(v[1]))


def main():
    A = json.load(open("arrangement.json"))
    arr = Arrangement([tuple(kparse(v) for v in p) for p in A["planes"]]).build()
    rng = np.random.default_rng(1)
    items, meta = [], []
    for f in A["faces"]:
        if f["dim"] == 0:
            continue
        s = tuple(f["signs"])
        vs = [x for x, t in arr.vertices if all(a == 0 or a == b for a, b in zip(t, s))]
        pts = []
        while len(pts) < 5:
            w = [int(v) for v in rng.integers(1, 50, len(vs))]
            tot = sum(w)
            q = tuple(sum((x[i] * wi for wi, x in zip(w, vs)), K(0)) / tot for i in range(4))
            if arr.signs(q) == s:
                pts.append([float(v) for v in q])
        items.append((f["point_float"], pts))
        meta.append(f)
    with Pool(4) as pool:
        R = pool.map(job, items, chunksize=2)
    bad = [(m["dim"], m["point_float"], b) for m, b in zip(meta, R) if b]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "faces_checked": len(items),
               "points_checked": 5 * len(items), "faces_not_constant": bad}, open("face_check.json", "w"), indent=1)
    print(f"{len(items)} faces, {5 * len(items)} points: {len(bad)} faces not constant")
    for b in bad[:10]:
        print(b)


if __name__ == "__main__":
    main()
