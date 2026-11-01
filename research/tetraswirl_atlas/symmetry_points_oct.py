"""Points where the octahedral swirl domain's extra-symmetry lines meet, and the symmetry there (symmetry_points_oct.json).

The lines: the normalizer's half-turn axes (normalizer_oct.json, 96 symmetries) and the 6/3 gyrogon ghost girdle
M01-M12 (144). Their ends, every crossing of two of them, and every crossing of one with a ring of the group (its
swirl ring and cross rings) (straight segments in the betas: the seed is linear in
them) are found exactly, the crossings by solving the two lines' equations; at each point the polychoron's vertices
are counted and its symmetry measured against every group found containing the group (its normalizer and the library's
supergroups of it): the number of their maps that keep the seed's orbit.

    SWIRL_GROUP=oct python symmetry_points_oct.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import domain as D
import normalizer_oct as nz
import supergroup_lines_oct as sl
from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

OUT = "symmetry_points_oct.json"
S3 = 3 ** 0.5
EXACT = {0.0: "0", 0.5: "1/2", (S3 - 1) / 2: "(√3−1)/2", (3 - S3) / 6: "(3−√3)/6", 1 - S3 / 2: "1−√3/2", 2 - S3: "2−√3",
         1 / S3: "1/√3", S3 - 1: "√3−1", (S3 + 1) / 6: "(√3+1)/6", 1 / 3: "1/3", 1 / 4: "1/4", 1.0: "1",
         (3 + S3) / 6: "(3+√3)/6", S3 / 6: "√3/6", (2 - S3) / 2: "(2−√3)/2", (S3 - 1) / 4: "(√3−1)/4",
         (3 - S3) / 4: "(3−√3)/4", 1 / 6: "1/6", 2 / 3: "2/3", S3 / 3: "√3/3", (S3 + 1) / 4: "(√3+1)/4"}


def exact_text(b):
    out = []
    for x in b:
        k = min(EXACT, key=lambda v: abs(v - x))
        out.append(EXACT[k] if abs(k - x) < 1e-9 else f"{x:.9f}")
    return "(" + ", ".join(out) + ")"


def vertices(b):
    x = np.asarray(b, float) @ D.CELL
    return len(nz.orbit(x / np.linalg.norm(x)))


def main():
    N = json.load(open("normalizer_oct.json"))
    segs = [("half-turn axis", np.array(l["from"]), np.array(l["to"])) for l in N["loci"] if l["dim"] == 1]
    segs.append(("6/3 gyrogon girdle", np.array([0.5, 0.5, 0, 0.0]), np.array([0, 0.5, 0.5, 0.0])))
    # (the group's own rings too: where an extra-symmetry line crosses one, the point there has both symmetries)
    from make_atlas_oct import RINGS
    rings = [(name.split(" (")[0], np.array(a, float), np.array(b, float)) for name, kind, nv, a, b in RINGS]
    # (the groups to measure symmetry against: the normalizer and every library group containing G)
    groups = []
    for nm in [f"h4_tubical_octahedral_{f}_{n}" for f in ("dihedral", "dihedral_index2_2n", "dihedral_index6_6n")
               for n in range(1, 13)]:
        try:
            H = np.stack(group_elements(named_symmetry(nm), 20000))
        except Exception:
            continue
        if sl.GK <= {tuple(np.round(h, 6).ravel()) for h in H} and len(H) > 48:
            groups.append((nm, H))
    import normalizer_oct
    A = nz.orbit(D.CELL[0] / np.linalg.norm(D.CELL[0])); F = np.array([0, 1, 1, 1.0]) @ D.CELL
    Bo = nz.orbit(F / np.linalg.norm(F)); S = np.vstack([A, Bo])
    idx = next(list(c) for c in itertools.combinations(range(16), 4) if abs(np.linalg.det(S[list(c)])) > 1e-6)
    P0 = S[idx]; Gram = P0 @ P0.T; Nm = {}
    for t in itertools.permutations(range(16), 4):
        Q = S[list(t)]
        if np.abs(Q @ Q.T - Gram).max() > 1e-9:
            continue
        k = np.linalg.solve(P0, Q).T
        if np.abs(k @ k.T - np.eye(4)).max() < 1e-9 and all(np.min(np.linalg.norm(S - s, axis=1)) < 1e-9 for s in S @ k.T):
            Nm[nz.key(k)] = k
    groups.append(("normalizer", np.stack(list(Nm.values()))))

    def symmetry(b):
        x = np.asarray(b, float) @ D.CELL
        x = x / np.linalg.norm(x)
        O = nz.orbit(x)
        best = (48, "the group")
        keep = {nz.key(g) for g in D.ELEMENTS}
        for nm, H in groups:
            ok = [h for h in H if all(np.min(np.linalg.norm(O - q, axis=1)) < 1e-7 for q in O @ h.T)]
            for h in ok:
                keep.add(nz.key(h))
        # (every map found that keeps the orbit: together the polychoron's symmetries within the groups tested)
        return len(keep)
    pts = []
    pairs = list(itertools.combinations(segs, 2)) + [(r, x) for r in rings for x in segs]
    for (n1, a1, b1), (n2, a2, b2) in pairs:
        # (where the two segments meet: a1 + s (b1 - a1) = a2 + t (b2 - a2), as betas summing to 1)
        M = np.stack([b1 - a1, -(b2 - a2)], 1)
        st, res, *_ = np.linalg.lstsq(M, a2 - a1, rcond=None)
        s, t = st
        x = a1 + s * (b1 - a1)
        if np.linalg.norm(M @ st - (a2 - a1)) < 1e-9 and -1e-9 <= s <= 1 + 1e-9 and -1e-9 <= t <= 1 + 1e-9:
            pts.append(("meet", f"{n1} and {n2}", x))
    for n, a, b in segs:
        for e in (a, b):
            pts.append(("end", n, e))
    seen, out = [], []
    for kind, what, x in pts:
        x = np.where(np.abs(x) < 1e-12, 0, x)
        x = x / x.sum()
        if any(np.linalg.norm(x - y) < 1e-8 for y in seen):
            for o in out:
                if np.linalg.norm(np.array(o["beta"]) - x) < 1e-8 and what not in o["on"]:
                    o["on"].append(what)
            continue
        seen.append(x)
        out.append({"beta": x.tolist(), "exact": exact_text(x), "on": [what], "kind": kind,
                    "vertices": vertices(x), "symmetry": symmetry(x)})
    for o in out:
        o["lines_through"] = len(o["on"])
        print(f"{o['exact']:52s} vertices {o['vertices']:3d}  symmetry {o['symmetry']}  on: {'; '.join(o['on'])}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "points": out}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
