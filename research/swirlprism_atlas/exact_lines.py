"""Each line's verified pieces put exactly on the line (exact_lines.json).

The pieces of each line (exact_defs.json) were traced to about 1e-6: a point of the line E1, which lies in the face
b4 = 0, had b4 up to 1.4e-7. Here every point of every piece is moved onto two of the surfaces the line lies on (the
library surfaces listed for that piece, the pair crossing most cleanly), by Newton steps at full double precision; each
end of a piece is put on the exact point there (exact_points.json, 40 digits) when one lies within 1e-5, or, where the
piece ends on a face of the half-cell (the line going on into the next cell), solved onto that face. How far each piece
moved is recorded.

    python exact_lines.py
"""
from __future__ import annotations

import json

import numpy as np

import core as osd
from pieces import grad, nrm, project

OUT = "exact_lines.json"


def main():
    ex = json.load(open("exact_defs.json"))
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    V = [np.array(p["float"]) for p in json.load(open("exact_points.json"))["points"] if p["exact"]]
    out, worst = {}, 0.0
    for L, v in ex["lines"].items():
        pieces, pairs, moved = [], [], []
        for P, names in zip(v["points"], v["surfaces"]):
            P = np.asarray(P, float)
            cand = [LIB[n] for n in names if n in LIB]
            if len(P) == 0 or len(cand) < 2:
                pieces.append(P.tolist())
                pairs.append([])
                moved.append(None)
                continue
            # (the two surfaces crossing most cleanly along the piece)
            mid = P[len(P) // 2]
            best, pair = -1.0, None
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    a, b = grad(cand[i], mid), grad(cand[j], mid)
                    a, b = a - a.mean(), b - b.mean()
                    s = np.sqrt(max(0.0, 1 - (a @ b) ** 2 / ((a @ a) * (b @ b) + 1e-300)))   # (sine of their angle)
                    if s > best:
                        best, pair = s, (cand[i], cand[j])
            Q = np.array([nrm(project(list(pair), p)) for p in P])
            for k in (0, -1):
                d = [np.linalg.norm(Q[k] - u) for u in V]
                if d and min(d) < 1e-5:
                    Q[k] = V[int(np.argmin(d))]
                    continue
                # (an end on a face of the half-cell, where the line goes on into the next cell: solved onto the face)
                for n in np.vstack([np.eye(4), [[0, 0, 1, -1.0]]]):
                    if abs(Q[k] @ n) < 1e-5:
                        Q[k] = nrm(project(list(pair) + [osd.Plane(n, "face")], Q[k]))
                        break
            m = float(np.linalg.norm(Q - P, axis=1).max())
            worst = max(worst, m)
            pieces.append(Q.tolist())
            pairs.append([pair[0].name, pair[1].name])
            moved.append(m)
        out[L] = {"points": pieces, "surfaces": pairs, "moved": moved}
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "lines": out}, open(OUT, "w"))
    print(f"{len(out)} lines; largest move {worst:.2e}")


if __name__ == "__main__":
    main()
