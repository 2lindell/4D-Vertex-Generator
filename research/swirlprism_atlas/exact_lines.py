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
from exact_math import GG, grad, nrm, project

OUT = "exact_lines.json"
GINV = np.linalg.inv(GG)


class Image:
    """A library surface carried by a symmetry g of the extended group: f(x) = s(g^-1 x)."""

    def __init__(self, s, gi):
        self.s, self.gi, self.name = s, gi, f"{s.name}@{gi}"

    def f(self, X):
        return self.s.f(np.atleast_2d(X) @ GINV[self.gi].T)


def second_surface(P, first, LIB):
    """The image of a library surface that carries the whole piece P (to 1e-5) and crosses the surface `first` there
    (not `first` itself): the one closest to every point of P. For a piece whose listed surfaces are all one plane."""
    mid = P[len(P) // 2]
    t = P[min(len(P) - 1, len(P) // 2 + 1)] - P[max(0, len(P) // 2 - 1)]
    g = grad(first, mid)
    side = np.cross(np.linalg.svd(np.ones((1, 4)))[2][1:] @ t, np.linalg.svd(np.ones((1, 4)))[2][1:] @ g) @ \
        np.linalg.svd(np.ones((1, 4)))[2][1:]
    off = mid + 0.02 * side / np.linalg.norm(side)        # (on `first`, off the piece: a copy of `first` vanishes here)
    best = (np.inf, None)
    for s in LIB.values():
        X = np.einsum("gij,pj->gpi", GINV, P)
        F = np.stack([np.ravel(s.f(X[k])) for k in range(len(GINV))])
        e = 1e-7
        Gr = (np.stack([np.stack([np.ravel(s.f(np.einsum("ij,pj->pi", GINV[k], P + e * u)))
                                  for u in np.eye(4)], -1) for k in range(len(GINV))]) - F[..., None]) / e
        Gr = Gr - Gr.mean(-1, keepdims=True)              # (within the betas' sum-zero space)
        D = (np.abs(F) / (np.linalg.norm(Gr, axis=-1) + 1e-300)).max(1)
        fo = np.abs(np.array([float(np.ravel(s.f((GINV[k] @ off)[None]))[0]) for k in range(len(GINV))]))
        D[fo < 1e-9] = np.inf
        k = int(D.argmin())
        if D[k] < best[0]:
            best = (D[k], Image(s, k))
    return best[1] if best[0] < 1e-5 else None


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
            if best < 1e-6:
                # (the listed surfaces are all one plane (a pair of opposite normals): the second is an image of a
                # library surface through the whole piece)
                S2 = second_surface(P, pair[0], LIB)
                if S2 is None:
                    raise ValueError(f"{L}: no second surface for a piece")
                pair = (pair[0], S2)
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
