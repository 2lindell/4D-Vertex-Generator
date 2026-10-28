"""Does every wall close? From the exact data alone, on the whole 3-sphere (closure.json).

A wall is closed when each of its edges meets the others at two points or more: going round its edges, each one runs
from a meeting point to another. The meeting points are the exact points (exact_points.json, solved to 40 digits),
carried by every symmetry; the wall is every image of its surfaces (so where it runs on into the next cell, that part
counts too); a line passes through a point when an image of one of its verified pieces does, a wall when one of its
surfaces does. No mesh is used.

    python closure.py [walls]
"""
from __future__ import annotations

import json
import sys

import mpmath as mp
import numpy as np

from core import FOLD, extended, in_half
from exact import SURF, value

OUT = "closure.json"
E_ = extended()
GG = np.concatenate([E_, E_ @ FOLD])


def dist(s, x):
    """Distance from x to the exact surface s (its coefficients exact, evaluated at full precision)."""
    return float(value(s, [mp.mpf(float(v)) for v in x]))


def images(P, tol=1e-9):
    Y = np.einsum("gij,nj->gni", GG, np.atleast_2d(np.asarray(P, float)))
    sm = Y.sum(-1, keepdims=True)
    ok = (np.abs(sm) > 1e-9).all((1, 2))
    Y = Y[ok] / sm[ok]
    return [y for y in Y if all(in_half(p, tol) for p in y)]


def main(only=None):
    W = json.load(open("working_data.json"))["walls"]
    ex = json.load(open("exact_defs.json"))
    pts = [np.array(p["float"]) for p in json.load(open("exact_points.json"))["points"] if p["exact"]]
    # (every image of every exact point in the half-cell, once)
    V = []
    for p in pts:
        for y in images(p, 1e-12):
            if not any(np.linalg.norm(y[0] - v) < 1e-9 for v in V):
                V.append(y[0])
    V = np.array(V)
    # (each line's verified pieces carried into the half-cell, as segments)
    segs = {}
    for L, v in ex["lines"].items():
        S = []
        for P in v["points"]:
            P = np.asarray(P, float)
            if len(P) >= 2:
                for Y in images(P, 1e-6):
                    S += list(zip(Y[:-1], Y[1:]))
        if S:
            segs[L] = (np.array([a for a, _ in S]), np.array([b for _, b in S]))

    def on_line(L, x, tol=1e-5):
        A, B = segs[L]
        d = B - A
        s = np.clip(((x - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
        return float(np.linalg.norm(A + s[:, None] * d - x, axis=1).min()) < tol
    bases = lambda w: {n.split(":")[0] for n in ex["walls"].get(w, {}).get("surface", [])}
    surfs = lambda w: [s for n, s in SURF.items() if n.split(":")[0] in bases(w)]
    out = {}
    for w, wd in W.items():
        if only and w not in only:
            continue
        Sw = surfs(w)
        edges = wd["edges"]
        Ys = {Y: surfs(Y) for Y in edges if Y not in ex["lines"]}
        corners = {e: [] for e in edges}
        for x in V:
            if not any(dist(s, x) < 1e-12 for s in Sw):
                continue
            thru = [e for e in edges if (e in segs and on_line(e, x)) or (e in Ys and any(dist(s, x) < 1e-12 for s in Ys[e]))]
            if len(thru) >= 2:
                for e in thru:
                    corners[e].append(x)
        short = {e: len(c) for e, c in corners.items() if len(c) < 2}
        out[w] = {"closed": not short, "corners_per_edge": {e: len(c) for e, c in corners.items()},
                  "edges_meeting_the_others_at_fewer_than_two_points": short}
        print(f"{w}: {'closed' if not short else 'OPEN'}; " + ", ".join(f"{e}:{len(c)}" for e, c in corners.items())
              + (f"; short: {short}" if short else ""), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{sum(v['closed'] for v in out.values())} of {len(out)} walls closed")


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
