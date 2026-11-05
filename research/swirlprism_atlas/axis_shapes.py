"""The shapes the extra-symmetry lines pass through (axis_shapes.json).

The light purple axes (half-turn axes of the 2400-element group, normalizer.coset_axes: 4 segments in the half-cell)
and the green girdle (order-3 axis of the 3600-element group, supergroups.girdle_segments: 1 segment): a seed on one
gives a polytope with 2400 (3600) symmetries, so every shape a segment passes through has seeds of extra symmetry
there, and where the segment crosses a wall or a line, that crossing is a point of extra symmetry too. Each segment
(straight in the betas: the seed is linear in them) is walked in steps of 5e-4, each change of shape bisected to
1e-12 and the point there classified as on a wall (the hull tolerance relaxed), with the shape's key (split names).

    python axis_shapes.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np

import normalizer as nz
import supergroups as sg
from core import shape_key

OUT = "axis_shapes.json"
STEP = 5e-4


def lab(b, tol=1e-9):
    return shape_key(b, tol)[0]


def crossing(args):
    a, b, la = args
    lo, hi = a, b
    for _ in range(40):
        m = (lo + hi) / 2
        if lab(m) == la:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    return m, lab(m, 1e-7), lab(m, 1e-8)


def main():
    segs = [("purple", a, b) for a, b in nz.coset_axes()] + [("girdle", a, b) for a, b in sg.girdle_segments()]
    out = []
    with Pool() as pool:
        for kind, a, b in segs:
            a, b = a / a.sum(), b / b.sum()
            n = max(2, int(np.linalg.norm(b - a) / STEP))
            ts = np.linspace(0, 1, n + 1)
            P = [a + t * (b - a) for t in ts]
            L = pool.map(lab, P, chunksize=16)
            ends = [lab(a, 1e-7), lab(b, 1e-7)]
            runs, changes = [], []
            for i in range(len(P)):
                if not runs or runs[-1]["shape"] != L[i]:
                    runs.append({"shape": L[i], "from_t": float(ts[i])})
                    if i:
                        changes.append((P[i - 1], P[i], L[i - 1]))
                runs[-1]["to_t"] = float(ts[i])
            X = pool.map(crossing, changes)
            out.append({"kind": kind, "from": a.tolist(), "to": b.tolist(), "ends": ends,
                        "runs": runs,
                        "crossings": [{"at": m.tolist(), "on": on, "on_1e-8": on8, "between": [runs[k]["shape"], runs[k + 1]["shape"]]}
                                      for k, (m, on, on8) in enumerate(X)]})
            print(kind, np.round(a, 4).tolist(), "->", np.round(b, 4).tolist(), flush=True)
            print("   ends:", ends, flush=True)
            print("   shapes along it:", " | ".join(f"{r['shape']} ({r['to_t'] - r['from_t']:.3f})" for r in runs), flush=True)
            for c in out[-1]["crossings"]:
                print(f"   crossing {c['between'][0]} -> {c['between'][1]} on {c['on']} (tol 1e-8: {c['on_1e-8']}) at {np.round(c['at'], 6).tolist()}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "segments": out}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
