"""A curved wall whose edges are curves already traced exactly (xcurves.json) and straight lines: its outline is
built from them directly, in order, and the inside filled as edge_snap.fill_mesh does (the old mesh only guides
which sheet of the surface the inner points go on). For a wall whose mesh broke up where it runs out thin.

    python fill_from_curves.py X52 X29:X39 X18:X35 line:X8      (curve:label past it, ...; a line closes the loop)
"""
from __future__ import annotations

import json
import sys

import numpy as np

import edge_snap as E
import xsurface


def main(tid, *parts):
    X = json.load(open("xsurfaces.json"))
    C = json.load(open("xcurves.json"))
    sf = X[tid]
    s0 = xsurface.surface(tid)
    W = E.Poly(s0["mons"], s0["coeffs"], tid)
    loop, corners = [], []
    for part in parts:
        name, lab = part.split(":", 1)
        if name == "line":
            continue
        P = [np.asarray(p, float) / np.sum(p) for p in C[name]["pts"]]
        if loop and np.linalg.norm(P[-1] - loop[-1][0]) < np.linalg.norm(P[0] - loop[-1][0]):
            P = P[::-1]
        corners += [P[0], P[-1]]
        loop += [(p, lab) for p in (P if not loop else P[1:])]
    # the closing straight line
    lab = [p.split(":", 1)[1] for p in parts if p.startswith("line:")][0]
    a, b = loop[-1][0], loop[0][0]
    n = max(2, int(np.linalg.norm(b - a) / 0.004))
    loop += [(a + (b - a) * t, lab) for t in np.linspace(0, 1, n + 1)[1:-1]]
    E.CORNERS.clear()
    E.CORNERS.extend(corners)
    outline = [[{"beta": p.tolist(), "beyond": l} for p, l in loop] + [{"beta": loop[0][0].tolist(), "beyond": loop[0][1]}]]
    tris = E.fill_mesh(W, outline, np.array(sf["tris"], float))
    X[tid] = {**sf, "tris": [np.asarray(t).tolist() for t in tris], "outline": outline, "snapped": True, "from_curves": list(parts)}
    json.dump(X, open("xsurfaces.json", "w"))
    print(f"{tid}: {len(tris)} triangles inside {', '.join(parts)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
