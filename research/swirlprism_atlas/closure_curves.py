"""Does every wall's outline close with the exact curves? (closure_curves.json)

A wall's outline in the half-cell is its edges: the pieces of its lines (exact_lines.json) and its edges along other
walls (exact_curves.json), every image of each in the half-cell, cut at the faces, kept where they lie on the wall's
own surface (some image of it). The outline closes when every end of every piece meets another piece (within 1e-6), or
lies on a face of the half-cell, where the wall goes on into the next cell. Ends that do neither are listed.

    python closure_curves.py [walls]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from domain_page import HALF, line_in_cell
from exact_curves import LIB, dist_images, unit
from exact_page import in_cell

OUT = "closure_curves.json"
TOL = 1e-6


def seg_dist(p, P):
    """Distance from p to the polyline P."""
    a, b = P[:-1], P[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())


def main(only=None):
    W = json.load(open("working_data.json"))["walls"]
    ex = json.load(open("exact_defs.json"))
    lines = json.load(open("exact_lines.json"))["lines"]
    curves = json.load(open("exact_curves.json"))
    out = {}
    for w, wd in W.items():
        if only and w not in only:
            continue
        surf = [LIB[n] for n in ex["walls"].get(w, {}).get("surface", []) if n in LIB]
        on_w = lambda P: any(dist_images(s, unit(P[len(P) // 2][None])[0]).min() < 1e-6 for s in surf)
        pieces = []
        for e in wd["edges"]:
            if e in lines:
                pieces += [(e, np.asarray(P, float)) for P in line_in_cell(lines[e]["points"])]
            else:
                pieces += [(e, P) for run in curves.get(w, {}).get(e, []) for P in in_cell(run)]
        pieces = [(e, P) for e, P in pieces if len(P) >= 2 and on_w(P)]
        open_ = []
        for i, (e, P) in enumerate(pieces):
            for p in (P[0], P[-1]):
                if (np.abs(HALF @ p) < TOL).any():
                    continue
                if any(seg_dist(p, Q) < TOL for j, (_, Q) in enumerate(pieces) if j != i):
                    continue
                open_.append({"edge": e, "at": np.round(p, 6).tolist()})
        out[w] = {"pieces": len(pieces), "open": open_}
        print(f"{w}: {len(pieces)} pieces, {len(open_)} open ends" +
              ("".join(f"\n   {o['edge']} at {o['at']}" for o in open_[:6])), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    bad = [w for w, r in out.items() if r["open"]]
    print(f"{len(out) - len(bad)}/{len(out)} walls close; open: {bad}")


if __name__ == "__main__":
    main(set(sys.argv[1:]) or None)
