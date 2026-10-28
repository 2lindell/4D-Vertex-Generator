"""The Symmetry Domain page's wall outlines and dashed cuts drawn from the exact equations, not from the meshes.

Each wall's edges along named lines are drawn from the lines' own exact pieces already. Its other curves are drawn here
from the equations:
  * an edge along another wall Y (solid): where the wall's surface meets Y's surface (wall_equations.json), traced
    exactly (Newton onto both surfaces at every step) from one of its corners to the next;
  * a cut by Y (dashed): Y's own edge on the wall, traced the same way on Y's surface and the wall's;
each carried by every symmetry into the half-cell, cut at its faces, and then into the dodecahedron views. The page's
outlines that came from the meshes' rims are replaced by these; the meshes themselves are kept as they are.

    python exact_outlines.py ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

import chamber_domain
import core as osd
import dodeca_view as dv
from domain_page import LINE_TOL, chains, chart, line_in_cell, rounded, thin_line, view
from pieces import trace

TAGS = {"cell": "atlas-data", "v1": "atlas-v1", "chambers": "atlas-chambers"}


def curves(owner, other, eq, LIB):
    """Where the wall owner meets the wall other, along owner's edge on it: exact runs (betas)."""
    out = []
    r = eq.get(owner, {}).get("edges", {}).get(other, {})
    own = [LIB[n] for n in eq.get(owner, {}).get("surface", []) if n in LIB]
    for rr in ([r] if r.get("surface") else []) + r.get("more", []):
        T = LIB.get(rr["surface"])
        C = [np.asarray(c, float) for c in rr.get("corners", [])]
        if T is None or not C:
            continue
        for S in own:
            if S is T:
                continue
            on = [c for c in C if abs(float(np.ravel(S.f(c))[0])) < 1e-6]
            for c in on:
                for run in trace(S, T, c, on):
                    # (the same stretch traced from another corner, or the other way, kept once)
                    if any(near_run(run[len(run) // 2], q) for q in out):
                        continue
                    out.append(run)
    return out


def near_run(p, Q, tol=1e-6):
    """The point p lies on the polyline Q."""
    A, B = Q[:-1], Q[1:]
    d = B - A
    t = np.clip(((p - A) * d).sum(1) / np.maximum((d * d).sum(1), 1e-30), 0, 1)
    return float(np.linalg.norm(A + t[:, None] * d - p, axis=1).min()) < tol


def in_cell(runs):
    """Every image of the runs in the half-cell, each stretch once."""
    out = []
    for r in line_in_cell(runs):
        r = np.asarray(r, float)
        if len(r) < 2 or any(near_run(r[len(r) // 2], q) for q in out):
            continue
        out.append(r)
    return out


def main(page):
    eq = json.load(open("wall_equations.json"))["walls"]
    work = json.load(open("working_data.json"))
    ex = json.load(open("exact_defs.json"))["lines"]
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    html = open(page).read()
    data = {}
    for k, tag in TAGS.items():
        m = re.search(rf'<script id="{tag}" type="application/json">(.*?)</script>', html, re.S)
        data[k] = (m, json.loads(m.group(1)))
    solid, dashed = [], []                                 # (wall, other, run in the half-cell)
    for w, wd in work["walls"].items():
        for Y in wd["edges"]:
            if Y in ex:
                continue                                   # (along a line: drawn from the line itself)
            solid += [(w, Y, r) for r in in_cell(curves(w, Y, eq, LIB))]
        for c in wd.get("cuts", []):
            Y = c["by"]
            dashed += [(w, Y, r) for r in in_cell(curves(Y, w, eq, LIB))]
    print(f"{len(solid)} solid and {len(dashed)} dashed exact curves in the half-cell", flush=True)
    item = lambda w, Y, r, copy, P: {"id": w, "outline": True, "copy": copy, "pts": P,
                                      "hover": f"{w} · edge of the wall" if not copy else
                                      f"{w} · invisible line: where {Y} meets it, between its pieces"}
    # (the half-cell view)
    cell = data["cell"][1]
    cell["xlines"] = [s for s in cell["xlines"] if not s.get("outline")] + \
        [item(w, Y, r, False, thin_line(view(r), LINE_TOL).tolist()) for w, Y, r in solid] + \
        [item(w, Y, r, True, thin_line(view(r), LINE_TOL).tolist()) for w, Y, r in dashed]
    # (the dodecahedron views: the same curves in the cell chart, carried by dodeca_view as the page builder does)
    empty = {"samples": {}, "uniform": [], "special": [], "rings": [], "main": [], "segments": [], "qaxes": [],
             "regular": [], "tpatches": [], "mirrors": [], "fdomain": [], "fcentre": [0, 0, 0], "xwalls": []}
    src = {**empty, "xlines": [item(w, Y, r, False, thin_line(chart(r), LINE_TOL / 3).tolist()) for w, Y, r in solid]
           + [item(w, Y, r, True, thin_line(chart(r), LINE_TOL / 3).tolist()) for w, Y, r in dashed]}
    dv.use_centre((1, 0, 0, 0))
    v1 = dv.view(src)
    ch = chamber_domain.view(v1)
    dv.use_centre((0, 0, 1, 1))
    length = lambda P: float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())
    for k, new in (("v1", v1), ("chambers", ch)):
        dv.tidy_lines(new)
        vw = data[k][1]
        # (the pieces of one curve joined end to end, one item each)
        groups = {}
        for s_ in new["xlines"]:
            groups.setdefault((s_["id"], s_["copy"], s_["hover"]), []).append(np.asarray(s_["pts"], float))
        joined = []
        for (i_, c_, h_), P_ in groups.items():
            segs = [(a, b) for P in P_ for a, b in zip(P[:-1], P[1:])]
            for line in chains(segs, 4):
                if length(line) >= 0.015:
                    joined.append({"id": i_, "outline": True, "copy": c_, "hover": h_,
                                   "pts": thin_line(np.asarray(line), 2 * LINE_TOL).tolist()})
        vw["xlines"] = [s_ for s_ in vw["xlines"] if not s_.get("outline")] + joined
    # (written back in place, the rest of the page untouched)
    for k, tag in TAGS.items():
        m, d = data[k]
        body = json.dumps(rounded(d, 4 if k == "cell" else 3), separators=(",", ":"))
        html = html.replace(m.group(0), f'<script id="{tag}" type="application/json">{body}</script>')
    open(page, "w").write(html)
    print(f"{page}: {len(html) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
