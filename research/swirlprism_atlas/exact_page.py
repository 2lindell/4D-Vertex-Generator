"""The Symmetry Domain page's wall edges along other walls and its dashed cuts, from the exact curves (exact_curves.json).

For each wall, its edges along other walls (solid) and the cuts other walls make on it (dashed: the cutting wall's own
edge on it) are the exact curves of exact_curves.py, found on the whole 3-sphere between the walls' exact corners;
every image of each in the half-cell is drawn, in each view. In the half-cell view they replace the outlines that came
from the meshes' rims, except a wall's outline along its own lines (points on the exact lines), which is kept; in the
dodecahedron views the dashed cuts are replaced.

    python exact_page.py ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

import chamber_domain
import dodeca_view as dv
from domain_page import GG, LINE_TOL, carry, chains, chart, clip_polyline, images, line_in_cell, rounded, thin_line, view

TAGS = {"cell": "atlas-data", "v1": "atlas-v1", "chambers": "atlas-chambers"}


def in_cell(run):
    """Every image of a run (unit vectors of betas) in the half-cell, cut at its faces (betas summing to 1)."""
    run = np.asarray(run, float)
    out = []
    for gi in images(run):
        Y = carry(run.copy(), GG[gi])
        ok = np.isfinite(Y).all(1)
        # (split where the betas' sum passes 0, far outside the half-cell)
        cur = []
        for y, k in zip(Y, ok):
            if k:
                cur.append(y)
            elif cur:
                out += [np.array(r) for r in clip_polyline(np.array(cur)) if len(r) >= 2]
                cur = []
        if len(cur) >= 2:
            out += [np.array(r) for r in clip_polyline(np.array(cur)) if len(r) >= 2]
    keep = []
    for r in out:
        m = r[len(r) // 2]
        if any(np.linalg.norm(q - m, axis=1).min() < 1e-7 for q in keep):
            continue                                       # (the same piece from another element)
        keep.append(r)
    return keep


def main(page):
    curves = json.load(open("exact_curves.json"))
    work = json.load(open("working_data.json"))["walls"]
    ex = json.load(open("exact_defs.json"))["lines"]
    html = open(page).read()
    data = {}
    for k, tag in TAGS.items():
        m = re.search(rf'<script id="{tag}" type="application/json">(.*?)</script>', html, re.S)
        data[k] = (m, json.loads(m.group(1)))
    solid, dashed = [], []
    for w, r in curves.items():
        for e, runs in r.items():
            for run in runs:
                solid += [(w, e, p) for p in in_cell(run)]
    for w, wd in work.items():
        for c in wd.get("cuts", []):
            for run in curves.get(c["by"], {}).get(w, []):
                dashed += [(w, c["by"], p) for p in in_cell(run)]
    # (several exact curves can join the same two corners, A1 and C1 being so symmetric: of each, only the pieces lying
    # on both walls as the page draws them are kept; the curves' places stay exact)
    from scipy.spatial import cKDTree
    cell0 = data["cell"][1]
    trees = {}
    for w_ in cell0["xwalls"]:
        trees.setdefault(w_["id"], []).append(np.asarray(w_["verts"], float))
    trees = {k: cKDTree(np.vstack(v)) for k, v in trees.items()}
    # (region files hold the walls' pieces too, but each wall's own mesh is enough here)

    def on_both(a, b, p, tol=0.012):
        q = view(p)
        m = q[[len(q) // 4, len(q) // 2, 3 * len(q) // 4]]
        return all(k in trees and (trees[k].query(m)[0] < tol).all() for k in (a, b))
    n0 = len(solid), len(dashed)
    # (an edge whose corners were given is drawn as found: its corners are confirmed)
    given = json.load(open("working_base.json"))["corrections"].get("edge_corners", {})
    sure = {(w, Y) for w, d_ in given.items() for Y in d_}
    solid = [(w, Y, p) for w, Y, p in solid if (w, Y) in sure or on_both(w, Y, p)]
    dashed = [(w, Y, p) for w, Y, p in dashed if (Y, w) in sure or on_both(w, Y, p)]
    print(f"kept on both walls: solid {n0[0]} -> {len(solid)}, dashed {n0[1]} -> {len(dashed)}", flush=True)
    print(f"{len(solid)} solid and {len(dashed)} dashed exact pieces in the half-cell", flush=True)
    hover = lambda w, Y, copy: (f"{w} · edge of the wall, along {Y} (exact)" if not copy else
                                f"{w} · invisible line: where {Y} meets it, between its pieces (exact)")

    # (a mesh outline kept only where it runs along one of the wall's own lines)
    cell = data["cell"][1]
    line_pts = {}
    for w, wd in work.items():
        P = []
        for L in wd["edges"]:
            if L in ex:
                P += [view(q) for q in line_in_cell(ex[L]["points"])]
        line_pts[w] = np.vstack(P) if P else np.zeros((0, 3))

    # (which walls have every edge along another wall exact, and which cuts are exact: there the mesh's curves go)
    drawn = {(w, Y) for w, Y, _ in solid}
    exact_edges = {w for w, wd in work.items()
                   if any(Y not in ex for Y in wd["edges"]) and all((w, Y) in drawn for Y in wd["edges"] if Y not in ex)}
    exact_cuts = {(w, Y) for w, Y, _ in dashed}

    def keep_mesh(s):
        if not s.get("outline"):
            return True
        if s.get("copy"):                                  # (a dashed cut: kept only where no exact one was found)
            m_ = re.search(r"where (\S+) meets it", s.get("hover", ""))
            return not (m_ and (s["id"], m_.group(1)) in exact_cuts)
        if s["id"] not in exact_edges:
            return True
        L = line_pts.get(s["id"], np.zeros((0, 3)))
        m = np.asarray(s["pts"][len(s["pts"]) // 2], float)
        return len(L) > 0 and np.linalg.norm(L - m, axis=1).min() < 2e-3
    cell["xlines"] = [s for s in cell["xlines"] if keep_mesh(s)] + \
        [{"id": w, "outline": True, "copy": False, "pts": thin_line(view(p), LINE_TOL).tolist(), "hover": hover(w, Y, False)}
         for w, Y, p in solid] + \
        [{"id": w, "outline": True, "copy": True, "pts": thin_line(view(p), LINE_TOL).tolist(), "hover": hover(w, Y, True)}
         for w, Y, p in dashed]
    # (the dodecahedron views: the same pieces in the cell chart, carried by dodeca_view as the page builder does)
    empty = {"samples": {}, "uniform": [], "special": [], "rings": [], "main": [], "segments": [], "qaxes": [],
             "regular": [], "tpatches": [], "mirrors": [], "fdomain": [], "fcentre": [0, 0, 0], "xwalls": []}
    src = {**empty, "xlines": [{"id": w, "outline": True, "copy": False, "pts": thin_line(chart(p), LINE_TOL / 3).tolist(),
                                "hover": hover(w, Y, False)} for w, Y, p in solid]
           + [{"id": w, "outline": True, "copy": True, "pts": thin_line(chart(p), LINE_TOL / 3).tolist(),
               "hover": hover(w, Y, True)} for w, Y, p in dashed]}
    dv.use_centre((1, 0, 0, 0))
    v1 = dv.view(src)
    ch = chamber_domain.view(v1)
    dv.use_centre((0, 0, 1, 1))
    length = lambda P: float(np.linalg.norm(np.diff(np.asarray(P, float), axis=0), axis=1).sum())
    for k, new in (("v1", v1), ("chambers", ch)):
        dv.tidy_lines(new)
        vw = data[k][1]
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
        # (in these views the dashed cuts are all replaced; the solid outlines from the meshes are kept beside the exact)
        vw["xlines"] = [s_ for s_ in vw["xlines"] if not (s_.get("outline") and s_.get("copy"))] + joined
    for k, tag in TAGS.items():
        m, d = data[k]
        body = json.dumps(rounded(d, 4 if k == "cell" else 3), separators=(",", ":"), ensure_ascii=False)
        html = html.replace(m.group(0), f'<script id="{tag}" type="application/json">{body}</script>')
    open(page, "w").write(html)
    print(f"{page}: {len(html) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
