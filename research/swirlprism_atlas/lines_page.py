"""The Symmetry Domain with its lines only (symmetry_domain_lines.html): no meshes, only edges and cuts, all exact.

From the built page: every wall's mesh is left out, and every outline or dashed cut that came from a mesh's rim. Kept and
added, all from exact data:
  * the named lines, from their exact points (as the page draws them), and each wall's own edge lines as its outline:
    every copy of the line lying on one of the wall's surfaces as placed in the half-cell;
  * each wall's edge along another wall (solid) and each cut another wall makes on it (dashed: the cutting wall's own
    edge on it), the exact curves of exact_curves.py, every image of each in the half-cell, kept where it lies exactly
    on one of the wall's own surfaces as placed in the half-cell and on one of the other wall's (strictly: no mesh is
    asked).
The region shells (meshes too) are not published beside it.

    python lines_page.py ../../assets/symmetry_domain.html symmetry_domain_lines.html
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

import chamber_domain
import dodeca_view as dv
from arrangement import LIB, off
from domain_page import LINE_TOL, chains, chart, rounded, thin_line, view
from exact_page import TAGS, in_cell


_EP = []


def split_at_points(P, only=None):
    """A line's piece (betas) split at the exact points it passes through (only those named in `only`, a wall's
    corners, when given: the wall's edge runs straight through any other): [(stretch, name at start, name at end)], an
    end on a face of the half-cell named "face"."""
    if not _EP:
        from arrangement import exact_points
        _EP.extend(exact_points())
    EP, EN = _EP
    a, b = P[:-1], P[1:]
    hits = []
    for x, n in zip(EP, EN):
        d = b - a
        t = np.clip(((x - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-300), 0, 1)
        dist = np.linalg.norm(a + t[:, None] * d - x, axis=1)
        i = int(dist.argmin())
        if dist[i] < 1e-7 and (only is None or n in only or i + t[i] < 1e-9 or i + t[i] > len(P) - 1 - 1e-9):
            hits.append((i + t[i], n, x))
    hits.sort(key=lambda h: h[0])
    face = lambda x: "face" if min(x[0], x[1], x[3], x[2] - x[3]) < 1e-9 else None
    out, cur, start = [], [P[0]], None
    names = {}
    for h in hits:
        names.setdefault(round(h[0], 9), (h[1], h[2]))
    cuts = sorted(names.items())
    pos, prev_name = 0.0, None
    if cuts and cuts[0][0] < 1e-9:
        prev_name = cuts[0][1][0]
        cuts = cuts[1:]
    prev_name = prev_name or face(P[0]) or "?"
    i0 = 0
    for c, (n, x) in cuts:
        i = int(np.floor(c))
        if c >= len(P) - 1 - 1e-9:
            break
        seg = np.vstack([cur, P[i0 + 1:i + 1], [x]]) if i >= i0 + 1 else np.vstack([cur, [x]])
        if len(seg) >= 2 and np.linalg.norm(seg[-1] - seg[0]) > 1e-9:
            out.append((seg, prev_name, n))
        cur, i0, prev_name = [x], i, n
    last = cuts[-1] if cuts and cuts[-1][0] >= len(P) - 1 - 1e-9 else None
    seg = np.vstack([cur, P[i0 + 1:]])
    end_name = last[1][0] if last else (face(P[-1]) or "?")
    if len(seg) >= 2 and np.linalg.norm(seg[-1] - seg[0]) > 1e-9:
        out.append((seg, prev_name, end_name))
    return out


def on_copy(names, P):
    from arrangement import GINV, fvals, grads
    k = sorted({len(P) // 4, len(P) // 2, (3 * len(P)) // 4})   # (inside the stretch: at a corner a surface can be singular)
    for n in names:
        if n in LIB:
            ok = np.ones(len(GINV), bool)
            for x in P[k]:
                X = np.einsum("gij,j->gi", GINV, x)
                ok &= np.abs(fvals(LIB[n], X)) / (np.linalg.norm(grads(LIB[n], X), axis=1) + 1e-300) < 1e-7
            if ok.any():
                return True
    return False


def on_placed(names, P):
    k = sorted({len(P) // 4, len(P) // 2, (3 * len(P)) // 4})   # (inside the stretch: at a corner a surface can be singular)
    return any(n in LIB and off(LIB[n], P[k]).max() < 1e-7 for n in names)


def main(src, dst):
    curves = json.load(open("exact_curves.json"))
    work = json.load(open("working_data.json"))
    walls, surf = work["walls"], work["exact"]["wall_surfaces"]
    html = open(src).read()
    data = {}
    for k, tag in TAGS.items():
        m = re.search(rf'<script id="{tag}" type="application/json">(.*?)</script>', html, re.S)
        data[k] = (m, json.loads(m.group(1)))
    solid, dashed = [], []
    # (each wall's edges: wall_faces.py, the stretches of its edge lines and its edges along other walls that have one
    # of its faces on one side only)
    WF = json.load(open("wall_faces.json"))
    for w, r in WF.items():
        solid += [(w, e["along"], np.asarray(e["points"], float)) for e in r["edges"]]
    open_ = {w: r["report"]["corners_met_once"] for w, r in WF.items() if r["report"]["corners_met_once"]}
    missing = {w: r["report"]["edges_not_drawn"] for w, r in WF.items() if r["report"]["edges_not_drawn"]}
    print("listed edges not drawn:", missing, flush=True)
    for w, wd in walls.items():
        for c in wd.get("cuts", []):
            for run in curves.get(c["by"], {}).get(w, []):
                dashed += [(w, c["by"], p) for p in in_cell(run)
                           if on_placed(surf.get(w, []), p) and on_placed(surf.get(c["by"], []), p)]
    print("walls whose drawn outline leaves a corner met by one edge only:", open_, flush=True)
    print(f"{len(solid)} solid and {len(dashed)} dashed exact pieces", flush=True)
    hover = lambda w, Y, copy: (f"{w} · edge of the wall, along {Y} (exact)" if not copy else
                                f"{w} · invisible line: where {Y} meets it, between its pieces (exact)")
    cell = data["cell"][1]
    cell["xwalls"] = []
    cell["xlines"] = [{"id": w, "outline": True, "copy": False, "pts": thin_line(view(p), LINE_TOL).tolist(),
                       "hover": hover(w, Y, False)} for w, Y, p in solid] + \
                     [{"id": w, "outline": True, "copy": True, "pts": thin_line(view(p), LINE_TOL).tolist(),
                       "hover": hover(w, Y, True)} for w, Y, p in dashed]
    empty = {"samples": {}, "uniform": [], "special": [], "rings": [], "main": [], "segments": [], "qaxes": [],
             "regular": [], "tpatches": [], "mirrors": [], "fdomain": [], "fcentre": [0, 0, 0], "xwalls": []}
    srcd = {**empty, "xlines": [{"id": w, "outline": True, "copy": c_, "pts": thin_line(chart(p), LINE_TOL / 3).tolist(),
                                 "hover": hover(w, Y, c_)} for c_, L_ in ((False, solid), (True, dashed)) for w, Y, p in L_]}
    dv.use_centre((1, 0, 0, 0))
    v1 = dv.view(srcd)
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
        vw["xwalls"] = []
        vw["xlines"] = joined
    for k, tag in TAGS.items():
        m, d = data[k]
        body = json.dumps(rounded(d, 4 if k == "cell" else 3), separators=(",", ":"), ensure_ascii=False)
        html = html.replace(m.group(0), f'<script id="{tag}" type="application/json">{body}</script>')
    html = html.replace("<title>", "<title>Lines · ", 1)
    open(dst, "w").write(html)
    print(f"{dst}: {len(html) / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
