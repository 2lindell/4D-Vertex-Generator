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
    for w, r in curves.items():
        for Y, runs in r.items():
            for run in runs:
                solid += [(w, Y, p) for p in in_cell(run) if on_placed(surf.get(w, []), p) and on_placed(surf.get(Y, []), p)]
    for w, wd in walls.items():
        for c in wd.get("cuts", []):
            for run in curves.get(c["by"], {}).get(w, []):
                dashed += [(w, c["by"], p) for p in in_cell(run)
                           if on_placed(surf.get(w, []), p) and on_placed(surf.get(c["by"], []), p)]
    # (each wall's own edge lines: every copy of the line's exact pieces in the half-cell lying on one of its placed
    # surfaces)
    from domain_page import line_in_cell
    XL = work["exact"]["lines"]
    copies = {L: [np.asarray(p, float) for p in line_in_cell(v["points"]) if len(p) >= 2] for L, v in XL.items()}
    open_ = {}
    for w, wd in walls.items():
        corners = set(wd["points"])
        mine = []
        for L in wd["edges"]:
            cand = [st for c_ in copies.get(L, []) for st in split_at_points(c_, corners)
                    if not st[1] == st[2] == "face" and {st[1], st[2]} <= corners | {"face"}]
            got = [(L, P, a, b) for P, a, b in cand if on_placed(surf.get(w, []), P)]
            if not got and L in XL:
                # (none on its surfaces as listed: on any copy of them, corner to corner; the list lacks that copy)
                got = [(L, P, a, b) for P, a, b in cand if on_copy(surf.get(w, []), P)]
                if got:
                    print(f"  {w}: its edge {L} lies on a copy of its surface the list lacks", flush=True)
            mine += got
        solid += [(w, L, P) for L, P, _, _ in mine]
        # (closed: every corner met by two of its edges, counting its edges along other walls)
        deg = {}
        for L, P, a, b in mine:
            for nm, x in ((a, P[0]), (b, P[-1])):
                if nm != "face":
                    k = (nm, tuple(np.round(x, 6)))
                    deg[k] = deg.get(k, 0) + 1
        for w2, Y, P in [t for t in solid if t[0] == w and t[1] not in XL]:
            for x in (P[0], P[-1]):
                d = [k for k in deg if np.linalg.norm(np.array(k[1]) - x) < 1e-6]
                for k in d:
                    deg[k] += 1
        onface = lambda x: min(x[0], x[1], x[3], x[2] - x[3]) < 1e-9
        bad = sorted({f"{k[0]} at {np.round(k[1], 3).tolist()}" for k, n in deg.items() if n < 2 and not onface(k[1])})
        if bad:
            open_[w] = bad
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
