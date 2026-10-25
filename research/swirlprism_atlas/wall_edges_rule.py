"""The rule for a wall's edges, read off the walls checked by hand (T2, X10, X17, X26, X38, X39, F1):

  A wall's edges are the lines and walls of its boundary list (boundaries.json; boundaries_extra.json where it has
  none) that it runs along for a stretch:
    * a line named on its traced edges (edge_lines.json: the line on an edge, or a line or wall beyond it), seen at
      two points or more (one point is a corner brushing past);
    * a line on a side of one of its main patches (fexact_patches.json; not its copies: a copy's edges are images
      of its main's, and not a stale patch that is another wall now, as T2's old region B is X96);
    * a face line of the half-cell where it ends, or where its copy or own mirror image lies across (wall_outline.py's
      goes_on_across): those are edges too;
  and not:
    * a region (a region beyond an edge says what is there, not what the edge is);
    * another wall on the same surface (X39, X52, X35 on X10's quadric are its neighbours there, not its edges);
    * a wall met where the face it lies in is crossed ("face:F4" past X39's edge: X39 passes through F4 there).
  A wall with fewer than two such edges takes the lines of its boundary list (X17's X2 is only there).

    python wall_edges_rule.py        (prints the rule's edges against wall_outline_checked.json and wall_outline.json)
"""
from __future__ import annotations

import json
import os

import numpy as np

import region_walls as rw
import xsurface


def same_surface(a, b, known):
    if a not in known or b not in known:
        return False
    try:
        return xsurface.surface(a)["equation"] == xsurface.surface(b)["equation"]
    except Exception:
        return False


def main_patches(fp, w, d):
    """The wall's main patches: those whose plane is the plane of a main (not copy) piece on the page."""
    mains = []
    for x in d["xwalls"]:
        if x["id"] == w and not x.get("copy"):
            V = np.array(x["verts"], float) * np.array([1, 1, -1.0])
            B = (np.c_[V, np.ones(len(V))] @ np.asarray(d["cellInv"], float).T)
            mains.append(np.linalg.svd(B)[2][-1])
    if w == "T2":
        for p in d["tpatches"]:
            if not p.get("copy"):
                Q = np.array(p["pts"], float) * np.array([1, 1, -1.0])
                mains.append(np.linalg.svd(np.c_[Q, np.ones(len(Q))] @ np.asarray(d["cellInv"], float).T)[2][-1])
    out = []
    for p in fp:
        if p["target"] != w:
            continue
        n = np.asarray(p["normal"], float)
        n = n / np.linalg.norm(n)
        if w == "T2" and len(p["corners"]) == 3 and abs(n @ np.array([0, 1, -1, 0]) / 1) > 0.99:
            continue                                  # (T2's former region B: X96)
        if any(abs(abs(n @ (m / np.linalg.norm(m))) - 1) < 1e-6 for m in mains):
            out.append(p)
    return out


def edges(w, ctx):
    dims, bnd, EL, fp, outline, d, known = ctx
    LW = lambda x: dims.get(x) in ("line", "wall") and x != w
    base = {x for x in bnd.get(w, []) if LW(x)}
    s = set()
    for past, on, *n in EL.get(w, []):
        count = n[0] if n else 1
        if str(past).startswith("face:") and dims.get(past[5:]) == "wall":
            continue                                  # (it passes through the wall lying in that face)
        for x in (on, str(past)[5:] if str(past).startswith("face:") else past):
            if LW(x) and count >= 2 and not (dims.get(x) == "wall" and same_surface(w, x, known)):
                s.add(x)
    for p in main_patches(fp, w, d):
        for e in p.get("edge_labels") or []:
            if LW(e[2]):
                s.add(e[2])
    for x in outline.get(w, {}).get("goes_on_across", []):
        if LW(x) and dims.get(x) == "line":
            s.add(x)
    g = base & s if base else s
    g = {x for x in g if not (dims.get(x) == "wall" and same_surface(w, x, known))}
    if len(g) < 2:
        g |= {x for x in base if dims.get(x) == "line"}
    return sorted(g)


def context():
    d = rw.page_data()
    dims = {t["id"]: t["dim"] for t in d["types"]}
    bnd = json.load(open("boundaries.json"))
    for w, v in json.load(open("boundaries_extra.json")).items():
        if not w.startswith("_"):
            bnd.setdefault(w, v)
    EL = json.load(open("edge_lines.json"))
    fp = [p for p in (lambda x: x if isinstance(x, list) else x.values())(json.load(open("fexact_patches.json")))
          if isinstance(p, dict)]
    outline = json.load(open("wall_outline.json")) if os.path.exists("wall_outline.json") else {}
    known = set(json.load(open("xsurfaces.json"))) | set(json.load(open("xsurface_equations.json")))
    return dims, bnd, EL, fp, outline, d, known


if __name__ == "__main__":
    ctx = context()
    dims = ctx[0]
    checked = {k: v for k, v in json.load(open("wall_outline_checked.json")).items() if not k.startswith("_")}
    checked["F1"] = {"edges": ["B1", "D3", "E2", "T2", "X13"]}
    cur = ctx[4]
    ok = 0
    for w in sorted(checked):
        g = edges(w, ctx)
        m = set(g) == set(checked[w]["edges"])
        ok += m
        print(f"{w}: rule {g}  checked {checked[w]['edges']}  {'OK' if m else 'DIFF'}")
    print(f"{ok}/{len(checked)} checked walls match\nother walls where the rule differs from the current list:")
    for w in sorted((t for t, dm in dims.items() if dm == "wall" and t not in checked),
                    key=lambda x: (x[0], int("".join(c for c in x[1:] if c.isdigit()) or 0))):
        g = edges(w, ctx)
        if set(g) != set(cur.get(w, {}).get("edges", [])):
            print(f"   {w}: rule {g}  current {cur.get(w, {}).get('edges')}")
