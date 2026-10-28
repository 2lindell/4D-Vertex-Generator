"""The shape map (shape_map.html): every shape as a node, linked to the shapes on its boundary and the shapes it bounds,
from the working data alone (working_data.json).

Links, each tagged with where it comes from:
  - "edges": a wall's edges and the points at their ends;
  - "separates": a wall's two sides;
  - "ends on it" / "crosses": a wall's cuts (the walls that end on it, or cross it);
  - "region walls": a region's walls (a wall bounding it only in part, as its pieces show, is marked as partial);
  - "region boundary": the lines on a region's boundary and their end points;
  - "ends on": a line's ends, solved exactly from its equations (line_ends.py): the points or lines it ends on.
A point has no boundary: its links are the lines, walls and regions that touch it. Each node carries its example seed and
its evenness; the nodes are laid out by the two main directions of their centres in the cell (each shape's own geometry:
its pieces, its exact line pieces or its exact point).

    python build_shape_map.py        (writes shape_map.html from shape_map_src.html)
"""
from __future__ import annotations

import collections
import json

import numpy as np

from core import CHART, home, seed_text


def main():
    work = json.load(open("working_data.json"))
    shapes = work["shapes"]
    WP = json.load(open("wall_parts.json"))
    ex = json.load(open("exact_defs.json"))["lines"]
    crp = json.load(open("cross_ring_points.json"))["points"]
    # (each shape's own geometry, as betas: its pieces, its exact line pieces, its exact points)
    geo = collections.defaultdict(list)
    for w, v in WP.items():
        for p in v["pieces"]:
            T_ = np.array(p["tris"], float).reshape(-1, 4)
            geo[w].append(T_)
            for r in p.get("between") or ():
                geo[r].append(T_)
    for L_, v in ex.items():
        geo[L_] += [np.asarray(P, float) for P in v["points"] if len(P)]
    for c in crp:
        geo[c["name"]].append(np.asarray([c["beta"]], float))
    nodes = []
    for tid, t in shapes.items():
        B_ = np.concatenate(geo[tid]) if geo.get(tid) else (np.asarray([t["seed"]], float) if t.get("seed") else None)
        if B_ is None:
            continue
        c = (B_ / B_.sum(1, keepdims=True)).mean(0) @ CHART
        n = {"id": tid, "dim": t["dim"], "name": t["name"], "c": np.round(c, 4).tolist(), "bounds": [], "sep": []}
        if t.get("seed"):
            n["seed"] = seed_text(home(t["seed"]))
        if t.get("evenness"):
            n["even"] = t["evenness"]
        nodes.append(n)
    ids = {n["id"] for n in nodes}
    N_DIM = {n["id"]: n["dim"] for n in nodes}
    E = {}

    def add(a, b, src, info=None, part=None):
        if a == b or a not in ids or b not in ids:
            return
        k = tuple(sorted((a, b)))
        e = E.setdefault(k, {"a": k[0], "b": k[1], "src": [], "faces": {}, "part": []})
        if src not in e["src"]:
            e["src"].append(src)
        if info:
            e["faces"].update(info)
        if part and part not in e["part"]:
            e["part"].append(part)
    # (every link from the working data (working_data.json, by working_data.py), so a shape lists only what it
    # touches: a wall its edges, the points at their ends, the regions on its two sides and the walls that end on it
    # (its invisible cuts); a region its walls; a line or point its own boundary list (the points or lines at its
    # ends). A region's full boundary list and the old cut's measured faces name shapes past its edges, so they are
    # not used)
    W = work["walls"]
    for n in nodes:
        i_ = n["id"]
        if n["dim"] == "wall" and i_ in W:
            sp = W[i_].get("sphere")
            if sp:
                n["note"] = (f"on the 3-sphere: one whole piece of {sp['patch_images']} half-cell patch images, "
                             f"carried onto itself by {sp['stabiliser']} symmetr{'y' if sp['stabiliser'] == 1 else 'ies'}"
                             + (f"; meets copies of itself along {', '.join(sp['meets_copies_along'])}"
                                if sp.get("meets_copies_along") else ""))
            n["bounds"] = W[i_]["edges"] + W[i_]["points"]
            n["sep"] = W[i_]["regions"]
            for b in n["bounds"]:
                add(i_, b, "edges")
            for r in W[i_]["regions"]:
                add(i_, r, "separates")
            for c in W[i_]["cuts"]:
                add(i_, c["by"], "crosses" if c.get("crossing") else "ends on it")
        elif n["dim"] == "line" and i_ in work.get("lines", {}):
            # (a line's ends, solved exactly from its equations: the points or lines it ends on)
            ends = [e for e in work["lines"][i_]["ends"] if e.get("status") in ("point", "line")]
            n["bounds"] = sorted({e["on"] for e in ends})
            n["ends_unresolved"] = [e.get("on", e.get("why")) for e in work["lines"][i_]["ends"]
                                    if e.get("status") not in ("point", "line")]
            for b in n["bounds"]:
                add(i_, b, "ends on")
            if n["ends_unresolved"]:
                n["note"] = "an end not yet named: " + "; ".join(str(x) for x in n["ends_unresolved"])
        elif n["dim"] == "point":
            n["bounds"] = []                              # (a point has no boundary: what touches it links to it)
        elif n["dim"] == "line":
            for b in n["bounds"]:
                if N_DIM.get(b) in ("line", "point"):
                    add(i_, b, "column")
        elif n["dim"] == "region":
            R = work["regions"].get(i_, {})
            ext = R.get("wall_extent", {})
            lines = R.get("lines", {})
            if isinstance(lines, list):
                lines = {L_: "full" for L_ in lines}
            n["bounds"] = R.get("walls", []) + list(lines) + R.get("points", [])
            for w in R.get("walls", []):
                add(i_, w, "region walls", part=i_ if ext.get(w) == "partial" else None)
            for x, how in lines.items():
                add(i_, x, "region boundary", part=i_ if how != "full" else None)
            for x in R.get("points", []):
                add(i_, x, "region boundary")
    edges = list(E.values())
    C = np.array([n["c"] for n in nodes])
    C0 = C - C.mean(0)
    U = np.linalg.svd(C0, full_matrices=False)[2][:2]
    for n, p in zip(nodes, C0 @ U.T):
        n["xy"] = np.round(p, 4).tolist()
    data = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, separators=(",", ":"))
    html = open("shape_map_src.html").read().replace("__DATA__", data.replace("</", "<\\/"))
    open("shape_map.html", "w").write(html)
    print(len(nodes), "nodes,", len(edges), "links:", dict(collections.Counter(s for e in edges for s in e["src"])))


if __name__ == "__main__":
    main()
