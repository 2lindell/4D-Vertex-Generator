"""The shape map (shape_map.html): every shape of the atlas as a node, linked to the shapes on its boundary and the
shapes it bounds, from the current database. Run after cell_atlas2.py (it reads the page's data) and region_view.py
--defs (region_db.json).

Links (all from working_data.json), each tagged with where it comes from:
  - "column": a shape's boundary list in the page's key (its points, lines and walls);
  - "separates": a wall's two sides;
  - "region walls": a region's walls (a wall bounding it only in part, as its piece shows, is marked as partial);
  - "region boundary": the lines on a region's boundary (where two of its walls meet, an edge of a wall lying wholly on
    it, or where one of its walls meets its own copy) and their end points;
  - "ends on": a line's ends, solved exactly from its equations (line_ends.py): the points or lines it ends on;
  - "faces": the measured sections of each wall between two regions (sections/, as region_build left them), with
    their areas.
Each node also carries its example seed's evenness (optimal_seeds.json). The nodes are laid out by the two main
directions of their centres in the cell.

    python build_shape_map.py        (writes shape_map.html from shape_map_src.html)
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np

import region_walls as rw


def main():
    d = rw.page_data()
    T = {t["id"]: t for t in d["types"]}
    pts = collections.defaultdict(list)
    for k, v in d["samples"].items():
        for p in v:
            pts[k].append(p[:3])
    geo = collections.defaultdict(list)
    for w in d["xwalls"]:
        geo[w["id"]] += [v for v in w["verts"]]
    for p in d["tpatches"]:
        geo["T2"] += p["pts"][:-1]
    for src in ("xlines", "rings", "segments", "main"):
        for x in d.get(src, []):
            if isinstance(x, dict) and not x.get("outline") and x.get("pts"):
                geo[x["id"]] += x["pts"]
    for src in ("uniform", "special"):
        for x in d.get(src, []):
            if isinstance(x, dict) and "q" in x:
                geo[x["id"]].append(x["q"])
    seeds = json.load(open("optimal_seeds.json")) if os.path.exists("optimal_seeds.json") else {}
    # (each wall's edges, checked: wall_edges.py)
    wall_edges = collections.defaultdict(list)
    if os.path.exists("wall_edges.json"):
        for t_ in json.load(open("wall_edges.json")):
            if "stray" in t_:
                continue
            for e in t_["edges"]:
                wall_edges[t_["wall"]].append({
                    "piece": t_["piece"], "kind": e["kind"],
                    "on": e.get("surface") or e.get("on"), "exact": e.get("exact", True),
                    "off": e.get("max_off"), "beyond": e.get("beyond"), "across": e.get("across")})
    # (the boundary lists as the database has them now (boundaries.json), over the page key's older copy)
    B = json.load(open("boundaries.json")) if os.path.exists("boundaries.json") else {}
    nodes = []
    for tid, t in T.items():
        P = geo.get(tid) or pts.get(tid)
        if not P:
            continue
        c = np.mean(np.array(P, float), axis=0)
        n = {"id": tid, "dim": t["dim"], "name": t["name"], "c": np.round(c, 4).tolist(),
             "bounds": B.get(tid) or t.get("bounds") or [], "sep": t.get("separates") or []}
        if tid in seeds and np.isfinite(seeds[tid].get("ratio", np.inf)):
            from cell_atlas import seed_text, to_upper
            n["seed"] = seed_text(to_upper(np.array(seeds[tid]["beta"], float)))
            n["even"] = round(float(seeds[tid]["ratio"]), 4)

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
    work = json.load(open("working_data.json"))
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
        elif n["dim"] in ("line", "point"):
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
