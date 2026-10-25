"""The shape map (shape_map.html): every shape of the atlas as a node, linked to the shapes on its boundary and the
shapes it bounds, from the current database. Run after cell_atlas2.py (it reads the page's data) and region_view.py
--defs (region_db.json).

Links, each tagged with where it comes from:
  - "column": a shape's boundary list in the page's key (its points, lines and walls);
  - "separates": a wall's two sides;
  - "region walls": a region's walls in region_defs.json (a wall bounding it only in part is marked as partial);
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
    nodes = []
    for tid, t in T.items():
        P = geo.get(tid) or pts.get(tid)
        if not P:
            continue
        c = np.mean(np.array(P, float), axis=0)
        n = {"id": tid, "dim": t["dim"], "name": t["name"], "c": np.round(c, 4).tolist(),
             "bounds": t.get("bounds") or [], "sep": t.get("separates") or []}
        if tid in seeds and np.isfinite(seeds[tid].get("ratio", np.inf)):
            from cell_atlas import seed_text, to_upper
            n["seed"] = seed_text(to_upper(np.array(seeds[tid]["beta"], float)))
            n["even"] = round(float(seeds[tid]["ratio"]), 4)
        if tid in wall_edges:
            n["edges"] = wall_edges[tid]
        nodes.append(n)
    ids = {n["id"] for n in nodes}
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
    for n in nodes:
        for b in n["bounds"]:
            add(n["id"], b, "column")
        for r in n["sep"]:
            add(n["id"], r, "separates")
    db = json.load(open("region_db.json"))
    defs = json.load(open("region_defs.json"))
    partial = db.get("partial", {})
    for r, v in defs.items():
        if not isinstance(v, dict):
            continue                                  # (the file's own notes)
        for w in v.get("walls", []):
            add(r, w, "region walls", part=r if w in partial.get(r, []) else None)
    # (each wall's measured faces: its sections' areas by the regions on their two sides, after region_build)
    import glob
    import gzip
    for fn in sorted(glob.glob("sections/*.json.gz")):
        w = os.path.basename(fn).split(".")[0]
        area = collections.Counter()
        for f in json.load(gzip.open(fn, "rt")):
            x, y = f["sides"]
            if x == y or x.startswith("(") or y.startswith("(") or "unknown" in (x, y):
                continue
            area[tuple(sorted((x, y)))] += f["area"]
        for (x, y), a in area.items():
            for r, o in ((x, y), (y, x)):
                add(w, r, "faces", {f"{r}|{o}": round(a, 6)})
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
