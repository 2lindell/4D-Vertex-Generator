"""The invisible lines on each wall: where the regions beside it change (invisible_cuts.json).

From the original data alone:
  * a wall's regions are the regions it separates (the page key's separates column); two regions are one face,
    so a wall beside n regions needs n - 2 cuts (one where a region on one side changes; a wall passing through it
    changes both sides at once and counts twice);
  * a cut is made by a wall touching it: another wall that ends on it (it is among that wall's edges,
    wall_outline.json) or meets it in the boundary lists (boundaries.json), and separates two of its regions;
  * the cut runs along the curve where the two walls meet, which is never a named line (a line both walls list is
    where their boundaries meet, not where one cuts the other).

    python invisible_cuts.py
"""
from __future__ import annotations

import json

import region_walls as rw


def main():
    d = rw.page_data()
    T = {t["id"]: t for t in d["types"]}
    dims = {k: t["dim"] for k, t in T.items()}
    sep = {k: set(t.get("separates") or []) for k, t in T.items() if t["dim"] == "wall"}
    # (F4, X20 and X41 as the database has them now (wall_sides.json): the page key's lists for them are older and
    # name regions that do not touch them (F4: X43, X60; X20: X59; X41: X49))
    sides = json.load(open("wall_sides.json"))
    for w in ("F4", "X20", "X41"):
        sep[w] = set(sides[w])
    bnd = {k: set(v) for k, v in json.load(open("boundaries.json")).items()}
    for k, v in json.load(open("boundaries_extra.json")).items():
        if not k.startswith("_"):
            bnd.setdefault(k, set(v))
    edges = {k: set(v["edges"]) for k, v in json.load(open("wall_outline.json")).items()}
    out = {}
    for w in sorted(sep, key=lambda x: (x[0], int(x[1:]))):
        R = sep[w]
        need = max(len(R) - 2, 0)
        cuts = []
        for y in sorted(sep):
            if y == w:
                continue
            ends_on = w in edges.get(y, set())
            meets = y in bnd.get(w, set()) or w in bnd.get(y, set())
            if not (ends_on or meets) or y in edges.get(w, set()) and not ends_on:
                continue                              # (w ends on y: y is w's edge, not a line across it)
            between = sorted(sep[y] & R)
            if len(between) < 2:
                continue
            cuts.append({"by": y, "between": between, "ends_on": ends_on, "along": f"{w} meets {y}"})
        out[w] = {"regions": sorted(R), "cuts_needed": need, "cuts": cuts}
    json.dump(out, open("invisible_cuts.json", "w"), indent=1)
    for w, r in out.items():
        flag = "" if len(r["cuts"]) == r["cuts_needed"] else f"   <-- {len(r['cuts'])} found, {r['cuts_needed']} needed"
        print(f"{w:4s} {'|'.join(r['regions']):24s} " + "; ".join(
            f"{c['by']}{'' if c['ends_on'] else '?'} ({'|'.join(c['between'])})" for c in r["cuts"]) + flag)


if __name__ == "__main__":
    main()
