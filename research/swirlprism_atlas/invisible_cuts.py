"""The invisible lines on each wall: where the regions beside it change (invisible_cuts.json).

From the original data alone:
  * a wall's regions are the regions it separates (the page key's separates column); two regions are one face,
    so a wall beside n regions needs n - 2 cuts (one where a region on one side changes; a wall passing through it
    changes both sides at once and counts twice);
  * a cut is made by a wall touching it: another wall that ends on it (it is among that wall's edges,
    wall_outline.json) or meets it in the boundary lists (boundaries.json), and separates two of its regions;
  * the line of the cut is a line both walls list as a boundary (the curve they meet along), else it is unnamed.

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
            line = sorted(x for x in bnd.get(w, set()) & bnd.get(y, set()) if dims.get(x) == "line"
                          and x not in edges.get(w, set()))
            cuts.append({"by": y, "between": between, "ends_on": ends_on, "line": line or None})
        out[w] = {"regions": sorted(R), "cuts_needed": need, "cuts": cuts}
    json.dump(out, open("invisible_cuts.json", "w"), indent=1)
    for w, r in out.items():
        flag = "" if len(r["cuts"]) == r["cuts_needed"] else f"   <-- {len(r['cuts'])} found, {r['cuts_needed']} needed"
        print(f"{w:4s} {'|'.join(r['regions']):24s} " + "; ".join(
            f"{c['by']}{'' if c['ends_on'] else '?'} ({'|'.join(c['between'])}) on {c['line'] or 'unnamed'}" for c in r["cuts"]) + flag)


if __name__ == "__main__":
    main()
