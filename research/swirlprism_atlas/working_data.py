"""The working data in one place (working_data.json): every wall with its edges, points, the regions beside it and
the invisible cuts across it, and every region with its walls. Built from the corrected wall edges
(wall_outline.json, by wall_edges_rule.py), the regions each wall separates (the page key; wall_sides.json where the
key is out of date) and each region's walls (region_defs.json), with the fills listed in FILLS, each from the original
boundary lists (boundaries.json). Then, treating the data as a closed set:
  * a cut on a wall: another wall touching it (ending on it, or in either one's boundary list) that separates two of
    its regions; it runs along the curve where the two meet (never a named line);
  * a region is closed where every edge of each of its walls is met inside the region: by another of its walls that
    has the same edge, passes through that line (goes on across it) or lists it, or is the wall that edge names.
What is still open is printed, and kept in the file under "open".

    python working_data.py
"""
from __future__ import annotations

import json

import region_walls as rw

# (wall, region, add or drop, why): each a fill of the original data, or a stale entry taken out
FILLS = [
    ("F4", "X43", "drop", "X43's boundary list has no F4 (and F4 has X49, X58, X61 in wall_sides.json)"),
    ("F4", "X60", "drop", "X60's boundary list has no F4"),
    ("X41", "X49", "drop", "X49's boundary list has no X41 (wall_sides.json: X58, X62)"),
    ("X20", "X59", "drop", "X59's boundary list has no X20 (wall_sides.json: X66, X95)"),
    ("X79", "X72", "add", "tested: both sides of X79 are X72 and X73 on all its patches (they swap where X26 crosses it)"),
    ("X96", "X58", "add", "X58's boundary list has T2: T2's former region B, the wall X96 (X41 ends on it there)"),
    ("X17", "X46", "add", "X46's boundary list has X17 (X19 ends on X17 there)"),
]


# (lines a wall runs straight through, on into the next cell: no edge there, but the wall is at that line)
THROUGH = {"X27": ["X2"], "X53": ["D1"]}
# (walls crossing each other: each passes through the other along a curve, where the two regions beside both swap
# sides; an invisible line on each, though no region is added)
CROSSINGS = [("X26", "X79", "tested: X26 lies in b3 = b2 + b4 and runs through X79 along b2 = b4, b3 = 2 b2; "
              "X72 and X73 swap sides on X79 there")]


def key(x):
    return (x[0], int("".join(c for c in x[1:] if c.isdigit()) or 0), x)


def main():
    d = rw.page_data()
    T = {t["id"]: t for t in d["types"]}
    dims = {k: t["dim"] for k, t in T.items()}
    B = {k: set(v) for k, v in json.load(open("boundaries.json")).items()}
    for k, v in json.load(open("boundaries_extra.json")).items():
        if not k.startswith("_"):
            B.setdefault(k, set(v))
    outline = json.load(open("wall_outline.json"))
    sides = json.load(open("wall_sides.json"))
    walls = sorted((k for k, v in dims.items() if v == "wall"), key=key)
    regions = sorted((k for k, v in dims.items() if v == "region"), key=key)
    sep = {w: set(T[w].get("separates") or []) for w in walls}
    for w in ("F4", "X20", "X41"):                    # (the page key's lists for these are out of date)
        sep[w] = set(sides[w])
    deduced = []
    rw_ = {r: set(v["walls"]) for r, v in json.load(open("region_defs.json")).items() if isinstance(v, dict)}
    for w, r, act, _ in FILLS:
        if act == "add":
            sep[w].add(r)
            rw_.setdefault(r, set()).add(w)
        else:
            sep[w].discard(r)
            rw_.get(r, set()).discard(w)
    # (a wall with two regions and no cut that ends on another wall: both its regions run along that wall)
    E0 = {w: set(outline.get(w, {}).get("edges", [])) for w in walls}
    for y in walls:
        if len(sep[y]) == 2:
            for w in E0[y] & set(walls):
                for r in sep[y] - sep[w]:
                    sep[w].add(r)
                    deduced.append({"wall": w, "region": r, "action": "add",
                                    "why": f"{y} ends on {w} with only {' and '.join(sorted(sep[y]))} beside it"})
    for w in walls:                                   # (a region beside a wall lists it, and the other way round)
        for r in sep[w]:
            rw_.setdefault(r, set()).add(w)
    for r, ws in rw_.items():
        for w in ws:
            sep.setdefault(w, set()).add(r)
    E = {w: set(outline.get(w, {}).get("edges", [])) for w in walls}
    across = {w: set(outline.get(w, {}).get("goes_on_across", [])) | set(THROUGH.get(w, [])) for w in walls}
    # (the cuts)
    cuts = {}
    for w in walls:
        cuts[w] = []
        for y in walls:
            if y == w:
                continue
            ends_on = w in E[y]
            if not (ends_on or y in B.get(w, set()) or w in B.get(y, set())) or (y in E[w] and not ends_on):
                continue
            between = sorted(sep[y] & sep[w], key=key)
            if len(between) >= 2:
                cuts[w].append({"by": y, "between": between, "along": f"{w} meets {y}"})
        for a, b_, why in CROSSINGS:
            if w in (a, b_):
                y = b_ if w == a else a
                cuts[w] = [c for c in cuts[w] if c["by"] != y]
                cuts[w].append({"by": y, "between": sorted(sep[w] & sep[y], key=key), "along": f"{w} crosses {y}",
                                "crossing": True})
    # (which regions reach each line: beyond a wall's edge on it (edge_lines.json), or beside a wall with two
    # regions and no cut that ends on it; a region of a wall with more regions may lie along its other edges only)
    EL = json.load(open("edge_lines.json"))
    reach = {}
    for w, v in EL.items():
        for past, on, *n in v:
            p = str(past)[5:] if str(past).startswith("face:") else past
            if dims.get(p) == "region" and dims.get(on) == "line":
                reach.setdefault(on, set()).add(p)
    for w in walls:
        if len(sep[w]) == 2:
            for e in E[w] | across[w]:
                if dims.get(e) == "line":
                    reach.setdefault(e, set()).update(sep[w])
    # (closure of each region)
    open_ = {}
    for r in regions:
        W = rw_.get(r, set())
        miss = []
        for w in W:
            for e in E[w]:
                if dims.get(e) == "wall":
                    if e not in W:
                        miss.append(f"{w} ends on {e}, not a wall of {r}")
                elif r in reach.get(e, ()) and not any(e in E[v] or e in across[v] or e in B.get(v, set())
                                                       for v in W if v != w):
                    miss.append(f"{w}'s edge {e} meets no other wall of {r}")
        if miss:
            open_[r] = sorted(miss)
    out = {
        "_about": __doc__.strip().split("\n\n")[0],
        "fills": [{"wall": w, "region": r, "action": a, "why": why} for w, r, a, why in FILLS] + deduced,
        "through": THROUGH,
        "crossings": [{"walls": [a, b_], "why": why} for a, b_, why in CROSSINGS],
        "walls": {w: {"edges": sorted(E[w], key=key), "through": THROUGH.get(w, []), "points": outline.get(w, {}).get("points", []),
                      "regions": sorted(sep[w], key=key), "cuts_needed": max(len(sep[w]) - 2, 0), "cuts": cuts[w]}
                  for w in walls},
        "regions": {r: {"walls": sorted(rw_.get(r, set()), key=key)} for r in regions},
        "open": open_,
    }
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2])
    print("walls short of cuts:", {w: (len(cuts[w]), max(len(sep[w]) - 2, 0)) for w in walls
                                   if len([c for c in cuts[w] if not c.get("crossing")]) < len(sep[w]) - 2})
    print(f"closed regions: {len(regions) - len(open_)}/{len(regions)}")
    for r, m in open_.items():
        print(f"  {r}: " + "; ".join(m))


if __name__ == "__main__":
    main()
