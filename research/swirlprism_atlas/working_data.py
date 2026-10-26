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
    ("X79", "X72", "add", "tested: both sides of X79 are X72 and X73 on all its patches (they swap where X26 crosses it)"),
    ("X96", "X58", "add", "X58's boundary list has T2: T2's former region B, the wall X96 (X41 ends on it there)"),
    ("X17", "X46", "add", "X46's boundary list has X17 (X19 ends on X17 there)"),
]


# (lines a wall runs straight through, on into the next cell: no edge there, but the wall is at that line)
THROUGH = {"X27": ["X2"], "X53": ["D1"], "X94": ["X78"]}
# (lines where a wall meets its own image, closing a wedge region by itself: ring test, X32 between two sheets of X15
# and X30 between two sheets of X83 around X78)
SELF_MEET = {"X15": ["X78"], "X83": ["X78"]}
# (edges the traced data misses, found by the ring test: the wall is met once around the line, so it ends there)
EDGE_FILLS = {"F5": ["E1"]}
# (edges the rule keeps that are not edges: the wall meets its own image across a face of the cell there (ring test
# with the neighbouring cell brought back by the group: the wall twice round the line), so it goes on through it)
EDGE_DROPS = {"X14": ["C2b"]}
# (cuts deduced in the closed set: the only wall separating the two regions that change on the wall; not yet traced)
CUT_FILLS = [("X20", "F5", "ring test round X8: X20's far side changes from X66 to X59 along X8 (X52 gives way to X39 "
              "there); F5 is the only wall between X59 and X66")]
# (walls with the same region on both sides: tested, X38 on its own quadric (its key sample and seed lie on it) has X68
# on both sides at 217 of 225 points, down to 1e-7 off it: a wall inside X68, between X68 and its mirror image)
SAME_BOTH_SIDES = {"X38": "X68"}
# (walls found that are not in the atlas yet, under a provisional name: their regions, surface and what they cut)
NEW_WALLS = {
    "X100": {
        "regions": ["X58", "X61"], "surface": "X7 (its own quadric; also X51's image 1127)",
        "signature": "1200: cells 120+120+600+600+600+1200+1200+1200 | faces 240+600+1200+1200+1200+1200+1200+1200+1200"
                     "+1200+1200+1200 | edges 600+600+600+600+600+600+1200+1200+1200+1200 | val 3:1800,4:4200,6:1800,"
                     "11:600",
        "why": "tested: 25 points bisected between X58 and X61 off F4's face piece (b1 = 0.001 to 0.015) lie on X7's "
               "quadric to 1e-7, and all classify as this shape; it ends on F4, where X58 changes to X61",
        "edges": ["C2a", "E1", "F4"],
        "edges_why": "ring test: met once around C2a and around E1, between X61 and X58; it ends on F4"},
}
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
    for w, es in EDGE_FILLS.items():
        outline[w]["edges"] = sorted(set(outline[w]["edges"]) | set(es))
    for w, es in EDGE_DROPS.items():
        outline[w]["edges"] = sorted(set(outline[w]["edges"]) - set(es))
        THROUGH.setdefault(w, [])
        THROUGH[w] = sorted(set(THROUGH[w]) | set(es))
    for nw, v in NEW_WALLS.items():                   # (new walls join the walls, as the page will list them)
        outline[nw] = {"edges": v["edges"], "points": [], "goes_on_across": []}
        dims[nw] = "wall"
        T[nw] = {"id": nw, "dim": "wall", "separates": v["regions"]}
    sides = json.load(open("wall_sides.json"))
    walls = sorted((k for k, v in dims.items() if v == "wall"), key=key)
    regions = sorted((k for k, v in dims.items() if v == "region"), key=key)
    base = json.load(open("wall_regions_base.json"))      # (frozen: a page rebuild does not change them)
    sep = {w: set(base.get(w) or T[w].get("separates") or []) for w in walls}
    for w in ("F4", "X41"):                          # (the page key's lists for these are out of date)
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
        for w_, y, why in CUT_FILLS:
            if w == w_ and not any(c["by"] == y for c in cuts[w]):
                cuts[w].append({"by": y, "between": sorted(sep[w] & sep[y], key=key), "along": f"{w} meets {y}",
                                "deduced": why})
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
        for w in W & set(walls):
            for e in E[w]:
                if dims.get(e) == "wall":
                    if e not in W and len(sep[w]) == 2:      # (a wall with more regions: maybe another's edge)
                        miss.append(f"{w} ends on {e}, not a wall of {r}")
                elif r in reach.get(e, ()) and e not in SELF_MEET.get(w, []) and not any(
                        e in E[v] or e in across[v] or e in B.get(v, set()) for v in W & set(walls) if v != w):
                    miss.append(f"{w}'s edge {e} meets no other wall of {r}")
        if miss:
            open_[r] = sorted(miss)
    out = {
        "_about": __doc__.strip().split("\n\n")[0],
        "fills": [{"wall": w, "region": r, "action": a, "why": why} for w, r, a, why in FILLS] + deduced,
        "through": THROUGH,
        "self_meet": SELF_MEET,
        "edge_fills": EDGE_FILLS,
        "edge_drops": EDGE_DROPS,
        "crossings": [{"walls": [a, b_], "why": why} for a, b_, why in CROSSINGS],
        "same_both_sides": SAME_BOTH_SIDES,
        "new_walls": NEW_WALLS,
        "walls": {w: {"edges": sorted(E[w], key=key), "through": THROUGH.get(w, []),
                      **({"both_sides": SAME_BOTH_SIDES[w]} if w in SAME_BOTH_SIDES else {}), "points": outline.get(w, {}).get("points", []),
                      "regions": sorted(sep[w], key=key), "cuts_needed": max(len(sep[w]) - 2, 0), "cuts": cuts[w]}
                  for w in walls},
        "regions": {r: {"walls": sorted(rw_.get(r, set()), key=key)} for r in regions},
        "open": open_,
    }
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2 and w not in SAME_BOTH_SIDES])
    print("walls short of cuts:", {w: (len(cuts[w]), max(len(sep[w]) - 2, 0)) for w in walls
                                   if len([c for c in cuts[w] if not c.get("crossing")]) < len(sep[w]) - 2})
    print(f"closed regions: {len(regions) - len(open_)}/{len(regions)}")
    for r, m in open_.items():
        print(f"  {r}: " + "; ".join(m))


if __name__ == "__main__":
    main()
