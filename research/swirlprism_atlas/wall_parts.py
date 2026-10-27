"""Each wall's pieces, named from the working data alone (wall_parts.json).

pieces.py builds each patch of a wall divided into its parts: grown out to the wall's edges and stopped at its cuts
(each cut the cutting wall's own edge on this wall), each triangle a cut passes through divided at the exact points where
it crosses, so each part fills exactly to its lines and cuts. Here the parts are named, as a puzzle (no point is
classified):
  * a part bounded by lines (a line running along a real share of its rim) is the working data's piece of the wall with
    those lines;
  * across a cut by Y between two real parts, the piece changes the way the working data's cut says (one of the regions
    it lists swapped for another);
  * across a seam (where two of a flat wall's planes meet), the piece stays the same;
  * a scrap (under 1% of its patch, left at a corner where a cut ends) decides nothing and joins its neighbour; where
    the parts cannot all fit, the smallest part in the contradiction is set aside the same way.
Parts of one patch that come out the same piece across a cut are joined, and that stretch of the cut dropped: the cutting
wall's surface goes on past the piece there, but it divides nothing. What is left of each cut is the wall's invisible line.

Each piece: its patch and surface, its triangles, its boundary (segments labelled with their edge, or "cut by Y") and
the two regions it lies between. Regions use these pieces directly.

    python wall_parts.py [X14,X41,...]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

OUT = "wall_parts.json"


def wall_pieces(w, rec, work_w, lines):
    """The wall's pieces, named by the working data: [{patch, surface, tris, boundary, between}], and notes."""
    known = [tuple(sorted(p["between"])) for p in work_w.get("pieces", [])]
    plines = {tuple(sorted(p["between"])): set(p["lines"]) for p in work_w.get("pieces", [])}
    if not known and len(work_w.get("regions", [])) == 2:
        known = [tuple(sorted(work_w["regions"]))]
        plines[known[0]] = set()
    if not known and len(work_w.get("regions", [])) == 1:   # (the same region on both sides: X38 inside X68)
        known = [(work_w["regions"][0], work_w["regions"][0])]
        plines[known[0]] = set()
    cut_sides = {c["by"]: tuple(c["between"]) for c in work_w.get("cuts", []) if len(c.get("between", [])) >= 2}
    parts, links = [], []
    for k, pt in enumerate(rec["patches"]):
        ps = pt.get("parts") or ([{"tris": pt["tris"], "boundary": pt["boundary"], "links": []}] if pt["tris"] else [])
        base = len(parts)
        for p in ps:
            parts.append({"patch": k, "surface": pt["surface"], "tris": p["tris"], "segs": p["boundary"]})
        for i, p in enumerate(ps):
            for j, y in p["links"]:
                if i < j:
                    links.append((base + i, base + j, y))
    # (seams: parts of different patches with seam segments lying together)
    seam = [(i, (np.asarray(x, float) + np.asarray(y, float)) / 2) for i, p in enumerate(parts)
            for (x, nx), (y, ny) in p["segs"] if nx.startswith("seam")]
    if seam:
        tree = cKDTree(np.array([m for _, m in seam]))
        for a, b in tree.query_pairs(0.004):
            i, j = seam[a][0], seam[b][0]
            if parts[i]["patch"] != parts[j]["patch"]:
                links.append((min(i, j), max(i, j), "seam"))
    links = sorted(set(links))
    size = Counter()
    for p in parts:
        size[p["patch"]] += len(p["tris"])
    scrap = [len(p["tris"]) < 0.01 * size[p["patch"]] for p in parts]
    start, extra_lines = [], []
    for p in parts:
        # (a line runs along the part: three of its segments or more, and a real share of the part's lines)
        cnt = Counter(n for (_, n), (_, m) in p["segs"] if n == m and n in lines)
        top = max(cnt.values(), default=0)
        Lp = {n for n, c in cnt.items() if c >= max(3, 0.1 * top)}
        if len(known) == 1 or not Lp:
            start.append(set(known))                       # (one piece only, or nothing to tell them apart)
            continue
        sub = {x for x in known if Lp <= plines[x]}
        if not sub:
            # (lines the working data does not list on any piece: the pieces sharing the most of the part's lines,
            # and the rest recorded as contacts the working data misses)
            best = max(len(Lp & plines[x]) for x in known)
            sub = {x for x in known if best and len(Lp & plines[x]) == best}
            extra_lines.append((len(start), Lp))
        start.append(sub)

    def swaps(x, y):
        s = cut_sides.get(y, ())
        return {tuple(sorted((set(x) - {a}) | {b})) for a in s if a in x for b in s if b not in x}

    def fits(x, z, how, i, j):
        if scrap[i] or scrap[j]:
            return True
        return z == x if how == "seam" else z in swaps(x, how)
    while True:
        dom = [set(d) for d in start]
        changed = True
        while changed:
            changed = False
            for i, j, how in links:
                for a, b in ((i, j), (j, i)):
                    keep = {x for x in dom[a] if any(fits(x, z, how, a, b) for z in dom[b])}
                    if keep != dom[a]:
                        dom[a], changed = keep, True
        empty = [i for i, d in enumerate(dom) if not d and not scrap[i] and start[i]]
        if not empty:
            break
        scrap[min(empty, key=lambda i: len(parts[i]["tris"]))] = True
    name = [next(iter(d)) if len(d) == 1 and not scrap[i] else next(iter(start[i])) if len(start[i]) == 1 else None
            for i, d in enumerate(dom)]
    root = list(range(len(parts)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i
    for i, j, how in links:
        if how != "seam" and name[i] and name[i] == name[j] and parts[i]["patch"] == parts[j]["patch"]:
            root[find(i)] = find(j)
    for i in range(len(parts)):                           # (a scrap joins its largest neighbour in its patch)
        if scrap[i]:
            nb = [j for a, b, how in links for j in (a, b) if i in (a, b) and j != i and how != "seam"
                  and parts[j]["patch"] == parts[i]["patch"] and not scrap[j]]
            if nb:
                root[find(i)] = find(max(nb, key=lambda j: len(parts[j]["tris"])))
    groups = {}
    for i in range(len(parts)):
        groups.setdefault(find(i), []).append(i)
    out, notes = [], []
    key = lambda x: tuple(np.round(np.asarray(x, float), 7))
    for mem in groups.values():
        mem = sorted(mem, key=lambda i: -len(parts[i]["tris"]))
        nm = next((name[i] for i in mem if name[i]), None)
        segs = [s for i in mem for s in parts[i]["segs"]]
        # (a stretch of cut with this piece on both sides divides nothing: dropped)
        cutk = Counter(tuple(sorted((key(x), key(y)))) for (x, nx), (y, ny) in segs if nx.startswith("cut by"))
        segs = [s for s in segs if not (s[0][1].startswith("cut by")
                                        and cutk[tuple(sorted((key(s[0][0]), key(s[1][0]))))] > 1)]
        tris = [t for i in mem for t in parts[i]["tris"]]
        if nm is None:
            d = set().union(*(dom[i] for i in mem))
            notes.append(f"a piece of {len(tris)} triangles on {parts[mem[0]]['surface']} not settled (could be "
                         f"{', '.join('|'.join(x) for x in sorted(d)) or 'none of the working data pieces'})")
        out.append({"patch": parts[mem[0]]["patch"], "surface": parts[mem[0]]["surface"], "tris": tris,
                    "boundary": segs, "between": list(nm) if nm else None})
        for i, Lp in extra_lines:
            if i in mem and nm:
                miss = sorted(Lp - plines.get(nm, set()))
                if miss:
                    notes.append(f"the piece {'|'.join(nm)} is bounded by {', '.join(miss)} too, which the working data "
                                 f"does not list for it")
    return out, notes


def main(only=None, src="pieces.json"):
    P = json.load(open(src))
    work = json.load(open("working_data.json"))["walls"]
    lines = set(json.load(open("exact_defs.json"))["lines"])
    out = json.load(open(OUT)) if only and os.path.exists(OUT) else {}
    for w, rec in P.items():
        if only and w not in only:
            continue
        pcs, notes = wall_pieces(w, rec, work.get(w, {}), lines)
        out[w] = {"pieces": pcs, "notes": notes}
        got = Counter("|".join(p["between"]) if p["between"] else "?" for p in pcs)
        print(f"{w}: {len(pcs)} pieces {dict(got)}" + (f"; {len(notes)} not settled" if notes else ""), flush=True)
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None,
         os.environ.get("PIECES_IN", "pieces.json"))
