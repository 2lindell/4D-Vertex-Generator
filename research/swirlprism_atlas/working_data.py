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

import numpy as np
import os

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
THROUGH = {"X53": ["D1"], "X94": ["X78"]}
# (lines where a wall meets its own image, closing a wedge region by itself: ring test, X32 between two sheets of X15
# and X30 between two sheets of X83 around X78)
SELF_MEET = {"X15": ["X78"], "X83": ["X78"], "X14": ["C2b"], "X25": ["C2b"], "X38": ["C2a"]}   # (X14, X25: a different copy of
# the wall across C2b, X38 across C2a (on the face b4=0): their edge, not a line they go through)
# (edges the traced data misses, found by the ring test: the wall is met once around the line, so it ends there)
EDGE_FILLS = {"F5": ["E1", "X20"], "X27": ["X2"], "X38": ["C2a"]}
# (named lines lying on a face of the half-cell along a wall's piece: a named shape there is always an edge, where the
# wall meets a different copy of itself across the face (sphere_complex.py found them; X53 goes through D1, verified))
FACE_EDGES = {"F1": ["D3"], "F4": ["D2"], "T2": ["D1"], "X3": ["C3", "D1"], "X9": ["C2b", "D2"], "X10": ["C3"],
              "X19": ["C2b", "D2"], "X20": ["D1"], "X23": ["D2"], "X26": ["D2", "D3"], "X40": ["D1", "D3"],
              "X42": ["D2", "D3"], "X87": ["C3", "D3"], "X88": ["D3"]}   # (F5 ends on X20, cutting it; X27 ends on X2: checked by hand)
# (edges the rule keeps that are not edges: the wall meets its own image across a face of the cell there (ring test
# with the neighbouring cell brought back by the group: the wall twice round the line), so it goes on through it)
EDGE_DROPS = {"F5": ["X8"]}   # (F5: the sides labelled X8 classify as F5)
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


def settle_ring(r1, r2, sides, same=None, touch=None):
    """The ring round a line stretch, settled from its two readings by the known contacts on the full 3-sphere: each
    wall crossed round the line must have the regions before and after it as its two sides, and must be known to touch
    the line (touch: the walls with the line as an edge, or passing through it, or meeting their own image along it).
    A crossing that breaks this hides what the ring stepped over (a wall it could not name, ERR, or thin regions): it is
    replaced by the shortest chain of regions and walls that fits, if there is just one (a tie is decided by what the
    other reading saw), keeping the wall that was seen. Settled when both readings come out the same; or when one has
    more (thin regions the other stepped over) and every extra wall touches the line. Else None.
    Returns (ring, notes on what was settled)."""
    same = same or {}
    touch = set(touch or ())
    adj = {}
    for w, rs_ in sides.items():
        for a in rs_:
            for b in rs_:
                if a != b:
                    adj.setdefault(a, []).append((w, b))
        if same.get(w):
            adj.setdefault(same[w], []).append((w, same[w]))

    def ok(w, a, b):
        if w not in sides:
            return False
        return (a == b and same.get(w) == a) or (a != b and {a, b} <= sides[w])

    def parse(ring):
        rs, ws = [], []
        for x in ring:
            if x.startswith("["):
                if len(ws) < len(rs):
                    ws.append(x.strip("[]"))
                elif ws:
                    ws[-1] = x.strip("[]")
            else:
                if rs and len(ws) < len(rs):
                    ws.append(None)                       # (two regions with nothing named between them)
                rs.append(x)
        if len(ws) < len(rs):
            ws.append(None)
        return rs, ws

    def chains(a, b, w, seen, depth=3, any_wall=False):
        """Shortest chains a [w1] x1 [w2] ... b, walls touching the line, the wall seen (w) among them if it is known."""
        known = w in sides
        out, frontier = [], [((), a)]
        for _ in range(depth):
            nxt = []
            for path, at in frontier:
                for v, x in adj.get(at, []):
                    if touch and v not in touch and v not in seen_ok and not any_wall:
                        continue
                    p = path + ((v, x),)
                    if x == b:
                        if not known or any(u == w for u, _ in p):
                            out.append(p)
                    elif x not in (a, b) and all(x != y for _, y in path):
                        nxt.append((p, x))
            if out:
                break
            frontier = nxt
        if len(out) > 1:                                  # (a tie: what the other reading saw decides)
            out = [p for p in out if all(v in seen and (x in seen or x == b) for v, x in p)] or out
        return out

    def fix(ring, seen):
        rs, ws = parse(ring)
        if any(r in ("ERR", "OUT") for r in rs):
            return None, []
        notes, out = [], []
        n = len(rs)
        for k in range(n):
            a, b, w = rs[k], rs[(k + 1) % n], ws[k]
            if a == b and (w is None or w == "ERR"):
                continue                                  # (the same region on both sides of nothing: one region)
            out.append(a)
            if ok(w, a, b):                               # (seen, with the right sides: evidence it touches the line)
                out.append(f"[{w}]")
                continue
            c = chains(a, b, w, seen)
            if not c:                                     # (no known contact fits: a chain that is the only one
                c = chains(a, b, w, seen, any_wall=True)  # possible at all, its walls then touch the line)
                if len(c) == 1:
                    inferred.update(v for v, _ in c[0] if v not in touch and v not in seen_ok)
            if len(c) != 1:
                return None, []
            for v, x in c[0]:
                out.append(f"[{v}]")
                if x != b:
                    out.append(x)
            notes.append(f"{a} [{w}] {b}: " + " ".join(f"[{v}] {x}" for v, x in c[0]))
        return out, notes

    def canon(ring):
        k = min(range(0, len(ring), 2), key=lambda i: tuple(ring[i:] + ring[:i])) if ring else 0
        return tuple(ring[k:] + ring[:k])

    def within(small, big):
        """Is the cyclic ring small the cyclic ring big with some stretches left out?"""
        S, B_ = [x for x in small if not x.startswith("[")], [x for x in big if not x.startswith("[")]
        for k in range(len(B_)):
            R_ = B_[k:] + B_[:k]
            j = 0
            for y in R_:
                if j < len(S) and y == S[j]:
                    j += 1
            if j == len(S):
                return True
        return False

    inferred = set()
    seen = {x.strip("[]") for x in r1 + r2}
    seen_ok = {r[k].strip("[]") for r in (r1, r2) for k in range(1, len(r), 2)
               if ok(r[k].strip("[]"), r[k - 1], r[(k + 1) % len(r)])}
    f1, n1 = fix(r1, seen)
    f2, n2 = fix(r2, seen)
    if f1 is None or f2 is None:
        return None, []
    notes = sorted(set(n1 + n2))
    if inferred:
        notes.append(f"walls that must touch the line though nothing lists it: {', '.join(sorted(inferred))}")
    if canon(f1) == canon(f2):
        return f1, notes + ([] if r1 == r2 else ["the two readings agree once settled"])
    big, small = (f1, f2) if len(f1) > len(f2) else (f2, f1)
    extra = {x.strip("[]") for x in big if x.startswith("[")} - {x.strip("[]") for x in small if x.startswith("[")}
    if within(small, big) and extra <= (touch | seen_ok):
        return big, notes + [f"one reading stepped over thin regions; their walls ({', '.join(sorted(extra))}) touch the line"]
    return None, []


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
    for w, es in list(EDGE_FILLS.items()) + list(FACE_EDGES.items()):
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
                elif r in reach.get(e, ()) and e not in SELF_MEET.get(w, []) + FACE_EDGES.get(w, []) and not any(
                        e in E[v] or e in across[v] or e in B.get(v, set()) for v in W & set(walls) if v != w):
                    miss.append(f"{w}'s edge {e} meets no other wall of {r}")
        if miss:
            open_[r] = sorted(miss)
    # (each region's walls, full or partial, its lines and points: from the regions beside the walls' pieces,
    # piece_regions.py, where they have been found)
    PR = json.load(open("piece_regions.json")) if os.path.exists("piece_regions.json") else {}
    to_check = {}

    LE = json.load(open("line_ends.json"))["lines"] if os.path.exists("line_ends.json") else {}
    line_pts = {L_: sorted({r["on"] for r in rs if r["type"] == "ends" and r["on"] not in LE
                            and not r["on"].startswith(("unnamed", "ERR", "new:"))
                            and r["on"] not in walls and r["on"] not in regions})
                for L_, rs in LE.items()}

    # (the closed set around each line: the ring test, the walls and regions in their cyclic order round the line at
    # points along it; a wall's piece between regions a and b has the line L where L's ring shows the wall between
    # a and b; a region has L where it appears in L's ring, all along it (full) or along part of it (partial))
    RT = json.load(open("ring_test.json"))["rings"] if os.path.exists("ring_test.json") else {}
    ring_lines, piece_lines, ring_unsure, ring_settled, contacts_found = {}, {}, {}, {}, {}
    sides_of = {w: set(sep[w]) for w in walls}
    # (the walls known to touch each line: it is one of their edges, they pass through it, or meet their own image along it)
    touch = {}
    for w in walls:
        for L_ in set(outline.get(w, {}).get("edges", [])) | set(THROUGH.get(w, [])) | set(SELF_MEET.get(w, [])):
            touch.setdefault(L_, set()).add(w)
    for L_, stretches in RT.items():
        if L_ not in line_pts:
            continue
        total, have = 0.0, {}
        for st in stretches:
            ring, how = settle_ring(st["ring"], st.get("ring2", st["ring"]), sides_of, SAME_BOTH_SIDES, touch.get(L_))
            if ring is None:
                ring_unsure.setdefault(L_, []).append([st.get("from"), st.get("to")])
                continue
            for k_ in range(1, len(ring), 2):           # (a wall seen round the line that its data does not list)
                w_ = ring[k_].strip("[]")
                if w_ not in touch.get(L_, set()):
                    n_ = sum(1 for y in ring if y == ring[k_])
                    contacts_found.setdefault(L_, {})[w_] = "ends on it (met once round it)" if n_ == 1 else \
                        f"passes through it or meets its own image along it (met {n_} times round it)"
            if how:
                ring_settled.setdefault(L_, []).append({"from": st.get("from"), "to": st.get("to"), "ring": " ".join(ring),
                                                        "how": how})
            ln = st.get("length", 1.0)
            total += ln
            for r in {x for x in ring if not x.startswith("[")}:
                have[r] = have.get(r, 0.0) + ln
            n = len(ring)
            for i, x in enumerate(ring):
                if x.startswith("["):
                    wall_ = x.strip("[]")
                    a_, b_ = ring[(i - 1) % n], ring[(i + 1) % n]
                    piece_lines.setdefault(wall_, {}).setdefault("|".join(sorted((a_, b_), key=key)), set()).add(L_)
        for r, ln in have.items():
            ring_lines.setdefault(r, {})[L_] = "full" if ln >= 0.98 * total else "partial"
    if ring_unsure:
        to_check["ring_stretches_unsure"] = ring_unsure
    if ring_settled:
        to_check["ring_stretches_settled"] = ring_settled
    if contacts_found:
        to_check["contacts_found_by_rings"] = contacts_found

    def piece_data(r):
        """A region's walls' extent (full or partial: from their pieces, piece_regions.py), and its lines and points
        from the walls' own boundaries: a line is on the region's boundary where two of its walls meet along it, where
        a wall lying wholly on it has it as an edge, or where one of its walls meets its own copy along it; the
        points are those lines' ends."""
        mine = rw_.get(r, set())
        extent = {}
        if r in PR.get("regions", {}):
            pr = PR["regions"][r]
            extent = {w: pr["walls"].get(w, "not found beside it in the pieces") for w in sorted(mine, key=key)}
            extra = sorted((w for w in pr["walls"] if w not in mine), key=key)
            if extra:
                to_check.setdefault("region_walls", {})[r] = extra
        lines_r = ring_lines.get(r, {})
        pts = sorted({p for L_ in lines_r for p in line_pts[L_]}, key=key)
        return {"wall_extent": extent, "lines": {L_: lines_r[L_] for L_ in sorted(lines_r, key=key)}, "points": pts}

    out = {
        "_about": __doc__.strip().split("\n\n")[0],
        "fills": [{"wall": w, "region": r, "action": a, "why": why} for w, r, a, why in FILLS] + deduced,
        "through": THROUGH,
        "self_meet": SELF_MEET,
        "edge_fills": EDGE_FILLS,
        "face_edges": FACE_EDGES,
        "edge_drops": EDGE_DROPS,
        "crossings": [{"walls": [a, b_], "why": why} for a, b_, why in CROSSINGS],
        "same_both_sides": SAME_BOTH_SIDES,
        "new_walls": NEW_WALLS,
        "walls": {w: {"edges": sorted(E[w], key=key),
                      "boundary_points": sorted({p for L_ in E[w] if L_ in line_pts for p in line_pts[L_]}, key=key),
                      "pieces": [{"between": pair.split("|"), "lines": sorted(ls, key=key),
                                  "points": sorted({p for L_ in ls for p in line_pts[L_]}, key=key)}
                                 for pair, ls in sorted(piece_lines.get(w, {}).items())], "through": THROUGH.get(w, []),
                      **({"both_sides": SAME_BOTH_SIDES[w]} if w in SAME_BOTH_SIDES else {}), "points": outline.get(w, {}).get("points", []),
                      "regions": sorted(sep[w], key=key), "cuts_needed": max(len(sep[w]) - 2, 0), "cuts": cuts[w]}
                  for w in walls},
        "regions": {r: {"walls": sorted(rw_.get(r, set()), key=key), **piece_data(r)} for r in regions},
        "pieces_to_check": to_check,
        "open": open_,
    }
    # (each line's ends, solved exactly from its equations: line_ends.py; a named point, another line, a new point not
    # yet named, or unresolved where the classifier cannot name the exact end)
    if os.path.exists("line_ends.json"):
        LE = json.load(open("line_ends.json"))["lines"]
        lines_out = {}
        for L_, rs in LE.items():
            ends = []
            for r in rs:
                if r["type"] == "goes on across a face":
                    continue
                if r["type"] != "ends":
                    ends.append({"status": "unresolved", "why": r["type"], "piece": r["piece"], "side": r["side"]})
                    continue
                on = r["on"]
                at = r.get("exact") or r.get("approx")
                if on.startswith("new:"):
                    st, name = "new point", on
                elif on.startswith("unnamed point"):
                    m = on.split("| ")[1].rstrip(")")
                    st, name = ("line", m) if m in LE else ("unresolved", on)
                elif on in LE:
                    st, name = "line", on
                elif on == "ERR" or on in walls or on in regions:
                    st, name = "unresolved", on
                else:
                    st, name = "point", on
                ends.append({"status": st, "on": name, "at": [round(v, 9) for v in at], "exact": "exact" in r})
            uniq = []
            for e in ends:
                if not any(e.get("on") == u.get("on") and e.get("at") and u.get("at") and
                           np.allclose(e["at"], u["at"], atol=1e-6) for u in uniq):
                    uniq.append(e)
            lines_out[L_] = {"ends": uniq, "bounds_walls": sorted((w for w in walls if L_ in E[w]), key=key)}
        out["lines"] = lines_out
    # (each wall as a whole on the 3-sphere: sphere_complex.py, its half-cell pieces carried by the symmetries and
    # joined across faces, at seams and where they are the same part)
    if os.path.exists("sphere_complex.json"):
        SC = json.load(open("sphere_complex.json"))
        for w, v in SC.items():
            if w in out["walls"]:
                out["walls"][w]["sphere"] = {
                    "patch_images": len(v["whole_piece"]), "stabiliser": len(v["stabiliser"]),
                    "joins": sorted({j["how"] for j in v["joins"]}),
                    "meets_copies_along": sorted(set(FACE_EDGES.get(w, [])) | set(SELF_MEET.get(w, [])))}
    for w, v in PR.get("walls", {}).items():                   # (sides the pieces show that the wall is not given)
        extra = {p: f for p, f in v["sides"].items() if any(x not in sep.get(w, set()) for x in p.split("|"))}
        if extra:
            to_check.setdefault("wall_sides", {})[w] = extra
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2 and w not in SAME_BOTH_SIDES])
    print("walls short of cuts:", {w: (len(cuts[w]), max(len(sep[w]) - 2, 0)) for w in walls
                                   if len([c for c in cuts[w] if not c.get("crossing")]) < len(sep[w]) - 2})
    print(f"closed regions: {len(regions) - len(open_)}/{len(regions)}")
    for k, v in to_check.items():
        print(f"pieces to check ({k}):")
        for x, y in v.items():
            print(f"  {x}: {y}")
    for r, m in open_.items():
        print(f"  {r}: " + "; ".join(m))


if __name__ == "__main__":
    main()
