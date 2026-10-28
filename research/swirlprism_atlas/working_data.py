"""The working data in one place (working_data.json), from the verified base facts and what is computed from them.

The base (working_base.json): each shape's kind, name and counts and its example seed; each wall's edges, the regions on
its sides, its cuts and the lines it goes through; each region's walls. It is the source, edited directly; nothing here
re-derives it from older files. Added to it, each from the pipeline's own outputs:
  * each line's ends (line_ends.json), and the points at the ends of each wall's edges;
  * the closed set round each line (ring_test.json, settled as a puzzle on the walls' known sides and contacts): the
    lines of each wall's pieces, and each region's lines (full or partial) and points;
  * each wall's pieces (wall_parts.json: built out to its edges and cuts, named from this data), so each region's walls
    full or partial: full where every piece of the wall has the region on one side;
  * each wall as a whole on the 3-sphere (sphere_complex.json);
  * a region is closed where each line on the rim of one of its pieces is met by another of its pieces there (or the
    wall meets its own copy along it).
What is still open, or does not fit, is listed under "pieces_to_check" and "open".

    python working_data.py
"""
from __future__ import annotations

import json
import os

import numpy as np


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
    base = json.load(open("working_base.json"))
    shapes = base["shapes"]
    dims = {k: v["dim"] for k, v in shapes.items()}
    walls = sorted(base["walls"], key=key)
    regions = sorted(base["regions"], key=key)
    SELF_MEET, FACE_EDGES, SAME_BOTH_SIDES = base["self_meet"], base["face_edges"], base["same_both_sides"]
    E = {w: set(base["walls"][w]["edges"]) for w in walls}
    sep = {w: set(base["walls"][w]["regions"]) for w in walls}
    cuts = {w: base["walls"][w]["cuts"] for w in walls}
    THROUGH = {w: base["walls"][w].get("through", []) for w in walls if base["walls"][w].get("through")}
    rw_ = {r: set(base["regions"][r]["walls"]) for r in regions}
    to_check, open_ = {}, {}

    LE = json.load(open("line_ends.json"))["lines"] if os.path.exists("line_ends.json") else {}
    line_pts = {L_: sorted({r["on"] for r in rs if r["type"] == "ends" and r["on"] not in LE
                            and not r["on"].startswith(("unnamed", "ERR", "new:"))
                            and r["on"] not in walls and r["on"] not in regions})
                for L_, rs in LE.items()}

    # (the closed set around each line: the ring test, the walls and regions in their cyclic order round the line at
    # points along it, settled as a puzzle on the walls' known sides and contacts; a wall's piece between regions a and
    # b has the line L where L's ring shows the wall between a and b; a region has L where it appears in L's ring, all
    # along it (full) or along part of it (partial))
    RT = json.load(open("ring_test.json"))["rings"] if os.path.exists("ring_test.json") else {}
    ring_lines, piece_lines, ring_unsure, ring_settled, contacts_found = {}, {}, {}, {}, {}
    sides_of = {w: set(sep[w]) for w in walls}
    touch = {}
    for w in walls:
        for L_ in E[w] | set(THROUGH.get(w, [])) | set(SELF_MEET.get(w, [])):
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
    for k_, v_ in (("ring_stretches_unsure", ring_unsure), ("ring_stretches_settled", ring_settled),
                   ("contacts_found_by_rings", contacts_found)):
        if v_:
            to_check[k_] = v_

    # (each wall's pieces, named from this data: wall_parts.py; a region's walls are full where every piece of the wall
    # has it on one side)
    WP = json.load(open("wall_parts.json")) if os.path.exists("wall_parts.json") else {}
    pieces_of = {w: [p.get("between") for p in v["pieces"]] for w, v in WP.items()}
    notes = {w: v["notes"] for w, v in WP.items() if v.get("notes")}
    if notes:
        to_check["wall_pieces"] = notes

    def extent(r, w):
        ps = pieces_of.get(w)
        if not ps:
            return "no pieces built"
        named = [p for p in ps if p]
        mine = [p for p in named if r in p]
        if not mine:
            return "not found beside it in the pieces"
        return "full" if len(mine) == len(ps) else "partial"

    # (closure: each line on the rim of one of a region's pieces is met by another of its pieces there, or the wall
    # meets its own copy along it, or lies along a face of the cell there)
    lines_set = set(LE) | {k for k, v in dims.items() if v == "line"}
    for r in regions:
        rims = []
        for w in rw_.get(r, set()):
            for p in WP.get(w, {}).get("pieces", []):
                if p.get("between") and r in p["between"]:
                    rims.append((w, {n for (_, n), (_, m) in p["boundary"] if n == m and n in lines_set}))
        miss = []
        for w, Ls in rims:
            for L_ in Ls:
                if L_ in SELF_MEET.get(w, []) + FACE_EDGES.get(w, []):
                    continue
                if not any(L_ in Ls2 for w2, Ls2 in rims if (w2, Ls2) is not (w, Ls)):
                    miss.append(f"{w}'s piece edge {L_} meets no other of its pieces")
        if miss:
            open_[r] = sorted(set(miss))

    def region_data(r):
        lines_r = ring_lines.get(r, {})
        pts = sorted({p for L_ in lines_r for p in line_pts.get(L_, [])}, key=key)
        return {"walls": sorted(rw_.get(r, set()), key=key),
                "wall_extent": {w: extent(r, w) for w in sorted(rw_.get(r, set()), key=key)},
                "lines": {L_: lines_r[L_] for L_ in sorted(lines_r, key=key)}, "points": pts}

    out = {
        "_about": __doc__.strip().split("\n\n")[0],
        "shapes": shapes,
        "self_meet": SELF_MEET, "face_edges": FACE_EDGES, "same_both_sides": SAME_BOTH_SIDES,
        "corrections": base["corrections"],
        "walls": {w: {"edges": sorted(E[w], key=key),
                      "boundary_points": sorted({p for L_ in E[w] if L_ in line_pts for p in line_pts[L_]}, key=key),
                      "pieces": [{"between": pair.split("|"), "lines": sorted(ls, key=key),
                                  "points": sorted({p for L_ in ls for p in line_pts.get(L_, [])}, key=key)}
                                 for pair, ls in sorted(piece_lines.get(w, {}).items())], "through": THROUGH.get(w, []),
                      **({"both_sides": SAME_BOTH_SIDES[w]} if w in SAME_BOTH_SIDES else {}),
                      "points": sorted({p for L_ in E[w] if L_ in line_pts for p in line_pts[L_]}, key=key),
                      "regions": sorted(sep[w], key=key), "cuts_needed": max(len(sep[w]) - 2, 0), "cuts": cuts[w]}
                  for w in walls},
        "regions": {r: region_data(r) for r in regions},
        "pieces_to_check": to_check,
        "open": open_,
    }
    # (each line's ends, solved exactly from its equations: line_ends.py; a named point, another line, a new point not
    # yet named, or unresolved where the classifier cannot name the exact end)
    if LE:
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
    for w, ps in pieces_of.items():                        # (sides the pieces show that the wall is not given)
        extra = sorted({"|".join(p) for p in ps if p and any(x not in sep.get(w, set()) for x in p)})
        if extra:
            to_check.setdefault("wall_sides", {})[w] = extra
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2 and w not in SAME_BOTH_SIDES])
    print(f"closed regions: {len(regions) - len(open_)}/{len(regions)}")
    for k, v in to_check.items():
        print(f"pieces to check ({k}): {len(v)}")
    for r, m in open_.items():
        print(f"  {r}: " + "; ".join(m))


if __name__ == "__main__":
    main()
