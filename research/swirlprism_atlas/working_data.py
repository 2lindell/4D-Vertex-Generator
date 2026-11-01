"""The working data in one place (working_data.json), from the verified base facts and what is computed from them.

The base (working_base.json): each shape's kind, name and counts and its example seed; each wall's edges, the regions on
its sides, its cuts and the lines it goes through; each region's walls. It is the source, edited directly; nothing here
re-derives it from older files. Added to it, each from the pipeline's own outputs:
  * each line's ends (line_ends.json), each named by the exact point it is at (exact_points.json, by position), and
    each wall's corners: the exact points at the ends of its edge lines on its own surface;
  * the exact geometry itself (exact_points.json, exact_lines.json, exact_defs.json), under "exact";
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


def extended_group():
    from core import FOLD, extended
    E_ = extended()
    return np.concatenate([E_, E_ @ FOLD])


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
    # (the exact geometry: every meeting point solved to 40 digits (exact_points.json), each line's pieces put exactly on
    # its surfaces (exact_lines.json), each wall's surfaces as placed (exact_defs.json); a line's end is named by the
    # exact point it is at, found by position among every image of every exact point, so a point named after a line
    # (X2, X29, X81, ...) is named, and the end is never labelled by what it was first seen on)
    XP = json.load(open("exact_points.json"))["points"] if os.path.exists("exact_points.json") else []
    XL = json.load(open("exact_lines.json"))["lines"] if os.path.exists("exact_lines.json") else {}
    XD = json.load(open("exact_defs.json")) if os.path.exists("exact_defs.json") else {"walls": {}}
    G_ = extended_group()
    imgs, img_names = [], []
    for p_ in XP:
        if not p_["exact"]:
            continue
        Y = G_ @ np.array(p_["float"])
        s_ = Y.sum(1)
        Y = Y[np.abs(s_) > 1e-12] / s_[np.abs(s_) > 1e-12, None]
        imgs.append(Y)
        img_names += [p_["name"]] * len(Y)
    imgs = np.vstack(imgs) if imgs else np.zeros((0, 4))

    def exact_name(at):
        if at is None or not len(imgs):
            return None
        x = np.asarray(at, float)
        x = x / x.sum()
        d = np.linalg.norm(imgs - x, axis=1)
        return img_names[int(d.argmin())] if d.min() < 1e-6 else None
    for L_, rs in LE.items():
        for r in rs:
            if r["type"] == "ends":
                nm_ = exact_name(r.get("exact") or r.get("approx"))
                if nm_:
                    r["on"] = nm_
    # (a wall's corners: the exact points at the ends of its own edge lines that lie on the wall's own surface, some
    # image of one of its surfaces as placed)
    from core import library
    planes_, curved_ = library()
    LIB_ = {s_.name: s_ for s_ in list(planes_) + list(curved_)}
    GI_ = np.linalg.inv(G_)

    def on_wall(w, at):
        x = np.asarray(at, float)
        for n_ in XD["walls"].get(w, {}).get("surface", []):
            if n_ in LIB_:
                v_ = np.ravel(LIB_[n_].f(np.einsum("gij,j->gi", GI_, x)))
                if (np.abs(v_) < 1e-7 * np.abs(v_).max()).any():
                    return True
        return False
    corners = {}
    for w in walls:
        c_ = set()
        for L_ in E[w]:
            for r in LE.get(L_, []):
                at = r.get("exact") or r.get("approx")
                if r["type"] == "ends" and exact_name(at) and on_wall(w, at):
                    c_.add(r["on"])
        corners[w] = c_

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
        return {"walls": sorted(rw_.get(r, set()), key=key),
                "wall_extent": {w: extent(r, w) for w in sorted(rw_.get(r, set()), key=key)}}

    out = {
        "_about": __doc__.strip().split("\n\n")[0],
        "shapes": shapes,
        "self_meet": SELF_MEET, "face_edges": FACE_EDGES, "same_both_sides": SAME_BOTH_SIDES,
        "corrections": base["corrections"],
        "walls": {w: {"edges": sorted(E[w], key=key),
                      "boundary_points": sorted(corners[w], key=key),
                      "through": THROUGH.get(w, []),
                      **({"both_sides": SAME_BOTH_SIDES[w]} if w in SAME_BOTH_SIDES else {}),
                      "points": sorted(corners[w], key=key),
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
                if exact_name(at):
                    st, name = "point", exact_name(at)
                elif on.startswith("new:"):
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
    # (the exact geometry itself, so this is the one database for it)
    out["exact"] = {
        "points": [{k_: p_[k_] for k_ in ("name", "b", "float", "surfaces", "from")} for p_ in XP if p_["exact"]],
        "lines": {L_: {"points": v_["points"], "surfaces": v_["surfaces"]} for L_, v_ in XL.items()},
        "wall_surfaces": {w: XD["walls"].get(w, {}).get("surface", []) for w in walls}}
    # (where the walls' pieces are not built (wall_parts.json missing), each region's wall extents are kept from the
    # last build that had them)
    if not WP and os.path.exists("working_data.json"):
        prev = json.load(open("working_data.json"))
        for r in out["regions"]:
            if r in prev.get("regions", {}):
                out["regions"][r]["wall_extent"] = prev["regions"][r].get("wall_extent", out["regions"][r]["wall_extent"])
        if "wall_pieces" in prev.get("pieces_to_check", {}):
            out["pieces_to_check"]["wall_pieces"] = prev["pieces_to_check"]["wall_pieces"]
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2 and w not in SAME_BOTH_SIDES])
    print(f"closed regions: {len(regions) - len(open_)}/{len(regions)}")
    for k, v in to_check.items():
        print(f"pieces to check ({k}): {len(v)}")
    for r, m in open_.items():
        print(f"  {r}: " + "; ".join(m))


if __name__ == "__main__":
    main()
