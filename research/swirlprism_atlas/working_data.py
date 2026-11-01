"""The working data in one place (working_data.json), from the verified base facts and what is computed from them.

The base (working_base.json): each shape's kind, name and counts and its example seed; each wall's edges, the regions on
its sides, its cuts and the lines it goes through; each region's walls. It is the source, edited directly; nothing here
re-derives it from older files. Added to it, each from the pipeline's own outputs:
  * each line's ends (line_ends.json), each named by the exact point it is at (exact_points.json, by position), and
    each wall's corners: the exact points at the ends of its edge lines on its own surface;
  * the exact geometry itself (exact_points.json, exact_lines.json, exact_defs.json), under "exact";

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
    walls = sorted(base["walls"], key=key)
    regions = sorted(base["regions"], key=key)
    SELF_MEET, FACE_EDGES, SAME_BOTH_SIDES = base["self_meet"], base["face_edges"], base["same_both_sides"]
    E = {w: set(base["walls"][w]["edges"]) for w in walls}
    sep = {w: set(base["walls"][w]["regions"]) for w in walls}
    cuts = {w: base["walls"][w]["cuts"] for w in walls}
    THROUGH = {w: base["walls"][w].get("through", []) for w in walls if base["walls"][w].get("through")}
    rw_ = {r: set(base["regions"][r]["walls"]) for r in regions}

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

    def region_data(r):
        return {"walls": sorted(rw_.get(r, set()), key=key)}

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
    # (the exact geometry itself, so this is the one database for it)
    out["exact"] = {
        "points": [{k_: p_[k_] for k_ in ("name", "b", "float", "surfaces", "from")} for p_ in XP if p_["exact"]],
        "lines": {L_: {"points": v_["points"], "surfaces": v_["surfaces"]} for L_, v_ in XL.items()},
        "wall_surfaces": {w: XD["walls"].get(w, {}).get("surface", []) for w in walls}}
    json.dump(out, open("working_data.json", "w"), indent=1)
    print("walls with one side:", [w for w in walls if len(sep[w]) < 2 and w not in SAME_BOTH_SIDES])


if __name__ == "__main__":
    main()
