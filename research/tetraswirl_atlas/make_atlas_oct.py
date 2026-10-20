"""atlas_oct.json, the traced data build_page.py plugs into the octahedral swirl page: the walls checked by the
classifier (walls_verified_oct.json) with their equations, the regions they separate and points on them; the lines
and points fixed by the group's rotations (seeds there have fewer vertices); and, once atlas_model.py has written
lines_points_oct.json, the lines and points where walls meet.

Every point found goes in "points" ({"id", "q": chart point, "beta", "hover"}) with a key row of its own ("types",
dim "point"); every wall and line has a key row and its points as samples ("samples": {id: [[x, y, z, seed, beta]]}).

    SWIRL_GROUP=oct python make_atlas_oct.py
"""
from __future__ import annotations

import json
import os
import re
from multiprocessing import Pool

import numpy as np

import domain as D

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def pretty(poly):
    """'b1*b3 - b2**2 + bV' -> 'β₁β₃ − β₂² + β_V = 0'."""
    def term(t):
        t = t.strip()
        c = ""
        m = re.match(r"^(\d+)\*(.*)$", t)
        if m:
            c, t = m.group(1), m.group(2)
        fs = []
        for f in t.split("*") if t else []:
            if not f:
                continue
            fs.append(f)
        out, i = "", 0
        while i < len(fs):
            v = fs[i]
            if i + 1 < len(fs) and fs[i + 1].isdigit():          # (from '**': split gives '', digit)
                out += _var(v) + fs[i + 1].translate(SUP)
                i += 2
            else:
                out += _var(v)
                i += 1
        return c + out
    s = poly.replace("**", "*")
    parts = re.split(r"\s([+-])\s", " + " + s if not s.startswith("-") else " " + s.replace("-", "- ", 1))
    out = ""
    for k in range(1, len(parts), 2):
        sign, t = parts[k], parts[k + 1]
        out += (" − " if sign == "-" else " + ") + term(t)
    out = out.strip()
    out = out[2:] if out.startswith("+ ") else ("−" + out[2:] if out.startswith("− ") else out)
    return out + " = 0"


def _var(v):
    return "β_V" if v == "bV" else "β" + v[1:].translate(SUB)


def chart(beta):
    return D.chart_from_seed(np.asarray(beta, float) @ D.CELL)


def seed_text(beta):
    x = np.asarray(beta, float) @ D.CELL
    x /= np.linalg.norm(x)
    return ", ".join(f"{v:.12f}" for v in x)


def beta_text(beta):
    b = np.asarray(beta, float)
    b = b / b.sum()
    return "(" + ", ".join(f"{v:.4f}" for v in b) + ")"


def sample_row(beta):
    return [round(float(x), 5) for x in chart(beta)] + [seed_text(beta), beta_text(beta)]


def label(beta):
    try:
        return D.signature(D.classify(np.asarray(beta, float) @ D.CELL / np.linalg.norm(np.asarray(beta, float) @ D.CELL)))
    except Exception:
        return "ERR"


def main():
    ver = json.load(open(f"walls_verified_{D.KEY}.json"))["walls"]
    ids = json.load(open(f"ids_{D.KEY}.json"))
    types, samples, points, lines = [], {}, [], []
    # walls: the confirmed surfaces, most crossed first
    walls = sorted([w for w in ver if w["verdict"] == "wall"], key=lambda w: -w["crossings"])
    for k, w in enumerate(walls, 1):
        wid = f"W{k}"
        sep = sorted({r for pr in w["separates"] for r in pr.split("|") if r.startswith("R")}, key=lambda r: int(r[1:]))
        on = w["on_wall"][0]["sig"] if w["on_wall"] else None
        pts = w["points"]
        types.append({"id": wid, "dim": "wall", "name": "", "sig": on if on and on[0].isdigit() else None,
                      "where": pretty(w["surface"]), "locus": f"checked by the classifier at {w['checked']} crossings "
                      f"({w['sides_differ']} with different shapes either side)",
                      "separates": sep, "bounds": [], "seed": seed_text(pts[0]) if pts else None,
                      "beta": beta_text(pts[0]) if pts else None, "equation": w["surface"]})
        samples[wid] = [sample_row(p) for p in pts]
    # lines and points fixed by rotations of the group (seeds there have 24, 16 or 8 vertices)
    third = 1 / 3
    sym_lines = [
        ("the axis V–F: β₁ = β₂ = β₃ (order-3 rotation; 16 vertices)", [1, 0, 0, 0], [0, third, third, third]),
        ("F–W₁: β_V = 0, β₂ = β₃ (half-turn; 24 vertices)", [0, third, third, third], [0, 1, 0, 0]),
        ("F–W₂: β_V = 0, β₁ = β₃ (half-turn; 24 vertices)", [0, third, third, third], [0, 0, 1, 0]),
        ("W₁ to the midpoint of VW₂: β₃ = 0, β_V = β₂ (half-turn; 24 vertices)", [0, 1, 0, 0], [0.5, 0, 0.5, 0]),
    ]
    sym_points = [
        ("V = β (1, 0, 0, 0): a vertex of the 16-cell", [1, 0, 0, 0]),
        ("W₁ = β (0, 1, 0, 0): a vertex of the 16-cell", [0, 1, 0, 0]),
        ("W₂ = β (0, 0, 1, 0): a vertex of the 16-cell", [0, 0, 1, 0]),
        ("F = β (0, ⅓, ⅓, ⅓): the centre of the face W₁W₂W₃", [0, third, third, third]),
        ("the midpoint of VW₁ = β (½, ½, 0, 0)", [0.5, 0.5, 0, 0]),
        ("the midpoint of W₁W₂ = β (0, ½, ½, 0)", [0, 0.5, 0.5, 0]),
    ]
    jobs = [np.array(a) * 0.5 + np.array(b) * 0.5 for _, a, b in sym_lines] + [np.array(p) for _, p in sym_points]
    with Pool(4) as pool:
        labs = pool.map(label, jobs)
    for k, ((name, a, b), lab) in enumerate(zip(sym_lines, labs), 1):
        lid = f"S{k}"
        ts = np.linspace(0.02, 0.98, 25)
        pts = [np.array(a) * (1 - t) + np.array(b) * t for t in ts]
        types.append({"id": lid, "dim": "line", "name": "", "sig": lab if lab[0].isdigit() else None,
                      "where": name, "bounds": [], "seed": seed_text(pts[12]), "beta": beta_text(pts[12])})
        lines.append({"id": lid, "pts": [[round(float(x), 5) for x in chart(p)] for p in [np.array(a), np.array(b)]],
                      "hover": f"{lid} · {name}"})
    for k, ((name, p), lab) in enumerate(zip(sym_points, labs[len(sym_lines):]), 1):
        pid = f"P{k}"
        types.append({"id": pid, "dim": "point", "name": "", "sig": lab if lab[0].isdigit() else None,
                      "where": name, "bounds": [], "seed": seed_text(p), "beta": beta_text(p)})
        points.append({"id": pid, "q": [round(float(x), 5) for x in chart(p)], "beta": beta_text(p), "hover": f"{pid} · {name}"})
    # lines and points where walls meet, when atlas_model.py has found them
    if os.path.exists(f"lines_points_{D.KEY}.json"):
        lp = json.load(open(f"lines_points_{D.KEY}.json"))
        wall_id = {t["equation"]: t["id"] for t in types if t.get("equation")}
        for k, l in enumerate(lp.get("lines", []), 1):
            lid = f"L{k}"
            ws = [wall_id.get(w, "?") for w in l["walls"]]
            types.append({"id": lid, "dim": "line", "name": "", "where": f"where {ws[0]} and {ws[1]} meet",
                          "bounds": [], "seed": seed_text(l["points"][0]), "beta": beta_text(l["points"][0]),
                          "sig": l.get("on") if str(l.get("on", ""))[:1].isdigit() else None})
            samples[lid] = [sample_row(p) for p in l["points"][:40]]
        for k, p in enumerate(lp.get("points", []), len(sym_points) + 1):
            pid = f"P{k}"
            ws = [wall_id.get(w, "?") for w in p["walls"]]
            types.append({"id": pid, "dim": "point", "name": "", "where": f"where {', '.join(ws)} meet",
                          "bounds": [], "seed": seed_text(p["beta"]), "beta": beta_text(p["beta"]),
                          "sig": p.get("on") if str(p.get("on", ""))[:1].isdigit() else None})
            points.append({"id": pid, "q": [round(float(x), 5) for x in chart(p["beta"])], "beta": beta_text(p["beta"]),
                           "hover": f"{pid} · where {', '.join(ws)} meet"})
    # (the regular polytopes among them, recognised by their counts)
    regular = {"8: cells 16x[4v4f]": "16-cell", "16: cells 8x[8v6f]": "tesseract", "24: cells 8x[6v8f]+16x[6v8f]": "24-cell"}
    for t in types:
        t.pop("equation", None)
        for k, v in regular.items():
            if (t.get("sig") or "").startswith(k + " |"):
                t["name"] = v
    json.dump({"types": types, "samples": samples, "lines": lines, "points": points}, open(f"atlas_{D.KEY}.json", "w"), indent=1)
    from collections import Counter
    print(f"atlas_{D.KEY}.json: {dict(Counter(t['dim'] for t in types))}; {len(points)} points")
    for t in types[:6]:
        print("  ", t["id"], t["where"], t.get("separates"))


if __name__ == "__main__":
    main()
