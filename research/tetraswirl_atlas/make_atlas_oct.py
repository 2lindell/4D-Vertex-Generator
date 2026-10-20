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
    # walls: one row per shape found on walls (a shape is one shape wherever it lies, as A1 in the swirlprism atlas
    # has two positions), each listing the surfaces it lies on; points on each confirmed surface where it really is a
    # wall (the exact model: atlas_model.wall_here), spread out, and classified
    from atlas_model import Surf, project, wall_here, in_domain, PIECES, TYPE
    from walls_exact import domain_betas
    surfs = [w for w in ver if w["verdict"] == "wall"]
    Bd = domain_betas(4000, seed=21)
    found = []                                       # (surface text, beta, region pair)
    for w in surfs:
        sf = Surf(w["surface"])
        B, ok = project(Bd, [sf])
        B = B[ok & in_domain(B)]
        near, far, okw = wall_here(B, sf)
        B, near, far = B[okw], near[okw], far[okw]
        # (only points well inside a patch: nudged 0.003 along the surface four ways, still the same wall between the
        # same pieces; points near a line, where walls cross or meet the domain's faces, carry a line's shape)
        if len(B):
            n = sf.grad(B)
            n = n - n.mean(1, keepdims=True)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            e = np.array([1.0, -1.0, 0.0, 0.0])
            t1 = e - (n @ e)[:, None] * n
            t1 = t1 - t1.mean(1, keepdims=True)
            t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
            t2 = np.cross(np.c_[n[:, 1:]], np.c_[t1[:, 1:]])          # (a second direction within sum = 1)
            t2 = np.c_[-t2.sum(1), t2]
            t2 = t2 - t2.mean(1, keepdims=True) - (np.einsum("ni,ni->n", t2, n))[:, None] * n
            t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
            keep = np.ones(len(B), bool)
            for d in (t1, -t1, t2, -t2):
                Q, okq = project(B + 3e-3 * d, [sf])
                nq, fq, wq = wall_here(Q, sf)
                keep &= okq & in_domain(Q) & wq & (nq == near) & (fq == far)
            B, near, far = B[keep], near[keep], far[keep]
        rng_ = np.random.default_rng(len(found))
        pick = rng_.permutation(len(B))[:20]
        for i in pick:
            found.append((w["surface"], B[i], "|".join(sorted([TYPE[PIECES[near[i]]], TYPE[PIECES[far[i]]]]))))
    with Pool(4) as pool:
        wl = pool.map(label, [f[1] for f in found])
    by_shape = {}
    for (surf, b, pair), sg in zip(found, wl):
        if not sg[:1].isdigit():
            continue
        e = by_shape.setdefault(sg, {"surfaces": {}, "pairs": set(), "pts": []})
        e["surfaces"][surf] = e["surfaces"].get(surf, 0) + 1
        e["pairs"].add(pair)
        e["pts"].append(b)
    # (a shape found on a surface that is one of the regions' shapes is not a transitional: the region reaches onto
    # its own boundary there, where a cell collapses flat; it is noted on the region's row, not listed as a wall)
    region_of = {sg: rid for sg, rid in ids.items() if rid.startswith("R")}
    reach = {}
    for sg in [x for x in by_shape if x in region_of]:
        e = by_shape.pop(sg)
        reach[region_of[sg]] = e
    for rid, e in reach.items():
        other = sorted({r for pr in e["pairs"] for r in pr.split("|")} - {rid}, key=lambda r: int(r[1:]))
        types.append({"id": rid, "dim": "region", "locus": f"also holds on parts of {len(e['surfaces'])} surface"
                      f"{'s' if len(e['surfaces']) > 1 else ''} where it meets {', '.join(other)} (no transitional shape there)"})
    for k, (sg, e) in enumerate(sorted(by_shape.items(), key=lambda kv: -len(kv[1]["pts"])), 1):
        wid = f"W{k}"
        surf_list = sorted(e["surfaces"].items(), key=lambda kv: -kv[1])
        where = "on " + "; ".join(pretty(f) for f, _ in surf_list[:4]) + (f"; and {len(surf_list) - 4} more surfaces" if len(surf_list) > 4 else "")
        sep = sorted({r for pr in e["pairs"] for r in pr.split("|")}, key=lambda r: int(r[1:]))
        types.append({"id": wid, "dim": "wall", "name": "", "sig": sg, "where": where,
                      "locus": f"{len(e['pts'])} points on {len(surf_list)} surface{'s' if len(surf_list) > 1 else ''}",
                      "separates": sep, "bounds": [], "seed": seed_text(e["pts"][0]), "beta": beta_text(e["pts"][0]),
                      "surfaces": [f for f, _ in surf_list]})
        samples[wid] = [sample_row(p) for p in e["pts"]]
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
        ("F = β (0, ⅓, ⅓, ⅓): the centre of the face W₁W₂W₃, a vertex of the group's other 16-cell", [0, third, third, third]),
        ("the midpoint of VW₁ = β (½, ½, 0, 0)", [0.5, 0.5, 0, 0]),
        ("the midpoint of W₁W₂ = β (0, ½, ½, 0)", [0, 0.5, 0.5, 0]),
    ]
    # (copies glued together: a point or line is listed once, drawn at each place it appears in the domain)
    from four_d_vertex_generator.generation import generate_vertices_from_seed

    def orbit(beta):
        x = np.asarray(beta, float) @ D.CELL
        return generate_vertices_from_seed(x / np.linalg.norm(x), D.ACTION)

    def same(A, B):
        return len(A) == len(B) and all(np.min(np.linalg.norm(B - a, axis=1)) < 1e-9 for a in A)

    def cluster(items, key):
        groups = []
        for it in items:
            o = orbit(key(it))
            for g in groups:
                if same(g["orbit"], o):
                    g["members"].append(it)
                    break
            else:
                groups.append({"orbit": o, "members": [it]})
        return [g["members"] for g in groups]
    mid = lambda l: np.array(l[1]) * 0.5 + np.array(l[2]) * 0.5
    with Pool(4) as pool:
        lab_l = pool.map(label, [mid(l) for l in sym_lines])
        lab_p = pool.map(label, [np.array(p[1]) for p in sym_points])

    def by_shape_then_position(items, labs_, key):
        """[(shape, [[items at one position], ...])]: one entry per shape, its positions (orbits) inside it."""
        out = {}
        for it, lb in zip(items, labs_):
            out.setdefault(lb, []).append(it)
        return [(lb, cluster(its, key)) for lb, its in out.items()]
    line_groups = by_shape_then_position(sym_lines, lab_l, mid)
    point_groups = by_shape_then_position(sym_points, lab_p, lambda p: p[1])
    labs = [lb for lb, _ in line_groups] + [lb for lb, _ in point_groups]
    pos_text = lambda poss: "; ".join(" = ".join(x[0].split(":")[0] for x in pos) for pos in poss)
    for k, ((lab, poss), _) in enumerate(zip(line_groups, labs), 1):
        lid = f"S{k}"
        g = [x for pos in poss for x in pos]
        name = g[0][0] if len(g) == 1 else (f"{len(poss)} position{'s' if len(poss) > 1 else ''}: {pos_text(poss)} ("
                                            + "; ".join(x[0] for x in g) + ")")
        p0 = mid(g[0])
        types.append({"id": lid, "dim": "line", "name": "", "sig": lab if lab[0].isdigit() else None,
                      "where": name, "bounds": [], "seed": seed_text(p0), "beta": beta_text(p0)})
        for _, a, b in g:
            lines.append({"id": lid, "pts": [[round(float(x), 5) for x in chart(p)] for p in (np.array(a), np.array(b))],
                          "hover": f"{lid} · {name}"})
    for k, (lab, poss) in enumerate(point_groups, 1):
        pid = f"P{k}"
        g = [x for pos in poss for x in pos]
        name = g[0][0] if len(g) == 1 else (f"{len(poss)} position{'s' if len(poss) > 1 else ''} (one per orbit; a "
                                            f"position at several places is one point, glued): {pos_text(poss)} ("
                                            + "; ".join(x[0] for x in g) + ")")
        p0 = g[0][1]
        types.append({"id": pid, "dim": "point", "name": "", "sig": lab if lab[0].isdigit() else None,
                      "where": name, "bounds": [], "seed": seed_text(p0), "beta": beta_text(p0)})
        for nm, p in g:
            points.append({"id": pid, "q": [round(float(x), 5) for x in chart(p)], "beta": beta_text(p), "hover": f"{pid} · {nm}"})
    n_sym_points = len(point_groups)
    # lines and points where walls meet (atlas_model.py), one row per shape found there, as for the walls
    if os.path.exists(f"lines_points_{D.KEY}.json"):
        lp = json.load(open(f"lines_points_{D.KEY}.json"))
        wall_id = {f: t["id"] for t in types for f in t.get("surfaces", [])}
        sym_sigs = {t.get("sig") for t in types if t["dim"] in ("line", "point")}
        groups = {}
        for l in lp.get("lines", []):
            sg = l.get("on", "")
            # (a line's shape that is a region's or a wall's is not a new transitional)
            wall_sigs = {t.get("sig") for t in types if t["dim"] == "wall"}
            if str(sg)[:1].isdigit() and sg not in sym_sigs and sg not in region_of and sg not in wall_sigs:
                groups.setdefault(sg, []).append(l)
        for k, (sg, ls) in enumerate(sorted(groups.items(), key=lambda kv: -sum(len(x["points"]) for x in kv[1])), 1):
            lid = f"L{k}"
            pairs = sorted({" × ".join(wall_id.get(w, "?") for w in l["walls"]) for l in ls})
            types.append({"id": lid, "dim": "line", "name": "", "sig": sg, "bounds": [],
                          "where": f"where walls meet: {', '.join(pairs[:6])}{f' and {len(pairs) - 6} more' if len(pairs) > 6 else ''}",
                          "locus": f"{len(ls)} curve{'s' if len(ls) > 1 else ''}",
                          "seed": seed_text(ls[0]["points"][0]), "beta": beta_text(ls[0]["points"][0])})
            samples[lid] = [sample_row(p) for l in ls for p in l["points"][:: max(1, len(l["points"]) // 12)]][:120]
        pgroups = {}
        for p in lp.get("points", []):
            sg = p.get("on", "")
            known = {t.get("sig") for t in types if t["dim"] in ("wall", "line")} | set(region_of)
            if str(sg)[:1].isdigit() and (sg not in known or any(t["dim"] == "point" and t.get("sig") == sg for t in types)):
                pgroups.setdefault(sg, []).append(p)
        n0 = len([t for t in types if t["dim"] == "point"])
        for k, (sg, ps) in enumerate(sorted(pgroups.items(), key=lambda kv: -len(kv[1])), n0 + 1):
            # (a shape already listed at a symmetry point: its new places are added to that row)
            old = next((t for t in types if t["dim"] == "point" and t.get("sig") == sg), None)
            pid = old["id"] if old else f"P{k}"
            if not old:
                walls_ = sorted({wall_id.get(w, "?") for p in ps for w in p["walls"]})
                types.append({"id": pid, "dim": "point", "name": "", "sig": sg, "bounds": [],
                              "where": f"where walls meet ({', '.join(walls_[:6])}{' …' if len(walls_) > 6 else ''})",
                              "locus": f"{len(ps)} place{'s' if len(ps) > 1 else ''}",
                              "seed": seed_text(ps[0]["beta"]), "beta": beta_text(ps[0]["beta"])})
            for p in ps:
                points.append({"id": pid, "q": [round(float(x), 5) for x in chart(p["beta"])], "beta": beta_text(p["beta"]),
                               "hover": f"{pid} · where {', '.join(wall_id.get(w, '?') for w in p['walls'])} meet"})
    # (the regular polytopes among them, recognised by their counts)
    regular = {"8: cells 16x[4v4f]": "16-cell", "16: cells 8x[8v6f]": "tesseract", "24: cells 8x[6v8f]+16x[6v8f]": "24-cell"}
    for t in types:
        t.pop("equation", None)
        t.pop("surfaces", None)
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
