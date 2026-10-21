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


_T = 1 / 3
_R2 = np.sqrt(2)
_V, _W1, _W2, _F = [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, _T, _T, _T]
# the rings of [3;3:1] (great circles fixed by a rotation: seeds there have fewer vertices) where they cross the domain:
# (name, kind, vertices, from, to); the corner names go in the text
RINGS = [
    ("swirl ring V–F (β₁ = β₂ = β₃)", "rectic swirl ring (order-3 rotation)", 16, _V, _F),
    ("cross ring F–W₁ (β_V = 0, β₂ = β₃)", "essic cross ring (half-turn)", 24, _F, _W1),
    ("cross ring F–W₂ (β_V = 0, β₁ = β₃)", "essic cross ring (half-turn)", 24, _F, _W2),
    ("cross ring F–M₁₂ (β_V = 0, β₁ = β₂)", "essic cross ring (half-turn)", 24, _F, [0, 0.5, 0.5, 0]),
    ("cross ring W₁–M₀₂ (β₃ = 0, β_V = β₂)", "essic cross ring (half-turn)", 24, _W1, [0.5, 0, 0.5, 0]),
]
# the ghost girdles (circles a rotation turns along itself, fixing no point) that cross the domain
GIRDLES = [
    ("6/3 gyrogon ghost girdle M₀₁–M₁₂ (β₃ = 0, β₁ = ½): 6 of the 48 images on one great circle",
     [0.5, 0.5, 0, 0], [0, 0.5, 0.5, 0]),
    ("skew square ghost girdle (8 of the 48 images on one great circle), from β (√2/4, ½ − √2/4, √2/4, ½ − √2/4) on "
     "β₁ = β₃ to β (1 − √2/2, 1 − √2/2, √2 − 1, 0) on β₃ = 0",
     [_R2 / 4, 0.5 - _R2 / 4, _R2 / 4, 0.5 - _R2 / 4], [1 - _R2 / 2, 1 - _R2 / 2, _R2 - 1, 0]),
    ("skew square ghost girdle (8 of the 48 images on one great circle), from β (√2 − 1, 1 − √2/2, 1 − √2/2, 0) on "
     "β₃ = 0 to β (√2/4, √2/4, ½ − √2/4, ½ − √2/4) on β₂ = β₃",
     [_R2 - 1, 1 - _R2 / 2, 1 - _R2 / 2, 0], [_R2 / 4, _R2 / 4, 0.5 - _R2 / 4, 0.5 - _R2 / 4]),
]
CORNERS = {(1, 0, 0, 0): "V", (0, 1, 0, 0): "W₁", (0, 0, 1, 0): "W₂", (0, _T, _T, _T): "F",
           (0.5, 0.5, 0, 0): "M₀₁ (the midpoint of VW₁)", (0, 0.5, 0.5, 0): "M₁₂ (the midpoint of W₁W₂)",
           (0.5, 0, 0.5, 0): "M₀₂ (the midpoint of VW₂)", (0.25, 0.25, 0.25, 0.25): "C (the cell's centre)",
           (_T, _T, _T, 0): "the centre of the face VW₁W₂"}


def seg_dist(B, a, b):
    """Distance of each beta in B from the segment a-b."""
    B = B / B.sum(axis=1, keepdims=True)
    d = b - a
    t = np.clip((B - a) @ d / (d @ d), 0, 1)
    return np.linalg.norm(B - a - t[:, None] * d, axis=1)


def exact_text(beta):
    """A place on a ring: its corner name, or its weights as fractions."""
    from fractions import Fraction
    b = np.asarray(beta, float)
    for k, v in CORNERS.items():
        if np.abs(b - np.array(k)).max() < 1e-9:
            return f"{v} = β ({', '.join(_frac(x) for x in k)})"
    fr = [Fraction(float(x)).limit_denominator(60) for x in b]
    if max(abs(float(f) - x) for f, x in zip(fr, b)) < 1e-9:
        return "β (" + ", ".join(_frac(float(f)) for f in fr) + ")"
    return "β " + beta_text(b)


def _frac(x):
    from fractions import Fraction
    f = Fraction(float(x)).limit_denominator(60)
    return {0: "0", 1: "1"}.get(f, "½" if f == Fraction(1, 2) else "⅓" if f == Fraction(1, 3) else "¼" if f == Fraction(1, 4)
                                      else "⅙" if f == Fraction(1, 6) else f"{f.numerator}/{f.denominator}")


def _snap(t):
    """A breakpoint's parameter, made exact when it is a simple fraction."""
    from fractions import Fraction
    f = Fraction(t).limit_denominator(60)
    return float(f) if abs(float(f) - t) < 1e-8 else t


def scan_rings(n=2000):
    """Each ring's piece in the domain split where its shape changes. Returns (pieces, spots): pieces {"sig", "ring",
    "kind", "copies", "a", "b", "mid", "from", "to"}; spots (the ends of the pieces and the corners, where the shape is
    not the pieces' either side) {"sig", "beta", "text", "rings", "copies"}."""
    pieces, spots = [], []
    with Pool(4) as pool:
        for name, kind, copies, a, b in RINGS:
            a, b = np.array(a, float), np.array(b, float)
            at = lambda t: (1 - t) * a + t * b
            ts = np.linspace(0, 1, n + 1)
            L = pool.map(label, [at(t) for t in ts], chunksize=25)
            runs = []
            for t, l in zip(ts, L):
                if runs and runs[-1][2] == l:
                    runs[-1][1] = t
                else:
                    runs.append([t, t, l])
            # (each change bisected to 1e-13 and snapped to a simple fraction when it is one)
            cuts = [0.0]
            for (_, t0, l0), (t1, _, _) in zip(runs, runs[1:]):
                lo, hi = t0, t1
                for _ in range(45):
                    m = (lo + hi) / 2
                    lo, hi = (m, hi) if label(at(m)) == l0 else (lo, m)
                c = _snap((lo + hi) / 2)
                if c - cuts[-1] > 1e-9:
                    cuts.append(c)
            if 1.0 - cuts[-1] > 1e-9:
                cuts.append(1.0)
            for t0, t1 in zip(cuts, cuts[1:]):
                sg = label(at((t0 + t1) / 2))
                if pieces and pieces[-1]["ring"] == name and pieces[-1]["sig"] == sg:
                    pieces[-1].update(b=at(t1), to=exact_text(at(t1)))
                    continue
                pieces.append({"sig": sg, "ring": name, "kind": kind, "copies": copies, "a": at(t0), "b": at(t1),
                               "mid": at((t0 + t1) / 2), "from": exact_text(at(t0)), "to": exact_text(at(t1))})
            mine = [p for p in pieces if p["ring"] == name]
            for t in cuts:
                sg = label(at(t))
                if any(sg == p["sig"] and (np.allclose(p["a"], at(t)) or np.allclose(p["b"], at(t))) for p in mine):
                    continue                       # (the end of a piece with the piece's own shape)
                hit = next((s for s in spots if np.allclose(s["beta"], at(t), atol=1e-9)), None)
                if hit:
                    hit["rings"].add(name.split(" (")[0])
                else:
                    spots.append({"sig": sg, "beta": at(t), "text": exact_text(at(t)), "rings": {name.split(" (")[0]},
                                  "copies": int(sg.split(":")[0]) if sg[:1].isdigit() else copies})
    return pieces, spots


def group_by(items):
    """[(shape, [items])], in order of first appearance."""
    out = {}
    for it in items:
        out.setdefault(it["sig"], []).append(it)
    return list(out.items())


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
    # (noted by hand: where a region's shape holds on a surface it may gain a 2-fold symmetry; Bowers' "2-astettic")
    extra_symmetry = {"R1": "2-astettic there (an extra 2-fold symmetry)", "R5": "2-astettic there (an extra 2-fold symmetry)"}
    for rid, e in reach.items():
        other = sorted({r for pr in e["pairs"] for r in pr.split("|")} - {rid}, key=lambda r: int(r[1:]))
        types.append({"id": rid, "dim": "region", "locus": f"also holds on parts of {len(e['surfaces'])} surface"
                      f"{'s' if len(e['surfaces']) > 1 else ''} where it meets {', '.join(other)} (no transitional shape there"
                      + (f"; {extra_symmetry[rid]}" if rid in extra_symmetry else "") + ")"})
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
    # lines and points fixed by rotations of the group (seeds there have 24, 16 or 8 vertices): the rings of [3;3:1]
    # where they cross the domain, each scanned with the classifier and split exactly where its shape changes
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
    pieces, spots = scan_rings()
    ring_segs = [(np.array(r[3], float), np.array(r[4], float)) for r in RINGS]
    for k, (sg, ps) in enumerate(group_by(pieces), 1):
        lid = f"S{k}"
        rings_ = sorted({p["ring"] for p in ps})
        types.append({"id": lid, "dim": "line", "name": "", "sig": sg if sg[:1].isdigit() else None,
                      "where": "; ".join(f"{p['ring']}, {p['from']} to {p['to']}" for p in ps),
                      "locus": f"{ps[0]['copies']} vertices: {' and '.join(sorted({p['kind'] for p in ps}))}"
                               + (f"; {len(ps)} pieces" if len(ps) > 1 else ""),
                      "bounds": [], "seed": seed_text(ps[0]["mid"]), "beta": beta_text(ps[0]["mid"])})
        for p in ps:
            lines.append({"id": lid, "pts": [[round(float(x), 5) for x in chart(q)] for q in (p["a"], p["b"])],
                          "hover": f"{lid} · {p['ring']}, {p['from']} to {p['to']}"})
            samples.setdefault(lid, []).extend(sample_row((1 - t) * p["a"] + t * p["b"]) for t in np.linspace(0.1, 0.9, 5))
    for k, (sg, ps) in enumerate(group_by(spots), 1):
        pid = f"P{k}"
        poss = cluster(ps, lambda p: p["beta"])
        text = "; ".join(" = ".join(p["text"] for p in pos) for pos in poss)
        types.append({"id": pid, "dim": "point", "name": "", "sig": sg if sg[:1].isdigit() else None,
                      "where": (f"{len(poss)} positions (one per orbit; a position at several places is one point, "
                                f"glued): " if len(poss) > 1 else "") + text,
                      "locus": f"{ps[0]['copies']} vertices; on {', '.join(sorted({r for p in ps for r in p['rings']}))}",
                      "bounds": [], "seed": seed_text(ps[0]["beta"]), "beta": beta_text(ps[0]["beta"])})
        for p in ps:
            points.append({"id": pid, "q": [round(float(x), 5) for x in chart(p["beta"])], "beta": beta_text(p["beta"]),
                           "hover": f"{pid} · {p['text']}"})
    # the ghost girdles: circles a rotation turns along itself; seeds there keep all 48 images (no fewer vertices),
    # but 6 or 8 of them lie on one great circle; drawn for reference
    for k, (name, a, b) in enumerate(GIRDLES, 1):
        gid = f"G{k}"
        types.append({"id": gid, "dim": "line", "group": "symmetry", "name": "", "sig": None, "where": name,
                      "locus": "48 vertices (no fewer: not a transition by itself)", "bounds": []})
        lines.append({"id": gid, "pts": [[round(float(x), 5) for x in chart(np.array(q, float))] for q in (a, b)],
                      "hover": f"{gid} · {name}"})
    n_sym_points = len([t for t in types if t["dim"] == "point"])
    # lines and points where walls meet (atlas_model.py), one row per shape found there, as for the walls
    if os.path.exists(f"lines_points_{D.KEY}.json"):
        lp = json.load(open(f"lines_points_{D.KEY}.json"))
        wall_id = {f: t["id"] for t in types for f in t.get("surfaces", [])}
        sym_sigs = {t.get("sig") for t in types if t["dim"] in ("line", "point")}
        groups = {}
        on_ring = lambda B, tol: min(seg_dist(np.atleast_2d(B), a_, b_).max() if tol is None else
                                     np.median(seg_dist(np.atleast_2d(B), a_, b_)) for a_, b_ in ring_segs)
        for l in lp.get("lines", []):
            sg = l.get("on", "")
            if on_ring(np.array(l["points"]), "median") < 1e-5:
                continue                       # (a ring's piece: listed from the exact scan above)
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
            if on_ring(np.array([p["beta"]]), None) < 1e-3:
                continue                       # (on or by a ring: its points come from the exact scan; one found a hair off
                                               # the ring carries a near-degenerate copy of the ring's shape)
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
                # (a line whose shape is a regular polytope's is regular only at its special spots: marked on it)
                if t["dim"] == "line" and v == "24-cell":
                    t["name"] = "24-cell (regular at M₀₁ = M₁₂ and at M₀₂)"
                    for nm, b in (("M₀₁ (the midpoint of VW₁)", [0.5, 0.5, 0, 0]), ("M₁₂ (the midpoint of W₁W₂)", [0, 0.5, 0.5, 0]),
                                  ("M₀₂ (the midpoint of VW₂)", [0.5, 0, 0.5, 0])):
                        points.append({"id": t["id"], "q": [round(float(x), 5) for x in chart(b)], "beta": beta_text(b),
                                       "hover": f"{t['id']} · the regular 24-cell, at {nm}"})
    json.dump({"types": types, "samples": samples, "lines": lines, "points": points}, open(f"atlas_{D.KEY}.json", "w"), indent=1)
    from collections import Counter
    print(f"atlas_{D.KEY}.json: {dict(Counter(t['dim'] for t in types))}; {len(points)} points")
    for t in types[:6]:
        print("  ", t["id"], t.get("where"), t.get("separates"))


if __name__ == "__main__":
    main()
