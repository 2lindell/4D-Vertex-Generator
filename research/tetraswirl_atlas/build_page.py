"""The swirl atlas page for one group, laid out like the Symmetry Domain page (atlas_template.html shares its styles,
key and controls), from whatever has been gathered so far:

  * the group and its domain (domain.py): corners, edges and faces, each face named by what it is;
  * the samples (survey_<group>.json, and dense_<group>*.json when there), each coloured by its polychoron; every
    signature keeps the id it was first given (ids_<group>.json), regions R1, R2, ... by how much of the domain
    they fill, other vertex counts Z1, ...;
  * the traced data, once the later stages write it (atlas_<group>.json, every key optional):
        {"types": [{"id", "name", "dim": "region|wall|line|point", "where", "locus", "bounds": [ids],
                    "separates": [ids], "sig", "seed"}],      (overrides or adds rows of the key)
         "walls":  [{"id", "verts": [[x, y, z]], "faces": [[i, j, k]], "hover"}],
         "lines":  [{"id", "pts": [[x, y, z]], "hover", "copy"}],
         "points": [{"id", "q": [x, y, z], "hover"}],
         "faces":  {"<domain face index>": [ids]}}            (the shapes lying in each face of the domain)
    all in the domain's chart coordinates.

    SWIRL_GROUP=oct python build_page.py        (writes ../../assets/tetraswirl_oct_domain.html)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

import domain as D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "swirlprism_atlas"))
from place_colors import DARK, LIGHT, _oklch_to_hex  # noqa: E402  (the Symmetry Domain page's colour rule)

KEY = D.KEY
GROUP = {
    "oct": {"title": "Octahedral Swirl Symmetry Domain",
            "short": "octahedral swirl (index 6)"},
    "tet": {"title": "Tetrahedral Swirl Symmetry Domain",
            "short": "tetrahedral swirl (index 3)"},
}[KEY]
ORDER = len(D.ELEMENTS)
DIM_N = {"region": 3, "wall": 2, "line": 1, "point": 0}
GROUPS = [
    ("regions", "Regions", "region", "none found yet"),
    ("walls", "Walls", "wall", "none traced yet: the wall stage (walls.py, build2.py) bisects between neighbouring samples "
                               "of different regions and names each wall by the five orbit points that become coplanar there"),
    ("lines", "Lines", "line", "none traced yet: lines are where two walls meet"),
    ("points", "Points", "point", "none traced yet: points are where lines meet"),
]


def domain_geometry(tol=1e-9):
    """Corners (merged), edges, and faces (ordered loops) of the domain, each face named."""
    V = D.domain_vertices()
    keep = []
    for v in V:
        if not any(np.linalg.norm(v - k) < 1e-7 for k in keep):
            keep.append(v)
    V = np.array(keep)
    # (the halfspaces, with what each one is)
    rows, names, kinds = zip(*D.domain_faces())
    faces, scale = [], np.abs(V).max()
    for r, name, kind in zip(rows, names, kinds):
        on = np.flatnonzero(np.abs(V @ r[:3] + r[3]) / np.linalg.norm(r[:3]) < 1e-7 * max(scale, 1))
        if len(on) < 3:
            continue
        c = V[on].mean(0)
        n = r[:3] / np.linalg.norm(r[:3])
        e1 = V[on[0]] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ang = np.arctan2((V[on] - c) @ e2, (V[on] - c) @ e1)
        faces.append({"loop": [int(i) for i in on[np.argsort(ang)]], "name": name, "kind": kind, "ids": []})
    edges = set()
    for f in faces:
        L = f["loop"]
        for a, b in zip(L, L[1:] + L[:1]):
            edges.add((min(a, b), max(a, b)))
    return V, sorted(edges), faces


def parse_sig(sig):
    """'48: cells A | faces B | edges C' -> (vertices, cells, faces, edges, valence text)."""
    m = re.match(r"(\d+): cells (.*) \| faces (.*) \| edges (.*)", sig)
    if not m:
        return None
    n, cells, faces, edges = m.groups()
    nice = lambda s: s.replace("x", "×").replace("+", " + ")
    val = Counter()
    for part in edges.split("+"):
        k, v = part.split("x")
        val[int(v)] += int(k)
    return int(n), nice(cells), nice(faces), nice(edges), ", ".join(f"{v}: {k}" for v, k in sorted(val.items()))


def seed_text(u):
    s = D.seed_from_chart(np.asarray(u, float))
    return ", ".join(f"{x:.6f}" for x in s)


def beta_text(u):
    """β in the 16-cell's tetrahedron, scaled so the weights sum to 1 (cell16 domain)."""
    if D.DOMAIN != "cell16":
        return ""
    b = D.beta_from_seed(D.seed_from_chart(np.asarray(u, float)))
    return "(" + ", ".join(f"{x:.4f}" for x in b) + ")"


def main():
    V, E, F = domain_geometry()
    # samples
    pts, sigs = [], []
    for path in [f"survey_{KEY}.json"] + sorted(glob.glob(f"dense_{KEY}*.json")):
        if os.path.exists(path):
            s = json.load(open(path))
            pts += s["points"]
            sigs += s["sigs"]
    pts = np.array(pts, float).reshape(-1, 3)
    inside = D.in_domain(pts, 1e-9) if len(pts) else np.zeros(0, bool)
    pts, sigs = pts[inside], [s for s, k in zip(sigs, inside) if k]
    count = Counter(s for s in sigs if s != "none")
    # ids: kept once given
    id_path = f"ids_{KEY}.json"
    ids = json.load(open(id_path)) if os.path.exists(id_path) else {}
    used = Counter(i[0] for i in ids.values())
    for sig, _ in count.most_common():
        if sig not in ids:
            p = parse_sig(sig)
            letter = "R" if p and p[0] == ORDER else "Z"
            used[letter] += 1
            ids[sig] = f"{letter}{used[letter]}"
    json.dump(ids, open(id_path, "w"), indent=1)
    types, samples = [], {}
    for sig, n in sorted(count.items(), key=lambda kv: (ids[kv[0]][0], int(ids[kv[0]][1:]))):
        tid = ids[sig]
        P = pts[[s == sig for s in sigs]]
        rep = P[np.argmin(np.linalg.norm(P - P.mean(0), axis=1))]
        p = parse_sig(sig)
        types.append({"id": tid, "name": "", "dim": "region", "group": "regions", "vertices": p[0] if p else None,
                      "cells": p[1] if p else None, "faces": p[2] if p else None, "edges": p[3] if p else None,
                      "valence": p[4] if p else None, "seed": seed_text(rep), "samples": int(n), "sig": sig,
                      "where": f"region ({100 * n / len(pts):.1f}% of the samples)", "bounds": [], "separates": []})
        samples[tid] = [[round(float(x), 5) for x in q] + [seed_text(q), beta_text(q)] for q in P]
        if D.DOMAIN == "cell16":
            types[-1]["beta"] = beta_text(rep)
    # traced data, when there
    traced = json.load(open(f"atlas_{KEY}.json")) if os.path.exists(f"atlas_{KEY}.json") else {}
    by = {t["id"]: t for t in types}
    for t in traced.get("types", []):
        if t["id"] in by:
            by[t["id"]].update(t)
        else:
            t = {"name": "", "bounds": [], "separates": [], **t}
            t.setdefault("group", {"region": "regions", "wall": "walls", "line": "lines", "point": "points"}[t["dim"]])
            if t.get("sig") and parse_sig(t["sig"]):
                p = parse_sig(t["sig"])
                t.update(vertices=p[0], cells=p[1], faces=p[2], edges=p[3], valence=p[4])
            types.append(t)
            by[t["id"]] = t
    for k, ids_ in traced.get("faces", {}).items():
        F[int(k)]["ids"] = ids_
    # colours: lightness by dimension (as the Symmetry Domain page), hues spread round the circle within each dimension
    light, dark = [], []
    for dim in DIM_N:
        members = [t for t in types if t["dim"] == dim]
        for k, t in enumerate(members):
            h = 25 + 360 * k / max(len(members), 1)
            t["color"] = f"--ty-{t['id']}"
            t["symbol"] = "circle" if (t.get("vertices") or ORDER) == ORDER else "diamond"
            light.append(f"--ty-{t['id']}: {_oklch_to_hex(*LIGHT[DIM_N[dim]], h)};")
            dark.append(f"--ty-{t['id']}: {_oklch_to_hex(*DARK[DIM_N[dim]], h)};")
    type_css = (f":root {{ {' '.join(light)} }}\n@media (prefers-color-scheme: dark) {{ :root:not([data-theme=\"light\"]) "
                f"{{ {' '.join(dark)} }} }}\n:root[data-theme=\"dark\"] {{ {' '.join(dark)} }}")
    # views: the domain, and the whole Voronoi cell (its images under the anchor's stabiliser)
    R = D.stabiliser_in_chart()
    ident = np.c_[np.eye(3), np.zeros(3)]
    maps_cell = [np.c_[np.linalg.matrix_power(R, k), np.zeros(3)] for k in range(D.M)]

    def bounds(maps):
        Q = np.vstack([V @ M[:, :3].T + M[:, 3] for M in maps])
        return [Q.min(0).round(5).tolist(), Q.max(0).round(5).tolist()]
    whole = "16-cell tetrahedron" if D.DOMAIN == "cell16" else "Voronoi cell"
    views = [{"id": "domain", "name": f"third of a 16-cell tetrahedron" if D.DOMAIN == "cell16" else f"fundamental domain (1/{D.M} of the cell)",
              "maps": [ident.round(12).tolist()], "bounds": bounds([ident])}]
    if D.DOMAIN == "cell16":                # (the wedge faced from its open side, halfway between W1 and W2)
        a_ = np.pi / D.M
        views[0]["camera"] = {"eye": {"x": 1.7 * np.cos(a_), "y": 1.7 * np.sin(a_), "z": 0.45},
                              "center": {"x": 0, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}}
    if D.M > 1:
        views.append({"id": "cell", "name": f"whole {whole} ({D.M} domains)", "maps": [M.round(12).tolist() for M in maps_cell],
                      "bounds": bounds(maps_cell)})
    zs = np.linspace(V[:, 2].min() - 1, V[:, 2].max() + 1, 4001)
    on = zs[D.in_domain(np.c_[np.zeros_like(zs), np.zeros_like(zs), zs], 1e-12)]
    axis = {"pts": [[0, 0, float(on.min())], [0, 0, float(on.max())]],
            "name": f"axis of the order-{D.M} rotation fixing the anchor (seeds on it have more symmetry)"} if len(on) else None
    n_none = sum(1 for s in sigs if s == "none")
    notes = [
        {"title": "The domain", "items": [
            f"The group is the {GROUP['short']} group <code>{D.NAME}</code>: {ORDER} rotations of the 3-sphere. A seed's "
            f"{ORDER} images are the vertices of an isogonal polychoron, and two seeds give the same polychoron when one "
            f"is an image of the other, so one fundamental domain holds every shape once.",
            *([f"The group maps two 16-cells to themselves and moves the 16 tetrahedral cells of each into one another; a "
               f"cell is fixed only by an order-3 rotation about one of its vertices, V, through the centre of the opposite "
               f"face W₁W₂W₃. So a third of the cell is a fundamental domain, as half of a 600-cell's tetrahedron is for "
               f"the swirlprism group: the tetrahedron with corners V, W₁, W₂ and the face centre F, where β₃ is the least "
               f"of β₁, β₂, β₃. Its cut faces β₁ = β₃ and β₂ = β₃ are mirrors of the 16-cell; the rotation carries the "
               f"other two thirds onto it.",
               "It is drawn in the gnomonic chart at the cell's centre, so great circles are straight lines and great "
               "spheres are planes, with V at the top. Points are written in barycentric weights β = (β_V, β₁, β₂, β₃) "
               "of the cell, summing to 1."] if D.DOMAIN == "cell16" else
              [f"The domain is drawn in the gnomonic chart at an anchor on the fixed circle of an order-{D.M} rotation, so "
               f"great circles are straight lines and great spheres are planes. It is the part of the anchor's Voronoi "
               f"cell (the points nearer the anchor than any of its images) between two cuts through the rotation "
               f"axis, 1/{D.M} of the cell: {len(V)} corners and {len(F)} faces."]),
            "Still to find: the extra symmetries that map the group to itself (its normalizer), which may halve the "
            "domain again, as the half-turn Q does for the swirlprism group.",
        ]},
        {"title": "Views", "items": [v["name"][0].upper() + v["name"][1:] + "." for v in views]
         + ["Each view keeps its own camera, so ticking boxes never moves the picture."]},
        {"title": "Names and markers", "items": [
            f"Regions are numbered R1, R2, … by how much of the domain they fill when first found, and keep their number. "
            f"Polychora with fewer than {ORDER} vertices (seeds with extra symmetry) are Z1, Z2, …",
            "Lightness shows dimension, as on the Symmetry Domain page: regions lightest, walls darker, lines darker "
            "still, points darkest.",
            "Counts are the classes of cells, faces and edges under the group; two shapes with the same counts are not "
            "yet proven the same, so the key may still split.",
        ]},
        {"title": "Status", "status": True, "items": [
            f"<b>Samples:</b> {len(pts)} seeds classified ({len(count)} kinds of polychoron"
            + (f"; {n_none} gave no polytope" if n_none else "") + ").",
            f"<b>Walls, lines and points:</b> {sum(t['dim'] == 'wall' for t in types)}, {sum(t['dim'] == 'line' for t in types)} "
            f"and {sum(t['dim'] == 'point' for t in types)} traced.",
            "<b>Next:</b> the normalizer and final domain; a denser survey; walls bisected and named by their coplanar "
            "orbit points; regions closed by their walls.",
        ]},
    ]
    data = {
        "group": {"key": KEY, "name": D.NAME, "order": ORDER, "title": GROUP["title"],
                  "eyebrow": f"{D.NAME} · {ORDER} rotations",
                  "lede": f"A map of every isogonal polychoron the {GROUP['short']} group produces from a single seed. Each "
                          f"point of one fundamental domain is a seed; it is coloured by the polychoron its {ORDER} images "
                          f"make. Regions, walls, lines and points are listed in the key as they are found."},
        "domain": {"verts": V.round(6).tolist(), "edges": [list(e) for e in E], "faces": F},
        "views": views, "axis": axis,
        "groups": [{"id": g, "title": title, "empty": empty} for g, title, _, empty in GROUPS],
        "types": types, "samples": samples, "totalSamples": len(pts),
        "walls": traced.get("walls", []), "lines": traced.get("lines", []), "points": traced.get("points", []),
        "notes": notes,
    }
    template = open("atlas_template.html").read()
    out = os.path.join("..", "..", "assets", f"tetraswirl_{KEY}_domain.html")
    page = (template.replace("__TITLE__", GROUP["title"]).replace("__TYPE_CSS__", type_css)
            .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    open(out, "w").write(page)
    print(f"{out}: {len(types)} types, {len(pts)} samples, {len(V)} corners, {len(F)} faces")


if __name__ == "__main__":
    main()
