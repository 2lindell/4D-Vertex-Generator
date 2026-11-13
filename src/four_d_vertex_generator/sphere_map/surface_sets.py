"""The shapes in each mirror, face, plane, surface and line of the half-cell (surface_sets.json).

The domain viewer's "Select the shapes in a mirror, face or surface" list: each entry a locus in the
half-cell and every named shape lying in it, read from the 3-sphere map's own data:

* walls: those with a piece in it (domain_pieces.json: the sheet the piece lies on);
* lines: those with a stretch in it, in the database's own place or as a copy on the domain's
  boundary (the curves domain_pieces.py builds from: line pieces and their copies);
* points: those on it (atlas.json: each named point's place).

Entries: every H4 mirror through the half-cell (the splitting mirror beta_3 = beta_4 its top face),
its faces, the lines where two of these meet that carry a named shape, and every other plane or
surface (a surface with its folds and copies, one family) holding at least two walls.

    python -m four_d_vertex_generator.sphere_map.surface_sets
"""
from __future__ import annotations

import itertools
import json
import pickle
import re
from collections import defaultdict

import numpy as np

from .frame import cell_vertices, data_dir

PHI = (1 + 5 ** 0.5) / 2


def _golden(c):
    """An exact coefficient [a, b, c] = (a + b sqrt5) / c as p + q phi (sqrt5 = 2 phi - 1)."""
    a, b, d = c
    p, q = (a - b) / d, 2 * b / d
    return p, q


def _num(p, q):
    """p + q phi, written."""
    def fr(x):
        return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"
    if abs(q) < 1e-12:
        return fr(p)
    qs = "φ" if abs(q - 1) < 1e-12 else "-φ" if abs(q + 1) < 1e-12 else f"{fr(q)}φ"
    if abs(p) < 1e-12:
        return qs
    return f"{fr(p)}{'+' if q > 0 else '-'}{qs.lstrip('-')}"


def _gdiv(x, y):
    """(p + q phi) / (r + s phi), exactly: 1 / (r + s phi) = (r + s - s phi) / (r^2 + r s - s^2)."""
    (p, q), (r, s_) = x, y
    n = r * r + r * s_ - s_ * s_
    a, b = (r + s_) / n, -s_ / n                 # (1 / (r + s phi) = a + b phi)
    return p * a + q * b, p * b + q * a + q * b   # ((p + q phi)(a + b phi), phi^2 = phi + 1)


def _plane_text(exact):
    """n . beta = 0 written with the terms of each sign on their own side, the first beta's
    coefficient 1: beta1 = (1+phi)*beta2."""
    gs = [_golden(c) for c in exact]
    first = next(g for g in gs if abs(g[0] + g[1] * PHI) > 1e-12)
    terms = [(_gdiv(g, first), i) for i, g in enumerate(gs)]
    lhs, rhs = [], []
    for (p, q), i in terms:
        v = p + q * PHI
        if abs(v) < 1e-12:
            continue
        side = lhs if v > 0 else rhs
        p, q = (p, q) if v > 0 else (-p, -q)
        k = _num(p, q)
        side.append(f"β{i + 1}" if k == "1" else f"({k})·β{i + 1}" if ("+" in k or "-" in k[1:]) else f"{k}·β{i + 1}")
    if not rhs:
        return " + ".join(lhs) + " = 0"
    return " + ".join(lhs) + " = " + " + ".join(rhs)


def _mirror_name(n):
    """beta_i = beta_j for an H4 mirror through the half-cell's corners, else None."""
    nz = [i for i in range(4) if abs(n[i]) > 1e-9]
    if len(nz) == 2 and abs(n[nz[0]] + n[nz[1]]) < 1e-9:
        return f"β{nz[0] + 1} = β{nz[1] + 1}"
    return None


def _chain(conds):
    """Two conditions on beta written as one line: beta1 = beta2 = beta3, beta1 = beta2 = 0, or
    beta1 = beta2 and beta3 = 0."""
    groups = []                                    # (indices equal to each other, equal to 0?)
    for c in conds:
        idx, zero = ({c[1], c[2]}, False) if c[0] == "eq" else ({c[1]}, True)
        for g in groups:
            if g[0] & idx or (zero and g[1]):        # (two betas both 0 are equal: one chain)
                g[0] |= idx
                g[1] = g[1] or zero
                break
        else:
            groups.append([set(idx), zero])
    parts = [" = ".join(f"β{i + 1}" for i in sorted(g[0])) + (" = 0" if g[1] else "") for g in groups]
    return "the line " + " and ".join(parts)


def build() -> dict:
    root = data_dir() / "sphere_map"
    lib = json.loads((root / "surface_library.json").read_text())["surfaces"]
    doc = json.loads((root / "domain_pieces.json").read_text())
    atlas = json.loads((root / "atlas.json").read_text())
    T = cell_vertices()
    Ti = np.linalg.inv(T)
    # the curves the pieces were built from: database line stretches and their copies on the boundary
    from . import domain_pieces as dp
    import __main__
    for k_, v_ in vars(dp).items():              # (the cache is pickled from the module run as __main__)
        if not k_.startswith("__") and not hasattr(__main__, k_):
            setattr(__main__, k_, v_)
    C = pickle.loads((root / ".cache" / "domain_curves.pkl").read_bytes())
    D = dp.Data()
    curves = [c for c in C if c["kind"] == "line"] + [c for c in dp.boundary_images(D, C) if c["kind"] == "line"]
    lines_of_lib = defaultdict(set)              # library surface -> lines with a stretch on it
    for c in curves:
        for i, s in enumerate(D.lib):
            if dp.dist_to(s, c["pts"]).max() < 1e-6:
                lines_of_lib[i].add(c["name"])
    faces = {"b1=0": 0, "b2=0": 1, "b4=0": 3, "b3=b4": 4}
    lines_of_face = defaultdict(set)
    for c in curves:
        for nm, k in faces.items():
            if np.abs(c["pts"] @ dp.FACE_N[k]).max() < 1e-9:
                lines_of_face[nm].add(c["name"])
    walls_of = defaultdict(set)                  # sheet -> walls with a piece on it
    for f in doc["faces"]:
        walls_of[f["sheet"]].add(f["wall"])
    # named points' places in the domain
    named = {k for k, v in json.loads((root.parent / "source" / "working_data.json").read_text())["shapes"].items()
             if v["dim"] == "point"}
    points = []
    for p in atlas["points"]:
        if p.get("atlas_point") in named:
            b = np.asarray(p["R4"], float) @ Ti
            points.append((p["atlas_point"], b / b.sum()))

    order = lambda k: (re.match(r"[A-Z]+", k).group(0), int(re.search(r"\d+", k).group(0)), k)
    out = []

    def add(name, kind, walls, lines, pts=()):
        ids = sorted(set(walls) | set(lines) | set(pts), key=order)
        if ids:
            out.append({"name": name, "kind": kind, "ids": ids, "walls": sorted(walls, key=order),
                        "lines": sorted(lines, key=order), "points": sorted(set(pts), key=order)})

    # the H4 mirrors through the half-cell; the splitting mirror is its top face
    for i, j in itertools.combinations(range(4), 2):
        n = np.zeros(4)
        n[i], n[j] = 1, -1
        idx = [k for k, d in enumerate(lib) if d["kind"] == "plane" and
               abs(abs(np.dot(d["normal"], n / np.linalg.norm(n))) - 1) < 1e-9]
        walls = set().union(*[walls_of.get(str(k), set()) for k in idx]) if idx else set()
        lines = set().union(*[lines_of_lib.get(k, set()) for k in idx]) if idx else set()
        if (i, j) == (2, 3):
            walls |= walls_of.get("face:b3=b4", set())
            lines |= lines_of_face["b3=b4"]
            add("the splitting mirror β3 = β4 (the top face of the half tetrahedron)", "mirror", walls, lines)
        else:
            add(f"the mirror β{i + 1} = β{j + 1}", "mirror", walls, lines)
    # the other faces
    add("the faces β1 = 0 and β2 = 0 (folds of each other)", "face",
        walls_of.get("face:b1=0", set()) | walls_of.get("face:b2=0", set()), lines_of_face["b1=0"] | lines_of_face["b2=0"])
    add("the face β4 = 0 (the bottom face)", "face", walls_of.get("face:b4=0", set()), lines_of_face["b4=0"])

    # lines where two of the half-cell's mirrors and faces meet, and the line through the corner
    # (1,1,1,0) and the point (2+phi, 2+phi, 1, 1): the named points and lines on each. Lines one
    # symmetry carries onto another are one entry (the others after "≅"); a line holding one shape
    # only (a corner's point, on every line through that corner) is left out
    from .frame import group
    G = group()
    eqs = {}
    for i, j in itertools.combinations(range(4), 2):
        eqs[(("eq", i, j),)] = np.eye(4)[i] - np.eye(4)[j]
    for i in range(4):
        eqs[(("zero", i),)] = np.eye(4)[i]
    loci = []
    for (ka, a), (kb, b) in itertools.combinations(eqs.items(), 2):
        M = np.array([a, b])
        if np.linalg.matrix_rank(M) == 2:
            loci.append((_chain(ka + kb), M))
    P1, P2 = np.array([1, 1, 1, 0.0]), np.array([2 + PHI, 2 + PHI, 1, 1])
    loci.append(("the line through β ∝ (1, 1, 1, 0) and (2+φ, 2+φ, 1, 1)",
                 np.linalg.svd(np.array([P1, P2]))[2][2:]))
    entries = {}
    for name, M in loci:
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        # (a point's stored place to 1e-6; a traced curve to 1e-7)
        on = lambda X, tol: (np.abs(np.atleast_2d(X) @ Mn.T) < tol).all(1)
        pts = sorted({nm for nm, b_ in points if on(b_, 1e-6)[0] and b_.min() > -1e-6 and b_[2] - b_[3] > -1e-6}, key=order)
        lns = sorted({c["name"] for c in curves if on(c["pts"], 1e-7).all()}, key=order)
        if len(pts) + len(lns) < 2:
            continue
        # the line on the sphere (R^4): the 2-plane its beta directions span
        B = np.linalg.svd(Mn)[2][2:] @ T
        B = np.linalg.qr(B.T)[0].T
        key = None
        for k_, e in entries.items():
            if (e["pts"], e["lns"]) != (pts, lns):
                continue
            # (one symmetry of the group carrying this line onto that one)
            Q = np.einsum("gij,kj->gki", G, B)
            res = np.linalg.norm(Q - Q @ e["B"].T @ e["B"], axis=(1, 2))
            if res.min() < 1e-6:
                key = k_
                break
        if key is None:
            entries[name] = {"pts": pts, "lns": lns, "B": B, "also": []}
        elif name != key and name.replace("the line ", "") not in entries[key]["also"]:
            entries[key]["also"].append(name.replace("the line ", ""))
    for name, e in entries.items():
        add(name + (f" (≅ {', '.join(e['also'])})" if e["also"] else ""), "line", [], e["lns"], e["pts"])

    # every other plane, and every surface (with its folds and copies), holding two walls or more
    fam = defaultdict(list)
    for k, d in enumerate(lib):
        if d["kind"] == "plane":
            if _mirror_name(d["normal"]) is None:
                fam[("plane", k)].append(k)
        else:
            fam[("surface", d["family"])].append(k)
    for (kind, key), idx in fam.items():
        walls = set().union(*[walls_of.get(str(k), set()) for k in idx])
        if len(walls) < 2:
            continue
        lines = set().union(*[lines_of_lib.get(k, set()) for k in idx])
        if kind == "plane":
            name = "the plane " + _plane_text(lib[idx[0]]["exact"])
        else:
            bases = sorted({lib[k]["base"].split(":")[0] for k in idx})
            folds = sorted({lib[k]["base"] for k in idx if ":fold" in lib[k]["base"]})
            # (the library names a surface after the wall it was found for)
            name = f"the surface found for {bases[0]}" + (" and its fold" if folds else "")
        add(name, kind, walls, lines)
    return {"_about": __doc__, "sets": out}


def run() -> str:
    data = build()
    path = data_dir() / "sphere_map" / "surface_sets.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for s in data["sets"]:
        print(f"{s['name']}: {', '.join(s['ids'])}")
    return str(path)


if __name__ == "__main__":
    print(run())
