"""The shapes in each mirror, face, plane, surface and line of the half-cell (surface_sets.json).

The domain viewer's "Select the shapes in a mirror, face or surface" list: each entry a locus in the
half-cell and every named shape lying in it, read from the 3-sphere map's own data:

* walls: those with a piece in it (domain_pieces.json: the sheet the piece lies on);
* lines: those with a stretch in it, in the database's own place or as a copy on the domain's
  boundary (the curves domain_pieces.py builds from: line pieces and their copies);
* points: those on it (atlas.json: each named point's place).

Each locus is taken whole on the 3-sphere: a shape is in it when a copy of it lies on it anywhere,
that is, when the shape (in the half-cell) lies on one of the locus's images under the group; and
loci that are images of each other are one entry (a mirror that is another plane's image is that
plane). Entries: every H4 mirror and face of the half-cell (the splitting mirror beta_3 = beta_4 its
top face), the lines where two of these meet holding two named shapes or more, the line through
(1,1,1,0) and (2+phi, 2+phi, 1, 1), and every other plane or surface (a surface with its folds and
copies, one family) holding at least two walls.

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
    from .frame import group
    G = group()
    Ms = np.stack([T @ g @ Ti for g in G])           # beta (domain) -> beta in copy g's frame
    # the curves the pieces were built from: database line stretches and their copies on the boundary
    from . import domain_pieces as dp
    import __main__
    for k_, v_ in vars(dp).items():              # (the cache is pickled from the module run as __main__)
        if not k_.startswith("__") and not hasattr(__main__, k_):
            setattr(__main__, k_, v_)
    C = pickle.loads((root / ".cache" / "domain_curves.pkl").read_bytes())
    D = dp.Data()
    curves = [c for c in C if c["kind"] == "line"] + [c for c in dp.boundary_images(D, C) if c["kind"] == "line"]
    # (each curve tested at a few of its points)
    cpts = [(c["name"], c["pts"][np.linspace(0, len(c["pts"]) - 1, min(7, len(c["pts"]))).astype(int)]) for c in curves]
    walls_of = defaultdict(set)                  # sheet -> walls with a piece on it
    for f in doc["faces"]:
        walls_of[f["sheet"]].add(f["wall"])
    named = {k for k, v in json.loads((root.parent / "source" / "working_data.json").read_text())["shapes"].items()
             if v["dim"] == "point"}
    points = []
    for p in atlas["points"]:
        if p.get("atlas_point") in named:
            b = np.asarray(p["R4"], float) @ Ti
            points.append((p["atlas_point"], b / b.sum()))
    order = lambda k: (re.match(r"[A-Z]+", k).group(0), int(re.search(r"\d+", k).group(0)), k)
    out = []

    def add(name, kind, walls, lines, pts):
        ids = sorted(set(walls) | set(lines) | set(pts), key=order)
        if ids:
            out.append({"name": name, "kind": kind, "ids": ids, "walls": sorted(set(walls), key=order),
                        "lines": sorted(set(lines), key=order), "points": sorted(set(pts), key=order)})

    # ---- planes: the half-cell's mirrors and faces and the library's planes, one entry per plane of
    # the 3-sphere up to the group (a mirror that is an image of another plane is that plane)
    cand = []                                    # (name, rank, normal in beta, sheets on it)
    for i, j in itertools.combinations(range(4), 2):
        n = np.eye(4)[i] - np.eye(4)[j]
        if (i, j) == (2, 3):
            cand.append(("the splitting mirror β3 = β4 (the top face of the half tetrahedron)", 0, n, ["face:b3=b4"]))
        else:
            cand.append((f"the mirror β{i + 1} = β{j + 1}", 1, n, []))
    cand.append(("the face β1 = 0", 2, np.eye(4)[0], ["face:b1=0"]))
    cand.append(("the face β2 = 0", 2, np.eye(4)[1], ["face:b2=0"]))
    cand.append(("the face β4 = 0 (the bottom face)", 2, np.eye(4)[3], ["face:b4=0"]))
    for k, d in enumerate(lib):
        if d["kind"] == "plane":
            cand.append(("the plane " + _plane_text(d["exact"]), 3, np.asarray(d["normal"], float), [str(k)]))
    n4 = lambda n: (lambda v: v / np.linalg.norm(v))(Ti @ n)        # (the plane's normal on R^4)
    orbits = []                                  # [members, images of the normal in beta]
    for c in cand:
        v = n4(c[2])
        for o in orbits:
            if np.abs(np.abs(o[2] @ v) - 1).min() < 1e-9:
                o[0].append(c)
                break
        else:
            imgs = np.einsum("gij,j->gi", G, v)
            orbits.append([[c], None, imgs])
    for members, _, imgs in orbits:
        nb = imgs @ T.T                          # (beta . (T n4') = X . n4')
        nb = nb / np.linalg.norm(nb, axis=1, keepdims=True)
        nb = np.unique(np.round(np.where(nb[:, :1] < 0, -nb, nb), 9), axis=0)
        sheets = {sh for c in members for sh in c[3]}
        # (the library's planes on any image of this one)
        for k, d in enumerate(lib):
            if d["kind"] == "plane" and np.abs(np.abs(nb @ (np.asarray(d["normal"]) / np.linalg.norm(d["normal"]))) - 1).min() < 1e-9:
                sheets.add(str(k))
        for fn, fv in (("face:b1=0", np.eye(4)[0]), ("face:b2=0", np.eye(4)[1]), ("face:b4=0", np.eye(4)[3]),
                       ("face:b3=b4", np.eye(4)[2] - np.eye(4)[3])):
            if np.abs(np.abs(nb @ (fv / np.linalg.norm(fv))) - 1).min() < 1e-9:
                sheets.add(fn)
        walls = set().union(*[walls_of.get(sh, set()) for sh in sheets])
        lines = {nm for nm, P in cpts if (np.abs(P @ nb.T) < 1e-7).all(0).any()}
        pts = {nm for nm, b in points if np.abs(b @ nb.T).min() < 1e-6}
        # (named by its simplest member: the splitting mirror, a mirror, a face, else the plane of the
        # library holding the most walls, the shortest written; the others it is the same great sphere
        # as listed when they are mirrors, faces or planes holding walls, with the count of all)
        own = lambda c: len(set().union(*[walls_of.get(sh, set()) for sh in c[3]]))
        members.sort(key=lambda c: (c[1], -own(c), len(c[0])))
        rank = members[0][1]
        if rank >= 3 and len(walls) < 2:
            continue                             # (a plane of the library holding fewer than two walls)
        others = []
        for c in members[1:]:
            short = c[0].replace("the ", "", 1)
            if c[1] >= 3 and (_mirror_name(c[2]) is not None or own(c) == 0):
                continue                         # (a library plane that is a mirror, or holds no wall)
            if short not in others and c[0] != members[0][0]:
                others.append(short)
        n_planes = len(nb)
        name = members[0][0] + (f" — the same great sphere, carried by the group, as the {'; the '.join(others)} "
                                f"({n_planes} planes through the half-cell's copies in all)" if others else "")
        add(name, "mirror" if rank <= 1 else "face" if rank == 2 else "plane", walls, lines, pts)

    # ---- curved surfaces: each family (a surface with its folds and copies), whole on the 3-sphere:
    # a line or point on any image of one of its surfaces is on it
    fam = defaultdict(list)
    for k, d in enumerate(lib):
        if d["kind"] != "plane":
            fam[d["family"]].append(k)
    Pinv = np.linalg.inv(Ms)
    for f_, idx in fam.items():
        walls = set().union(*[walls_of.get(str(k), set()) for k in idx])
        if len(walls) < 2:
            continue
        s0 = D.lib[idx[0]]
        # (its images: s(beta M_g) = 0)
        def on_any(X, tol):
            X = np.atleast_2d(X)
            Y = np.einsum("nj,gjk->gnk", X, Ms)
            Y = Y / Y.sum(-1, keepdims=True)
            vals = np.abs(np.stack([s0.f(y) for y in Y]))
            gr = np.stack([np.linalg.norm(s0.grad(y) - s0.grad(y).mean(1, keepdims=True), axis=1) for y in Y])
            return ((vals / np.maximum(gr, 1e-300)) < tol).all(1).any()
        lines = {nm for nm, P in cpts if on_any(P, 1e-6)}
        pts = {nm for nm, b in points if on_any(b, 1e-6)}
        bases = sorted({lib[k]["base"].split(":")[0] for k in idx})
        folds = any(":fold" in lib[k]["base"] for k in idx)
        add(f"the surface found for {bases[0]}" + (" and its fold" if folds else ""), "surface", walls, lines, pts)

    # ---- lines: where two of the half-cell's mirrors and faces meet, and the line through (1,1,1,0)
    # and (2+phi, 2+phi, 1, 1); each whole on the 3-sphere (its images through the half-cell), one
    # entry per line up to the group; a line holding one shape only is left out
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
    seen = []                                    # [name, names of its images, its 2-plane of R^4]
    for name, M in loci:
        B = np.linalg.svd(M)[2][2:] @ T            # (its 2-plane of R^4)
        B = np.linalg.qr(B.T)[0].T
        GB = np.einsum("gij,kj->gki", G, B)        # (its images)
        for e in seen:
            # (some symmetry carries it onto that line: each image row lies in that line's plane)
            if np.linalg.norm(GB - GB @ e[2].T @ e[2], axis=(1, 2)).min() < 1e-6:
                if name != e[0] and name not in e[1]:
                    e[1].append(name)
                break
        else:
            seen.append([name, [], B])
    for name, others, B in seen:
        imgs = np.einsum("gij,kj->gki", G, B)       # (its images, as 2-planes of R^4)
        def on_line(X, tol):
            Xr = np.atleast_2d(X) @ T
            Xr = Xr / np.linalg.norm(Xr, axis=1, keepdims=True)
            res = np.linalg.norm(Xr[None] - np.einsum("gnk,gkj->gnj", np.einsum("nj,gkj->gnk", Xr, imgs), imgs), axis=2)
            return (res < tol).all(1).any()
        pts = {nm for nm, b in points if on_line(b, 1e-6)}
        lns = {nm for nm, P in cpts if on_line(P, 1e-6)}
        if len(pts) + len(lns) < 2:
            continue
        add(name + (f" (the same line as {'; '.join(o.replace('the line ', '') for o in others)})" if others else ""),
            "line", [], lns, pts)
    return {"_about": __doc__, "sets": out}

def samples() -> dict:
    """What the domain viewer needs to list, for each view, every individual sphere, surface and
    line passing through it with the shapes lying on it there (selection_samples.json):
    * loci in the half-cell's betas: its mirrors and faces and the library's planes (a normal), the
      library's curved surfaces (their polynomial), and the lines where two mirrors or faces meet
      and the line through (1,1,1,0) and (2+phi, 2+phi, 1, 1) (two equations);
    * the shapes, as points in the half-cell: each wall piece at points spread over it, each named
      line's stretches (their own place and their copies on the domain's boundary) at points along
      them, and each named point."""
    root = data_dir() / "sphere_map"
    lib = json.loads((root / "surface_library.json").read_text())["surfaces"]
    doc = json.loads((root / "domain_pieces.json").read_text())
    z = np.load(root / "domain_pieces.npz")
    atlas = json.loads((root / "atlas.json").read_text())
    Ti = np.linalg.inv(cell_vertices())
    from . import domain_pieces as dp
    import __main__
    for k_, v_ in vars(dp).items():
        if not k_.startswith("__") and not hasattr(__main__, k_):
            setattr(__main__, k_, v_)
    C = pickle.loads((root / ".cache" / "domain_curves.pkl").read_bytes())
    D = dp.Data()
    curves = [c for c in C if c["kind"] == "line"] + [c for c in dp.boundary_images(D, C) if c["kind"] == "line"]
    planes = []
    for i, j in itertools.combinations(range(4), 2):
        n = np.eye(4)[i] - np.eye(4)[j]
        planes.append({"name": "the splitting mirror β3 = β4 (the top face of the half tetrahedron)" if (i, j) == (2, 3)
                       else f"the mirror β{i + 1} = β{j + 1}", "rank": 0 if (i, j) == (2, 3) else 1, "normal": n.tolist()})
    for nm, k in (("the face β1 = 0", 0), ("the face β2 = 0", 1), ("the face β4 = 0 (the bottom face)", 3)):
        planes.append({"name": nm, "rank": 2, "normal": np.eye(4)[k].tolist()})
    for k, d in enumerate(lib):
        if d["kind"] == "plane" and _mirror_name(d["normal"]) is None:
            planes.append({"name": "the plane " + _plane_text(d["exact"]), "rank": 3, "normal": d["normal"]})
    surfaces = [{"index": k, "name": d["name"], "family": d["family"], "base": d["base"], "monomials": d["monomials"],
                 "coefficients": d["coefficients"], "A": d["A"]} for k, d in enumerate(lib) if d["kind"] != "plane"]
    sheet_surface = {str(sf["index"]): i for i, sf in enumerate(surfaces)}
    lines = []
    eqs = {}
    for i, j in itertools.combinations(range(4), 2):
        eqs[(("eq", i, j),)] = np.eye(4)[i] - np.eye(4)[j]
    for i in range(4):
        eqs[(("zero", i),)] = np.eye(4)[i]
    for (ka, a), (kb, b) in itertools.combinations(eqs.items(), 2):
        M = np.array([a, b])
        if np.linalg.matrix_rank(M) == 2:
            lines.append({"name": _chain(ka + kb), "M": M.tolist()})
    P1, P2 = np.array([1, 1, 1, 0.0]), np.array([2 + PHI, 2 + PHI, 1, 1])
    lines.append({"name": "the line through β ∝ (1, 1, 1, 0) and (2+φ, 2+φ, 1, 1)",
                  "M": np.linalg.svd(np.array([P1, P2]))[2][2:].tolist()})
    walls = []
    rng = np.random.default_rng(0)
    for f in doc["faces"]:
        # (its mesh's own points: on the surface, as its triangles' middles are not where it curves)
        X = z["verts"][f["verts"][0]:f["verts"][1]]
        pick = X[rng.choice(len(X), min(10, len(X)), replace=False)]
        walls.append({"name": f["wall"], "sheet": f["sheet"], "pts": (pick / pick.sum(1, keepdims=True)).round(9).tolist()})
    line_pts = [{"name": c["name"], "pts": c["pts"][np.linspace(0, len(c["pts"]) - 1, min(9, len(c["pts"]))).astype(int)].round(10).tolist()}
                for c in curves]
    named = {k for k, v in json.loads((root.parent / "source" / "working_data.json").read_text())["shapes"].items()
             if v["dim"] == "point"}
    points = []
    for p in atlas["points"]:
        if p.get("atlas_point") in named:
            b = np.asarray(p["R4"], float) @ Ti
            points.append({"name": p["atlas_point"], "pt": (b / b.sum()).round(12).tolist()})
    return {"_about": samples.__doc__, "planes": planes, "surfaces": surfaces, "sheet_surface": sheet_surface, "lines": lines,
            "walls": walls, "line_stretches": line_pts, "points": points}


def run() -> str:
    (data_dir() / "sphere_map" / "selection_samples.json").write_text(json.dumps(samples(), ensure_ascii=False))
    data = build()
    path = data_dir() / "sphere_map" / "surface_sets.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    for s in data["sets"]:
        print(f"{s['name']}: {', '.join(s['ids'])}")
    return str(path)


if __name__ == "__main__":
    print(run())
