"""The 3-sphere map of an A4 symmetry on the A4 chamber (a4_map_<group>.json), for the A4 domain viewer (build_page.py).

The domain is the tetrahedral chamber of the A4 Coxeter group [3,3,3]: seeds beta >= 0, beta_k the distance to the k-th
mirror (the generator's Coordinates mode takes the seed s = R^-1 beta, R the generator's simple roots,
fundamental_chamber_roots("a4")). It is drawn in the gnomonic chart of the 3-sphere at the chamber's centre, so the
mirrors and every plane of beta are planes, and great circles are straight lines.

  * pennic [3,3,3] (order 120): a reflection group, so the polychoron is the same on each open face of the chamber:
    the fifteen Wythoffian polychora of the 5-cell family, one region, four walls, six lines and four points.
  * kipennic [3,3,3]+ (order 60): the rotations of [3,3,3]. On the chamber's faces a seed's images are the pennic
    ones (each reflection's image is a rotation's), so the faces carry the same Wythoffian polychora; the open chamber
    is one region, snad (the snub decachoron, its kipennic version: 5+5 snits, 10+10 gyro traps, 60 irregular tets).
    The chamber is half a fundamental domain (its mirror image is the other half).
  * decaic [[3,3,3]] (order 240): [3,3,3] with the swap of the 5-cell and its dual (beta read backwards). Its map is the
    exact arrangement of planes of decaic_atlas/arrangement.json (half the chamber, beta_1 >= beta_4), each face
    classified by its polychoron (decaic_atlas/shapes.json, named in decaic_shapes.json), and carried by the swap onto
    the other half, so the whole chamber shows every decaic shape twice, mirrored.

Every face of the map is drawn where its polychoron has that many degrees of variation (a face of a region's plane
where nothing changes is not a wall); each shape has an example seed in every piece (walls, lines: the most even
seed; regions: the central seed), and each region its extent, the walls round it. The polychora are classified by decaic_atlas/group.py (DECAIC_SUB = pen or full).

    python a4_map.py pennic
    python a4_map.py kipennic
    python a4_map.py decaic
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEC = os.path.join(ROOT, "decaic_atlas")
SUBS = {"pennic": "pen", "kipennic": "kipen", "decaic": "full"}

# (the Wythoffian names, by the ringed nodes: beta_k > 0)
WYTHOFF = {(1,): ("pen", "pentachoron"), (2,): ("rap", "rectified pentachoron"), (3,): ("rap", "rectified pentachoron"),
           (4,): ("pen", "pentachoron"), (1, 2): ("tip", "truncated pentachoron"), (3, 4): ("tip", "truncated pentachoron"),
           (1, 3): ("srip", "small rhombated pentachoron"), (2, 4): ("srip", "small rhombated pentachoron"),
           (1, 4): ("spid", "small prismatodecachoron"), (2, 3): ("deca", "decachoron"),
           (1, 2, 3): ("grip", "great rhombated pentachoron"), (2, 3, 4): ("grip", "great rhombated pentachoron"),
           (1, 2, 4): ("prip", "prismatorhombated pentachoron"), (1, 3, 4): ("prip", "prismatorhombated pentachoron"),
           (1, 2, 3, 4): ("gippid", "great prismatodecachoron")}


def frame():
    """The generator's simple roots R (seed s = R^-1 beta), the chart's centre c and frame F (3 x 4)."""
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
    from four_d_vertex_generator.library import fundamental_chamber_roots
    R = np.asarray(fundamental_chamber_roots("a4"), float)
    Ri = np.linalg.inv(R)
    corners = (Ri @ np.eye(4)).T
    corners /= np.linalg.norm(corners, axis=1)[:, None]
    c = corners.sum(0)
    c /= np.linalg.norm(c)
    # (axes: z towards beta_1's corner, x towards beta_4's, both orthogonal to c)
    z = corners[0] - (corners[0] @ c) * c
    z /= np.linalg.norm(z)
    x = corners[3] - (corners[3] @ c) * c - (corners[3] @ z) * z
    x /= np.linalg.norm(x)
    y = np.linalg.svd(np.vstack([c, z, x]))[2][-1]
    return Ri, c, np.vstack([x, y, z])


RI, C, F = frame()


def seed(beta):
    s = RI @ np.asarray(beta, float)
    return s / np.linalg.norm(s)


def chart(beta):
    """beta (any scale, beta >= 0) -> the gnomonic chart at the chamber's centre."""
    s = RI @ np.asarray(beta, float)
    return F @ (s / (s @ C))


def seed_text(beta):
    return ", ".join(f"{v:.12f}" for v in seed(beta))


def beta_text(beta):
    b = np.asarray(beta, float)
    b = b / b.sum()
    return "(" + ", ".join(f"{v:.4f}" for v in b) + ")"


def classify(points, sub):
    """(signature, key) of each beta, by decaic_atlas/group.py under DECAIC_SUB = sub (in a process of its own)."""
    code = ("import json, sys, numpy as np\nimport group as G\nP = json.loads(sys.stdin.read())\n"
            "print(json.dumps([list(G.label(np.asarray(p, float) / np.sum(p), 1e-9)) for p in P]))")
    r = subprocess.run([sys.executable, "-c", code], input=json.dumps([[float(v) for v in p] for p in points]),
                       capture_output=True, text=True, cwd=DEC, env={**os.environ, "DECAIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def poly_mesh(P):
    """Convex polygon (its corners in any order, in one plane) -> ordered corners and fan triangles."""
    P = np.asarray(P, float)
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    e1 = P[0] - c
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    P = P[np.argsort(np.arctan2((P - c) @ e2, (P - c) @ e1))]
    return P, [[0, k, k + 1] for k in range(1, len(P) - 1)]


def r5(v):
    return [round(float(x), 6) for x in v]


def domain():
    """The chamber: four corners (beta = e_k), six edges, four faces (the mirrors beta_k = 0)."""
    V = [chart(e) for e in np.eye(4)]
    faces = []
    for k in range(4):
        loop = [i for i in range(4) if i != k]
        faces.append({"loop": loop, "name": f"the mirror β{k + 1} = 0", "kind": "mirror", "ids": []})
    return {"verts": [r5(v) for v in V], "edges": [list(e) for e in itertools.combinations(range(4), 2)], "faces": faces}


def parse_sig(sig):
    import re
    m = re.match(r"(\d+): cells (.*) \| faces (.*) \| edges (.*)", sig or "")
    if not m:
        return None
    n, cells, faces, edges = m.groups()
    nice = lambda s: s.replace("x", "×").replace("+", " + ")
    val = Counter()
    for part in edges.split("+"):
        k, v = part.split("x")
        val[int(v)] += int(k)
    return {"vertices": int(n), "cells": nice(cells), "faces": nice(faces), "edges": nice(edges),
            "valence": ", ".join(f"{v}: {k}" for v, k in sorted(val.items()))}


def plane_text(p):
    """A plane a . beta = 0 written as an equation in the betas."""
    pos = " + ".join((f"{a}·" if a != 1 else "") + f"β{i + 1}" for i, a in enumerate(p) if a > 0)
    neg = " + ".join((f"{-a}·" if a != -1 else "") + f"β{i + 1}" for i, a in enumerate(p) if a < 0)
    return f"{pos or '0'} = {neg or '0'}"


# ---------------------------------------------------------------------------------------------------------- the faces
def faces_pennic():
    """The chamber's fifteen faces: (dim, signs over beta_k >= 0, exact point, corner betas)."""
    out = []
    for r in range(1, 5):
        for ring in itertools.combinations(range(4), r):
            s = tuple(1 if k in ring else 0 for k in range(4))
            pt = [Fraction(1, r) if k in ring else Fraction(0) for k in range(4)]
            out.append({"dim": r - 1, "signs": s, "point": pt, "corners": [list(np.eye(4)[k]) for k in ring],
                        "ring": tuple(k + 1 for k in ring), "lies_on": [f"β{k + 1} = 0" for k in range(4) if k not in ring]})
    return out, [list(e) for e in np.eye(4).astype(int)]


def faces_decaic():
    A = json.load(open(os.path.join(DEC, "arrangement.json")))
    planes = [tuple(b) for b in A["boundary"]] + [tuple(p) for p in A["planes"] if tuple(p) not in map(tuple, A["boundary"])]
    F_ = [{"dim": f["dim"], "signs": tuple(f["signs"]), "point": [Fraction(v) for v in f["point"]]} for f in A["faces"]]
    verts = [f for f in F_ if f["dim"] == 0]
    for f in F_:
        s = f["signs"]
        f["corners"] = [[float(v) for v in g["point"]] for g in verts
                        if all((a == 0 and b == 0) or (a != 0 and b in (0, a)) for a, b in zip(s, g["signs"]))]
        f["lies_on"] = [("the fold " if p == (1, 0, 0, -1) else "") + plane_text(p) for p, v in zip(planes, s) if v == 0]
    return F_, planes


# ------------------------------------------------------------------------------------------------------------ the map
def build(group):
    sub = SUBS[group]
    order = {"pennic": 120, "kipennic": 60, "decaic": 240}[group]
    faces, planes = faces_pennic() if group in ("pennic", "kipennic") else faces_decaic()
    lab = classify([f["point"] for f in faces], sub)
    for f, (sig, key) in zip(faces, lab):
        f["sig"], f["key"] = sig, key
    # the shapes: one per key, named
    names = {}
    if group in ("pennic", "kipennic"):
        for f in faces:                         # (kipennic: on the mirrors a seed's rotations make the Wythoffian
            nm = WYTHOFF[f["ring"]]             #  polychoron still; inside, the kipennic snub decachoron)
            if group == "kipennic" and f["dim"] == 3:
                nm = ("snad", "snub decachoron (its kipennic version)")
            names.setdefault(f["key"], nm)
    else:
        S = json.load(open(os.path.join(DEC, "shapes.json")))["shapes"]
        N = json.load(open(os.path.join(DEC, "decaic_shapes.json")))["shapes"]
        full = {}                                       # (the list's full names: "Name - (pron). full name. It has")
        import re
        for line in open(os.path.join(ROOT, "pages", "decaic_source.txt")):
            m = re.match(r"(\w+) - (?:\([^)]*\)\.?\s*)?(?:the |The )?(.+?)(?:\.\s| has | is |,)", line)
            if m:
                full.setdefault(m.group(1).lower(), m.group(2).strip())
        for s, n in zip(S, N):
            assert s["vertices"] == n["vertices"]
            for k in s["keys"]:
                names[k] = (n["name"], full.get(n["name"], ""))
    by_key = {}
    for f in faces:
        by_key.setdefault(f["key"], []).append(f)
    types = []
    DIM = {0: "point", 1: "line", 2: "wall", 3: "region"}
    for key, fs in by_key.items():
        deg = max(f["dim"] for f in fs)                    # (the shape's degrees of variation: its largest face)
        own = [f for f in fs if f["dim"] == deg]
        rep = own[0]
        p = parse_sig(rep["sig"]) or {}
        name, full = names.get(key, (key[:6], ""))
        b = [float(v) for v in rep["point"]]
        common = sorted(set.intersection(*[set(f["lies_on"]) for f in own])) if own else []
        types.append({"id": name, "name": full, "dim": DIM[deg], "group": DIM[deg] + "s", "key": key, "sig": rep["sig"],
                      **p, "seed": seed_text(b), "beta": beta_text(b), "pieces": len(own), "degrees": deg,
                      "locus": ("on " + " and ".join(common)) if common and len(common) <= 2 and deg > 0 else "",
                      "symbol": "circle" if p.get("vertices") == order else "diamond"})
    # (two keys with one name: kept apart by a suffix)
    cnt = Counter(t["id"] for t in types)
    seen = Counter()
    for t in types:
        if cnt[t["id"]] > 1:
            seen[t["id"]] += 1
            t["id"] = f"{t['id']}_{seen[t['id']]}"
    tid = {t["key"]: t["id"] for t in types}
    for t in types:                                    # (the mirrors beta_k = 0 its pieces lie in, the swap's images too)
        ks = set()
        for f in by_key[t["key"]]:
            if f["dim"] == t["degrees"]:
                ks |= {k for k in range(4) if f["signs"][k] == 0}
                if group == "decaic":
                    ks |= {3 - k for k in range(4) if f["signs"][k] == 0}
        t["mirrors"] = sorted(ks)
    deg_of = {t["key"]: t["degrees"] for t in types}
    # the geometry: each face drawn where it is its shape's own (its dimension is the shape's degrees), and its image
    # under the swap for decaic (beta read backwards)
    images = [lambda b: b] + ([lambda b: b[::-1]] if group == "decaic" else [])
    walls, lines, points = [], [], []
    for f in faces:
        if f["dim"] != deg_of[f["key"]] or f["dim"] == 3:
            continue
        t = tid[f["key"]]
        for im in images:
            Q = [chart(im(np.asarray(c, float))) for c in f["corners"]]
            if f["dim"] == 2:
                P, tri = poly_mesh(Q)
                walls.append({"id": t, "verts": [r5(q) for q in P], "faces": tri})
            elif f["dim"] == 1:
                lines.append({"id": t, "pts": [r5(q) for q in Q]})
            else:
                b = im(np.asarray([float(v) for v in f["point"]]))
                points.append({"id": t, "q": r5(chart(b)), "beta": beta_text(b), "seed": seed_text(b)})
    # (a point on the fold is its own image: kept once)
    uniq = []
    for p in points:
        if not any(p["id"] == u["id"] and np.allclose(p["q"], u["q"], atol=1e-6) for u in uniq):
            uniq.append(p)
    points = uniq
    # bounds: the shapes on a shape's faces' boundaries
    def under(f, g):
        return g["dim"] < f["dim"] and all((a == 0 and b == 0) or (a != 0 and b in (0, a))
                                           for a, b in zip(f["signs"], g["signs"]))
    bounds = {t["id"]: set() for t in types}
    for f in faces:
        if f["dim"] != deg_of[f["key"]]:
            continue
        for g in faces:
            if g["dim"] == deg_of[g["key"]] and g["key"] != f["key"] and under(f, g):
                bounds[tid[f["key"]]].add(tid[g["key"]])
    order_ = {"region": 0, "wall": 1, "line": 2, "point": 3}
    for t in types:
        t["bounds"] = sorted(bounds[t["id"]], key=lambda i: (order_[next(u["dim"] for u in types if u["id"] == i)], i))
    sep = {t["id"]: set() for t in types}
    for t in types:
        for b in t["bounds"]:
            sep[b].add(t["id"])
    for t in types:
        t["separates"] = sorted(i for i in sep[t["id"]] if next(u["degrees"] for u in types if u["id"] == i) == t["degrees"] + 1)
    # the region a beta lies in (by the planes' signs; decaic: the half beta_1 >= beta_4 holds the arrangement)
    cell = {f["signs"]: f for f in faces if f["dim"] == 3}

    def region_at(b):
        b = np.asarray(b, float)
        if b.min() < 0:
            return None
        if group != "decaic":
            return tid[next(f["key"] for f in faces if f["dim"] == 3)]
        bb = b if b[0] >= b[3] else b[::-1]
        f = cell.get(tuple(int(np.sign(np.dot(p, bb))) for p in planes))
        return tid[f["key"]] if f else None

    # example seeds: each wall, line and point at its most even seed (decaic: optimize_seeds.py's, just inside the
    # shape where the optimum is on its edge; pennic, kipennic: the uniform Wythoffian, equal betas on its ringed
    # nodes), each region at its central seed (decaic: central_seeds.py's deepest point; pennic, kipennic: the
    # chamber's centre), in every piece of the shape
    ex = {}
    if group == "decaic":
        opt = {r["name"]: r["off_seed"] for r in json.load(open(os.path.join(DEC, "optimized_seeds.json")))["shapes"]}
        cen = {r["name"]: r["centre"] for r in json.load(open(os.path.join(DEC, "central_seeds.json")))["shapes"]}
        for t in types:
            b = np.asarray(cen[t["id"]] if t["dim"] == "region" else opt[t["id"]], float)
            b = np.clip(b, 0, None)
            ex[t["id"]] = [b] + ([b[::-1]] if not np.allclose(b, b[::-1]) else [])
    else:
        for t in types:
            ex[t["id"]] = [np.array([1.0 if k + 1 in f["ring"] else 0.0 for k in range(4)])
                           for f in by_key[t["key"]] if f["dim"] == t["degrees"]]
    chk = classify([b for t in types for b in ex[t["id"]]], sub)
    k = 0
    examples = []
    for t in types:
        for b in ex[t["id"]]:
            if chk[k][1] != t["key"]:
                raise SystemExit(f"{group}: the example seed of {t['id']} {b} is not the shape")
            k += 1
            examples.append({"id": t["id"], "q": r5(chart(b)), "beta": beta_text(b), "seed": seed_text(b)})
        b0 = ex[t["id"]][0]
        t["seed"], t["beta"] = seed_text(b0), beta_text(b0)
        t["example"] = "central seed (the deepest point of the region)" if t["dim"] == "region" else (
            "the point" if t["dim"] == "point" else "most even seed (least longest / shortest edge)" if group == "decaic"
            else "uniform seed (all edges equal)")
    # each region's extent: the faces of the map between it and anything else (the walls round it), with their edges,
    # each found by stepping off the face both ways
    shells = {}
    for f in faces:
        if f["dim"] != 2:
            continue
        P = np.array(f["corners"], float)
        c = P.mean(0)
        n = np.linalg.svd(np.vstack([P - c, np.ones(4)]))[2][-1]
        sides = [region_at(c + e * n) for e in (1e-7, -1e-7)]
        for r in set(sides) - {None}:
            if sides.count(r) == 1:
                for im in images:
                    Q, tri = poly_mesh([chart(im(q)) for q in P])
                    sh = shells.setdefault(r, {"verts": [], "faces": [], "edges": {}})
                    o = len(sh["verts"])
                    sh["verts"] += [r5(q) for q in Q]
                    sh["faces"] += [[a + o, b + o, c_ + o] for a, b, c_ in tri]
                    for a in range(len(Q)):
                        e = tuple(sorted((tuple(r5(Q[a])), tuple(r5(Q[(a + 1) % len(Q)])))))
                        sh["edges"].setdefault(e, []).append(np.cross(Q[1] - Q[0], Q[2] - Q[0]))
    for r, sh in shells.items():               # (an edge between two pieces of one flat wall is not an edge of it)
        keep = []
        for e, ns in sh["edges"].items():
            flat = len(ns) == 2 and abs(abs(ns[0] @ ns[1]) / np.linalg.norm(ns[0]) / np.linalg.norm(ns[1]) - 1) < 1e-6
            if not flat:
                keep.append([list(e[0]), list(e[1])])
        sh["edges"] = keep
    for t in types:
        t["where"] = {"region": "region", "wall": "wall", "line": "line", "point": "point"}[t["dim"]] + (
            f" ({t['pieces']} pieces)" if t["pieces"] > 1 else "")
    order_t = {"region": 0, "wall": 1, "line": 2, "point": 3}
    types.sort(key=lambda t: (order_t[t["dim"]], -(t.get("vertices") or 0), t["id"]))
    out = {"group": group, "order": order, "domain": domain(), "types": types, "walls": walls, "lines": lines,
           "points": points, "examples": examples, "shells": shells}
    path = os.path.join(HERE, f"a4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({Counter(t['dim'] for t in types)}), {len(walls)} wall pieces, {len(lines)} "
          f"line pieces, {len(points)} points, {len(examples)} example seeds, {len(shells)} region shells")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["pennic", "kipennic", "decaic"]:
        build(g)
