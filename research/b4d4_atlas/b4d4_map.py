"""The B4 and D4 symmetry domains: the 3-sphere maps of the tessic family's groups (tessic_atlas/group.py), for the
B4 and D4 pages of the domain viewer (built by ../a4_atlas/build_page.py b4 | d4).

Every group of the family has no walls inside the B4 chamber (tessic_atlas/tessic_subgroups.md: one region fills it,
checked there by the plane arrangement, or by the curved-wall map, the edge-zone survey and the region closure), so a
map is the chamber's fifteen faces (interior, four faces, six edges, four corners), each classified by the polychoron a
seed there makes (tessic_atlas/group.py under TESSIC_SUB; three more seeds in each face check it is one shape), and
named from the tessic list (pages/tessic_source.txt) by the atlas's seeds.

    B4 page: tessic [4,3,3] (full), kitessic [4,3,3]+ (ki), tessipyritic [4,(3,3)+] (pyr), on the B4 chamber.
    D4 page: demitessic [3^{1,1,1}] (demi), kidemitessic D4+ (kidemi), on the D4 chamber: the B4 chamber and its mirror
             image in beta_4 = 0, which is no D4 mirror but a fold (the reflection in it normalizes D4, so a seed and
             its image make congruent polychora). Every shape is drawn in both halves, as decaic's are about its swap.

Seeds are given in the generator's coordinates (s = R^-1 beta, R = fundamental_chamber_roots("b4"), whose mirrors are
the atlas's: beta_1 the hexadecachoron's node, beta_4 the tesseract's), and the chart is the gnomonic one at the
domain's centre, as on the A4 page.

    python b4d4_map.py            # all five maps (b4d4_map_<group>.json)
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TES = os.path.join(ROOT, "tessic_atlas")
sys.path.insert(0, os.path.join(ROOT, "a4_atlas"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
import a4_map as M  # noqa: E402  (poly_mesh, parse_sig, r5, faces_pennic: the A4 page's own)
from four_d_vertex_generator.library import fundamental_chamber_roots  # noqa: E402

GROUPS = {"tessic": ("b4", "full", 384), "kitessic": ("b4", "ki", 192), "tessipyritic": ("b4", "pyr", 192),
          "demitessic": ("d4", "demi", 192), "kidemitessic": ("d4", "kidemi", 96)}
MAIN = {"b4": "tessic", "d4": "demitessic"}       # (the family's reflection group: the others are seeded as it is)
MAIN_SEED = {}                                    # (family -> a face's signs -> the main group's seed there)
R = np.asarray(fundamental_chamber_roots("b4"), float)
RI = np.linalg.inv(R)
U4 = R[3] / np.linalg.norm(R[3])
REFL4 = np.eye(4) - 2 * np.outer(U4, U4)                  # (the reflection in beta_4 = 0: D4's fold)


def d4_seed(signs, dist=None):
    """The D4-symmetric seed of a face of the B4 chamber (signs: which beta are nonzero there), as beta: equal
    distance to each D4 mirror the face is off. D4's mirrors are beta_1, beta_2, beta_3 and beta_3' (beta_3's image in
    beta_4 = 0); a point is off beta_3' unless beta_3 = beta_4 = 0. Put back in the B4 chamber if it falls in the
    other half."""
    s1, s2, s3, s4 = (int(v != 0) for v in signs)
    d = np.array([s1, s2, s3, 1.0 if (s3 or s4) else 0.0]) * (1.0 if dist is None else np.asarray(dist, float))
    D = np.vstack([R[0], R[1], R[2], REFL4 @ R[2]])
    x = np.linalg.solve(D, d)
    b = R @ x
    if b[3] < -1e-12:
        x = REFL4 @ x
        b = R @ x
    b = np.clip(b, 0, None)
    return b / b.sum()


def corner(beta):
    s = RI @ np.asarray(beta, float)
    return s / np.linalg.norm(s)


class Chart:
    """The domain's corners (in seed space) and the gnomonic chart at its centre."""

    def __init__(self, fam):
        c = [corner(e) for e in np.eye(4)]
        self.fam = fam
        # (D4: hex's corner, beta_2's, the tesseract's and its image; beta_3's corner is the middle of the last edge)
        self.corners = c if fam == "b4" else [c[0], c[1], c[3], REFL4 @ c[3]]
        C = np.sum(self.corners, 0)
        self.C = C / np.linalg.norm(C)
        z = self.corners[0] - (self.corners[0] @ self.C) * self.C
        z /= np.linalg.norm(z)
        x = self.corners[-1] - (self.corners[-1] @ self.C) * self.C - (self.corners[-1] @ z) * z
        x /= np.linalg.norm(x)
        y = np.linalg.svd(np.vstack([self.C, z, x]))[2][-1]
        self.F = np.vstack([x, y, z])
        self.images = [np.eye(4)] + ([REFL4] if fam == "d4" else [])

    def __call__(self, beta, im=None):
        s = (np.eye(4) if im is None else im) @ (RI @ np.asarray(beta, float))
        return M.r5(self.F @ (s / (s @ self.C)))

    def domain(self):
        V = [M.r5(self.F @ (q / (q @ self.C))) for q in self.corners]
        if self.fam == "b4":
            faces = [{"loop": [i for i in range(4) if i != k], "name": f"the mirror β{k + 1} = 0", "kind": "mirror",
                      "ids": []} for k in range(4)]
            labels = ["β1", "β2", "β3", "β4"]
        else:
            faces = [{"loop": [1, 2, 3], "name": "the mirror β1 = 0", "kind": "mirror", "ids": []},
                     {"loop": [0, 2, 3], "name": "the mirror β2 = 0", "kind": "mirror", "ids": []},
                     {"loop": [0, 1, 2], "name": "the mirror β3 = 0", "kind": "mirror", "ids": []},
                     {"loop": [0, 1, 3], "name": "the mirror β3′ = 0 (β3's image in the fold)", "kind": "mirror",
                      "ids": []}]
            labels = ["β1", "β2", "β4", "β4′"]
        return {"verts": V, "labels": labels, "edges": [list(e) for e in itertools.combinations(range(4), 2)],
                "faces": faces}


def seed_text(beta, im=None):
    s = (np.eye(4) if im is None else im) @ (RI @ np.asarray(beta, float))
    s = s / np.linalg.norm(s)
    return ", ".join(f"{v:.12f}" for v in s)


def classify(points, sub, tol=1e-9):
    """(signature, key) of each beta, by tessic_atlas/group.py under TESSIC_SUB = sub (in a process of its own)."""
    code = ("import json, sys, numpy as np\nimport group as G\nP = json.loads(sys.stdin.read())\n"
            f"print(json.dumps([list(G.label(np.asarray(p, float) / np.sum(p), {tol})) for p in P]))")
    r = subprocess.run([sys.executable, "-c", code], input=json.dumps([[float(v) for v in p] for p in points]),
                       capture_output=True, text=True, cwd=TES, env={**os.environ, "TESSIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def full_names():
    """The list's full names: "Name - the full name has ..." (pages/tessic_source.txt)."""
    out = {}
    for line in open(os.path.join(ROOT, "pages", "tessic_source.txt")):
        m = re.match(r"(\w+) - (?:\([^)]*\)\.?\s*)?(?:the |The )?(.+?)(?:\.\s| has | is |,|\.$)", line.strip())
        if m:
            out.setdefault(m.group(1).lower(), m.group(2).strip())
    return out


def name_table():
    """Arrangement key -> name: the atlas's named seeds of every group (a key is the polychoron's combinatorial type,
    the same under every group), the tessic shapes by their ringed nodes."""
    pts, nms, subs = [], [], []
    for s in json.load(open(os.path.join(TES, "tessic_shapes.json")))["shapes"]:
        pts.append(s["seed"]); nms.append(s["name"]); subs.append("full")
    for sub in ("ki", "pyr", "demi", "kidemi"):
        for f in ("optimized_seeds", "central_seeds"):
            for s in json.load(open(os.path.join(TES, f"{f}_{sub}.json")))["shapes"]:
                pts.append(np.clip(s["best"] if f == "optimized_seeds" else s["centre"], 0, None))
                nms.append(s["name"]); subs.append(sub)
    table = {}
    for sub in sorted(set(subs)):
        idx = [i for i, s in enumerate(subs) if s == sub]
        for i, (_, key) in zip(idx, classify([pts[i] for i in idx], sub, 1e-7)):
            if key:
                table.setdefault(key, nms[i])
    return table


def build(group, table, full):
    fam, sub, order = GROUPS[group]
    ch = Chart(fam)
    faces, _ = M.faces_pennic()
    lab = classify([f["point"] for f in faces], sub)
    for f, (sig, key) in zip(faces, lab):
        f["sig"], f["key"] = sig, key
    # (one shape over each face: three more seeds in it, each the same polychoron)
    rng = np.random.default_rng(7)
    chk_pts, owner = [], []
    for i, f in enumerate(faces):
        if f["dim"] == 0:
            continue
        for w in rng.dirichlet(np.ones(len(f["corners"])), 3):
            chk_pts.append(np.asarray(f["corners"], float).T @ w)
            owner.append(i)
    bad = [(faces[i]["ring"], k) for i, (_, k) in zip(owner, classify(chk_pts, sub)) if k != faces[i]["key"]]
    if bad:
        raise SystemExit(f"{group}: a face is not one shape: {bad[:4]}")
    # (a key the lists do not name: where the group's orbit is the whole tessic one (as many vertices), the polychoron
    #  is the tessic shape there, its cells only classed otherwise by the smaller group)
    unnamed = [f for f in faces if f["key"] not in table]
    if unnamed:
        for f, (sig, k) in zip(unnamed, classify([f["point"] for f in unnamed], "full")):
            if k in table and (M.parse_sig(sig) or {}).get("vertices") == (M.parse_sig(f["sig"]) or {}).get("vertices"):
                table[f["key"]] = table[k]
                print(f"  {group}: {f['key'][:6]} on {f['lies_on']} is {table[k]} (the whole tessic orbit)", flush=True)
    by_key = {}
    for f in faces:
        by_key.setdefault(f["key"], []).append(f)
    DIM = {0: "point", 1: "line", 2: "wall", 3: "region"}
    types = []
    for key, fs in by_key.items():
        deg = max(f["dim"] for f in fs)
        own = [f for f in fs if f["dim"] == deg]
        p = M.parse_sig(own[0]["sig"]) or {}
        nm = table.get(key, key[:6])
        common = sorted(set.intersection(*[set(f["lies_on"]) for f in own]))
        # (the mirrors its pieces lie in, as the domain's faces: D4's beta_4 = 0 is the fold, and beta_3 = 0's image
        #  is the domain's fourth face)
        ks = sorted({k for f in own for k in range(4) if f["signs"][k] == 0})
        if fam == "d4":
            ks = sorted({k for k in ks if k < 3} | ({3} if 2 in ks else set()))
        types.append({"id": nm, "name": full.get(nm, ""), "dim": DIM[deg], "group": DIM[deg] + "s", "key": key,
                      "sig": own[0]["sig"], **p, "pieces": len(own) * len(ch.images), "degrees": deg,
                      "locus": ("on " + " and ".join(common)) if common and deg > 0 else "", "mirrors": ks,
                      "symbol": "circle" if p.get("vertices") == order else "diamond", "_own": own})
    tid = {t["key"]: t["id"] for t in types}
    under = lambda f, g: g["dim"] < f["dim"] and all((a == 0 and b == 0) or (a != 0 and b in (0, a))
                                                       for a, b in zip(f["signs"], g["signs"]))
    # geometry: each face where it is its shape's own, and (D4) its image in the fold
    walls, lines, points = [], [], []
    for t in types:
        for f in t["_own"]:
            for im in ch.images:
                if f["dim"] == 2:
                    P, tri = M.poly_mesh([ch(c, im) for c in f["corners"]])
                    walls.append({"id": t["id"], "verts": [M.r5(q) for q in P], "faces": tri})
                elif f["dim"] == 1:
                    lines.append({"id": t["id"], "pts": [ch(c, im) for c in f["corners"]]})
                elif f["dim"] == 0:
                    points.append({"id": t["id"], "q": ch(f["point"], im), "beta": M.beta_text(f["point"]),
                                   "seed": seed_text(f["point"], im)})
    uniq = []                                    # (a point on the fold is its own image)
    for p in points:
        if not any(p["id"] == u["id"] and np.allclose(p["q"], u["q"], atol=1e-6) for u in uniq):
            uniq.append(p)
    points = uniq
    if fam == "d4":
        # (D4: a piece and its image in the fold are one piece where they meet there: the piece has a face in the fold
        #  of one dimension less and the same shape, D4 seeing no boundary there; a piece in the fold is its own image)
        key_of = {tuple(f["signs"]): f["key"] for f in faces}
        for t in types:
            n = 0
            for f in t["_own"]:
                if f["signs"][3] == 0:
                    n += 1
                    continue
                joined = any(g["dim"] == f["dim"] - 1 and g["signs"][3] == 0 and g["key"] == f["key"]
                             and under(f, g) for g in faces)
                n += 1 if joined else 2
            t["pieces"] = n
    # bounds and separates
    deg_of = {t["key"]: t["degrees"] for t in types}
    bounds = {t["id"]: set() for t in types}
    for f in faces:
        if f["dim"] != deg_of[f["key"]]:
            continue
        for g in faces:
            if g["dim"] == deg_of[g["key"]] and g["key"] != f["key"] and under(f, g):
                bounds[tid[f["key"]]].add(tid[g["key"]])
    rank = {"region": 0, "wall": 1, "line": 2, "point": 3}
    dim_of = {t["id"]: t["dim"] for t in types}
    for t in types:
        t["bounds"] = sorted(bounds[t["id"]], key=lambda i: (rank[dim_of[i]], i))
    sep = {t["id"]: set() for t in types}
    for t in types:
        for b in t["bounds"]:
            sep[b].add(t["id"])
    for t in types:
        t["separates"] = sorted(i for i in sep[t["id"]] if rank[dim_of[i]] == rank[t["dim"]] - 1)
    # example seeds: the atlas's most even seed of the shape where it lies in the shape's own face (else the uniform
    # seed, equal beta on the ringed nodes); a region at the chamber's centre (its central seed)
    opt = {}
    files = [("optimized_seeds" + ("" if sub == "full" else "_" + sub), "best")]
    for fn, k in files:
        for s in json.load(open(os.path.join(TES, fn + ".json")))["shapes"]:
            opt[s["name"]] = np.clip(np.asarray(s[k], float), 0, None)
    examples = []
    main = MAIN[fam]
    if group != main and fam not in MAIN_SEED:
        build(main, table, full)                 # (its seeds first: this group's are taken from it)
    for t in types:
        f0 = t["_own"][0]
        if fam == "d4":
            # (D4's own symmetric seeds: equal distance to the D4 mirrors ringed on the face (beta_1, beta_2, beta_3
            #  and beta_3', beta_3's image in beta_4 = 0); all edges equal on every Wythoffian shape, the region's
            #  at the chamber's centre, on its triangular symmetry's axis)
            if t["dim"] == "region":
                b = d4_seed((1, 1, 1, 1))
                how = ("uniform seed (the D4 chamber's centre: equal distance to its four mirrors, all edges equal)"
                       if group == "demitessic" else
                       "the D4 chamber's centre (demitessic's uniform seed; a snub, sadi is not uniform there: edges "
                       "0.943 to 1.155), and sadi's uniform seed, also on the axis: distances 1, 1/φ, 1, 1 to the "
                       "mirrors β1, β2, β3, β3′ (all edges equal)")
                t["_ex"] = [(b, how)]
                if group == "kidemitessic":
                    phi = (1 + 5 ** 0.5) / 2
                    t["_ex"].append((d4_seed((1, 1, 1, 1), (1, 1 / phi, 1, 1)), how))
            else:
                bs = []
                for f in t["_own"]:
                    b = d4_seed(f["signs"])
                    if not any(np.allclose(b, c) for c in bs):
                        bs.append(b)
                how = ("the point" if t["dim"] == "point" else
                       "uniform seed (equal distance to its ringed D4 mirrors: all edges equal)")
                t["_ex"] = [(b, how) for b in bs]
            continue
        if group != main:
            # (the same chamber layout as the main symmetry's: each face seeded where the main one seeds it)
            if t["dim"] == "region":
                t["_ex"] = [(np.full(4, 0.25), f"{main}'s central seed (the chamber's centre)")]
            else:
                t["_ex"] = [(MAIN_SEED[fam][f["signs"]], f"{main}'s seed there (the same place as on {main})")
                            for f in t["_own"]]
                if t["id"] == "sadi":            # (and the snub's own uniform seed in that face, all edges equal)
                    phi = (1 + 5 ** 0.5) / 2
                    t["_ex"].append((np.array([1 / phi ** 2, 1 / phi ** 3, 1 / phi ** 2, 0.0]) / (2 / phi ** 2 + 1 / phi ** 3),
                                     f"{main}'s seed there (the same place as on {main}), and sadi's uniform seed "
                                     "(β1 : β2 : β3 = 1 : 1/φ : 1, all edges equal)"))
                    t["_ex"] = [(b, t["_ex"][-1][1]) for b, _ in t["_ex"]]
            continue
        uni = np.array([1.0 if k + 1 in f0["ring"] else 0.0 for k in range(4)])
        if t["dim"] == "region":
            b, how = np.full(4, 0.25), "central seed (the chamber's centre)"
        elif t["id"] in opt and len(t["_own"]) == 1 and all((opt[t["id"]][k] > 1e-9) == (f0["signs"][k] != 0)
                                                             or (fam == "d4" and k == 3 and opt[t["id"]][k] <= 1e-9)
                                                             for k in range(4)):
            # (D4: a seed in the fold lies in the piece too, the piece going on across it)
            b, how = opt[t["id"]], "most even seed (least longest / shortest edge)"
        else:
            b, how = uni, "the point" if t["dim"] == "point" else "uniform seed (equal β on its ringed nodes)"
        t["_ex"] = [(b, how)] if t["dim"] != "region" or True else []
        if t["dim"] != "region" and len(t["_own"]) > 1:   # (every piece: its own uniform seed)
            t["_ex"] = [(np.array([1.0 if k + 1 in f["ring"] else 0.0 for k in range(4)]), how) for f in t["_own"]]
        if t["dim"] != "region":
            for f in t["_own"]:
                MAIN_SEED.setdefault(fam, {})[f["signs"]] = (t["_ex"][0][0] if len(t["_own"]) == 1 else
                                                             np.array([1.0 if k + 1 in f["ring"] else 0.0
                                                                       for k in range(4)]))
    flat = [(t, b, how) for t in types for b, how in t["_ex"]]
    chk = classify([b for _, b, _ in flat], sub, 1e-7)
    for (t, b, how), (_, k) in zip(flat, chk):
        if k != t["key"]:
            raise SystemExit(f"{group}: the example seed of {t['id']} {b} is not the shape")
        for im in ch.images:
            q = ch(b, im)
            if any(e["id"] == t["id"] and np.allclose(e["q"], q, atol=1e-6) for e in examples):
                continue
            examples.append({"id": t["id"], "q": q, "beta": M.beta_text(b), "seed": seed_text(b, im)})
        t["seed"], t["beta"], t["example"] = seed_text(t["_ex"][0][0]), M.beta_text(t["_ex"][0][0]), t["_ex"][0][1]
    # the region's extent: the domain's faces (not D4's fold)
    shells = {}
    reg = [t for t in types if t["dim"] == "region"]
    if reg:
        dom = ch.domain()
        V = np.array(dom["verts"])
        sh = {"verts": [], "faces": [], "edges": []}
        for f in dom["faces"]:
            o = len(sh["verts"])
            sh["verts"] += [list(V[i]) for i in f["loop"]]
            sh["faces"].append([o, o + 1, o + 2])
        sh["edges"] = [[list(V[a]), list(V[b])] for a, b in dom["edges"]]
        shells[reg[0]["id"]] = sh
    for t in types:
        t["where"] = t["dim"] + (f" ({t['pieces']} pieces)" if t["pieces"] > 1 else "")
        for k_ in ("_own", "_ex"):
            t.pop(k_, None)
    types.sort(key=lambda t: (rank[t["dim"]], -(t.get("vertices") or 0), t["id"]))
    out = {"group": group, "family": fam, "order": order, "domain": ch.domain(), "types": types, "walls": walls,
           "lines": lines, "points": points, "examples": examples, "shells": shells}
    # (D4: β4 = 0, where the B4 chamber the maps are found on meets its image, is no D4 mirror and no boundary of
    #  any shape, one of the three planes of the chamber's triangular symmetry: not drawn)
    path = os.path.join(HERE, f"b4d4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({dict(Counter(t['dim'] for t in types))}): "
          + ", ".join(f"{t['id']} {t['where']}" for t in types), flush=True)


if __name__ == "__main__":
    table, full = name_table(), full_names()
    for g in sys.argv[1:] or list(GROUPS):
        build(g, table, full)
