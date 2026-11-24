"""The 3-sphere maps of the F4 symmetries with flat walls on the F4 chamber (f4_map_<group>.json), for the F4 domain
viewer (../a4_atlas/build_page.py f4), laid out as the A4 maps (../a4_atlas/a4_map.py).

The domain is the chamber of [3,4,3]: seeds beta >= 0, beta_k the distance to the k-th mirror (mirrors 1, 2 long, 3, 4
short, as ../contic_atlas/group.py; the generator's Coordinates mode takes s = R^-1 beta, R =
fundamental_chamber_roots("f4")), drawn in the gnomonic chart at the chamber's centre.

  * icoic [3,4,3] (1152): the chamber's fifteen faces, the Wythoffian polychora of the 24-cell family.
  * kiicoic [3,4,3]+ (576): the same faces (on the mirrors a seed's rotations give the Wythoffian polychoron), and
    inside one region, snoc (the snub tetracontoctachoron, its kiicoic version).
  * contic [[3,4,3]] (2304): the exact plane arrangement over Q(sqrt2) of ../contic_atlas/arrangement.json (half the
    chamber, beta_1 >= beta_4), each face classified, carried by the swap (beta read backwards) onto the other half.
  * icopyritic [3,4,3+] (576) and toxitic [3+,4,3+] (288): the chamber's faces classified under each; one region fills
    the chamber (pysnapi, sebtidi). Names from ../toxitic_atlas/opt_central_<sheet>.json (icopyritic's second sheet,
    icpb, is its swap image: its seeds read backwards).

    python f4_map.py [icoic kiicoic contic icopyritic toxitic]
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CON = os.path.join(ROOT, "contic_atlas")
TOX = os.path.join(ROOT, "toxitic_atlas")
sys.path.insert(0, os.path.join(ROOT, "a4_atlas"))
sys.path.insert(0, os.path.join(ROOT, "pages"))
from a4_map import parse_sig, poly_mesh, r5  # noqa: E402
import contic_names as NAMES  # noqa: E402

SUBS = {"icoic": "ico", "kiicoic": "kiico", "contic": "full", "icopyritic": "icp", "toxitic": "tox",
        "kicontic": "ki", "iocontic": "io", "oxitic": "ox"}
ORDER = {"icoic": 1152, "kiicoic": 576, "contic": 2304, "icopyritic": 576, "toxitic": 288, "kicontic": 1152,
         "iocontic": 1152, "oxitic": 576}
WYTHOFF = {(1,): "ico", (4,): "ico", (2,): "rico", (3,): "rico", (1, 2): "tico", (3, 4): "tico", (1, 3): "srico",
           (2, 4): "srico", (1, 4): "sidpii", (2, 3): "icoi", (1, 2, 3): "grico", (2, 3, 4): "grico",
           (1, 2, 4): "prico", (1, 3, 4): "prico", (1, 2, 3, 4): "gidpii"}
FULL = {**NAMES.ICOIC, **NAMES.CONTIC, **NAMES.KICONTIC, **NAMES.IOCONTIC, **NAMES.ICOPYRITIC, **NAMES.TOXITIC,
        **NAMES.OXITIC}


def frame():
    sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "src"))
    from four_d_vertex_generator.library import fundamental_chamber_roots
    R = np.asarray(fundamental_chamber_roots("f4"), float)
    Ri = np.linalg.inv(R)
    corners = (Ri @ np.eye(4)).T
    corners /= np.linalg.norm(corners, axis=1)[:, None]
    c = corners.sum(0)
    c /= np.linalg.norm(c)
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
    s = RI @ np.asarray(beta, float)
    return F @ (s / (s @ C))


def seed_text(beta):
    return ", ".join(f"{v:.12f}" for v in seed(beta))


def beta_text(beta):
    b = np.asarray(beta, float)
    b = b / b.sum()
    return "(" + ", ".join(f"{v:.4f}" for v in b) + ")"


def classify(points, sub, tol=1e-9):
    """(signature, key) of each beta, by ../contic_atlas/group.py under CONTIC_SUB = sub."""
    code = ("import json, sys, numpy as np\nimport group as G\nP = json.loads(sys.stdin.read())\n"
            f"print(json.dumps([list(G.label(np.asarray(p, float) / np.sum(p), {tol})) for p in P]))")
    r = subprocess.run([sys.executable, "-c", code], input=json.dumps([[float(v) for v in p] for p in points]),
                       capture_output=True, text=True, cwd=CON, env={**os.environ, "CONTIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def domain():
    V = [chart(e) for e in np.eye(4)]
    faces = [{"loop": [i for i in range(4) if i != k], "name": f"the mirror β{k + 1} = 0", "kind": "mirror", "ids": []}
             for k in range(4)]
    return {"verts": [r5(v) for v in V], "edges": [list(e) for e in itertools.combinations(range(4), 2)], "faces": faces}


def q2(v):
    """[a, b] (strings) -> a + b sqrt2 as a float."""
    from fractions import Fraction
    return float(Fraction(v[0])) + float(Fraction(v[1])) * np.sqrt(2)


def q2_text(v):
    from fractions import Fraction
    a, b = Fraction(v[0]), Fraction(v[1])
    if b == 0:
        return str(a)
    r = ("" if abs(b) == 1 else f"{abs(b)}·") + "√2"
    if a == 0:
        return ("-" if b < 0 else "") + r
    return f"({a} {'+' if b > 0 else '−'} {r})"


def plane_text(p):
    terms = []
    for i, c in enumerate(p):
        if q2(c) == 0:
            continue
        t = q2_text(c)
        terms.append((q2(c) > 0, ("" if t in ("1", "-1") else t.lstrip("-") + "·") + f"β{i + 1}"))
    pos = " + ".join(t for s, t in terms if s)
    neg = " + ".join(t for s, t in terms if not s)
    return f"{pos or '0'} = {neg or '0'}"


def faces_wythoff():
    out = []
    for r in range(1, 5):
        for ring in itertools.combinations(range(4), r):
            pt = [1.0 / r if k in ring else 0.0 for k in range(4)]
            out.append({"dim": r - 1, "signs": tuple(1 if k in ring else 0 for k in range(4)), "point": pt,
                        "corners": [list(np.eye(4)[k]) for k in ring], "ring": tuple(k + 1 for k in ring)})
    return out, [tuple(e) for e in np.eye(4).astype(int)]


def faces_contic():
    A = json.load(open(os.path.join(CON, "arrangement.json")))
    planes = [tuple(map(tuple, b)) for b in A["boundary"]]
    planes += [tuple(map(tuple, p)) for p in A["planes"] if tuple(map(tuple, p)) not in planes]
    fold = planes[4]
    F_ = [{"dim": f["dim"], "signs": tuple(f["signs"]), "point": f["point_float"]} for f in A["faces"]]
    verts = [f for f in F_ if f["dim"] == 0]
    for f in F_:
        s = f["signs"]
        f["corners"] = [list(g["point"]) for g in verts
                        if all((a == 0 and b == 0) or (a != 0 and b in (0, a)) for a, b in zip(s, g["signs"]))]
        f["lies_on"] = [("the fold " if p == fold else "") + plane_text(p) for p, v in zip(planes, s) if v == 0]
    P = np.array([[q2(c) for c in p] for p in planes])
    return F_, P


def sheet_seeds(sheets):
    """name -> (kind, seed, optimized, central) from ../toxitic_atlas/opt_central_<sheet>.json, in the full chamber
    (a sheet whose name ends in b is its group's swap image: read backwards)."""
    out = {}
    for sh in sheets:
        for r in json.load(open(os.path.join(TOX, f"opt_central_{sh}.json")))["shapes"]:
            rv = (lambda b: list(b)[::-1]) if sh.endswith("b") else (lambda b: list(b))
            out.setdefault(r["name"], []).append({"kind": r["kind"], "seed": rv(r["seed"]),
                                                  "optimized": rv(r["optimized"]), "central": rv(r["central"]),
                                                  "inside": r["optimized_status"].startswith("minimum inside")})
    return out


def build(group):
    sub = SUBS[group]
    order = ORDER[group]
    if group == "contic":
        faces, planes = faces_contic()
    else:
        faces, planes = faces_wythoff()
        for f in faces:
            f["lies_on"] = [f"β{k + 1} = 0" for k in range(4) if f["signs"][k] == 0]
    lab = classify([f["point"] for f in faces], sub)
    for f, (sig, key) in zip(faces, lab):
        f["sig"], f["key"] = sig, key
    # the names: by the atlas's seeds of each shape, classified here
    names, cand = {}, {}
    if group in ("icoic", "kiicoic"):
        for f in faces:
            if group == "kiicoic" and f["dim"] == 3:
                names.setdefault(f["key"], "snoc")
            else:
                names.setdefault(f["key"], WYTHOFF[f["ring"]])
    elif group == "contic":
        S = json.load(open(os.path.join(CON, "shapes.json")))["shapes"]
        N = json.load(open(os.path.join(CON, "contic_shapes.json")))["shapes"]
        for s, n in zip(S, N):
            assert s["vertices"] == n["vertices"]
            for k in s["keys"]:
                names[k] = n["name"]
    else:
        cand = sheet_seeds({"icopyritic": ["icp", "icpb"], "toxitic": ["tox"]}[group])
        items = [(nm, r["seed"]) for nm, rs in cand.items() for r in rs]
        for (nm, _), (_, key) in zip(items, classify([b for _, b in items], sub, 1e-7)):
            names.setdefault(key, nm)
        # (a face of the chamber whose shape the sheets do not list: the icoic name where the vertex counts agree)
        ico = {k: WYTHOFF[f["ring"]] for f, (_, k) in zip(faces, classify([f["point"] for f in faces], "ico"))}
        icok = classify([f["point"] for f in faces], "ico")
        for f, (isig, ik) in zip(faces, icok):
            if f["key"] not in names and isig.split(":")[0] == f["sig"].split(":")[0]:
                names[f["key"]] = ico[ik]
    by_key = {}
    for f in faces:
        by_key.setdefault(f["key"], []).append(f)
    DIM = {0: "point", 1: "line", 2: "wall", 3: "region"}
    types = []
    for key, fs in by_key.items():
        deg = max(f["dim"] for f in fs)
        own = [f for f in fs if f["dim"] == deg]
        rep = own[0]
        p = parse_sig(rep["sig"]) or {}
        name = names.get(key, key[:6])
        full = FULL.get(name, ("", ""))[0]
        if group == "kiicoic" and deg == 3:
            full += " (its kiicoic version)"
        b = [float(v) for v in rep["point"]]
        common = sorted(set.intersection(*[set(f["lies_on"]) for f in own])) if own else []
        types.append({"id": name, "name": full, "dim": DIM[deg], "group": DIM[deg] + "s", "key": key,
                      "sig": rep["sig"], **p, "seed": seed_text(b), "beta": beta_text(b), "pieces": len(own),
                      "degrees": deg, "locus": ("on " + " and ".join(common)) if common and len(common) <= 2 and deg > 0
                      else "", "symbol": "circle" if p.get("vertices") == order else "diamond"})
    cnt = Counter(t["id"] for t in types)
    seen = Counter()
    for t in types:
        if cnt[t["id"]] > 1:
            seen[t["id"]] += 1
            t["id"] = f"{t['id']}_{seen[t['id']]}"
    tid = {t["key"]: t["id"] for t in types}
    swap = group == "contic"
    for t in types:
        ks = set()
        for f in by_key[t["key"]]:
            if f["dim"] == t["degrees"]:
                ks |= {k for k in range(4) if f["signs"][k] == 0}
                if swap:
                    ks |= {3 - k for k in range(4) if f["signs"][k] == 0}
        t["mirrors"] = sorted(ks)
    deg_of = {t["key"]: t["degrees"] for t in types}
    images = [lambda b: b] + ([lambda b: b[::-1]] if swap else [])
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
                b = im(np.asarray(f["point"], float))
                points.append({"id": t, "q": r5(chart(b)), "beta": beta_text(b), "seed": seed_text(b)})
    uniq = []
    for p in points:
        if not any(p["id"] == u["id"] and np.allclose(p["q"], u["q"], atol=1e-6) for u in uniq):
            uniq.append(p)
    points = uniq

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
    dim_of = {t["id"]: t["dim"] for t in types}
    for t in types:
        t["bounds"] = sorted(bounds[t["id"]], key=lambda i: (order_[dim_of[i]], i))
    sep = {t["id"]: set() for t in types}
    for t in types:
        for b in t["bounds"]:
            sep[b].add(t["id"])
    deg_id = {t["id"]: t["degrees"] for t in types}
    for t in types:
        t["separates"] = sorted(i for i in sep[t["id"]] if deg_id[i] == t["degrees"] + 1)
    cell = {f["signs"]: f for f in faces if f["dim"] == 3}

    def region_at(b):
        b = np.asarray(b, float)
        if b.min() < 0:
            return None
        if not swap:
            return tid[next(f["key"] for f in faces if f["dim"] == 3)]
        bb = b if b[0] >= b[3] else b[::-1]
        f = cell.get(tuple(int(np.sign(np.dot(p, bb))) for p in planes))
        return tid[f["key"]] if f else None

    # example seeds: walls, lines, points at their most even seed (where it is inside the shape; else its central
    # seed), regions at their central seed; every piece of the shape
    ex, how = {}, {}
    if group == "contic":
        O = {r["name"]: r for r in json.load(open(os.path.join(CON, "optimized_seeds.json")))["shapes"]}
        cen = {r["name"]: r["centre"] for r in json.load(open(os.path.join(CON, "central_seeds.json")))["shapes"]}
        for t in types:
            nm = t["id"]
            edge = t["dim"] != "region" and O[nm]["status"].startswith("infimum")
            b = np.clip(np.asarray(cen[nm] if t["dim"] == "region" or edge else O[nm]["off_seed"], float), 0, None)
            ex[nm] = [b] + ([b[::-1]] if not np.allclose(b, b[::-1]) else [])
            how[nm] = "central" if t["dim"] == "region" else "edge" if edge else "even"
    elif group in ("icoic", "kiicoic"):
        I = {r["name"]: r for r in json.load(open(os.path.join(CON, "icoic_shapes.json")))["shapes"]}
        for t in types:
            fs = [f for f in by_key[t["key"]] if f["dim"] == t["degrees"]]
            if t["dim"] == "region":
                ex[t["id"]] = [np.array(I["gidpii"]["central"] if group == "icoic" else [.25] * 4, float)]
                how[t["id"]] = "central" if group == "icoic" else "uniform"
            else:
                ex[t["id"]] = [np.array([1.0 if k + 1 in f["ring"] else 0.0 for k in range(4)]) for f in fs]
                how[t["id"]] = "uniform"
    else:
        for t in types:
            fs = [f for f in by_key[t["key"]] if f["dim"] == t["degrees"]]
            rs = cand.get(t["id"].split("_")[0], [])
            got = []
            for f in fs:                        # (each piece: the listed optimum or central seed lying in it)
                pick = None
                for r in rs:
                    for kind_, b in (("even", r["optimized"]), ("central", r["central"])):
                        if kind_ == "even" and (t["dim"] == "region" or not r["inside"]):
                            continue
                        b = np.clip(np.asarray(b, float), 0, None)
                        if all((s == 0) == (abs(v) < 1e-9) for s, v in zip(f["signs"], b)):
                            pick = pick or (b, kind_)
                if pick is None:
                    pick = (np.asarray(f["point"], float), "face")
                got.append(pick)
            ex[t["id"]] = [b for b, _ in got]
            how[t["id"]] = got[0][1]
    chk = classify([b for t in types for b in ex[t["id"]]], sub)
    k = 0
    examples = []
    for t in types:
        keep = []
        for b in ex[t["id"]]:
            if chk[k][1] == t["key"]:
                keep.append(b)
            k += 1
        if not keep:
            raise SystemExit(f"{group}: no example seed of {t['id']} is the shape")
        for b in keep:
            examples.append({"id": t["id"], "q": r5(chart(b)), "beta": beta_text(b), "seed": seed_text(b)})
        t["seed"], t["beta"] = seed_text(keep[0]), beta_text(keep[0])
        h = how[t["id"]]
        t["example"] = ("the point" if t["dim"] == "point" else
                        "central seed (the deepest point of the region)" if h == "central" and t["dim"] == "region" else
                        "central seed (its most even seed is on its edge)" if h in ("edge", "central") else
                        "uniform seed (all edges equal)" if h == "uniform" else
                        "most even seed (least longest / shortest edge)" if h == "even" else "a seed in the face")
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
    for r, sh in shells.items():
        keep = []
        for e, ns in sh["edges"].items():
            flat = len(ns) == 2 and abs(abs(ns[0] @ ns[1]) / np.linalg.norm(ns[0]) / np.linalg.norm(ns[1]) - 1) < 1e-6
            if not flat:
                keep.append([list(e[0]), list(e[1])])
        sh["edges"] = keep
    for t in types:
        t["where"] = t["dim"] + (f" ({t['pieces']} pieces)" if t["pieces"] > 1 else "")
    types.sort(key=lambda t: (order_[t["dim"]], -(t.get("vertices") or 0), t["id"]))
    out = {"group": group, "order": order, "domain": domain(), "types": types, "walls": walls, "lines": lines,
           "points": points, "examples": examples, "shells": shells}
    path = os.path.join(HERE, f"f4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({dict(Counter(t['dim'] for t in types))}), {len(walls)} wall pieces, "
          f"{len(lines)} line pieces, {len(points)} points, {len(examples)} example seeds; "
          f"unnamed: {[t['id'] for t in types if t['id'] not in FULL and t['id'].split('_')[0] not in FULL]}")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["icoic", "kiicoic", "contic", "icopyritic", "toxitic"]:
        build(g)
