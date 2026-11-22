"""The 3-sphere maps of kidecaic [[3,3,3]]+ and iodecaic [[3,3,3]+] on the A4 chamber (a4_map_kidecaic.json,
a4_map_iodecaic.json), for the A4 domain viewer: the same format as a4_map.py's.

Under these subgroups of decaic the walls are curved (quadrics to quartics in beta: decaic_atlas/snub_map.py), so the
map is drawn from what the atlas found and is checked on a grid:

  * the shapes: the decaic page's list of each group's own shapes (the named ones of decaic_atlas/decaic_snub_shapes.md,
    the atlas's new ones in new_shapes.json with their given names, the crossing lines) and the line on the swap's axis
    (snad, gippid); each seed classified (decaic_atlas/group.py, DECAIC_SUB = ki or io);
  * a grid of seeds over the chamber (beta = lattice / N, nudged off the rational walls), each classified, half of them
    (beta_1 >= beta_4) and the rest by the swap, which maps the group to itself;
  * the walls: each surface the map found between two regions (snub_map_<sub>.json), meshed where it changes sign in a
    grid tetrahedron whose seeds are those two regions (marching tetrahedra), in the colour of the shape on it;
  * the lines: the swap's axis where its shape holds, and each crossing line traced along its two surfaces;
  * on the mirrors a seed's images are the decaic ones, so the mirrors carry decaic's shapes there (a4_map_decaic.json);
  * example seeds: the most even (decaic_atlas/optimize_seeds.py), or the central seed (central_seeds.py) for a region
    or where the most even seed is on the shape's edge, each checked to be its shape; and each region's extent, the
    walls and mirror patches round it.

    python a4_snub.py [kidecaic] [iodecaic]
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEC = os.path.join(ROOT, "decaic_atlas")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pages"))
import a4_map as M  # noqa: E402

SUB = {"kidecaic": "ki", "iodecaic": "io"}
AXIS_LINE = {"kidecaic": "snad", "iodecaic": "gippid"}
N = int(os.environ.get("A4_GRID", "36"))
NUDGE = 1e-5 * np.array([0.41421356, -0.26794919, 0.17157288, -0.31783725])


def classify_many(points, sub, workers=4, tol=1e-9):
    """(signature, key) of each beta, in a few processes at once."""
    points = [list(map(float, p)) for p in points]
    if not points:
        return []
    k = max(1, -(-len(points) // workers))
    chunks = [points[i:i + k] for i in range(0, len(points), k)]
    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(lambda c: M.classify(c, sub, tol), chunks))
    return [x for p in parts for x in p]


def full_names():
    out = {}
    for line in open(os.path.join(ROOT, "pages", "decaic_source.txt")):
        m = re.match(r"(\w+) - (?:\([^)]*\)\.?\s*)?(?:the |The )?(.+?)(?:\.\s| has | is |,)", line)
        if m:
            out.setdefault(m.group(1).lower(), m.group(2).strip())
    return out


def own_shapes(group):
    """The group's own shapes as the decaic page lists them: name, full name, kind, degrees, seed."""
    import make_decaic_page as P
    from decaic_given_names import GIVEN, RENAMED
    sub = SUB[group]
    full = full_names()
    out = []
    title = {"ki": "## Kidecaic: own shapes", "io": "## Iodecaic: own shapes"}[sub]
    for n, b in P.own_named(os.path.join(DEC, "decaic_snub_shapes.md"), title).items():
        short, fname = RENAMED.get(n.capitalize(), (n.capitalize(), full.get(n, "")))
        kind = None
        out.append({"atlas": n, "id": short.lower(), "name": fname, "kind": kind, "seed": b})
    for r in json.load(open(os.path.join(ROOT, "new_shapes", "new_shapes.json")))["shapes"]:
        if r["group"] == group:
            g = GIVEN.get(r["name"])
            out.append({"atlas": r["name"], "id": (g[0] if g else P.display(r["name"])).lower().replace(" ", "_"),
                        "name": g[1] if g else "", "kind": r["kind"], "seed": r["beta"]})
    try:
        for t in json.load(open(os.path.join(DEC, f"crossings_{sub}.json")))["shapes"]:
            stem = os.path.basename(t["seed_off"])[:-len("_seed.off")]
            g = GIVEN.get(stem)
            out.append({"atlas": stem, "id": (g[0] if g else P.display(stem)).lower().replace(" ", "_"),
                        "name": g[1] if g else "", "kind": t["kind"], "seed": t["seed"], "locus": t["locus"]})
    except FileNotFoundError:
        pass
    O = {r["name"]: r for r in json.load(open(os.path.join(DEC, f"optimized_seeds_{sub}.json")))["shapes"]}
    a = AXIS_LINE[group]
    out.append({"atlas": a, "id": a, "name": full.get(a, ""), "kind": "line", "seed": O[a]["start"], "axis": True})
    return out


def kuhn_tets(n):
    """The grid's tetrahedra: each cube of (i, j, k) cut into six (Kuhn), kept where all four betas are >= 0."""
    perms = list(itertools.permutations(range(3)))
    out = []
    for i in range(n):
        for j in range(n - i):
            for k in range(n - i - j):
                base = np.array([i, j, k])
                for p in perms:
                    v = [base.copy()]
                    for ax in p:
                        nxt = v[-1].copy()
                        nxt[ax] += 1
                        v.append(nxt)
                    if all(x.sum() <= n for x in v):
                        out.append([tuple(x) for x in v])
    return out


def beta_of(v, n):
    i, j, k = v
    return np.array([i, j, k, n - i - j - k], float) / n


def march(tet, fv):
    """The piece of the surface f = 0 in a tetrahedron (corners' betas, f there): a triangle, two or none."""
    pos = [x for x, f in zip(tet, fv) if f > 0]
    neg = [x for x, f in zip(tet, fv) if f <= 0]
    fp = [f for f in fv if f > 0]
    fn = [f for f in fv if f <= 0]
    if not pos or not neg:
        return []
    cut = lambda a, fa, b, fb: a + (b - a) * (fa / (fa - fb))
    P = [cut(a, fa, b, fb) for a, fa in zip(pos, fp) for b, fb in zip(neg, fn)]
    if len(P) == 3:
        return [P]
    # (two on each side: a quadrilateral, the cuts a1b1, a1b2, a2b2, a2b1 in order)
    return [[P[0], P[1], P[3]], [P[0], P[3], P[2]]]


def surface_fn(text):
    import sympy as sp
    B = sp.symbols("b1 b2 b3 b4")
    return sp.lambdify(B, sp.sympify(text), "numpy")


def trace_line(seed, locus, key, sub, steps=200, h=2e-3):
    """A crossing line: from its seed both ways along its two surfaces (Newton back onto both each step), while its
    polychoron holds (checked every few steps)."""
    import sympy as sp
    B = sp.symbols("b1 b2 b3 b4")
    E = [sp.sympify(t) for t in locus]
    fs = [sp.lambdify(B, e, "numpy") for e in E]
    gs = [[sp.lambdify(B, sp.diff(e, v), "numpy") for v in B] for e in E]

    def J(x):
        return np.array([[float(g(*x)) for g in G_] for G_ in gs] + [np.ones(4)])

    def proj(x):
        for _ in range(40):
            r = np.array([f(*x) for f in fs] + [x.sum() - 1], float)
            st = np.linalg.lstsq(J(x), r, rcond=None)[0]
            x = x - st
            if np.abs(st).max() < 1e-15:
                break
        return x
    x0 = proj(np.asarray(seed, float) / np.sum(seed))
    out = []
    for sgn in (1, -1):
        x, pts, prev = x0.copy(), [], None
        t = np.linalg.svd(J(x))[2][-1] * sgn
        for s in range(steps):
            tn = np.linalg.svd(J(x))[2][-1]
            tn = tn if tn @ t > 0 else -tn
            y = proj(x + h * tn)
            if y.min() < 0:
                break
            pts.append(y)
            x, t = y, tn
        # (keep while the shape holds)
        lab = classify_many(pts, sub)
        keep = []
        for p_, (_, k) in zip(pts, lab):
            if k != key:
                break
            keep.append(p_)
        out.append(keep)
    return out[1][::-1] + [x0] + out[0]


def build(group):
    sub = SUB[group]
    order = 120
    S = own_shapes(group)
    lab = classify_many([s["seed"] for s in S], sub)
    for s, (sig, key) in zip(S, lab):
        s["sig"], s["key"] = sig, key
    by_key = {s["key"]: s for s in S}
    # the grid: every interior lattice seed (nudged), classified on the half beta_1 >= beta_4, the rest by the swap
    pts = [(i, j, k) for i in range(N + 1) for j in range(N + 1 - i) for k in range(N + 1 - i - j)]
    inner = [v for v in pts if min(beta_of(v, N)) > 0]
    half = []
    for v in inner:
        b = beta_of(v, N)
        half.append(tuple(b if b[0] >= b[3] else b[::-1]))
    uniq = sorted(set(half))
    L = dict(zip(uniq, classify_many([np.asarray(b) + NUDGE for b in uniq], sub)))
    label = {v: L[h][1] for v, h in zip(inner, half)}
    grid_keys = Counter(label.values())
    print(group, "grid", len(inner), "seeds:", len(grid_keys), "polychora", flush=True)
    unknown = [k for k in grid_keys if k not in by_key]
    if unknown:
        print("  grid polychora not in the list:", unknown, flush=True)
    regions = {k for k in grid_keys if k in by_key}
    for s in S:
        if s["kind"] is None:
            s["kind"] = "region" if s["key"] in regions else "wall"
    DIM = {"region": 3, "wall": 2, "line": 1, "point": 0}
    types = []
    for s in S:
        p = M.parse_sig(s["sig"]) or {}
        types.append({"id": s["id"], "name": s["name"], "dim": s["kind"], "group": s["kind"] + "s", "key": s["key"],
                      "sig": s["sig"], **p, "degrees": DIM[s["kind"]], "pieces": 1, "locus": "", "bounds": [],
                      "separates": [], "atlas": s["atlas"], "symbol": "circle"})
    tid = {t["key"]: t["id"] for t in types}
    # the walls: each wall shape on every surface of the map through its seed (and the fold b1 = b4), meshed where the
    # surface crosses a grid tetrahedron (marching tetrahedra, on the half beta_1 >= beta_4, then the swap), each
    # triangle kept if its centre, put on the surface, is that wall's polychoron (hull tolerance 1e-7, as the map's
    # walls); the regions round it are its tetrahedra's grid seeds
    import sympy as sp
    SM = json.load(open(os.path.join(DEC, f"snub_map_{sub}.json")))
    Bs = sp.symbols("b1 b2 b3 b4")
    surf = []
    for text in SM["surfaces"] + ["b1 - b4"]:
        e = sp.sympify(text)
        surf.append((text, sp.lambdify(Bs, e, "numpy"), [sp.lambdify(Bs, sp.diff(e, v), "numpy") for v in Bs]))
    tets = [t_ for t_ in kuhn_tets(N) if np.mean([beta_of(v, N) for v in t_], 0)[0] >= np.mean([beta_of(v, N) for v in t_], 0)[3] - 1e-12]
    walls, shells_tri, sep = [], defaultdict(list), defaultdict(set)
    unmeshed = []
    for t in types:
        if t["dim"] != "wall":
            continue
        b0 = np.asarray(by_key[t["key"]]["seed"], float)
        b0 = b0 / b0.sum()
        mine = [(tx, f, g) for tx, f, g in surf
                if abs(f(*b0)) / (np.linalg.norm([gg(*b0) for gg in g]) + 1e-300) < 1e-6]
        cand = []
        for tx, f, g in mine:
            for tet in tets:
                Bt = [beta_of(v, N) for v in tet]
                fv = [float(f(*x)) for x in Bt]
                for tri in march(Bt, fv):
                    c = np.mean(tri, 0)
                    for _ in range(6):                  # (onto the surface)
                        gr = np.array([float(gg(*c)) for gg in g])
                        gr = gr - gr.mean()
                        c = c - float(f(*c)) * gr / (gr @ gr)
                    cand.append((tri, c, tet))
        if not cand:
            unmeshed.append(t["id"])
            continue
        labs = classify_many([c for _, c, _ in cand], sub, tol=1e-7)
        kept = [(tri, tet) for (tri, c, tet), (_, k) in zip(cand, labs) if k == t["key"]]
        if not kept:
            unmeshed.append(t["id"])
            continue
        for im in (lambda x: x, lambda x: x[::-1]):
            V, F = [], []
            for tri, tet in kept:
                o = len(V)
                V += [M.r5(M.chart(im(q))) for q in tri]
                F.append([o, o + 1, o + 2])
            walls.append({"id": t["id"], "verts": V, "faces": F})
        for tri, tet in kept:
            for r in {label.get(v) for v in tet} - {None, t["key"]}:
                if r in tid:
                    sep[t["id"]].add(tid[r])
                    for im in (lambda x: x, lambda x: x[::-1]):
                        shells_tri[tid[r]].append(([M.r5(M.chart(im(q))) for q in tri], [[0, 1, 2]]))
        print("  wall", t["id"], len(kept), "of", len(cand), "triangles on", [m[0][:24] for m in mine], flush=True)
    if unmeshed:
        print("  walls not meshed:", unmeshed, flush=True)
    for t in types:
        t["separates"] = sorted(sep.get(t["id"], ()))
        t["pieces"] = 1
    # each region on the mirrors: the grid triangles there whose tetrahedron's seeds are that region
    for tet in kuhn_tets(N):
        for k in range(4):
            on = [v for v in tet if (beta_of(v, N)[k] == 0)]
            if len(on) != 3:
                continue
            labs = {label.get(v) for v in tet if v not in on} - {None}
            if len(labs) == 1:
                r = labs.pop()
                if r in tid:
                    V = [M.r5(M.chart(beta_of(v, N))) for v in on]
                    shells_tri[tid[r]].append((V, [[0, 1, 2]]))
    shells = {}
    for r, parts in shells_tri.items():
        V, F, E = [], [], Counter()
        for v_, f_ in parts:
            o = len(V)
            V += v_
            F += [[x + o for x in f] for f in f_]
        shells[r] = {"verts": V, "faces": F, "edges": []}
    # the lines: the swap's axis where its shape holds, and the crossing lines traced
    lines = []
    for s in S:
        if s["kind"] != "line":
            continue
        if s.get("axis"):
            ts = np.linspace(0.002, 0.498, 249)
            P = [np.array([t, 0.5 - t, 0.5 - t, t]) for t in ts]
            labs = classify_many([p + NUDGE * 0 for p in P], sub)
            run = []
            for p_, (_, k) in zip(P, labs):
                if k == s["key"]:
                    run.append(p_)
                elif run:
                    lines.append({"id": s["id"], "pts": [M.r5(M.chart(q)) for q in run]})
                    run = []
            if run:
                lines.append({"id": s["id"], "pts": [M.r5(M.chart(q)) for q in run]})
        elif s.get("locus"):
            P = trace_line(s["seed"], s["locus"], s["key"], sub)
            for im in (lambda x: x, lambda x: x[::-1]):
                lines.append({"id": s["id"], "pts": [M.r5(M.chart(im(q))) for q in P]})
    # the mirrors: decaic's shapes there (a seed on a mirror has the decaic images)
    D = json.load(open(os.path.join(HERE, "a4_map_decaic.json")))
    dom = M.domain()
    Vd = np.array(dom["verts"])
    planes = []
    for f in dom["faces"]:
        P_ = Vd[f["loop"]]
        n = np.cross(P_[1] - P_[0], P_[2] - P_[0])
        planes.append((n / np.linalg.norm(n), P_[0]))
    on_mirror = lambda Q: any(all(abs((np.asarray(q) - p0) @ n) < 1e-6 for q in Q) for n, p0 in planes)
    mt = {}
    for w in D["walls"]:
        if on_mirror(w["verts"]):
            walls.append(w)
            mt[w["id"]] = "wall"
    for l_ in D["lines"]:
        if on_mirror(l_["pts"]):
            lines.append(l_)
            mt[l_["id"]] = "line"
    points = [p for p in D["points"] if on_mirror([p["q"]])]
    for p in points:
        mt[p["id"]] = "point"
    examples = [e for e in D["examples"] if e["id"] in mt and on_mirror([e["q"]])]
    for t in D["types"]:
        if t["id"] in mt and t["id"] not in tid.values():
            u = {k: v for k, v in t.items() if k not in ("color",)}
            u.update(dim=mt[t["id"]], group=mt[t["id"]] + "s", bounds=[], separates=[],
                     where=f"on the mirrors (shared with decaic: a seed there has the decaic images)", locus="")
            types.append(u)
    # example seeds of the group's own shapes
    O = json.load(open(os.path.join(DEC, f"optimized_seeds_{sub}.json")))["shapes"]
    C = {r["name"]: r for r in json.load(open(os.path.join(DEC, f"central_seeds_{sub}.json")))["shapes"]}
    ok_ = classify_many([r["start"] for r in O], sub)
    ex_of = {}
    for r, (_, k) in zip(O, ok_):
        edge = not r["status"].startswith("minimum inside") and r["kind"] != "point"
        use_c = r["kind"] == "region" or edge
        b = C[r["name"]]["centre"] if use_c and r["name"] in C else r["off_seed"]
        ex_of[k] = (b, "central seed (the deepest point of the region)" if r["kind"] == "region" else
                    "central seed (its most even seed is on its edge)" if use_c else "most even seed (least longest / shortest edge)")
    own = [t for t in types if t["key"] in by_key and "where" not in t]
    cands = [(t, *ex_of.get(t["key"], (by_key[t["key"]]["seed"], "the atlas's seed"))) for t in own]
    chk = classify_many([b for _, b, _ in cands], sub)
    for (t, b, how), (_, k) in zip(cands, chk):
        if k != t["key"]:                       # (not the shape: the atlas's own seed instead)
            b, how = by_key[t["key"]]["seed"], "the atlas's seed"
        b = np.clip(np.asarray(b, float), 0, None)
        t["seed"], t["beta"], t["example"] = M.seed_text(b), M.beta_text(b), how
        for q in [b] + ([b[::-1]] if not np.allclose(b, b[::-1]) else []):
            examples.append({"id": t["id"], "q": M.r5(M.chart(q)), "beta": M.beta_text(q), "seed": M.seed_text(q)})
        t["where"] = t["dim"] + (" (on the swap's axis)" if by_key[t["key"]].get("axis") else
                                 " (where two of the map's walls cross)" if by_key[t["key"]].get("locus") else "")
        t["samples"] = grid_keys.get(t["key"], 0)
        if t["dim"] == "region":
            t["where"] = f"region ({100 * grid_keys.get(t['key'], 0) / len(inner):.1f}% of the grid)"
    for t in types:
        t.setdefault("mirrors", [])
    order_t = {"region": 0, "wall": 1, "line": 2, "point": 3}
    types.sort(key=lambda t: (order_t[t["dim"]], "mirrors" in t.get("where", ""), t["id"]))
    out = {"group": group, "order": order, "domain": dom, "types": types, "walls": walls, "lines": lines,
           "points": points, "examples": examples, "shells": shells, "grid": N, "unlisted": unknown,
           "unmeshed": unmeshed}
    path = os.path.join(HERE, f"a4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({Counter(t['dim'] for t in types)}), {len(walls)} wall pieces, "
          f"{len(lines)} line pieces, {len(examples)} example seeds, {len(shells)} region shells")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["kidecaic", "iodecaic"]:
        build(g)
