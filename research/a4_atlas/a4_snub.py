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
    O = {r["name"]: r for r in json.load(open(os.path.join(DEC, f"optimized_seeds_{sub}.json")))["shapes"]}
    title = {"ki": "## Kidecaic: own shapes", "io": "## Iodecaic: own shapes"}[sub]
    for n, b in P.own_named(os.path.join(DEC, "decaic_snub_shapes.md"), title).items():
        short, fname = RENAMED.get(n.capitalize(), (n.capitalize(), full.get(n, "")))
        kind = O[n]["kind"] if n in O else None
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
    a = AXIS_LINE[group]
    out.append({"atlas": a, "id": a, "name": full.get(a, ""), "kind": "line", "seed": O[a]["start"], "axis": True})
    return out


def kuhn_tets(n):
    """The grid's tetrahedra, tiling the chamber exactly: the Kuhn cubes in cumulative coordinates
    (c1, c2, c3) = (i, i + j, i + j + k), 0 <= c1 <= c2 <= c3 <= n, kept where every corner is in order, then back to
    (i, j, k)."""
    perms = list(itertools.permutations(range(3)))
    out = []
    for a in range(n):
        for b in range(a, n):
            for c in range(b, n):
                base = np.array([a, b, c])
                for p in perms:
                    v = [base.copy()]
                    for ax in p:
                        nxt = v[-1].copy()
                        nxt[ax] += 1
                        v.append(nxt)
                    if all(x[0] <= x[1] <= x[2] <= n for x in v):
                        out.append([(int(x[0]), int(x[1] - x[0]), int(x[2] - x[1])) for x in v])
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
    """The page's map from the exact pieces (a4_exact.py: a4_exact_<group>.pkl)."""
    import pickle
    sub = SUB[group]
    # (a4_pieces.py's pieces when made: each meshed inside its exact loop, the lines carried to their ends)
    pp = os.path.join(HERE, f"a4_pieces_{group}.pkl")
    X = pickle.load(open(pp if os.path.exists(pp) else os.path.join(HERE, f"a4_exact_{group}.pkl"), "rb"))
    S = X["shapes"]
    by_key = {s["key"]: s for s in S}
    types = []
    for s in S:
        p = M.parse_sig(s["sig"]) or {}
        kind = s["kind"] or "region"
        types.append({"id": s["id"], "name": s["name"], "dim": kind, "group": kind + "s", "key": s["key"],
                      "sig": s["sig"], **p, "degrees": {"region": 3, "wall": 2, "line": 1, "point": 0}[kind],
                      "pieces": 0, "locus": "", "bounds": [], "separates": [], "atlas": s["atlas"], "symbol": "circle"})
    tid = {t["key"]: t["id"] for t in types}
    wall_keys = {t["key"] for t in types if t["dim"] == "wall"}
    region_keys = {t["key"] for t in types if t["dim"] == "region"}
    chartv = lambda V: [M.r5(M.chart(np.clip(np.asarray(v, float), 0, None))) for v in V]
    CELL = float(os.environ.get("A4_MESH_CELL", "0.004"))
    Pexact = np.array([M.chart(np.clip(np.asarray(q, float), 0, None)) for q in X["points"]]).reshape(-1, 3)
    from scipy.spatial import cKDTree
    Ptree = cKDTree(Pexact) if len(Pexact) else None

    def simplify(V, T):
        """The mesh in the chart, its inner vertices merged within a cell (CELL, in the chart's units) and the
        triangles left flat dropped; the piece's outline keeps its vertices, exact on the traced lines."""
        V = np.array([M.chart(np.clip(np.asarray(v, float), 0, None)) for v in V])
        ec = Counter(tuple(sorted(e)) for t in T for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
        rim = {a for e, c in ec.items() if c == 1 for a in e}
        rep, cells = {}, {}
        for i, v in enumerate(V):
            if i in rim:                         # (the outline: merged in a finer cell, the map's points kept)
                if Ptree is not None and Ptree.query(v)[0] < 1e-9:
                    rep[i] = i
                else:
                    rep[i] = cells.setdefault(("rim",) + tuple(np.floor(v / (CELL / 2)).astype(int)), i)
            else:
                rep[i] = cells.setdefault(tuple(np.floor(v / CELL).astype(int)), i)
        T2 = [[rep[a] for a in t] for t in T]
        T2 = [t for t in T2 if len(set(t)) == 3]
        used = sorted({a for t in T2 for a in t})
        new = {a: k for k, a in enumerate(used)}
        return {"verts": [M.r5(V[a]) for a in used], "faces": [[new[a] for a in t] for t in T2]}
    # walls: the exact pieces that are wall shapes; each region's extent: the pieces round it (walls on either side,
    # mirror patches on its side)
    walls, shells_tri, sep, unlisted = [], defaultdict(list), defaultdict(set), Counter()
    # (a4_exact fills a piece's gap at an exact point near its outline; that is right only where the point is a corner
    #  left uncovered: where another piece of the same surface covers it, the fan would lie over that piece. Such
    #  fills are dropped here)
    def own_tris(F):
        fl = set(F.get("filled", []))
        return np.array([t for t in F["mesh"]["faces"] if not fl.intersection(t)], int).reshape(-1, 3)

    by_surf = defaultdict(list)
    for fi, F in enumerate(X["faces"]):
        by_surf[F["surface"]].append(fi)
    dropped = 0
    for si_, fis in by_surf.items():
        if not any(X["faces"][fi].get("filled") for fi in fis):
            continue
        A, E0, E1, own = [], [], [], []              # (every piece's own triangles on this surface)
        for fi in fis:
            V = np.asarray(X["faces"][fi]["mesh"]["verts"], float)
            T = own_tris(X["faces"][fi])
            if len(T):
                A.append(V[T[:, 0]]); E0.append(V[T[:, 1]] - V[T[:, 0]]); E1.append(V[T[:, 2]] - V[T[:, 0]])
                own += [fi] * len(T)
        A, E0, E1, own = np.concatenate(A), np.concatenate(E0), np.concatenate(E1), np.array(own)
        a_, b_, c_ = (E0 * E0).sum(1), (E0 * E1).sum(1), (E1 * E1).sum(1)
        den = a_ * c_ - b_ * b_ + 1e-300
        cover = {}

        def covered_by(p):
            k = tuple(np.round(p, 12))
            if k not in cover:
                w = p - A
                d, e = (E0 * w).sum(1), (E1 * w).sum(1)
                u, v = (c_ * d - b_ * e) / den, (a_ * e - b_ * d) / den
                ok = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
                Q = A[ok] + u[ok, None] * E0[ok] + v[ok, None] * E1[ok]
                cover[k] = set(own[ok][np.linalg.norm(Q - p, axis=1) < 1e-7].tolist())
            return cover[k]
        for fi in fis:
            F = X["faces"][fi]
            fl = F.get("filled", [])
            if not fl:
                continue
            V = np.asarray(F["mesh"]["verts"], float)
            bad = {i for i in fl if covered_by(V[i]) - {fi}}
            if bad:
                F["mesh"] = dict(F["mesh"], faces=[t for t in F["mesh"]["faces"] if not bad.intersection(t)])
                dropped += len(bad)
    print("  fills dropped (the point covered by another piece):", dropped, flush=True)
    for F in X["faces"]:
        mesh = ({"verts": chartv(F["mesh"]["verts"]), "faces": [list(map(int, t)) for t in F["mesh"]["faces"]]}
                if F.get("exact_loop") else simplify(F["mesh"]["verts"], F["mesh"]["faces"]))
        on_mirror = F["surface"] < 4
        if F["key"] in wall_keys and not on_mirror:
            walls.append({"id": tid[F["key"]], **mesh})
            t = next(t for t in types if t["key"] == F["key"])
            t["pieces"] += 1
            for r in F["sides"]:
                if r in region_keys:
                    shells_tri[tid[r]].append(len(walls) - 1)     # (a wall piece: by its index in walls)
                    sep[tid[F["key"]]].add(tid[r])
        elif on_mirror and F["sides"][0] in region_keys:
            shells_tri[tid[F["sides"][0]]].append(mesh)
        elif not on_mirror and F["key"] not in region_keys and F["key"] not in wall_keys and np.min(F["inner"]) > 0:
            unlisted[F["key"]] += 1
    shells = {}
    for r, parts in shells_tri.items():
        V, Fa, W = [], [], []
        for m in parts:
            if isinstance(m, int):
                W.append(m)
                continue
            o = len(V)
            V += m["verts"]
            Fa += [[a + o for a in f] for f in m["faces"]]
        shells[r] = {"verts": V, "faces": Fa, "walls": sorted(set(W)), "edges": []}
    for t in types:
        t["separates"] = sorted(sep.get(t["id"], ()))
    # lines: the traced segments that are line shapes; edges: the others (where walls meet, or meet a mirror), drawn
    # thin; a chamber edge (on two mirrors) is the domain's own
    lines, edges = [], []
    # (the swap's axis is drawn once, exactly, as its line shape: many pairs of surfaces through it traced
    #  overlapping bits of it)
    def on_axis(P):
        P = np.asarray(P)
        return np.abs(P[:, 0] - P[:, 3]).max() < 1e-6 and np.abs(P[:, 1] - P[:, 2]).max() < 1e-6
    ax = [s_ for s_ in S if s_.get("axis")]
    if ax:
        tv = np.linspace(0, 0.5, 41)
        lines.append({"id": ax[0]["id"], "pts": chartv(np.stack([tv, 0.5 - tv, 0.5 - tv, tv], 1)),
                      "ends": ("axis", "axis")})
    for sg in X["segments"]:
        if ax and on_axis(sg["pts"]):
            continue
        L_ = np.linalg.norm(np.diff(np.asarray(sg["pts"]), axis=0), axis=1).sum()
        if L_ < 1e-5:
            continue
        pts = chartv(sg["pts"][:: max(1, len(sg["pts"]) // 40)] + [sg["pts"][-1]])
        if sg["key"] in tid and by_key[sg["key"]]["kind"] == "line":
            lines.append({"id": tid[sg["key"]], "pts": pts, "ends": tuple(sg["ends"])})
        elif len([k for k in sg["on"] if k < 4]) < 2 and sg["key"] not in region_keys:
            # (only where walls meet, or meet a mirror: a crossing inside a region, of surfaces carrying no wall
            #  there, is not drawn)
            edges.append({"pts": pts, "hover": "an edge of the map: " + " and ".join(
                X["surfaces"][k] for k in sorted(sg["on"])[:3])})
    # (a line's pieces: its runs, segments joined at shared points; a line is cut wherever any surface crosses it,
    #  but goes on as the same shape there)
    for t in types:
        if t["dim"] == "line":
            par = {}

            def root(a):
                while par.setdefault(a, a) != a:
                    a = par[a]
                return a
            for l_ in lines:
                if l_["id"] == t["id"]:
                    a, b = l_["ends"]
                    par[root(a)] = root(b)
            t["pieces"] = len({root(a) for a in list(par)})
    for l_ in lines:
        l_.pop("ends", None)
    # the mirrors: decaic's shapes there (a seed on a mirror has the decaic images)
    D = json.load(open(os.path.join(HERE, "a4_map_decaic.json")))
    dom = M.domain()
    Vd = np.array(dom["verts"])
    planes = []
    for f in dom["faces"]:
        P_ = Vd[f["loop"]]
        n = np.cross(P_[1] - P_[0], P_[2] - P_[0])
        planes.append((n / np.linalg.norm(n), P_[0]))
    on_m = lambda Q: any(all(abs((np.asarray(q) - p0) @ n) < 1e-6 for q in Q) for n, p0 in planes)
    mt = {}
    for w in D["walls"]:
        if on_m(w["verts"]):
            walls.append(w)
            mt[w["id"]] = "wall"
    for l_ in D["lines"]:
        if on_m(l_["pts"]):
            lines.append(l_)
            mt[l_["id"]] = "line"
    points = [p for p in D["points"] if on_m([p["q"]])]
    for p in points:
        mt[p["id"]] = "point"
    examples = [e for e in D["examples"] if e["id"] in mt and on_m([e["q"]])]
    for t in D["types"]:
        if t["id"] in mt and t["id"] not in tid.values():
            u = {k: v for k, v in t.items() if k not in ("color",)}
            u.update(dim=mt[t["id"]], group=mt[t["id"]] + "s", bounds=[], separates=[],
                     where="on the mirrors (shared with decaic: a seed there has the decaic images)", locus="")
            types.append(u)
    # example seeds of the group's own shapes (central for a region, or where the most even seed is on the shape's
    # edge; each checked to be its shape, else the atlas's seed)
    O = json.load(open(os.path.join(DEC, f"optimized_seeds_{sub}.json")))["shapes"]
    C = {r["name"]: r for r in json.load(open(os.path.join(DEC, f"central_seeds_{sub}.json")))["shapes"]}
    ok_ = classify_many([r["start"] for r in O], sub)
    ex_of = {}
    for r, (_, k) in zip(O, ok_):
        edge = not r["status"].startswith("minimum inside") and r["kind"] != "point"
        use_c = r["kind"] == "region" or edge or r["name"] in M.CENTRAL
        b = C[r["name"]]["centre"] if use_c and r["name"] in C else r["off_seed"]
        ex_of[k] = (b, "central seed (the deepest point of the region)" if r["kind"] == "region" else
                    "central seed (its most even seed is on its edge)" if edge else
                    "central seed (the deepest point of the shape)" if use_c else "most even seed (least longest / shortest edge)")
    own = [t for t in types if t["key"] in by_key and "mirrors" not in t.get("where", "")]
    cands = [(t, *ex_of.get(t["key"], (by_key[t["key"]]["seed"], "the atlas's seed"))) for t in own]
    chk = classify_many([b for _, b, _ in cands], sub, tol=1e-7)
    # (how much of the chamber each region fills: random seeds)
    rng = np.random.default_rng(3)
    R_ = rng.dirichlet([1, 1, 1, 1], 3000)
    frac = Counter(k for _, k in classify_many(R_, sub))
    for (t, b, how), (_, k) in zip(cands, chk):
        if k != t["key"]:
            b, how = by_key[t["key"]]["seed"], "the atlas's seed"
        b = np.clip(np.asarray(b, float), 0, None)
        t["seed"], t["beta"], t["example"] = M.seed_text(b), M.beta_text(b), how
        for q in [b] + ([b[::-1]] if not np.allclose(b, b[::-1]) else []):
            examples.append({"id": t["id"], "q": M.r5(M.chart(q)), "beta": M.beta_text(q), "seed": M.seed_text(q)})
        if t["dim"] == "region":
            pc = 100 * frac.get(t["key"], 0) / len(R_)
            t["where"] = f"region ({pc:.1f}% of the chamber)" if pc >= 0.05 else "region (under 0.1% of the chamber: thin)"
        else:
            t["where"] = t["dim"] + (f" ({t['pieces']} pieces)" if t["pieces"] > 1 else "") + (
                " (on the swap's axis)" if by_key[t["key"]].get("axis") else "")
    for t in types:
        t.setdefault("mirrors", [])
    order_t = {"region": 0, "wall": 1, "line": 2, "point": 3}
    types.sort(key=lambda t: (order_t[t["dim"]], "mirrors" in t.get("where", ""), t["id"]))
    out = {"group": group, "order": 120, "domain": dom, "types": types, "walls": walls, "lines": lines, "edges": edges,
           "points": points, "examples": examples, "shells": shells, "exact": True,
           "unlisted": {k: v for k, v in unlisted.items()}}
    path = os.path.join(HERE, f"a4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({Counter(t['dim'] for t in types)}), {len(walls)} wall pieces, "
          f"{len(lines)} line pieces, {len(edges)} edges, {len(examples)} example seeds, {len(shells)} region shells; "
          f"unlisted pieces: {dict(unlisted)}")


if __name__ == "__main__":
    for g in sys.argv[1:] or ["kidecaic", "iodecaic"]:
        build(g)
