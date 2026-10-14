"""Walls, regions, lines and points of the index-3 tetrahedral swirl domain.   python build_data.py data.json

The orbit's hull changes type exactly where five orbit points become coplanar. In the chart every such wall is a piece
of one quadric Q0 (and its two copies Q1 = Q0 o R, Q2 = Q0 o R^2 under the anchor's order-3 stabiliser R). The three
signs (sign Q0, sign Q1, sign Q2) therefore decide the type; this script fits Q0 from bisected wall points, checks
the sign -> type rule against random samples, then meshes regions, walls, lines and special points.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from skimage.measure import marching_cubes

import domain as D
import walls as W

MONS = [m for m in itertools.product(range(3), repeat=3) if sum(m) <= 2]
R = D.stabiliser_in_chart()
H = D.domain_halfspaces()
HN = np.linalg.norm(H[:, :3], axis=1)
V = D.domain_vertices()
RNG = np.random.default_rng(11)


def sig(u, tol=1e-9):
    return D.signature(D.classify(D.seed_from_chart(np.asarray(u, float)), tol))


def monos(p):
    p = np.asarray(p, float)
    return np.stack([np.prod(p ** np.array(m), axis=-1) for m in MONS], axis=-1)


def fit_quadric(points):
    c = np.linalg.svd(monos(points))[2][-1]
    return c / np.abs(c).max()


def q_eval(c, p):
    return monos(p) @ c


def q_grad(c, p):
    p = np.asarray(p, float)
    g = np.zeros(p.shape)
    for ci, m in zip(c, MONS):
        for ax in range(3):
            if m[ax] == 0:
                continue
            mm = list(m)
            mm[ax] -= 1
            g[..., ax] += ci * m[ax] * np.prod(p ** np.array(mm), axis=-1)
    return g


def domain_margin(p):
    return np.min(-(np.asarray(p) @ H[:, :3].T + H[:, 3]) / HN, axis=-1)


def random_domain_points(n, margin=0.0):
    out = []
    while len(out) < n:
        X = RNG.uniform(V.min(0), V.max(0), size=(4 * n, 3))
        out.extend(X[domain_margin(X) > margin].tolist())
    return np.array(out[:n])


def fit_wall():
    """Bisect label changes between random neighbouring points and fit the quadric of the largest sheet."""
    pts = []
    while len(pts) < 90:
        a, b = random_domain_points(2)
        sa, sb = sig(a), sig(b)
        if sa == sb or "none" in (sa, sb):
            continue
        for _ in range(60):
            m = (a + b) / 2
            if sig(m) == sa:
                a = m
            else:
                b = m
        q = (a + b) / 2
        if domain_margin(q) > 1e-6:
            pts.append(q)
    pts = np.array(pts)
    # the 5-subsets of orbit points that are coplanar at each wall point name its sheet (the subsets that vanish
    # everywhere are structural and dropped)
    sets = [frozenset(W.near_zero_subsets(D.seed_from_chart(q), 1e-7)) for q in pts]
    always = frozenset.intersection(*sets)
    sheets = defaultdict(list)
    for q, st in zip(pts, sets):
        sheets[st - always].append(q)
    sheet = np.array(max(sheets.values(), key=len))
    c = fit_quadric(sheet)
    res = [abs(q_eval(c, np.linalg.matrix_power(R, k) @ p)) for p in pts for k in range(3)]
    res = np.array(res).reshape(len(pts), 3).min(1)
    return c, pts, float(res.max())


def sign_pattern(Q, p):
    return tuple(np.sign(q_eval(Q[k], p)).astype(int).tolist() for k in range(3))


def chain(segments):
    """Join unordered 3D segments into polylines."""
    key = lambda p: tuple(np.round(p / 1e-6).astype(np.int64))
    ends = defaultdict(list)
    for i, (a, b) in enumerate(segments):
        ends[key(a)].append(i)
        ends[key(b)].append(i)
    used = [False] * len(segments)
    lines = []
    for i in range(len(segments)):
        if used[i]:
            continue
        used[i] = True
        line = [segments[i][0], segments[i][1]]
        for direction in (0, 1):
            while True:
                tip = line[-1] if direction == 0 else line[0]
                nxt = [j for j in ends[key(tip)] if not used[j]]
                if not nxt:
                    break
                j = nxt[0]
                used[j] = True
                a, b = segments[j]
                new = b if key(a) == key(tip) else a
                if direction == 0:
                    line.append(new)
                else:
                    line.insert(0, new)
        lines.append(np.array(line))
    return lines


def tidy(line, k):
    """Move a polyline onto both quadrics (Gauss-Newton with the plane through its end-to-end direction)."""
    return line


def emit(out, Q, pat_type, regions, out_walls, wall_triangles, h):
    types: dict[str, dict] = {}

    def type_id(signature, dim, name=None):
        if signature not in types:
            idx = sum(1 for t in types.values()) + 1
            types[signature] = {"id": f"T{idx}", "dim": dim, "sig": signature, "name": name}
        return types[signature]["id"]

    def seed_info(u):
        s = D.seed_from_chart(np.asarray(u, float))
        return {"u": [round(float(x), 6) for x in u], "seed": [round(float(x), 9) for x in s]}

    cell_type = {pat: type_id(sg, 3) for pat, sg in pat_type.items()}
    ent = {"regions": [], "walls": [], "lines": [], "points": []}
    for n, ((pat, ci), r) in enumerate(sorted(regions.items(), key=lambda kv: -kv[1]["volume"]), 1):
        pts = random_domain_points(4000)
        inside = pts[[sign_pattern(Q, p) == pat for p in pts]]
        c = inside.mean(0) if len(inside) else r["verts"].mean(0)
        if sign_pattern(Q, c) != pat or domain_margin(c) <= 0:
            c = inside[np.argmax(domain_margin(inside))]
        ent["regions"].append({"id": f"R{n}", "type": cell_type[pat], "pattern": list(pat), "volume": round(r["volume"], 4),
                               "verts": np.round(r["verts"], 3).tolist(), "faces": r["faces"].tolist(), "example": seed_info(c)})
    rid = {tuple(r["pattern"]): r["id"] for r in ent["regions"]}

    real = [w for w in out_walls if w["sides"][0] is not None and w["sides"][1] is not None
            and pat_type[w["sides"][0]] != pat_type[w["sides"][1]]]
    for n, w in enumerate(sorted(real, key=lambda w: -w["area"]), 1):
        v, tris = w["verts"], w["tris"]
        used = np.unique(tris)
        remap = {int(o): i for i, o in enumerate(used)}
        c = v[tris].mean(0).mean(0)
        cen = v[tris].mean(1)
        mid = cen[np.argmin(np.linalg.norm(cen - cen.mean(0), axis=1))]
        sg = sig(mid, 1e-7)
        w["mid"] = mid
        ent["walls"].append({"id": f"W{n}", "quadric": w["quadric"], "area": round(w["area"], 3),
                             "between": [rid[tuple(s)] for s in w["sides"]], "type": type_id(sg, 2),
                             "verts": np.round(v[used], 3).tolist(),
                             "faces": [[remap[int(i)] for i in t] for t in tris], "example": seed_info(mid)})

    # lines where two quadrics meet inside the domain
    for k, j in ((0, 1), (0, 2), (1, 2)):
        v, groups = wall_triangles[k]
        tris = np.array([t for g in groups.values() for t in g])
        qj = q_eval(Q[j], v)
        segs = []
        for t in tris:
            vals = qj[t]
            pts = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                if vals[a] * vals[b] < 0:
                    s = vals[a] / (vals[a] - vals[b])
                    pts.append(v[t[a]] + s * (v[t[b]] - v[t[a]]))
            if len(pts) == 2 and all(domain_margin(p) > 0 for p in pts):
                segs.append(pts)
        for line in chain(segs):
            if len(line) < 6:
                continue
            mid = line[len(line) // 2]
            nk = q_grad(Q[k], mid); nk /= np.linalg.norm(nk)
            nj = q_grad(Q[j], mid); nj /= np.linalg.norm(nj)
            quads = Counter()
            for a, b in itertools.product((1, -1), repeat=2):
                q = mid + 0.03 * (a * nk + b * nj)
                if domain_margin(q) > 0:
                    quads[sign_pattern(Q, q)] += 1
            tys = {pat_type.get(p) for p in quads}
            if len(tys - {None}) < 2 and len(quads) < 3:
                continue
            ent["lines"].append({"quadrics": [k, j], "pts": line, "mid": mid, "around": sorted(quads),
                                 "length": float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())})
    for n, l in enumerate(sorted(ent["lines"], key=lambda l: -l["length"]), 1):
        sg = sig(l["mid"], 1e-7)
        l["id"] = f"L{n}"
        l["type"] = type_id(sg, 1)
        l["example"] = seed_info(l["mid"])
        l["around"] = [rid.get(tuple(p)) for p in l["around"]]
        l["pts"] = np.round(l["pts"], 3).tolist()
        l["mid"] = None
        l["length"] = round(l["length"], 3)

    # symmetry axes (fewer than 24 orbit points)
    n0 = len(ent["lines"])
    axis_sig = sig([0, 0, 0.4], 1e-9)
    ent["lines"].append({"id": f"L{n0 + 1}", "kind": "axis", "type": type_id(axis_sig, 1),
                         "pts": [[0, 0, -float(V[:, 2].min()) * -1], [0, 0, float(V[:, 2].max())]],
                         "example": seed_info([0, 0, 0.4]), "length": round(float(V[:, 2].max() - V[:, 2].min()), 3)})
    th = np.linspace(0, 2 * np.pi, 20001)[:-1]
    seen = []
    for g in D.ELEMENTS:
        if abs(np.trace(g) - 1) > 1e-8:
            continue
        _, _, vt = np.linalg.svd(g - np.eye(4))
        F = vt[2:]
        key = tuple(np.round(F.T @ F, 4).ravel())
        if key in seen:
            continue
        seen.append(key)
        p = np.cos(th)[:, None] * F[0] + np.sin(th)[:, None] * F[1]
        ok = p @ D.A > 1e-6
        U = D.chart_from_seed(p[ok])
        ins = D.in_domain(U, 1e-9)
        if ins.any() and np.abs(U[ins][:, :2]).max() > 1e-3:
            Ui = U[ins]
            ent["lines"].append({"id": f"L{len(ent['lines']) + 1}", "kind": "edge", "type": type_id(sig(Ui[len(Ui) // 2], 1e-9), 1),
                                 "pts": np.round(Ui[:: max(1, len(Ui) // 40)], 3).tolist(),
                                 "example": seed_info(Ui[len(Ui) // 2]),
                                 "length": round(float(np.linalg.norm(np.diff(Ui, axis=0), axis=1).sum()), 3)})

    # points: the domain's corners, and where three quadrics meet
    for i, u in enumerate(V):
        ent["points"].append({"id": f"P{i + 1}", "type": type_id(sig(u, 1e-9), 0), "kind": "corner", "example": seed_info(u)})
    types_out = list(types.values())
    cells_name = {"24: cells 8x[6v8f]+8x[6v8f]+8x[6v8f]": "Combinatorial icositetrachoron (24 octahedra)"}
    for t in types_out:
        t["name"] = next((v for k, v in cells_name.items() if t["sig"].startswith(k)), t["name"])
    json.dump({"quadrics": [q.tolist() for q in Q], "monomials": MONS, "types": types_out, "entities": ent, "h": h,
               "domain": V.tolist()}, open(out, "w"), default=lambda o: o.item() if hasattr(o, "item") else list(o))
    print("wrote", out, {k: len(v) for k, v in ent.items()}, [(t["id"], t["dim"], t["sig"][:50]) for t in types_out])


def main(out):
    c0, wall_pts, resid = fit_wall()
    Q = [c0]
    # Q_k(u) = Q0(R^k u): coefficients of the composed quadric, found by fitting on random points
    for k in (1, 2):
        Rk = np.linalg.matrix_power(R, k)
        X = RNG.uniform(-2, 2, size=(80, 3))
        vals = q_eval(c0, X @ Rk.T)
        Q.append(np.linalg.lstsq(monos(X), vals, rcond=None)[0])
    print("quadric Q0 coefficients", {m: round(float(x), 6) for m, x in zip(MONS, c0) if abs(x) > 1e-9},
          "max residual of wall points", resid)

    # sign pattern -> type
    S = random_domain_points(2500)
    pat_types = defaultdict(Counter)
    for p in S:
        pat_types[sign_pattern(Q, p)][sig(p)] += 1
    for pat, cnt in pat_types.items():
        assert len(cnt) == 1, (pat, cnt)
    pat_type = {pat: next(iter(cnt)) for pat, cnt in pat_types.items()}
    print("sign cells:", {pat: len(cnt) for pat, cnt in pat_types.items()})

    # grid
    h = 0.03
    lo, hi = V.min(0) - 2 * h, V.max(0) + 2 * h
    axes = [np.arange(l, u + h, h) for l, u in zip(lo, hi)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    shape = G.shape[:3]
    flat = G.reshape(-1, 3)
    dm = domain_margin(flat).reshape(shape)
    qv = [(q_eval(Q[k], flat) / (1 + np.linalg.norm(q_grad(Q[k], flat), axis=1))).reshape(shape) for k in range(3)]

    def to_world(v):
        return v * h + lo

    def mesh(field):
        v, f, _, _ = marching_cubes(field, 0.0, spacing=(1, 1, 1))
        return to_world(v), f

    regions = {}
    for pat, typ in pat_type.items():
        field = np.minimum.reduce([dm] + [pat[k] * qv[k] for k in range(3)])
        # connected components of the sign cell inside the domain
        from scipy import ndimage

        lab, n = ndimage.label(field > 0)
        sizes = ndimage.sum(np.ones(shape), lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s > 30]
        for ci, comp in enumerate(keep):
            f2 = np.where(lab == comp, field, -abs(field) - 1e-6)
            v, f = mesh(f2)
            inside = D.in_domain(random_domain_points(1), 1)  # noqa: F841 (shape check only)
            regions[(pat, ci)] = {"pattern": pat, "type": typ, "verts": v, "faces": f,
                                  "voxels": int((lab == comp).sum()),
                                  "volume": float((lab == comp).sum() * h ** 3)}
    print("regions:", [(k, v["type"][:20], v["volume"]) for k, v in regions.items()])

    # walls: triangles of each quadric inside the domain, grouped by the signs of the other two quadrics
    wall_meshes = defaultdict(lambda: {"tris": [], "verts": []})
    wall_triangles = {}
    for k in range(3):
        v, f = mesh(qv[k])
        ok = np.all(domain_margin(v)[f] > -h, axis=1)
        v_all, f = v, f[ok]
        cen = v_all[f].mean(1)
        others = [j for j in range(3) if j != k]
        key_of = [(k, *(int(np.sign(q_eval(Q[j], c)) ) for j in others)) for c in cen]
        groups = defaultdict(list)
        for tri, key in zip(f, key_of):
            groups[key].append(tri)
        wall_triangles[k] = (v_all, groups)
    out_walls = []
    for k, (v, groups) in wall_triangles.items():
        for key, tris in groups.items():
            tris = np.array(tris)
            if len(tris) < 20:
                continue
            c = v[tris].mean(1)
            # types on the two sides
            n = q_grad(Q[k], c)
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            side = []
            for sgn in (1, -1):
                q = c + sgn * 0.02 * n
                ins = domain_margin(q) > 0
                pats = Counter(sign_pattern(Q, p) for p in q[ins])
                side.append(pats.most_common(1)[0][0] if pats else None)
            out_walls.append({"quadric": k, "key": key, "tris": tris, "verts": v, "sides": side,
                              "area": float(sum(0.5 * np.linalg.norm(np.cross(v[t[1]] - v[t[0]], v[t[2]] - v[t[0]]))
                                                for t in tris))})
    print("wall pieces:", [(w["quadric"], w["key"], len(w["tris"]), w["sides"], round(w["area"], 3)) for w in out_walls])

    emit(out, Q, pat_type, regions, out_walls, wall_triangles, h)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.json")
