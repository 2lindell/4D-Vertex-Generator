"""The F4 domain viewer's maps of the curved F4 symmetries (kicontic, iocontic, oxitic) from their exact maps
(../a4_atlas: a4_curves -> a4_arcs -> a4_exactmap -> a4_walls exact -> a4_exactpieces), as f4_map_<group>.json in the
A4 maps' layout (../a4_atlas/a4_snub.build), for ../a4_atlas/build_page.py f4.

The map's coordinates are exactspec's b: the chamber's betas for kicontic and iocontic; for oxitic, whose domain is
two chambers' worth, the tetrahedron T = C u r1 C u r4 C u r1 r4 C, b = (beta2, beta1 + beta2, beta3, beta3 + beta4)
(beta = M b, exactspec's M). T is drawn in the gnomonic chart at its own centre, as the chamber is.

  * walls: the exact wall pieces off the domain's faces, each meshed inside its exact loop; on the domain's faces the
    pieces reading a wall (contic's walls on the mirrors, under kicontic and iocontic);
  * each region's shell: the wall pieces either side of it and the domain-face pieces on its side;
  * lines: the traced arcs reading a line shape; edges: the other arcs where walls meet;
  * points: the map's points reading a point shape;
  * names: the atlas's shapes (exactspec.f4_shapes) read at their seeds; their examples there (and their images
    under the group's symmetries of the domain).

    python f4_curved.py kicontic iocontic [oxitic]
"""
from __future__ import annotations

import itertools
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
A4 = os.path.join(ROOT, "a4_atlas")
sys.path.insert(0, A4)
sys.path.insert(0, HERE)
import f4_map as F4M  # noqa: E402
from a4_map import parse_sig, r5  # noqa: E402


def frame(Mb):
    """The gnomonic chart of the domain whose corners are the betas M e_k, at its centre (f4_map.frame when M = I)."""
    corners = (F4M.RI @ Mb).T
    corners = corners / np.linalg.norm(corners, axis=1)[:, None]
    c = corners.sum(0)
    c /= np.linalg.norm(c)
    z = corners[0] - (corners[0] @ c) * c
    z /= np.linalg.norm(z)
    x = corners[3] - (corners[3] @ c) * c - (corners[3] @ z) * z
    x /= np.linalg.norm(x)
    y = np.linalg.svd(np.vstack([c, z, x]))[2][-1]
    if np.linalg.det(np.vstack([x, y, z, c])) * np.linalg.det(np.vstack([*F4M.F, F4M.C])) < 0:
        y = -y
    return c, np.vstack([x, y, z])


BETA_OF_B = {"oxitic": ["β2", "β1 + β2", "β3", "β3 + β4"]}


def build(group):
    import a4_walls as W
    import exactspec
    SP = exactspec.get(group)
    Mb = np.asarray(SP.M, float)
    C, F = frame(Mb)

    def beta(b):
        return Mb @ np.clip(np.asarray(b, float), 0, None)

    def chart(b):
        s = F4M.RI @ beta(b)
        return F @ (s / (s @ C))

    chartv = lambda V: [r5(chart(v)) for v in V]   # noqa: E731
    X = pickle.load(open(os.path.join(A4, f"a4_pieces_{group}_exact.pkl"), "rb"))
    sub, order = F4M.SUBS[group], F4M.ORDER[group]
    # the names: the atlas's shapes read at their seeds put exactly on their sets (a4_walls.dims_of_f4)
    dim, name = W.dims_of_f4(group, SP)
    # every key the map holds, with a point of it (to read its signature) and its kind where the atlas names none
    rep, kind_seen = {}, defaultdict(set)
    for F_ in X["faces"]:
        rep.setdefault(F_["key"], F_["inner"])
        kind_seen[F_["key"]].add("wall")
        for r in F_["sides"]:
            if r is not None:
                kind_seen[r].add("region")
    for sg in X["segments"]:
        mid = sg["pts"][len(sg["pts"]) // 2]
        rep.setdefault(sg["key"], mid)
        kind_seen[sg["key"]].add("line")
    # (a region's point: a step off a wall piece it borders, on its side)
    import a4_exact as E
    import a4_pieces as P
    surfs = [E.Surf(e, n) for n, e in zip(SP.names, SP.exprs)]
    for F_ in X["faces"]:
        x = np.asarray(F_["inner"], float)
        k = F_["surface"]
        for side, r in enumerate(F_["sides"]):
            if r is None or r in rep:
                continue
            if k < 4:
                y = x.copy()
                y[k] += 1e-4
            else:
                y = np.clip(x + (1 if side == 0 else -1) * 1e-4 * P.normal(surfs[k], x), 0, None)
            rep[r] = y / y.sum()
    rank = {"point": 0, "line": 1, "wall": 2, "region": 3}
    keys = sorted(k for k in rep if k is not None)
    for k in keys:
        if k not in dim:
            dim[k] = max(kind_seen[k], key=lambda d: rank[d])
    # (the atlas's example seeds of each named shape, in b; read again here: kept where they read the shape)
    seeds = defaultdict(list)
    for nm_, kind, x in exactspec.f4_shapes(group):
        seeds[nm_].append(np.asarray(x, float) / np.sum(x))
    print(f"  {group}: {len(keys)} keys; reading their signatures", flush=True)
    lab = F4M.classify([beta(rep[k]) for k in keys], sub, 1e-7)
    sig = {}
    for k, (s_, kk) in zip(keys, lab):
        sig[k] = s_
        if kk != k:
            print(f"  {group}: the point of {k} reads {kk}", flush=True)
    DEG = {"region": 3, "wall": 2, "line": 1, "point": 0}
    types = []
    for k in keys:
        nm_ = name.get(k, "")
        p = parse_sig(sig[k]) or {}
        tid = nm_ or f"{dim[k]}_{k[:6]}"
        types.append({"id": tid, "name": F4M.FULL.get(nm_, ("", ""))[0], "dim": dim[k], "group": dim[k] + "s",
                      "key": k, "sig": sig[k], **p, "degrees": DEG[dim[k]], "pieces": 0, "locus": "", "bounds": [],
                      "separates": [], "mirrors": [], "symbol": "circle" if p.get("vertices") == order else "diamond"})
    cnt, seen = Counter(t["id"] for t in types), Counter()
    for t in types:
        if cnt[t["id"]] > 1:
            seen[t["id"]] += 1
            t["id"] = f"{t['id']}_{seen[t['id']]}"
    T = {t["key"]: t for t in types}
    # walls and the regions' shells
    walls, shells_tri, sep, mirrors = [], defaultdict(list), defaultdict(set), defaultdict(set)
    for F_ in X["faces"]:
        mesh = {"verts": chartv(F_["mesh"]["verts"]), "faces": [list(map(int, t)) for t in F_["mesh"]["faces"]]}
        t = T[F_["key"]]
        on_face = F_["surface"] < 4
        if t["dim"] == "wall":
            walls.append({"id": t["id"], **mesh})
            t["pieces"] += 1
            if on_face:
                mirrors[t["key"]].add(F_["surface"])
            for r in F_["sides"]:
                if r in T and T[r]["dim"] == "region":
                    sep[t["key"]].add(T[r]["id"])
                    if not on_face:
                        shells_tri[r].append(len(walls) - 1)
        if on_face and F_["sides"][0] in T and T[F_["sides"][0]]["dim"] == "region":
            shells_tri[F_["sides"][0]].append(mesh)
    shells = {}
    for r, parts in shells_tri.items():
        V, Fa, Wi = [], [], []
        for m in parts:
            if isinstance(m, int):
                Wi.append(m)
                continue
            o = len(V)
            V += m["verts"]
            Fa += [[a + o for a in f] for f in m["faces"]]
        shells[T[r]["id"]] = {"verts": V, "faces": Fa, "walls": sorted(set(Wi)), "edges": []}
    for t in types:
        t["separates"] = sorted(sep.get(t["key"], ()))
        t["mirrors"] = sorted(mirrors.get(t["key"], ()))
    # lines (by their runs: segments joined at shared points), and the edges where walls meet
    lines, edges, runs = [], [], defaultdict(dict)
    for sg in X["segments"]:
        Q = [np.asarray(q, float) for q in sg["pts"]]
        if np.linalg.norm(np.diff(np.array(Q), axis=0), axis=1).sum() < 1e-5:
            continue
        pts = chartv(Q[:: max(1, len(Q) // 40)] + [Q[-1]])
        t = T.get(sg["key"])
        if t is not None and t["dim"] == "line":
            lines.append({"id": t["id"], "pts": pts})
            par = runs[t["key"]]

            def root(a):
                while par.setdefault(a, a) != a:
                    a = par[a]
                return a
            a, b = sg["ends"]
            par[root(a)] = root(b)
        elif sg.get("edge") and len([k for k in sg["on"] if k < 4]) < 2:
            edges.append({"pts": pts, "hover": "an edge of the map: " + " and ".join(
                X["surfaces"][k] for k in sorted(sg["on"])[:3])})
    for k, par in runs.items():
        T[k]["pieces"] = sum(1 for a in par if par[a] == a)
    # points: the map's points reading a point shape
    live = sorted({e for sg in X["segments"] for e in sg["ends"]})
    print(f"  {group}: reading {len(live)} map points", flush=True)
    plab = SP.classify([X["points"][v] for v in live])
    points, unl = [], Counter()
    for v, kk in zip(live, plab):
        if kk in T and T[kk]["dim"] == "point":
            b = beta(X["points"][v])
            points.append({"id": T[kk]["id"], "q": r5(chart(X["points"][v])), "beta": F4M.beta_text(b),
                           "seed": F4M.seed_text(b)})
        elif kk not in T:
            unl[kk] += 1
    # examples: the atlas's seeds (and their images under the domain's symmetries) where they read their shape
    imgs = [lambda b: b] + [(lambda p: (lambda b: np.asarray(b)[p]))(perm) for perm in (SP.syms or [])]
    cand = [(t, y) for t in types for x in seeds.get(t["id"].split("_")[0], [])
            for y in {tuple(np.round(im(x), 15)) for im in imgs}]
    chk = SP.classify([np.asarray(y) for _, y in cand]) if cand else []
    examples, got = [], Counter()
    for (t, y), kk in zip(cand, chk):
        if kk != t["key"]:
            continue
        b = beta(y)
        examples.append({"id": t["id"], "q": r5(chart(y)), "beta": F4M.beta_text(b), "seed": F4M.seed_text(b)})
        if not got[t["id"]]:
            t["seed"], t["beta"], t["example"] = F4M.seed_text(b), F4M.beta_text(b), (
                "central seed (the deepest point of the region)" if t["dim"] == "region" else "the atlas's seed")
        got[t["id"]] += 1
    for t in types:
        if t["id"] not in got:          # (no atlas seed: the point the map read it at)
            b = beta(rep[t["key"]])
            t["seed"], t["beta"], t["example"] = F4M.seed_text(b), F4M.beta_text(b), "a point of the map"
            examples.append({"id": t["id"], "q": r5(chart(rep[t["key"]])), "beta": t["beta"], "seed": t["seed"]})
    rng = np.random.default_rng(3)
    R_ = rng.dirichlet([1, 1, 1, 1], 1000)
    frac = Counter(SP.classify(list(R_), tol=1e-9))
    for t in types:
        if t["dim"] == "region":
            pc = 100 * frac.get(t["key"], 0) / len(R_)
            t["where"] = (f"region ({pc:.1f}% of the domain)" if pc >= 0.05 else
                          "region (under 0.1% of the domain: thin)")
        else:
            t["where"] = t["dim"] + (f" ({t['pieces']} pieces)" if t["pieces"] > 1 else "") + (
                " on the domain's faces" if t["mirrors"] and t["dim"] == "wall" else "")
    bn = BETA_OF_B.get(group, [f"β{k + 1}" for k in range(4)])
    dom = {"verts": [r5(chart(e)) for e in np.eye(4)], "edges": [list(e) for e in itertools.combinations(range(4), 2)],
           "faces": [{"loop": [i for i in range(4) if i != k], "name": f"the mirror {bn[k]} = 0", "kind": "mirror",
                      "ids": []} for k in range(4)]}
    order_ = {"region": 0, "wall": 1, "line": 2, "point": 3}
    types.sort(key=lambda t: (order_[t["dim"]], -(t.get("vertices") or 0), t["id"]))
    out = {"group": group, "order": order, "domain": dom, "types": types, "walls": walls, "lines": lines,
           "edges": edges, "points": points, "examples": examples, "shells": shells, "exact": True,
           "unlisted": sum(unl.values())}
    path = os.environ.get("F4_OUT") or os.path.join(HERE, f"f4_map_{group}.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print(f"{path}: {len(types)} shapes ({dict(Counter(t['dim'] for t in types))}), {len(walls)} wall pieces, "
          f"{len(lines)} line arcs, {len(edges)} edges, {len(points)} points, {len(examples)} examples; "
          f"unnamed: {[t['id'] for t in types if not t['name']]}; map points of no listed shape: {len(unl)} keys",
          flush=True)


if __name__ == "__main__":
    for g in sys.argv[1:] or ["kicontic", "iocontic"]:
        build(g)
