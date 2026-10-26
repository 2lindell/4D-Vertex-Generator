"""Exact definitions for the working data (exact_defs.json): every wall's surface, and each of its edges, invisible
lines (cuts and crossings) and lines it goes on through, as the exact surface that meets the wall along it, with the
side of that surface the wall lies on. What cannot be pinned down exactly is listed as unverified, not filled in.

  * a line: the exact surfaces (planes of the half-cell's arrangement under the extended group, and the curved walls'
    surfaces and their images: optimal_seeds.library()) that hold every traced point of it (the traced points lie on
    the curve; only the chords between them do not), to OFF;
  * a wall's own surface: a curved wall's equation; a flat wall's planes (its main exact patches);
  * an edge on a line L: a surface holding L other than the wall's own (a plane first), crossing the wall there;
  * an edge on, or a cut by, another wall Y: the surface of Y (or an image of it) through the curve where they meet:
    the points of the wall's traced boundary next to Y (edge_curves.json) lie on it;
  * the side: the sign of that surface at points of the wall just inside it, next to the curve.

    python exact_defs.py
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

import edge_curves as ec
import optimal_seeds as osd
import region_walls as rw

OFF = 3e-6                   # (the page keeps 6 decimals of a line's points)


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def dist(s, P):
    P = np.atleast_2d(P)
    if isinstance(s, osd.Plane):
        n = s.n - s.n.mean()
        return np.abs(P @ s.n) / (np.linalg.norm(n) + 1e-300)
    g = np.array([s.grad(p) for p in P], float)
    g = g - g.mean(1, keepdims=True)
    return np.abs(s.f(P).ravel()) / (np.linalg.norm(g, axis=1) + 1e-300)


def main():
    work = json.load(open("working_data.json"))
    d = rw.page_data()
    dims = {t["id"]: t["dim"] for t in d["types"]}
    planes, curved = osd.library()
    LIB = list(planes) + list(curved)
    byname = {s.name: s for s in LIB}
    # (lines: the surfaces through every traced point of a piece of it)
    Ci = np.asarray(d["cellInv"], float)
    pieces = {}
    for src in ("xlines", "rings", "segments", "main"):
        for x in d.get(src, []):
            if isinstance(x, dict) and x.get("id") and x.get("pts") and not x.get("outline") and x.get("kind") is None:
                V = np.array(x["pts"], float) * [1, 1, -1.0]
                pieces.setdefault(x["id"], []).append(nrm(np.c_[V, np.ones(len(V))] @ Ci.T))
    line_defs = {}
    for L, Ps in pieces.items():
        per = []
        for P in Ps:
            if len(P) == 2:                               # (a straight piece: its ends and middle fix it)
                P = np.vstack([P[0], nrm(P[0] + P[1]), P[1]])
            per.append(sorted(s.name for s in LIB if dist(s, P).max() < OFF))
        line_defs[L] = {"pieces": len(Ps), "surfaces": per,
                        "exact": bool(per) and all(len(x) >= 2 for x in per)}
    # (walls' own surfaces)
    fp = [p for p in (lambda x: x if isinstance(x, list) else x.values())(json.load(open("fexact_patches.json")))
          if isinstance(p, dict)]
    X = json.load(open("xsurfaces.json"))
    own = {}
    for w in work["walls"]:
        if w in X:
            V = nrm(np.array(X[w]["tris"], float).reshape(-1, 4))[::7][:300]
            fits = sorted((float(np.median(dist(c, V))), c.name) for c in curved)
            own[w] = [fits[0][1]] if fits and fits[0][0] < 1e-8 else []
        else:
            ns = []
            for p in fp:
                if p["target"] != w or (w == "T2" and {e[2] for e in p.get("edge_labels") or []} == {"T1", "X5", "C2a"}):
                    continue
                n = np.asarray(p["normal"], float)
                n = n / np.linalg.norm(n) * np.sign(n[np.argmax(np.abs(n))])
                name = min((s for s in planes), key=lambda s: np.linalg.norm(s.n / np.linalg.norm(s.n) - n)).name
                ns.append(name)
            if w == "X96":
                ns.append(min(planes, key=lambda s: np.linalg.norm(s.n / np.linalg.norm(s.n) - np.array([0, 1, -1, 0]) / 2 ** 0.5)).name)
            own[w] = sorted(set(ns))
    E = json.load(open("edge_curves.json"))
    import edge_display
    WP, _ = edge_display._points(work)                     # (every wall's exact points, all images)
    YP = {y: np.asarray(V, float) for y, (V, h) in WP.items() if len(V)}
    YT = {y: cKDTree(V) for y, V in YP.items()}
    walls = {}
    unverified = []
    for w, v in work["walls"].items():
        base = ec.base_mesh(w, X, fp).reshape(-1, 4)
        tb = cKDTree(base) if len(base) else None
        rec = {"surface": own.get(w, []), "edges": {}, "cuts": {}, "through": {}}
        if not rec["surface"]:
            unverified.append(f"{w}: its own surface is not in the library")

        def side(s, C):
            """The sign of s at the wall's own points next to the curve (C), away from it."""
            if tb is None or not len(C):
                return None
            near = base[sorted({j for c in C for j in tb.query_ball_point(c, 0.03)})]
            far = near[dist(s, near) > 2e-3]
            if not len(far):
                return None
            sg = Counter(np.sign(s.f(far).ravel()).astype(int).tolist())
            tot = sum(sg.values())
            best, n = sg.most_common(1)[0]
            return {"sign": best, "share": round(n / tot, 3)}

        def for_line(L):
            ld = line_defs.get(L)
            if not ld or not ld["surfaces"] or tb is None:
                return {"status": "unverified: the line has no exact surface through its points"}
            # (the pieces of the line next to the wall's own piece, and the surfaces through each of them)
            # (the piece of the line nearest the wall's own piece, and the surfaces through it, but the one the wall
            # lies in there: a flat wall's other planes may bound it, where its patches meet)
            ds = [(tb.query(P)[0].min(), k) for k, P in enumerate(pieces[L])]
            dmin, k = min(ds)
            if dmin > 0.02:
                return {"status": "unverified: no piece of the line comes near the wall"}
            P, S = pieces[L][k], ld["surfaces"][k]
            if not S:
                return {"status": "unverified: the line has no exact surface through its points"}
            loc = base[sorted({j for p in P for j in tb.query_ball_point(p, 0.02)})]
            lies_in = {n for n in S if n in byname and np.mean(dist(byname[n], loc) < 1e-6) > 0.5}
            cand = [n for n in S if n not in lies_in]
            near = [(P, S)]
            cand.sort(key=lambda n: (not n.startswith("plane"), n))
            if not cand:
                return {"status": "unverified: no surface through the line other than the wall's own"}
            pts = np.vstack([P for P, _ in near])
            return {"status": "exact", "surface": cand[0], "also": cand[1:6],
                    **({"side": side(byname[cand[0]], pts)} if cand[0] in byname else {})}

        def for_wall(Y, C):
            """The surface of Y where it meets the wall: Y's own exact points (all its images) next to the curve lie
            on it, and the wall does not."""
            if len(C) and tb is not None:                  # (the curve next to the wall's own piece only)
                C = C[tb.query(C)[0] < 0.015]
            if not len(C):
                return {"status": "unverified: no points where they meet"}
            if Y not in YT:
                return {"status": f"unverified: {Y} has no points"}
            QY = YP[Y][sorted({j for c in C for j in YT[Y].query_ball_point(c, 0.02)})]
            if len(QY) < 6:
                return {"status": f"unverified: {Y} has no points next to the curve"}
            loc = base[sorted({j for c in C for j in tb.query_ball_point(c, 0.02)})]
            rng = np.random.default_rng(0)
            QY = QY[rng.choice(len(QY), min(150, len(QY)), replace=False)]
            loc = loc[rng.choice(len(loc), min(150, len(loc)), replace=False)] if len(loc) else loc
            fits = []
            for s_ in LIB:
                if np.mean(dist(s_, QY[:20]) < 1e-6) < 0.15:      # (quick rejection on a few points)
                    continue
                dy = dist(s_, QY)
                share = float(np.mean(dy < 1e-6))
                if share > 0.3 and np.mean(dist(s_, loc) < 1e-6) < 0.5:
                    fits.append((share, float(np.median(dist(s_, C))), s_.name))
            if not fits:
                return {"status": f"unverified: no surface holds {Y}'s points next to the curve"}
            share, off, n = max(fits, key=lambda x: (x[0], -x[1]))
            return {"status": "exact", "surface": n, "share_of_Y": round(share, 3), "curve_off": off,
                    "side": side(byname[n], C)}

        for e in v["edges"]:
            if dims.get(e) == "line" or e in pieces:
                r = for_line(e)
            else:
                C = np.array(E.get(w, {}).get("edges", {}).get(e, {}).get("curve", []), float)
                r = for_wall(e, C)
            rec["edges"][e] = r
            if r["status"] != "exact":
                unverified.append(f"{w} edge {e}: {r['status']}")
        for c in v.get("cuts", []):
            Y = c["by"]
            C = np.array(E.get(Y, {}).get("edges", {}).get(w, {}).get("curve", []), float)
            r = for_wall(Y, C) if len(C) else {"status": "unverified: the cutting wall's edge on it is not traced"}
            rec["cuts"][Y] = {**r, "between": c["between"], **({"crossing": True} if c.get("crossing") else {})}
            if r["status"] != "exact":
                unverified.append(f"{w} cut by {Y}: {r['status']}")
        for L in v.get("through", []):
            rec["through"][L] = for_line(L)
        walls[w] = rec
    out = {"_about": __doc__.strip().split("\n\n")[0], "lines": line_defs, "walls": walls, "unverified": unverified}
    json.dump(out, open("exact_defs.json", "w"), indent=1)
    nl = sum(1 for x in line_defs.values() if x["exact"])
    print(f"lines: {nl}/{len(line_defs)} exact (two or more surfaces through every piece)")
    print(f"walls: {sum(1 for w in walls.values() if w['surface'])}/{len(walls)} with their surface")
    n_e = sum(len(w["edges"]) for w in walls.values())
    print(f"edges: {sum(1 for w in walls.values() for r in w['edges'].values() if r['status'] == 'exact')}/{n_e} exact; "
          f"cuts: {sum(1 for w in walls.values() for r in w['cuts'].values() if r['status'] == 'exact')}/"
          f"{sum(len(w['cuts']) for w in walls.values())} exact")
    print("unverified:")
    for u in unverified:
        print("  ", u)


if __name__ == "__main__":
    main()
