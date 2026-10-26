"""Every wall's edges and cuts as equations, from the verified definitions alone (wall_equations.json).

Reads only working_data.json (which edges, cuts and crossings each wall has), exact_defs.json (each wall's surface or
surfaces, and each line as the exact surfaces through it with its verified points) and the surfaces' equations
(optimal_seeds.library()). No mesh, outline or traced edge is used.

  * an edge on a line L: L lies on the wall's surface (one of the wall's surfaces is among L's); the edge is the wall's
    surface meeting another surface through L that crosses the wall there (a plane first); the wall's side of it is
    the side holding the wall's other edge lines;
  * an edge on, or a cut by, another wall Y: the corners where Y's own edge lines meet the wall's surface are solved
    exactly (Newton on the line's surfaces and the wall's); the curve is the wall's surface meeting the surface of Y
    (Y's own, or an image of it under the group) through those corners;
  * a crossing: the same, with Y's own surface.
What does not follow from the equations is listed, with why.

    python edges_from_equations.py
"""
from __future__ import annotations

import json

import numpy as np

import optimal_seeds as osd
from t2_pieces import extended

FOLD = np.eye(4)[[1, 0, 3, 2]]
# (known end points of a cut, as exact betas: the cut's surface must pass through them)
CUT_ENDS = {("X96", "X41"): {"C4": [0, 1, 1, 1]}, ("X14", "X25"): {"C1": [1, 1, 1, 1]}}
TOL = 3e-6


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def grad(s, P):
    P = np.atleast_2d(np.asarray(P, float))
    if isinstance(s, osd.Plane):
        return np.tile(s.n - s.n.mean(), (len(P), 1))
    e = 1e-7
    g = np.stack([(s.f(P + e * u).ravel() - s.f(P - e * u).ravel()) / (2 * e) for u in np.eye(4)], 1)
    return g - g.mean(1, keepdims=True)


def dist(s, P):
    P = np.atleast_2d(np.asarray(P, float))
    return np.abs(s.f(P).ravel()) / (np.linalg.norm(grad(s, P), axis=1) + 1e-300)


def crosses(a, b, P):
    """a and b meet transversally at the points P (their normals not parallel there)."""
    ga, gb = grad(a, P), grad(b, P)
    c = np.abs((ga * gb).sum(1)) / (np.linalg.norm(ga, axis=1) * np.linalg.norm(gb, axis=1) + 1e-300)
    return bool(c.max() < 0.999)


def solve(surfs, x0, steps=40):
    """The point on all the surfaces (three of them, with sum(b) = 1) nearest x0, by Newton steps."""
    x = np.asarray(x0, float).copy()
    for _ in range(steps):
        F = np.array([s.f(x[None]).ravel()[0] for s in surfs] + [x.sum() - 1])
        if np.abs(F).max() < 1e-15:
            break
        J = np.vstack([grad(s, x)[0] + 0 * x for s in surfs] + [np.ones(4)])
        J[:-1] += np.outer([0] * len(surfs), np.ones(4))
        dx = np.linalg.lstsq(J, -F, rcond=None)[0]
        x = x + dx
    return x


def main():
    work = json.load(open("working_data.json"))
    ex = json.load(open("exact_defs.json"))
    planes, curved = osd.library()
    LIB = {s.name: s for s in list(planes) + list(curved)}
    G = extended()
    GG = np.concatenate([G, G @ FOLD])
    lines = ex["lines"]
    own = {w: r["surface"] for w, r in ex["walls"].items()}

    def images_of(Y):
        """Names of the library surfaces that are Y's own surface carried by the group."""
        out = set()
        for n in own.get(Y, []):
            s = LIB[n]
            if isinstance(s, osd.Plane):
                N = np.array([s.n @ np.linalg.inv(g) for g in GG])
                N = N / np.linalg.norm(N, axis=1, keepdims=True)
                for p in planes:
                    m = p.n / np.linalg.norm(p.n)
                    if np.abs(N @ m).max() > 1 - 1e-9:
                        out.add(p.name)
            else:
                base = n.split(":")[0]
                out |= {c.name for c in curved if c.name.split(":")[0] == base}
        return out

    IMG = {}

    def surfaces_of(w):
        """The wall's surface on the 3-sphere: its own surfaces and every image of them under the group."""
        if w not in IMG:
            IMG[w] = set(own.get(w, [])) | images_of(w)
        return IMG[w]

    def on_wall(w, L):
        """The pieces of line L lying on the wall (on its surface or an image of it): (points, the line's surfaces,
        the wall's surface there)."""
        out = []
        mine_all = surfaces_of(w)
        for P, S in zip(lines.get(L, {}).get("points", []), lines.get(L, {}).get("surfaces", [])):
            mine = [n for n in own[w] if n in S] + sorted(n for n in S if n in mine_all and n not in own[w])
            if mine:
                out.append((np.array(P, float), S, mine[0]))
        return out

    def line_edge(w, L, others):
        pcs = on_wall(w, L)
        if not pcs:
            return {"status": f"the line {L} does not lie on the wall's surface"}
        P, S, mine = pcs[0]
        if len(P) == 2:
            P = np.vstack([P[0], nrm(P[0] + P[1]), P[1]])
        cand = [n for n in S if n != mine and crosses(LIB[n], LIB[mine], P)]
        cand.sort(key=lambda n: (not n.startswith("plane"), n))
        if not cand:
            return {"status": f"no surface through {L} crosses the wall's surface there"}
        s = LIB[cand[0]]
        # (the side: where the wall's other edge lines are, on its surface and off this one)
        sides = []
        for L2 in others:
            for P2, _, _ in on_wall(w, L2):
                v = s.f(np.asarray(P2, float)).ravel()
                v = v[dist(s, P2) > 1e-6]
                sides += np.sign(v).astype(int).tolist()
        sg = int(np.sign(np.sum(sides))) if sides else 0
        return {"status": "exact", "line": L, "wall_surface": mine, "surface": cand[0], "also": cand[1:4],
                "side": sg if sg else None, "side_votes": len(sides)}

    def crossings_on(P, S, s):
        """The exact points where surface s crosses the line piece (points P, the line's surfaces S)."""
        out = []
        v = s.f(P).ravel()
        for i in np.flatnonzero(np.sign(v[:-1]) * np.sign(v[1:]) <= 0):
            t = v[i] / (v[i] - v[i + 1]) if v[i] != v[i + 1] else 0.5
            x0 = nrm(P[i] + t * (P[i + 1] - P[i]))
            a = LIB[S[0]]
            b = next((LIB[n] for n in S[1:] if crosses(LIB[n], a, P[i:i + 2])), None)
            if b is None:
                continue
            x = solve([a, b, s], x0)
            if (x > -1e-9).all() and max(dist(q, x)[0] for q in (a, b, s)) < 1e-10:
                out.append(x)
        return out

    def wall_meet(w, Y):
        """The surface of Y (its own, or an image) meeting the wall's own surface along the curve where they meet: the
        one through the most corners, a corner being where one of Y's own edge lines meets the wall's surface, or
        one of the wall's own edge lines meets that surface of Y."""
        score = {}
        for L in work["walls"].get(Y, {}).get("edges", []):            # (Y's lines, on Y, meeting the wall)
            if L not in lines:
                continue
            for P, S, mineY in on_wall(Y, L):
                P = np.asarray(P, float)
                if len(P) == 2:
                    P = np.vstack([P[0], nrm(P[0] + P[1]), P[1]])
                for n_w in own[w]:
                    for x in crossings_on(P, S, LIB[n_w]):
                        for n in (n_ for n_ in S if n_ in surfaces_of(Y)):
                            score.setdefault(n, []).append(("Y", L, x))
        for L in [e for e in work["walls"][w]["edges"] if e in lines]:  # (the wall's lines, meeting a surface of Y)
            for P, S, mine in on_wall(w, L):
                if mine not in own[w]:
                    continue
                P = np.asarray(P, float)
                if len(P) == 2:
                    P = np.vstack([P[0], nrm(P[0] + P[1]), P[1]])
                for n in surfaces_of(Y):
                    if n in own[w]:
                        continue
                    for x in crossings_on(P, S, LIB[n]):
                        score.setdefault(n, []).append(("W", L, x))
        # (corners at the ends of either wall's edge lines lying on both surfaces: a line lying in Y's surface, or
        # ending on it, crosses it nowhere)
        for who, wall_, Ls in (("W", w, work["walls"][w]["edges"]), ("Y", Y, work["walls"].get(Y, {}).get("edges", []))):
            for L in Ls:
                if L not in lines:
                    continue
                for P, S, _m in on_wall(wall_, L):
                    for p in (np.asarray(P[0], float), np.asarray(P[-1], float)):
                        if dist(LIB[own[w][0]], p)[0] > 1e-5:
                            continue
                        for n in own.get(Y, []):
                            if n in own[w] or dist(LIB[n], p)[0] > 1e-5:
                                continue
                            x = solve([LIB[own[w][0]], LIB[n]] + [LIB[q] for q in S if q not in (own[w][0], n)][:1], p)
                            if max(dist(q, x)[0] for q in (LIB[own[w][0]], LIB[n])) < 1e-9:
                                score.setdefault(n, []).append((who, L, x))
        score = {n: c for n, c in score.items() if n not in own[w] and crosses(LIB[n], LIB[own[w][0]], np.array([x for _, _, x in c]))
                 if isinstance(LIB[own[w][0]], osd.Plane) or True}
        if not score:
            return {"status": f"no corner: no surface of {Y} meets the wall's surface at its edge lines"}
        # (first the surface holding Y's own edge lines at the corners, then the one through the most corners)
        own_side = lambda k: sum(1 for who, _, _ in score[k] if who == "Y")
        ends = CUT_ENDS.get((w, Y), {})
        if ends:
            E_ = nrm(np.array(list(ends.values()), float))
            score = {k: c for k, c in score.items() if dist(LIB[k], E_).max() < 1e-12}
            if not score:
                return {"status": f"no surface of {Y} through its known ends {sorted(ends)}"}
        # (Y's own surface first, where it meets the wall at all; then the one holding Y's own edge lines at the
        # corners, then the one through the most corners)
        n = max(score, key=lambda k: (k in own.get(Y, []), own_side(k), len(score[k]), not k.startswith("plane"), k))
        C = np.array([x for _, _, x in score[n]])
        # (a wall that is several flat patches may meet the wall on more than one of them: each with its corners)
        more = [{"surface": k, "corners": np.array([x for _, _, x in score[k]]).round(12).tolist()}
                for k in own.get(Y, []) if k in score and k != n]
        return {"status": "exact", "surface": n, "corners": C.round(12).tolist(), "corner_count": len(C),
                **({"more": more} if more else {}),
                "corner_lines": sorted({f"{who}:{L}" for who, L, _ in score[n]}),
                "on_Y_lines": own_side(n),
                "next_best": sorted(((own_side(k), len(c), k) for k, c in score.items() if k != n), reverse=True)[:2]}

    out, problems = {}, []
    for w, v in work["walls"].items():
        if w not in own or not own[w]:
            problems.append(f"{w}: no verified surface")
            continue
        rec = {"surface": own[w], "edges": {}, "cuts": {}}
        line_edges = [e for e in v["edges"] if e in lines]
        for e in v["edges"]:
            r = line_edge(w, e, [x for x in line_edges if x != e]) if e in lines else wall_meet(w, e)
            rec["edges"][e] = r
            if r["status"] != "exact":
                problems.append(f"{w} edge {e}: {r['status']}")
        for c in v.get("cuts", []):
            r = wall_meet(w, c["by"])
            rec["cuts"][c["by"]] = {**r, "between": c["between"], **({"crossing": True} if c.get("crossing") else {})}
            if r["status"] != "exact":
                problems.append(f"{w} cut by {c['by']}: {r['status']}")
        out[w] = rec
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "walls": out, "not_from_equations": problems},
              open("wall_equations.json", "w"), indent=1)
    ne = sum(len(r["edges"]) for r in out.values())
    nc = sum(len(r["cuts"]) for r in out.values())
    print(f"edges: {sum(1 for r in out.values() for x in r['edges'].values() if x['status'] == 'exact')}/{ne}; "
          f"cuts: {sum(1 for r in out.values() for x in r['cuts'].values() if x['status'] == 'exact')}/{nc}")
    for p in problems:
        print("  ", p)


if __name__ == "__main__":
    main()
