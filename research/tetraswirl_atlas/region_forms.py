"""Each region of the octahedral swirl domain as an exact set of sign conditions, and its walls (16-cell domain).

A seed's polychoron changes only where one of its cells stops being supporting: a vertex reaches the cell's 3-flat
(a vertex of a neighbouring cell, or, where symmetry makes several cross together, a whole ring of them). With the seed's vertices G_k beta (signed rearrangements of beta, walls_exact.py), that is a five-point
determinant, a quartic form in beta with integer coefficients. So a region, found from one of its seeds, is the set
where every pair of neighbouring cells of its polychoron keeps the sign it has there:

    region = { beta : sign f_AB(beta) = sign f_AB(seed) for every neighbouring pair of cells A, B }.

Checked against the survey: every sample of the region meets all its conditions, and every other sample fails one.
A condition is one of the region's walls where it alone fails at points just outside (random points of the domain
that fail it and no other condition), and the wall is the factor of the form that vanishes there.

    SWIRL_GROUP=oct python region_forms.py          (writes region_forms_oct.json)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import sympy as sp

import domain as D
from walls_exact import NAMES, domain_betas, frame_group, monomials
from four_d_vertex_generator.off import compute_convex_hull

B_SYM = sp.symbols("bV b1 b2 b3")
MONS = monomials()


def form_of(G, five):
    """Integer coefficients of det[G_b x - G_a x, ...] for the five vertex indices (a, b, c, d, e)."""
    rng = np.random.default_rng(3)
    S = rng.integers(-4, 5, size=(len(MONS) + 10, 4))
    M = np.array([[np.prod(s.astype(float) ** np.array(e)) for e in MONS] for s in S])
    P = np.einsum("kij,sj->ski", G[list(five)], S).astype(float)
    vals = np.round(np.linalg.det(P[:, 1:] - P[:, :1]))
    c, *_ = np.linalg.lstsq(M, vals, rcond=None)
    c = np.round(c).astype(np.int64)
    assert np.abs(M @ c - vals).max() < 1e-6
    return c


def evaluate(c, B):
    return np.array([np.prod(B ** np.array(e), axis=1) for e in MONS]).T @ c


def hull_pairs(G, beta):
    """The polychoron of beta: its vertices (as group indices) and, for each pair of neighbouring cells, a five-set
    (four independent vertices of one cell and a vertex of the other off its 3-flat)."""
    X = np.einsum("kij,j->ki", G, beta)
    faces, cells = compute_convex_hull(X, tol=1e-9)
    cv = [sorted({v for f in c for v in faces[f]}) for c in cells]
    owner = defaultdict(list)
    for ci, c in enumerate(cells):
        for f in c:
            owner[f].append(ci)
    out = set()
    for f, (a, b) in ((f, o) for f, o in owner.items() if len(o) == 2):
        for A, Bc in ((a, b), (b, a)):
            base = cv[A]
            # four independent vertices of A, starting with the shared face's
            pick = list(faces[f][:3])
            for v in base:
                if v not in pick and np.linalg.svd(X[pick[1:] + [v]] - X[pick[0]], compute_uv=False)[-1] > 1e-7:
                    pick.append(v)
                    break
            apex = [v for v in cv[Bc] if v not in base]
            if len(pick) == 4 and apex:
                out.add(tuple(pick + [apex[0]]))
    return X, out


def support_fives(G, beta):
    """For one cell of each kind of beta's polychoron (the rest are its images: the forms are the same), four
    independent vertices of it and each vertex off it: the hull is unchanged while every such vertex stays on the
    inner side of the cell's 3-flat (the cells stay supporting), and only then."""
    from walls_exact import mult_table
    X = np.einsum("kij,j->ki", G, beta)
    faces, cells = compute_convex_hull(X, tol=1e-9)
    cv = [frozenset(v for f in c for v in faces[f]) for c in cells]
    if not hasattr(support_fives, "T"):
        support_fives.T = mult_table(G)
    T = support_fives.T
    seen, out = set(), set()
    for c in cv:
        if c in seen:
            continue
        seen |= {frozenset(T[g][list(c)].tolist()) for g in range(len(G))}
        base = sorted(c)
        pick = [base[0]]
        for v in base[1:]:
            if len(pick) < 4 and np.linalg.svd(X[pick[1:] + [v]] - X[pick[0]], compute_uv=False)[-1] > 1e-7:
                pick.append(v)
        for q in range(len(X)):
            if q not in c:
                out.add(tuple(pick + [q]))
    return X, out


def conditions(G, beta):
    """The oriented forms (positive at beta) that keep every cell of beta's polychoron supporting."""
    _, fives = support_fives(G, beta)
    conds = {}
    for five in fives:
        c = form_of(G, five)
        if not np.abs(c).any():
            continue
        c = c // np.gcd.reduce(np.abs(c[c != 0]))
        sgn = int(np.sign(evaluate(c, beta[None])[0]))
        if sgn:
            conds.setdefault(tuple(c * sgn), five)
    return conds


def factor_text(c):
    return sp.factor_list(sp.Add(*[int(a) * sp.Mul(*[v ** e for v, e in zip(B_SYM, m)]) for a, m in zip(c, MONS) if a]))[1]


def main():
    from scipy.spatial import cKDTree
    G = frame_group()
    s = json.load(open(f"survey_{D.KEY}.json"))
    U = np.array(s["points"])
    sig = np.array(s["sigs"])
    Bs = D.beta_from_seed(D.seed_from_chart(U))
    ids = json.load(open(f"ids_{D.KEY}.json"))
    # pieces: each shape's samples covered by the sign conditions of one example seed after another
    piece_of = np.full(len(Bs), -1)
    pieces = []
    tree = {sg: cKDTree(U[sig != sg]) for sg in set(sig)}
    for sg, tid in sorted(ids.items(), key=lambda kv: int(kv[1][1:])):
        left = np.flatnonzero(sig == sg)
        k = 0
        while len(left):
            # (the example: the sample farthest from every sample of another shape, so it is well inside; if its
            # conditions let in another shape's samples, the next farthest, up to eight tries)
            best = None
            for i in left[np.argsort(-tree[sg].query(U[left])[0])][:8]:
                C_ = np.array(list(conditions(G, Bs[i])))
                ok_ = (evaluate(C_.T, Bs) > 0).all(1)
                w_ = int((ok_ & (sig != sg)).sum())
                if best is None or w_ < best[0] or (w_ == best[0] and ok_[left].sum() > best[2][left].sum()):
                    best = (w_, C_, ok_, Bs[i])
                if w_ == 0:
                    break
            wrong, C, ok, rep = best
            got = left[ok[left]]
            if not len(got):                       # (the example itself must pass: a borderline seed; skip it)
                left = left[1:]
                continue
            name = tid if k == 0 else f"{tid}.{k + 1}"
            piece_of[got] = len(pieces)
            pieces.append({"id": name, "type": tid, "sig": sg, "conds": C, "rep": rep, "samples": len(got), "others_inside": wrong})
            left = np.setdiff1d(left, got)
            k += 1
    for p in pieces:
        print(f"{p['id']:7s} {len(p['conds']):3d} conditions, {p['samples']:5d} samples, {p['others_inside']} other shapes inside", flush=True)
    # walls: from each sample to a neighbour of another piece, the first condition of its own piece to change sign
    pairs = cKDTree(U).query_pairs(1.01 * s["h"], output_type="ndarray")
    pairs = pairs[(piece_of[pairs[:, 0]] != piece_of[pairs[:, 1]]) & (piece_of[pairs] >= 0).all(1)]
    t = np.linspace(0, 1, 401)[1:]
    found = defaultdict(Counter)            # (piece, surface) -> other pieces
    fac_fns = {}

    def factors_of(pi, k):
        if (pi, k) not in fac_fns:
            fac_fns[(pi, k)] = [(str(sp.expand(f)), sp.lambdify(B_SYM, f, "numpy"))
                                for f, _ in factor_text(pieces[pi]["conds"][k]) if f.free_symbols]
        return fac_fns[(pi, k)]
    for a, b in np.vstack([pairs, pairs[:, ::-1]]):
        p = pieces[piece_of[a]]
        path = (1 - t)[:, None] * Bs[a] + t[:, None] * Bs[b]
        V = evaluate(p["conds"].T, path)                        # (t, cond)
        neg = V <= 0
        if not neg.any():
            continue
        first = np.argmax(neg.any(1))
        before = Bs[a] if first == 0 else path[first - 1]
        for k in np.flatnonzero(neg[first]):
            # (the factor of the form that vanishes there: the one changing sign over the last step)
            for name, fn in factors_of(piece_of[a], k):
                if np.sign(fn(*before)) != np.sign(fn(*path[first])):
                    found[(piece_of[a], name)][pieces[piece_of[b]]["id"]] += 1
    surfaces = {}
    out = {}
    for pi, p in enumerate(pieces):
        walls = []
        for (qi, name), other in sorted(found.items()):
            if qi != pi:
                continue
            key = name
            surfaces.setdefault(key, {"factor": name, "pieces": set()})["pieces"].add(p["id"])
            walls.append({"surface": name, "across": dict(other), "crossings": int(sum(other.values()))})
        out[p["id"]] = {"type": p["type"], "sig": p["sig"], "conditions": [list(map(int, c)) for c in p["conds"]],
                        "samples": p["samples"], "other_shapes_inside": p["others_inside"],
                        "example_beta": p["rep"].round(6).tolist(), "walls": walls}
    surf = sorted(({"factor": v["factor"], "pieces": sorted(v["pieces"])} for v in surfaces.values()),
                  key=lambda w: -len(w["pieces"]))
    print(f"{len(pieces)} pieces; {len(surf)} distinct wall surfaces")
    for w in surf:
        print(f"   {', '.join(w['pieces']):28s} {w['factor']} = 0")
    json.dump({"group": D.NAME, "domain": D.DOMAIN, "variables": ["bV", "b1", "b2", "b3"], "monomials": MONS,
               "pieces": out, "surfaces": surf}, open(f"region_forms_{D.KEY}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
