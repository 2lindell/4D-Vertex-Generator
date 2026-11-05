"""Every possible wall of the octahedral swirl domain, exactly (16-cell domain, SWIRL_GROUP=oct).

In the frame of the 16-cell the group fixes, its 48 rotations are signed permutation matrices, so the polychoron of a
seed with weights beta = (beta_V, beta_1, beta_2, beta_3) has the 48 vertices G_k beta: signed rearrangements of the
weights. A wall is where five vertices fall into one 3-flat:

    det[G_b beta - G_a beta, G_c beta - G_a beta, G_d beta - G_a beta, G_e beta - G_a beta] = 0,

a quartic form in beta with integer coefficients. Five vertices and their images under the group give the same form
(the rotations have determinant 1), so one five-set per orbit is enough. Each form is found from its values at 35
integer points (exactly: the values are integers), factored, and every distinct factor whose zero set crosses the
domain is a candidate wall. A candidate is a wall where the shapes on its two sides differ: neighbouring survey samples
of different shapes must have a wall between them, so each such pair is matched to the candidates that change sign
between its two ends.

    SWIRL_GROUP=oct python walls_exact.py          (writes walls_exact_oct.json)
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import domain as D

NAMES = ("V", "1", "2", "3")


def frame_group():
    G = np.array([np.round(D.CELL @ g @ D.CELL.T).astype(int) for g in D.ELEMENTS])
    assert all(sorted(np.abs(g).sum(0)) == [1] * 4 for g in G), "not signed permutations in the 16-cell's frame"
    return G


def mult_table(G):
    """T[g, k] = index of G_g G_k."""
    key = {tuple(g.ravel()): i for i, g in enumerate(G)}
    return np.array([[key[tuple((G[g] @ G[k]).ravel())] for k in range(len(G))] for g in range(len(G))])


def subset_orbits(T, n=5):
    """One five-set (sorted indices) per orbit of the group acting on the vertices."""
    N = T.shape[0]
    combos = np.array(list(itertools.combinations(range(N), n)), dtype=np.int64)
    base = np.int64(N)
    enc = lambda A: (((A[:, 0] * base + A[:, 1]) * base + A[:, 2]) * base + A[:, 3]) * base + A[:, 4]
    best = enc(combos)
    for g in range(N):
        img = np.sort(T[g][combos], axis=1)
        best = np.minimum(best, enc(img))
    reps = np.unique(best)
    out = np.zeros((len(reps), n), dtype=np.int64)
    r = reps.copy()
    for k in range(n - 1, -1, -1):
        out[:, k] = r % base
        r //= base
    return out


def monomials(deg=4, nv=4):
    return [e for e in itertools.product(range(deg + 1), repeat=nv) if sum(e) == deg]


def quartics(G, reps):
    """Integer coefficients (over monomials()) of each five-set's coplanarity form."""
    mons = monomials()
    rng = np.random.default_rng(3)
    S = rng.integers(-4, 5, size=(len(mons) + 10, 4))
    M = np.array([[np.prod(s.astype(float) ** np.array(e)) for e in mons] for s in S])
    P = np.einsum("kij,sj->ski", G, S).astype(float)             # (sample, vertex, coord)
    A = P[:, reps[:, 1:], :] - P[:, reps[:, :1], :]              # (sample, rep, 4, 4)
    vals = np.round(np.linalg.det(A)).T                         # (rep, sample): integers
    coef, *_ = np.linalg.lstsq(M, vals.T, rcond=None)
    C = np.round(coef.T).astype(np.int64)
    assert np.abs(C @ M.T - vals).max() < 1e-6, "forms not recovered exactly"
    return C, mons


def as_sympy(c, mons):
    import sympy as sp
    b = sp.symbols("bV b1 b2 b3")
    return sp.Add(*[int(k) * sp.Mul(*[v ** e for v, e in zip(b, m)]) for k, m in zip(c, mons) if k]), b


def domain_betas(n=60000, seed=0):
    """Points spread through the domain, as weights (beta_V, beta_1, beta_2, beta_3) summing to 1."""
    rng = np.random.default_rng(seed)
    B = rng.dirichlet(np.ones(4), size=4 * n)
    B = B[(B[:, 3] <= B[:, 1]) & (B[:, 3] <= B[:, 2])]
    return B[:n]


def main(out=f"walls_exact_{D.KEY}.json"):
    import sympy as sp
    G = frame_group()
    T = mult_table(G)
    reps = subset_orbits(T)
    print(f"{len(G)} rotations; {len(reps)} five-sets up to symmetry", flush=True)
    C, mons = quartics(G, reps)
    nz = np.abs(C).sum(1) > 0
    print(f"  {nz.sum()} give a form that is not identically zero", flush=True)
    # (distinct forms up to sign, with how many five-sets give each)
    uniq = {}
    for c, r in zip(C[nz], reps[nz]):
        c = c // np.gcd.reduce(np.abs(c[c != 0]))
        if c[np.flatnonzero(c)[0]] < 0:
            c = -c
        uniq.setdefault(tuple(c), []).append(r.tolist())
    print(f"  {len(uniq)} distinct forms", flush=True)
    # factors, each kept once (up to sign)
    b = sp.symbols("bV b1 b2 b3")
    factors = {}
    for k, (c, rs) in enumerate(uniq.items()):
        f, _ = as_sympy(c, mons)
        for fac, _mult in sp.factor_list(f)[1]:
            if fac.free_symbols:
                p = sp.Poly(fac, *b)
                lc = p.coeffs()[0]
                key = sp.srepr(sp.expand(fac if lc > 0 else -fac))
                e = factors.setdefault(key, {"expr": sp.expand(fac if lc > 0 else -fac), "sets": 0})
                e["sets"] += len(rs)
        if (k + 1) % 2000 == 0:
            print(f"    factored {k + 1} of {len(uniq)}", flush=True)
    print(f"  {len(factors)} distinct factors", flush=True)
    # which cross the domain (change sign inside it)
    Bd = domain_betas()
    cand = []
    for key, e in factors.items():
        fn = sp.lambdify(b, e["expr"], "numpy")
        v = fn(*Bd.T) * np.ones(len(Bd))
        if (v > 1e-12).any() and (v < -1e-12).any():
            cand.append({"expr": e["expr"], "fn": fn, "sets": e["sets"], "deg": sp.Poly(e["expr"], *b).total_degree()})
    print(f"  {len(cand)} cross the domain (by degree: {dict(Counter(c['deg'] for c in cand))})", flush=True)
    # which separate neighbouring survey samples of different shapes
    s = json.load(open(f"survey_{D.KEY}.json"))
    U = np.array(s["points"])
    sig = s["sigs"]
    Bs = D.beta_from_seed(D.seed_from_chart(U))
    h = s["h"]
    from scipy.spatial import cKDTree
    pairs = cKDTree(U).query_pairs(1.01 * h, output_type="ndarray")
    diff = pairs[[sig[a] != sig[b] for a, b in pairs]]
    print(f"  survey: {len(U)} samples, {len(diff)} neighbouring pairs of different shapes", flush=True)
    vals = np.array([c["fn"](*Bs.T) * np.ones(len(Bs)) for c in cand])           # (candidate, sample)
    flips = np.sign(vals[:, diff[:, 0]]) != np.sign(vals[:, diff[:, 1]])         # (candidate, pair)
    explained = flips.any(0)
    # a pair crossed by exactly one candidate pins that candidate as a wall
    only = flips[:, flips.sum(0) == 1]
    pinned = only.sum(1)
    used = flips.sum(1)
    for c, p, u in zip(cand, pinned, used):
        c["pinned"], c["crossings"] = int(p), int(u)
    walls = [c for c in cand if c["pinned"]]
    print(f"  {explained.sum()} of {len(diff)} label changes cross a candidate; "
          f"{len(walls)} candidates are the only one crossed by some change (certain walls)", flush=True)
    json.dump({"group": D.NAME, "domain": D.DOMAIN, "variables": list(NAMES),
               "fivesets": int(len(reps)), "forms": len(uniq), "factors": len(factors),
               "candidates": [{"poly": str(c["expr"]), "degree": c["deg"], "fivesets": c["sets"],
                               "changes_crossed": c["crossings"], "changes_only_this": c["pinned"]}
                              for c in sorted(cand, key=lambda c: (-c["pinned"], -c["crossings"]))],
               "unexplained_changes": int((~explained).sum())},
              open(out, "w"), indent=1)
    for c in sorted(walls, key=lambda c: -c["pinned"])[:30]:
        print(f"    {c['pinned']:4d} changes only here, {c['crossings']:4d} crossed: {c['expr']} = 0")


if __name__ == "__main__":
    main(*sys.argv[1:])
