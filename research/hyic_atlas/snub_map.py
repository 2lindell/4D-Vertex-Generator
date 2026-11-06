"""A hyic subgroup's domain (the H4 chamber) mapped over curved walls (snub_map<tag>.json); as
decaic_atlas/snub_map.py and contic_atlas/snub_map.py, over Q(sqrt5), with no fold (the hyic family).

Under the snub subgroups the forms (forms.py) may no longer all factor into planes: many walls are quadrics, cubics or
quartics in the betas. So the plane arrangement is replaced by:

  1. surfaces: the contic planes (arrangement.json) and every mixed-sign factor of the forms of every grid region
     found below, exact over Q(sqrt2); a wall crossing with no surface through it adds the forms of both sides;
  2. regions: a grid in the betas (all betas > 0, b1 > b4; shifted off the lattice by small irrational amounts, since
     lattice points lie exactly on rational walls) classified (group.label); grid neighbours of the same
     polytope joined (and joined across the fold, where the swap makes (a,b,c,a+) and (a+,c,b,a) the same seed);
  3. walls: between grid neighbours of different polytopes the change is bisected to ~1e-11, the surface vanishing there
     identified (|f| / |grad f| < 1e-7), the point put exactly on it (Newton) and its polychoron classified;
  4. the fold plane b1 = b4 (the domain's edge inside the chamber) mapped the same way in two dimensions.

On the mirrors (some beta_k = 0) the subgroup's orbit is the contic one: those shapes are the contic atlas's.

    HYIC_SUB=ki python snub_map.py [N] [CAP]
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import sympy as sp

import group as G

B = sp.symbols("b1 b2 b3 b4")
OUT = G.fname("snub_map")
CAP = 12


class Surface:
    def __init__(self, text):
        self.text = text
        e = sp.sympify(text)
        self.f = sp.lambdify(B, e, "numpy")
        self.g = [sp.lambdify(B, sp.diff(e, v), "numpy") for v in B]

    def val(self, x):
        return float(self.f(*x))

    def grad(self, x):
        g = np.array([float(gi(*x)) for gi in self.g])
        return g - g.mean()                    # (along beta summing to 1)

    def dist(self, x):
        return abs(self.val(x)) / (np.linalg.norm(self.grad(x)) + 1e-300)

    def project(self, x):
        x = np.array(x, float)
        for _ in range(60):
            g = self.grad(x)
            step = self.val(x) / (g @ g) * g
            x = x - step
            if np.abs(step).max() < 1e-16:
                break
        return x


def mixed(text):
    """Does the factor take both signs on the open chamber? (its coefficients, in Q(sqrt2), have both signs; a factor
    with one-signed coefficients vanishes nowhere in it)"""
    p = sp.Poly(sp.sympify(text), *B)
    c = [float(v) for v in p.coeffs()]
    return min(c) < 0 < max(c)


def surfaces_from_files():
    """This group's arrangement planes (arrangement<tag>.json) and the curved factors its log records (over Q(sqrt2))."""
    from fractions import Fraction
    out = set()
    try:
        A = json.load(open(G.fname("arrangement")))
        for p in A["planes"]:
            t = " + ".join(f"({Fraction(a)} + {Fraction(b)}*sqrt(2))*{v}" for (a, b), v in zip(p, ["b1", "b2", "b3", "b4"]))
            out.add(str(sp.expand(sp.sympify(t))))
    except FileNotFoundError:
        pass
    try:
        text = open(G.fname("arrangement", "log")).read()
        for seg in re.split(r"\s*NONLINEAR at \[[^\]]*\]: ", text)[1:]:
            f = re.split(r"\s{2,}|\n", seg)[0].strip()
            try:
                if mixed(f):
                    out.add(str(sp.expand(sp.sympify(f))))
            except (sp.SympifyError, sp.PolynomialError):
                pass
    except FileNotFoundError:
        pass
    return sorted(out)


def forms_job(beta):
    """The mixed-sign irreducible factors (over Q(sqrt2)) of the seed's forms, planes and curved."""
    import forms as F
    out = set()
    try:
        for fx in F.region_forms(beta):
            for fac in F._factor(str(fx)):
                if mixed(str(fac)):
                    out.add(str(sp.expand(fac)))
    except Exception:
        pass
    return out


def label_job(b):
    sg, k = G.label(b)
    return sg, k


def wall_label_job(b):
    sg, k = G.label(b, 1e-7)
    return sg, k


def bisect_job(item):
    """Bisect the change between a and b (labels la, lb) to ~1e-11: the point, and the polytope on each side."""
    a, b, la, lb = item
    a, b = np.array(a, float), np.array(b, float)

    def same(l, m, ref, refpt):
        return l[1] == ref[1] or (l[0] == ref[0] and G.same_polytope(m, refpt))
    lo, hi, ka, kb, pa, pb = a, b, la, lb, a, b
    for _ in range(30):
        m = (lo + hi) / 2
        l = G.label(m)
        if same(l, m, ka, pa):
            lo = m
        else:
            hi = m
            if l[1] is not None:                      # (whatever is on the far side: maybe a third polytope; a hull
                kb, pb = l, m                         #  too near the wall to classify keeps the last one classified)
    return (lo + hi) / 2, ka, kb, pa, pb


def grid(N, fold=False):
    pts = []
    for i in range(1, N):
        for j in range(1, N - i):
            for k in range(1, N - i - j):
                l = N - i - j - k
                if l < 1:
                    continue
                if (not G.FOLD and not fold) or (i > l and not fold) or (i == l and fold):
                    pts.append((i, j, k, l))
    return pts


SHIFT = np.array([np.sqrt(2) - 1.4, np.sqrt(3) - 1.7, np.sqrt(7) - 2.6, np.sqrt(11) - 3.3]) * 0.37  # (no sqrt5: the walls are over Q(sqrt5))


def generic(P, N, fold=False):
    """The grid points shifted off the lattice by small irrational amounts (rational points lie exactly on the
    rational walls), kept on the fold b1 = b4 for the fold grid."""
    s = SHIFT.copy()
    if fold:
        s[3] = s[0]
    X = np.array(P, float) + s
    return X / X.sum(1, keepdims=True)


class Types:
    """Polytope types: arrangement keys joined when their polytopes are the same (group.same_polytope)."""
    def __init__(self):
        self.key2t, self.reps = {}, []          # (key -> type id), (type id -> (signature, beta))

    def of(self, sg, key, beta):
        if key is None:
            return -1
        if key in self.key2t:
            return self.key2t[key]
        nv = sg.split(":")[0]
        for t, (sg2, b2) in enumerate(self.reps):
            if sg2.split(":")[0] == nv and G.same_polytope(beta, b2):
                self.key2t[key] = t
                return t
        self.reps.append((sg, list(map(float, beta))))
        self.key2t[key] = len(self.reps) - 1
        return len(self.reps) - 1


def main():
    t0 = time.time()
    global CAP
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    pool = Pool(4)
    T = Types()
    # (regions: the grid)
    P = grid(N)
    Bp = generic(P, N)
    R = pool.map(label_job, list(map(tuple, Bp)), chunksize=16)
    typ = np.array([T.of(sg, k, b) for (sg, k), b in zip(R, Bp)])
    print(f"grid N={N}: {len(P)} points, {len(T.reps)} types, {time.time() - t0:.0f}s", flush=True)
    index = {p: i for i, p in enumerate(P)}
    moves = [tuple(int(a == u) - int(a == w) for a in range(4)) for u in range(4) for w in range(4) if u != w]
    parent = list(range(len(P)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pairs = []
    for i, p in enumerate(P):
        for m in moves:
            q = tuple(a + b for a, b in zip(p, m))
            j = index.get(q)
            if j is None or j < i:
                continue
            if typ[i] == typ[j]:
                parent[find(i)] = find(j)
            else:
                pairs.append((i, j))
    # (across the fold: (i,j,k,l) with i = l+1 next to its swap image (l+1... ) -- the swap of (i,j,k,l) is (l,k,j,i);
    #  a point just inside (i = l + 1) and the swap image of its neighbour (i-1, j, k, l+1)... are the same seed region)
    for i, p in enumerate(P if G.FOLD else []):
        if p[0] == p[3] + 1:
            j = index.get((p[0], p[2], p[1], p[3]))         # (the point whose swap image is p's mirror neighbour)
            if j is not None and typ[i] == typ[j]:
                parent[find(i)] = find(j)
    comps = defaultdict(list)
    for i in range(len(P)):
        comps[find(i)].append(i)
    region = {}
    for c, members in comps.items():
        t = int(typ[members[0]])
        region.setdefault(t, []).append(len(members))
    # (surfaces: the files' and every region component's forms, until no new one)
    surf_text = set(surfaces_from_files())
    reps = [P[members[len(members) // 2]] for members in comps.values()]
    more = pool.map(forms_job, [tuple(np.array(r, float) / N) for r in reps], chunksize=1)
    for m in more:
        surf_text |= m
    surfs = [Surface(t) for t in sorted(surf_text)]
    print(f"{len(comps)} grid regions; {len(surfs)} candidate surfaces; {time.time() - t0:.0f}s", flush=True)
    # (walls: bisect each change between grid neighbours; up to CAP crossings for each pair of polytopes)
    bypair = defaultdict(list)
    for i, j in pairs:
        bypair[tuple(sorted((int(typ[i]), int(typ[j]))))].append((i, j) if typ[i] < typ[j] else (j, i))
    todo = []
    for pr, lst in bypair.items():
        for i, j in lst[:: max(1, len(lst) // CAP)][:CAP]:
            todo.append((Bp[i], Bp[j], R[i], R[j]))
    res = pool.map(bisect_job, todo, chunksize=1)
    ends = [(x, T.of(*la, a), T.of(*lb, b)) for (x, la, lb, a, b) in res]
    print(f"{len(ends)} wall crossings bisected; {time.time() - t0:.0f}s", flush=True)
    # (crossings with no known surface through them: add the forms of the polytopes on both sides)
    miss = [(x, a, b) for (x, ta, tb), (_, _, _, a, b) in zip(ends, res) if min((s.dist(x) for s in surfs), default=1.0) >= 1e-7]
    if miss:
        more = pool.map(forms_job, [tuple(p) for _, a, b in miss for p in (a, b)], chunksize=1)
        new = set().union(*more) - surf_text
        surf_text |= new
        surfs = [Surface(t) for t in sorted(surf_text)]
        print(f"{len(miss)} crossings on no known surface: {len(new)} surfaces added", flush=True)
    walls, on = [], []
    for x, ta, tb in ends:
        d = np.array([s.dist(x) for s in surfs] or [1.0])     # (no surfaces (ixoic: no exact forms): the bisected point)
        si = int(d.argmin())
        y = surfs[si].project(x) if surfs and d[si] < 1e-7 else x
        on.append((y, si if surfs and d[si] < 1e-7 else -1, ta, tb, float(d[si])))
    W = pool.map(wall_label_job, [tuple(y) for y, *_ in on], chunksize=8)
    for (y, si, ta, tb, d), (sg, k) in zip(on, W):
        walls.append({"surface": surfs[si].text if si >= 0 else f"none (nearest {d:.0e})", "sides": sorted([ta, tb]),
                      "wall_type": T.of(sg, k, y), "point": y.round(13).tolist()})
    # (the fold plane: a 2D grid on b1 = b4)
    F = grid(N, fold=True) if G.FOLD else []
    Bf = generic(F, N, fold=True) if F else np.zeros((0, 4))
    Rf = pool.map(label_job, list(map(tuple, Bf)), chunksize=16) if len(Bf) else []
    ftyp = [T.of(sg, k, b) for (sg, k), b in zip(Rf, Bf)]
    pool.close()
    # (summary)
    info = []
    for t, (sg, b) in enumerate(T.reps):
        info.append({"type": t, "signature": sg, "vertices": int(sg.split(":")[0]) if sg != "none" else 0, "example": b,
                     "grid_regions": region.get(t, []), "grid_points": int((typ == t).sum()),
                     "fold_points": int(sum(1 for f in ftyp if f == t)),
                     "wall_of": sorted({(w["surface"], tuple(w["sides"])) for w in walls if w["wall_type"] == t
                                        and t not in w["sides"]})})
    wsum = Counter((w["surface"], tuple(w["sides"]), w["wall_type"]) for w in walls)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "N": N, "types": info,
               "walls": [{"surface": s, "sides": list(sd), "wall_type": wt, "crossings": n} for (s, sd, wt), n in wsum.most_common()],
               "wall_points": walls, "surfaces": [s.text for s in surfs], "seconds": round(time.time() - t0, 1)},
              open(OUT, "w"), indent=1)
    for d in info:
        print(f"type {d['type']:2}: {d['signature'][:70]:70}  regions {d['grid_regions']} fold {d['fold_points']}")
    for (s, sd, wt), n in wsum.most_common():
        print(f"  {n:4} crossings  {sd[0]:2}|{sd[1]:2} wall type {wt:2}  on {s[:80]}")
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
