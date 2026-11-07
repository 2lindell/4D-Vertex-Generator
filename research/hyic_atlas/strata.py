"""The toxitic / oxitic shapes on the domain's faces, edges and special lines (strata_<sub>.json).

The domain is the contic half-chamber beta >= 0, b1 >= b4 (contic_atlas/group.py's frame); its interior is mapped by
contic_atlas/snub_map.py. This maps the lower strata:

  faces   b2 = 0, b3 = 0, b4 = 0 (F4 mirrors) and the fold b1 = b4;
  lines   the chamber's edges in the domain (b2 = b3 = 0, b2 = b4 = 0, b3 = b4 = 0, b1 = b4 = 0, and the fold's edges
          b1 = b4 with b2 = 0 or b3 = 0) and the fold line b1 = b4, b2 = b3;
  points  the domain's corners, and (0,1,1,0) where the fold line meets the edge b1 = b4 = 0.

Each stratum gets a grid (shifted off the lattice inside it), classified (group.label); grid neighbours of different
polytopes are bisected inside the stratum and the polytope at the change classified with the wall tolerance (1e-7).
Every polytope found gets its symmetry: the elements of H4 [5,3,3] (14400) keeping its vertex set, so a
shape with more than the subgroup's symmetry (on a mirror: icopyritic for toxitic; on a ring or a point) is told apart.

    HYIC_SUB=ki|ix python strata.py [N]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import group as G  # noqa: E402

FULL = None


def contic():
    """H4 (the full hyic group): the symmetries a polytope's vertex set is tested against."""
    global FULL
    if FULL is None:
        FULL = G._closure(G.REFL)
    return FULL


def symmetry(beta):
    """The order of the polytope's symmetry inside [[3,4,3]], and whether it has improper (mirror) symmetries."""
    V, _ = G.vertices(beta)
    tree = cKDTree(V)
    tol = 1e-7 * np.abs(V).max()
    n, improper = 0, 0
    for g in contic():
        d, _ = tree.query(V @ g.T)
        if d.max() < tol:
            n += 1
            improper += int(np.linalg.det(g) < 0)
    return n, improper


# (the strata: name, the zero betas, fold b1 = b4, fold line b2 = b3 too)
STRATA = [("face b1 = 0", [0], False, False), ("face b2 = 0", [1], False, False), ("face b3 = 0", [2], False, False),
          ("face b4 = 0", [3], False, False)] + \
         [(f"line b{i + 1} = b{j + 1} = 0", [i, j], False, False) for i in range(4) for j in range(i + 1, 4)] + \
         [(f"point {tuple(int(k == m) for k in range(4))}".replace(" ", ""), [k for k in range(4) if k != m], False, False)
          for m in range(4)]


def lattice(N, zero, fold, line):
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            for k in range(N + 1 - i - j):
                p = (i, j, k, N - i - j - k)
                if any(p[z] != 0 for z in zero) or any(p[z] == 0 for z in range(4) if z not in zero):
                    continue
                if fold and p[0] != p[3]:
                    continue
                if not fold and G.FOLD and p[0] <= p[3]:
                    continue
                if line and p[1] != p[2]:
                    continue
                pts.append(p)
    return pts


SHIFT = np.array([np.sqrt(2) - 1.4, np.sqrt(3) - 1.7, np.sqrt(7) - 2.6, np.sqrt(11) - 3.3]) * 0.37  # (no sqrt5: the walls are over Q(sqrt5))


def generic(P, zero, fold, line):
    s = SHIFT.copy()
    s[zero] = 0
    if fold:
        s[3] = s[0]
    if line:
        s[2] = s[1]
    X = np.array(P, float) + s
    X[:, zero] = 0
    return X / X.sum(1, keepdims=True)


def label_job(item):
    b, tol = item
    try:
        sg, k = G.label(np.asarray(b), tol)
    except Exception:
        return "none", None
    return sg, k


def bisect_job(item):
    a, b, ka, kb = item
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = a, b
    for _ in range(36):
        m = (lo + hi) / 2
        k = label_job((m, 1e-9))[1]
        if k == ka:
            lo = m
        else:
            hi = m
    x = (lo + hi) / 2
    sg, k = label_job((x, 1e-7))
    return x.tolist(), sg, k


def sym_job(b):
    return symmetry(np.asarray(b))


def main():
    t0 = time.time()
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pool = Pool(4)
    types = {}                                    # key -> {signature, example, strata}
    out = {"_about": __doc__.strip().split("\n\n")[0], "group": G.SUB, "order": G.ORDER, "N": N, "strata": []}
    # (checkpoint after each stratum, so a restart resumes where it stopped)
    ck = os.path.join(HERE, f"strata_{G.SUB}.partial.json")
    if os.path.exists(ck):
        c = json.load(open(ck))
        if c["N"] == N:
            out["strata"] = c["strata"]
            types = {(None if k == "None" else k): {**t, "strata": set(t["strata"])} for k, t in c["types"].items()}
            print(f"resumed after {len(out['strata'])} strata", flush=True)
    done = {s["stratum"] for s in out["strata"]}
    for name, zero, fold, line in STRATA:
        if name in done:
            continue
        P = lattice(N if "point" not in name else 2, zero, fold, line)
        if "point" in name:
            X = np.zeros((1, 4))
            X[0, [z for z in range(4) if z not in zero]] = 1
            X /= X.sum()
            P = [tuple(X[0])]
        else:
            X = generic(P, zero, fold, line)
        L = pool.map(label_job, [(tuple(x), 1e-9) for x in X], chunksize=8)
        # (neighbours: lattice points within the stratum's step)
        pairs = []
        if len(P) > 1:
            T = cKDTree(np.array(P, float))
            pairs = [(i, j) for i, j in T.query_pairs(2.5) if L[i][1] != L[j][1]]
        bycount = defaultdict(list)
        for i, j in pairs:
            bycount[tuple(sorted((str(L[i][1]), str(L[j][1]))))].append((i, j))
        todo = []
        for lst in bycount.values():
            for i, j in lst[:: max(1, len(lst) // 6)][:6]:
                todo.append((tuple(X[i]), tuple(X[j]), L[i][1], L[j][1]))
        C = pool.map(bisect_job, todo, chunksize=1)
        found = Counter(k for _, k in L)
        changes = Counter()
        for (x, sg, k), (a, b, ka, kb) in zip(C, todo):
            changes[(tuple(sorted((str(ka), str(kb)))), str(k))] += 1
            types.setdefault(k, {"signature": sg, "example": x, "strata": set(), "at_change": True})
            types[k]["strata"].add(name)
        for (sg, k), x in zip(L, X):
            t = types.setdefault(k, {"signature": sg, "example": x.tolist(), "strata": set(), "at_change": False})
            t["strata"].add(name)
            t["at_change"] = False
        out["strata"].append({"stratum": name, "points": len(P), "types": {str(k): n for k, n in found.items()},
                              "changes": [{"sides": list(s), "at": k, "crossings": n} for (s, k), n in changes.items()]})
        json.dump({"N": N, "strata": out["strata"],
                   "types": {str(k): {**t, "strata": sorted(t["strata"])} for k, t in types.items()}}, open(ck, "w"))
        print(f"{name:24} {len(P):5} points  {len(found)} types  {len(changes)} kinds of change  {time.time() - t0:.0f}s",
              flush=True)
    keys = [k for k in types if k is not None]
    S = pool.map(sym_job, [types[k]["example"] for k in keys], chunksize=1)
    pool.close()
    out["types"] = []
    for k, (n, imp) in zip(keys, S):
        t = types[k]
        out["types"].append({"key": k, "signature": t["signature"], "vertices": int(t["signature"].split(":")[0]),
                             "example": list(map(float, t["example"])), "strata": sorted(t["strata"]),
                             "only_at_a_change": t["at_change"], "symmetry_in_contic": int(n), "improper": int(imp)})
    for t in sorted(out["types"], key=lambda t: (t["vertices"], t["strata"])):
        print(f"  {t['key']}  {t['vertices']:4}v  sym {t['symmetry_in_contic']:4} ({t['improper']} improper)  "
              f"{'change ' if t['only_at_a_change'] else '        '}{', '.join(t['strata'])[:70]:70}  "
              f"{t['signature'].split('|')[0][5:80]}", flush=True)
    json.dump(out, open(os.path.join(HERE, f"strata_{G.SUB}.json"), "w"), indent=1)
    os.remove(ck)
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
