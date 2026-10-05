"""How many connected pieces each region has inside a given fundamental domain, read from the classified seeds
(cohesive_domain.samples). Same rule as cohesive_domain.link_fast's inside stage: close same-region neighbours
join without classifying; links that would join separate pieces are checked by classifying points along them
(at most 8 per piece). Used to compare the 12-chamber domain with the half-cell.

    python domain_pieces.py chambers
"""
from __future__ import annotations

import collections
import json
import sys
from multiprocessing import Pool

import chamber_domain as chd
import cohesive_domain as cd
import dodeca_view as V
import numpy as np
from cellframe import TINV
from scipy.spatial import cKDTree


def place_in_chambers(X):
    """Chart positions (V1 dodecahedron chart) of each seed's image inside the 12 chosen chambers."""
    V.use_centre((1, 0, 0, 0))
    sel = chd.choose({})[0]
    Hs = [chd._halfspaces(c) for c in sel]
    Y = np.einsum("gij,nj->gni", V.E, X)
    den = Y @ V.C
    U = (Y @ V.B.T) / np.where(den > 0, den, 1)[..., None]
    out = np.full((len(X), 3), np.nan)
    for n in range(len(X)):
        for g in np.flatnonzero(den[:, n] > 0.3):
            if any(all(nv @ U[g, n] + k >= -1e-9 for nv, k in H) for H in Hs):
                out[n] = U[g, n]
                break
    return out, (V.C.copy(), V.B.copy())


def _check(job):
    from probe import _one
    seeds, lab = job
    for x in seeds:
        b = TINV @ x
        if _one(b / b.sum())[0] != lab:
            return False
    return True


def pieces(Q, lab, to_seed, tries=8, k=10, procs=4, log=print):
    N = len(Q)
    kd = cKDTree(Q)
    nn = kd.query(Q, k=2)[0][:, 1]
    uf = cd.UF(N)
    rest = []
    for t in set(lab):
        idx = np.array([i for i in range(N) if lab[i] == t])
        if len(idx) < 2:
            continue
        d, nb = cKDTree(Q[idx]).query(Q[idx], k=min(k + 1, len(idx)))
        for a, ds, js in zip(idx, d, nb):
            for dist, jj in zip(ds[1:], js[1:]):
                b = idx[jj]
                if dist <= 1.5 * max(nn[a], nn[b]) and lab[kd.query((Q[a] + Q[b]) / 2)[1]] == t:
                    uf.union(a, b)
                else:
                    rest.append((float(dist), int(a), int(b)))
    rest.sort()
    used = {}
    with Pool(procs) as pool:
        while True:
            pick, seen = [], set()
            for c in rest:
                ra, rb = uf.find(c[1]), uf.find(c[2])
                if ra == rb or used.get(ra, 0) >= tries or used.get(rb, 0) >= tries or frozenset((ra, rb)) in seen:
                    continue
                seen.add(frozenset((ra, rb)))
                pick.append(c)
                if len(pick) >= 200:
                    break
            if not pick:
                break
            jobs = []
            for _, a, b in pick:
                n = max(1, min(5, int(np.ceil(np.linalg.norm(Q[b] - Q[a]) / 0.012)) - 1))
                jobs.append(([to_seed(Q[a] + (Q[b] - Q[a]) * (j + 1) / (n + 1)) for j in range(n)], lab[a]))
            ok = pool.map(_check, jobs, chunksize=2)
            for (_, a, b), good in zip(pick, ok):
                ra, rb = uf.find(a), uf.find(b)
                used[ra] = used.get(ra, 0) + 1
                used[rb] = used.get(rb, 0) + 1
                if good and ra != rb:
                    uf.union(a, b)
                    used[uf.find(a)] = 0
            done = {id(c) for c in pick}
            rest = [c for c in rest if id(c) not in done]
            log(f"checked {len(pick)}, {sum(ok)} good, {len({uf.find(i) for i in range(N)})} pieces", flush=True)
    c = collections.defaultdict(set)
    for i in range(N):
        c[lab[i]].add(uf.find(i))
    return {t: len(s) for t, s in c.items()}


if __name__ == "__main__":
    pts, _ = cd.samples()
    lab = [t for _, t in pts]
    X = np.array([cd.seed(b) for b, _ in pts])
    Q, (C, B) = place_in_chambers(X)

    def to_seed(u):
        x = C + B.T @ u
        return x / np.linalg.norm(x)
    res = pieces(Q, lab, to_seed)
    json.dump(res, open("chamber_pieces.json", "w"), indent=1)
    for t, n in sorted(res.items(), key=lambda kv: -kv[1]):
        print(t, n)
