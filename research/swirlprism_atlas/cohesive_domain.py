"""A fundamental domain in which every region (3-dimensional class) is one connected piece.

The half-cell cuts most regions into several pieces. Each piece can be replaced by any of its images under the
swirlprism group: the result is still a fundamental domain. Pieces of one region that meet across the half-cell's
faces (in the 3-sphere) are glued by moving one onto the other: a spanning tree of those contacts gives every
piece its move. This only works for a region whose pieces are all linked, through contacts, to each other: a
region whose copies in the 3-sphere fall into several orbits can never be one piece in any fundamental domain.

Connectivity is read from classified seeds: the golden grid, the dense search and cohesive_samples.json, with
their images under the group near the cell. Two seeds of the same region are linked when they are neighbours in
the Delaunay tetrahedralisation (and the segment between them stays near them); links that join different
pieces are checked by classifying points along them.
"""
from __future__ import annotations

import json
import os

import numpy as np
from cellframe import T, TINV
from scipy.spatial import Delaunay, cKDTree

from dodeca_view import E

P_CELL = None


def regions():
    r1 = json.load(open("xloci_step1.json"))
    out = {t for t, r in r1.items() if r["kept"] == r["of"]}
    for s in json.load(open("extra_samples.json")):
        if s.get("kind") == "region":
            out.add(_relabel(s["sig"]))
    return out


def _relabel(sig_or_label):
    from probe import label_of_sig
    s = sig_or_label
    if s.startswith("new:"):
        s = s[4:]
    if ": cells" in s:
        return label_of_sig(s)
    return s


def samples():
    pts = []
    for b, lab in json.load(open("dense_search.json")):
        pts.append((b, _relabel(lab)))
    if os.path.exists("cohesive_samples.json"):
        for b, lab in json.load(open("cohesive_samples.json")):
            pts.append((b, _relabel(lab)))
    reg = regions()
    pts = [(np.asarray(b, float) / sum(b), lab) for b, lab in pts if lab in reg]
    return pts, reg


def seed(b):
    x = b @ T
    return x / np.linalg.norm(x)


def bary(X):
    B = X @ TINV.T
    return B / B.sum(axis=-1, keepdims=True)


def near_images(X, margin=0.35):
    """Images g.x of every seed whose barycentric coordinates (cell) are all >= -margin."""
    Y = np.einsum("gij,nj->gni", E, X)                      # (g, n, 4)
    S = Y @ TINV.T
    s = S.sum(axis=-1)
    ok = s > 1e-6
    B = S / np.where(ok, s, 1)[..., None]
    ok &= np.all(B >= -margin, axis=-1)
    g, n = np.nonzero(ok)
    return g, n, B[g, n]


def chart(B):
    from cell_atlas import P
    return B @ P


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[a] = b


def contacts(pts):
    """Delaunay links between same-region seeds (and their images): list of (j1, j2, h) meaning seed j1 is next to
    h . seed j2, h a group element index (identity for links inside the half-cell)."""
    X = np.array([seed(b) for b, _ in pts])
    lab = [l for _, l in pts]
    g, n, B = near_images(X)
    Q = chart(B)
    tri = Delaunay(Q)
    kd = cKDTree(Q)
    edges = set()
    for s in tri.simplices:
        for a in range(4):
            for c in range(a + 1, 4):
                i, j = s[a], s[c]
                if lab[n[i]] == lab[n[j]]:
                    edges.add((min(i, j), max(i, j)))
    key = {tuple(np.round(m.ravel(), 6)): k for k, m in enumerate(E)}
    ident = key[tuple(np.round(np.eye(4).ravel(), 6))]
    out = []
    for i, j in edges:
        mid = (Q[i] + Q[j]) / 2
        _, k = kd.query(mid)                                    # the midpoint must be nearest one of the ends
        if lab[n[k]] != lab[n[i]]:
            continue
        h = key[tuple(np.round((E[g[i]].T @ E[g[j]]).ravel(), 6))]
        out.append((int(n[i]), int(n[j]), int(h), Q[i], Q[j]))
    return out, ident, lab


def analyse(pts=None):
    if pts is None:
        pts, _ = samples()
    links, ident, lab = contacts(pts)
    N = len(pts)
    inside, quot = UF(N), UF(N)
    for i, j, h, *_ in links:
        quot.union(i, j)
        if h == ident:
            inside.union(i, j)
    by = {}
    for k in range(N):
        by.setdefault(lab[k], []).append(k)
    report = {}
    for t, ks in sorted(by.items(), key=lambda kv: -len(kv[1])):
        pieces = {inside.find(k) for k in ks}
        orbits = {quot.find(k) for k in ks}
        report[t] = {"samples": len(ks), "piecesInHalf": len(pieces), "orbits": len(orbits)}
    return report, links, inside, ident, lab


def to_beta(q):
    """Barycentric coordinates (cell) of a chart point, inside the cell or not."""
    from cell_atlas import P
    return np.linalg.solve(np.vstack([P.T, np.ones(4)]), np.append(q, 1.0))


def _check(job):
    """Do all the points on a link classify as its region?"""
    from probe import _one
    pts, lab = job
    return all(_one(to_beta(q))[0] == lab for q in pts)


def _probe_points(a, b, step=0.012):
    m = max(1, min(5, int(np.ceil(np.linalg.norm(b - a) / step)) - 1))
    return [a + (b - a) * (k + 1) / (m + 1) for k in range(m)]


def candidates(pts, k=10):
    """Same-region neighbour pairs among the seeds and their nearby images: (dist, j1, j2, h, qa, qb)."""
    X = np.array([seed(b) for b, _ in pts])
    lab = [l for _, l in pts]
    g, n, B = near_images(X)
    Q = chart(B)
    key = {tuple(np.round(m.ravel(), 6)): i for i, m in enumerate(E)}
    ident = key[tuple(np.round(np.eye(4).ravel(), 6))]
    home = np.flatnonzero(g == ident)                          # the seeds themselves
    out = {}
    for t in set(lab):
        idx = np.flatnonzero(np.array([lab[x] == t for x in n]))
        if len(idx) < 2:
            continue
        kd = cKDTree(Q[idx])
        mine = [i for i in home if lab[n[i]] == t]
        d, nb = kd.query(Q[mine], k=min(k + 1, len(idx)))
        for i, ds, js in zip(mine, d, nb):
            for dist, jj in zip(ds[1:], js[1:]):
                j = idx[jj]
                h = int(g[j])                                  # seed n[i] is next to h . seed n[j]
                if h == ident and n[j] == n[i]:
                    continue
                kk = (int(n[i]), int(n[j]), h)
                if kk not in out:
                    out[kk] = (float(dist), Q[i], Q[j])
    return [(d, a, b, h, qa, qb) for (a, b, h), (d, qa, qb) in sorted(out.items(), key=lambda kv: kv[1][0])], ident, lab


def link(pts, procs=4, batch=400, log=print):
    """Grow the pieces: inside the half-cell first (links with no move), then across its faces. Each link is
    accepted only if points along it classify as the region."""
    from multiprocessing import Pool
    cands, ident, lab = candidates(pts)
    N = len(pts)
    inside, quot = UF(N), UF(N)
    cross = []
    with Pool(procs) as pool:
        for stage in ("inside", "across"):
            todo = [c for c in cands if (c[3] == ident) == (stage == "inside")]
            while todo:
                uf = inside if stage == "inside" else quot
                todo = [c for c in todo if uf.find(c[1]) != uf.find(c[2])]
                if not todo:
                    break
                head, todo = todo[:batch], todo[batch:]
                ok = pool.map(_check, [(_probe_points(c[4], c[5]), lab[c[1]]) for c in head], chunksize=4)
                for c, good in zip(head, ok):
                    if good and uf.find(c[1]) != uf.find(c[2]):
                        uf.union(c[1], c[2])
                        if stage == "across":
                            cross.append(c)
                log(f"{stage}: checked {len(head)}, {sum(ok)} good, {len(todo)} left")
            if stage == "inside":
                for a in range(N):
                    quot.union(a, inside.find(a))
    return inside, quot, cross, ident, lab


def link_fast(pts, procs=4, batch=200, tries=8, log=print):
    """Faster linking. Links between close neighbours inside the half-cell (no farther apart than 1.5 times the
    local seed spacing, with the midpoint nearest a seed of the same region) are taken without classifying. Only
    links that would join separate pieces are checked by classifying points along them, nearest first, at most
    `tries` per piece and stage."""
    from multiprocessing import Pool
    cands, ident, lab = candidates(pts)
    N = len(pts)
    X = np.array([seed(b) for b, _ in pts])
    Q = chart(bary(X))
    kd = cKDTree(Q)
    nn = kd.query(Q, k=2)[0][:, 1]
    inside, quot = UF(N), UF(N)
    rest = []
    for c in cands:
        d, a, b, h, qa, qb = c
        if h == ident and d <= 1.5 * max(nn[a], nn[b]):
            _, m = kd.query((qa + qb) / 2)
            if lab[m] == lab[a]:
                inside.union(a, b)
                continue
        rest.append(c)
    log(f"{N} seeds; unchecked short links leave {len({inside.find(k) for k in range(N)})} pieces")
    cross = []
    with Pool(procs) as pool:
        for stage in ("inside", "across"):
            uf = inside if stage == "inside" else quot
            if stage == "across":
                for a in range(N):
                    quot.union(a, inside.find(a))
            todo = [c for c in rest if (c[3] == ident) == (stage == "inside")]
            used = {}
            while True:
                pick, seen = [], set()
                for c in todo:
                    ra, rb = uf.find(c[1]), uf.find(c[2])
                    if ra == rb or used.get(ra, 0) >= tries or used.get(rb, 0) >= tries:
                        continue
                    key = frozenset((ra, rb))
                    if key in seen:
                        continue
                    seen.add(key)
                    pick.append(c)
                    if len(pick) >= batch:
                        break
                if not pick:
                    break
                ok = pool.map(_check, [(_probe_points(c[4], c[5]), lab[c[1]]) for c in pick], chunksize=2)
                for c, good in zip(pick, ok):
                    ra, rb = uf.find(c[1]), uf.find(c[2])
                    used[ra] = used.get(ra, 0) + 1
                    used[rb] = used.get(rb, 0) + 1
                    if good and ra != rb:
                        uf.union(c[1], c[2])
                        used[uf.find(c[1])] = 0
                        if stage == "across":
                            cross.append(c)
                done = {id(c) for c in pick}
                todo = [t for t in todo if id(t) not in done]
                log(f"{stage}: checked {len(pick)}, {sum(ok)} good, "
                    f"{len({uf.find(k) for k in range(N)})} pieces or orbits left")
    return inside, quot, cross, ident, lab


def report(pts, inside, quot, lab):
    by = {}
    for k in range(len(pts)):
        by.setdefault(lab[k], []).append(k)
    return {t: {"samples": len(ks), "piecesInHalf": len({inside.find(k) for k in ks}),
                "orbits": len({quot.find(k) for k in ks})} for t, ks in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def placements(pts, inside, cross, ident, lab):
    """A group element for every piece: grow each region from its biggest piece, adding pieces across checked
    links and keeping the moved pieces as close to the cell centre as the links allow."""
    import heapq
    X = np.array([seed(b) for b, _ in pts])
    c0 = seed(np.array([1, 1, 1, 1.0]))
    members = {}
    for k in range(len(pts)):
        members.setdefault(inside.find(k), []).append(k)
    cen = {p: X[ks].mean(axis=0) for p, ks in members.items()}
    adj = {}
    for _, a, b, h, *_ in cross:
        pa, pb = inside.find(a), inside.find(b)
        adj.setdefault(pa, []).append((pb, E[h]))
        adj.setdefault(pb, []).append((pa, E[h].T))
    place = {}
    by = {}
    for p, ks in members.items():
        by.setdefault(lab[ks[0]], []).append(p)
    for t, ps in by.items():
        for root in sorted(ps, key=lambda p: -len(members[p])):
            if root in place:
                continue                                       # another orbit of this region: it stays apart
            heap, n = [(0.0, 0, root, np.eye(4))], 1
            while heap:                                        # best first: the nearest placement wins
                _, _, a, M = heapq.heappop(heap)
                if a in place:
                    continue
                place[a] = M
                for b, h in adj.get(a, []):
                    if b not in place:
                        Mb = M @ h
                        heapq.heappush(heap, (-float((Mb @ cen[b]) @ c0), n, b, Mb))
                        n += 1
    return place, members


def build_view(pts, place, members, lab):
    """The moved region seeds in a gnomonic chart about their mean."""
    X = np.array([seed(b) for b, _ in pts])
    Y = np.zeros_like(X)
    for p, ks in members.items():
        Y[ks] = X[ks] @ place[p].T
    c = Y.mean(axis=0)
    c /= np.linalg.norm(c)
    perp = np.linalg.svd(c[None])[2][1:]
    U = (Y @ perp.T) / (Y @ c)[:, None]
    U -= U.mean(axis=0)
    w, v = np.linalg.eigh(np.cov(U.T))                         # the long axis horizontal, the short one vertical
    U = U @ v[:, ::-1]
    samples = {}
    for k, (b, t) in enumerate(pts):
        bb = np.round(b / b.max(), 4)
        samples.setdefault(t, []).append(U[k].tolist() + [f"β ∝ ({', '.join(f'{x:g}' for x in bb)}), moved with its piece"])
    out = {"samples": samples, "uniform": [], "special": [], "rings": [], "main": [], "segments": [], "qaxes": [],
           "regular": [], "xlines": [], "xwalls": [], "tpatches": [], "mirrors": [], "fdomain": [], "edges": [],
           "edgesOnRing": [], "dodecaFaces": [], "split": [], "fcentre": [0, 0, 0],
           "bounds": [U.min(axis=0).tolist(), U.max(axis=0).tolist()], "minDotCentre": float((Y @ c).min())}
    return out


def main():
    pts, _ = samples()
    inside, quot, cross, ident, lab = link_fast(pts)
    rep = report(pts, inside, quot, lab)
    place, members = placements(pts, inside, cross, ident, lab)
    view = build_view(pts, place, members, lab)
    view["cohesion"] = rep
    json.dump(view, open("cohesive_view.json", "w"))
    for t, r in rep.items():
        print(t, r)


if __name__ == "__main__":
    main()
