"""Regions, walls, lines and points of a swirl group's fundamental domain.

    SWIRL_GROUP=tet python build2.py survey_tet.json data_tet.json

Method (all numbers are measured, none assumed):
 1. The survey gives the volume types (combinatorial type of the orbit's hull) on a grid.
 2. Label changes between neighbouring samples are bisected to wall points. At a wall point the 5-subsets of orbit
    points that are coplanar name the wall: F_S(u) = det[1, g_i s(u)] vanishes on it. Wall points with the same
    coplanar subsets lie on one surface F_S = 0.
 3. Sign patterns of the surfaces' F decide the type (checked on samples), so regions are meshed from a fine grid,
    walls from the zero sets of F where the two sides are different regions, lines from where two walls meet.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

import domain as D
import walls as W

E = D.ELEMENTS
RNG = np.random.default_rng(5)
HS = D.domain_halfspaces()
HN = np.linalg.norm(HS[:, :3], axis=1)
V = D.domain_vertices()
LO, HI = V.min(0), V.max(0)
TOL_SUBSET = 1e-7


def margin(p):
    return np.min(-(np.asarray(p) @ HS[:, :3].T + HS[:, 3]) / HN, axis=-1)


def sig(u, tol=1e-9):
    return D.signature(D.classify(D.seed_from_chart(np.asarray(u, float)), tol))


def random_points(n):
    out = []
    while len(out) < n:
        X = RNG.uniform(LO, HI, size=(4 * n, 3))
        out.extend(X[margin(X) > 0].tolist())
    return np.array(out[:n])


def F(sub, U):
    """Coplanarity determinant of the orbit points g_i s for the subset, at chart points U (N x 3)."""
    U = np.atleast_2d(U)
    s = D.seed_from_chart(U)
    pts = np.einsum("kab,nb->nka", E[list(sub)], s)
    M = np.concatenate([np.ones((len(U), 5, 1)), pts], axis=2)
    return np.linalg.det(M)


def grad(sub, U, eps=1e-6):
    U = np.atleast_2d(U)
    g = np.zeros_like(U)
    for ax in range(3):
        d = np.zeros(3)
        d[ax] = eps
        g[:, ax] = (F(sub, U + d) - F(sub, U - d)) / (2 * eps)
    return g


def project(subs, p, iters=30):
    """Newton projection of p onto the common zero set of F_sub for the subsets (least-norm steps)."""
    p = np.array(p, float)
    for _ in range(iters):
        vals = np.array([F(s, p)[0] for s in subs])
        if np.abs(vals).max() < 1e-14:
            break
        J = np.array([grad(s, p)[0] for s in subs])
        p = p - np.linalg.pinv(J) @ vals
    return p


def bisect(a, b, la, steps=50):
    for _ in range(steps):
        m = (a + b) / 2
        if sig(m) == la:
            a = m
        else:
            b = m
    return (a + b) / 2


def _bis(t):
    return bisect(*t, steps=40)


def find_walls(survey, n_pairs):
    d = json.load(open(survey))
    P = np.array(d["points"])
    S = d["sigs"]
    h = d["h"]
    look = {tuple(np.round((p - P.min(0)) / h).astype(int)): i for i, p in enumerate(P)}
    pairs = []
    for k, i in look.items():
        for ax in range(3):
            k2 = list(k)
            k2[ax] += 1
            j = look.get(tuple(k2))
            if j is not None and S[i] != S[j] and "none" not in (S[i], S[j]):
                pairs.append((i, j))
    print(len(pairs), "label changes in the survey; types:", len(set(S)))
    # sample so that each unordered type pair gets coverage
    by_pair = defaultdict(list)
    for i, j in pairs:
        by_pair[frozenset((S[i], S[j]))].append((i, j))
    chosen = []
    for k, lst in by_pair.items():
        idx = RNG.choice(len(lst), min(len(lst), n_pairs), replace=False)
        chosen += [lst[t] for t in idx]
    print(len(by_pair), "type pairs,", len(chosen), "bisections", flush=True)
    from multiprocessing import Pool
    with Pool(2) as pool:
        qs = pool.map(_bis, [(P[i], P[j], S[i]) for i, j in chosen], chunksize=8)
    pts = [q for q in qs if margin(q) > 1e-6]
    return np.array(pts), sorted(set(S))


def cluster_surfaces(pts):
    structural = None
    for u in random_points(4):
        s = W.near_zero_subsets(D.seed_from_chart(u), TOL_SUBSET)
        structural = s if structural is None else structural & s
    sets = [frozenset(W.near_zero_subsets(D.seed_from_chart(q), TOL_SUBSET)) - structural for q in pts]
    groups = defaultdict(list)
    for q, s in zip(pts, sets):
        groups[s].append(q)
    # a point on a junction has the union of the sets of the surfaces through it: drop those
    keys = list(groups)
    clusters = []
    for k in keys:
        if not k:
            continue
        if any(o < k for o in keys) or any(len(o | p) == len(k) and o != k and p != k and o | p == k for o in keys for p in keys if o < k and p < k):
            continue
        clusters.append((k, np.array(groups[k])))
    clusters.sort(key=lambda kv: -len(kv[1]))
    return clusters, structural


def monos(d):
    return [(i, j, k) for t in range(d + 1) for i in range(t + 1) for j in range(t + 1 - i) for k in [t - i - j]]


def design(P, ms):
    P = np.atleast_2d(P)
    return np.stack([P[:, 0] ** i * P[:, 1] ** j * P[:, 2] ** k for i, j, k in ms], 1)


def fit_surface(P):
    for d in range(1, 7):
        ms = monos(d)
        if len(P) < 2 * len(ms):
            return None
        A = design(P, ms)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        _, sv, vt = np.linalg.svd(A)
        if sv[-1] < 1e-7 * sv[0] and sv[-2] > 1e-5 * sv[0]:
            return d, ms, vt[-1]
    return None


def main(survey, out):
    pts, vol_sigs = find_walls(survey, int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    clusters, structural = cluster_surfaces(pts)
    print(len(pts), "wall points,", len(clusters), "surfaces with point counts", [len(c[1]) for c in clusters])
    fits = []
    for k, P in clusters:
        ft = fit_surface(P)
        print(" surface with", len(P), "points:", None if ft is None else f"degree {ft[0]}")
        if ft is not None:
            fits.append(ft)
    reps = fits

    # grid
    vol = np.prod(HI - LO)
    h = float((vol / 1.3e6) ** (1 / 3))
    h = max(h, 0.012)
    lo, hi = LO - 2 * h, HI + 2 * h
    axes = [np.arange(a, b + h, h) for a, b in zip(lo, hi)]
    shape = tuple(len(a) for a in axes)
    flat = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    print("grid", shape, "h", round(h, 4))
    dm = margin(flat)
    inside = dm > 0
    Fv = []
    for sub in reps:
        d_, ms, co = sub
        f = np.zeros(len(flat), np.float32)
        for a in range(0, len(flat), 200000):
            f[a:a + 200000] = design(flat[a:a + 200000], ms) @ co
        Fv.append(f.reshape(shape))
    Fv = np.array(Fv)
    gn = [np.sqrt(sum(g ** 2 for g in np.gradient(f, h)) ) + 1e-12 for f in Fv]
    dist = np.array([np.abs(f) / g for f, g in zip(Fv, gn)])      # distance to each surface (first order)
    sign = np.sign(Fv).astype(np.int8)

    # pattern -> type
    pat_id = np.zeros(shape, np.int64)
    for c in range(len(reps)):
        pat_id = pat_id * 3 + (sign[c] + 1)
    pids = np.unique(pat_id[inside.reshape(shape)])
    pat_sig = {}
    inside3 = inside.reshape(shape)
    margin3 = dm.reshape(shape)
    for pid in pids:
        mask = (pat_id == pid) & inside3
        idx = np.flatnonzero(mask.ravel())
        deep = idx[np.argsort(-margin3.ravel()[idx])[:3]]
        sigs = {sig(flat[i]) for i in deep}
        if len(sigs) > 1:
            print("warning: pattern", pid, "has several types", sigs)
        pat_sig[int(pid)] = sorted(sigs)[0]
    type_sigs = sorted(set(pat_sig.values()))
    tid = {s: i + 1 for i, s in enumerate(type_sigs)}
    type_grid = np.zeros(shape, np.int32)
    for pid, s in pat_sig.items():
        type_grid[(pat_id == pid) & inside3] = tid[s]
    print("volume types found on the grid:", len(type_sigs))

    # regions
    regions = []
    lab_grid = np.zeros(shape, np.int32)
    for s, t in tid.items():
        lab, n = ndimage.label(type_grid == t)
        for c in range(1, n + 1):
            cnt = int((lab == c).sum())
            if cnt < 6:
                continue
            regions.append({"type_sig": s, "mask": lab == c, "voxels": cnt})
    regions.sort(key=lambda r: -r["voxels"])
    for i, r in enumerate(regions, 1):
        lab_grid[r["mask"]] = i
        r["id"] = f"R{i}"
    print("regions:", [(r["id"], tid[r["type_sig"]], r["voxels"]) for r in regions])

    def to_world(v):
        return v * h + lo

    dmin = np.minimum(margin3, dist.min(0)) if len(dist) else margin3
    for r in regions:
        field = np.where(r["mask"], dmin, -dmin)
        field = np.where(inside3 | r["mask"], field, -np.abs(margin3))
        v, f, _, _ = marching_cubes(field, 0.0)
        r["verts"], r["faces"] = to_world(v), f
        r["volume"] = r["voxels"] * h ** 3

    def region_at(p):
        idx = np.round((p - lo) / h).astype(int)
        idx = np.clip(idx, 0, np.array(shape) - 1)
        return lab_grid[tuple(idx.T)]

    # walls
    wall_surf = []
    meshes = []
    for c, sub in enumerate(reps):
        v, f, _, _ = marching_cubes(Fv[c], 0.0)
        v = to_world(v)
        ok = np.all(margin(v)[f] > -h, axis=1)
        f = f[ok]
        cen = v[f].mean(1)
        # one Newton step onto the surface to cut the voxel error before probing the sides
        d_, ms, co = sub
        n = np.stack([(design(cen + e * 1e-5, ms) - design(cen - e * 1e-5, ms)) @ co / 2e-5 for e in np.eye(3)], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        a = region_at(cen + 1.5 * h * n)
        b = region_at(cen - 1.5 * h * n)
        meshes.append((v, f, cen, a, b))
        groups = defaultdict(list)
        for t, ra, rb in zip(f, a, b):
            if ra and rb and ra != rb:
                groups[(min(ra, rb), max(ra, rb))].append(t)
        for pair, tris in groups.items():
            tris = np.array(tris)
            area = float(0.5 * np.linalg.norm(np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]), axis=1).sum())
            if area < 0.004:
                continue
            wall_surf.append({"surface": c, "pair": pair, "tris": tris, "verts": v, "area": area})
    wall_surf.sort(key=lambda w: -w["area"])
    print("walls:", [(w["surface"], w["pair"], round(w["area"], 3)) for w in wall_surf])

    return dict(h=h, lo=lo, shape=shape, reps=reps, clusters=clusters, regions=regions, wall_surf=wall_surf, meshes=meshes,
                lab_grid=lab_grid, tid=tid, pat_sig=pat_sig, margin3=margin3, Fv=Fv, out=out)


if __name__ == "__main__":
    import pickle

    res = main(sys.argv[1], sys.argv[2])
    pickle.dump(res, open(sys.argv[2] + ".pkl", "wb"))
