"""Walls as sections of exact surfaces, split along every line where another surface meets them.

Every wall lies on an exact surface: a plane (fexact_*.json, through its three corners) or a curved surface
(xsurface.surface: a polynomial in the betas), together with the surface's images under the group elements the page
copies it by (copyOps), folded into the half-cell as the page folds them. Inside the half-cell these surfaces cut each
other along curves: where two walls meet (a line of the atlas), or where one surface passes through another (an
invisible line).

A wall's surface is meshed whole over the half-cell, then cut along each other surface's zero set, exactly: every mesh
triangle a surface's sign changes across is split along it, the new vertices put on both surfaces. So every line lies
along mesh edges, and the mesh falls into sections: neighbouring triangles on the same side of every surface. Each
section has one region on each side, the same over the whole section. A region is a cell of the arrangement (the
points on given sides of every surface), named once, by one classification, and kept in a table by its sides; a
section with the same region on both sides is not part of a wall and is dropped.

    python arrangement.py X3,X7,X10,X83,T2,X11      (writes arr_pieces/<wall>.json.gz, as region_pieces/)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

import region_cells as rc
import region_walls as rw

PHI = (1 + 5 ** 0.5) / 2
OUT = "arr_pieces"
H = 0.004                 # (mesh spacing in the cell's chart)
STEP = 2e-5               # (off a section, along the surface's normal, to name the regions either side)
FOLD = np.eye(4)[[1, 0, 3, 2]]


class Surface:
    """f(B) = 0 for betas B (n, 4), homogeneous (no normalising, so the sign is continuous)."""

    def __init__(self, name, f, walls, kind):
        self.name, self.f, self.walls, self.kind = name, f, walls, kind

    def at(self, X):
        """Value at cell points X (n, 3)."""
        return self.f(np.c_[X, np.ones(len(X))] @ rc.M_INV.T)

    def grad(self, X, e=1e-7):
        return np.stack([(self.at(X + e * np.eye(3)[k]) - self.at(X - e * np.eye(3)[k])) / (2 * e) for k in range(3)], 1)


def _poly(mons, coeffs):
    mons = [tuple(m) for m in mons]
    coeffs = np.asarray(coeffs, float)
    return lambda B: sum(c * np.prod(B[:, list(m)], axis=1) for c, m in zip(coeffs, mons))


def surfaces():
    """Every surface a wall lies on, and its images in the half-cell; the same surface found twice is kept once."""
    import xsurface
    d = rw.page_data()
    out = []
    # planes of the flat walls, each through its three corners
    for fn in sorted(glob.glob("fexact_*.json")):
        e = json.load(open(fn))
        if not isinstance(e, dict) or "corners" not in e:
            continue                          # (the grids beside them)
        C = np.array(json.loads(e["corners"]) if isinstance(e["corners"], str) else e["corners"], float)
        n = np.linalg.svd(C)[2][-1]
        out.append(Surface(f"{e['target']}:{e['wall']}", (lambda n: lambda B: B @ n)(n), [e["target"]], "plane"))
    # (and the planes written in the page's key for the flat walls: "wall: β1 = β2, β2 = β3 + (1+φ)·β4 (exact patches)")
    import sympy as sp
    b = sp.symbols("b1:5")
    phi_ = (1 + sp.sqrt(5)) / 2
    for t in d["types"]:
        loc = str(t.get("locus") or "")
        if t.get("dim") != "wall" or not loc.startswith("wall: ") or "(exact patches)" not in loc:
            continue
        for eq in loc[6:loc.index("(exact patches)")].split(","):
            if "=" not in eq:
                continue
            txt = eq.replace("β", "b").replace("φ", "phi").replace("·", "*").replace("−", "-")
            lhs, rhs = txt.split("=")
            try:
                e = sp.sympify(lhs, {"phi": phi_, **{f"b{k + 1}": b[k] for k in range(4)}}) - \
                    sp.sympify(rhs, {"phi": phi_, **{f"b{k + 1}": b[k] for k in range(4)}})
            except (sp.SympifyError, SyntaxError, TypeError):
                continue
            n = np.array([float(sp.diff(e, v)) for v in b])
            out.append(Surface(f"{t['id']}:{eq.strip()}", (lambda n: lambda B: B @ n)(n), [t["id"]], "plane"))
    # curved walls: the equation, and its images by the page's copy elements (each folded too)
    eqs = set(json.load(open("xsurface_equations.json"))) | {"X10", "X38", "X83", "X48", "X51", "X33"}
    ops = defaultdict(set)
    for w in d["xwalls"]:
        ops[w["id"]] |= set(map(str, w.get("copyOps", [])))
    for tid in sorted(eqs):
        s = xsurface.surface(tid)
        f = _poly(s["mons"], s["coeffs"])
        out.append(Surface(tid, f, [tid], "orig"))
        out.append(Surface(f"{tid}:fold", (lambda f: lambda B: f(B @ FOLD.T))(f), [tid], "copy"))
        for g in sorted(ops[tid]):
            A = np.linalg.inv(np.asarray(d["copyMats"][g], float))
            out.append(Surface(f"{tid}:{g}", (lambda f, A: lambda B: f(B @ A.T))(f, A), [tid], "copy"))
            out.append(Surface(f"{tid}:{g}:fold", (lambda f, A: lambda B: f(B @ FOLD.T @ A.T))(f, A), [tid], "copy"))
    # (one of each: compared by their values at random points, up to a factor; and only those passing through the
    # half-cell, which change sign over it)
    rng = np.random.default_rng(0)
    P = rc.xyz(rng.dirichlet(np.ones(4), 64))
    Bh = rng.dirichlet(np.ones(4), 200000)
    Bh = Bh[Bh[:, 2] >= Bh[:, 3]]
    Ph = rc.xyz(Bh)
    keep, seen = [], []
    for s in out:
        v = s.at(P)
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            continue
        vh = s.at(Ph)
        if vh.min() > 0 or vh.max() < 0:
            continue
        v = v / nv
        hit = next((k for k, u in seen if min(np.linalg.norm(v - u), np.linalg.norm(v + u)) < 1e-9), None)
        if hit is None:
            seen.append((len(keep), v))
            keep.append(s)
        else:
            for w in s.walls:
                if w not in keep[hit].walls:
                    keep[hit].walls.append(w)
    return keep


# the half-cell: beta >= 0 and beta3 >= beta4
DOMAIN = [Surface(f"b{k + 1}>=0", (lambda k: lambda B: B[:, k])(k), [], "domain") for k in range(4)] + \
         [Surface("b3>=b4", lambda B: B[:, 2] - B[:, 3], [], "domain")]


def project(X, surfs, iters=30):
    """Points moved onto the common zero set of the surfaces (least-norm Newton steps)."""
    X = np.array(X, float)
    for _ in range(iters):
        F = np.stack([s.at(X) for s in surfs], 1)
        if np.abs(F).max() < 1e-15:
            break
        J = np.stack([s.grad(X) for s in surfs], 1)                    # (n, k, 3)
        X = X - np.einsum("nik,nk->ni", np.linalg.pinv(J), F)
    return X


class Mesh:
    def __init__(self, V, T, surf):
        self.V, self.T, self.surf = np.asarray(V, float), [tuple(t) for t in T], surf
        self.on = defaultdict(set)          # vertex -> the cutting surfaces it lies on

    def cut(self, s, si):
        """Split every triangle the surface s changes sign across, along it; new vertices on both surfaces."""
        g = s.at(self.V)
        scale = np.abs(g).max()
        if scale < 1e-14:
            return 0
        g[np.abs(g) < 1e-12 * scale] = 0.0
        new_v, cache, out, n_cut = [], {}, [], 0

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                t = g[i] / (g[i] - g[j])
                cache[k] = len(self.V) + len(new_v)
                new_v.append((1 - t) * self.V[i] + t * self.V[j])
                self.on[cache[k]] |= (self.on[i] & self.on[j]) | {si}
            return cache[k]
        for tri in self.T:
            sg = np.sign(g[list(tri)])
            if not ((sg > 0).any() and (sg < 0).any()):
                out.append(tri)
                continue
            n_cut += 1
            a, b, c = tri
            # (rotated so the vertex on the surface, if one is, else the one alone on its side, comes first)
            zero = 0 in sg
            for _ in range(3):
                sa, sb, sc = np.sign(g[[a, b, c]])
                if (sa == 0) if zero else (sa != sb and sa != sc):
                    break
                a, b, c = b, c, a
            if zero:                         # (b and c on opposite sides)
                p = mid(b, c)
                self.on[a].add(si)
                out += [(a, b, p), (a, p, c)]
            else:                            # (a alone on its side)
                p, q = mid(a, b), mid(a, c)
                out += [(a, p, q), (p, b, c), (p, c, q)]
        if new_v:
            W = np.array(new_v)
            # (each new vertex put on this mesh's surface and on the cutting surface)
            W = project(W, [s] if self.surf is None else [self.surf, s])
            self.V = np.vstack([self.V, W])
        self.T = out
        return n_cut


def carriers(surfs, path="arr_carriers.json"):
    """The surfaces that hold the page's walls (most of a piece of a wall's mesh lies on it), with those walls: only
    they cut the others, along the lines where they meet them and on past (invisible lines)."""
    names = [s.name for s in surfs]
    if os.path.exists(path):
        c = json.load(open(path))
        if c.get("names") == names:
            return c["carry"]
    d = rw.page_data()
    rng = np.random.default_rng(0)
    carry = defaultdict(set)
    for w, k, i, T in rw.pieces_of(d, {x["id"] for x in d["xwalls"]} | {"T2"}):
        j = rng.integers(0, len(T), 400)
        u, v = rng.random(400), rng.random(400)
        fl = u + v > 1
        u[fl], v[fl] = 1 - u[fl], 1 - v[fl]
        V = T[j, 0] + u[:, None] * (T[j, 1] - T[j, 0]) + v[:, None] * (T[j, 2] - T[j, 0])
        for s in surfs:
            g = np.abs(s.at(V)) / (np.linalg.norm(s.grad(V), axis=1) + 1e-30)
            if (g < 3e-4).mean() > 0.3:
                carry[s.name].add(w)
    carry = {k: sorted(v) for k, v in carry.items()}
    json.dump({"names": names, "carry": carry}, open(path, "w"))
    return carry


def mesh_surface(s, h=H):
    """The surface's zero set over the half-cell, marching cubes on the cell's chart, vertices put on it exactly."""
    from skimage.measure import marching_cubes
    corners = rc.xyz(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.5, 0.5]], float))
    lo, hi = corners.min(0) - 2 * h, corners.max(0) + 2 * h
    n = np.ceil((hi - lo) / h).astype(int) + 1
    G = np.stack(np.meshgrid(*[lo[k] + h * np.arange(n[k]) for k in range(3)], indexing="ij"), -1)
    vals = s.at(G.reshape(-1, 3)).reshape(G.shape[:3])
    if vals.min() > 0 or vals.max() < 0:
        return None
    V, T, _, _ = marching_cubes(vals, 0.0, spacing=(h, h, h))
    V = project(V + lo, [s])
    m = Mesh(V, T, s)
    # (cut to the half-cell, and the outside dropped)
    for k, dom in enumerate(DOMAIN):
        m.cut(dom, -1 - k)
    B = lambda X: np.c_[X, np.ones(len(X))] @ rc.M_INV.T
    C = np.array([m.V[list(t)].mean(0) for t in m.T])
    Bc = B(C)
    inside = (Bc >= -1e-12).all(1) & (Bc[:, 2] - Bc[:, 3] >= -1e-12)
    m.T = [t for t, ok in zip(m.T, inside) if ok]
    return m


def sections(m, cutters):
    """Sections of the mesh: neighbouring triangles on the same side of every cutting surface."""
    C = np.array([m.V[list(t)].mean(0) for t in m.T])
    C = project(C, [m.surf])
    # (a triangle's side of a cutting surface: that of its corner furthest from it; the cuts leave every triangle on
    # one side, with its other corners on the surface, so a sliver beside a line is not split off by rounding)
    Ti = np.array(m.T)
    S = np.zeros((len(m.T), len(cutters)), np.int8)
    for k, s in enumerate(cutters):
        g = s.at(m.V)[Ti]                                          # (triangles, 3)
        S[:, k] = np.sign(g[np.arange(len(Ti)), np.abs(g).argmax(1)])
    parent = list(range(len(m.T)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_edge = defaultdict(list)
    for i, (a, b, c) in enumerate(m.T):
        for e in ((a, b), (b, c), (c, a)):
            by_edge[(min(e), max(e))].append(i)
    for tris in by_edge.values():
        for i in tris[1:]:
            if (S[i] == S[tris[0]]).all():
                parent[find(i)] = find(tris[0])
    return np.array([find(i) for i in range(len(m.T))]), S, C


def cell_key(surfs, X):
    """A hash of the sides of every surface at points X: points with the same key lie in the same cell."""
    rng = np.random.default_rng(12345)
    W = rng.integers(1, 2 ** 62, size=len(surfs), dtype=np.int64)
    h = np.zeros(len(X), np.int64)
    with np.errstate(over="ignore"):
        for w, s in zip(W, surfs):
            h += w * (np.sign(s.at(X)).astype(np.int64) + 1)
    return h


def cell_table(surfs, path="arr_cells.npz"):
    """The region database's voxels (region_grid.json) sorted into the arrangement's cells: for each cell, the
    region most of its voxels belong to and how many. Kept on disk for the same list of surfaces."""
    names = np.array([s.name for s in surfs])
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if list(z["names"]) == list(names):
            return dict(zip(z["keys"].tolist(), zip(z["regions"].tolist(), z["counts"].tolist(), z["shares"].tolist())))
    g = json.load(open("region_grid.json"))
    sh, lo, h = tuple(g["shape"]), np.array(g["lo"]), g["h"]
    regs = sorted(g["runs"])
    idx, lab = [], []
    for k, r in enumerate(regs):
        ii = np.concatenate([np.arange(a, a + n) for a, n in g["runs"][r]]) if g["runs"][r] else np.zeros(0, int)
        idx.append(ii)
        lab.append(np.full(len(ii), k))
    idx, lab = np.concatenate(idx), np.concatenate(lab)
    X = lo + (np.array(np.unravel_index(idx, sh)).T + 0.5) * h
    keys = np.concatenate([cell_key(surfs, X[i:i + 500000]) for i in range(0, len(X), 500000)])
    order = np.lexsort((lab, keys))
    keys, lab = keys[order], lab[order]
    table = {}
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    for a, b in zip(starts, ends):
        c = np.bincount(lab[a:b])
        table[int(keys[a])] = (regs[int(c.argmax())], int(b - a), float(c.max() / (b - a)))
    np.savez(path, names=names, keys=np.array(list(table)), regions=np.array([v[0] for v in table.values()]),
             counts=np.array([v[1] for v in table.values()]), shares=np.array([v[2] for v in table.values()]))
    return table


def main(walls):
    from region_db import Labels
    labels = Labels()
    surfs = surfaces()
    carry = carriers(surfs)
    for s in surfs:
        s.walls = carry.get(s.name, s.walls)
    surfs = [s for s in surfs if s.name in carry]          # (the arrangement: the surfaces walls lie on)
    want = set(walls.split(","))
    print(f"{len(surfs)} surfaces ({sum(s.kind == 'plane' for s in surfs)} planes)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    table = cell_table(surfs)
    print(f"{len(table)} cells of the arrangement hold the database's voxels", flush=True)
    for s in surfs:
        if not set(s.walls) & want:
            continue
        m = mesh_surface(s)
        if m is None or not m.T:
            continue
        others = [c for c in surfs if c is not s]
        n_cuts = sum(m.cut(c, i) > 0 for i, c in enumerate(others))
        comp, S, C = sections(m, others)
        A = np.array([0.5 * np.linalg.norm(np.cross(m.V[t[1]] - m.V[t[0]], m.V[t[2]] - m.V[t[0]])) for t in m.T])
        # (each section: the middle of its biggest triangle, a short step either side along the surface's normal)
        lead = {}
        for i, c in enumerate(comp):
            if c not in lead or A[i] > A[lead[c]]:
                lead[c] = i
        ids = sorted(lead)
        P = np.array([C[lead[c]] for c in ids])
        N = s.grad(P)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        # (each side's cell: named from the region database's voxels in it; the classifier only for a cell too thin
        # to hold any)
        Q = np.concatenate([P + STEP * N, P - STEP * N])
        K = cell_key(surfs, Q)
        name = {}
        thin = [j for j, k in enumerate(K) if k not in table or table[k][1] < 3]
        got = labels(rc.betas(Q[thin])) if thin else []
        for j, l in zip(thin, got):
            name[j] = l
        for j, k in enumerate(K):
            if j not in name:
                name[j] = table[k][0]
        sides = {c: (name[j], name[len(ids) + j]) for j, c in enumerate(ids)}
        n_thin = len(thin)
        faces = []
        kept = 0.0
        for c, p, n in zip(ids, P, N):
            a, b = sides[c]
            if a == b:
                continue
            T = [m.T[i] for i in np.flatnonzero(comp == c)]
            tris = np.array([m.V[list(t)] for t in T])
            kept += A[comp == c].sum()
            faces.append({"sides": [str(a), str(b)], "kind": s.kind, "mirror": False, "surface": s.name,
                          "cut": [bool(~rc.inside(rc.betas((p + sg * 4e-4 * n)[None]))[0]) for sg in (1, -1)],
                          "tris": tris.reshape(-1, 9).round(7).tolist()})
        wid = next(w for w in s.walls if w in want)
        fn = os.path.join(OUT, f"{wid}.json.gz")
        old = json.load(gzip.open(fn, "rt")) if os.path.exists(fn) else []
        old = [f for f in old if f.get("surface") != s.name] + faces
        json.dump(old, gzip.open(fn, "wt"), separators=(",", ":"))
        print(f"  {s.name:28s} {len(m.T):6d} triangles, cut by {n_cuts} surfaces, {len(ids)} sections, "
              f"{len(faces)} of them wall ({kept / max(A.sum(), 1e-30):.0%} of the area); "
              f"{2 * len(ids) - n_thin} sides named from the database, {n_thin} classified (cells too thin for voxels)",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
