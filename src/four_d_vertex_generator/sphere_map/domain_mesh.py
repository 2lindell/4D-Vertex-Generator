"""Exact meshes of every wall piece in the domain, for the domain viewer.

Each wall patch (`patches.json`) is exactly the points of its surface where every other library
surface has the sign its two pieces share (a face patch: the points of a domain face where side a
has piece a's signs and side b, read in its copy, piece b's). It is meshed so:

* the surface is triangulated over the whole domain (marching tetrahedra on a grid; a domain
  face on a lattice), every vertex solved onto the surface exactly;
* each patch keeps the triangles where its signs hold, and the triangles its rim crosses are cut
  at the rim: every new vertex solved exactly onto the surface and the crossing surface (onto
  both crossing surfaces where two meet: the patch's corners);
* each rim stretch is named by what lies round it in the pieces: a named line (the atlas's line
  pieces, checked on their own exact equations), another wall meeting this one, the domain's face,
  or a seam (the wall goes on across it: no curve there);
* patches are joined into wall pieces as `wall_pieces.py` joins them (cut only by named lines and
  wall-wall meeting curves, over every copy), and wherever two patches of one wall share a seam.

Writes `domain_mesh.npz` (vertices in domain betas, triangles, rims) and `domain_mesh.json`.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .chords import FACE_N, FACE_NAMES
from .frame import cell_vertices, data_dir, group

N_GRID = 72                 # grid steps per unit beta (marching tetrahedra)
N_FACE = 96                 # lattice steps along a domain face's side
FACE_TRI = {"b1=0": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5]],
            "b2=0": [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5]],
            "b3=b4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, .5, .5]]}
KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]


# ------------------------------------------------------------------------------- functions

class Fn:
    """A signed constraint c(x) >= 0 on domain betas x (sum 1): value and sum-zero gradient."""

    def __init__(self, surf=None, sign=1.0, M=None, normal=None, tag=None):
        self.surf, self.sign, self.M, self.normal, self.tag = surf, float(sign), M, normal, tag

    def v(self, X):
        X = np.atleast_2d(X)
        if self.normal is not None:
            return self.sign * (X @ self.normal)
        Y = X if self.M is None else X @ self.M
        return self.sign * self.surf.f(Y)

    def g(self, x):
        if self.normal is not None:
            gr = self.sign * self.normal
        elif self.M is None:
            gr = self.sign * self.surf.grad(x[None])[0]
        else:
            gr = self.sign * (self.M @ self.surf.grad((x @ self.M)[None])[0])
        return gr - gr.mean()

    def __call__(self, x):
        return float(self.v(x)[0]), self.g(x)


def solve(fs, x, iters=30):
    x = np.asarray(x, float).copy()
    for _ in range(iters):
        v = np.array([f(x)[0] for f in fs])
        if np.abs(v).max() < 1e-15:
            break
        J = np.array([f(x)[1] for f in fs])
        x = x + np.linalg.lstsq(np.vstack([J, np.ones(4)]), np.r_[-v, 1.0 - x.sum()], rcond=None)[0]
    return x, bool(np.abs([f(x)[0] for f in fs]).max() < 1e-12)


def project(surf, X, iters=8):
    X = X.copy()
    for _ in range(iters):
        f = surf.f(X)
        g = surf.grad(X)
        g = g - g.mean(1, keepdims=True)
        gg = np.einsum("ij,ij->i", g, g)
        X = X - (f / np.where(gg > 0, gg, 1.0))[:, None] * g
    return X


# ------------------------------------------------------------------------------- base meshes

def surface_mesh(surf, n=N_GRID):
    """The surface over the domain: marching tetrahedra on a grid in (b1, b2, b3), every vertex
    solved onto the surface. Returns (V betas, F)."""
    h = 1.0 / n
    ii = np.array([(i, j, k) for i in range(n) for j in range(n - i) for k in range(n - i - j)
                   if 2 * (k + 1) >= n - i - j - 2])               # (cubes reaching b3 >= b4)
    corner = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])
    G = (ii[:, None, :] + corner[None]).reshape(-1, 3)
    key = G[:, 0] * (n + 2) ** 2 + G[:, 1] * (n + 2) + G[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    P = np.stack([uk // (n + 2) ** 2, (uk // (n + 2)) % (n + 2), uk % (n + 2)], 1) * h
    B = np.c_[P, 1 - P.sum(1)]
    val = surf.f(B)
    cid = inv.reshape(-1, 8)
    tets = np.concatenate([cid[:, list(t)] for t in KUHN])
    s = val[tets] > 0
    cnt = s.sum(1)
    tris, emap, epts = [], {}, []

    def ev(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in emap:
            emap[k] = len(epts)
            fa, fb = val[a], val[b]
            t = fa / (fa - fb)
            epts.append(B[a] + t * (B[b] - B[a]))
        return emap[k]
    for T, sg, c in zip(tets[(cnt > 0) & (cnt < 4)], s[(cnt > 0) & (cnt < 4)], cnt[(cnt > 0) & (cnt < 4)]):
        pos, neg = T[sg], T[~sg]
        if c == 1 or c == 3:
            lone, rest = (pos[0], neg) if c == 1 else (neg[0], pos)
            tris.append([ev(lone, r) for r in rest])
        else:
            a, b = ev(pos[0], neg[0]), ev(pos[0], neg[1])
            c_, d = ev(pos[1], neg[1]), ev(pos[1], neg[0])
            tris += [[a, b, c_], [a, c_, d]]
    V = np.array(epts)
    W = project(surf, V)
    ok = (np.abs(surf.f(W)) < 1e-12) & (np.linalg.norm(W - V, axis=1) < 2 * h)
    F = np.array(tris, int)
    F = F[ok[F].all(1)]
    return W, F


def face_mesh(face, n=N_FACE):
    A = np.asarray(FACE_TRI[face], float)
    pts, at = [], {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            at[i, j] = len(pts)
            pts.append(((n - i - j) * A[0] + i * A[1] + j * A[2]) / n)
    F = []
    for i in range(n):
        for j in range(n - i):
            F.append([at[i, j], at[i + 1, j], at[i, j + 1]])
            if i + j < n - 1:
                F.append([at[i + 1, j], at[i + 1, j + 1], at[i, j + 1]])
    return np.array(pts), np.array(F)


# ------------------------------------------------------------------------------- clipping

def clip(poly, cons, base):
    """A convex polygon [(x, tags)] on the base surface(s) cut by constraints [(tag, Fn)] (kept
    where >= 0). New vertices are solved onto the base, the constraint, and a constraint both ends
    of the cut edge lie on (a corner). Returns the polygon."""
    for tag, fn in cons:
        if len(poly) < 3:
            return []
        vals = [float(fn.v(x)[0]) if tag not in tg else 0.0 for x, tg in poly]
        if min(vals) >= 0:
            continue
        if max(vals) <= 0:
            return []
        out = []
        for i in range(len(poly)):
            (xa, ta), (xb, tb) = poly[i], poly[(i + 1) % len(poly)]
            fa, fb = vals[i], vals[(i + 1) % len(poly)]
            if fa >= 0:
                out.append((xa, ta))
            if (fa > 0 and fb < 0) or (fa < 0 and fb > 0):
                x0 = xa + fa / (fa - fb) * (xb - xa)
                common = [t for t in ta & tb if t in base["fns"]]
                x, ok = (None, False)
                if common:
                    x, ok = solve(base["on"] + [base["fns"][common[0]], fn], x0)
                    if ok and np.linalg.norm(x - x0) < 4 * base["h"]:
                        out.append((x, frozenset({tag, common[0]})))
                        continue
                x, ok = solve(base["on"] + [fn], x0)
                if not ok or np.linalg.norm(x - x0) > 4 * base["h"]:
                    x = x0
                out.append((x, frozenset({tag})))
        poly = out
    return poly if len(poly) >= 3 else []


# ------------------------------------------------------------------------------- the build

def _inv_table(mult):
    return np.array([int(np.flatnonzero(mult[g] == 0)[0]) for g in range(len(mult))])


def build(log=print, only=None, keys_wanted=None):
    from .atlas_db import _library_surfaces, _piece_keys
    from .frame import multiplication_table
    from .wall_meets import resolve
    root = data_dir() / "sphere_map"
    surfs = _library_surfaces(root)
    keys = _piece_keys(root)
    pieces = json.loads((root / "pieces.json").read_text())["pieces"]
    patches = json.loads((root / "patches.json").read_text())["patches"]
    sign = np.array([[1 if c == "+" else -1 if c == "-" else 0 for c in p["signs"]] for p in pieces],
                    np.int8)
    region = [p["region"] for p in pieces]
    pw = {}
    for p in patches:
        if p["face"] is None and p["wall"]:
            a, b = p["pieces"]
            pw[(min(a, b), max(a, b), p["surface"])] = p
    T = cell_vertices()
    g = group()
    Tinv = np.linalg.inv(T)
    mult = multiplication_table()
    L = len(surfs)
    face_fns = {f"face:{FACE_NAMES[i]}": Fn(normal=FACE_N[i], tag=f"face:{FACE_NAMES[i]}")
                for i in range(len(FACE_N))}

    # named lines in the domain: the atlas's line pieces, each segment with its exact equations
    atlas = json.loads((root / "atlas.json").read_text())
    lpts, lseg = [], []
    seg_fns = []
    for lp in atlas["line_pieces"]:
        for s in lp["segments"]:
            P = np.asarray(s["polyline_beta"], float)
            P = P / P.sum(1, keepdims=True)
            try:
                fns = [resolve(e) for e in s["equations"]]
            except KeyError:
                fns = None
            k = len(seg_fns)
            seg_fns.append((lp["line_shape"], fns))
            for a, b in zip(P[:-1], P[1:]):
                m = max(1, int(np.ceil(np.linalg.norm(b - a) / 2e-3)))
                for t in np.linspace(0, 1, m + 1):
                    lpts.append(a + (b - a) * t)
                    lseg.append(k)
    ltree = cKDTree(np.array(lpts))

    def named_line(x):
        """The named line through x (domain beta), from the line pieces' exact equations."""
        idx = ltree.query_ball_point(x, 2e-2)
        for k in sorted({lseg[i] for i in idx}):
            name, fns = seg_fns[k]
            if fns and all(abs(float(f.f(x[None])[0])) < 1e-9 * (1 + np.linalg.norm(f.grad(x[None])[0]))
                           for f in fns):
                return name
        return None

    def piece_of(sg):
        return keys.get(np.asarray(sg, np.int8).tobytes())

    def classify(pt, wall, s, t, a):
        """What the rim of a patch of `wall` on surface s along surface t is, at pt: the walls on
        the four sides round the curve s = t = 0 (pieces from piece a's signs)."""
        sa = sign[a].copy()
        quad = {}
        for i in (1, -1):
            for j in (1, -1):
                q = sa.copy()
                q[s], q[t] = i, j
                quad[i, j] = piece_of(q)
        here = int(sa[t])
        def wall_between(p, q, on):
            if p is None or q is None:
                return None
            r = pw.get((min(p, q), max(p, q), on))
            return r["wall"] if r else None
        beyond_s = wall_between(quad[1, -here], quad[-1, -here], s)
        on_t = {wall_between(quad[i, 1], quad[i, -1], t) for i in (1, -1)} - {None}
        line = named_line(pt)
        cont = beyond_s == wall or wall in on_t
        others = sorted(on_t - {wall} | ({beyond_s} - {wall, None} if False else set()))
        if line:
            return ("line", line, cont)
        if others:
            return ("meet", "/".join(others), cont)
        if cont:
            return ("seam", None, True)
        return ("open", None, False)

    out_V, out_F, patch_tris = [], [], {}
    rims = defaultdict(list)                 # patch id -> [(kind, name, cont, a, b)]
    nv = 0
    by_surface = defaultdict(list)
    for p in patches:
        if not p["wall"] or (only and p["wall"] not in only):
            continue
        by_surface[p["surface"] if p["face"] is None else "face:" + p["face"]].append(p)
    stats = defaultdict(int)
    for key, plist in sorted(by_surface.items(), key=lambda kv: str(kv[0])):
        if keys_wanted is not None and str(key) not in keys_wanted:
            continue
        if isinstance(key, str):
            V, F = face_mesh(key[5:])
            ffn = face_fns[key]
            base = {"on": [Fn(normal=FACE_N[FACE_NAMES.index(key[5:])])], "fns": {}, "h": 1.0 / N_FACE}
            s_idx = None
        else:
            V, F = surface_mesh(surfs[key])
            base = {"on": [Fn(surfs[key])], "fns": {}, "h": 1.0 / N_GRID}
            s_idx = key
        Sv = np.sign(np.stack([sf.f(V) for sf in surfs], 1)).astype(np.int8)
        FV = V @ FACE_N.T
        for p in plist:
            a, b = p["pieces"]
            sa = sign[a].astype(np.int8).copy()
            if s_idx is not None:
                sa[s_idx] = 0
            mism = (Sv * sa) < 0
            fmis = FV < -1e-13
            if isinstance(key, str):
                fmis[:, FACE_NAMES.index(key[5:])] = False
            cons_all = {}
            for j in range(L):
                if sa[j]:
                    cons_all[j] = Fn(surfs[j], sa[j], tag=j)
            for i in range(len(FACE_N)):
                cons_all[f"face:{FACE_NAMES[i]}"] = face_fns[f"face:{FACE_NAMES[i]}"]
            Mb = None
            if isinstance(key, str):
                c = p["side_b_copy_shift"]
                Mb = T @ g[c] @ Tinv
                Sb = np.sign(np.stack([sf.f(V @ Mb) for sf in surfs], 1)).astype(np.int8)
                mism_b = (Sb * sign[b]) < 0
                for j in range(L):
                    if sign[b][j]:
                        cons_all[f"b:{j}"] = Fn(surfs[j], sign[b][j], M=Mb, tag=f"b:{j}")
            else:
                mism_b = np.zeros((len(V), 0), bool)
            M = np.concatenate([mism, fmis, mism_b], 1)
            names = list(range(L)) + [f"face:{n}" for n in FACE_NAMES] + [f"b:{j}" for j in range(mism_b.shape[1])]
            excl = (M[F[:, 0]] & M[F[:, 1]] & M[F[:, 2]]).any(1)
            bad_v = M.any(1)
            full = ~bad_v[F].any(1)
            part = ~excl & ~full
            tris = [V[F[full]]]
            base["fns"] = cons_all
            rim = []
            for f in np.flatnonzero(part):
                cols = np.flatnonzero(M[F[f]].any(0))
                cons = [(names[c_], cons_all[names[c_]]) for c_ in cols if names[c_] in cons_all]
                poly = clip([(V[v], frozenset()) for v in F[f]], cons, base)
                if not poly:
                    continue
                X = np.array([x for x, _ in poly])
                for k in range(1, len(X) - 1):
                    tris.append(X[[0, k, k + 1]][None])
                for k in range(len(poly)):
                    (xa, ta), (xb, tb) = poly[k], poly[(k + 1) % len(poly)]
                    com = ta & tb
                    if com:
                        rim.append((sorted(com, key=str)[0], xa, xb))
            Tp = np.concatenate(tris) if tris else np.zeros((0, 3, 4))
            patch_tris[p["id"]] = Tp
            # name each rim stretch (by its middle, solved onto both surfaces)
            for tag, xa, xb in rim:
                if isinstance(tag, str) and tag.startswith("face:"):
                    rims[p["id"]].append(("face", tag[5:], False, xa, xb))
                    stats["face"] += 1
                    continue
                m0 = (xa + xb) / 2
                m, ok = solve(base["on"] + [cons_all[tag]], m0)
                if not ok:
                    m = m0
                if isinstance(tag, str):          # (a face patch's side b)
                    line = named_line(m) or named_line(m @ Mb / (m @ Mb).sum())
                    kind = ("line", line, False) if line else ("meet", None, False)
                elif s_idx is None:
                    line = named_line(m)
                    kind = ("line", line, False) if line else ("meet", None, False)
                else:
                    kind = classify(m, p["wall"], s_idx, tag, a)
                rims[p["id"]].append((*kind, xa, xb))
                stats[kind[0]] += 1
        log(f"surface {key}: {len(plist)} patches, "
            f"{sum(len(patch_tris[p['id']]) for p in plist)} triangles")
    log(dict(stats))
    return patch_tris, dict(rims)


def _job(keys):
    return build(log=lambda *_: None, keys_wanted=set(keys))


def build_parallel(log=print, workers=4):
    """build() over the surfaces, shared out between processes."""
    from multiprocessing import Pool
    patches = json.loads((data_dir() / "sphere_map" / "patches.json").read_text())["patches"]
    load = defaultdict(int)
    for p in patches:
        if p["wall"]:
            load[str(p["surface"]) if p["face"] is None else "face:" + p["face"]] += 1
    jobs = [[] for _ in range(4 * workers)]
    tot = [0] * len(jobs)
    for k, n in sorted(load.items(), key=lambda kv: -kv[1]):
        i = int(np.argmin(tot))
        jobs[i].append(k)
        tot[i] += n
    pt, rm = {}, {}
    with Pool(workers) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(_job, [j for j in jobs if j])):
            pt.update(a)
            rm.update(b)
            log(f"mesh jobs done: {i + 1}")
    return pt, rm


def exact_lines(step=4e-3):
    """Every named line's stretches in the domain (the atlas's line pieces), resampled every
    `step` and each point solved onto the segment's two exact surfaces."""
    from .wall_meets import resolve
    root = data_dir() / "sphere_map"
    atlas = json.loads((root / "atlas.json").read_text())
    out, seen = [], set()
    for lp in atlas["line_pieces"]:
        for s in lp["segments"]:
            P = np.asarray(s["polyline_beta"], float)
            P = P / P.sum(1, keepdims=True)
            key = tuple(np.round(np.sort(np.r_[P[0], P[-1], P[len(P) // 2]]), 6))
            if key in seen:
                continue
            seen.add(key)
            fns = [resolve(e) for e in s["equations"]]
            fs = [Fn(f) for f in fns]
            pts = [P[0]]
            for a, b in zip(P[:-1], P[1:]):
                m = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
                for t in np.linspace(0, 1, m + 1)[1:]:
                    x0 = a + (b - a) * t
                    x, ok = solve(fs, x0)
                    pts.append(x if ok and np.linalg.norm(x - x0) < 1e-2 else x0)
            out.append({"line": lp["line_shape"], "piece": lp["id"], "copy": s["copy"],
                        "pts": np.array(pts)})
    return out


def export(log=print):
    """domain_mesh.npz / domain_mesh.json: the wall pieces in the domain, their rims, the lines."""
    from . import wall_pieces
    root = data_dir() / "sphere_map"
    patches = json.loads((root / "patches.json").read_text())["patches"]
    import pickle
    cache = root / ".cache" / "domain_mesh_build.pkl"
    if cache.exists() and cache.stat().st_mtime > (root / "patches.json").stat().st_mtime:
        patch_tris, rims = pickle.loads(cache.read_bytes())
    else:
        patch_tris, rims = build_parallel(log=log)
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps((patch_tris, rims)))
    wp = wall_pieces.build(log=lambda *_: None)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    cls_of = {}
    for wall, v in wp.items():
        for pid, (root_, ci) in v["_domain"].items():
            join(("p", pid), ("r", wall, root_))
            cls_of[pid] = (wall, ci)
    seam_at = defaultdict(list)
    for pid, rl in rims.items():
        for kind, name, cont, xa, xb in rl:
            if kind == "seam":
                k = tuple(sorted([tuple(np.round(xa, 8)), tuple(np.round(xb, 8))]))
                seam_at[(patches[pid]["wall"], k)].append(pid)
    joined_by_seam = 0
    for ps in seam_at.values():
        for q in ps[1:]:
            if find(("p", q)) != find(("p", ps[0])):
                joined_by_seam += 1
            join(("p", ps[0]), ("p", q))
    groups = defaultdict(list)
    for pid in patch_tris:
        groups[find(("p", pid))].append(pid)
    pieces, T, TP, R, RK = [], [], [], [], []
    kinds = ["line", "meet", "open", "seam", "face"]
    rim_names = []
    for g_, pids in sorted(groups.items(), key=lambda kv: (patches[kv[1][0]]["wall"], min(kv[1]))):
        wall = patches[pids[0]]["wall"]
        k = len(pieces)
        regs = sorted({"|".join(sorted(patches[p]["regions"])) for p in pids})
        cl = sorted({cls_of[p][1] for p in pids if p in cls_of})
        on_sphere = sum(wp[wall]["classes"][c]["count_on_sphere"] for c in cl) if wall in wp else None
        ntri = 0
        for p in sorted(pids):
            T.append(patch_tris[p])
            TP.append(np.full(len(patch_tris[p]), k))
            ntri += len(patch_tris[p])
            for kind, name, cont, xa, xb in rims.get(p, []):
                R.append(np.stack([xa, xb]))
                nm = name or ""
                if nm not in rim_names:
                    rim_names.append(nm)
                RK.append([k, kinds.index(kind), rim_names.index(nm), int(bool(cont)), p])
        pieces.append({"id": k, "wall": wall, "between": regs, "patches": sorted(pids),
                       "classes": cl, "classes_on_sphere": on_sphere, "triangles": ntri})
    lines = exact_lines()
    np.savez_compressed(
        root / "domain_mesh.npz",
        tris=np.concatenate(T).astype(np.float32), tri_piece=np.concatenate(TP).astype(np.int32),
        rims=np.array(R, np.float32), rim_info=np.array(RK, np.int32),
        line_pts=np.concatenate([l["pts"] for l in lines]).astype(np.float32),
        line_len=np.array([len(l["pts"]) for l in lines], np.int32))
    doc = {"_about": __doc__ + "\nnpz: tris (n,3,4) betas, tri_piece (n) piece id; rims (m,2,4) "
           "betas, rim_info (m,5) [piece, kind, name, the wall goes on across it, patch]; "
           "line_pts / line_len: the named lines' stretches (lines below, in order).",
           "rim_kinds": kinds, "rim_names": rim_names, "pieces": pieces,
           "lines": [{k: v for k, v in l.items() if k != "pts"} for l in lines],
           "joined_by_seam": joined_by_seam,
           "rim_counts": {kd: int(sum(1 for r in RK if r[1] == i)) for i, kd in enumerate(kinds)}}
    (root / "domain_mesh.json").write_text(json.dumps(doc, indent=1))
    log(f"{len(pieces)} wall pieces, {sum(p['triangles'] for p in pieces)} triangles, "
        f"{len(R)} rim segments {doc['rim_counts']}, {len(lines)} line stretches; "
        f"{joined_by_seam} joins by shared seams")
    return doc


if __name__ == "__main__":
    export()
