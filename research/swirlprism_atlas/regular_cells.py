"""Where pentagonal-prism and pentagonal-antiprism cells of the swirlprism orbits become regular.

A cell is found by its faces: two pentagons and five quadrilaterals (prism) or two pentagons and ten
triangles (antiprism). Its defect is a vector of conditions that all vanish exactly when it is regular
(all edges one length, every face regular); the zero set of the defect is the locus asked for.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from cellframe import seed_from_beta

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.library import named_symmetry
from four_d_vertex_generator.off import compute_convex_hull

ACTION = named_symmetry("h4_swirlprism")
KINDS = {"prism": (5, 4, 5), "antiprism": (5, 3, 10)}   # two pentagons plus n faces of this size


def hull(beta):
    v = generate_vertices_from_seed(seed_from_beta(np.asarray(beta, float)), ACTION, tol=1e-9)
    faces, cells = compute_convex_hull(v, tol=1e-9)
    return v, faces, cells


def kind_of(cell, faces):
    sizes = sorted(len(faces[f]) for f in cell)
    for name, (p, q, n) in KINDS.items():
        if sizes == sorted([p, p] + [q] * n):
            return name
    return None


def defect(v, cell, faces):
    """Relative deviations from regularity: edge lengths, and each face's diagonals (equal in a regular face)."""
    edges = {tuple(sorted(e)) for f in cell for e in zip(faces[f], faces[f][1:] + faces[f][:1])}
    lengths = np.array([np.linalg.norm(v[a] - v[b]) for a, b in edges])
    unit = lengths.mean()
    out = list(lengths / unit - 1)
    for f in cell:
        face = faces[f]
        if len(face) > 3:
            k = len(face)
            diag = [np.linalg.norm(v[face[i]] - v[face[(i + 2) % k]]) for i in range(k)]
            want = unit * (2 * np.cos(np.pi / k) if k == 5 else 2 ** 0.5)   # diagonal of a regular k-gon
            out += [d / unit - want / unit for d in diag]
    return np.array(out)


def cell_defects(beta):
    """For each cell kind present: (count of cells, smallest max |defect| among them, components of the best)."""
    v, faces, cells = hull(beta)
    best = {}
    for c in cells:
        k = kind_of(c, faces)
        if not k:
            continue
        d = defect(v, c, faces)
        n, m, _ = best.get(k, (0, np.inf, None))
        best[k] = (n + 1, min(m, np.abs(d).max()), d if np.abs(d).max() < m else best.get(k, (0, 0, None))[2])
    return best


def tracked_defect(beta, kind, ref):
    """Defect of the cell of this kind whose centre points closest to the direction ref."""
    b = np.abs(np.asarray(beta, float))
    v, faces, cells = hull(b / b.sum())
    best, d = -2.0, None
    for c in cells:
        if kind_of(c, faces) != kind:
            continue
        centre = v[sorted({x for f in c for x in faces[f]})].mean(axis=0)
        cos = centre @ ref / np.linalg.norm(centre)
        if cos > best:
            best, d = cos, defect(v, c, faces)
    return d


def _solve(job):
    """Gauss-Newton from b0, following the one cell nearest the start's most regular cell."""
    from scipy.optimize import least_squares
    kind, b0 = job
    v, faces, cells = hull(b0)
    cand = [c for c in cells if kind_of(c, faces) == kind]
    c0 = min(cand, key=lambda c: np.linalg.norm(defect(v, c, faces)))
    ref = v[sorted({x for f in c0 for x in faces[f]})].mean(axis=0)
    ref /= np.linalg.norm(ref)
    n = len(tracked_defect(b0, kind, ref))

    def fun(x):
        try:
            d = tracked_defect(x, kind, ref)
        except Exception:
            return np.full(n + 1, 10.0)
        if d is None or len(d) != n:
            return np.full(n + 1, 10.0)
        return np.append(d, np.abs(x).sum() - 1)      # fix the scale
    try:
        r = least_squares(fun, b0, x_scale=0.1, diff_step=1e-6, xtol=1e-14, ftol=1e-14, max_nfev=120)
    except Exception as exc:
        return kind, b0.tolist(), None, repr(exc)
    b = np.abs(r.x) / np.abs(r.x).sum()
    return kind, b0.tolist(), b.tolist(), float(np.abs(r.fun[:-1]).max())


def find(kind, types):
    """Solve for a regular cell from the deepest golden sample of each given type (plus two random ones)."""
    from multiprocessing import Pool

    from cell_atlas2 import identify, load_refs
    refs = load_refs()[0]
    xids = json.load(open("atlas_xids.json"))
    by_type = {}
    for s in json.load(open("golden_22.json")):
        cid, xkey, _ = identify(s["sig"], refs)
        t = cid or xids.get(xkey)
        if t in types:
            b = np.array(s["beta"], float)
            by_type.setdefault(t, []).append(b / b.sum())
    rng = np.random.default_rng(0)
    starts = []
    for t, bs in sorted(by_type.items()):
        picks = [max(bs, key=lambda x: x.min())] + [bs[i] for i in rng.choice(len(bs), min(2, len(bs)), replace=False)]
        starts += [(kind, b) for b in picks]
    out = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(_solve, starts):
            out.append(r)
            print(np.round(r[1], 4).tolist(), "->", r[2] and np.round(r[2], 8).tolist(), r[3], flush=True)
    json.dump(out, open(f"regular_{kind}_minima.json", "w"))


def prism_residual(beta):
    """(side edge / pentagon edge - 1, full defect) for the pentagonal prisms at beta; None without prisms.

    Each prism class gives one value; they are returned sorted."""
    try:
        v, faces, cells = hull(beta)
    except Exception:
        return None
    vals = {}
    for c in cells:
        if kind_of(c, faces) != "prism":
            continue
        pent = [faces[f] for f in c if len(faces[f]) == 5]
        base = {tuple(sorted(e)) for f in pent for e in zip(f, f[1:] + f[:1])}
        edges = {tuple(sorted(e)) for f in c for e in zip(faces[f], faces[f][1:] + faces[f][:1])}
        side = edges - base
        lb = np.mean([np.linalg.norm(v[a] - v[b]) for a, b in base])
        ls = np.mean([np.linalg.norm(v[a] - v[b]) for a, b in side])
        r = ls / lb - 1
        vals.setdefault(round(r, 7), (r, float(np.abs(defect(v, c, faces)).max())))
    return sorted(vals.values()) or None


PRISM_PLANES = {   # the walls and faces that carry pentagonal prisms (golden samples), and their half-turn images
    "b2=0": [0, 1, 0, 0], "b1=0": [1, 0, 0, 0], "b1=b2": [1, -1, 0, 0],
    "b1=phi2*b2": [1, -(1 + 5 ** 0.5) / 2 - 1, 0, 0], "b2=phi2*b1": [-(1 + 5 ** 0.5) / 2 - 1, 1, 0, 0],
}


def _res(b):
    return prism_residual(b)


def _bis_prism(job, steps=40):
    a, b, ia, ib = job              # which prism class (by index) changes sign between a and b
    ra = prism_residual(a)[ia][0]
    for _ in range(steps):
        m = (a + b) / 2
        rm = prism_residual(m)
        if not rm:
            return None
        r = min(rm, key=lambda x: abs(x[0] - ra))[0] if len(rm) > 1 else rm[0][0]
        if np.sign(r) == np.sign(ra):
            a, ra = m, r
        else:
            b = m
    rm = prism_residual((a + b) / 2)
    return ((a + b) / 2).tolist(), rm


def prism_curves(n=12):
    """Marching triangles on each prism-carrying plane for side/base = 1, kept in the displayed half."""
    from multiprocessing import Pool

    from xloci import _order, _section
    jobs, tri = [], []
    for name, nrm in PRISM_PLANES.items():
        nrm = np.array(nrm, float)
        poly = _order(_section([nrm]), nrm)
        c = np.mean(poly, axis=0)
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            idx = {}
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    idx[(i, j)] = len(jobs)
                    jobs.append((i * a + j * b + (n - i - j) * c) / n)
            for i in range(n):
                for j in range(n - i):
                    tri.append((name, idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
                    if i + j + 2 <= n:
                        tri.append((name, idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    print(len(jobs), "grid points", flush=True)
    with Pool(4) as pool:
        res = pool.map(_res, jobs, chunksize=8)
        edges = {}
        for name, *vs in tri:
            for u, w in ((vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])):
                ru, rw = res[u], res[w]
                if ru and rw and len(ru) == len(rw):
                    for k in range(len(ru)):
                        if np.sign(ru[k][0]) != np.sign(rw[k][0]):
                            edges.setdefault((min(u, w), max(u, w), k), (name, jobs[u], jobs[w], k))
        keys = list(edges)
        hits = pool.map(_bis_prism, [(edges[e][1], edges[e][2], e[2], e[2]) for e in keys])
    crossing = {e: h for e, h in zip(keys, hits) if h}
    segs = []
    for name, *vs in tri:
        pts = [crossing[(min(u, w), max(u, w), k)][0] for u, w in ((vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0]))
               for k in range(3) if (min(u, w), max(u, w), k) in crossing]
        if len(pts) == 2:
            segs.append({"plane": name, "pts": pts})
    out = {"segments": segs, "points": [{"beta": h[0], "res": h[1]} for h in crossing.values()]}
    json.dump(out, open("regular_prism_curves.json", "w"))
    worst = max((r[1] for h in crossing.values() for r in (h[1] or [])), default=None)
    print(len(segs), "segments,", len(crossing), "crossings; largest full defect on them", worst)


def antiprism_residuals(v, faces, cells):
    """For each pentagonal antiprism: (centre, r1, r2, full defect), r1 and r2 the two kinds of side edge over
    the pentagon edge, minus 1. The kinds are told apart by their turn as seen from outside the cell, so the
    labels are the same for every antiprism and every seed."""
    out = []
    for c in cells:
        if kind_of(c, faces) != "antiprism":
            continue
        cv = sorted({x for f in c for x in faces[f]})
        centre = v[cv].mean(axis=0)
        frame = np.linalg.svd(v[cv] - centre)[2][:3]           # the cell's own 3-space
        local = {x: (v[x] - centre) @ frame.T for x in cv}
        pents = [faces[f] for f in c if len(faces[f]) == 5]
        base_len = np.mean([np.linalg.norm(v[a] - v[b]) for f in pents for a, b in zip(f, f[1:] + f[:1])])
        tris = [faces[f] for f in c if len(faces[f]) == 3]
        kinds = {1: [], -1: []}
        for top in pents:
            n = np.mean([local[x] for x in top], axis=0)        # outward, from the cell centre
            ring = list(top)
            a0, a1 = local[ring[0]], local[ring[1]]
            if np.cross(a0, a1) @ n < 0:                         # counter-clockwise seen from outside
                ring = ring[::-1]
            for k in range(5):
                a, b = ring[k], ring[(k + 1) % 5]
                apex = next(t for t in tris if a in t and b in t)
                w = next(x for x in apex if x not in (a, b))
                kinds[1].append(np.linalg.norm(v[a] - v[w]))
                kinds[-1].append(np.linalg.norm(v[b] - v[w]))
        out.append((centre, np.mean(kinds[1]) / base_len - 1, np.mean(kinds[-1]) / base_len - 1,
                    float(np.abs(defect(v, c, faces)).max())))
    return out


def _plane_basis():
    return np.linalg.svd(np.ones((1, 4)))[2][1:]                # three directions keeping sum(beta)


def _anti_track(beta, ref):
    b = np.abs(np.asarray(beta, float))
    v, faces, cells = hull(b / b.sum())
    res = antiprism_residuals(v, faces, cells)
    if not res:
        return None
    best = max(res, key=lambda r: r[0] @ ref / np.linalg.norm(r[0]))
    return np.array([best[1], best[2]]), best[3], best[0] / np.linalg.norm(best[0])


def _anti_solve(b0):
    """Gauss-Newton for r1 = r2 = 0, following the antiprism nearest the start's most regular one."""
    from scipy.optimize import least_squares
    v, faces, cells = hull(b0)
    res = antiprism_residuals(v, faces, cells)
    if not res:
        return b0.tolist(), None, "no antiprisms"
    c0 = min(res, key=lambda r: r[1] ** 2 + r[2] ** 2)
    ref = c0[0] / np.linalg.norm(c0[0])
    B = _plane_basis()

    def fun(x):
        try:
            t = _anti_track(b0 + x @ B, ref)
        except Exception:
            return np.array([10.0, 10.0])
        return np.array([10.0, 10.0]) if t is None else t[0]
    try:
        r = least_squares(fun, np.zeros(3), diff_step=1e-7, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=80)
    except Exception as exc:
        return b0.tolist(), None, repr(exc)
    b = np.abs(b0 + r.x @ B)
    b = b / b.sum()
    t = _anti_track(b, ref)
    return b0.tolist(), b.tolist(), None if t is None else (float(np.abs(t[0]).max()), t[1])


def antiprism_find(count=48):
    """Solve from golden samples spread over the cell (only those whose antiprisms are already close)."""
    from multiprocessing import Pool
    S = json.load(open("golden_22.json"))
    rng = np.random.default_rng(2)
    B = [np.array(s["beta"], float) / sum(s["beta"]) for s in S]
    B = [b for b in B if b.min() > 0.02]                        # inside, away from faces
    starts = [B[i] for i in rng.choice(len(B), min(count, len(B)), replace=False)]
    out = []
    with Pool(4) as pool:
        for r in pool.imap_unordered(_anti_solve, starts):
            out.append(r)
            print(np.round(r[0], 4).tolist(), "->", r[1] and np.round(r[1], 8).tolist(), r[2], flush=True)
            json.dump(out, open("regular_antiprism_minima.json", "w"))


def _anti_ref(b):
    v, faces, cells = hull(b)
    res = antiprism_residuals(v, faces, cells)
    c0 = min(res, key=lambda r: r[1] ** 2 + r[2] ** 2)
    return c0[0] / np.linalg.norm(c0[0])


def _trace_one(job, h=0.02, max_steps=150):
    """Follow the curve r1 = r2 = 0 from a solution, one way (sign), by predictor-corrector steps."""
    from scipy.optimize import least_squares
    b, sign = job
    b = np.asarray(b, float)
    ref = _anti_ref(b)
    B = _plane_basis()
    path, prev_t = [b.tolist()], None
    from probe import _one
    labels = [_one(b)[0]]
    for _ in range(max_steps):
        r0 = _anti_track(b, ref)[0]
        J = np.column_stack([(_anti_track(b + 1e-7 * B[k], ref)[0] - r0) / 1e-7 for k in range(3)])
        t = np.linalg.svd(J)[2][-1] @ B                       # tangent, in beta space
        t /= np.linalg.norm(t)
        if prev_t is None:
            t *= sign
        elif t @ prev_t < 0:
            t = -t
        pred = b + h * t
        N = np.linalg.svd(np.vstack([np.ones(4), t]))[2][2:]  # correct across the curve only

        def fun(x, pred=pred, N=N):
            try:
                tr = _anti_track(pred + x @ N, ref)
            except Exception:
                return np.array([10.0, 10.0])
            return np.array([10.0, 10.0]) if tr is None else tr[0]
        try:
            r = least_squares(fun, np.zeros(2), diff_step=1e-7, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=40)
        except Exception:
            break
        nb = pred + r.x @ N
        if nb.min() < -1e-12 or np.abs(r.fun).max() > 1e-8:
            break
        tr = _anti_track(nb, ref)
        ref = tr[2]
        prev_t, b = t, nb
        path.append(b.tolist())
        labels.append(_one(b)[0])
    return {"path": path, "labels": labels}


def antiprism_trace():
    """Trace the regular-antiprism curves from the distinct solutions found by antiprism_find."""
    from multiprocessing import Pool
    sols = [np.array(r[1]) for r in json.load(open("regular_antiprism_minima.json"))
            if r[1] and isinstance(r[2], list) and r[2][0] < 1e-8]
    seeds = []
    for s in sols:
        if not any(np.linalg.norm(s - q) < 1e-6 for q in seeds):
            seeds.append(s)
    print(len(seeds), "distinct solutions", flush=True)
    with Pool(4) as pool:
        out = pool.map(_trace_one, [(s.tolist(), sg) for s in seeds for sg in (1, -1)])
    json.dump(out, open("regular_antiprism_curves.json", "w"))
    for o in out:
        print(len(o["path"]), "steps", np.round(o["path"][0], 4).tolist(), "->", np.round(o["path"][-1], 4).tolist(),
              sorted(set(o["labels"])))


def main():
    if sys.argv[1] == "antiprism_trace":
        antiprism_trace()
        return
    if sys.argv[1] == "antiprism_find":
        antiprism_find()
        return
    if sys.argv[1] == "prism_curves":
        prism_curves()
        return
    if sys.argv[1] == "find":
        find(sys.argv[2], sys.argv[3].split(","))
        return
    if sys.argv[1] == "survey":
        from cell_atlas2 import identify, load_refs
        refs = load_refs()[0]
        xids = json.load(open("atlas_xids.json"))
        seen = {}
        for s in json.load(open("golden_22.json")):
            cid, xkey, _ = identify(s["sig"], refs)
            tid = cid or xids.get(xkey)
            b = np.array(s["beta"], float)
            if tid and (tid not in seen or b.min() > seen[tid].min()):
                seen[tid] = b
        for tid, b in sorted(seen.items()):
            r = cell_defects(b / b.sum())
            if r:
                print(tid, {k: (n, round(m, 4)) for k, (n, m, _) in r.items()}, flush=True)


if __name__ == "__main__":
    main()
