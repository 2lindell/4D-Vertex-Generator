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


def main():
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
