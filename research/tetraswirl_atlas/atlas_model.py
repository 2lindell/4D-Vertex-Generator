"""The octahedral swirl atlas as an exact model: a seed's shape from the pieces' sign conditions (region_forms_oct.json),
no classifier needed; walls, lines and points found from it, and checked by the classifier where it matters.

    shape_at(B)            the region (piece) each beta lies in, or None (no piece: unexplored, or on a wall)
    on_surface(f, n)       points of the domain on the surface f = 0 (Newton from random seeds)
    SWIRL_GROUP=oct python atlas_model.py      coverage, each wall's extent, and the lines and points where walls meet
                                               (writes lines_points_oct.json)
"""
from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import sympy as sp

import domain as D
from region_forms import B_SYM, MONS, evaluate
from walls_exact import domain_betas

MODEL = json.load(open(f"region_forms_{D.KEY}.json"))
PIECES = list(MODEL["pieces"])
TYPE = {p: v["type"] for p, v in MODEL["pieces"].items()}
_C = [np.array(MODEL["pieces"][p]["conditions"]) for p in PIECES]
_MON = np.array(MONS)


def _monos(B):
    return np.prod(np.asarray(B, float)[:, None, :] ** _MON[None], axis=2)      # (n, monomials)


def piece_at(B):
    """Index into PIECES of the piece each beta lies in (-1: none)."""
    Mo = _monos(B)
    out = np.full(len(Mo), -1)
    for i, C in enumerate(_C):
        ok = (Mo @ C.T > 0).all(1)
        out[(out < 0) & ok] = i
    return out


def shape_at(B):
    return [TYPE[PIECES[i]] if i >= 0 else None for i in piece_at(B)]


def in_domain(B, eps=0.0):
    B = np.asarray(B)
    return (B.min(1) >= -eps) & (B[:, 3] <= B[:, 1] + eps) & (B[:, 3] <= B[:, 2] + eps)


class Surf:
    def __init__(self, text):
        self.text = text
        e = sp.sympify(text)
        self.f = sp.lambdify(B_SYM, e, "numpy")
        self.g = sp.lambdify(B_SYM, [sp.diff(e, v) for v in B_SYM], "numpy")

    def val(self, B):
        B = np.atleast_2d(B)
        return np.asarray(self.f(*B.T), float) * np.ones(len(B))

    def grad(self, B):
        B = np.atleast_2d(B)
        return np.stack([np.asarray(x, float) * np.ones(len(B)) for x in self.g(*B.T)], 1)


def project(B, surfs, iters=40):
    """Newton steps (least norm, within beta summing to 1) onto the common zero set of the surfaces."""
    B = np.array(B, float)
    for _ in range(iters):
        F = np.stack([s.val(B) for s in surfs], 1)                       # (n, k)
        J = np.stack([s.grad(B) for s in surfs], 1)                      # (n, k, 4)
        bad = ~(np.isfinite(F).all(1) & np.isfinite(J).all((1, 2)) & np.isfinite(B).all(1))
        F[bad], J[bad] = 0.0, 0.0                                        # (a point gone astray stops; dropped below)
        B[bad] = np.nan_to_num(B[bad])
        J = J - J.mean(2, keepdims=True)                                 # (stay on sum = 1)
        step = np.einsum("nki,nk->ni", np.linalg.pinv(J).transpose(0, 2, 1) if False else
                         np.transpose(np.linalg.pinv(J), (0, 2, 1)), F)
        B = B - step
        if np.abs(F).max() < 1e-13:
            break
    with np.errstate(invalid="ignore"):
        ok = np.stack([np.abs(s.val(B)) < 1e-10 for s in surfs], 1).all(1) & np.isfinite(B).all(1)
    return B, ok


def sides(B, normals, d=2e-4):
    """Pieces at small steps around each point: along +-normals (one per surface), every sign combination."""
    out = []
    for signs in itertools.product((1, -1), repeat=normals.shape[1]):
        X = B + d * np.einsum("k,nki->ni", np.array(signs, float), normals)
        out.append(piece_at(X))
    return np.stack(out, 1)


def wall_here(B, s, d=2e-4, tol=1e-9):
    """Pieces either side of points B on surface s, and whether s is the wall between them there: a condition of
    the near piece that fails on the far side vanishes at B itself (not a surface merely passing close to a wall)."""
    N = unit_normals(B, [s])[:, 0]
    near, far = piece_at(B - d * N), piece_at(B + d * N)
    Mo, Mf = _monos(B), _monos(B + d * N)
    ok = np.zeros(len(B), bool)
    for i in range(len(B)):
        p, q = near[i], far[i]
        if p < 0 or q < 0 or TYPE[PIECES[p]] == TYPE[PIECES[q]]:
            continue
        C = _C[p]
        fail = (Mf[i] @ C.T) <= 0
        ok[i] = bool((np.abs(Mo[i] @ C[fail].T) < tol * np.abs(C[fail]).sum(1)).any())
    return near, far, ok


def unit_normals(B, surfs):
    N = np.stack([s.grad(B) for s in surfs], 1)
    N = N - N.mean(2, keepdims=True)
    return N / np.linalg.norm(N, axis=2, keepdims=True)


def label(beta):
    b = np.asarray(beta, float)
    try:
        return D.signature(D.classify(D.seed_from_chart(D.chart_from_seed(b @ D.CELL))))
    except Exception:
        return "ERR"


def main():
    rng = np.random.default_rng(7)
    ver = json.load(open(f"walls_verified_{D.KEY}.json"))
    walls = [w["surface"] for w in ver["walls"] if w["verdict"] == "wall"]
    S = {f: Surf(f) for f in walls}
    # coverage: how much of the domain the pieces account for
    Bd = domain_betas(200000, seed=11)
    pc = piece_at(Bd)
    print(f"coverage: {np.mean(pc >= 0):.4f} of the domain lies in a known piece "
          f"({np.sum(pc < 0)} of {len(Bd)} random points in none)", flush=True)
    # each wall's extent in the model: points on it whose two sides are pieces of different shapes
    ext = {}
    for f, s in S.items():
        B, ok = project(Bd[rng.permutation(len(Bd))[:6000]], [s])
        B = B[ok & in_domain(B)]
        near, far, sep = wall_here(B, s)
        ext[f] = B[sep]
        prs = Counter("|".join(sorted([TYPE[PIECES[a]], TYPE[PIECES[b]]])) for a, b, k in zip(near, far, sep) if k)
        print(f"  {f[:58]:58s} {sep.mean():6.3f} of its points separate: {dict(prs.most_common(4))}", flush=True)
    # lines: where two walls both separate shapes at once (three or four shapes round the point)
    lines = []
    fl = list(S)
    for n_, (a, b) in enumerate(itertools.combinations(range(len(fl)), 2)):
        if n_ % 50 == 0:
            print(f"  pair {n_} of {len(fl) * (len(fl) - 1) // 2}; {len(lines)} lines so far", flush=True)
        if not len(ext[fl[a]]) or not len(ext[fl[b]]):
            continue
        seed = np.vstack([ext[fl[a]][:300], ext[fl[b]][:300]])
        B, ok = project(seed, [S[fl[a]], S[fl[b]]])
        B = B[ok & in_domain(B, 1e-12)]
        if not len(B):
            continue
        B = np.unique(B.round(9), axis=0)
        sd = sides(B, unit_normals(B, [S[fl[a]], S[fl[b]]]))
        kinds = [set(TYPE[PIECES[i]] for i in row if i >= 0) for row in sd]
        # (each surface is the wall just off the line, on one side of the other surface at least)
        Ng = unit_normals(B, [S[fl[b]]])[:, 0]
        Nf = unit_normals(B, [S[fl[a]]])[:, 0]
        wa = wall_here(project(B + 3e-4 * Ng, [S[fl[a]]])[0], S[fl[a]])[2] | wall_here(project(B - 3e-4 * Ng, [S[fl[a]]])[0], S[fl[a]])[2]
        wb = wall_here(project(B + 3e-4 * Nf, [S[fl[b]]])[0], S[fl[b]])[2] | wall_here(project(B - 3e-4 * Nf, [S[fl[b]]])[0], S[fl[b]])[2]
        real = wa & wb & np.array([len(k) >= 2 for k in kinds])
        if real.sum() < 2:
            continue
        around = Counter(" ".join(sorted(k)) for k, r in zip(kinds, real) if r)
        lines.append({"walls": [fl[a], fl[b]], "points": B[real].round(8).tolist(), "around": dict(around)})
    print(f"{len(lines)} lines where two walls meet", flush=True)
    json.dump({"coverage": float(np.mean(pc >= 0)), "lines": lines}, open(f"lines_{D.KEY}.json", "w"))
    # points: where three walls meet on lines
    pts, seen_pts = [], set()
    print(f"  finding points among {len(lines)} lines", flush=True)
    for la, lb in itertools.combinations(range(len(lines)), 2):
        ws = sorted(set(lines[la]["walls"]) | set(lines[lb]["walls"]))
        if len(ws) != 3:
            continue
        seed = np.vstack([lines[la]["points"][:100], lines[lb]["points"][:100]])
        B, ok = project(seed, [S[w] for w in ws])
        B = B[ok & in_domain(B, 1e-12)]
        for p in np.unique(B.round(7), axis=0):
            key = tuple(p.round(6))
            if key not in seen_pts:
                seen_pts.add(key)
                pts.append({"beta": p.tolist(), "walls": ws})
    print(f"{len(pts)} points where three walls meet", flush=True)
    # the classifier on each line and point: what lies there
    jobs = [np.mean(l["points"], 0) if False else np.array(l["points"][len(l["points"]) // 2]) for l in lines] + [np.array(p["beta"]) for p in pts]
    with Pool(4) as pool:
        labs = pool.map(label, jobs)
    ids = json.load(open(f"ids_{D.KEY}.json"))
    for l, lab in zip(lines + pts, labs):
        l["on"] = ids.get(lab, lab)
    for l in lines:
        print(f"  line  {l['walls'][0][:28]:28s} x {l['walls'][1][:28]:28s} {len(l['points']):4d} pts; around {list(l['around'])[:2]}; on it {l['on'][:60]}")
    for p in pts:
        print(f"  point beta {np.round(p['beta'], 5)}; walls {[w[:20] for w in p['walls']]}; {p['on'][:60]}")
    json.dump({"coverage": float(np.mean(pc >= 0)), "walls": {f: {"extent_points": int(len(ext[f]))} for f in S},
               "lines": lines, "points": pts}, open(f"lines_points_{D.KEY}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
