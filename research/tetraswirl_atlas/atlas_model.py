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
        J = J - J.mean(2, keepdims=True)                                 # (stay on sum = 1)
        step = np.einsum("nki,nk->ni", np.linalg.pinv(J).transpose(0, 2, 1) if False else
                         np.transpose(np.linalg.pinv(J), (0, 2, 1)), F)
        B = B - step
        if np.abs(F).max() < 1e-13:
            break
    ok = np.stack([np.abs(s.val(B)) < 1e-10 for s in surfs], 1).all(1)
    return B, ok


def sides(B, normals, d=2e-4):
    """Pieces at small steps around each point: along +-normals (one per surface), every sign combination."""
    out = []
    for signs in itertools.product((1, -1), repeat=normals.shape[1]):
        X = B + d * np.einsum("k,nki->ni", np.array(signs, float), normals)
        out.append(piece_at(X))
    return np.stack(out, 1)


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
        sd = sides(B, unit_normals(B, [s]))
        sep = np.array([a >= 0 and b >= 0 and TYPE[PIECES[a]] != TYPE[PIECES[b]] for a, b in sd])
        ext[f] = B[sep]
        prs = Counter("|".join(sorted([TYPE[PIECES[a]], TYPE[PIECES[b]]])) for (a, b), k in zip(sd, sep) if k)
        print(f"  {f[:58]:58s} {sep.mean():6.3f} of its points separate: {dict(prs.most_common(4))}", flush=True)
    # lines: where two walls both separate shapes at once (three or four shapes round the point)
    lines = []
    fl = list(S)
    for a, b in itertools.combinations(range(len(fl)), 2):
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
        real = np.array([len(k) >= 3 or (len(k) == 2 and (row >= 0).all()) for k, row in zip(kinds, sd)])
        if real.sum() < 2:
            continue
        around = Counter(" ".join(sorted(k)) for k, r in zip(kinds, real) if r)
        lines.append({"walls": [fl[a], fl[b]], "points": B[real].round(8).tolist(), "around": dict(around)})
    print(f"{len(lines)} lines where two walls meet", flush=True)
    # points: where three walls meet on lines
    pts = []
    for la, lb in itertools.combinations(range(len(lines)), 2):
        ws = sorted(set(lines[la]["walls"]) | set(lines[lb]["walls"]))
        if len(ws) != 3:
            continue
        seed = np.vstack([lines[la]["points"][:100], lines[lb]["points"][:100]])
        B, ok = project(seed, [S[w] for w in ws])
        B = B[ok & in_domain(B, 1e-12)]
        for p in np.unique(B.round(8), axis=0):
            if not any(np.allclose(p, q["beta"], atol=1e-7) for q in pts):
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
