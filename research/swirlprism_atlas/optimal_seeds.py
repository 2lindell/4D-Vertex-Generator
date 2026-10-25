"""The optimal seed of every shape: the point of its locus whose polychoron has the most even edges, i.e. the lowest
ratio of longest to shortest hull edge. Written to optimal_seeds.json; the page shows it as the shape's example seed.

A shape is a point, a line, a wall or a region of the half-cell, and its locus is exact:
  - a point is a single seed (nothing to choose);
  - a region is open: the seed moves freely in the half-cell's three dimensions;
  - a wall lies on one exact surface and a line on two (where two of them meet).
The surfaces are taken from a library of exact ones: every plane of the half-cell (its faces, its splitting mirror
and each flat wall's plane) carried by the 4800 elements of the extended group, and every curved wall's polynomial
surface with the images the page copies it by (arrangement.surfaces). For each starting sample, the surfaces passing
through it (to 1e-9) are the candidates, and the defining ones are those the shape stays on: a step along the
candidate surface (a wall), or along the curve two of them share (a line), keeps the sample's classification. So a
surface that only passes through a sample by chance is never taken as a constraint.

The search then runs in the locus's own coordinates (3, 2 or 1 of them): each trial point is moved back onto the
constraint surfaces by Newton steps in the betas, classified, and scored by its edge ratio (infinite if it is not the
shape). Starts: the shape's samples with the lowest ratios, each refined by a Nelder-Mead simplex.

    python optimal_seeds.py              (all shapes; resumes, keeping shapes already in optimal_seeds.json)
    python optimal_seeds.py X99,F2       (just these, recomputed)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import region_walls as rw
from cellframe import seed_from_beta
from classify import ACTION
from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.local_view import hull_edges
from four_d_vertex_generator.off import compute_convex_hull
from probe import _one, label_of_sig

OUT = "optimal_seeds.json"
FOLD = np.eye(4)[[1, 0, 3, 2]]
ON = 1e-9                  # (a sample lies on a surface: |f| / |grad f| below this, betas summing to 1)
STEP = 1e-4                # (the step along a candidate locus that must keep the classification)
STARTS = 2                 # (refined starts per shape)
SPREAD = 20                # (samples scored per shape, spread over its list)


def ratio_of(b):
    """Longest over shortest hull edge of the polychoron from beta b."""
    V = generate_vertices_from_seed(seed_from_beta(b), ACTION, tol=1e-9)
    faces, _ = compute_convex_hull(V, tol=1e-9)
    E = np.array([tuple(e) for e in hull_edges(faces)])
    L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(L.max() / L.min())


def home(b):
    """Normalised and moved into the half-cell (beta3 >= beta4) by the fold, which keeps every shape."""
    b = np.asarray(b, float)
    b = b / b.sum()
    b = np.where(np.abs(b) < 1e-15, 0.0, b)              # (on a face of the half-cell: exactly on it)
    return b @ FOLD.T if b[2] < b[3] else b


# --- constraint surfaces, as functions of the betas (homogeneous) --------------------------------------------------

class Plane:
    def __init__(self, n, name):
        self.n, self.name = np.asarray(n, float), name

    def f(self, B):
        return np.atleast_2d(B) @ self.n

    def grad(self, b):
        return self.n


class Curved:
    def __init__(self, s):
        self.s, self.name = s, s.name

    def f(self, B):
        return self.s.f(np.atleast_2d(B))

    def grad(self, b, e=1e-7):
        b = np.asarray(b, float)
        return np.array([(self.f(b + e * np.eye(4)[k])[0] - self.f(b - e * np.eye(4)[k])[0]) / (2 * e)
                         for k in range(4)])


class Conic:
    """A conic edge of a flat patch (fexact_patches.json), as the quadric cone it lies on: a homogeneous quadratic
    in three of the betas (x², y², z², xy, xz, yz)."""

    def __init__(self, label, var, coeffs, name):
        self.label, self.v, self.c, self.name = label, list(var), np.asarray(coeffs, float), name

    def f(self, B):
        B = np.atleast_2d(B)
        x, y, z = (B[:, k] for k in self.v)
        a = self.c
        return a[0] * x * x + a[1] * y * y + a[2] * z * z + a[3] * x * y + a[4] * x * z + a[5] * y * z

    def grad(self, b, e=1e-7):
        b = np.asarray(b, float)
        return np.array([(self.f(b + e * np.eye(4)[k])[0] - self.f(b - e * np.eye(4)[k])[0]) / (2 * e)
                         for k in range(4)])


def conic_pairs():
    """Each line shape traced as a conic edge of a flat patch: (its patch's plane, the conic), from fexact_patches."""
    out = defaultdict(list)
    d = json.load(open("fexact_patches.json"))
    for p in (d if isinstance(d, list) else d.values()):
        if not isinstance(p, dict):
            continue
        for c in p.get("conics", []):
            out[c["label"]].append([Plane(p["normal"], f"plane of {p['target']} ({p['wall']})"),
                                    Conic(c["label"], c["vars"], c["coeffs"], f"conic {c['label']} in {p['wall']}")])
    return out


CONICS = {}


def library():
    """Every exact plane of the half-cell carried by the extended group, and the curved walls' surfaces."""
    from arrangement import surfaces
    from t2_pieces import extended
    surfs = surfaces()
    base = [np.eye(4)[k] for k in range(4)] + [np.array([0, 0, 1, -1.0])]
    curved = []
    for s in surfs:
        if s.kind == "plane":
            base.append(s.f(np.eye(4)))
        else:
            curved.append(Curved(s))
    G = extended()
    GG = np.concatenate([G, G @ FOLD])
    N = np.concatenate([np.stack([n @ np.linalg.inv(g) for g in GG]) for n in base])
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    N *= np.sign(N[np.arange(len(N)), np.abs(N).argmax(1)])[:, None]
    N = np.unique(np.round(N, 9), axis=0)
    planes = [Plane(n, "plane (" + ", ".join(f"{x:.6g}" for x in n) + ")") for n in N]
    return planes, curved


def through(b, planes, curved):
    """The library surfaces passing through b (one of each zero set, as seen at b)."""
    out = []
    P = np.array([p.n for p in planes])
    for i in np.flatnonzero(np.abs(P @ b) / np.linalg.norm(P, axis=1) < ON):
        out.append(planes[i])
    for c in curved:
        g = c.grad(b)
        if np.linalg.norm(g) > 1e-12 and abs(c.f(b)[0]) / np.linalg.norm(g) < ON:
            out.append(c)
    # (the same zero set twice: the same normal at b, up to sign)
    keep, seen = [], []
    for s in out:
        g = s.grad(b)
        g = g - g.mean()                                  # (in the plane of betas summing to 1)
        g = g / (np.linalg.norm(g) + 1e-30)
        if any(min(np.linalg.norm(g - u), np.linalg.norm(g + u)) < 1e-7 for u in seen):
            continue
        seen.append(g)
        keep.append(s)
    return keep


def onto(b, cons, iters=40):
    """b moved onto every constraint surface and onto the betas' sum 1 (least-norm Newton steps)."""
    b = np.asarray(b, float).copy()
    for _ in range(iters):
        F = np.array([c.f(b)[0] for c in cons] + [b.sum() - 1])
        if np.abs(F).max() < 1e-15:
            break
        J = np.array([c.grad(b) for c in cons] + [np.ones(4)])
        b = b - np.linalg.pinv(J) @ F
    return b


def tangent(b, cons):
    """An orthonormal basis of the directions along the constraints (and the betas' sum)."""
    J = np.array([c.grad(b) for c in cons] + [np.ones(4)])
    _, s, vt = np.linalg.svd(J)
    rank = int((s > 1e-9 * s[0]).sum())
    return vt[rank:]


# --- per shape -----------------------------------------------------------------------------------------------------

TID = None
LIB = None


def label(b):
    b = np.asarray(b, float)
    if (b < -1e-12).any():
        return None
    return _one(b / b.sum())[0]


def stays(b, cons, tid):
    """A step each way along the locus the constraints leave (in every direction of it) keeps the shape."""
    U = tangent(b, cons)
    if not len(U):
        return False
    for u in U:
        # (each way along it, unless that way leaves the half-cell at once: a locus ending on its face)
        labs = [label(onto(b + sgn * STEP * u, cons)) for sgn in (1, -1)]
        if any(x not in (tid, None) for x in labs) or all(x is None for x in labs):
            return False
    return True


def constraints(b, tid, dim):
    """The surfaces through b defining the shape's locus there: none (a region), one (a wall) or two (a line)."""
    if dim == "region":
        return []
    planes, curved = LIB
    cand = through(b, planes, curved)
    k = {"wall": 1, "line": 2}[dim]
    if dim == "wall":
        sets = [[c] for c in cand]
    else:
        sets = [[c1, c2] for i, c1 in enumerate(cand) for c2 in cand[i + 1:]]
    if dim == "line":                                  # (a conic edge traced exactly: its plane and its conic)
        sets = [pc for pc in CONICS.get(tid, []) if all(abs(c.f(b)[0]) / (np.linalg.norm(c.grad(b)) + 1e-30) < 1e-7
                                                         for c in pc)] + sets
    for cons in sets:
        if len(tangent(b, cons)) != 3 - k:
            continue
        if stays(b, cons, tid):
            return cons
    # (a curved wall at a sample where it meets other shapes, so no step keeps it: its own surface, named for it)
    own = [c for c in cand if isinstance(c, Curved) and c.name.split(":")[0] == tid and ":" not in c.name]
    if dim == "wall" and own:
        return own[:1]
    return None


def score(b, tid):
    b = np.asarray(b, float)
    if (b < -1e-12).any() or label(b) != tid:
        return np.inf
    try:
        return ratio_of(b / b.sum())
    except Exception:
        return np.inf


def refine(args):
    tid, dim, b0 = args
    b0 = np.asarray(b0, float)
    b0 = b0 / b0.sum()
    cons = constraints(b0, tid, dim)
    if cons is None:
        return {"ratio": score(b0, tid), "beta": b0.tolist(), "constraints": None, "start": b0.tolist()}
    U = tangent(b0, cons)
    at = lambda y: onto(b0 + y @ U, cons) if cons else (b0 + y @ U) / (b0 + y @ U).sum()
    f = lambda y: score(at(y), tid)
    r0 = f(np.zeros(len(U)))
    h = {"region": 0.02, "wall": 0.01, "line": 0.01}[dim]
    simplex = np.vstack([np.zeros(len(U)), h * np.eye(len(U))])
    res = minimize(f, np.zeros(len(U)), method="Nelder-Mead",
                   options={"initial_simplex": simplex, "xatol": 1e-7, "fatol": 1e-8, "maxiter": 60 * len(U)})
    best = at(res.x) if res.fun < r0 else b0
    return {"ratio": float(min(res.fun, r0)), "beta": home(best).tolist(), "start": home(b0).tolist(),
            "start_ratio": float(r0), "constraints": [c.name for c in cons]}


def samples_by_shape():
    S = json.load(open("golden_22.json"))
    if os.path.exists("extra_samples.json"):
        S += json.load(open("extra_samples.json"))
    by = defaultdict(list)
    for s in S:
        by[label_of_sig(s["sig"])].append(home(s["beta"]))
    return by


DIMS, BY, MESH = {}, {}, {}


def shape(tid):
    """The optimal seed of one shape (run in a worker)."""
    dim, B = DIMS[tid], np.array(BY.get(tid, []))
    if dim == "wall" and tid in MESH:
        # (a curved wall's mesh lies on its exact surface: its vertices, spread over it and put back on the
        # surface, are starts too; its samples can sit too near an edge to classify as the wall)
        V = MESH[tid]
        V = V[np.linspace(0, len(V) - 1, min(len(V), SPREAD)).astype(int)]
        # (the surface the mesh lies on: named for the wall, or another wall's surface it shares)
        res_ = lambda c: max(abs(c.f(v)[0]) / (np.linalg.norm(c.grad(v)) + 1e-30) for v in V)
        own = [c for c in LIB[1] if c.name == tid] or [c for c in LIB[1] if res_(c) < 1e-6][:1]
        extra = [home(onto(v, own)) for v in V] if own else []
        B = np.array(list(B) + extra) if len(B) else np.array(extra)
    if dim == "line" and tid in CONICS and len(B):
        # (a conic line: its samples put onto the conic, and points spread along it, as starts)
        extra = []
        for pc in CONICS[tid]:
            for b in B[:4]:
                x = onto(b, pc)
                if np.isfinite(x).all() and (x > -1e-12).all():
                    U = tangent(x, pc)
                    extra += [home(onto(x + t * U[0], pc)) for t in np.linspace(-0.2, 0.2, 9)] if len(U) else []
        extra = [x for x in extra if np.isfinite(x).all() and (x > -1e-12).all()]
        B = np.array(list(B) + extra)
    if not len(B):
        return tid, None
    if dim == "point":
        b = B[0]
        return tid, {"dim": dim, "beta": b.tolist(), "ratio": score(b, tid), "constraints": "the point itself"}
    # (the samples' own ratios, spread over the locus: the best few are the starts)
    idx = np.unique(np.linspace(0, len(B) - 1, min(len(B), SPREAD)).astype(int))
    r = [score(B[i], tid) for i in idx]
    order = [idx[i] for i in np.argsort(r) if np.isfinite(r[i])]
    if not order:
        return tid, None
    # (the best starts whose locus is found; failing those, the best sample as it is)
    res = []
    for i in order[:3 * STARTS]:
        x = refine((tid, dim, B[i]))
        if x["constraints"] is not None:
            res.append(x)
            if len(res) == STARTS:
                break
    if not res:
        res = [refine((tid, dim, B[order[0]]))]
    return tid, {"dim": dim, **min(res, key=lambda x: x["ratio"])}


def main(only=None):
    global LIB, DIMS, BY, MESH, CONICS
    d = rw.page_data()
    DIMS = {t["id"]: t.get("dim") for t in d["types"]}
    BY = samples_by_shape()
    X = json.load(open("xsurfaces.json"))
    for tid, sf in X.items():
        V = np.unique(np.round(np.array(sf["tris"], float).reshape(-1, 4), 12), axis=0)
        V = V / V.sum(1, keepdims=True)
        V = V[(V > 0.01).all(1)]                          # (away from the half-cell's faces)
        MESH[tid] = V
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [t for t in DIMS if (only is None and t not in done) or (only is not None and t in only)]
    LIB = library()                                       # (the workers share it by forking)
    CONICS.update(conic_pairs())
    print(f"{len(LIB[0])} planes, {len(LIB[1])} curved surfaces in the library; {len(todo)} shapes", flush=True)
    with Pool(int(os.environ.get("PROCS", 4))) as pool:
        for tid, e in pool.imap_unordered(shape, todo):
            if e is None:
                print(f"{tid}: no sample classifies as the shape", flush=True)
                continue
            done[tid] = e
            print(f"{tid} ({e['dim']}): ratio {e.get('start_ratio', e['ratio']):.4f} -> {e['ratio']:.6f} at beta "
                  f"{np.round(e['beta'], 8).tolist()}; locus {e['constraints']}", flush=True)
            json.dump(done, open(OUT, "w"), indent=1)


def robust(b, cons, tid, step=STEP):
    """The shape holds a step each way along its locus from b (or one way, where the other leaves the half-cell)."""
    return label(b) == tid and stays(b, cons, tid) if cons else label(b) == tid and all(
        label(home(b + sgn * step * u)) in (tid, None) for u in np.eye(4)[:3] - 0.25 for sgn in (1, -1))


def settle_one(args):
    """An optimum that sits on the edge of its shape (where it meets another, the ratio still falling) moved back
    toward its start, by bisection along the locus, to the last point where the shape holds all round."""
    tid, e = args
    names = e.get("constraints")
    if not isinstance(names, list) or "start" not in e:
        return tid, e
    planes, curved = LIB
    by_name = {p.name: p for p in planes} | {c.name: c for c in curved} | \
        {c.name: c for pcs in conic_pairs().values() for pc in pcs for c in pc}
    cons = [by_name[n] for n in names if n in by_name]
    b1, b0 = np.array(e["beta"]), np.array(e["start"])
    if robust(b1, cons, tid):
        return tid, {**e, "settled": "the optimum itself"}
    if not robust(b0, cons, tid):
        return tid, {**e, "settled": "not robust (kept)"}
    at = lambda t: onto(b0 + t * (b1 - b0), cons) if cons else b0 + t * (b1 - b0)
    lo, hi = 0.0, 1.0
    for _ in range(14):
        m = (lo + hi) / 2
        if robust(at(m), cons, tid):
            lo = m
        else:
            hi = m
    b = home(at(lo))
    return tid, {**e, "beta": b.tolist(), "ratio": score(b, tid), "optimum_at_edge": e["ratio"],
                 "settled": f"backed off to {lo:.4f} of the way from the start"}


def settle():
    global LIB
    LIB = library()
    done = json.load(open(OUT))
    with Pool(int(os.environ.get("PROCS", 4))) as pool:
        for tid, e in pool.imap_unordered(settle_one, list(done.items())):
            done[tid] = e
            print(f"{tid}: {e.get('settled')}, ratio {e['ratio']:.6f}", flush=True)
            json.dump(done, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--settle"]:
        settle()
    else:
        main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None)
