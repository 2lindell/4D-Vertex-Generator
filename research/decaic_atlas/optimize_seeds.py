"""The most even seed of every decaic, kidecaic and iodecaic shape (optimized_seeds<tag>.json, optimized .off files).

As swirlprism_atlas/exact_seeds.py. A shape's vertices are the orbit of x = beta W5 (exact, integer in beta). Every edge
joins the seed's vertex x to an image g x (g = sign * perm), and its squared length is a quadratic form in the betas,

    q_g(beta) = |x - g x|^2 = beta M M^T beta^T,  M = W5 - sign * W5[:, perm]      (integer M).

While the seed stays in the shape its edges are the same orbits, so the ratio of the longest to the shortest edge is
R(beta) = sqrt(max_g q_g / min_g q_g), a function of beta alone. The most even seed minimises R over the shape:

  * the locus: a point is fixed; a line, a wall: its equations (the arrangement's planes the shape's face lies on, or
    the curved surface a kidecaic / iodecaic wall lies on); a region: free;
  * staying in the shape: every form of the seed's polychoron (forms.region_conditions) that is not zero at the seed
    keeps its sign (these are where the polychoron changes);
  * domain: beta >= 0, b1 >= b4.

Found as: minimise t subject to t >= q_g >= 1 for every edge orbit (the forms are homogeneous: the scale is free), the
locus, and the sign conditions (SLSQP); then polished: the longest orbits within 1e-7 of the longest made equal, the
shortest likewise, the active sign conditions held at 0, and the minimum's own condition (Lagrange) solved by Newton's
method at 40 digits. Checked: the point is still the shape (same polytope: arrangement key or incidences), every
orbit lies between the shortest and longest, R rises on a ring of small steps on the locus. A minimum where a sign
condition is active is on the shape's edge: the shape gets as even as that only in the limit, and its .off uses a seed
a little inside (the last point still the shape on the way from the start).

    DECAIC_SUB=full|ki|io python optimize_seeds.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.optimize import minimize

import forms as F
import group as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
from four_d_vertex_generator.local_view import hull_edges  # noqa: E402
from four_d_vertex_generator.off import to_4off  # noqa: E402

mp.mp.dps = 40
B = sp.symbols("b1 b2 b3 b4")
OFF_DIR = os.path.join(ROOT, "optimized_off", {"full": "decaic", "ki": "kidecaic", "io": "iodecaic", "pen": "pennic", "kipen": "kipennic"}[G.SUB])
W5I = G.W5.astype(int)


# ---------------------------------------------------------------------------------------------------------------- shapes
def plane_text(c):
    return " + ".join(f"({int(a)})*{v}" for a, v in zip(c, ["b1", "b2", "b3", "b4"]) if a)


def decaic_shapes():
    """The 27 decaic shapes: exact seed (the arrangement face's point) and the planes its face lies on."""
    S = json.load(open("decaic_shapes.json"))["shapes"]
    A = json.load(open("arrangement.json"))
    planes = [tuple(b) for b in A["boundary"]] + [tuple(p) for p in A["planes"] if list(p) not in A["boundary"]]
    out = []
    for s in S:
        seed = [float(Fraction(v)) for v in s["seed_beta_exact"]]
        face = next(f for f in A["faces"] if [float(Fraction(v)) for v in f["point"]] == seed)
        zero = [planes[i] for i, v in enumerate(face["signs"]) if v == 0]
        out.append({"name": s["name"], "kind": {3: "region", 2: "wall", 1: "line", 0: "point"}[s["degrees"]],
                    "degrees": s["degrees"], "seed": seed, "locus": independent([plane_text(c) for c in zero], seed),
                    "hull_tol": 1e-9})
    add_pieces(out, [(f["dim"], [float(Fraction(v)) for v in f["point"]],
                      [plane_text(planes[i]) for i, v in enumerate(f["signs"]) if v == 0]) for f in A["faces"]],
               {k: n["name"] for s_, n in zip(json.load(open("shapes.json"))["shapes"], S) for k in s_["keys"]})
    return out


def _face_key(item):
    return G.label(item)[1]


def add_pieces(shapes, faces, key_to_name):
    """A shape can lie on more than one face of the domain with different loci (sabipard: on b2 = 0 and on b3 = 0).
    Every face of the shape's own dimension is classified, and each distinct locus becomes a piece (a seed on it and
    its equations): optimise_best / centre_best try every piece and keep the best."""
    with Pool(4) as pool:
        K = pool.map(_face_key, [f[1] for f in faces], chunksize=4)
    by = {s["name"]: s for s in shapes}
    for s in shapes:
        s["pieces"] = [{"seed": s["seed"], "locus": s["locus"]}]
    seen = {s["name"]: {tuple(sorted(s["locus"]))} for s in shapes}
    for (dim, point, zeros), k in zip(faces, K):
        n = key_to_name.get(k)
        if n is None or n not in by or dim != by[n]["degrees"] or dim in (0, 3):
            continue
        loc = independent(zeros, point)
        sig = tuple(sorted(loc))
        if sig not in seen[n]:
            seen[n].add(sig)
            by[n]["pieces"].append({"seed": point, "locus": loc})


def optimise_best(shape):
    """optimise on every piece of the shape; the best (least R) is the shape's, the others are recorded."""
    pieces = shape.get("pieces") or [{"seed": shape["seed"], "locus": shape["locus"]}]
    res = []
    for i, pc in enumerate(pieces):
        r = optimise({**shape, "seed": pc["seed"], "locus": pc["locus"]})
        r["piece"] = i
        res.append(r)
    best = min(res, key=lambda r: r["R"])
    if len(res) > 1:
        best["pieces"] = [{"piece": r["piece"], "locus": r["locus"], "R": r["R"], "best": r["best"], "status": r["status"]}
                          for r in res]
    return best


EDGE_NAMES = {"ki": [("sabosid", {3: 120, 4: 420, 5: 120, 8: 60}), ("apbosid", {3: 360, 4: 300, 8: 120}),
                     ("sigwebosid", {3: 300, 4: 300, 7: 60}), ("gigwebosid", {3: 240, 4: 360, 8: 60}),
                     ("tetbosid", {3: 300, 4: 120, 5: 60, 6: 60}), ("petbosid", {3: 420, 4: 60, 5: 60, 6: 60})],
              "io": [("sosbid", {3: 120, 4: 300, 5: 240}), ("gosbid", {3: 360, 4: 60, 5: 240, 6: 60}),
                     ("tetosbid", {3: 240, 4: 240}), ("petosbid", {3: 360, 4: 180})]}


def snub_shapes():
    """The kidecaic / iodecaic shapes of their own (not on a mirror): the map's regions and walls, and the gippid line
    (snad under kidecaic, gippid under iodecaic). Shapes on the mirrors are the decaic ones (same orbits)."""
    W = json.load(open(G.fname("wall_shapes")))["shapes"]
    out, k = [], {"region": 0, "wall": 0}
    for t in W:
        e = {int(a): v for a, v in t["edges_by_valence"].items()}
        kind = "region" if t["kind"] == "region" else "wall"
        name = next((n for n, ed in EDGE_NAMES[G.SUB] if ed == e), None)
        if name is None and e == {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60}:
            name = "mebosid" if t["map_type"] == 0 else "gabosid"            # (map type 0: nearer gabred)
        if name is None:
            k[kind] += 1
            name = f"new_{kind}_{k[kind]:02d}_{t['vertices']}v_{t['cells']}c"
        if t["kind"] == "wall (fold)":
            locus = ["b1 - b4"]
        elif kind == "wall":
            locus = [max(t["surfaces"], key=t["surfaces"].get)]
        else:
            locus = []
        out.append({"name": name, "kind": kind, "degrees": 3 if kind == "region" else 2, "seed": t["seed_beta"],
                    "locus": locus, "hull_tol": 1e-9 if kind == "region" else 1e-7})
    # (regions the dense edge survey added, told apart by their arrangement keys)
    seen = {G.label(np.asarray(o["seed"]) / np.sum(o["seed"]))[1] for o in out if o["kind"] == "region"}
    try:
        zones = json.load(open(G.fname("edge_survey")))["zones"]
    except FileNotFoundError:
        zones = {}
    for z in zones.values():
        for t in z["types"]:
            if t["known_region"] is not None:
                continue
            kk = G.label(t["example"])[1]
            if kk in seen:
                continue
            seen.add(kk)
            sg = t["signature"]
            nc = sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))
            k["region"] += 1
            out.append({"name": f"new_region_{k['region']:02d}_{sg.split(':')[0]}v_{nc}c", "kind": "region", "degrees": 3,
                        "seed": t["example"], "locus": [], "hull_tol": 1e-9})
    # (wall shapes the closure check found)
    try:
        nw = json.load(open(G.fname("snub_closure")))["new_wall_shapes"]
    except FileNotFoundError:
        nw = {}
    for w in nw.values():
        k["wall"] += 1
        nc = sum(int(a) for a in re.findall(r"(\d+)x\[", w["signature"].split("|")[0]))
        out.append({"name": f"new_wall_{k['wall']:02d}_120v_{nc}c",
                    "kind": "wall", "degrees": 2, "seed": w["seed"], "locus": [w["surface"]], "hull_tol": 1e-7})
    g = np.array([1, 11, 11, 1]) / 24
    out.append({"name": "snad" if G.SUB == "ki" else "gippid", "kind": "line", "degrees": 1, "seed": g.tolist(),
                "locus": ["b1 - b4", "b2 - b3"], "hull_tol": 1e-9})
    return out


def independent(texts, seed):
    """The locus equations, reduced to independent ones (by their gradients at the seed)."""
    keep, grads = [], []
    for t in texts:
        e = sp.sympify(t)
        g = np.array([float(sp.diff(e, v).subs(dict(zip(B, seed)))) for v in B])
        if np.linalg.matrix_rank(np.array(grads + [g]), 1e-9) > len(grads):
            keep.append(t)
            grads.append(g)
    return keep


# ------------------------------------------------------------------------------------------------------------ the forms
def edge_forms(seed, hull_tol):
    """The edge orbits at the seed's vertex x (vertex 0): each as its integer matrix Q (q = beta Q beta^T)."""
    V5, V, faces, cells = G.polytope(seed, hull_tol)
    x = G.x5(seed)
    assert np.abs(V5[0] - x).max() < 1e-12
    labels = F.vertex_labels(seed)
    Qs = {}
    for e in hull_edges(faces):
        e = tuple(e)
        if 0 not in e:
            continue
        w = e[1] if e[0] == 0 else e[0]
        s, p = labels[w]
        M = W5I - s * W5I[:, list(p)]
        Q = M @ M.T
        Qs[tuple(Q.ravel())] = Q
    return list(Qs.values())


def sign_conditions(seed):
    """The seed's forms that are not zero there, with their signs: (coefficients, sign)."""
    out = []
    for co, s in F.region_conditions(seed).items():
        v = F.evaluate(co, seed)[0]
        scale = np.abs(np.asarray(co, float)).sum()
        if abs(v) > 1e-9 * scale:
            out.append((np.asarray(co, float) / scale, int(np.sign(v)), co))
    return out


def same_shape(b, seed, key0, hull_tol):
    try:
        sg, k = G.label(b, hull_tol)
    except Exception:
        return False
    return k == key0 or (k is not None and G.same_polytope(b, seed))


# ------------------------------------------------------------------------------------------------------------- optimise
def optimise(shape):
    t0 = time.time()
    seed = np.asarray(shape["seed"], float)
    seed = seed / seed.sum()
    tol = shape["hull_tol"]
    key0 = G.label(seed, tol)[1]
    res = {"name": shape["name"], "kind": shape["kind"], "degrees": shape["degrees"], "locus": shape["locus"],
           "start": seed.tolist()}
    Qs = edge_forms(seed, tol)
    q = lambda b: np.array([b @ Q @ b for Q in Qs])
    R0 = float(np.sqrt(q(seed).max() / q(seed).min()))
    res["start_R"] = R0
    if shape["degrees"] == 0:
        res.update({"status": "point (no freedom)", "best": seed.tolist(), "R": R0, "off_seed": seed.tolist()})
        return res
    zero = [i for i in range(4) if seed[i] < 1e-12]                 # (on a mirror: stays there)
    free = [i for i in range(4) if i not in zero]
    # (locus equations not already met by the mirrors' zeros)
    loc = [sp.sympify(t) for t in shape["locus"]]
    loc = [e for e in loc if sp.expand(e.subs({B[i]: 0 for i in zero})) != 0]
    loc_f = [sp.lambdify(B, e, "numpy") for e in loc]
    loc_g = [[sp.lambdify(B, sp.diff(e, v), "numpy") for v in B] for e in loc]
    conds = sign_conditions(seed)

    def full(z):
        b = np.zeros(4)
        b[free] = z[:len(free)]
        return b
    # (SLSQP on z = (free betas, t): min t; t - q >= 0; q - 1 >= 0; s * form >= 0; locus = 0; domain)
    s0 = seed / np.sqrt(q(seed).min())
    z0 = np.concatenate([s0[free], [q(s0).max()]])
    nf = len(free)
    cons = [{"type": "ineq", "fun": lambda z: z[nf] - q(full(z))},
            {"type": "ineq", "fun": lambda z: q(full(z)) - 1.0},
            {"type": "ineq", "fun": lambda z: np.concatenate([z[:nf], [full(z)[0] - full(z)[3]] if 0 in free else []])}]
    condf = None
    if conds:
        C = np.array([c[0] for c in conds])
        Sg = np.array([c[1] for c in conds], float)
        Mx = np.array(F.MONS)

        def condf(z):
            b = full(z)
            return Sg * (np.prod(b[None, :] ** Mx, axis=1) @ C.T) / (b.sum() ** 4)
        cons.append({"type": "ineq", "fun": condf})
    deg = [sp.Poly(e, *B).total_degree() for e in loc]
    if loc:
        cons.append({"type": "eq", "fun": lambda z: np.array([f(*full(z)) / full(z).sum() ** d for f, d in zip(loc_f, deg)])})
    def feasible(z):
        ok = all((np.atleast_1d(c["fun"](z)) >= -1e-7).all() if c["type"] == "ineq" else
                 (np.abs(np.atleast_1d(c["fun"](z))) <= 1e-7).all() for c in cons)
        if not ok:
            return False
        bb = full(z)
        return bb.sum() > 0 and G.in_domain((bb / bb.sum())[None], 1e-12)[0]
    # (several starts: the seed, and small steps from it along the locus inside the shape; the best feasible result
    #  that is no worse than the seed; else the seed itself)
    rng = np.random.default_rng(3)
    starts = [seed]
    for _ in range(6):
        d = rng.normal(size=4)
        d[zero] = 0
        p = project(seed + 0.02 * d / np.linalg.norm(d) * seed[free].min(), loc_f, loc_g, zero)
        if G.in_domain(p[None])[0] and same_shape(p, seed, key0, tol):
            starts.append(p)
    best, msg = (z0, "start"), "start"
    bestR = R0
    for p in starts:
        p = p / np.sqrt(q(p).min())
        zp = np.concatenate([p[free], [q(p).max()]])
        r = minimize(lambda z: z[nf], zp, method="SLSQP", constraints=cons, options={"ftol": 1e-15, "maxiter": 2000})
        bb = full(r.x)
        if bb.sum() <= 0 or not feasible(r.x):
            continue
        bb = bb / bb.sum()
        Rb = float(np.sqrt(q(bb).max() / q(bb).min()))
        if Rb < bestR - 1e-15 and (same_shape(bb, seed, key0, tol) or
                                   (conds and min(condf(np.concatenate([bb[free], [0]]))) < 1e-7)):
            best, bestR, msg = (r.x, "slsqp"), Rb, r.message
    b = full(best[0])
    b = b / b.sum()
    r = type("R", (), {"message": msg})
    qs = q(b)
    R = float(np.sqrt(qs.max() / qs.min()))
    active_conds = [i for i, v in enumerate(condf(np.concatenate([b[free], [0]]))) if v < 1e-9] if conds else []
    res["numeric"] = {"best": b.tolist(), "R": R, "slsqp": r.message, "active_sign_conditions": len(active_conds)}
    # (polish at 40 digits: equal longest, equal shortest, active conditions, locus, Lagrange)
    pb, polished = polish(b, Qs, loc, zero, [conds[i] for i in active_conds])
    if polished is not None:
        b2 = np.array([float(v) for v in pb])
        if np.abs(b2 - b).max() < 1e-6:
            b = b2
            res["exact_digits"] = 40
            res["best_40_digits"] = [mp.nstr(v, 40) for v in pb]
            res["equations"] = polished
    qs = q(b)
    R = float(np.sqrt(qs.max() / qs.min()))
    on_edge = bool(active_conds)
    inside = same_shape(b, seed, key0, tol)
    off = b
    if not inside or on_edge:
        # (the last point still the shape on the way from the start: a seed just inside)
        lo, hi = 0.0, 1.0
        for _ in range(50):
            m = (lo + hi) / 2
            p = (1 - m) * seed + m * b
            p = project(p, loc_f, loc_g, zero)
            if same_shape(p, seed, key0, tol):
                lo = m
            else:
                hi = m
        off = project((1 - lo * 0.999) * seed + lo * 0.999 * b, loc_f, loc_g, zero)
    ring = ring_check(b, q, loc_f, loc_g, zero, R) if inside and not on_edge else None
    if R >= R0 - 1e-12:
        inside, on_edge, ring = True, False, None
        status = "no progress: the optimiser found no feasible step from the seed (seed kept)"
    else:
        status = None
    res.update({"best": b.tolist(), "R": R, "orbits": len(Qs),
                "status": status or (("minimum inside the shape" + (" (not unique: R is flat along the locus there)" if ring and ring["flat_directions"] else ""))
                           if inside and not on_edge else
                           "infimum on the shape's edge (a sign condition is active): not attained"),
                "ring_check_rises": ring, "off_seed": np.asarray(off).tolist(), "off_R": float(np.sqrt(q(off).max() / q(off).min())),
                "seconds": round(time.time() - t0, 1)})
    return res


def project(p, loc_f, loc_g, zero):
    p = np.array(p, float)
    p[zero] = 0
    for _ in range(50):
        if not loc_f:
            break
        f = np.array([fi(*p) for fi in loc_f])
        J = np.array([[gi(*p) for gi in g] for g in loc_g])
        J[:, zero] = 0
        step = np.linalg.lstsq(J, f, rcond=None)[0]
        p = p - step
        if np.abs(step).max() < 1e-16:
            break
    return p / p.sum()


def ring_check(b, q, loc_f, loc_g, zero, R):
    """Does R rise at every point of a ring of small steps round b, on the locus?"""
    rng = np.random.default_rng(0)
    worse = flat = 0
    for _ in range(40):
        d = rng.normal(size=4)
        d[zero] = 0
        d -= d.mean()
        p = project(b + 1e-5 * d / np.linalg.norm(d), loc_f, loc_g, zero)
        if not G.in_domain(p[None])[0]:
            continue
        qs = q(p)
        Rp = np.sqrt(qs.max() / qs.min())
        if Rp < R - 1e-13:
            worse += 1
        elif Rp < R + 1e-11:
            flat += 1
    return {"rises": worse == 0, "flat_directions": flat > 0}


def polish(b, Qs, loc, zero, active):
    """Newton at 40 digits on: the active orbits equal (longest, shortest), the locus, the mirrors, the active sign
    conditions, beta summing to 1, and (with freedom left) the Lagrange condition."""
    qs = np.array([b @ Q @ b for Q in Qs])
    L = [i for i in range(len(Qs)) if qs[i] > qs.max() * (1 - 1e-7)]
    S = [i for i in range(len(Qs)) if qs[i] < qs.min() * (1 + 1e-7)]
    bv = sp.Matrix(B)
    qe = [sp.expand((bv.T * sp.Matrix(Qs[i]) * bv)[0]) for i in range(len(Qs))]
    cons = [qe[i] - qe[L[0]] for i in L[1:]] + [qe[i] - qe[S[0]] for i in S[1:]] + list(loc) + [B[i] for i in zero]
    cons += [F.poly(c[2]) for c in active]
    # (independent constraints only)
    keep, grads = [], []
    for c in cons:
        g = np.array([float(sp.diff(c, v).subs(dict(zip(B, b)))) for v in B] + [1.0]) if c.free_symbols else None
        if g is None:
            continue
        gg = g[:4] - g[:4].mean()
        if np.linalg.matrix_rank(np.array(grads + [gg]), 1e-7) > len(grads):
            keep.append(c)
            grads.append(gg)
    n = len(keep)
    if n > 3:
        return None, None
    lam = sp.symbols(f"l0:{n}") if n else ()
    mu = sp.Symbol("mu")
    phi = sp.log(qe[L[0]]) - sp.log(qe[S[0]])
    if n == 3:
        eqs = keep + [sum(B) - 1]
        unknowns = list(B)
        start = list(b)
    else:
        eqs = [sp.diff(phi, v) - sum(l * sp.diff(c, v) for l, c in zip(lam, keep)) - mu for v in B] + keep + [sum(B) - 1]
        unknowns = list(B) + list(lam) + [mu]
        # (multipliers from least squares at b)
        gphi = np.array([float(sp.diff(phi, v).subs(dict(zip(B, b)))) for v in B])
        A = np.array([[float(sp.diff(c, v).subs(dict(zip(B, b)))) for c in keep] + [1.0] for v in B])
        sol = np.linalg.lstsq(A, gphi, rcond=None)[0]
        start = list(b) + list(sol)
    fn = sp.lambdify(unknowns, eqs, "mpmath")
    try:
        x = mp.findroot(lambda *u: fn(*u), [mp.mpf(v) for v in start], tol=mp.mpf(10) ** -35, maxsteps=100)
    except Exception:
        return None, None
    x = [x[i] for i in range(len(unknowns))] if hasattr(x, "rows") else [x]
    return x[:4], [str(e) for e in keep]


# --------------------------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(OFF_DIR, exist_ok=True)
    shapes = decaic_shapes() if G.SUB == "full" else snub_shapes()
    with Pool(4) as pool:
        R = pool.map(optimise_best, shapes, chunksize=1)
    for s, r in zip(shapes, R):
        b = np.asarray(r["off_seed"], float)
        _, V, faces, cells = G.polytope(b, s["hull_tol"])
        name = f"{r['name']}_{len(V)}v_{len(cells)}c_opt"
        open(os.path.join(OFF_DIR, name + ".off"), "w").write(to_4off(V, faces, cells))
        r["off"] = os.path.relpath(os.path.join(OFF_DIR, name + ".off"), ROOT)
        r["vertices"], r["cells"] = len(V), len(cells)
        r["seed_vector"] = (G.x5(b) / np.abs(G.x5(b)).max()).tolist()
        print(f"{r['name']:34} {r['kind']:6} R {r['start_R']:.6f} -> {r['R']:.9f}  {r['status'][:40]:40} "
              f"exact {r.get('exact_digits', '-')}  beta {np.round(r['best'], 9).tolist()}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "shapes": R}, open(G.fname("optimized_seeds"), "w"), indent=1)


if __name__ == "__main__":
    main()
