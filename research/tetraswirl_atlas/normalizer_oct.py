"""The octahedral swirl group's normalizer, and the seeds whose polychoron gains symmetry from it (normalizer_oct.json).

A seed's polychoron is the orbit G p. A map k outside G that normalizes it (k G k^-1 = G) carries the orbit to G (k p),
so the polychoron is symmetric under k exactly when k p lies in G p: when p is fixed by g^-1 k for some g, itself an
element of N \\ G. So the seeds of extra symmetry are the fixed sets of the elements of N \\ G: great spheres (a
reflection), great circles (lines in the chart) or points.

N is found as a finite group: it permutes G's orbits of 8 points with stabiliser of order 6 (the two sets of eight
kitrippic points: the 16-cell holding V, W1, W2, W3 and the one holding F), so it lies in the symmetry group of those
16 points; that group is enumerated (each map fixed by where it sends four independent points) and N is the part that
normalizes G.

Each fixed set is clipped to the domain (b >= 0, b3 <= b1, b3 <= b2), and the symmetry a seed there has (the elements
of N that keep its orbit) is counted.

    SWIRL_GROUP=oct python normalizer_oct.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import domain as D

OUT = "normalizer_oct.json"
G = D.ELEMENTS
CELL = D.CELL


def key(M):
    return tuple(np.round(M, 6).ravel())


def orbit(p):
    O = []
    for q in G @ p:
        if not any(np.linalg.norm(q - u) < 1e-8 for u in O):
            O.append(q)
    return np.array(O)


def beta(x):
    b = np.linalg.solve(CELL.T, np.asarray(x, float).T).T
    return b


def in_domain(b, eps=1e-12):
    s = b.sum(-1, keepdims=True)
    b = b / s
    return (s[..., 0] > 0) & (b.min(-1) >= -eps) & (b[..., 3] <= b[..., 1] + eps) & (b[..., 3] <= b[..., 2] + eps)


def sym_order(p, N):
    O = orbit(p)
    n = 0
    for k in N:
        Q = O @ k.T
        if all(np.min(np.linalg.norm(O - q, axis=1)) < 1e-7 for q in Q):
            n += 1
    return n


def main():
    Gk = {key(g) for g in G}
    A = orbit(CELL[0] / np.linalg.norm(CELL[0]))
    F = np.array([0, 1, 1, 1.0]) @ CELL
    B = orbit(F / np.linalg.norm(F))
    S = np.vstack([A, B])
    print(f"|G| = {len(G)}; kitrippic sets of {len(A)} and {len(B)} points", flush=True)
    # (the symmetries of the 16 points: each fixed by the images of four independent points with the same angles)
    idx = None
    for c in itertools.combinations(range(len(S)), 4):
        if abs(np.linalg.det(S[list(c)])) > 1e-6:
            idx = list(c)
            break
    P = S[idx]
    Gram = P @ P.T
    sym = {}
    for t in itertools.permutations(range(len(S)), 4):
        Q = S[list(t)]
        if np.abs(Q @ Q.T - Gram).max() > 1e-9:
            continue
        k = np.linalg.solve(P, Q).T                   # (k P_i = Q_i)
        if np.abs(k @ k.T - np.eye(4)).max() > 1e-9:
            continue
        if all(np.min(np.linalg.norm(S - s, axis=1)) < 1e-9 for s in S @ k.T):
            sym[key(k)] = k
    Sym = list(sym.values())
    N = [k for k in Sym if {key(k @ g @ k.T) for g in G} == Gk]
    rot = sum(1 for k in N if np.linalg.det(k) > 0)
    print(f"symmetries of the 16 points: {len(Sym)}; normalizer N: {len(N)} ({rot} rotations, {len(N) - rot} with "
          f"det -1); index of G in N: {len(N) // len(G)}", flush=True)
    # (the fixed sets of N \ G, clipped to the domain)
    found = {}
    for k in N:
        if key(k) in Gk:
            continue
        w, V = np.linalg.eig(k)
        F1 = np.real(V[:, np.abs(w - 1) < 1e-8])
        if F1.size == 0:
            continue
        Fx = np.linalg.svd(F1.T)[2][:np.linalg.matrix_rank(F1, 1e-8)]      # (orthonormal basis of the fixed space)
        dim = len(Fx)
        proj = key(Fx.T @ Fx)
        if proj in found:
            found[proj]["elements"] += 1
            continue
        found[proj] = {"dim": dim, "basis": Fx, "elements": 1, "det": float(np.linalg.det(k)),
                       "order": next(n for n in range(1, 13) if np.allclose(np.linalg.matrix_power(k, n), np.eye(4), atol=1e-8))}
    loci = []
    for proj, f in found.items():
        Fx = f["basis"]
        if f["dim"] == 2:
            th = np.linspace(0, 2 * np.pi, 7201)
            X = np.cos(th)[:, None] * Fx[0] + np.sin(th)[:, None] * Fx[1]
            ok = in_domain(beta(X))
            if not ok.any():
                continue
            # (the arcs inside the domain, ends refined by bisection)
            runs, cur = [], None
            for i, o in enumerate(ok):
                if o and cur is None:
                    cur = i
                if not o and cur is not None:
                    runs.append((cur, i - 1)); cur = None
            if cur is not None:
                runs.append((cur, len(th) - 1))
            for a, b in runs:
                def edge(i_in, i_out):
                    lo, hi = th[i_in], th[i_out]
                    for _ in range(60):
                        m = (lo + hi) / 2
                        if in_domain(beta(np.cos(m) * Fx[0] + np.sin(m) * Fx[1])[None])[0]:
                            lo = m
                        else:
                            hi = m
                    return lo
                t0 = edge(a, max(a - 1, 0)) if a > 0 else th[a]
                t1 = edge(b, min(b + 1, len(th) - 1)) if b < len(th) - 1 else th[b]
                if t1 - t0 < 1e-9:
                    continue
                ends = [beta(np.cos(t) * Fx[0] + np.sin(t) * Fx[1]) for t in (t0, t1)]
                ends = [e / e.sum() for e in ends]
                mid = np.cos((t0 + t1) / 2) * Fx[0] + np.sin((t0 + t1) / 2) * Fx[1]
                loci.append({"dim": 1, "from": ends[0].tolist(), "to": ends[1].tolist(),
                             "symmetry": sym_order(mid, N), "elements": f["elements"], "kind": f"order-{f['order']} "
                             + ("rotation" if f["det"] > 0 else "rotoreflection")})
        elif f["dim"] == 3:
            # (a great sphere: a plane in the chart; its normal in the betas)
            n4 = np.linalg.svd(Fx)[2][3]                # (the sphere's normal in R4)
            nb = CELL @ n4                              # (as a linear form in the betas: n . (beta CELL))
            # (does it cross the domain? test points of the domain on both sides)
            test = np.random.default_rng(0).dirichlet(np.ones(4), 20000)
            test = test[(test[:, 3] <= test[:, 1]) & (test[:, 3] <= test[:, 2])]
            v = test @ nb
            if (v > 1e-9).any() and (v < -1e-9).any():
                mid = test[np.argmin(np.abs(v))] @ CELL
                loci.append({"dim": 2, "plane_in_betas": (nb / np.abs(nb).max()).round(9).tolist(),
                             "symmetry": sym_order(mid / np.linalg.norm(mid), N), "elements": f["elements"],
                             "kind": "reflection"})
        elif f["dim"] == 1:
            x = Fx[0]
            for s in (1, -1):
                b = beta(s * x)
                if in_domain(b[None])[0]:
                    loci.append({"dim": 0, "at": (b / b.sum()).tolist(), "symmetry": sym_order(s * x, N),
                                 "elements": f["elements"], "kind": f"order-{f['order']}"})
    for l in loci:
        if l["dim"] == 1:
            print(f"line  {np.round(l['from'], 5).tolist()} -> {np.round(l['to'], 5).tolist()}  symmetry {l['symmetry']}  "
                  f"({l['kind']})", flush=True)
        elif l["dim"] == 2:
            print(f"plane {l['plane_in_betas']}  symmetry {l['symmetry']}", flush=True)
        else:
            print(f"point {np.round(l['at'], 5).tolist()}  symmetry {l['symmetry']}", flush=True)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "normalizer_order": len(N), "rotations": rot,
               "loci": loci}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
