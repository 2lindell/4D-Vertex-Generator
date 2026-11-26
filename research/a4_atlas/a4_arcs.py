"""The exact curves of a4_curves cut into arcs between their exact points, and traced (a4_arcs_<group>.pkl).

Every curve's known points (its crossings with the other surfaces and the mirrors, its singular points and its critical
points for the generic linear function L) cut it into arcs, and along each arc L is monotone (a critical point of L on
it would be known). So a level L = c strictly between two consecutive values of L at a curve's known points crosses
every arc spanning it: the curve's points on such levels (solved exactly, 30 digits, Singular) give at least one smooth
point inside every arc. From each, the curve is followed both ways (each step put back onto the curve's own equations by
Newton's method; the step never more than a third of the way to the nearest known point of the curve, so none is
passed) until it arrives at a known point (within 1e-10, where it ends on the exact point). An arc found from two
levels is kept once.

The result is the map's segments, in the form a4_faces / a4_merge / a4_walls read: points (each exact point once,
merged within 1e-12), segments with their ends, their traced points and every surface they lie on (the curve's).

    python a4_arcs.py kidecaic
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sympy as sp
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_curves as AC  # noqa: E402

B = sp.symbols("b1 b2 b3 b4")
LV = np.array([1.0, 2.0, 5.0, 0.0])               # (a4_curves.L)


def levels_of(cur):
    """Rational levels strictly between consecutive values of L at the curve's known points (and one each side)."""
    vals = sorted({round(float(np.dot(LV, np.array([float(v) for v in p["beta"]]))), 13) for p in cur["points"]})
    lv = [(a + b) / 2 for a, b in zip(vals, vals[1:]) if b - a > 1e-12]
    return [sp.Rational(round(v * 10 ** 12), 10 ** 12) for v in lv]


def level_points(cur, levels, digits=30):
    if not levels:
        return []
    body = ['LIB "solve.lib";', "ring r=0,(b1,b2,b3,b4),dp; option(redSB);",
            f"ideal G = std(ideal({cur['ideal']}));", "ideal J; int i; int k; def R;"]
    for c in levels:
        body.append(
            f'J = std(G + ideal(b1+2*b2+5*b3-({c})*(b1+b2+b3+b4)) + ideal(b1+b2+b3+b4-1)); '
            f'if(dim(J)==0 && vdim(J)>0){{ R = solve(J, {digits}, 0, "nodisplay"); setring R; '
            f'for(i=1;i<=size(SOL);i++){{ string t="P"; for(k=1;k<=4;k++){{t=t+" "+string(SOL[i][k]);}} '
            f'print(t); }} kill R; setring r; def R; }}')
    out = AC.sing("\n".join(body))
    pts = []
    for line in out.splitlines():
        if line.startswith("P "):
            z = [AC.parse_c(v) for v in line.split()[1:5]]
            if max(abs(v.imag) for v in z) > 1e-20:
                continue
            x = np.array([v.real for v in z])
            if x.min() < -1e-20:
                continue
            pts.append(x)
    return pts


class Curve:
    def __init__(self, ideal):
        gens = [sp.sympify(g.replace("^", "**")) for g in ideal.split(",")]
        self.f = sp.lambdify([B], gens, "numpy")
        self.J = sp.lambdify([B], [[sp.diff(g, b) for b in B] for g in gens], "numpy")

    def F(self, x):
        return np.array(self.f(x), float)

    def Jac(self, x):
        return np.array(self.J(x), float)

    def tangent(self, x):
        M = np.vstack([self.Jac(x), np.ones(4)])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-300)
        t = np.linalg.svd(M)[2][-1]
        return t / np.linalg.norm(t)

    def correct(self, y, t, x0):
        """Newton onto the curve, in the plane through y across t."""
        for _ in range(30):
            r = np.concatenate([self.F(y), [y.sum() - 1, t @ (y - x0)]])
            M = np.vstack([self.Jac(y), np.ones(4), t])
            st = np.linalg.lstsq(M, r, rcond=None)[0]
            y = y - st
            if np.abs(st).max() < 1e-15:
                break
        ok = np.abs(self.F(y)).max() < 1e-12 * max(1.0, np.abs(self.Jac(y)).max()) and abs(y.sum() - 1) < 1e-13
        return y, ok


def follow(C, x, d, known, max_steps=200000):
    """From x along direction d to the known point it arrives at: (index, points) or (None, points, why)."""
    pts = [x.copy()]
    t = C.tangent(x)
    if t @ d < 0:
        t = -t
    # (arrival: within 1e-6 of a known point and a hundredth of its distance to the curve's next known point, so no
    #  other can be meant; near a point where many curves meet, Newton's method stalls a little short of it)
    if len(known) > 1:
        dd, _ = cKDTree(known).query(known, k=2)
        reach = np.minimum(1e-6, dd[:, 1] / 100)
    else:
        reach = np.full(len(known), 1e-6)
    for _ in range(max_steps):
        dist = np.linalg.norm(known - x, axis=1)
        j = int(np.argmin(dist))
        if dist[j] < max(1e-10, reach[j]) and len(pts) > 1:
            pts[-1] = known[j]
            return j, np.array(pts)
        # (ahead of us and close: aim straight at it once within reach)
        h = min(2e-3, dist[j] / 3) if dist[j] > 1e-10 else 2e-3
        h = max(h, 1e-13)
        for _try in range(30):
            y0 = x + h * t
            y, ok = C.correct(y0, t, y0)
            if ok and np.linalg.norm(y - x) < 3 * h:
                break
            h /= 2
        else:
            return None, np.array(pts), "Newton failed"
        tn = C.tangent(y)
        if tn @ t < 0:
            tn = -tn
        x, t = y, tn
        pts.append(x.copy())
        if x.min() < -1e-9:
            return None, np.array(pts), "left the chamber"
    return None, np.array(pts), "too many steps"


def arcs_of(c, cur, known_ids, Pts):
    C = Curve(cur["ideal"])
    known = np.array([Pts[i] for i in known_ids])
    seeds = level_points(cur, levels_of(cur))
    out, bad = [], []
    for s in seeds:
        ends = []
        for sgn in (1.0, -1.0):
            r = follow(C, s, sgn * C.tangent(s), known)
            if r[0] is None:
                bad.append((c, r[2], s.tolist()))
                break
            ends.append(r)
        if len(ends) < 2:
            continue
        (ja, Qa), (jb, Qb) = ends
        Q = np.vstack([Qa[::-1], Qb[1:]])
        out.append({"curve": c, "ends": (known_ids[ja], known_ids[jb]), "pts": Q, "on": set(cur["on"])})
    return out, bad, len(seeds)


def main(group):
    D = json.load(open(os.path.join(HERE, f"a4_curves_{group}.json")))
    curves = D["curves"]
    # (every exact point once)
    allp, owner = [], []
    for c, cur in enumerate(curves):
        for p in cur["points"]:
            allp.append(np.array([float(v) for v in p["beta"]]))
            owner.append(c)
    P_ = np.array(allp)
    tree = cKDTree(P_)
    rep = list(range(len(P_)))
    for i, j in sorted(tree.query_pairs(1e-12)):
        a, b = rep[i], rep[j]
        while rep[a] != a:
            a = rep[a]
        while rep[b] != b:
            b = rep[b]
        if a != b:
            rep[max(a, b)] = min(a, b)

    def r_(i):
        while rep[i] != i:
            i = rep[i]
        return i
    ids = sorted({r_(i) for i in range(len(P_))})
    pid = {i: n for n, i in enumerate(ids)}
    Pts = [P_[i] for i in ids]
    known_of = {}
    for i, c in enumerate(owner):
        known_of.setdefault(c, set()).add(pid[r_(i)])
    print(f"  {len(Pts)} exact points", flush=True)
    t0 = time.time()
    jobs = [(c, curves[c], sorted(known_of[c])) for c in sorted(known_of)]

    def job(a):
        c, cur, kid = a
        try:
            return arcs_of(c, cur, kid, Pts)
        except Exception as e:  # noqa: BLE001
            return [], [(c, f"error {e}", None)], 0
    segs, bad, nseeds = [], [], 0
    with ThreadPoolExecutor(4) as ex:
        for n, (o, b, k) in enumerate(ex.map(job, jobs)):
            segs += o
            bad += b
            nseeds += k
            if n % 25 == 0:
                print(f"  curve {n}/{len(jobs)}: {len(segs)} arcs, {len(bad)} failures ({time.time() - t0:.0f}s)",
                      flush=True)
    # (an arc found from two levels: kept once)
    keep, seen = [], []
    for s in segs:
        Q = s["pts"]
        mid = Q[len(Q) // 2]
        dup = False
        for t in seen:
            if t["curve"] == s["curve"] and set(t["ends"]) == set(s["ends"]):
                R = t["pts"]
                if np.min(np.linalg.norm(R - mid, axis=1)) < 1e-6:
                    dup = True
                    break
        if not dup:
            seen.append(s)
            keep.append(s)
    X = {"surfaces": D["surfaces"], "points": Pts, "segments": keep}
    pickle.dump(X, open(os.path.join(HERE, f"a4_arcs_{group}.pkl"), "wb"))
    print(f"{len(keep)} arcs ({len(segs) - len(keep)} found twice) from {nseeds} level points; "
          f"{len(bad)} traces failed")
    for b in bad[:20]:
        print("   ", b[0], b[1])


if __name__ == "__main__":
    for g in sys.argv[1:]:
        main(g)
