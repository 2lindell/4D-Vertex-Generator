"""Where each line ends, exactly, from the equations (line_ends.json).

A line is the curve where two surfaces meet; it ends on a point or on another line. Each verified piece of each line
(exact_defs.json) is followed from each of its ends along that curve (Newton steps onto both surfaces), outward,
classifying as it goes, until either:
  - it stops being the line: the change is bisected along the curve, and the end solved exactly as the meeting of the
    line's two surfaces with a third surface through it (one crossing the curve); the exact end is named by the
    classifier (a point, or the line it ends on); or
  - it leaves the half-cell while still the line: it goes on into the next cell (another image of the line's pieces
    carries it on there), so that piece end is not an end.
The followed stretch (from the stored end to the exact end) extends the piece where it stopped short.

    python line_ends.py [lines]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
from edge_relabel import lab_edge
from pieces import fval, grad, in_half, nrm, project

HALF = [("b1=0", np.eye(4)[0]), ("b2=0", np.eye(4)[1]), ("b3=0", np.eye(4)[2]), ("b4=0", np.eye(4)[3]),
        ("b3=b4", np.array([0, 0, 1, -1.0]))]
LIB = None


def lib():
    global LIB
    if LIB is None:
        planes, curved = osd.library()
        LIB = {s.name: s for s in list(planes) + list(curved)}
    return LIB


def lab(x):
    """The classifier's name at x, its rounding residues cleared (a coordinate within 1e-10 of 0, or beta3 within
    1e-10 of beta4, made exact), retried at other hull tolerances where the hull fails."""
    from classify import classify, signature
    from probe import label_of_sig, seed_from_beta
    x = nrm(x)
    x = np.where(np.abs(x) < 1e-10, 0.0, x)
    if abs(x[2] - x[3]) < 1e-10:
        x[2] = x[3] = (x[2] + x[3]) / 2
    x = osd.home(nrm(x))
    for tol in (1e-7, 1e-9, 1e-6):
        try:
            info = classify(seed_from_beta(x), hull_tol=tol)
        except Exception:
            continue
        return label_of_sig(signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items()))
    return "ERR"


def dist(s, x):
    return abs(fval(s, x)) / (np.linalg.norm(grad(s, x)) + 1e-300)


def pair(P, names):
    """Two of the line's surfaces meeting transversally along the piece (planes first: the steadiest)."""
    L = lib()
    x = P[len(P) // 2]
    names = sorted((n for n in names if n in L), key=lambda n: (not n.startswith("plane"), n))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = grad(L[a], x), grad(L[b], x)
            if abs(ga @ gb) / (np.linalg.norm(ga) * np.linalg.norm(gb) + 1e-300) < 0.99:
                return a, b
    return None


def tangent(A, B, x, prev):
    t = np.linalg.svd(np.vstack([grad(A, x), grad(B, x), np.ones(4)]))[2][-1]
    return t if t @ prev >= 0 else -t


def follow(job):
    try:
        return _follow(job)
    except Exception as e:
        return {"line": job[0], "piece": job[1], "side": job[2], "type": f"failed: {type(e).__name__}: {e}"}


def _follow(job):
    """One end of one piece: followed outward until the line ends or leaves the half-cell."""
    L_, k, side, P, names = job
    L = lib()
    P = np.asarray(P, float)
    if side == "start":
        P = P[::-1]
    ab = pair(P, names)
    if ab is None:
        return {"line": L_, "piece": k, "side": side, "type": "no surfaces"}
    A, B = L[ab[0]], L[ab[1]]
    # (a start on the line: the stored end, or the nearest point inward that is the line)
    i0 = len(P) - 1
    x = nrm(project([A, B], P[i0]))
    tries = 0
    while lab(x) != L_ and i0 > 0 and tries < 12:
        i0 -= max(1, len(P) // 12)
        i0 = max(i0, 0)
        x = nrm(project([A, B], P[i0]))
        tries += 1
    if lab(x) != L_:
        return {"line": L_, "piece": k, "side": side, "type": "not the line at its end", "surfaces": ab}
    prev = P[-1] - P[max(0, len(P) - 2)] if len(P) > 1 else np.ones(4)
    path, step, a = [x], 1e-3, x
    for _ in range(400):
        t = tangent(A, B, a, prev)
        y = nrm(project([A, B], a + step * t / np.linalg.norm(t)))
        if not in_half(y, 1e-12):
            # (the face it reaches: the exact crossing, and whether it is still the line there)
            n, v = min(HALF, key=lambda nv: y @ nv[1])           # (the face it is furthest past)
            fa = max(a @ v, 0.0)
            f = nrm(project([A, B, osd.Plane(v, n)], a + fa / (fa - y @ v) * (y - a)))
            path.append(f)
            if lab(f) == L_:
                return {"line": L_, "piece": k, "side": side, "type": "goes on across a face", "face": n,
                        "at_face": f.round(12).tolist(), "surfaces": ab, "path": np.array(path).round(12).tolist()}
            y = f
        ly = lab(y)
        if ly != L_:
            # (bisect the change along the curve)
            lo, hi = a, y
            for _ in range(36):
                m = nrm(project([A, B], (lo + hi) / 2))
                if lab(m) == L_:
                    lo = m
                else:
                    hi = m
                if np.linalg.norm(hi - lo) < 1e-10:
                    break
            m = (lo + hi) / 2
            best = None
            for cn, C in L.items():                     # (a third surface through it, crossing the curve)
                if cn in ab or dist(C, m) > 1e-6:
                    continue
                J = np.vstack([grad(A, m), grad(B, m), grad(C, m), np.ones(4)])
                if np.linalg.svd(J, compute_uv=False)[-1] < 1e-6:
                    continue
                e = nrm(project([A, B, C], m))
                d = np.linalg.norm(e - m)
                if best is None or d < best[0]:
                    best = (d, cn, e)
            beyond = lab(nrm(project([A, B], hi + 3e-4 * (hi - lo) / (np.linalg.norm(hi - lo) + 1e-300))))
            path.append(best[2] if best else m)
            out = {"line": L_, "piece": k, "side": side, "type": "ends", "surfaces": ab,
                   "beyond": beyond, "path": np.array(path).round(12).tolist()}
            if best and best[0] < 1e-5:
                out.update({"exact": best[2].round(12).tolist(), "third_surface": best[1], "on": lab(best[2]),
                            "off_bisected": float(best[0])})
                if out["on"] == L_:                       # (the end point itself reads as the line: where it meets
                    out["on"] = f"unnamed point ({L_} | {beyond})"   # the next shape, a point not yet named)
            else:
                out.update({"approx": m.round(12).tolist(), "on": lab(m)})
            return out
        path.append(y)
        prev, a = y - a, y
        step = min(step * 1.5, 0.01)
    return {"line": L_, "piece": k, "side": side, "type": "no end within reach", "surfaces": ab}


def main(only=None):
    ex = json.load(open("exact_defs.json"))["lines"]
    jobs = []
    for L_, v in ex.items():
        if only and L_ not in only:
            continue
        for k, (P, names) in enumerate(zip(v["points"], v["surfaces"])):
            if len(P) >= 2:
                jobs += [(L_, k, "start", P, names), (L_, k, "end", P, names)]
    with Pool(4) as pool:
        res = pool.map(follow, jobs, chunksize=1)
    out = {}
    for r in res:
        out.setdefault(r["line"], []).append(r)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "lines": out}, open("line_ends.json", "w"), indent=1)
    for L_ in sorted(out):
        ends = [r for r in out[L_] if r["type"] == "ends"]
        print(f"{L_}: ends on {sorted({r['on'] for r in ends})} "
              f"(exact {sum('exact' in r for r in ends)}/{len(ends)}); "
              f"across faces {sum(r['type'] == 'goes on across a face' for r in out[L_])}; "
              f"other {[r['type'] for r in out[L_] if r['type'] not in ('ends', 'goes on across a face')]}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
