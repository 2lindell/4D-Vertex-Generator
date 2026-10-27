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
    if len(P) < 24:                                      # (a short piece: points along it, on the curve)
        Q = [P[0]]
        for p, q in zip(P[:-1], P[1:]):
            Q += [nrm(project([A, B], p + s * (q - p))) for s in np.linspace(0, 1, max(2, 24 // (len(P) - 1)) + 1)[1:]]
        P = np.array(Q)
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


# (stored pieces that are not their line: classified along the curve, line_ends.py --check; X45 is a region, its one
# stored piece lies on X27; X80's piece 2 lies on X31; X18's piece 1 is regions throughout)
NOT_LINES = {"X45"}
# (ends given by hand where the classifier reads something else: X81 ends on X2 and on A1, like X80 (its pieces 1 and
# 3 run to the vertex, A1; the stretch before it reads as X79))
# X29 runs from X8 to A1 the same way (its pieces 1 and 3 run to the vertex; the stretch before it reads as F5)
USER_ENDS = {("X81", 1, "start"): "A1", ("X81", 3, "start"): "A1", ("X29", 1, "end"): "A1", ("X29", 3, "end"): "A1"}


def user_ends():
    d = json.load(open("line_ends.json"))
    ex = json.load(open("exact_defs.json"))["lines"]
    for (L_, k, side), name in USER_ENDS.items():
        for r in d["lines"].get(L_, []):
            if r["piece"] == k and r["side"] == side:
                P = np.asarray(ex[L_]["points"][k], float)
                at = P[0] if side == "start" else P[-1]
                r.update({"type": "ends", "on": name, "exact": nrm(at).round(12).tolist(), "given": True})
                r.pop("path", None)
    json.dump(d, open("line_ends.json", "w"), indent=1)
DROP_PIECES = {("X80", 2), ("X18", 1)}


def main(only=None):
    ex = json.load(open("exact_defs.json"))["lines"]
    ex = {L_: v for L_, v in ex.items() if L_ not in NOT_LINES}
    jobs = []
    for L_, v in ex.items():
        if only and L_ not in only:
            continue
        for k, (P, names) in enumerate(zip(v["points"], v["surfaces"])):
            if len(P) >= 2 and (L_, k) not in DROP_PIECES:
                jobs += [(L_, k, "start", P, names), (L_, k, "end", P, names)]
    with Pool(4) as pool:
        res = pool.map(follow, jobs, chunksize=1)
    old = json.load(open("line_ends.json"))["lines"] if only else {}
    out = {L_: rs for L_, rs in old.items() if L_ not in only and L_ not in NOT_LINES} if only else {}
    for r in res:
        out.setdefault(r["line"], []).append(r)
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "lines": out}, open("line_ends.json", "w"), indent=1)
    for L_ in sorted(out):
        ends = [r for r in out[L_] if r["type"] == "ends"]
        print(f"{L_}: ends on {sorted({r['on'] for r in ends})} "
              f"(exact {sum('exact' in r for r in ends)}/{len(ends)}); "
              f"across faces {sum(r['type'] == 'goes on across a face' for r in out[L_])}; "
              f"other {[r['type'] for r in out[L_] if r['type'] not in ('ends', 'goes on across a face')]}", flush=True)




def refine(r):
    """An end solved on every surface through it at once (least squares: where many surfaces meet, one third surface
    leaves it slightly off); surfaces that only pass near it are dropped and it is solved again; then named."""
    L = lib()
    x = np.asarray(r.get("exact") or r.get("approx"), float)
    A, B = L[r["surfaces"][0]], L[r["surfaces"][1]]

    def jac(s, y):
        if isinstance(s, osd.Plane):
            return np.asarray(s.n, float)
        return np.array([(fval(s, y + 1e-7 * u) - fval(s, y - 1e-7 * u)) / 2e-7 for u in np.eye(4)])

    thru = [s for n, s in L.items() if dist(s, x) < 1e-4]
    y = x.copy()
    for _ in range(4):
        if len(thru) < 3:
            break
        z = y.copy()
        for _ in range(40):
            F = np.array([fval(s, z) for s in thru] + [z.sum() - 1])
            J = np.vstack([jac(s, z) for s in thru] + [np.ones(4)])
            dz = np.linalg.lstsq(J, -F, rcond=None)[0]
            z = z + dz
            if np.linalg.norm(dz) < 1e-15:
                break
        d = np.array([dist(s, z) for s in thru])
        keep = [s for s, e in zip(thru, d) if e < 1e-8]
        y = z
        if len(keep) == len(thru):
            break
        thru = keep
    ok = len(thru) >= 3 and dist(A, y) < 1e-8 and dist(B, y) < 1e-8 and np.linalg.norm(y - x) < 1e-4
    if not ok:
        return {"on": lab(x), "surfaces_through": 0, "unrefined": True}
    y = nrm(np.where(np.abs(y) < 1e-12, 0.0, y))
    return {"exact": y.round(12).tolist(), "on": lab(y), "surfaces_through": len(thru),
            "moved": float(np.linalg.norm(y - x))}


def refine_all():
    d = json.load(open("line_ends.json"))
    jobs = [(L_, i) for L_, rs in d["lines"].items() for i, r in enumerate(rs)
            if r["type"] == "ends" and (r.get("exact") or r.get("approx"))]
    with Pool(4) as pool:
        res = pool.map(refine, [d["lines"][L_][i] for L_, i in jobs], chunksize=1)
    for (L_, i), u in zip(jobs, res):
        r = d["lines"][L_][i]
        r.update({k: v for k, v in u.items()})
        if r["on"] == L_:
            r["on"] = f"unnamed point ({L_} | {r.get('beyond')})"
    snap(d)
    json.dump(d, open("line_ends.json", "w"), indent=1)
    for L_ in sorted(d["lines"]):
        ends = [r for r in d["lines"][L_] if r["type"] == "ends"]
        print(f"{L_}: ends on {sorted({r['on'] for r in ends})}", flush=True)


def snap(d):
    """Ends left unsure (not refined, not classified, or reading as their own line) moved onto the nearest point known
    exactly (a vertex of the half-cell, or another end that was refined and named) lying on the line's two surfaces."""
    L = lib()
    cross = [(p["name"], np.array(p["beta"], float)) for p in json.load(open("cross_ring_points.json"))["points"]]
    # (an end at a cross-ring point is that point: named by the cross ring's sequence, which the classifier cannot always
    # read at the exact point)
    for rs in d["lines"].values():
        for r in rs:
            if r["type"] == "ends" and (r.get("exact") or r.get("approx")):
                x = np.asarray(r.get("exact") or r.get("approx"), float)
                A, B = L[r["surfaces"][0]], L[r["surfaces"][1]]
                near = min(cross, key=lambda c: np.linalg.norm(c[1] - x))
                dn = np.linalg.norm(near[1] - x)
                # (close by, or a little further where the point lies exactly on the line's own two surfaces)
                if dn < 1e-4 or (dn < 5e-3 and dist(A, near[1]) < 1e-8 and dist(B, near[1]) < 1e-8):
                    r.update({"exact": near[1].round(15).tolist(), "on": near[0], "cross_ring": True})
                    r.pop("unrefined", None)
    known = [np.array(v, float) for v in ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, .5, .5])]
    for rs in d["lines"].values():
        for r in rs:
            if r["type"] == "ends" and not r.get("unrefined") and r.get("exact") and not r["on"].startswith(
                    ("ERR", "unnamed", "new:")) and r["on"] not in d["lines"]:
                known.append(np.array(r["exact"], float))
    for L_, rs in d["lines"].items():
        for r in rs:
            if r["type"] != "ends":
                continue
            unsure = r.get("unrefined") or r["on"] == "ERR" or r["on"].startswith(f"unnamed point ({L_} | {L_}") \
                or "ERR" in r["on"]
            if not unsure:
                continue
            x = np.asarray(r.get("exact") or r.get("approx"), float)
            A, B = L[r["surfaces"][0]], L[r["surfaces"][1]]
            near = sorted(((np.linalg.norm(k - x), i) for i, k in enumerate(known)
                           if dist(A, k) < 1e-8 and dist(B, k) < 1e-8))
            if near and near[0][0] < 1e-4:
                k = known[near[0][1]]
                r.update({"exact": k.round(12).tolist(), "on": lab(k), "snapped": float(near[0][0])})
                r.pop("unrefined", None)
    # (an end still reading as a wall or region, or unnamed: where the line meets a copy of another line)
    names = set(d["lines"])
    ex = {k: v for k, v in json.load(open("exact_defs.json"))["lines"].items() if k not in NOT_LINES}
    todo = []
    for L_, rs in d["lines"].items():
        for r in rs:
            if r["type"] != "ends" or r.get("cross_ring"):
                continue
            on = r["on"]
            if on == "ERR" or on.startswith("unnamed") or (on not in names and on.startswith(("X", "F", "T"))):
                todo.append((L_, r))
    with Pool(4) as pool:
        res = pool.map(meet_line, [(L_, r, ex) for L_, r in todo], chunksize=1)
    for (L_, r), m in zip(todo, res):
        if m:
            r.update(m)


class Image:
    """A surface carried by a group element g: the points y with s(g^-1 y) = 0."""

    def __init__(self, s, gi):
        self.s, self.gi, self.name = s, gi, f"{s.name} @g{gi}"

    def f(self, B):
        return self.s.f(np.atleast_2d(B) @ GINV()[self.gi].T)


_G = {}


def GROUP():
    if "GG" not in _G:
        from t2_pieces import FOLD, extended
        G = extended()
        _G["GG"] = np.concatenate([G, G @ FOLD])
        _G["GINV"] = np.linalg.inv(_G["GG"])
    return _G["GG"]


def GINV():
    GROUP()
    return _G["GINV"]


def meet_line(job, reach=3e-3):
    """Where the line L_ (its curve, near the end r) meets any copy of another line: the other line's own two surfaces
    carried by the symmetry that brings its piece there; solved exactly on the line's two surfaces and one of those,
    checked on the other."""
    L_, r, ex = job
    L = lib()
    GG = GROUP()
    x = np.asarray(r.get("exact") or r.get("approx"), float)
    A, B = L[r["surfaces"][0]], L[r["surfaces"][1]]
    best = None
    for M, v in ex.items():
        if M == L_:
            continue
        for P, names in zip(v["points"], v["surfaces"]):
            P = np.asarray(P, float)
            if len(P) < 2:
                continue
            ab = pair(P, names)
            if ab is None:
                continue
            Y = np.einsum("gij,nj->gni", GG, P)
            sm = Y.sum(-1, keepdims=True)
            ok = (np.abs(sm) > 1e-9).all((1, 2))
            idx = np.flatnonzero(ok)
            Y = Y[ok] / sm[ok]
            near = np.linalg.norm(Y - x, axis=-1).min(1) < reach
            for gi in idx[near]:
                C1, C2 = Image(L[ab[0]], gi), Image(L[ab[1]], gi)
                for C, D in ((C1, C2), (C2, C1)):
                    J = np.vstack([grad(A, x), grad(B, x), grad(C, x), np.ones(4)])
                    if np.linalg.svd(J, compute_uv=False)[-1] < 1e-6:
                        continue
                    y = nrm(project([A, B, C], x))
                    if np.linalg.norm(y - x) > reach or dist(D, y) > 1e-8:
                        continue
                    if best is None or np.linalg.norm(y - x) < best[0]:
                        best = (np.linalg.norm(y - x), M, y)
    if best is None:
        return None
    y = best[2]
    nm = lab(y)
    return {"exact": y.round(12).tolist(), "on": nm if nm not in (L_, "ERR") else best[1], "meets_line": best[1],
            "moved": float(best[0])}


def check_piece(job):
    """Is a stored piece its line? Points along it moved onto its two surfaces, classified."""
    L_, k, P, names = job
    P = np.asarray(P, float)
    ab = pair(P, names)
    if ab is None:
        return (L_, k, len(P), None, [])
    A, B = lib()[ab[0]], lib()[ab[1]]
    Q = [nrm(P[0] * (1 - s) + P[1] * s) for s in np.linspace(0.1, 0.9, 5)] if len(P) == 2 else \
        [P[i] for i in np.linspace(1, len(P) - 2, 5).astype(int)]
    return (L_, k, len(P), ab, [lab(nrm(project([A, B], q))) for q in Q])


def check_all():
    ex = json.load(open("exact_defs.json"))["lines"]
    jobs = [(L_, k, P, ex[L_]["surfaces"][k]) for L_ in ex for k, P in enumerate(ex[L_]["points"]) if len(P) >= 2]
    with Pool(4) as pool:
        for L_, k, n, ab, labs in pool.imap_unordered(check_piece, jobs):
            print(("OK  " if labs and all(x == L_ for x in labs) else "BAD "), L_, k, n, ab, labs, flush=True)



def passes():
    """A followed path that passes through a known point lying on the line's curve (a cross-ring point, or another
    line's named end) stops there: the line ends at that point (steps along the curve can pass over a single point)."""
    d = json.load(open("line_ends.json"))
    stored = json.load(open("exact_defs.json"))["lines"]
    L = lib()
    known = [(p["name"], np.array(p["beta"], float)) for p in json.load(open("cross_ring_points.json"))["points"]]
    for rs in d["lines"].values():
        for r in rs:
            if r["type"] == "ends" and r.get("exact") and not r["on"].startswith(("unnamed", "new:", "ERR")) \
                    and r["on"] not in d["lines"]:
                known.append((r["on"], np.array(r["exact"], float)))
    changed = []
    for L_, rs in d["lines"].items():
        for r in rs:
            if not r.get("path") or r["type"] not in ("ends", "goes on across a face"):
                continue
            A, B = L[r["surfaces"][0]], L[r["surfaces"][1]]
            on_curve = [(n, k) for n, k in known if dist(A, k) < 1e-8 and dist(B, k) < 1e-8]
            P = np.asarray(r["path"], float)
            # (only past the stored piece's own end: a line goes on through a point where other lines end on it, so
            # the verified piece is never cut at one)
            S0 = np.asarray(stored[L_]["points"][r["piece"]], float)
            own = S0[-1] if r["side"] == "end" else S0[0]
            i_own = int(np.argmin(np.linalg.norm(P - own, axis=1)))
            for i in range(i_own, len(P) - 1):
                a, y = P[i], P[i + 1]
                hit = [(np.linalg.norm(k - a), n, k) for n, k in on_curve
                       if np.linalg.norm(k - a) > 1e-9 and
                       np.linalg.norm(k - a) + np.linalg.norm(y - k) - np.linalg.norm(y - a) < 1e-9]
                if hit and i + 1 < len(P) - 1:
                    _, n, k = min(hit, key=lambda h: h[0])
                    r.update({"path": np.vstack([P[:i + 1], k]).round(12).tolist(), "exact": k.round(15).tolist(),
                              "on": n, "type": "ends", "passed_point": True})
                    changed.append((L_, r["piece"], r["side"], n))
                    break
    json.dump(d, open("line_ends.json", "w"), indent=1)
    for c in changed:
        print("stops at a point it passed:", c)


def extend():
    """The line pieces in exact_defs.json brought to their exact ends: a piece that stopped short goes on along its
    curve to the end found; one that ran past it is cut back there; a piece that reaches a face of the half-cell stops
    exactly on it. Pieces found not to be their line are dropped, and X45 (a region) is no longer a line."""
    ex = json.load(open("exact_defs.json"))
    d = json.load(open("line_ends.json"))["lines"]
    for L_ in NOT_LINES:
        ex["lines"].pop(L_, None)
    W = json.load(open("working_data.json"))
    not_named = set(W["walls"]) | set(W["regions"])
    unresolved = lambda on: on in not_named or on.startswith(("ERR", "unnamed", "new:"))
    for L_, v in ex["lines"].items():
        pts = [np.asarray(P, float) for P in v["points"]]
        for r in sorted(d.get(L_, []), key=lambda r: r["side"] != "end"):
            if r["type"] not in ("ends", "goes on across a face") or not r.get("path"):
                continue
            if r["type"] == "ends" and unresolved(r["on"]):     # (an end not yet named: the stored extent kept)
                print(f"  {L_} piece {r['piece']} {r['side']}: end unresolved ({r['on'][:30]}); the stored end kept")
                continue
            P = pts[r["piece"]]
            path = np.asarray(r["path"], float)
            own, other = (P[-1], P[0]) if r["side"] == "end" else (P[0], P[-1])
            if len(P) > 2 and np.linalg.norm(path[0] - own) < 1e-3 and \
                    np.linalg.norm(path[-1] - other) < np.linalg.norm(path[-1] - own):
                print(f"  {L_} piece {r['piece']} {r['side']}: the followed path turns back along the line; "
                      f"the stored end kept")
                continue
            if r.get("exact"):
                # (the end kept exactly on the line's own two surfaces: a point named from nearby is projected onto
                # them)
                A, B = lib()[r["surfaces"][0]], lib()[r["surfaces"][1]]
                e = np.asarray(r["exact"], float)
                path[-1] = e if dist(A, e) < 1e-9 and dist(B, e) < 1e-9 else nrm(project([A, B], e))
            if r["side"] == "end":
                i0 = int(np.argmin(np.linalg.norm(P - path[0], axis=1)))
                pts[r["piece"]] = np.vstack([P[:i0], path])
            else:
                i0 = int(np.argmin(np.linalg.norm(P - path[0], axis=1)))
                pts[r["piece"]] = np.vstack([path[::-1], P[i0 + 1:]])
        keep = [k for k in range(len(pts)) if (L_, k) not in DROP_PIECES]
        v["points"] = [pts[k].round(12).tolist() for k in keep]
        v["surfaces"] = [v["surfaces"][k] for k in keep]
        v["pieces"] = len(keep)
        v["ends"] = [{"on": r["on"], "at": r.get("exact") or r.get("approx")} for r in d.get(L_, [])
                     if r["type"] == "ends"]
    ex["_about_ends"] = "line pieces brought to their exact ends (line_ends.py --extend)"
    json.dump(ex, open("exact_defs.json", "w"), indent=1)
    for L_, v in sorted(ex["lines"].items()):
        print(L_, [len(P) for P in v["points"]], sorted({e["on"] for e in v["ends"]}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--user"]:
        user_ends()
        sys.exit()
    if sys.argv[1:2] == ["--passes"]:
        passes()
        sys.exit()
    if sys.argv[1:2] == ["--extend"]:
        extend()
        sys.exit()
    if sys.argv[1:2] == ["--check"]:
        check_all()
        sys.exit()
    if sys.argv[1:2] == ["--refine"]:
        refine_all()
    elif sys.argv[1:2] == ["--snap"]:
        d = json.load(open("line_ends.json"))
        snap(d)
        json.dump(d, open("line_ends.json", "w"), indent=1)
        for L_ in sorted(d["lines"]):
            print(f"{L_}: ends on {sorted({r['on'] for r in d['lines'][L_] if r['type'] == 'ends'})}")
    else:
        main(set(sys.argv[1].split(",")) if len(sys.argv) > 1 and sys.argv[1] else None)
