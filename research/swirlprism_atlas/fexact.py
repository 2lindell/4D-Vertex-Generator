"""Exact extent of the named wall shapes (F1, F4, F5) inside their walls.

Each wall meets the half-cell in a polygon. A triangular grid on it is classified, every grid edge across which
the target shape starts or stops is bisected to its boundary point (with the class on the other side), and the
boundary points are then fitted with exact lines (great circles) in the wall.

    python fexact.py grid F4,F5 b2=b4     # classify the wall's grid and bisect each shape's boundary
    python fexact.py fit F1 b1=b2         # fit exact lines to a shape's boundary points
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np
from probe import _one

_P = (1 + 5 ** 0.5) / 2
NORMALS = {   # each wall n . beta = 0
    "b1=b2": [1, -1, 0, 0],
    "b1=phi2*b2": [1, -_P ** 2, 0, 0],
    "b1=b3": [1, 0, -1, 0],
    "b1=phi-2*b2": [1, -_P ** -2, 0, 0],
    "b2=b4": [0, 1, 0, -1],
    "b1=b4": [1, 0, 0, -1],
    "b3=phi-2*(b1-b4)": [-_P ** -2, 0, 1, _P ** -2],     # walls found as half-turn copies of the patches above
    "b4=phi-2*(b2-b3)": [0, -_P ** -2, _P ** -2, 1],
    "b1=phi-2*(b3-b4)": [1, 0, -_P ** -2, _P ** -2],
    "b1=0": [1, 0, 0, 0],
    "b2=0": [0, 1, 0, 0],
    "b4=0": [0, 0, 0, 1],
}
HALF = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, -1]], float)  # the half-cell: rows . beta >= 0


def wall_polygon(normal):
    """Corners (barycentric, in order) of the wall's polygon inside the half-cell."""
    from itertools import combinations
    n = np.asarray(normal, float)
    pts = []
    for i, j in combinations(range(len(HALF)), 2):
        A = np.vstack([n, HALF[i], HALF[j], np.ones(4)])
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        b = np.linalg.solve(A, [0, 0, 0, 1])
        if np.all(HALF @ b >= -1e-12) and not any(np.allclose(b, q) for q in pts):
            pts.append(b)
    P = np.array(pts)
    c = P.mean(axis=0)
    u = np.linalg.svd(P - c)[2][:2]
    ang = np.arctan2(*((P - c) @ u.T)[:, ::-1].T)
    return P[np.argsort(ang)]


WALLS = {w: wall_polygon(n).tolist() for w, n in NORMALS.items()}


def _lab(b):
    b = np.asarray(b, float)
    return _one(b / b.sum())[0]


def grid(corners, n):
    """A triangular grid on each triangle of the polygon's fan (shared edges are sampled twice)."""
    C = np.array(corners, float)
    pts, idx = [], {}
    for f in range(1, len(C) - 1):
        T = (C[0], C[f], C[f + 1])
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                idx[(f, i, j)] = len(pts)
                pts.append((i * T[0] + j * T[1] + k * T[2]) / n)
    return pts, idx


def _bisect(job, steps=34):
    a, b, la, lb = job
    a, b = np.array(a), np.array(b)
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        m = (lo + hi) / 2
        if _lab(a + m * (b - a)) == la:
            lo = m
        else:
            hi = m
    return {"beta": (a + (lo + hi) / 2 * (b - a)).tolist(), "inside": la, "outside": _lab(a + hi * (b - a)),
            "far": lb, "t": [lo, hi]}


def run_grid(targets, wall, n=24, procs=4):
    pts, idx = grid(WALLS[wall], n)
    with Pool(procs) as pool:
        labs = pool.map(_lab, pts, chunksize=4)
        edges = set()
        for (f, i, j), a in idx.items():
            for di, dj in ((1, 0), (0, 1), (-1, 1)):
                b = idx.get((f, i + di, j + dj))
                if b is not None:
                    edges.add((min(a, b), max(a, b)))
        from collections import Counter
        print(wall, "grid labels:", Counter(labs).most_common(), flush=True)
        for target in targets:
            jobs = []
            for a, b in sorted(edges):
                if (labs[a] == target) != (labs[b] == target):
                    ins, out = (a, b) if labs[a] == target else (b, a)
                    jobs.append((pts[ins].tolist(), pts[out].tolist(), target, labs[out]))
            bnd = pool.map(_bisect, jobs, chunksize=2)
            json.dump({"target": target, "wall": wall, "n": n, "corners": WALLS[wall],
                       "points": [p.tolist() for p in pts], "labels": labs, "boundary": bnd},
                      open(f"fexact_{target}_{wall}.json", "w"))
            print(" ", target, len(jobs), "boundary crossings; across:", Counter(b["outside"] for b in bnd).most_common(),
                  "| one step further:", Counter(b["far"] for b in bnd).most_common(), flush=True)


PHI = (1 + 5 ** 0.5) / 2


def golden_form(x, maxq=12, tol=1e-7):
    """x as (a + b phi)/q with small integers, or None."""
    best = None
    for q in range(1, maxq + 1):
        for b in range(-24, 25):
            a = round(x * q - b * PHI)
            if abs((a + b * PHI) / q - x) < tol:
                cand = (abs(a) + abs(b) + q, a, b, q)
                if best is None or cand < best:
                    best = cand
    return best and best[1:]


def fmt_golden(f):
    a, b, q = f
    if b == 0:
        s = f"{a}"
    elif a == 0:
        s = f"{'' if b == 1 else '-' if b == -1 else b}φ"
    else:
        s = f"{a}{'+' if b > 0 else '-'}{'' if abs(b) == 1 else abs(b)}φ"
    return s if q == 1 else f"({s})/{q}"


def plane_through(B, wall_normal):
    """The second plane (normal m, m . beta = 0) holding all points B of a great circle in the wall. It is only
    defined up to adding multiples of the wall's normal, so each multiple that zeroes one coordinate is tried and
    the simplest golden form kept; returns (m, residual, text)."""
    B = np.asarray(B, float)
    w = np.asarray(wall_normal, float)
    wn = w / np.linalg.norm(w)
    P = B - np.outer(B @ wn, wn)
    m = np.linalg.svd(np.vstack([P, wn]))[2][-1]
    m = m - (m @ wn) * wn
    best = None
    cands = [m] + [m - (m[i] / w[i]) * w for i in range(4) if abs(w[i]) > 1e-12]
    for c in cands:
        c = c / c[np.argmax(np.abs(c))]
        snapped = [golden_form(v) for v in c]
        if any(f is None for f in snapped):
            continue
        mm = np.array([(a + b * PHI) / q for a, b, q in snapped])
        res = float(np.abs(B @ mm).max())
        if res > 1e-6:
            continue
        cost = sum(abs(a) + abs(b) + q for a, b, q in snapped)
        if best is None or cost < best[0]:
            terms = [f"{fmt_golden(f)}·β{i + 1}" for i, f in enumerate(snapped) if f[0] or f[1]]
            best = (cost, mm, res, " + ".join(terms).replace("+ -", "− ") + " = 0")
    if best:
        return best[1], best[2], best[3]
    return m / np.abs(m).max(), float(np.abs(B @ m).max()), "not golden"


def _lines(B, normal, tol=1e-7):
    """Split boundary points into straight runs (great circles in the wall): repeatedly take the plane through the
    most points (RANSAC over pairs), keep its inliers."""
    B = [np.asarray(b, float) / np.sum(b) for b in B]
    out = []
    rest = list(range(len(B)))
    w = np.asarray(normal, float)
    while len(rest) >= 2:
        best = None
        for a in range(len(rest)):
            for c in range(a + 1, len(rest)):
                x, y = B[rest[a]], B[rest[c]]
                M = np.vstack([x, y, w])
                m = np.linalg.svd(M)[2][-1]
                inl = [k for k in rest if abs(B[k] @ m) < tol]
                if best is None or len(inl) > len(best):
                    best = inl
        if len(best) < 2:
            break
        out.append(best)
        rest = [k for k in rest if k not in best]
    return out, rest


def fit(target, wall, normal):
    d = json.load(open(f"fexact_{target}_{wall}.json"))
    from collections import defaultdict
    groups = defaultdict(list)
    for b in d["boundary"]:
        groups[b["outside"] if b["outside"] != "ERR" else b["far"] + "*"].append(b["beta"])
    out = []
    for nb, B in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        runs, left = _lines(B, normal)
        for r in runs:
            P = np.array([B[k] for k in r])
            m, res, text = plane_through(P, normal)
            Pn = P / P.sum(axis=1, keepdims=True)
            out.append({"neighbour": nb, "points": len(r), "plane": m.tolist(), "residual": res, "equation": text,
                        "span": [Pn[np.argmin(Pn @ np.ones(4))].tolist()]})
            print(f"{target} on {wall} | across {nb:6s} {len(r):3d} pts -> {text}  (max off {res:.1e})")
        if left:
            print(f"{target} on {wall} | across {nb:6s} {len(left):3d} isolated points", [np.round(np.array(B[k]) / np.sum(B[k]), 4).tolist() for k in left][:4])
    return out


# ---- exact patches ------------------------------------------------------------------------------------------
CONICS = {   # curved edges, as quadratic forms in beta (fitted from the boundary, golden coefficients, verified)
    ("F5", "b1=phi-2*b2"): lambda b: ((3 - 2 * PHI) * b[1] ** 2 + (2 - PHI) * b[1] * b[2] + (2 * PHI - 3) * b[1] * b[3]
                                      - b[2] ** 2 + b[3] ** 2),
    ("F5", "b2=b4"): lambda b: (5 * b[0] * b[1] - (3 - PHI) * b[0] * b[2] - (3 - PHI) * b[1] ** 2
                                - (2 * PHI - 1) * b[1] * b[2]),
}
PATCHES = [("F1", "b1=b2"), ("F1", "b1=phi2*b2"), ("F1", "b1=b3"), ("F4", "b1=0"), ("F4", "b2=0"),
           ("F4", "b1=phi-2*b2"), ("F5", "b1=phi-2*b2"), ("F5", "b2=b4"),
           ("F1", "b3=phi-2*(b1-b4)"), ("F1", "b4=phi-2*(b2-b3)"), ("F4", "b1=phi-2*(b3-b4)")]


def _clip(poly, m):
    """Keep the part of a polygon (barycentric corners) with m . beta >= 0."""
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        fa, fb = a @ m, b @ m
        if fa >= -1e-13:
            out.append(a)
        if (fa >= -1e-13) != (fb >= -1e-13):
            out.append(a + fa / (fa - fb) * (b - a))
    return out


def _clip_conic(poly, Q, inside_pt, n=64):
    """Keep the part with Q's sign at inside_pt: edges are cut where they cross the conic, and the conic arcs between
    an exit and the next entry are sampled from a pencil of lines through inside_pt."""
    sgn = np.sign(Q(inside_pt))
    f = lambda x: sgn * Q(x)

    def cross(a, b):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (f(a + mid * (b - a)) >= 0) == (f(a) >= 0):
                lo = mid
            else:
                hi = mid
        return a + (lo + hi) / 2 * (b - a)

    def on_conic(d):
        lo, hi = 0.0, 2.0
        while f(inside_pt + hi * d) >= 0 and hi < 64:
            hi *= 2
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(inside_pt + mid * d) >= 0:
                lo = mid
            else:
                hi = mid
        return inside_pt + lo * d
    out, exit_pt = [], None
    m = len(poly)
    for k in range(m):
        a, b = poly[k], poly[(k + 1) % m]
        ins_a, ins_b = f(a) >= 0, f(b) >= 0
        if ins_a:
            out.append(a)
        if ins_a and not ins_b:
            exit_pt = cross(a, b)
            out.append(exit_pt)
        elif not ins_a and ins_b:
            entry = cross(a, b)
            if exit_pt is not None:
                for t in np.linspace(0, 1, n)[1:-1]:
                    d = (1 - t) * (exit_pt - inside_pt) + t * (entry - inside_pt)
                    out.append(on_conic(d))
            out.append(entry)
    if out and exit_pt is not None and f(poly[0]) < 0:          # the arc closing the polygon
        entry = out[0]
        arc = [on_conic((1 - t) * (exit_pt - inside_pt) + t * (entry - inside_pt)) for t in np.linspace(0, 1, n)[1:-1]]
        out = out + arc
    return out


def patch(target, wall, tol=1e-7):
    """The exact patch: the wall polygon clipped by every fitted line (on the side holding the shape's grid points)
    and, for F5, by its conic. Returns corners (barycentric) and the edge list."""
    d = json.load(open(f"fexact_{target}_{wall}.json"))
    pts = np.array(d["points"], float)
    pts = pts / pts.sum(axis=1, keepdims=True)
    # the side of each edge is read from the shape's grid points strictly inside the wall polygon (a shape can also
    # run along the polygon's edge, where the wall meets one of its other walls or faces)
    normal = NORMALS[wall]
    nn = np.asarray(normal, float) / np.linalg.norm(normal)
    sides = np.array([h for h in HALF if abs(abs(h @ nn) - np.linalg.norm(h)) > 1e-9])   # not the wall itself
    interior = np.all(pts @ sides.T > 1e-9, axis=1)
    ins = pts[[k for k, l in enumerate(d["labels"]) if l == target and interior[k]]]
    from collections import defaultdict
    groups = defaultdict(list)
    for b in d["boundary"]:
        groups[b["outside"] if b["outside"] != "ERR" else b["far"] + "*"].append(b["beta"])
    edges = []
    for nb, B in groups.items():
        runs, _ = _lines(B, normal)
        for r in runs:
            if len(r) < 3:
                continue
            P = np.array([B[k] for k in r])
            m, res, text = plane_through(P, normal)
            if res > 1e-6 or text == "not golden":
                continue
            m = np.asarray(m, float) * (1 if np.mean(ins @ m) >= 0 else -1)
            if not any(np.allclose(m / np.abs(m).max(), e[0] / np.abs(e[0]).max(), atol=1e-6) for e in edges):
                edges.append((m, nb.rstrip("*"), text))
    poly = [np.array(c, float) / np.sum(c) for c in WALLS[wall]]
    for m, nb, text in edges:
        poly = _clip(poly, m)
    if (target, wall) in CONICS:
        Q = CONICS[(target, wall)]
        centre = ins.mean(axis=0)
        poly = _clip_conic(poly, Q, centre)
    return [p.tolist() for p in poly], [(m.tolist(), nb, text) for m, nb, text in edges]


def _in_polygon(q, V):
    inside = False
    for k in range(len(V)):
        (x1, y1), (x2, y2) = V[k], V[(k + 1) % len(V)]
        if (y1 > q[1]) != (y2 > q[1]) and q[0] < x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _dist_to_boundary(q, V):
    best = np.inf
    for k in range(len(V)):
        a, b = V[k], V[(k + 1) % len(V)]
        t = np.clip((q - a) @ (b - a) / max((b - a) @ (b - a), 1e-30), 0, 1)
        best = min(best, np.linalg.norm(a + t * (b - a) - q))
    return best


def check_patch(target, wall, poly, margin=0.01):
    """Grid points away from the patch's edge: the shape's inside the patch, the others outside."""
    d = json.load(open(f"fexact_{target}_{wall}.json"))
    P = np.array(d["points"], float)
    P = P / P.sum(axis=1, keepdims=True)
    n = np.asarray(NORMALS[wall], float)
    basis = np.linalg.svd(np.vstack([n, np.ones(4)]))[2][2:]
    V = np.array(poly) @ basis.T
    good = bad_out = bad_in = 0
    for q, lab in zip(P @ basis.T, d["labels"]):
        if _dist_to_boundary(q, V) < margin:
            continue
        inside = _in_polygon(q, V)
        if (lab == target) == inside:
            good += 1
        elif lab == target:
            bad_out += 1
        else:
            bad_in += 1
    return good, bad_out, bad_in


# ---- any wall shape: cached grids, automatic walls, curved edges, closure under the half-turn ------------------
GRID_N = 24


def wall_key(normal):
    """A name for a wall from its normal (golden coefficients)."""
    m = np.asarray(normal, float)
    m = m / m[np.argmax(np.abs(m))]
    parts = []
    for v in m:
        f = golden_form(v, tol=1e-6)
        parts.append(fmt_golden(f) if f else f"{v:.6f}")
    return "n(" + ",".join(parts) + ")"


def load_walls():
    """Every wall with a normal: the named ones and those found later (fexact_walls.json)."""
    import os
    extra = json.load(open("fexact_walls.json")) if os.path.exists("fexact_walls.json") else {}
    for k, n in extra.items():
        if k not in NORMALS:
            NORMALS[k] = n
            WALLS[k] = wall_polygon(n).tolist()


def add_wall(normal):
    import os
    key = wall_key(normal)
    for k, n in NORMALS.items():
        a, b = np.asarray(n, float), np.asarray(normal, float)
        if abs(abs(a @ b) - np.linalg.norm(a) * np.linalg.norm(b)) < 1e-9:
            return k
    extra = json.load(open("fexact_walls.json")) if os.path.exists("fexact_walls.json") else {}
    extra[key] = list(map(float, normal))
    json.dump(extra, open("fexact_walls.json", "w"), indent=1)
    NORMALS[key] = list(map(float, normal))
    WALLS[key] = wall_polygon(normal).tolist()
    return key


def grid_labels(wall, procs=4):
    """The classified grid of a wall, from any earlier run on it, else classified now (fexact_grid_<wall>.json)."""
    import glob
    import os
    path = f"fexact_grid_{wall}.json"
    if os.path.exists(path):
        d = json.load(open(path))
        return d["points"], d["labels"]
    for f in glob.glob("fexact_*_" + glob.escape(wall) + ".json"):
        d = json.load(open(f))
        if "labels" in d and d.get("n") == GRID_N:
            json.dump({"points": d["points"], "labels": d["labels"]}, open(path, "w"))
            return d["points"], d["labels"]
    pts, _ = grid(WALLS[wall], GRID_N)
    with Pool(procs) as pool:
        labs = pool.map(_lab, pts, chunksize=4)
    pts = [np.asarray(p).tolist() for p in pts]
    json.dump({"points": pts, "labels": labs}, open(path, "w"))
    return pts, labs


def trace(target, wall, procs=4):
    """Bisect the target's boundary on a wall from its (cached) grid."""
    import os
    path = f"fexact_{target}_{wall}.json"
    if os.path.exists(path):
        return
    pts, labs = grid_labels(wall)
    pts = [np.asarray(p, float) for p in pts]
    _, idx = grid(WALLS[wall], GRID_N)
    edges = set()
    for (f, i, j), a in idx.items():
        for di, dj in ((1, 0), (0, 1), (-1, 1)):
            b = idx.get((f, i + di, j + dj))
            if b is not None:
                edges.add((min(a, b), max(a, b)))
    jobs = []
    for a, b in sorted(edges):
        if (labs[a] == target) != (labs[b] == target):
            ins, out = (a, b) if labs[a] == target else (b, a)
            jobs.append((pts[ins].tolist(), pts[out].tolist(), target, labs[out]))
    with Pool(procs) as pool:
        bnd = pool.map(_bisect, jobs, chunksize=2)
    json.dump({"target": target, "wall": wall, "n": GRID_N, "corners": WALLS[wall], "points": [p.tolist() for p in pts],
               "labels": labs, "boundary": bnd}, open(path, "w"))


def interior_count(target, wall):
    pts, labs = grid_labels(wall)
    pts = np.array(pts, float)
    pts = pts / pts.sum(axis=1, keepdims=True)
    nn = np.asarray(NORMALS[wall], float) / np.linalg.norm(NORMALS[wall])
    sides = np.array([h for h in HALF if abs(abs(h @ nn) - np.linalg.norm(h)) > 1e-9])
    inner = np.all(pts @ sides.T > 1e-9, axis=1)
    return int(sum(1 for k, l in enumerate(labs) if l == target and inner[k]))


def fit_conic(B, normal):
    """A conic through boundary points B on a wall, as a function of beta (None if they do not fit one)."""
    B = np.asarray(B, float)
    n = np.asarray(normal, float)
    basis = np.linalg.svd(n[None])[2][1:]
    Y = B @ basis.T
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    mons = [(i, j) for i in range(3) for j in range(i, 3)]
    A = np.stack([Y[:, i] * Y[:, j] for i, j in mons], 1)
    u, s, vt = np.linalg.svd(A)
    if len(B) < 6 or s[-1] > 1e-7 or s[-2] < 1e-5:
        return None
    c = vt[-1]
    S = np.zeros((3, 3))
    for (i, j), v in zip(mons, c):
        S[i, j] += v / (1 if i == j else 2)
        S[j, i] = S[i, j]
    M = basis.T @ S @ basis
    return lambda b: float(np.asarray(b) @ M @ np.asarray(b))


def patch_general(target, wall):
    """Like patch(), with curved edges found automatically: boundary points no straight run explains, if they fit a
    conic, cut the polygon along it."""
    d = json.load(open(f"fexact_{target}_{wall}.json"))
    pts = np.array(d["points"], float)
    pts = pts / pts.sum(axis=1, keepdims=True)
    normal = NORMALS[wall]
    nn = np.asarray(normal, float) / np.linalg.norm(normal)
    sides = np.array([h for h in HALF if abs(abs(h @ nn) - np.linalg.norm(h)) > 1e-9])
    interior = np.all(pts @ sides.T > 1e-9, axis=1)
    ins = pts[[k for k, l in enumerate(d["labels"]) if l == target and interior[k]]]
    if not len(ins):
        return None, [], []
    from collections import defaultdict
    groups = defaultdict(list)
    for b in d["boundary"]:
        groups[b["outside"] if b["outside"] != "ERR" else b["far"] + "*"].append(b["beta"])
    edges, curved = [], []
    for nb, B in groups.items():
        runs, left = _lines(B, normal)
        for r in runs:
            if len(r) < 3:
                left += r
                continue
            P = np.array([B[k] for k in r])
            m, res, text = plane_through(P, normal)
            if res > 1e-6 or text == "not golden":
                left += r
                continue
            m = np.asarray(m, float) * (1 if np.mean(ins @ m) >= 0 else -1)
            if not any(np.allclose(m / np.abs(m).max(), e[0] / np.abs(e[0]).max(), atol=1e-6) for e in edges):
                edges.append((m, nb.rstrip("*"), text))
        if len(left) >= 6:
            Qf = fit_conic([B[k] for k in left], normal)
            if Qf is not None:
                curved.append((Qf, nb.rstrip("*"), [B[k] for k in left]))
    poly = [np.array(c, float) / np.sum(c) for c in WALLS[wall]]
    for m, nb, text in edges:
        poly = _clip(poly, m)
    for Qf, nb, B in curved:
        side = np.sign(np.mean([Qf(p) for p in ins]))
        if side == 0 or len(poly) < 3:
            continue
        inside_pt = ins[np.argmax([side * Qf(p) for p in ins])]
        poly = _clip_conic(poly, Qf, inside_pt)
    return [p.tolist() for p in poly], [(m.tolist(), nb, text) for m, nb, text in edges], [nb for _, nb, _ in curved]


def _reps(x):
    from cell_atlas import to_upper
    from cellframe import TINV
    from dodeca_view import E
    out = []
    for g in E:
        b = TINV @ (g @ x)
        if b.sum() > 0 and (b / b.sum()).min() >= -1e-9:
            out.append((g, to_upper(np.clip(b / b.sum(), 0, None))))
    return out


def on_patch(b, p, tol=1e-7):
    n = np.array(NORMALS[p["wall"]], float)
    if abs(b @ n) > tol * np.abs(n).sum():
        return False
    basis = np.linalg.svd(np.vstack([n, np.ones(4)]))[2][2:]
    V = np.array(p["corners"]) @ basis.T
    q = b @ basis.T
    return _in_polygon(q, V) or _dist_to_boundary(q, V) < 1e-7


def missing_walls(target, patches, samples=50, seed=0):
    """Walls holding half-turn copies of the target's patches that land on none of them."""
    from cellframe import T, TINV, seed_from_beta
    from dodeca_view import E
    from normalizer import halfturn
    Qm = halfturn()
    rng = np.random.default_rng(seed)
    found = {}
    Tm = np.array(T)
    for p in patches:
        C = np.array(p["corners"], float)
        n = np.array(NORMALS[p["wall"]], float)
        for _ in range(samples):
            b = rng.dirichlet(np.ones(len(C))) @ C
            if not on_patch(b, p):
                continue
            x = seed_from_beta(b)
            for g in E:
                M = g @ Qm
                bb = TINV @ (M @ x)
                if bb.sum() <= 0 or (bb / bb.sum()).min() < -1e-9:
                    continue
                from cell_atlas import to_upper
                r = to_upper(np.clip(bb / bb.sum(), 0, None))
                if any(on_patch(r, q) for q in patches):
                    continue
                A = TINV @ M @ Tm.T
                if bb[2] < bb[3] - 1e-12:
                    A = A[[1, 0, 3, 2]]
                nn = n @ np.linalg.inv(A)
                nn = nn / nn[np.argmax(np.abs(nn))]
                key = tuple(np.round(nn, 6))
                found.setdefault(key, []).append(r)
    return found


def shape(target, rounds=4, min_points=6, log=print):
    """Trace a wall shape on every wall it occupies, adding the walls its half-turn copies reach."""
    import glob
    import os
    load_walls()
    allp = json.load(open("fexact_patches.json")) if os.path.exists("fexact_patches.json") else {}
    walls_done = set()
    for rnd in range(rounds):
        walls = [w for w in NORMALS if w not in walls_done and (os.path.exists(f"fexact_grid_{w}.json")
                 or glob.glob("fexact_*_" + glob.escape(w) + ".json")) and interior_count(target, w) >= min_points]
        for w in walls:
            trace(target, w)
            poly, edges, curved = patch_general(target, w)
            walls_done.add(w)
            if poly is None or len(poly) < 3:
                continue
            good, bad_out, bad_in = check_patch(target, w, poly)
            allp[f"{target} {w}"] = {"target": target, "wall": w, "normal": NORMALS[w], "corners": poly,
                                     "edges": edges, "curved": curved, "check": [good, bad_out, bad_in]}
            log(f"  {target} on {w}: {len(poly)} corners, grid agreeing {good}, misplaced {bad_out}+{bad_in}"
                + (f", curved edge against {curved}" if curved else ""))
        mine = [p for p in allp.values() if p["target"] == target]
        miss = missing_walls(target, mine)
        new = [k for k in miss if len(miss[k]) >= 3]
        log(f"  round {rnd + 1}: {len(mine)} patches; copies off them on {len(new)} new wall(s)")
        if not new:
            break
        for k in new:
            w = add_wall(np.array(k))
            grid_labels(w)
            log(f"    new wall {w}: {interior_count(target, w)} interior {target} grid points")
    json.dump(allp, open("fexact_patches.json", "w"), indent=1)
    return allp


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        run_grid(sys.argv[2].split(","), sys.argv[3])
    elif sys.argv[1] == "patches":
        out = {}
        for target, wall in PATCHES:
            poly, edges = patch(target, wall)
            a, b, c = check_patch(target, wall, poly)
            out[f"{target} {wall}"] = {"target": target, "wall": wall, "normal": NORMALS[wall], "corners": poly, "edges": edges}
            print(f"{target} on {wall}: {len(poly)} corners; grid points agreeing {a}, {target} outside {b}, others inside {c}")
            for m, nb, text in edges:
                print("    edge", text, "| across:", nb)
        json.dump(out, open("fexact_patches.json", "w"), indent=1)
    elif sys.argv[1] == "shape":
        for t in sys.argv[2:]:
            print(t, flush=True)
            shape(t, log=lambda *a: print(*a, flush=True))
    elif sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], NORMALS[sys.argv[3]])
