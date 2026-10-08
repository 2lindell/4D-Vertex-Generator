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
    """The second plane (normal m, m . beta = 0) holding all points B of a great circle in the wall, snapped to
    golden coefficients; returns (m, residual, text)."""
    B = np.asarray(B, float)
    w = np.asarray(wall_normal, float) / np.linalg.norm(wall_normal)
    P = B - np.outer(B @ w, w)                         # remove the wall direction
    u, s, vt = np.linalg.svd(np.vstack([P, w]))
    m = vt[-1]
    m = m - (m @ w) * w
    m /= np.abs(m).max()
    m *= np.sign(m[np.argmax(np.abs(m))])
    res = float(np.abs(B @ m).max())
    k = np.argmax(np.abs(m))
    snapped = []
    for v in m / m[k]:
        f = golden_form(v)
        snapped.append(f)
    if all(f is not None for f in snapped):
        mm = np.array([(a + b * PHI) / q for a, b, q in snapped])
        terms = [f"{fmt_golden(f)}·β{i + 1}" for i, f in enumerate(snapped) if f[0] or f[1]]
        return mm, float(np.abs(B @ mm).max()), " + ".join(terms).replace("+ -", "− ") + " = 0"
    return m, res, "not golden"


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
           ("F4", "b1=phi-2*b2"), ("F5", "b1=phi-2*b2"), ("F5", "b2=b4")]


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
    ins = pts[[k for k, l in enumerate(d["labels"]) if l == target]]
    normal = NORMALS[wall]
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


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        run_grid(sys.argv[2].split(","), sys.argv[3])
    elif sys.argv[1] == "patches":
        out = {}
        for target, wall in PATCHES:
            poly, edges = patch(target, wall)
            a, b, c = check_patch(target, wall, poly)
            out[f"{target} {wall}"] = {"target": target, "wall": wall, "corners": poly, "edges": edges}
            print(f"{target} on {wall}: {len(poly)} corners; grid points agreeing {a}, {target} outside {b}, others inside {c}")
            for m, nb, text in edges:
                print("    edge", text, "| across:", nb)
        json.dump(out, open("fexact_patches.json", "w"), indent=1)
    elif sys.argv[1] == "fit":
        fit(sys.argv[2], sys.argv[3], NORMALS[sys.argv[3]])
