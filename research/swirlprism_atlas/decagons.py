"""Where the polytope has regular decagon faces.

A decagon face here is two regular pentagons merged into one plane: the orbits of a seed image under a simple
rotation R of order 5 (it turns one plane by 72 degrees and fixes the plane orthogonal to it), for two seed images
x and k.x. The two pentagons share a plane exactly when x and k.x have the same component in R's fixed plane, and
the decagon is regular exactly when, in R's turning plane, k.x is x turned by 36 degrees (the two pentagons then
interleave evenly). Both conditions are linear in the seed, so each solution set is a great circle: a straight
line in the cell chart. They are found for every R and k in the group, kept where the decagon really is a face of
the hull, and written to regular_decagons.json.

    python decagons.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
from cellframe import TINV
from dodeca_view import E

ROT36 = np.array([[np.cos(np.pi / 5), -np.sin(np.pi / 5)], [np.sin(np.pi / 5), np.cos(np.pi / 5)]])


def simple_fives():
    """(R, fixed-plane basis F, turning-plane basis W oriented so R turns it by +72 degrees)."""
    out = []
    for R in E:
        if not np.allclose(np.linalg.matrix_power(R, 5), np.eye(4), atol=1e-9) or np.allclose(R, np.eye(4)):
            continue
        u, sv, vt = np.linalg.svd(R - np.eye(4))
        if np.sum(sv > 1e-9) != 2:
            continue
        F = vt[2:]                                                         # rows: orthonormal fixed plane
        Wp = vt[:2]                                                        # the plane it turns
        a = Wp[0]
        b = Wp @ (R @ a)
        if abs(np.degrees(np.arctan2(b[1], b[0])) - 72) > 1e-6:
            if abs(np.degrees(np.arctan2(b[1], b[0])) + 72) < 1e-6:
                Wp = Wp[[1, 0]]
            else:
                continue                                                   # turns by 144 degrees: skip, its square turns by 72
        out.append((R, F, Wp))
    return out


def circles():
    """Great circles (orthonormal 2-plane bases) of seeds giving a regular decagon, with (R index, k index)."""
    found = []
    for ri, (R, F, W) in enumerate(simple_fives()):
        powers = [np.linalg.matrix_power(R, j) for j in range(5)]
        for ki, k in enumerate(E):
            if any(np.allclose(k, p) for p in powers):
                continue
            M = np.vstack([F @ (k - np.eye(4)), W @ k - ROT36 @ W])
            s = np.linalg.svd(M)
            null = s[2][np.sum(s[1] > 1e-9):]
            if len(null) >= 2:
                found.append((ri, ki, null[:2]))
    return found


H_CELL = [TINV[i] for i in range(4)] + [TINV[2] - TINV[3]]      # beta_i >= 0 and beta3 >= beta4: the half-cell


def pieces_in_half(cs, n=1440):
    """The circles' pieces inside the half-cell (seed arrays), each piece once."""
    from dodeca_view import clip_images
    out, keys = [], set()
    for _, _, N in cs:
        th = np.linspace(0, 2 * np.pi, n + 1)
        X = np.outer(np.cos(th), N[0]) + np.outer(np.sin(th), N[1])
        for pc in clip_images(X, H_CELL, elements=np.eye(4)[None]):
            B = pc @ TINV.T
            B = B / B.sum(axis=1, keepdims=True)
            if np.linalg.norm(B[-1] - B[0]) < 1e-7:
                continue
            key = frozenset([tuple(np.round(B[0], 6)), tuple(np.round(B[-1], 6))])
            if key not in keys:
                keys.add(key)
                out.append(pc)
    return out


def regular_decagon(beta, tol=1e-7):
    """Does the hull at beta have a regular decagon face?"""
    from regular_cells import hull
    try:
        v, faces, _ = hull(beta)
    except Exception:
        return False
    for f in faces:
        if len(f) != 10:
            continue
        P = v[f]
        c = P.mean(axis=0)
        U = np.linalg.svd(P - c)[2][:2]
        Z = (P - c) @ U.T
        Z = Z[np.argsort(np.arctan2(Z[:, 1], Z[:, 0]))]
        r = np.linalg.norm(Z, axis=1)
        e = np.linalg.norm(Z - np.roll(Z, 1, axis=0), axis=1)
        if np.ptp(r) < tol * r.mean() and np.ptp(e) < tol * e.mean():
            return True
    return False


def _beta(x):
    b = TINV @ x
    return b / b.sum()


def _job(x):
    return regular_decagon(_beta(x))


def trace(pieces, samples=17, steps=30, procs=4):
    """Along each piece, the stretches where the regular decagon really is a face, ends bisected."""
    def at(pc, t):                               # t in [0, 1] along the piece's chords
        s = t * (len(pc) - 1)
        i = min(int(s), len(pc) - 2)
        x = pc[i] + (s - i) * (pc[i + 1] - pc[i])
        return x / np.linalg.norm(x)
    out = []
    with Pool(procs) as pool:
        for pc in pieces:
            ts = np.linspace(0, 1, samples)
            ok = pool.map(_job, [at(pc, t) for t in ts])
            print("piece", np.round(_beta(pc[0]), 4).tolist(), "->", np.round(_beta(pc[-1]), 4).tolist(),
                  "".join("#" if o else "." for o in ok), flush=True)
            k = 0
            while k < samples:
                if not ok[k]:
                    k += 1
                    continue
                k0 = k
                while k + 1 < samples and ok[k + 1]:
                    k += 1
                lo, hi = ts[k0], ts[k]
                ends = []
                for inner, outer in ((lo, ts[k0 - 1] if k0 > 0 else None), (hi, ts[k + 1] if k + 1 < samples else None)):
                    if outer is None:
                        ends.append(inner)
                        continue
                    a, b = inner, outer
                    for _ in range(steps):
                        m = (a + b) / 2
                        if _job(at(pc, m)):
                            a = m
                        else:
                            b = m
                    ends.append(a)
                n = 24
                pts = [at(pc, ends[0] + (ends[1] - ends[0]) * j / n) for j in range(n + 1)]
                out.append({"seeds": [p.tolist() for p in pts], "a": _beta(pts[0]).tolist(), "b": _beta(pts[-1]).tolist()})
                k += 1
    return out


def tidy(lines):
    """Drop stretches lying on another one (a piece split where the clipping met a corner), and mark copies:
    the extra half-turn Q (not in the group) swaps decagon lines in pairs, so as for the prism lines, the
    stretch through M34 = beta (0, 0, 1, 1) is drawn solid and its Q-images dashed."""
    from cell_atlas import to_upper
    from normalizer import halfturn
    Q = halfturn()
    B = [np.array([_beta(np.array(x)) for x in s["seeds"]]) for s in lines]

    def dist(b, Bj):
        return min(np.linalg.norm(a + np.clip((b - a) @ (c - a) / ((c - a) @ (c - a)), 0, 1) * (c - a) - b)
                   for a, c in zip(Bj[:-1], Bj[1:]))
    keep = [i for i in range(len(lines))
            if not any(j != i and len(B[j]) and all(dist(b, B[j]) < 1e-7 for b in B[i])
                       and not (j > i and all(dist(b, B[i]) < 1e-7 for b in B[j])) for j in range(len(lines)))]
    lines = [lines[i] for i in keep]
    B = [B[i] for i in keep]

    def reps(x):
        out = []
        for g in E:
            b = TINV @ (g @ x)
            if b.sum() > 0 and (b / b.sum()).min() >= -1e-9:
                out.append(to_upper(np.clip(b / b.sum(), 0, None)))
        return out
    m34 = np.array([0, 0, 0.5, 0.5])
    main = {i for i, Bi in enumerate(B) if min(np.linalg.norm(Bi[0] - m34), np.linalg.norm(Bi[-1] - m34)) < 1e-6}
    for i, s in enumerate(lines):
        mid = np.array(s["seeds"][len(s["seeds"]) // 2])
        targets = {j for r in reps(Q @ mid) for j, Bj in enumerate(B) if dist(r, Bj) < 1e-7}
        s["copy"] = i not in main and bool(targets & main)
    return lines


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["tidy"]:
        lines = tidy(json.load(open("regular_decagons.json")))
        json.dump(lines, open("regular_decagons.json", "w"), indent=1)
        for s in lines:
            print(np.round(s["a"], 4).tolist(), "->", np.round(s["b"], 4).tolist(), "copy" if s["copy"] else "main")
        sys.exit()
    fives = simple_fives()
    print(len(fives), "simple rotations of order 5")
    cs = circles()
    print(len(cs), "(R, k) pairs with a circle of solutions")
    pcs = pieces_in_half(cs)
    print(len(pcs), "pieces in the half-cell", flush=True)
    lines = tidy(trace(pcs))
    json.dump(lines, open("regular_decagons.json", "w"), indent=1)
    print(len(lines), "stretches with regular decagon faces")
