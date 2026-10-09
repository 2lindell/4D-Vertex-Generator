"""Further lines of the line shapes, from their stray samples (samples on none of the shape's drawn pieces, under any
image of the 2400-element group).

For each stray, candidate lines are the straight lines through it and at least two further images of the shape's
samples (collinear triples), or, failing that, the directions from it (toward images of the shape's samples, the cell's
corners and centre) along which the shape goes on just past it. Each candidate line is labelled across the half
tetrahedron; the runs of the shape on it have their ends bisected and snapped to golden betas, with the shapes found
just past each end.

    python stray_lines.py strays_all.json line_pieces.json stray_lines.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

from cellframe import T, TINV, golden_form
from fexact import _lab_loose
from normalizer import extended_group

PHI = (1 + 5 ** 0.5) / 2
M = np.einsum("ij,gjk,kl->gil", TINV, np.array(extended_group(), float), np.array(T, float).T)
FOLD = [1, 0, 3, 2]
HALF = [np.eye(4)[k] for k in range(4)] + [np.array([0, 0, 1.0, -1.0])]


def in_half(x):
    return all(h @ x > -1e-12 for h in HALF)


def lab(x):
    x = np.asarray(x, float)
    return _lab_loose(x / x.sum()) if in_half(x) else "OUT"


def images(b):
    out = []
    for v in M @ b:
        if v.sum() <= 0:
            continue
        v = v / v.sum()
        if v.min() < -1e-9:
            continue
        v = np.clip(v, 0, None)
        v = v if v[2] >= v[3] - 1e-12 else v[FOLD]
        if not any(np.abs(v - w).max() < 1e-9 for w in out):
            out.append(v)
    return out


def segment(a, b):
    """The line through a and b inside the half tetrahedron, as its two ends."""
    ends = []
    for h in HALF:
        x = (h @ b) * a - (h @ a) * b
        if np.abs(x).max() < 1e-14 or abs(x.sum()) < 1e-14:
            continue
        x = x if x.sum() > 0 else -x
        x = x / x.sum()
        if in_half(x) and not any(np.abs(x - e).max() < 1e-9 for e in ends):
            ends.append(x)
    if len(ends) < 2:
        return None
    return max(((p, q) for p in ends for q in ends), key=lambda pq: np.linalg.norm(pq[0] - pq[1]))


def snap(x):
    """Golden text for beta x (scaled so one coordinate is 1), or rounded numbers."""
    for k in range(4):
        if x[k] < 1e-9:
            continue
        r = x / x[k]
        forms = []
        for v in r:
            got = None
            for den in (1, 2, 3, 4, 5, 10):
                for bb in range(-30, 31):
                    aa = round((v - bb * PHI) * den)
                    if abs((aa + bb * PHI) / den - v) < 1e-7:
                        got = ((aa + bb * PHI) / den, golden_form((aa + bb * PHI) / den))
                        break
                if got:
                    break
            forms.append(got)
        if all(forms):
            return [f[0] for f in forms], "(" + ", ".join(f[1] for f in forms) + ")"
    return None, str(np.round(x / x.max(), 6).tolist())


def _probe(args):
    q, d = args
    return lab(q + 1e-4 * d)


def _label(x):
    return lab(x)


def trace(pool, tid, a, b, n=49):
    seg = segment(a, b)
    if seg is None:
        return None
    A, B = seg
    ts = np.linspace(0, 1, n)
    labs = pool.map(_label, [A + t * (B - A) for t in ts])
    runs = []
    for t, l in zip(ts, labs):
        if runs and runs[-1][2] == l:
            runs[-1][1] = t
        else:
            runs.append([t, t, l])
    pieces = []
    for k, (t0, t1, l) in enumerate(runs):
        if l != tid:
            continue
        ends = []
        for side, (tin, tout) in ((0, (t0, runs[k - 1][1] if k else None)), (1, (t1, runs[k + 1][0] if k + 1 < len(runs) else None))):
            if tout is None:                            # the run reaches the end of the segment (a face)
                x = A + tin * (B - A)
                ends.append({"t": tin, "beta": x.tolist(), "beyond": "face"})
                continue
            lo, hi = tin, tout
            for _ in range(30):
                m = (lo + hi) / 2
                if lab(A + m * (B - A)) == tid:
                    lo = m
                else:
                    hi = m
            x = A + (lo + hi) / 2 * (B - A)
            ends.append({"t": (lo + hi) / 2, "beta": x.tolist(),
                         "at": lab(x), "beyond": lab(A + min(1, hi + 1e-6) * (B - A)) if side else
                         lab(A + max(0, lo - 1e-6) * (B - A))})
        for e in ends:
            g, txt = snap(np.asarray(e["beta"]))
            e["text"] = txt
            if g is not None:
                e["beta"] = (np.array(g) / sum(g)).tolist()
        pieces.append(ends)
    return {"seg": [A.tolist(), B.tolist()], "runs": [(round(r[0], 4), round(r[1], 4), r[2][:20]) for r in runs],
            "pieces": pieces}


def same_line(P1, P2):
    return np.abs(P1 - P2).max() < 1e-7


def proj(a, b):
    U = np.linalg.qr(np.vstack([a, b]).T)[0]
    return U @ U.T


def main(strays_path, pieces_path, out_path):
    S = json.load(open(strays_path))
    D = json.load(open(pieces_path))
    out = {}
    extra = [np.eye(4)[k] for k in range(4)] + [np.ones(4) / 4] + \
            [np.array(v, float) / sum(v) for v in ([1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1])]
    with Pool(4) as pool:
        for tid, found in S.items():
            if not found:
                continue
            samp = [np.array(b) / sum(b) for b in D[tid]["samples"]]
            imgs = []
            for b in samp:
                for v in images(b):
                    if not any(np.abs(v - w).max() < 1e-9 for w in imgs):
                        imgs.append(v)
            cands = []
            for f in found:
                q = np.array(f["stray"])
                if any(np.linalg.norm(q - P @ q) < 1e-9 for P, _ in cands):
                    continue                            # (already on a candidate line)
                got = [np.array(ln[1][0]) for ln in f["lines"] if ln[0] >= 2]
                if not got:                             # probe the directions from the stray
                    dirs = []
                    for c in imgs + extra:
                        d = c - q
                        if np.linalg.norm(d) < 1e-9:
                            continue
                        d = d / np.linalg.norm(d)
                        for s in (1, -1):
                            if not any(np.abs(s * d - e).max() < 1e-6 for e in dirs):
                                dirs.append(s * d)
                    labs = pool.map(_probe, [(q, d) for d in dirs])
                    got = [q + d for d, l in zip(dirs, labs) if l == tid]
                    print(f"{tid}: stray {np.round(q, 4).tolist()} probed {len(dirs)} directions, {len(got)} along the shape",
                          flush=True)
                for g in got:
                    P = proj(q, g)
                    if not any(same_line(P, P2) for P2, _ in cands):
                        cands.append((P, (q, g)))
            res = []
            for P, (q, g) in cands:
                r = trace(pool, tid, q, g)
                if r is None:
                    continue
                covered = [f["stray"] for f in found if np.linalg.norm(np.array(f["stray"]) - P @ np.array(f["stray"])) < 1e-9]
                r["strays"] = covered
                res.append(r)
                print(f"{tid}: line through {snap(q)[1]} — runs {r['runs']}", flush=True)
                for pc in r["pieces"]:
                    print(f"    piece {pc[0]['text']} ({pc[0].get('beyond')}) to {pc[1]['text']} ({pc[1].get('beyond')})", flush=True)
            out[tid] = res
            json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
