"""Ring test around a line: at points along the line, a small circle around it (in the plane normal to it, within the
half-cell's 3-space) is classified at evenly spaced angles; the cyclic order of what is met (regions, with the walls
between them) says which regions meet at the line and which walls separate them there.

    python ring_test.py X78 C2a X5 E1        (prints each line's rings; ring_test.json)
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

import numpy as np

import optimal_seeds as osd
import region_walls as rw
from edge_relabel import lab_edge
from probe import _one

NEW = {}


def short(x):
    """A new shape's signature, shortened to a stable tag (new#1, new#2, ...; the signatures are in ring_test.json)."""
    if x.startswith("[new") or x.startswith("new"):
        k = x.strip("[]")
        NEW.setdefault(k, f"new#{len(NEW) + 1}")
        return NEW[k] if not x.startswith("[") else f"[{NEW[k]}]"
    return x

R_ = 1.5e-3
K = 36


def nrm(b):
    b = np.asarray(b, float)
    return b / b.sum(-1, keepdims=True)


def line_points(d, L):
    """The line's verified pieces, exact (exact_defs.json, brought to their ends by line_ends.py)."""
    ex = json.load(open("exact_defs.json"))["lines"]
    return [np.asarray(P, float) for P in ex.get(L, {}).get("points", []) if len(P) >= 2]


def full_home(b):
    """A point anywhere on the 3-sphere (betas, possibly negative: past a face of the cell) brought into the half-cell
    by the extended group."""
    from edge_curves import group
    for g in group():
        y = g @ b
        s = y.sum()
        if abs(s) > 1e-12:
            y = y / s
            if (y > -1e-12).all() and y[2] >= y[3] - 1e-12:
                return y
    return None


def lab(b):
    b = np.asarray(b, float)
    if (b < -1e-12).any():                               # (past a face of the cell: the neighbouring cell's point)
        h = full_home(b)
        return _one(h)[0] if h is not None else "OUT"
    h = osd.home(nrm(b))
    return _one(h)[0] if np.isfinite(h).all() else "ERR"


def ring(args):
    p, t = args
    t = t - t.mean()
    t /= np.linalg.norm(t)
    B = np.linalg.svd(np.vstack([np.ones(4), t]))[2][2:]          # (directions with sum 0, normal to the line)
    A = np.linspace(0, 2 * np.pi, K, endpoint=False)
    pt = lambda a: p + R_ * (np.cos(a) * B[0] + np.sin(a) * B[1])
    seq = [lab(pt(a)) for a in A]
    out = []
    for i in range(K):                                               # (sectors, cyclically, with the wall between)
        j = (i + 1) % K
        if seq[i] == seq[j] or "OUT" in (seq[i], seq[j]):
            if seq[i] != seq[j]:
                out += [seq[j]]
            continue
        lo, hi = A[i], A[i] + 2 * np.pi / K                          # (the wall between: bisected, then named with
        for _ in range(30):                                           # the hull tolerance relaxed)
            m = (lo + hi) / 2
            if lab(pt(m)) == seq[i]:
                lo = m
            else:
                hi = m
        w = lab_edge(osd.home(nrm(pt((lo + hi) / 2))))
        out += [f"[{w}]", seq[j]]
    k = next((i for i, x in enumerate(out) if not x.startswith("[")), 0)
    out = out[k:] + out[:k]
    return out, lab_edge(osd.home(nrm(p)))


def events_on(P):
    """The exact points along a line piece where the walls round it can change: the corners of every wall's edges and
    cuts (wall_equations.json), the named points and the lines' ends (line_ends.json, cross_ring_points.json): each as
    its arclength along the piece, with what it is."""
    import os
    E = json.load(open("wall_equations.json"))["walls"]
    cand = []
    for w, rec in E.items():
        for kind in ("edges", "cuts"):
            for Y, r in rec[kind].items():
                for rr in [r] + r.get("more", []):
                    for c in rr.get("corners", []):
                        cand.append((f"{w} {'cut by' if kind == 'cuts' else 'meets'} {Y}", np.asarray(c, float)))
    for p in json.load(open("cross_ring_points.json"))["points"]:
        cand.append((p["name"], np.asarray(p["beta"], float)))
    if os.path.exists("line_ends.json"):
        for L, rs in json.load(open("line_ends.json"))["lines"].items():
            for r in rs:
                if r["type"] == "ends" and (r.get("exact") or r.get("approx")):
                    cand.append((f"{L} ends on {r['on']}", np.asarray(r.get("exact") or r.get("approx"), float)))
    seg = np.diff(P, axis=0)
    s_ = np.r_[0, np.cumsum(np.linalg.norm(seg, axis=1))]
    out = []
    for name, x in cand:
        a, d = P[:-1], seg
        t = np.clip(((x - a) * d).sum(1) / ((d * d).sum(1) + 1e-300), 0, 1)
        dist = np.linalg.norm(a + t[:, None] * d - x, axis=1)
        k = int(np.argmin(dist))
        if dist[k] < 2e-5:
            out.append((float(s_[k] + t[k] * np.linalg.norm(d[k])), name))
    return s_, sorted(out)


def main(lines):
    """Each line, split at the exact points where the walls round it can change; the ring taken twice inside each
    stretch (at a third and two thirds): the two agreeing say the stretch is one piece of each wall round it."""
    import os
    old = json.load(open("ring_test.json")) if os.path.exists("ring_test.json") else {}
    res = dict(old.get("rings", {}))
    if not lines:
        lines = sorted(json.load(open("exact_defs.json"))["lines"])
    with Pool(3) as pool:
        for L in lines:
            jobs, meta = [], []
            for k, P in enumerate(line_points(None, L)):
                s_, ev = events_on(P)
                cuts = sorted({0.0, float(s_[-1])} | {e for e, _ in ev if 1e-4 < e < s_[-1] - 1e-4})
                merged = [cuts[0]]
                for c in cuts[1:]:
                    if c - merged[-1] > 1e-4:
                        merged.append(c)
                for a_, b_ in zip(merged[:-1], merged[1:]):
                    if b_ - a_ < 4e-3:                                 # (too short to take a ring inside)
                        continue
                    for f in (1 / 3, 2 / 3):
                        t = a_ + f * (b_ - a_)
                        i = max(1, min(np.searchsorted(s_, t), len(P) - 1))
                        g = (t - s_[i - 1]) / max(s_[i] - s_[i - 1], 1e-300)
                        jobs.append((nrm(P[i - 1] + g * (P[i] - P[i - 1])), P[i] - P[i - 1]))
                        meta.append((k, a_, b_, sorted({n for e, n in ev if abs(e - a_) < 1e-4}),
                                     sorted({n for e, n in ev if abs(e - b_) < 1e-4})))
            R = pool.map(ring, jobs)
            out = []
            for idx in range(0, len(R), 2):
                (k, a_, b_, at_a, at_b), r1, r2 = meta[idx], R[idx], R[idx + 1]
                out.append({"piece": k, "from": round(a_, 6), "to": round(b_, 6), "length": round(b_ - a_, 6),
                            "starts_at": at_a, "ends_at": at_b, "at": np.round(jobs[idx][0], 5).tolist(),
                            "on": short(r1[1]), "ring": [short(x) for x in r1[0]],
                            "agree": [short(x) for x in r1[0]] == [short(x) for x in r2[0]],
                            "ring2": [short(x) for x in r2[0]]})
            res[L] = out
            json.dump({"rings": res, "new shapes": {v: k for k, v in NEW.items()}}, open("ring_test.json", "w"),
                      indent=1)
            print(L, flush=True)
            for x in out:
                print(f"   piece {x['piece']} {x['from']:.4f}-{x['to']:.4f} {'' if x['agree'] else 'DISAGREE '}"
                      f"{' '.join(x['ring'])}  | from {x['starts_at']} to {x['ends_at']}", flush=True)

if __name__ == "__main__":
    main(sys.argv[1:])
