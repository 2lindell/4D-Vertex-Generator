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


def main(lines):
    d = None
    old = json.load(open("ring_test.json")) if __import__("os").path.exists("ring_test.json") else {}
    res = dict(old.get("rings", {}))                          # (earlier lines kept; the ones run now replaced)
    if not lines:
        lines = sorted(json.load(open("exact_defs.json"))["lines"])
    with Pool(3) as pool:
        for L in lines:
            jobs = []
            for P in line_points(d, L):                                # (six points along each piece, by length)
                s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                for t in np.linspace(0.08, 0.92, 6) * s_[-1]:
                    i = min(np.searchsorted(s_, t), len(P) - 1)
                    i = max(i, 1)
                    f = (t - s_[i - 1]) / max(s_[i] - s_[i - 1], 1e-300)
                    jobs.append((nrm(P[i - 1] + f * (P[i] - P[i - 1])), P[i] - P[i - 1]))
            R = pool.map(ring, jobs)
            json.dump({"rings": {**res, L: [{"at": np.round(j[0], 4).tolist(), "on": short(r[1]),
                                             "ring": [short(x) for x in r[0]]} for j, r in zip(jobs, R)]},
                       "new shapes": {v: k for k, v in NEW.items()}}, open("ring_test.json", "w"), indent=1)
            res[L] = [{"at": np.round(j[0], 4).tolist(), "on": short(r[1]), "ring": [short(x) for x in r[0]]}
                      for j, r in zip(jobs, R)]
            print(L)
            for x in res[L]:
                print(f"   {x['at']} on {x['on']}: {' '.join(x['ring'])}")
    json.dump({"rings": res, "new shapes": {v: k for k, v in NEW.items()}}, open("ring_test.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
