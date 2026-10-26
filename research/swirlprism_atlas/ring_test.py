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
    Ci = np.asarray(d["cellInv"], float)
    out = []
    for src in ("xlines", "rings", "segments", "main"):
        for x in d.get(src, []):
            if isinstance(x, dict) and x.get("id") == L and x.get("pts") and not x.get("outline"):
                V = np.array(x["pts"], float) * [1, 1, -1.0]
                out.append(nrm(np.c_[V, np.ones(len(V))] @ Ci.T))
    return out


def lab(b):
    if (b < -1e-12).any():
        return "OUT"
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
    d = rw.page_data()
    res = {}
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
            res[L] = [{"at": np.round(j[0], 4).tolist(), "on": short(r[1]), "ring": [short(x) for x in r[0]]}
                      for j, r in zip(jobs, R)]
            print(L)
            for x in res[L]:
                print(f"   {x['at']} on {x['on']}: {' '.join(x['ring'])}")
    json.dump({"rings": res, "new shapes": {v: k for k, v in NEW.items()}}, open("ring_test.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
