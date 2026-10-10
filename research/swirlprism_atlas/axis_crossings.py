"""Points where an extra-symmetry axis (the purple 2400 axes and the green order-3 girdle) crosses a transitional
class: walk along each axis segment, bisect every change of shape to the crossing, and classify the crossing
point itself. A transitional class there (one that differs from the shapes on either side) is a wall, line or
point that the axis passes through. Writes axis_crossings.json.

    python axis_crossings.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool

import numpy as np
from normalizer import coset_axes
from probe import _one
from supergroups import girdle_families, girdle_segments

N = 400


def _bisect(job, steps=50):
    a, b, la = job
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        m = (lo + hi) / 2
        if _one(a + m * (b - a))[0] == la:
            lo = m
        else:
            hi = m
    t = (lo + hi) / 2
    p = a + t * (b - a)
    return p.tolist(), _one(p)[0], _one(a + lo * (b - a))[0], _one(a + hi * (b - a))[0]


def _label_at(p, tol=1e-8):
    """The class exactly on a crossing: the bisected point is within ~1e-15 of the wall, so the hull is built
    with a merge tolerance of 1e-8 (the default 1e-9 refuses such nearly degenerate hulls)."""
    from cellframe import seed_from_beta
    from classify import classify, signature
    from probe import label_of_sig
    try:
        info = classify(seed_from_beta(np.asarray(p, float)), hull_tol=tol)
    except Exception:
        return "ERR"
    return label_of_sig(signature(info) + " | val " + ",".join(f"{k}:{v}" for k, v in info["valence"].items()))


def relabel():
    """Classify the crossings found by main() properly, and the shapes a short step either side along the axis."""
    data = json.load(open("axis_crossings.json"))
    segs = {f"purple axis ({girdle_families()(a, b)}-family)": [] for a, b in coset_axes()}
    for c in data:
        p = np.array(c["beta"])
        axes = [(a, b) for a, b in coset_axes()] + [(a, b) for a, b in girdle_segments()]
        a, b = min(axes, key=lambda ab: np.linalg.norm(np.cross(np.r_[ab[1] - ab[0]][:3], (p - ab[0])[:3])))
        d = (b - a) / np.linalg.norm(b - a)
        c["on"] = _label_at(p)
        c["below"] = _one(p - 1e-5 * d)[0] if (p - 1e-5 * d).min() >= 0 else "outside"
        c["above"] = _one(p + 1e-5 * d)[0] if (p + 1e-5 * d).min() >= 0 else "outside"
        flag = "TRANSITIONAL" if c["on"] not in (c["below"], c["above"]) else ""
        print(f"{c['axis']:32s} {c['below']:7s} | {c['on']:8s} | {c['above']:7s} at β {np.round(p, 6).tolist()} {flag}")
    del segs
    json.dump(data, open("axis_crossings.json", "w"), indent=1)


def main():
    fam = girdle_families()
    segs = [(f"purple axis ({fam(a, b)}-family)", a, b) for a, b in coset_axes()]
    segs += [("green order-3 girdle", a, b) for a, b in girdle_segments()]
    jobs, meta = [], []
    for k, (name, a, b) in enumerate(segs):
        for i in range(N + 1):
            jobs.append(a + (b - a) * i / N)
            meta.append((k, i))
    with Pool(4) as pool:
        labels = [x[0] for x in pool.map(_one, jobs, chunksize=8)]
        changes = []
        for j in range(len(jobs) - 1):
            if meta[j][0] == meta[j + 1][0] and labels[j] != labels[j + 1]:
                changes.append((meta[j][0], jobs[j], jobs[j + 1], labels[j]))
        res = pool.map(_bisect, [(p, q, la) for _, p, q, la in changes])
    out = []
    for (k, _, _, _), (p, on, below, above) in zip(changes, res):
        name = segs[k][0]
        out.append({"axis": name, "beta": p, "on": on, "below": below, "above": above})
        flag = "transitional" if on not in (below, above) else ""
        print(f"{name:32s} {below:6s} | {on:8s} | {above:6s} at β {np.round(p, 6).tolist()} {flag}", flush=True)
    json.dump(out, open("axis_crossings.json", "w"), indent=1)


if __name__ == "__main__":
    import sys
    relabel() if sys.argv[1:] == ["relabel"] else main()
