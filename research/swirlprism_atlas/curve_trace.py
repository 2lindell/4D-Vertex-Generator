"""Trace the curve where two exact surfaces meet (xsurface.py equations), from a point on it, both ways, classifying as it
goes; the run of the target shape is cut at its ends (bisected along the curve) and saved for drawing.

    python curve_trace.py X78 X15 X83 b1,b2,b3,b4 out.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from fexact import _lab, _lab_loose
from xsurface import surface, F, grad


def main(tid, sa, sb, start, out, step=0.004, max_len=1.5):
    A, B = surface(sa), surface(sb)

    def onto(x):
        for _ in range(60):
            r = np.array([F(A, x)[0], F(B, x)[0]])
            J = np.vstack([grad(A, x), grad(B, x), np.ones(4)])
            dx = np.linalg.lstsq(J, np.concatenate([-r, [0]]), rcond=None)[0]
            x = x + dx
            if np.abs(dx).max() < 1e-17:
                break
        return x / x.sum()

    def tangent(x):
        t = np.linalg.svd(np.vstack([grad(A, x), grad(B, x), np.ones(4)]))[2][-1]
        return t / np.linalg.norm(t)

    def lab(x):
        if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
            return "OUT"
        return _lab_loose(x)

    x0 = onto(np.asarray(start, float))
    sides = []
    for sgn in (1, -1):
        x, t, run = x0, sgn * tangent(x0), [x0]
        travelled = 0.0
        while travelled < max_len:
            y = onto(x + step * t)
            tn = tangent(y)
            t = tn if tn @ t > 0 else -tn
            l = lab(y)
            if l != tid:                               # bisect along the curve for the end
                lo, hi = x, y
                for _ in range(30):
                    m = onto((lo + hi) / 2)
                    if lab(m) == tid:
                        lo = m
                    else:
                        hi = m
                run.append(lo)
                sides.append({"end": lo.tolist(), "beyond": l, "at": _lab_loose(lo)})
                print(f"{tid}: end after {travelled:.3f}; beyond {l[:30]}", flush=True)
                break
            run.append(y)
            x = y
            travelled += step
        else:
            sides.append({"end": x.tolist(), "beyond": "unfinished"})
        if sgn == 1:
            first = run[::-1]
        else:
            pts = first + run[1:]
    data = json.load(open(out)) if out and __import__("os").path.exists(out) else {}
    data[tid] = {"surfaces": [sa, sb], "equations": [A["equation"], B["equation"]], "pts": [p.tolist() for p in pts],
                 "ends": sides}
    json.dump(data, open(out, "w"), indent=1)
    print(f"{tid}: {len(pts)} points", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4].split(",")], sys.argv[5])
