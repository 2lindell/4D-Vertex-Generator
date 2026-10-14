"""Trace a seam on a curved wall: the curve on the wall's surface (xsurface.py equation) between the wall and the shape
past its edge. From a point near the seam it walks both ways along it, re-finding the seam at each step by bisecting
across it on the surface, until the shape on either side changes; that end is bisected along the walk. The polyline is
saved in xcurves.json for drawing.

    python seam_trace.py X84 X51 X74 b1,b2,b3,b4
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from fexact import _lab, _lab_loose
from xsurface import surface, project, grad

U = np.linalg.svd(np.ones((1, 4)))[2][1:]


def main(sid, wall, beyond, start, step=0.004, max_len=1.5, out="xcurves.json"):
    sf = surface(wall)

    def lab(x):
        if x.min() < -1e-12 or x[2] < x[3] - 1e-12:
            return "OUT"
        return _lab(x)

    def cross(x, d, w):
        """The seam point on the chord through x along d (on the surface), or None if the chord does not cross it."""
        a, b = project(sf, x - w * d), project(sf, x + w * d)
        la, lb = lab(a), lab(b)
        if {la, lb} != {wall, beyond}:
            return None
        if lb == wall:
            a, b = b, a
        for _ in range(34):
            m = project(sf, (a + b) / 2)
            if lab(m) == wall:
                a = m
            else:
                b = m
        return project(sf, (a + b) / 2)

    def across(x, t):
        n = grad(sf, x)
        n /= np.linalg.norm(n)
        d = U.T @ np.cross(U @ t, U @ n)
        return d / np.linalg.norm(d)

    x0 = np.asarray(start, float)
    x0 /= x0.sum()
    # find the seam near the start and its direction
    rng = np.random.default_rng(1)
    n = grad(sf, x0)
    n /= np.linalg.norm(n)
    p0 = None
    for _ in range(40):
        d = U.T @ rng.normal(size=3)
        d -= (d @ n) * n
        d /= np.linalg.norm(d)
        p0 = cross(x0, d, 3e-3)
        if p0 is not None:
            break
    if p0 is None:
        print(f"{sid}: no seam found near the start")
        return
    t0 = U.T @ np.cross(U @ d, U @ n)
    t0 /= np.linalg.norm(t0)
    runs = []
    for sgn in (1, -1):
        x, t, run, travelled, end = p0, sgn * t0, [p0], 0.0, None
        while travelled < max_len:
            y = project(sf, x + step * t)
            d = across(y, t)
            z = None
            for w in (step, 3 * step):
                z = cross(y, d, w)
                if z is not None:
                    break
            if z is None:                              # the seam stops: bisect the last step for its end
                lo, hi = 0.0, step
                for _ in range(20):
                    mid = (lo + hi) / 2
                    if cross(project(sf, x + mid * t), across(x, t), mid + 1e-4) is not None:
                        lo = mid
                    else:
                        hi = mid
                e = project(sf, x + lo * t)
                e = cross(e, across(e, t), 1e-4) if cross(e, across(e, t), 1e-4) is not None else e
                run.append(e)
                end = {"end": e.tolist(), "at": _lab_loose(e)}
                break
            tn = z - x
            t = tn / np.linalg.norm(tn)
            run.append(z)
            x = z
            travelled += step
        runs.append((run, end))
        print(f"{sid}: {len(run)} points one way; end {None if end is None else np.round(np.array(end['end']) / max(end['end']), 5).tolist()} "
              f"({None if end is None else end['at'][:30]})", flush=True)
    pts = runs[0][0][::-1] + runs[1][0][1:]
    data = json.load(open(out)) if os.path.exists(out) else {}
    data[sid] = {"surfaces": [wall], "equations": [sf["equation"]], "seam": [wall, beyond],
                 "pts": [p.tolist() for p in pts], "ends": [runs[0][1], runs[1][1]]}
    json.dump(data, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [float(v) for v in sys.argv[4].split(",")])
