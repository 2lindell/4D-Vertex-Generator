"""The page's wall copies (cell_atlas2_template.html, wallCopies), made the same way in Python: each curved wall's
mesh mapped by the group elements listed for it (copyOps), clipped to the cell, the part below the splitting mirror
folded up, and an image lying on the wall or on a copy already made skipped. Returned as triangles in betas."""
from __future__ import annotations

import numpy as np


def _clip(poly, hp):
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        fp, fq = hp @ p, hp @ q
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp / (fp - fq)
            r = p + (q - p) * t
            out.append(r / r.sum())
    return out


def _into_half(poly):
    for a in range(4):
        poly = _clip(poly, np.eye(4)[a])
        if len(poly) < 3:
            return []
    up = _clip(poly, np.array([0, 0, 1, -1.0]))
    down = [b[[1, 0, 3, 2]] for b in _clip(poly, np.array([0, 0, -1, 1.0]))]
    return [pc for pc in (up, down) if len(pc) >= 3]


def to_beta(q, cell_inv):
    """Page point (z flipped) -> beta."""
    v = np.array([q[0], q[1], -q[2], 1.0])
    return np.asarray(cell_inv) @ v


def copies(w, data, H=0.012):
    """Triangles (3 x 4 betas, normalised) of the wall's copies, in the page's order."""
    Minv = np.asarray(data["cellInv"], float)
    B = np.array([to_beta(q, Minv) for q in w["verts"]])
    Bn = B / B.sum(axis=1, keepdims=True)
    F = w["faces"]
    key = lambda pc: ",".join(f"{x:.5f}" for x in np.mean(pc, axis=0))
    seen = {key([Bn[k] for k in f]) for f in F}
    grid = set()
    cell = lambda b: tuple(np.floor(b[:3] / H).astype(int))
    for b in Bn:
        grid.add(cell(b))

    def near(b):
        c = cell(b)
        return any((c[0] + dx, c[1] + dy, c[2] + dz) in grid for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1))
    out = []
    for g in w.get("copyOps", []):
        M = np.asarray(data["copyMats"][str(g)], float)
        V = B @ M.T
        polys = []
        for f in F:
            s = V[f].sum(axis=1)
            if not (np.sign(s[0]) == np.sign(s[1]) == np.sign(s[2])) or np.abs(s).min() <= 1e-12:
                continue
            t = V[f] / s[:, None]
            if not all(b.max() > -1e-9 for b in t):
                continue
            if max(np.linalg.norm(t[0] - t[1]), np.linalg.norm(t[1] - t[2]), np.linalg.norm(t[2] - t[0])) > 0.12:
                continue
            polys += _into_half(list(t))
        if not polys:
            continue
        on = sum(near(np.mean(pc, axis=0)) for pc in polys)
        if on > 0.5 * len(polys):
            continue
        for pc in polys:
            kk = key(pc)
            if kk in seen:
                continue
            seen.add(kk)
            for b in pc:
                grid.add(cell(b))
            for m in range(1, len(pc) - 1):
                out.append(np.array([pc[0], pc[m], pc[m + 1]]))
    return out
