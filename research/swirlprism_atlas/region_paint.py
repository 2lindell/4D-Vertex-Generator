"""Which wall triangles of a dodecahedron view border a region's cell (region_cells.py), so the page can colour them in
the region's colour: no new surfaces, only the numbers of the views' own triangles.

Each triangle is stepped off on both sides; the step point, in the view's gnomonic chart at its centre vertex, is
taken back to the 3-sphere, brought into the half-cell by the group, and looked up in the region's voxels.
"""
from __future__ import annotations

import numpy as np

from cellframe import T
from cell_atlas import P

_TI = np.linalg.inv(T)
_M_INV = np.linalg.inv(np.vstack([P.T, np.ones(4)]))


def _member(grid, runs):
    m = np.zeros(int(np.prod(grid["shape"])), bool)
    for a, n in runs:
        m[a:a + n] = True
    return m.reshape(grid["shape"])


def paint(view, centre_frame, grid, runs_by_region, steps=(0.004, 0.007, 0.011)):
    """Sets w["paint"] = {region: [face indices]} on every wall of the view (after its walls are indexed into
    verts/faces). centre_frame = (C, B): the view's chart, X ~ C + B^T u."""
    import edge_snap
    C, B = centre_frame
    G = edge_snap.group()
    lo, h, shape = np.array(grid["lo"]), grid["h"], np.array(grid["shape"])
    members = {r: _member(grid, runs) for r, runs in runs_by_region.items()}

    def regions_at(U):
        """The regions (of those given) whose cell holds each chart point U (n, 3): a list of sets."""
        X = C[None] + U @ B
        Bt = X @ _TI
        Bt = Bt / Bt.sum(axis=1, keepdims=True)
        Y = np.einsum("gij,nj->ngi", G, Bt)
        s = Y.sum(axis=-1, keepdims=True)
        Y = Y / np.where(np.abs(s) < 1e-12, 1e-12, s)
        ok = (Y.min(axis=-1) > 1e-9) & (Y[..., 2] > Y[..., 3]) & (s[..., 0] > 0)
        out = [set() for _ in range(len(U))]
        n_, g_ = np.nonzero(ok)
        if not len(n_):
            return out
        Q = (np.c_[Y[n_, g_], ] @ np.vstack([P.T, np.ones(4)]).T)[:, :3]
        idx = np.floor((Q - lo) / h).astype(int)
        good = np.all((idx >= 0) & (idx < shape), axis=1)
        for n, i, gd in zip(n_, idx, good):
            if gd:
                for r, m in members.items():
                    if m[tuple(i)]:
                        out[n].add(r)
        return out
    total = {r: 0 for r in members}
    for w in view.get("xwalls", []):
        V, F = np.asarray(w.get("verts", []), float), np.asarray(w.get("faces", []), int)
        if not len(F):
            continue
        A, Bv, Cv = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        c = (A + Bv + Cv) / 3
        n = np.cross(Bv - A, Cv - A)
        ln = np.linalg.norm(n, axis=1)
        n = n / np.maximum(ln, 1e-300)[:, None]
        hit = {r: np.zeros(len(F), bool) for r in members}
        for s in (1, -1):
            for st in steps:
                got = regions_at(c + s * st * n)
                for k, rs in enumerate(got):
                    for r in rs:
                        hit[r][k] = True
        w["paint"] = {r: np.flatnonzero(hv & (ln > 1e-14)).tolist() for r, hv in hit.items() if hv.any()}
        for r in w["paint"]:
            total[r] += len(w["paint"][r])
    return total
