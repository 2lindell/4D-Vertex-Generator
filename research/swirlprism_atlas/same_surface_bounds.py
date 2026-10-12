"""Walls on one surface that bound each other: two pieces of one surface (or of a surface and its image under the
group) meet either along a line shape (X39 and X52 along X29), or along the edge of a third shape they both reach (X52
beside X10 along X8), or directly, with no line between them: only then is the other piece on the wall's boundary.
Read from the mesher's own outlines (xsurfaces_mesher_outlines.json, labelled with the shape just past each point): a
stretch past a piece of the same surface is a direct meeting when it runs on unbroken for 20 points at least (short
stretches alternating with another label are the mesher flickering along that edge: X52 and X39 along X10's X8 edge)
and fewer than half its points lie within 0.004 of a drawn line shape. Written to same_surface_bounds.json as {wall: [other walls]}.

    python same_surface_bounds.py
"""
from __future__ import annotations

import json

import numpy as np

import edge_snap as E
import xsurface


def main():
    X = json.load(open("xsurfaces.json"))
    O = json.load(open("xsurfaces_mesher_outlines.json"))
    eqs = set(X) | set(json.load(open("xsurface_equations.json")))
    LPTS, DIMS = E.shape_points()
    allL = np.vstack(list(LPTS.values()))
    G = E.group()
    Ginv = np.linalg.inv(G)
    GT = np.transpose(Ginv, (0, 2, 1))
    polys = {w: (lambda sf: E.Poly(sf["mons"], sf["coeffs"], w))(xsurface.surface(w)) for w in eqs}

    def on_image(w, P):
        """Do points P lie on w's surface or one of its images?"""
        b = polys[w]
        Y = np.einsum("nj,gjk->gnk", P, GT)
        f = b(Y.reshape(-1, 4)).reshape(len(G), -1)
        gy = b.grad(Y.reshape(-1, 4)).reshape(len(G), -1, 4)
        gx = np.einsum("gnj,gjk->gnk", gy, Ginv)
        gx = gx - gx.mean(axis=2, keepdims=True)
        return np.min(np.median(np.abs(f) / np.maximum(np.linalg.norm(gx, axis=2), 1e-300), axis=1)) < 1e-9

    out = {}
    for tid in X:
        V = np.asarray(X[tid]["tris"], float).reshape(-1, 4)
        inner = V[np.random.default_rng(0).choice(len(V), size=min(40, len(V)), replace=False)]
        same = {w for w in eqs - {tid} if on_image(w, inner)}
        if not same:
            continue
        for l in O.get(tid, {}).get("outline", []):
            labs = [q["beyond"] for q in l]
            B = np.array([q["beta"] for q in l])
            k = 0
            while k < len(labs):
                j = k
                while j < len(labs) and labs[j] == labs[k]:
                    j += 1
                if labs[k] in same and j - k >= 20:
                    P = B[k:j]
                    near = np.min(np.linalg.norm(P[:, None] - allL[None], axis=2), axis=1) < 0.004
                    verdict = "direct" if near.mean() < 0.5 else "along a line"
                    print(f"{tid}: {j - k} points past {labs[k]} (same surface): {verdict} ({near.mean():.0%} near a line)", flush=True)
                    if verdict == "direct":
                        out.setdefault(tid, set()).add(labs[k])
                k = j
    for t, ws in list(out.items()):                  # (a meeting is seen from either side: both have it)
        for w in ws:
            out.setdefault(w, set()).add(t)
    json.dump({t: sorted(v) for t, v in out.items()}, open("same_surface_bounds.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
