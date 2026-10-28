"""F5's patch in the mirror beta2 = beta4 made the exact image of its trimmed patch in beta1 = phi^-2 beta2 (the copy
under the extra half-turn), and X98's crescent carried there too: the two patches are one piece of the wall and its
image, so they must match. The traced patch in beta2 = beta4 had been left untrimmed, F5 running on over where X98's
image lies (fexact.py grid F5,X98 traced only the other plane).

Every corner, straight edge and conic is carried by the one element of the extended group taking the plane
beta1 = phi^-2 beta2 onto beta2 = beta4 and the patch inside the half-cell; nothing is traced again.

    python fix_x98_f5.py          (rewrites fexact_patches.json)
"""
from __future__ import annotations

import json

import numpy as np
from shapely.geometry import Polygon

from t2_pieces import extended

MONS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
FOLD = np.eye(4)[[1, 0, 3, 2]]
HALF = [np.eye(4)[a] for a in range(4)] + [np.array([0, 0, 1, -1.0])]


def arr(x):
    return np.asarray(json.loads(x) if isinstance(x, str) else x, float)


def frame(n):
    return np.linalg.svd(np.vstack([n, np.ones(4)]))[2][2:]


def poly2(C, n):
    C = np.asarray(C, float)
    return Polygon((C / C.sum(1, keepdims=True)) @ frame(n).T).buffer(0)


def conic_matrix(c):
    """The conic as a symmetric 4x4 form on beta."""
    A = np.zeros((4, 4))
    for ci, (i, j) in zip(c["coeffs"], MONS):
        a, b = c["vars"][i], c["vars"][j]
        A[a, b] += ci / 2
        A[b, a] += ci / 2
    return A


def conic_in(A, keep, sub):
    """A 4x4 form restricted to a plane by sub (4x3: beta from the three kept betas), as vars/coeffs."""
    R = sub.T @ A @ sub
    co = [R[i, i] if i == j else 2 * R[i, j] for i, j in MONS]
    s = max(abs(x) for x in co)
    return {"vars": keep, "coeffs": [x / s for x in co]}


def main():
    P = json.load(open("fexact_patches.json"))
    if "X98 b2=b4" in P:
        print("already done")
        return
    key = lambda t, w: next(k for k, p in P.items() if p["target"] == t and p["wall"] == w)
    kc, km, kx = key("F5", "b1=phi-2*b2"), key("F5", "b2=b4"), key("X98", "b1=phi-2*b2")
    nc, nm = arr(P[kc]["normal"]), arr(P[km]["normal"])
    Cc, Cm, Cx = arr(P[kc]["corners"]), arr(P[km]["corners"]), arr(P[kx]["corners"])
    Pm = poly2(Cm, nm)
    # (the element: the plane onto the mirror, the trimmed patch onto the traced one, inside the half-cell)
    best = None
    for M in extended():
        for F in (np.eye(4), FOLD):
            g = F @ M
            Y = Cc @ g.T
            if (Y.sum(1) <= 1e-12).any() or np.abs(Y @ nm).max() > 1e-9 * np.abs(Y).max():
                continue
            Y = Y / Y.sum(1, keepdims=True)
            if min(float((Y @ h).min()) for h in HALF) < -1e-9:
                continue
            common = poly2(Y, nm).intersection(Pm).area
            if best is None or common > best[0]:
                best = (common, g)
    common, g = best
    gi = np.linalg.inv(g)
    img = lambda C: (lambda Y: Y / Y.sum(1, keepdims=True))(np.asarray(C) @ g.T)
    Ic, Ix = img(Cc), img(Cx)
    a_m, a_c, a_x = Pm.area, poly2(Ic, nm).area, poly2(Ix, nm).area
    over = poly2(Ix, nm).intersection(poly2(Ic, nm)).area
    print(f"F5 in beta2 = beta4 as traced: {a_m:.5f}; image of the trimmed patch {a_c:.5f}; image of X98's crescent "
          f"{a_x:.5f} (overlapping the F5 image by {over:.1e}); the two together {a_c + a_x:.5f}, inside the traced "
          f"patch: {poly2(Ix, nm).difference(Pm).area:.1e} outside")
    # (the plane beta2 = beta4: beta from (beta1, beta2, beta3))
    sub = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0.0]])

    def carried(p, C):
        out = dict(p)
        out["normal"] = nm.tolist()
        out["wall"] = "b2=b4"
        out["corners"] = C.tolist()
        out["edges"] = [[(arr(m) @ gi).tolist(), nb, txt] for m, nb, txt in
                        (json.loads(p["edges"]) if isinstance(p["edges"], str) else p["edges"])]
        cs = json.loads(p["conics"]) if isinstance(p.get("conics"), str) else p.get("conics") or []
        out["conics"] = []
        for c in cs:
            A = gi.T @ conic_matrix(c) @ gi
            cc = conic_in(A, [0, 1, 2], sub)
            q = lambda b: sum(ci * b[cc["vars"][i]] * b[cc["vars"][j]] for ci, (i, j) in zip(cc["coeffs"], MONS))
            on = sum(abs(q(b)) < 1e-8 for b in C)
            from exact_conics import conic_text, snap
            vals, forms = snap(np.asarray(cc["coeffs"]))
            if vals is not None:
                cc["coeffs"] = vals.tolist()
            out["conics"].append({**c, **cc, "text": conic_text(cc["vars"], forms) if forms else f"the image of {c['text']}"})
            print(f"  {p['target']}: conic {c['label']} carried, {on} of {len(C)} corners on it")
        out["built"] = (f"the image of the {p['target']} patch in beta1 = phi^-2 beta2 under the extended group "
                        f"(fix_x98_f5.py)")
        return out
    traced = P[km]
    P[km] = carried(P[kc], Ic)
    # (each straight edge written without its beta2 part (the mirror's normal added), and the name of what lies past
    # it kept as the traced patch had it when it is the same line)
    def plain_line(m):
        m = np.asarray(m, float)
        m = m + m[1] * np.array([0, -1, 0, 1.0])
        return m / np.abs(m).max()
    old_edges = [(plain_line(m), nb) for m, nb, _ in traced["edges"]]
    for e in P[km]["edges"]:
        e[0] = plain_line(e[0]).tolist()
        for m, nb in old_edges:
            if min(np.abs(m - e[0]).max(), np.abs(m + e[0]).max()) < 1e-9:
                e[1] = nb
    P["X98 b2=b4"] = carried(P[kx], Ix)
    P["X98 b2=b4"]["target"] = "X98"
    for e in P["X98 b2=b4"]["edges"]:
        e[0] = plain_line(e[0]).tolist()
    json.dump(P, open("fexact_patches.json", "w"), indent=1)
    print("fexact_patches.json: F5 in beta2 = beta4 trimmed, X98 added there")


if __name__ == "__main__":
    main()
