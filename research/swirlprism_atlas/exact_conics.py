"""Exact (golden) conics for the curved edges of the exact wall patches.

A curved edge is first found as a conic fitted to bisected boundary points, good to ~1e-9 but not exact. Each run of
arc points with one edge label is fitted again in the wall's own coordinates (three of the betas), its coefficients
are snapped to the simplest golden numbers, the snapped conic is kept only if it still passes through every arc point
to 1e-7, and the arc points are then moved onto it exactly (along the line from the patch centre). The equation is
stored with the patch as patch["conics"] = [{"label", "vars", "coeffs", "text"}].

    python exact_conics.py          # report only
    python exact_conics.py write    # also update fexact_patches.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from cellframe import golden_form

PHI = (1 + 5 ** 0.5) / 2
MONS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def wall_vars(normal):
    """Three betas that coordinatise the wall (the fourth is a linear function of them), and that function."""
    n = np.asarray(normal, float)
    drop = int(np.argmax(np.abs(n)))
    keep = [i for i in range(4) if i != drop]
    coef = -n[keep] / n[drop]          # beta[drop] = coef . beta[keep]
    return keep, drop, coef


GOLDEN = sorted({(a / d, b / d) for d in (1, 2, 4, 5) for a in range(-12, 13) for b in range(-12, 13)},
                key=lambda ab: (abs(ab[0]) + abs(ab[1]), abs(ab[1])))


def _golden(v, tol):
    """The simplest a + b*phi (small rationals a, b) within tol of v, as (value, text), or None."""
    for a, b in GOLDEN:
        x = a + b * PHI
        if abs(x - v) < tol:
            return x, golden_form(x)
    return None


def snap(c, tol=2e-5):
    """Scale by each nonzero coefficient in turn; keep the first scaling where every coefficient is golden."""
    for k in np.argsort(-np.abs(c)):
        if abs(c[k]) < 1e-3 * np.abs(c).max():
            continue
        d = c / c[k]
        got = [(0.0, "0") if abs(v) < tol else _golden(v, tol) for v in d]
        if all(g is not None for g in got):
            return np.array([g[0] for g in got]), [g[1] for g in got]
    return None, None


def conic_text(keep, forms):
    names = [f"β{i + 1}" for i in keep]
    terms = []
    for (i, j), t in zip(MONS, forms):
        if t == "0":
            continue
        m = f"{names[i]}²" if i == j else f"{names[i]}{names[j]}"
        terms.append(m if t == "1" else f"-{m}" if t == "-1" else f"({t})·{m}")
    return " + ".join(terms).replace("+ -", "− ") + " = 0"


def quad(c, y):
    return sum(ci * y[i] * y[j] for ci, (i, j) in zip(c, MONS))


def onto(c, keep, drop, coef, p, centre):
    """Move beta p along the line from the patch centre until it lies on the conic (the nearer root)."""
    def y(b):
        return np.asarray(b)[keep]
    d = p - centre
    # q(centre + s d) is quadratic in s
    q0, q1, qm = quad(c, y(centre)), quad(c, y(centre + d)), quad(c, y(centre - d))
    A = (q1 + qm) / 2 - q0
    B = (q1 - qm) / 2
    roots = np.roots([A, B, q0]) if abs(A) > 1e-300 else np.array([-q0 / B])
    roots = roots[np.isreal(roots)].real
    if not len(roots) or np.min(np.abs(roots - 1)) > 1e-4:     # not on this conic after all (a straight-edge corner)
        return p
    s = roots[np.argmin(np.abs(roots - 1))]
    b = centre + s * d
    return b / b.sum()


def main(write=False):
    allp = json.load(open("fexact_patches.json"))
    known = {}
    # patches whose conics snap most easily first, so the others can reuse them
    for key, p in sorted(allp.items(), key=lambda kv: (kv[1]["target"] not in ("F5", "X31", "X79"), kv[0])):
        C = np.array(p["corners"], float)
        n = len(C)
        if n < 12 or not p.get("edge_labels"):
            continue
        C = C / C.sum(axis=1, keepdims=True)
        keep, drop, coef = wall_vars(p["normal"])
        centre = C.mean(axis=0)
        runs = {}
        for a, b, lab, *kind in p["edge_labels"]:
            if kind:
                continue
            runs.setdefault(lab, set()).update(k % n for k in range(a + 1, a + 6))
        conics = []
        newC = C.copy()
        for lab, ks in runs.items():
            ks = sorted(ks)
            if len(ks) < 8:
                continue
            Y = C[ks][:, keep]
            vals = None
            for kc in known.get(key.split(" ", 1)[1], []):          # a conic already found on this wall
                r = np.array([abs(quad(kc[0], y)) for y in Y])
                if np.mean(r < 1e-7) > 0.7:
                    vals, forms = kc
                    break
            idx = np.arange(len(Y))
            for _ in range(6 if vals is None else 0):             # fit, dropping the worst points (stray corners)
                A = np.stack([Y[idx, i] * Y[idx, j] for i, j in MONS], 1)
                c = np.linalg.svd(A)[2][-1]
                vals, forms = snap(c)
                if vals is not None:
                    break
                r = np.abs(A @ c)
                idx = idx[r < np.quantile(r, 0.85)] if len(idx) > 10 else idx
            if vals is None:
                print(f"{key}: {lab[:20]} — no golden conic (fit {np.round(c / np.abs(c).max(), 6)})")
                continue
            res = np.array([abs(quad(vals, y)) for y in Y])
            resid = float(np.quantile(res, 0.8))
            if resid > 1e-7:
                print(f"{key}: {lab[:20]} — snapped conic misses the arc by {resid:.1e}")
                continue
            text = conic_text(keep, forms)
            moved = 0.0
            # move the run's own arc points (and its end corners) onto the conic: near a corner two conics meet, so
            # a point of the next arc can nearly lie on this one too
            own = set(ks) | {(min(ks) - 1) % n, (max(ks) + 1) % n}
            for k in sorted(own):
                if abs(quad(vals, C[k][keep])) < 1e-6:
                    nb = onto(vals, keep, drop, coef, C[k], centre)
                    moved = max(moved, np.linalg.norm(nb - C[k]))
                    newC[k] = nb
            conics.append({"label": lab, "vars": keep, "coeffs": vals.tolist(), "text": text})
            known.setdefault(key.split(" ", 1)[1], []).append((vals, forms))
            print(f"{key}: {lab[:20]:20s} {text}   (fit residual {resid:.1e}, points moved ≤ {moved:.1e})")
        if write and conics:
            p["corners"] = newC.tolist()
            p["conics"] = conics
    if write:
        json.dump(allp, open("fexact_patches.json", "w"), indent=1)


if __name__ == "__main__":
    main(write="write" in sys.argv)
