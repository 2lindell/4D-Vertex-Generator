"""The curves of a curved A4 map found exactly, by algebra (no grid, no seed search).

Every surface is a polynomial with rational coefficients, homogeneous in the betas. Two surfaces S, T meet in a curve
whose pieces are found by elimination: the resultant of S and T in b4 is a form R(b1, b2, b3), the curve's shadow on
the plane b4 = 0, and R's factors over Q are the shadows of the curve's irreducible pieces (with the multiplicity
they meet with). A factor is a piece only if a generic point of it lifts to a common root of S and T (a factor from
both leading coefficients in b4 vanishing does not): one point of each factor, taken at random rational b1, b2 and
solved to 50 digits, decides.

Each piece is then named by where it lies: in a mirror (beta_k = 0) or on a chamber edge, in a plane, or a curve
through the chamber; and the swap (beta read backwards) carries pieces of (S, T) onto pieces of their swap images,
checked by carrying a point.

    python a4_algebra.py kidecaic        (a4_algebra_<group>.json)
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_exact as E  # noqa: E402
import a4_pieces as P  # noqa: E402

b1, b2, b3, b4 = E.B
mp.mp.dps = 50
SWAP = {b1: b4, b2: b3, b3: b2, b4: b1}


def lift(S, T, F, rng, tries=8):
    """A point of the factor F(b1, b2, b3) = 0 lifted to S = T = 0: b1, b2 rational at random, b3 a root of F, b4 a
    common root of S and T there. None when no root lifts (F is no piece of the curve)."""
    for _ in range(tries):
        v1, v2 = sp.Rational(rng.randint(1, 97), 97), sp.Rational(rng.randint(1, 97), 97)
        f3 = sp.Poly(F.subs({b1: v1, b2: v2}), b3)
        if f3.degree() <= 0:
            continue
        for r3 in sp.Poly(f3, b3).nroots(n=50, maxsteps=200):
            s4 = sp.Poly(S.subs({b1: v1, b2: v2, b3: r3}), b4)
            t4 = sp.Poly(T.subs({b1: v1, b2: v2, b3: r3}), b4)
            if s4.degree() <= 0 or t4.degree() <= 0:
                continue
            for r4 in s4.nroots(n=50, maxsteps=200):
                val = t4.eval(r4)
                if abs(complex(val)) < 1e-30 * max(1.0, abs(complex(r4)) ** t4.degree()):
                    return [v1, v2, r3, r4]
    return None


def kind_of(F):
    """Where the piece whose shadow is F lies, from F itself."""
    Fp = sp.Poly(F, b1, b2, b3)
    if Fp.total_degree() == 1:
        c = [Fp.coeff_monomial(m) for m in (b1, b2, b3)]
        if sum(x != 0 for x in c) == 1:
            return f"in the plane b{c.index(next(x for x in c if x != 0)) + 1} = 0"
        return f"in the plane {sp.sstr(F)} = 0"
    return "curved"


def main(group):
    X = pickle.load(open(os.path.join(HERE, f"a4_closed_{group}.pkl"), "rb"))
    names = X["surfaces"]
    exprs = [sp.expand(P.surf_of(n).expr) for n in names]
    # (each surface's swap image in the list: its expression reversed, a constant multiple of one there)
    sw = []
    for e in exprs:
        r = sp.expand(e.subs(SWAP, simultaneous=True))
        sw.append(next(j for j, f in enumerate(exprs) if sp.cancel(r / f).is_number))
    rng = random.Random(1)
    out = {"surfaces": names, "swap": sw, "pairs": []}
    t0 = time.time()
    for i in range(len(exprs)):
        for j in range(i + 1, len(exprs)):
            S, T = exprs[i], exprs[j]
            R = sp.resultant(S, T, b4)
            pieces = []
            if R == 0:
                pieces.append({"shadow": "0", "note": "S and T share a factor: a whole surface in common"})
            else:
                lc = sp.gcd(sp.Poly(S, b4).LC(), sp.Poly(T, b4).LC())
                for F, m in sp.factor_list(R)[1]:
                    if not F.free_symbols:
                        continue
                    # (R vanishes on F only where S and T share a root in b4, or both their leading coefficients in
                    #  b4 vanish: F dividing neither is a piece for certain; else a generic point of F decides)
                    sure = sp.rem(sp.Poly(lc, b1, b2, b3), sp.Poly(F, b1, b2, b3)).is_zero is False
                    try:
                        pt = lift(S, T, F, rng)
                    except Exception:
                        pt = None
                    if sure and pt is None:
                        pt = []
                    pieces.append({"shadow": sp.sstr(F), "degree": int(sp.Poly(F, b1, b2, b3).total_degree()),
                                   "multiplicity": int(m), "kind": kind_of(F), "piece": pt is not None,
                                   "point": [str(sp.N(v, 30)) for v in pt] if pt else None,
                                   "sure": bool(sure)})
            out["pairs"].append({"pair": [i, j], "pieces": pieces})
        print(f"  surface {i}: done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(HERE, f"a4_algebra_{group}.json"), "w"), indent=1)
    real = [(p["pair"], q) for p in out["pairs"] for q in p["pieces"] if q.get("piece")]
    print(f"{len(out['pairs'])} pairs; {len(real)} pieces "
          f"({sum(q['kind'] == 'curved' for _, q in real)} curved, "
          f"{sum(q['kind'].startswith('in the plane b') for _, q in real)} in a mirror, "
          f"{sum(q['kind'].startswith('in the plane') and not q['kind'].startswith('in the plane b') for _, q in real)}"
          f" in another plane); "
          f"{sum(1 for p in out['pairs'] for q in p['pieces'] if q.get('piece') is False)} factors no piece")


if __name__ == "__main__":
    for g in sys.argv[1:]:
        main(g)
