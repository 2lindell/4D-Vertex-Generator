"""What the exact pipeline (a4_curves, a4_arcs, a4_exactmap, a4_walls, a4_exactpieces) needs to know of a group:
its surfaces, the field, the domain it is drawn on, the swap, how a point is classified, and its shapes.

The pipeline always works in coordinates b1..b4 >= 0 (summing to 1) of its domain tetrahedron, whose four faces are
b_k = 0 (the first four surfaces). For the A4 groups the domain is the A4 chamber and b is beta. For the F4 groups the
domain is the union of F4 chambers that holds the whole fundamental domain (as one tetrahedron):

  * kicontic [[3,4,3]]+ and iocontic [[3,4,3]+]: the chamber C (b = beta);
  * oxitic [[3+,4,3+]]: C, r1 C, r4 C and r1 r4 C, the tetrahedron with faces beta2 = 0, beta1 + beta2 = 0, beta3 = 0 and
    beta3 + beta4 = 0 (r1 and r4 commute with the far mirrors, so the four chambers make one tetrahedron); b is those four
    forms. The swap maps it to itself (r1 C <-> r4 C) and oxitic's domain is half of it. In C and r1 r4 C the walls are
    oxitic's (sheet ox: r1 r4 normalises oxitic), in r1 C and r4 C those of its mirror image (sheet oxb), carried there.

F4's forms are over Q(sqrt2): Singular gets sqrt2 as a fifth variable s with s^2 - 2 in every ideal, and only points with
s = +sqrt2 are kept.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CON = os.path.join(ROOT, "contic_atlas")
B = sp.symbols("b1 b2 b3 b4")
F4_GROUPS = {"kicontic": "ki", "iocontic": "io", "oxitic": "ox"}


class Spec:
    group = None
    sub = None
    f4 = False
    vars = "b1,b2,b3,b4"
    extra = ""                      # (more generators for every ideal: s^2 - 2 for F4)
    nvar = 4
    L = "b1+b4"
    LV = np.array([1.0, 0.0, 0.0, 1.0])
    perm = [3, 2, 1, 0]             # (the swap, as a permutation of b; None: no swap)
    M = np.eye(4)                   # (beta = M b: the domain's coordinates to the chamber's betas)

    # surfaces: names and sympy expressions in B (sqrt(2) allowed)
    names = []
    exprs = []

    def swap(self, x):
        return None if self.perm is None else np.asarray(x)[self.perm]

    def keep_point(self, z):
        """z: the solution's coordinates (complex). The real ones of the domain, with s = +sqrt2 for F4."""
        if self.nvar == 5 and not (abs(z[4].imag) < 1e-20 and z[4].real > 0):
            return False
        return max(abs(v.imag) for v in z[:4]) <= 1e-20 and min(v.real for v in z[:4]) >= -1e-20

    def poly_str(self, e):
        s = sp.sstr(sp.expand(e)).replace("**", "^")
        return s.replace("sqrt(2)", "s")

    def sympify(self, text):
        return sp.sympify(text.replace("^", "**"), locals={"s": sp.sqrt(2)})


def a4_spec(group):
    import a4_pieces as P
    sp_ = Spec()
    sp_.group = group
    path = os.path.join(HERE, f"a4_curves_{group}_stage1.json")
    if os.path.exists(os.path.join(HERE, f"a4_closed_{group}.pkl")):
        names = pickle.load(open(os.path.join(HERE, f"a4_closed_{group}.pkl"), "rb"))["surfaces"]
    else:
        names = json.load(open(os.path.join(HERE, f"a4_curves_{group}.json")))["surfaces"]
    sp_.names = names
    sp_.exprs = [P.surf_of(n).expr for n in names]
    import a4_snub as SN
    sp_.sub = SN.SUB[group]
    sp_.classify = lambda pts, tol=1e-7: [k for _, k in SN.classify_many(pts, sp_.sub, tol=tol)]
    return sp_


# ------------------------------------------------------------------------------------------------ F4
def f4_frame():
    """Exact unit roots of F4 (contic_atlas/group.py), and each reflection's action on beta: beta(r y) = W beta(y)."""
    sys.path.insert(0, CON)
    r2 = sp.sqrt(2)
    roots = sp.Matrix([[0, 1, -1, 0], [0, 0, 1, -1], [0, 0, 0, 1],
                       [sp.Rational(1, 2), -sp.Rational(1, 2), -sp.Rational(1, 2), -sp.Rational(1, 2)]])
    U = sp.Matrix([[v / sp.sqrt(sum(w ** 2 for w in roots.row(i))) for v in roots.row(i)] for i in range(4)])
    Ui = sp.simplify(U.inv())
    W = []
    for k in range(4):
        u = U.row(k).T
        R = sp.eye(4) - 2 * u * u.T
        W.append(sp.simplify(U * R * Ui))           # (beta(r_k y) = U r_k y = U r_k U^-1 beta(y))
    return U, Ui, W


def f4_spec(group):
    sub = F4_GROUPS[group]
    sp_ = Spec()
    sp_.group, sp_.sub, sp_.f4 = group, sub, True
    sp_.vars, sp_.extra, sp_.nvar = "b1,b2,b3,b4,s", "s^2-2", 5
    U, Ui, W = f4_frame()
    I4 = sp.eye(4)
    if group == "oxitic":
        # (b = (beta2, beta1 + beta2, beta3, beta3 + beta4): the faces of C u r1 C u r4 C u r1 r4 C)
        Bm = sp.Matrix([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
        sheets = [("ox", I4), ("ox", W[0] * W[3]), ("oxb", W[0]), ("oxb", W[3])]
        sp_.perm = [2, 3, 0, 1]
        sp_.L, sp_.LV = "b1+b3", np.array([1.0, 0.0, 1.0, 0.0])
    else:
        Bm = I4
        sheets = [(sub, I4)]
    Minv = Bm.inv()                                   # (beta = Minv b)
    sp_.M = np.array(Minv.evalf(), float)
    bvec = sp.Matrix(B)
    names, exprs = [f"b{k + 1}" for k in range(4)], list(B)

    def same(e, f):
        r = sp.cancel(sp.expand(e) / sp.expand(f))
        return r.is_number

    for sheet, Wg in sheets:
        data = json.load(open(os.path.join(CON, f"snub_map_{sheet}.json")))
        for text in data["surfaces"]:
            e = sp.sympify(text)
            # (the sheet's surface, a form in the chamber's betas, carried to the chamber g C: a point y there is on it
            #  when g^-1 y is, beta(g^-1 y) = W_g beta(y) (W_g its own inverse for a reflection; r1 r4 = r4 r1))
            bt = Wg * (Minv * bvec)
            f = sp.expand(e.subs({sp.Symbol(f"b{k + 1}"): bt[k] for k in range(4)}, simultaneous=True))
            if not (f.free_symbols & set(B)):
                continue
            for fac, _ in sp.factor_list(sp.Poly(f, *B, extension=sp.sqrt(2)))[1]:
                fac = fac.as_expr()
                if not fac.free_symbols & set(B):
                    continue
                if any(same(fac, g) for g in exprs):
                    continue
                names.append(f"{sheet}{'' if Wg == I4 else ' moved'}: {sp.sstr(fac)[:60]}")
                exprs.append(sp.expand(fac))
    # (each surface's swap image too: the atlas found the walls on the half beta1 >= beta4 only)
    if sp_.perm is not None:
        k0 = len(exprs)
        for i in range(4, k0):
            g = exprs[i].subs({B[j]: B[sp_.perm[j]] for j in range(4)}, simultaneous=True)
            if not any(same(g, h) for h in exprs):
                names.append("swap of " + names[i])
                exprs.append(sp.expand(g))
    sp_.names, sp_.exprs = names, exprs
    Mnp = sp_.M

    def classify(points, tol=1e-7):
        betas = [Mnp @ np.asarray(p, float) for p in points]
        betas = [list(map(float, b / np.linalg.norm(b))) for b in betas]
        code = ("import json, sys, numpy as np\nimport group as G\nP = json.loads(sys.stdin.read())\n"
                f"print(json.dumps([G.label(np.asarray(p, float), {tol})[1] for p in P]))")
        from concurrent.futures import ThreadPoolExecutor

        def run(chunk):
            r = subprocess.run([sys.executable, "-c", code], input=json.dumps(chunk), capture_output=True, text=True,
                               cwd=CON, env={**os.environ, "CONTIC_SUB": sub}, check=True)
            return json.loads(r.stdout)
        if not betas:
            return []
        k = max(1, -(-len(betas) // 4))
        chunks = [betas[i:i + k] for i in range(0, len(betas), k)]
        with ThreadPoolExecutor(4) as ex:
            parts = list(ex.map(run, chunks))
        return [x for p in parts for x in p]
    sp_.classify = classify
    return sp_


def _v(x):
    return json.loads(x) if isinstance(x, str) else x


def f4_shapes(group):
    """The group's shapes from the contic / toxitic atlases: (name, kind, point in the domain's coordinates). A wall,
    line or point at the atlas's own seed (found on it), a region at its deepest point; a shape of the mirror sheet
    oxb carried to r1 C, where it lies under oxitic."""
    U, Ui, W = f4_frame()
    out = []
    if group in ("kicontic", "iocontic"):
        sub = F4_GROUPS[group]
        O = json.load(open(os.path.join(CON, f"optimized_seeds_{sub}.json")))["shapes"]
        C = {r["name"]: r for r in json.load(open(os.path.join(CON, f"central_seeds_{sub}.json")))["shapes"]}
        for r in O:
            b = _v(C[r["name"]]["centre"]) if r["kind"] == "region" and r["name"] in C else _v(r["start"])
            out.append((r["name"], r["kind"], np.asarray(b, float)))
        return out
    Bm = np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], float)
    W1 = np.array(W[0].evalf(), float)
    for sheet in ("ox", "oxb"):
        o = json.load(open(os.path.join(ROOT, "toxitic_atlas", f"opt_central_{sheet}.json")))
        o = o["shapes"] if isinstance(o, dict) and "shapes" in o else o
        for r in o:
            b = np.asarray(_v(r["central"] if r["kind"] == "region" and r.get("central") else r["seed"]), float)
            if sheet == "oxb":
                b = W1 @ b
            out.append((r["name"], r["kind"], Bm @ b))
    return out


_CACHE = {}


def get(group):
    if group not in _CACHE:
        _CACHE[group] = f4_spec(group) if group in F4_GROUPS else a4_spec(group)
    return _CACHE[group]


if __name__ == "__main__":
    for g in sys.argv[1:]:
        s = get(g)
        print(g, len(s.names), "surfaces;", "M =", np.round(s.M, 3).tolist())
        for n, e in list(zip(s.names, s.exprs))[:8]:
            print("  ", n[:50], "|", s.poly_str(e)[:80])
