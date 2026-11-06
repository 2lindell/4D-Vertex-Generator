"""Optimized oxitic seeds that stopped on a mirror: carried on into the other sheet (updates opt_central_<sub>.json and
the optimized .off files).

For oxitic the F4 mirrors are seams, not edges: a seed just beyond mirror k is, under oxitic, the reflection of a
chamber seed under the mirror-image group oxb (keys agree: an oxb polychoron has the arrangement key of the oxitic one
of the reflected seed). The optimiser keeps beta >= 0, so a shape whose optimum lands on a mirror it did not start on
is optimised again in the other sheet from that point (nudged inside), its wall surface mapped by the reflection
(beta -> UNIT r_k X beta), and so on while it keeps landing on mirrors (at most 4 crossings). The best R is kept.

    python cross_seam.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTIC = os.path.join(ROOT, "contic_atlas")

STEP = r'''
import json, sys, os
import numpy as np, sympy as sp
sys.path.insert(0, %r)
import optimize_seeds as C
G = C.G
a = json.loads(sys.argv[1])
s = {"name": a["name"], "kind": a["kind"], "degrees": a["degrees"], "seed": a["seed"], "locus": a["locus"],
     "hull_tol": a["hull_tol"]}
r = C.O.optimise(s)
b = np.asarray(r.get("off_seed", r["best"]), float)
print(json.dumps({"best": list(map(float, r["best"])), "off_seed": list(map(float, b)), "R": r["R"], "status": r["status"],
                  "key": G.label(b / b.sum(), a["hull_tol"])[1]}))
'''


def reflect_map(k):
    """beta (chamber, other sheet) -> beta of the reflected point (this sheet): the 4x4 matrix T."""
    sys.path.insert(0, CONTIC)
    import numpy as np
    os.environ.setdefault("CONTIC_SUB", "ox")
    import group as G
    return G.UNIT @ G.REFL[k] @ G.X


def main():
    import numpy as np
    import sympy as sp
    B = sp.symbols("b1 b2 b3 b4")
    T = {k: reflect_map(k) for k in range(4)}
    rows = {sub: json.load(open(os.path.join(HERE, f"opt_central_{sub}.json"))) for sub in ("ox", "oxb")}
    for sub in ("ox", "oxb"):
        for row in rows[sub]["shapes"]:
            if not (min(row["seed"]) > 1e-9 and min(row["optimized"]) < 1e-9):
                continue
            cur_sub, b, locus = sub, np.asarray(row["optimized"], float), list(row["locus"])
            best = (row["off_R"], cur_sub, b.tolist())
            trail = [f"{sub}: R {row['off_R']:.6f} on mirror b{int(np.argmin(b)) + 1} = 0"]
            for _ in range(4):
                k = int(np.argmin(b))
                if b[k] > 1e-9:
                    break
                Tk = T[k]                                  # (the same matrix both ways: reflections are involutions)
                other = "oxb" if cur_sub == "ox" else "ox"
                start = b.copy()
                start[k] = 0.004 * start.max()             # (just across the mirror, in the other sheet's chamber)
                # (the wall surface in the other sheet's betas: f(T beta))
                sub_map = dict(zip(B, [sum(float(Tk[i, j]) * B[j] for j in range(4)) for i in range(4)]))
                loc2 = [str(sp.expand(sp.sympify(t).subs(sub_map, simultaneous=True))) for t in locus]
                if loc2:                                   # (put the start on the mapped surface)
                    fs = [sp.lambdify(B, sp.sympify(t)) for t in loc2]
                    gs = [[sp.lambdify(B, sp.diff(sp.sympify(t), v)) for v in B] for t in loc2]
                    for _ in range(50):
                        for f, g in zip(fs, gs):
                            gr = np.array([gi(*start) for gi in g], float)
                            gr[k] = gr[k]                    # (all coordinates move)
                            gr -= gr.mean()
                            start = start - f(*start) / (gr @ gr) * gr
                arg = {"name": row["name"], "kind": row["kind"], "degrees": row["degrees"], "seed": start.tolist(),
                       "locus": loc2, "hull_tol": 1e-9 if row["kind"] in ("region", "line") else 1e-7}
                out = subprocess.run([sys.executable, "-c", STEP % CONTIC, json.dumps(arg)], capture_output=True, text=True,
                                     cwd=CONTIC, env={**os.environ, "CONTIC_SUB": other})
                try:
                    r = json.loads(out.stdout.strip().splitlines()[-1])
                except Exception:
                    trail.append(f"{other}: failed {out.stderr[-200:]}")
                    break
                if r["key"] != row["key"]:
                    trail.append(f"{other}: another polytope across the mirror (key {r['key']}): the shape ends there")
                    break
                b2 = np.asarray(r["off_seed"], float)
                b2 = b2 / b2.sum()
                trail.append(f"{other}: R {r['R']:.6f} ({r['status'][:40]})")
                if r["R"] < best[0] - 1e-12:
                    best = (r["R"], other, b2.tolist())
                cur_sub, b, locus = other, b2, loc2
                if b.min() > 1e-9:
                    break
            row["seam_trail"] = trail
            print(row["name"], " | ".join(trail), flush=True)
            if best[1] != sub or best[2] != row["optimized"]:
                row["optimized"], row["off_R"], row["optimized_sheet"] = best[2], best[0], best[1]
                row["optimized_status"] += f"; continued across the mirror into sheet {best[1]}: R {best[0]:.6f}"
                code = (f"import sys, numpy as np; sys.path.insert(0, {CONTIC!r}); import group as G; "
                        f"from four_d_vertex_generator.off import to_4off; "
                        f"V, Vn, i, f, c = G.polytope(np.array({best[2]}), {1e-9 if row['kind'] == 'region' else 1e-7}); "
                        f"open({os.path.join(ROOT, row['opt_off'])!r}, 'w').write(to_4off(Vn, f, c)); print(len(V), len(c))")
                o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=CONTIC,
                                   env={**os.environ, "CONTIC_SUB": best[1]})
                print("  rewritten", o.stdout.strip(), o.stderr[-200:], flush=True)
    for sub in ("ox", "oxb"):
        json.dump(rows[sub], open(os.path.join(HERE, f"opt_central_{sub}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
