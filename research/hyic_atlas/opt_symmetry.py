"""Full symmetry (fullsym.py) and ixoic elements kept, of every ixitic shape at its seed, optimized and central betas
(opt_central_ixitic.json): which optima land on a point of higher symmetry.

    python opt_symmetry.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = r'''
import json, sys, numpy as np
sys.path.insert(0, %r)
import group as G
from fullsym import full_symmetry
from scipy.spatial import cKDTree
g = G.sheet_cosets()[G.SHEET]
O = G._cached("ixoic", lambda: None); O = np.einsum("ji,njk,kl->nil", g, O, g)
out = []
for name, bs in json.loads(sys.stdin.read()):
    row = [name]
    for b in bs:
        b = np.asarray(b, float); b /= b.sum()
        V, _ = G.vertices(b)
        n, _ = full_symmetry(V)
        T = cKDTree(V); tol = 1e-7 * np.abs(V).max()
        ok = T.query(O @ V[0])[0] < tol
        kept = sum(1 for h in O[ok] if T.query(V @ h.T)[0].max() < tol)
        row.append([int(n), int(kept)])
    out.append(row)
print(json.dumps(out))
'''


def main():
    R = json.load(open(os.path.join(HERE, "opt_central_ixitic.json")))["shapes"]
    by = {}
    for r in R:
        by.setdefault(r["sheet"], []).append([r["name"], [r["seed"], r["optimized"], r["central"]]])
    res = []
    for sub, items in sorted(by.items()):
        p = subprocess.run([sys.executable, "-c", CODE % HERE], input=json.dumps(items), capture_output=True, text=True,
                           cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        res += json.loads(p.stdout)
    for name, s, o, c in res:
        flag = "  <-- higher" if max(s[0], o[0], c[0]) > 1440 else ""
        print(f"{name:24} seed {s[0]:6} ({s[1]} ixoic)  optimized {o[0]:6} ({o[1]})  central {c[0]:6} ({c[1]}){flag}")
    json.dump(res, open(os.path.join(HERE, "opt_symmetry.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
