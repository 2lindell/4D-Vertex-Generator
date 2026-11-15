"""Check that the ixoic shapes listed as points (fixed_lines.json) are points, not lines (point_check.md).

A point's seed x = X beta is fixed by its stabiliser in the sheet's group; the betas keeping that stabiliser (and
summing to 1) are its locus: a lone point when the stabiliser's fixed set is a line through the origin (no freedom:
the shape is a point whatever its walls do), else a line of the half-turn's fixed set, along which the seed is stepped
1e-5, 1e-4 and 1e-3 both ways. A point keeps its key nowhere along it.

    python point_check.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = r'''
import json, sys
import numpy as np
sys.path.insert(0, %r)
import group as G
def lab(b, t=1e-7):
    try:
        return G.label(np.asarray(b, float) / np.sum(b), t)[1]
    except Exception:
        return None
out = {}
for k, b in json.loads(sys.stdin.read()).items():
    b = np.asarray(b, float); b = b / b.sum(); x = G.X @ b
    stab = [g for g in G.ELEMENTS if np.allclose(g @ x, x, atol=1e-9)]
    A = np.vstack([(g - np.eye(4)) @ G.X for g in stab] + [np.ones((1, 4))])
    _, s, Vt = np.linalg.svd(A)
    null = Vt[(s > 1e-9).sum():]
    steps = [[e, lab(b + e * d), lab(b - e * d)] for d in null for e in (1e-5, 1e-4, 1e-3)]
    out[k] = {"stabiliser": len(stab), "free": len(null), "key": lab(b), "steps": steps,
              "point": all(s1 != k and s2 != k for _, s1, s2 in steps)}
print(json.dumps(out))
'''


def main():
    F = json.load(open(os.path.join(HERE, "fixed_lines.json")))
    by = {}
    seen = set()
    for sub, o in F.items():
        for k, r in o["shapes"].items():
            if r["kind"] == "point" and k not in seen:
                seen.add(k)
                by.setdefault(sub, {})[k] = r["seed"]
    L = ["# Ixoic points checked", "", __doc__.strip().split("\n\n")[1], "",
         "| key | sheet | stabiliser | free directions | keys stepped 1e-5 / 1e-3 either way | point |", "|---|---|---|---|---|---|"]
    for sub, items in sorted(by.items()):
        p = subprocess.run([sys.executable, "-c", CODE % HERE], input=json.dumps(items), capture_output=True, text=True,
                           cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        for k, r in json.loads(p.stdout.strip().splitlines()[-1]).items():
            st = "; ".join(f"{s[0]:g}: {s[1]} / {s[2]}" for s in r["steps"] if s[0] in (1e-5, 1e-3)) or "none (a lone point)"
            L.append(f"| {k} | {sub} | {r['stabiliser']} | {r['free']} | {st} | {'yes' if r['point'] else 'NO'} |")
            print(sub, k, r["stabiliser"], r["free"], r["point"], flush=True)
    open(os.path.join(HERE, "point_check.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
