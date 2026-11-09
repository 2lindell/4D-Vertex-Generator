"""Check every ixitic 'line' shape (ixitic_shapes.json) for a degree of freedom along its line: the seed moved by
+-1e-5 and +-1e-3 along the line (its two nonzero betas traded), classified on its sheet. A shape whose key is lost on
both sides at once is an isolated point on the line (a change point), not a line.

    python line_check.py
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
out = []
for name, seed, key in json.loads(sys.stdin.read()):
    b = np.asarray(seed, float); b /= b.sum()
    nz = [i for i in range(4) if b[i] > 1e-9]
    d = np.zeros(4); d[nz[0]] = 1; d[nz[1]] = -1
    keys = {e: G.label(b + e * d, 1e-9)[1] for e in (-1e-3, -1e-5, 1e-5, 1e-3)}
    out.append([name, len(nz), {str(e): k == key for e, k in keys.items()}])
print(json.dumps(out))
'''


def main():
    S = [s for s in json.load(open(os.path.join(HERE, "ixitic_shapes.json")))["shapes"] if s["kind"] == "line"]
    by = {}
    for s in S:
        by.setdefault(s["sheet"], []).append([s.get("name") or s["key"], s["seed"], s["key"]])
    res = []
    for sub, items in by.items():
        r = subprocess.run([sys.executable, "-c", CODE % HERE], input=json.dumps(items), capture_output=True, text=True,
                           cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        res += json.loads(r.stdout)
    for name, nz, k in res:
        point = not any(k.values())
        print(f"{name:12} nonzero betas {nz}  kept at {k}  -> {'POINT' if point else 'line'}")
    return res


if __name__ == "__main__":
    main()
