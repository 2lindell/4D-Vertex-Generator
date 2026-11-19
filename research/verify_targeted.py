"""crossing_verify.py's checks on the targeted search's candidates (crossing_search/targeted_<atlas>_<sheet>.json).

A targeted candidate first found where two cells meet carries that line's two surfaces even when its shape fills a
wall; so first, for each two-surface candidate, the shape is stepped 1e-4 within each of its surfaces (away from the
other, re-projected): if it keeps its key both ways on one, it is a wall on that surface alone. Then each sheet's
candidates go through crossing_verify.py's checks (stable label, holds along its locus, changes off it), four sheets at
a time. Report: crossing_search/verified_targeted.json (crossing_search/verified.json is left as it is).

    python verify_targeted.py hyic [ixo0 ixo1 ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crossing_verify import ATLAS, CODE, OUT  # noqa: E402

PRE = r'''
import json, sys
import numpy as np
sys.path.insert(0, %(atlas)r)
import group as G
from snub_map import Surface
C = json.loads(sys.stdin.read())


def lab(b):
    try:
        return G.label(np.asarray(b, float), 1e-7)[1]
    except Exception:
        return None


def onto(x, S, n=200):
    x = np.asarray(x, float)
    for _ in range(n):
        for s in S:
            x = s.project(x)
            x = x / x.sum()
    return x if max(s.dist(x) for s in S) < 1e-10 and x.min() > 0 else None


for k, c in C.items():
    if len(c["surfaces"]) != 2:
        continue
    S = [Surface(t) for t in c["surfaces"]]
    x = onto(c["seed"], S)
    if x is None:
        continue
    for a, b in ((0, 1), (1, 0)):
        ga, gb = S[a].grad(x), S[b].grad(x)
        d = gb - (gb @ ga) / (ga @ ga) * ga
        d -= d.mean()
        if np.linalg.norm(d) < 1e-12:
            continue
        d /= np.linalg.norm(d)
        ys = [onto(x + e * d, [S[a]]) for e in (1e-4, -1e-4)]
        if all(y is not None and lab(y) == k for y in ys):
            c["surfaces"] = [c["surfaces"][a]]
            c["seed"] = ys[0].tolist()
            c["kind"] = "wall"
            break
    else:
        c["kind"] = "line"
print(json.dumps(C))
'''


def one(atlas, sheet):
    f = os.path.join(OUT, f"targeted_{atlas}_{sheet}.json")
    d = json.load(open(f))
    cands = d.get("candidates", {})
    if not cands:
        return sheet, {}
    adir, var = ATLAS[atlas]
    adir = os.path.join(HERE, adir)
    env = {**os.environ, var: sheet}
    p = subprocess.run([sys.executable, "-c", PRE % {"atlas": adir}], input=json.dumps(cands), capture_output=True,
                       text=True, cwd=adir, env=env)
    if p.returncode:
        print(sheet, "FAILED (walls)", p.stderr[-1500:], flush=True)
        return sheet, None
    cands = json.loads(p.stdout.strip().splitlines()[-1])
    p = subprocess.run([sys.executable, "-c", CODE % {"atlas": adir}], input=json.dumps({"cands": cands}),
                       capture_output=True, text=True, cwd=adir, env=env)
    if p.returncode:
        print(sheet, "FAILED (verify)", p.stderr[-1500:], flush=True)
        return sheet, None
    o = json.loads(p.stdout.strip().splitlines()[-1])
    for k, r in o.items():
        r["signature"] = cands[k]["signature"]
        print(f"{atlas} {sheet} {k} {r['kind']}: {r['verdict']} (degrees {r.get('degrees')}); along {r.get('along')}; "
              f"off {r.get('changes_off')}", flush=True)
    return sheet, o


def main(atlas, sheets):
    dst = os.path.join(OUT, "verified_targeted.json")
    res = json.load(open(dst)) if os.path.exists(dst) else {}
    with ThreadPoolExecutor(int(os.environ.get("NPROC", "4"))) as ex:
        for sheet, o in ex.map(lambda s: one(atlas, s), sheets):
            if o is not None:
                res[f"{atlas}_{sheet}"] = o
                json.dump(res, open(dst, "w"), indent=1)    # (after each sheet: a restart keeps what is done)


if __name__ == "__main__":
    atlas = sys.argv[1]
    sheets = sys.argv[2:] or [f.split("_", 2)[2][:-5] for f in sorted(os.listdir(OUT))
                              if f.startswith(f"targeted_{atlas}_") and f.endswith(".json")]
    main(atlas, sheets)
