"""Exact equations of every ixoic (or ixitic) line (ixoic_line_equations.json, and .md; ixitic_line_equations.*).

A line of the atlas was stored as one seed and its polytope's key; the surfaces through it were judged numerically
(those within 1e-7 of the seed), and near a point where many surfaces cross that judgement fails (gypsobted: 27 of them,
nearly tangent). Here each line is given two exact equations, b in the chamber with f1(b) = f2(b) = 0:

  * a half-turn's fixed line (fixed_lines.py): (g - 1) X beta = 0 for the half-turn g fixing the seed, the same as
    (UNIT g X - 1) beta = 0, whose entries are exact over Q(sqrt2, sqrt5) (group.exact): two independent rows, linear;
  * every other line: two of the map's exact surfaces (snub_map, polynomials over Q(sqrt2, sqrt5)) or mirrors
    (b_k = 0), those through the seed (within 1e-6). A pair is the line's only if the curve f1 = f2 = 0 traced from the
    seed, both ways, is the line's polytope: each pair is tried (the most transversal first) with steps of 1e-5 both
    ways; the pairs that keep the polytope are traced on in steps of 1e-3 (up to 40 each way) while the polytope stays
    the same, and the pair traced furthest is kept. The other candidate surfaces that vanish all along the traced
    curve are listed (the line lies on them too).

A line whose seed no pair can leave both ways is either at the line's end (one way kept: "line, the seed at its end")
or has no line through it ("no line through the seed: a point"). Each seed is put exactly on its line at 40 digits
(mpmath: f1 = f2 = 0, sum 1, and its position along the line kept).

    python ixoic_line_equations.py [ixitic]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CODE = r'''
import json, sys, itertools
import numpy as np, sympy as sp, mpmath as mp
sys.path.insert(0, %(here)r)
import group as G
from snub_map import Surface, B
L = json.loads(sys.stdin.read())
M = json.load(open(G.fname("snub_map")))
SURF = [Surface(t) for t in M["surfaces"]]
ONE = np.ones(4)


def label_ok(b, key, tol):
    try:
        return G.label(np.asarray(b, float) / np.sum(b), tol)[1] == key
    except Exception:
        return False


def project(x, S):
    x = np.array(x, float)
    for _ in range(80):
        r = np.array([s.val(x) for s in S] + [x.sum() - 1])
        J = np.array([[float(gi(*x)) for gi in s.g] for s in S] + [ONE])
        st = np.linalg.lstsq(J, r, rcond=None)[0]
        x = x - st
        if np.abs(st).max() < 1e-16:
            break
    ok = max(abs(s.val(x)) for s in S) < 1e-12
    return x, ok


def tangent(x, S):
    J = np.array([[float(gi(*x)) for gi in s.g] for s in S] + [ONE])
    t = np.linalg.svd(J)[2][-1]
    return t / np.linalg.norm(t)


def trace(x0, S, key, tol, sgn, h=1e-3, n=40):
    x, k = x0.copy(), 0
    for _ in range(n):
        t = tangent(x, S)
        if k == 0:
            t = t * np.sign(t @ T0) * sgn
        else:
            t = t * np.sign(t @ (x - xp))
        xp = x
        y, ok = project(x + h * t, S)
        if not ok or y.min() < -1e-12 or not label_ok(y, key, tol):
            break
        x, k = y, k + 1
    return k, x


out = {}
for s in L:
    b = np.asarray(s["seed"], float); b = b / b.sum()
    tol = next((t for t in (1e-9, 1e-8, 1e-7) if label_ok(b, s["key"], t)), 1e-7)
    r = {"sheet": G.SUB, "key": s["key"], "source": s["source"], "hull_tol": tol, "seed": b.tolist()}
    fixed = None
    if True:                                   # (a half-turn fixing the seed: its fixed line, tried first)
        for g in G.ELEMENTS:
            if (np.allclose(g @ g, np.eye(4)) and not np.allclose(g, np.eye(4)) and not np.allclose(g, -np.eye(4))
                    and np.abs((g - np.eye(4)) @ G.X @ b).max() < 1e-9):
                A = G.exact(G.UNIT @ g @ G.X) - sp.eye(4)     # ((g - 1) X b = 0 <=> (UNIT g X - 1) b = 0: exact)
                rows = [sp.nsimplify(0) + sum(sp.radsimp(A[i, j]) * B[j] for j in range(4)) for i in range(4)]
                pick = []
                for i in range(4):
                    Mn = np.array([[float(sp.diff(rows[k], v)) for v in B] for k in pick + [i]])
                    if np.linalg.matrix_rank(Mn, 1e-9) == len(pick) + 1:
                        pick.append(i)
                    if len(pick) == 2:
                        break
                if len(pick) == 2:
                    fixed = tuple(Surface(str(sp.expand(rows[i]))) for i in pick)
                    break
    if s["source"] == "fixed line" and fixed:
        cand, pairs = list(fixed), [fixed]
    else:
        cand = [x for x in SURF if x.dist(b) < 1e-6] + [Surface(f"b{k + 1}") for k in range(4) if b[k] < 1e-12]
        sc = []
        for p_, q_ in itertools.combinations(cand, 2):
            u, v = p_.grad(b), q_.grad(b)
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-12 or nv < 1e-12:
                continue
            sn = np.sqrt(max(0.0, 1 - (u @ v / nu / nv) ** 2))
            if sn > 1e-4:
                sc.append((-sn, p_, q_))
        sc.sort(key=lambda z: z[0])
        pairs = ([fixed] if fixed else []) + [(p_, q_) for _, p_, q_ in sc[:400]]
    r["candidates"] = len(cand)
    best, tried, passing = None, 0, 0
    for P in pairs:
        S = list(P)
        tried += 1
        T0 = tangent(b, S)
        way = []
        for sgn in (1, -1):
            y, ok = project(b + sgn * 1e-5 * T0, S)
            way.append(ok and y.min() >= -1e-12 and label_ok(y, s["key"], tol))
        if not any(way):
            continue
        passing += 1
        ext = [trace(b, S, s["key"], tol, sgn)[0] if w else -1 for sgn, w in zip((1, -1), way)]
        score = (all(way), sum(e for e in ext if e > 0))
        if best is None or score > best[0]:
            best = (score, S, ext, way)
        if all(way) and min(ext) >= 40 or passing >= 6:
            break
    r["pairs_tried"] = tried
    if best is None:
        r["status"] = "no line through the seed: a point (no pair of its surfaces keeps the polytope either way)"
    else:
        _, S, ext, way = best
        r["status"] = "line" if all(way) else "line, the seed at its end (kept one way only)"
        r["equations"] = [x.text for x in S]
        r["traced_steps_1e-3"] = {"plus": ext[0], "minus": ext[1]}
        pts = []
        for sgn, e in zip((1, -1), ext):
            if e > 0:
                T0 = tangent(b, S)
                pts += [trace(b, S, s["key"], tol, sgn, n=k)[1] for k in sorted({1, max(1, e // 2), e})]
        others = [x for x in cand if x not in S]
        r["also_on"] = [x.text for x in others if all(x.dist(p) < 1e-9 for p in pts + [b])]
        mp.mp.dps = 50
        fs = [sp.lambdify(B, sp.sympify(x.text), "mpmath") for x in S]
        t = tangent(b, S)
        k = int(np.argmax(np.abs(t)))
        bb = b.copy()
        try:
            sol = mp.findroot(lambda a, c, d, e: [fs[0](a, c, d, e), fs[1](a, c, d, e), a + c + d + e - 1,
                                                  [a, c, d, e][k] - mp.mpf(bb[k])], [mp.mpf(v) for v in bb])
            r["seed_40"] = [mp.nstr(v, 40) for v in sol]
            r["seed_moved"] = float(max(abs(float(v) - w) for v, w in zip(sol, bb)))
        except Exception as e:
            r["seed_40"] = None
            r["seed_40_error"] = str(e)[:120]
    out[s["name"]] = r
    print(f"{s['name']:24} {r['status'][:40]:40} cand {len(cand):3} tried {tried:3} "
          f"{r.get('traced_steps_1e-3')} also_on {len(r.get('also_on', []))} moved {r.get('seed_moved')}", flush=True)
print(json.dumps(out))
'''


GROUP = "ixoic"


def lines():
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "pages"))
    import ixitic_offs as X
    X.GROUP = GROUP
    if GROUP == "ixoic":
        from ixoic_match import atlas
        A = atlas()
    else:                                       # (ixitic: the merged shapes; a half-turn's line is found all the same)
        A = [{"key": s["key"], "source": "merged"} for s in json.load(open(os.path.join(HERE, f"{GROUP}_shapes.json")))["shapes"]]
    out = []
    for s, a in zip(X.shapes(), A):
        assert s["key"] == a["key"]
        if s["kind"] == "line":
            out.append({"name": s["name"], "key": s["key"], "seed": s["seed"], "sheet": s["sheet"], "source": a["source"]})
    return out



def run(sub, L):
    p = subprocess.run([sys.executable, "-c", CODE % {"here": HERE}], input=json.dumps(L), capture_output=True,
                       text=True, cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
    lines_ = p.stdout.strip().splitlines()
    for ln in lines_[:-1]:
        print(sub, ln, flush=True)
    try:
        return json.loads(lines_[-1])
    except Exception:
        print(sub, "FAILED", p.stderr[-1500:], flush=True)
        return {}


def main():
    from concurrent.futures import ThreadPoolExecutor
    by = {}
    global GROUP
    args = sys.argv[1:]
    if args and args[0] in ("ixoic", "ixitic"):
        GROUP, args = args[0], args[1:]
    only = set(args)
    for s in lines():
        if only and s["name"] not in only:
            continue
        by.setdefault(s["sheet"], []).append(s)
    with ThreadPoolExecutor(4) as ex:
        R = list(ex.map(lambda kv: run(*kv), sorted(by.items())))
    res = {k: v for r in R for k, v in r.items()}
    if only:
        print(json.dumps(res, indent=1)[:3000])
        return
    path = os.path.join(HERE, f"{GROUP}_line_equations.json")
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "lines": res}, open(path, "w"), indent=1)
    md = [f"# Exact equations of the {GROUP} lines", "", "Made by research/hyic_atlas/ixoic_line_equations.py "
          f"(equations in {GROUP}_line_equations.json).", "",
          "| line | sheet | source | status | equations (degrees) | traced (1e-3 steps, + / -) | also on | seed moved |",
          "|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(res.items(), key=lambda kv: (kv[1]["sheet"], kv[0])):
        deg = ", ".join(str(sp_degree(t)) for t in r.get("equations", [])) or "-"
        tr = r.get("traced_steps_1e-3")
        md.append(f"| {n} | {r['sheet']} | {r['source']} | {r['status']} | {deg} | "
                  f"{(str(tr['plus']) + ' / ' + str(tr['minus'])) if tr else '-'} | {len(r.get('also_on', []))} | "
                  f"{('%.1e' % r['seed_moved']) if r.get('seed_moved') is not None else '-'} |")
    open(os.path.join(HERE, f"{GROUP}_line_equations.md"), "w").write("\n".join(md) + "\n")
    print(len(res), "lines")


def sp_degree(t):
    import sympy as sp
    return sp.Poly(sp.sympify(t), *sp.symbols("b1 b2 b3 b4")).total_degree()


if __name__ == "__main__":
    main()
