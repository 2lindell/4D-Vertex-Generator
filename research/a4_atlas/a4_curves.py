"""The exact curves and points of a curved A4 map, by computer algebra (Singular), with no grid and no seed search.

  1. curves: for every pair of surfaces (the four mirrors among them), the prime components of the ideal (S, T)
     (Singular's minAssChar, characteristic sets; minAssGTZ stalls on a few pairs): each one an irreducible curve, given by its reduced Groebner basis, so the same curve
     found from several pairs is recognised exactly. Each curve's surfaces: every surface lying in its ideal (its
     polynomial reduces to 0), decided exactly.
  2. points: on each curve, where every surface not containing it crosses it (the ideal of the curve and the surface,
     0-dimensional), and the curve's critical points for a generic linear function L (the 4x4 minors of its Jacobian
     with grad L and (1,1,1,1)), which are its singular points and a point on every closed loop of it. Each solved
     to 30 digits (solve.lib); the real ones in the chamber (beta >= 0, sum 1) kept.

A curve has a real stretch in the chamber only if it has a point here: a stretch either ends on the chamber's
boundary (a crossing with a mirror) or is a closed loop (where L has a minimum).

    python a4_curves.py kidecaic        (a4_curves_<group>.json)
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import a4_pieces as P  # noqa: E402

L = "b1+b4"     # (the linear function whose critical points are added: swap-symmetric, so a curve's and its swap
                #  image's are each other's images; constant on no curve of the map but the chamber edge b1 = b4 = 0)


def sing(script, timeout=None):
    with tempfile.NamedTemporaryFile("w", suffix=".sing", delete=False) as f:
        f.write(script + "\nquit;\n")
        path = f.name
    try:
        r = subprocess.run(["Singular", "-q", path], capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stdout


def poly_str(e):
    return sp.sstr(sp.expand(e)).replace("**", "^")


VARS, EXTRA, NVAR = "b1,b2,b3,b4", "", 4      # (set from exactspec: F4 adds s = sqrt2, s^2 - 2 in every ideal)
KEEP = None
HEAD = 'ring r=0,(b1,b2,b3,b4),dp; option(redSB); LIB "primdec.lib";\n'


def set_spec(sp_):
    global VARS, EXTRA, NVAR, KEEP, HEAD, L
    VARS, NVAR, KEEP, L = sp_.vars, sp_.nvar, sp_.keep_point, sp_.L
    EXTRA = ("," + sp_.extra) if sp_.extra else ""
    HEAD = f'ring r=0,({VARS}),dp; option(redSB); LIB "primdec.lib";\n'


def ring():
    return f"ring r=0,({VARS}),dp; option(redSB);"


def crit_expr():
    """The critical points of L along the curve G, and its singular points: the 4x4 minors of its Jacobian in
    b1..b4 (not s) with grad L and (1,1,1,1)."""
    return ("matrix JT[4][size(G)+2]; for(jj=1;jj<=4;jj++){ for(ii=1;ii<=size(G);ii++){ JT[jj,ii]=diff(G[ii],var(jj)); } "
            f"JT[jj,size(G)+1]=diff({L},var(jj)); JT[jj,size(G)+2]=1; }} ideal CR = minor(JT,4);")


def keep(z):
    if KEEP is not None:
        return KEEP(z)
    return max(abs(v.imag) for v in z[:4]) <= 1e-20 and min(v.real for v in z[:4]) >= -1e-20


def curves(polys, log=print):
    """Stage 1: every pair's components, deduplicated; each curve's surfaces."""
    n = len(polys)
    from concurrent.futures import ThreadPoolExecutor

    def row(i):
        body = [HEAD] + [f"poly s{k} = {p};" for k, p in enumerate(polys)] + ["list L; int c; ideal G;"]
        for j in range(i + 1, n):
            body.append(f'L = minAssChar(ideal(s{i},s{j}{EXTRA})); for(c=1;c<=size(L);c++){{G=std(L[c]); '
                        f'if(dim(G)==2){{print("C {i} {j} "+string(G));}}}}')
        r = sing("\n".join(body), timeout=None)
        log(f"  pairs of surface {i}: done")
        return r
    with ThreadPoolExecutor(4) as ex:
        out = "\n".join(ex.map(row, range(n)))
    found = {}
    for line in out.splitlines():
        if not line.startswith("C "):
            continue
        _, i, j, gens = line.split(" ", 3)
        key = ",".join(sorted(g.strip() for g in gens.split(",")))
        found.setdefault(key, set()).update({int(i), int(j)})
    keys = sorted(found)
    log(f"  {len(keys)} distinct curves from {n * (n - 1) // 2} pairs")
    # (each curve's surfaces, exactly: every polynomial lying in its ideal)
    body = [HEAD] + [f"poly s{i} = {p};" for i, p in enumerate(polys)] + ["ideal G; int k; string on;"]
    for c, key in enumerate(keys):
        body.append(f"G = std(ideal({key})); on = \"\"; for(k=0;k<={n - 1};k++){{ }}")
        tests = " ".join(f'if(reduce(s{k},G)==0){{on=on+" {k}";}}' for k in range(n))
        body[-1] = f'G = std(ideal({key})); on = ""; {tests} print("O {c}"+on);'
    out = sing("\n".join(body))
    on = {}
    for line in out.splitlines():
        if line.startswith("O "):
            parts = line.split()
            on[int(parts[1])] = sorted(int(x) for x in parts[2:])
    return [{"ideal": key, "on": on[c]} for c, key in enumerate(keys)]


NUM = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?")


def parse_c(s):
    """A Singular complex number (0.5, -0.5, 0.5-i*0.5, (i*2), ...) as a Python complex."""
    s = s.strip().replace(" ", "").replace("I", "i")
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    if "i" not in s:
        return complex(float(s), 0.0)
    s = s.replace("i*", "1j*").replace("*i", "*1j")
    s = re.sub(r"(?<![\w.])i(?![\w])", "1j", s)
    return complex(eval(s, {"__builtins__": {}}, {}))


def points(polys, C, log=print, digits=30):
    """Stage 2: each curve's points in the chamber (crossings with every surface not on it; critical points of L)."""
    n = len(polys)
    out_all = []
    for c, cur in enumerate(C):
        body = ['LIB "solve.lib"; LIB "primdec.lib";', ring(),
                f"ideal G = std(ideal({cur['ideal']}));"]
        body += [f"poly s{i} = {p};" for i, p in enumerate(polys)]
        body.append("ideal J; int i; int k; int ii; int jj; def R; list S;")
        lin = all(len(re.findall(r"\^|\*b", g)) == 0 for g in cur["ideal"].split(","))
        jobs = [] if cur.get("crossings_known") else [(k, f"s{k}") for k in range(n) if k not in cur["on"]]
        if not lin:
            body.append(crit_expr())
            jobs.append(("crit", "CR"))
        for tag, expr in jobs:
            ex = expr if tag == "crit" else f"ideal({expr})"
            body.append(
                f'J = std(G + {ex} + ideal(b1+b2+b3+b4-1)); '
                f'if(dim(J)==0 && vdim(J)>0){{ R = solve(J, {digits}, 0, "nodisplay"); setring R; '
                f'for(i=1;i<=size(SOL);i++){{ string t="P {tag}"; for(k=1;k<={NVAR};k++){{t=t+" "+string(SOL[i][k]);}} '
                f'print(t); }} kill R; setring r; def R; }} '
                f'if(dim(J)>0){{ print("D {tag}"); }}')
        out = sing("\n".join(body))
        pts, deg = [], []
        for line in out.splitlines():
            if line.startswith("P "):
                parts = line.split()
                tag = parts[1]
                z = [parse_c(v) for v in parts[2:2 + NVAR]]
                if not keep(z):
                    continue
                x = [v.real for v in z[:4]]
                pts.append({"by": tag, "beta": [repr(v) for v in x], "exact": parts[2:6]})
            elif line.startswith("D "):
                deg.append(line.split()[1])
        if cur.get("crossings_known"):
            pts = [p for p in cur["points"] if p["by"] != "crit"] + pts
            deg = [d for d in cur.get("degenerate", []) if d != "crit"] + deg
        out_all.append(dict({k: v for k, v in cur.items() if k != "crossings_known"}, points=pts, degenerate=deg))
        if c % 20 == 0:
            log(f"  curve {c}/{len(C)}: {len(pts)} points in the chamber")
    return out_all


def meets(C, log=print, digits=30):
    """Where two curves meet: for every two curves sharing a surface, the 0-dimensional ideal of both, solved. A
    vertex of the map is where two of its curves meet; a curve's crossing with a surface does not cover them all (two
    curves on surfaces that both contain one of them, e.g. a chamber edge lying on many surfaces, meet at a point no
    surface crosses the edge at)."""
    from concurrent.futures import ThreadPoolExecutor
    from collections import defaultdict
    live = [c for c, cur in enumerate(C) if cur["points"]]
    on = {c: set(C[c]["on"]) for c in live}
    pairs = defaultdict(list)
    for a in live:
        for b in live:
            if b > a and on[a] & on[b]:
                pairs[a].append(b)

    def job(a):
        body = ['LIB "solve.lib";', ring(),
                f"ideal G = std(ideal({C[a]['ideal']}));", "ideal J; int i; int k; def R;"]
        for b in pairs[a]:
            body.append(
                f'J = std(G + ideal({C[b]["ideal"]}) + ideal(b1+b2+b3+b4-1)); '
                f'if(dim(J)==0 && vdim(J)>0){{ R = solve(J, {digits}, 0, "nodisplay"); setring R; '
                f'for(i=1;i<=size(SOL);i++){{ string t="P {b}"; for(k=1;k<={NVAR};k++){{t=t+" "+string(SOL[i][k]);}} '
                f'print(t); }} kill R; setring r; def R; }}')
        out = sing("\n".join(body)) if pairs[a] else ""
        got = []
        for line in out.splitlines():
            if line.startswith("P "):
                parts = line.split()
                z = [parse_c(v) for v in parts[2:2 + NVAR]]
                if not keep(z):
                    continue
                got.append((int(parts[1]), {"by": f"curve {parts[1]}", "beta": [repr(v.real) for v in z[:4]],
                                            "exact": parts[2:6]}))
        return a, got
    n = 0
    with ThreadPoolExecutor(4) as ex:
        for a, got in ex.map(job, sorted(pairs)):
            for b, p in got:
                C[a]["points"].append(p)
                C[b]["points"].append(dict(p, by=f"curve {a}"))
            n += 1
            if n % 25 == 0:
                log(f"  meets: {n}/{len(pairs)} curves")
    return C


def main(group):
    import functools
    global print
    print = functools.partial(print, flush=True)
    import exactspec
    SP = exactspec.get(group)
    set_spec(SP)
    X = {"surfaces": SP.names}
    polys = [SP.poly_str(e) for e in SP.exprs]
    t = time.time()
    st1 = os.path.join(HERE, f"a4_curves_{group}_stage1.json")
    if os.path.exists(st1) and json.load(open(st1)).get("polys") == polys:
        C = json.load(open(st1))["curves"]
    else:
        C = curves(polys, log=print)
        json.dump({"polys": polys, "curves": C}, open(st1, "w"))
    print(f"  curves: {time.time() - t:.0f}s", flush=True)
    prev = os.path.join(HERE, f"a4_curves_{group}.json")
    if "meets-only" in sys.argv and os.path.exists(prev):
        C = json.load(open(prev))["curves"]
        for cur in C:
            cur["points"] = [p for p in cur["points"] if not p["by"].startswith("curve ")]
        C = meets(C, log=print)
        json.dump({"surfaces": X["surfaces"], "polys": polys, "curves": C}, open(prev, "w"), indent=1)
        print(f"{sum(len(c['points']) for c in C)} points in all, with where curves meet")
        return
    if "crit-only" in sys.argv and os.path.exists(prev):
        old = {c["ideal"]: c for c in json.load(open(prev))["curves"]}
        C = [dict(old[c["ideal"]], crossings_known=True) for c in C]
    C = points(polys, C, log=print)
    C = meets(C, log=print)
    inside = [c for c in C if c["points"]]
    json.dump({"surfaces": X["surfaces"], "polys": polys, "curves": C},
              open(os.path.join(HERE, f"a4_curves_{group}.json"), "w"), indent=1)
    print(f"{len(C)} curves; {len(inside)} with points in the chamber; "
          f"{sum(len(c['points']) for c in C)} points in all ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    for g in [a for a in sys.argv[1:] if not a.startswith(("crit", "meets"))]:
        main(g)
