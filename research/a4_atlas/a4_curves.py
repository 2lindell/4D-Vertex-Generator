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
    if "   ? " in r.stdout or "   ? " in r.stderr:
        # (Singular prints an error and goes on: a polynomial it could not read would drop out of every pair)
        bad = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("   ? ")]
        raise RuntimeError("Singular: " + " | ".join(bad[:6]))
    return r.stdout


def poly_str(e):
    return sp.sstr(sp.expand(e)).replace("**", "^")


VARS, EXTRA, NVAR = "b1,b2,b3,b4", "", 4      # (set from exactspec: F4 adds s = sqrt2, s^2 - 2 in every ideal)
KEEP = None
TAGS = None                                   # (per surface, its chambers of the domain (exactspec.Spec.tags), or None)
HEAD = 'ring r=0,(b1,b2,b3,b4),dp; option(redSB); LIB "primdec.lib";\n'


def set_spec(sp_):
    global VARS, EXTRA, NVAR, KEEP, HEAD, L, TAGS
    VARS, NVAR, KEEP, L = sp_.vars, sp_.nvar, sp_.keep_point, sp_.L
    TAGS = None if sp_.tags is None else [set(x) for x in sp_.tags]
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


HARD = set()           # (pairs known not to finish whole: their swap image is used first when it is done)
SYMS = []        # (the domain's symmetries as permutations of b1..b4, each with every surface's image's index)
PAIR_CACHE = None      # (each pair's components, kept as they are found: a rerun redoes only the pairs not done)


def shadow_pair(p, q, i, j, log=print):
    """A pair over Q(sqrt2) too hard to decompose whole: by its shadows. The resultants of p and q in b4 and in b1
    (every curve of theirs is over a factor of one of them; a line through the corner b4 = 1 is over a point of the
    first, so the second) factored over Q(sqrt2); each factor's piece, the ideal (p, q, factor), decomposed (minAssGTZ,
    two minutes), or else its top-dimensional radical part taken (equidimMax, radical): the curve over an irreducible
    factor, one curve when a generic point of the factor lifts to one common point. Each written over Q with s."""
    import subprocess as _sp
    PR = 'ring r=(0,s),(b1,b2,b3,b4),dp; minpoly=s^2-2; option(redSB); LIB "primdec.lib";'
    body = [PR, f"poly p = {p}; poly q = {q};", "list F; int c; int e; poly R;",
            'for(e=1;e<=2;e++){ if(e==1){R=resultant(p,q,b4);} else {R=resultant(p,q,b1);} F=factorize(R); '
            'for(c=1;c<=size(F[1]);c++){ if(deg(F[1][c])>0){ print("F "+string(cleardenom(F[1][c]))); } } }']
    try:
        out = sing("\n".join(body), timeout=600)
    except _sp.TimeoutExpired:
        return None
    facs = sorted({ln[2:].strip() for ln in out.splitlines() if ln.startswith("F ")})
    comps = []
    for f in facs:
        got = None
        for how in ("L = minAssGTZ(I);", "L = list(radical(equidimMax(I)));"):
            body = [PR, f"poly p = {p}; poly q = {q}; poly f = {f};", "ideal I = std(ideal(p,q,f)); list L; int c; int k; ideal G;",
                    f'if(dim(I)==2){{ {how} for(c=1;c<=size(L);c++){{ G=std(L[c]); if(dim(G)==2){{ '
                    'for(k=1;k<=size(G);k++){ G[k]=cleardenom(G[k]); } print("P "+string(G)); } } }']
            try:
                got = [ln[2:].strip() for ln in sing("\n".join(body), timeout=120 if "GTZ" in how else 900).splitlines()
                       if ln.startswith("P ")]
                if "equidim" in how:
                    log(f"  pair ({i}, {j}): the curve over a factor taken as its radical top part (not decomposed)")
                break
            except _sp.TimeoutExpired:
                continue
        if got is None:
            return None
        comps += got
    # (each over Q with s a variable: the same reduced basis a pair decomposed whole gives)
    body = [HEAD, "ideal G; int k;"]
    for c in sorted(set(comps)):
        body.append(f"G = std(ideal({c}{EXTRA})); if(dim(G)==2){{ print(\"C {i} {j} \"+string(G)); }}")
    return sing("\n".join(body), timeout=600)


def curves(polys, log=print):
    """Stage 1: every pair's components, deduplicated; each curve's surfaces."""
    n = len(polys)
    from concurrent.futures import ThreadPoolExecutor

    import subprocess as _sp
    cache_path = PAIR_CACHE
    cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}
    if cache.get("polys") != polys:
        cache = {"polys": polys, "pairs": {}}
    HARD.update(map(tuple, cache.get("hard", [])))

    def swapped(i, j):
        """Pair (i, j)'s components as a symmetry's image of its image pair's, if that one is done (the symmetries
        are involutions: the image pair's components carried by the same permutation)."""
        for perm, of in SYMS:
            a, b = sorted((of[i], of[j]))
            src = cache["pairs"].get(f"{a},{b}")
            if src is not None and (a, b) != (i, j):
                break
        else:
            return None
        img = ",".join(f"b{perm[k] + 1}" for k in range(4)) + (",s" if NVAR == 5 else "")
        body = [HEAD, f"map w = r, {img};", "ideal G; ideal I0;"]
        for ln in src.splitlines():
            if ln.startswith("C "):
                body.append(f'I0 = ideal({ln.split(" ", 3)[3]}); G = std(w(I0)); print("C {i} {j} "+string(G));')
        log(f"  pair ({i}, {j}): the swap image of pair ({a}, {b})")
        return sing("\n".join(body), timeout=600)

    def pair(ij):
        """One pair's components: characteristic sets (minAssChar) within a minute, else Gianni-Trager-Zacharias
        (minAssGTZ) within ten, else by shadows (shadow_pair); a pair none finishes is reported, never left out."""
        i, j = ij
        if f"{i},{j}" in cache["pairs"]:
            return cache["pairs"][f"{i},{j}"], None
        # (a pair whose swap image pair is already done: its curves are those, carried by the swap, exactly)
        if SYMS:
            r = swapped(i, j)
            if r is not None:
                return r, None
        for how, limit in (("minAssChar", 60), ("minAssGTZ", 600)):
            body = [HEAD, f"poly p = {polys[i]};", f"poly q = {polys[j]};", "list L; int c; ideal G;",
                    f'L = {how}(ideal(p,q{EXTRA})); for(c=1;c<=size(L);c++){{G=std(L[c]); '
                    f'if(dim(G)==2){{print("C {i} {j} "+string(G));}}}}']
            try:
                return sing("\n".join(body), timeout=limit), None
            except _sp.TimeoutExpired:
                continue
        if EXTRA:
            r = shadow_pair(polys[i], polys[j], i, j, log)
            if r is not None:
                return r, None
        return "", (i, j)
    jobs = [(i, j) for i in range(n) for j in range(i + 1, n) if TAGS is None or TAGS[i] & TAGS[j]]
    outs, failed = [], []
    with ThreadPoolExecutor(4) as ex:
        for k, ((i, j), (r, bad)) in enumerate(zip(jobs, ex.map(pair, jobs))):
            outs.append(r)
            if bad:
                failed.append(bad)
                log(f"  pair {bad}: no decomposition within the limits")
            else:
                cache["pairs"][f"{i},{j}"] = r
            if (k + 1) % 200 == 0:
                log(f"  pairs: {k + 1}/{len(jobs)}")
                if cache_path:
                    json.dump(cache, open(cache_path, "w"))
    # (a pair not decomposed is the swap image of another: its curves are that pair's, carried by the swap exactly)
    if failed and SYMS:
        still = []
        for i, j in failed:
            r = swapped(i, j)
            if r is None:
                still.append((i, j))
                continue
            cache["pairs"][f"{i},{j}"] = r
            outs.append(r)
        failed = still
    if cache_path:
        cache["hard"] = sorted(set(map(tuple, cache.get("hard", []))) | set(failed))
        json.dump(cache, open(cache_path, "w"))
    if failed:
        raise RuntimeError(f"pairs not decomposed: {failed}")
    out = "\n".join(outs)
    found, ctag = {}, {}
    for line in out.splitlines():
        if not line.startswith("C "):
            continue
        _, i, j, gens = line.split(" ", 3)
        key = ",".join(sorted(g.strip() for g in gens.split(",")))
        found.setdefault(key, set()).update({int(i), int(j)})
        if TAGS is not None:
            ctag.setdefault(key, set()).update(TAGS[int(i)] & TAGS[int(j)])
    keys = sorted(found)
    log(f"  {len(keys)} distinct curves from {len(jobs)} pairs")
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
    return [dict({"ideal": key, "on": on[c]}, **({"tags": sorted(ctag[key])} if TAGS is not None else {}))
            for c, key in enumerate(keys)]


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
        # (with chambers: only the surfaces of the curve's own chambers cross it where it matters)
        jobs = [] if cur.get("crossings_known") else [(k, f"s{k}") for k in range(n) if k not in cur["on"] and
                                                      (TAGS is None or TAGS[k] & set(cur["tags"]))]
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
            if b > a and on[a] & on[b] and (TAGS is None or set(C[a]["tags"]) & set(C[b]["tags"])):
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
    global PAIR_CACHE
    SP = exactspec.get(group)
    set_spec(SP)
    PAIR_CACHE = os.path.join(HERE, f"a4_curves_{group}_pairs.json")
    SYMS.clear()
    for perm in SP.syms:
        B_ = exactspec.B
        of = {}
        for i, e in enumerate(SP.exprs):
            g = e.subs({B_[k]: B_[perm[k]] for k in range(4)}, simultaneous=True)
            of[i] = next(k for k, f in enumerate(SP.exprs) if exactspec.same_surface(g, f))
        SYMS.append((perm, of))
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
