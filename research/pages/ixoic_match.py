"""Match the accepted ixoic list (ixoic_list.txt) to the atlas's ixoic shapes (ixoic_names.json, and a report).

The list's fifth column gives, for one vertex, how many faces meet at each of its edges: "5568,44,3^6" is edges in
5, 5, 6, 8, 4, 4 and six 3 faces; numbers of two digits are set apart by spaces ("999 18" is 9, 9, 9, 18). The atlas's
edge counts by valence give the same per vertex (count x 2 / vertices). Shapes are matched on vertices and this edge
profile; where several fit, the listed cell orbits (sizes) decide, and what still ties is reported. Atlas shapes: the
merged ixoic list (hyic_atlas/ixoic_shapes.json), the half-turns' fixed lines (hyic_atlas/fixed_lines.json) and
the regions the closure check found (hyic_atlas/closure_<sheet>.json).

    python ixoic_match.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
HY = os.path.join(os.path.dirname(HERE), "hyic_atlas")


def parse_profile(text):
    """'999 18,4^9,3^9' -> Counter({9: 3, 18: 1, 4: 9, 3: 9}); None when unreadable."""
    t = text.replace("?", "").strip()
    m = re.fullmatch(r"\((.*)\)\^(\d+)", t)               # ('(6,12,333)^4')
    if m:
        c = parse_profile(m.group(1))
        return Counter({k: v * int(m.group(2)) for k, v in c.items()}) if c else None
    out = Counter()
    for g in t.split(","):
        g = g.strip()
        if not g:
            continue
        m = re.fullmatch(r"(\d+)\^(\d+)", g)
        if m:
            out[int(m.group(1))] += int(m.group(2))
            continue
        if not re.fullmatch(r"[\d ]+", g):
            return None
        for p in g.split():                             # (no edge has fewer than 3 faces: a 1 starts a two-digit number)
            for n in re.findall(r"1\d|[3-9]", p):
                out[int(n)] += 1
    return out


def entries():
    rows = []
    for line in open(os.path.join(HERE, "ixoic_list.txt")):
        c = line.rstrip("\n").split("\t")
        if len(c) < 5 or c[2].strip().lower() != "ixoic":
            continue
        cells = " ".join(x for x in c[7:15] if x.strip())
        orbits = Counter()
        for n, mult in re.findall(r"(\d+)(?:x(\d))?\s", cells + " "):
            orbits[int(n)] += int(mult or 1)
        rows.append({"name": c[0].strip(), "full": c[1].strip(), "vertices": int(c[3]) if c[3].strip() else None,
                     "profile_text": c[4].strip(), "profile": parse_profile(c[4]), "figure": c[5].strip(),
                     "cells": cells, "orbits": orbits, "related": c[15].strip() if len(c) > 15 else "",
                     "ref": c[19].strip() if len(c) > 19 else ""})
    return rows


def atlas():
    A = []
    for s in json.load(open(os.path.join(HY, "ixoic_shapes.json")))["shapes"]:
        A.append({"key": s["key"], "kind": s["kind"], "vertices": s["vertices"], "edges": s["edges"],
                  "signature": s["signature"], "seed": s["seed"], "sheet": s["sheet"], "source": "merged"})
    try:
        F = json.load(open(os.path.join(HY, "fixed_lines.json")))
    except FileNotFoundError:
        F = {}
    seen = {a["key"] for a in A}
    for sub, o in F.items():
        for k, r in o["shapes"].items():
            if k in seen:
                continue
            seen.add(k)
            sg = r["signature"]
            V = int(sg.split(":")[0])
            e = Counter()
            for n_, k_ in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
                e[int(k_)] += int(n_)
            A.append({"key": k, "kind": r["kind"], "vertices": V, "edges": dict(e), "signature": sg, "seed": r["seed"],
                      "sheet": sub, "source": "fixed line"})
    for sub in sorted(F) or [f"ixo{i}" for i in range(10)]:     # (regions the closure check found: closure_<sheet>.json)
        try:
            Cl = json.load(open(os.path.join(HY, f"closure_{sub}.json")))
        except FileNotFoundError:
            continue
        need = [[k, v["at"]] for k, v in Cl["new_regions"].items() if "signature" not in v]
        if need:
            import subprocess
            import sys
            r = subprocess.run([sys.executable, "-c", SIG % HY], input=json.dumps(need), capture_output=True, text=True,
                               cwd=HY, env={**os.environ, "HYIC_SUB": sub})
            for k, sg in json.loads(r.stdout.strip().splitlines()[-1]).items():
                Cl["new_regions"][k]["signature"] = sg
            json.dump(Cl, open(os.path.join(HY, f"closure_{sub}.json"), "w"), indent=1)
        for k, v in Cl["new_regions"].items():
            if k in seen or "signature" not in v:
                continue
            seen.add(k)
            sg = v["signature"]
            e = Counter()
            for n_, k_ in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
                e[int(k_)] += int(n_)
            A.append({"key": k, "kind": "region", "vertices": int(sg.split(":")[0]), "edges": dict(e), "signature": sg,
                      "seed": v["at"], "sheet": sub, "source": "closure"})
    for a in A:
        a["profile"] = Counter({int(k): v * 2 // a["vertices"] for k, v in a["edges"].items()})
        a["orbits"] = Counter(int(n) for n, _ in re.findall(r"(\d+)x\[(\d+v\d+f)\]", a["signature"].split("|")[0]))
    return A


SIG = r'''
import json, sys
import numpy as np
sys.path.insert(0, %r)
import group as G
out = {}
for key, seed in json.loads(sys.stdin.read()):
    for t in (1e-9, 1e-8):
        try:
            out[key] = G.label(np.asarray(seed, float), t)[0]
            break
        except Exception:
            pass
print(json.dumps(out))
'''


FIG = r'''
import json, sys
import numpy as np
from collections import Counter
sys.path.insert(0, %r)
import group as G
out = {}
for key, seed, tol in json.loads(sys.stdin.read()):
    try:
        V, Vn, idx, faces, cells = G.polytope(np.asarray(seed, float), tol)
    except Exception:
        continue
    c = Counter()
    for cl in cells:
        nb = set()
        for f in cl:
            fv = list(faces[f])
            if 0 in fv:
                i = fv.index(0)
                nb |= {fv[i - 1], fv[(i + 1) %% len(fv)]}
        if nb:
            c[len(nb)] += 1
    out[key] = dict(c)
print(json.dumps(out))
'''


def figures(A):
    """The vertex figure of each atlas shape: for each cell at a vertex, its edges there (cached)."""
    import subprocess
    import sys
    path = os.path.join(HERE, "ixoic_figures.json")
    C = json.load(open(path)) if os.path.exists(path) else {}
    by = {}
    for a in A:
        if a["key"] not in C:
            by.setdefault(a["sheet"], []).append([a["key"], a["seed"], 1e-9 if a["kind"] == "region" else 1e-7])
    for sub, items in by.items():
        r = subprocess.run([sys.executable, "-c", FIG % HY], input=json.dumps(items), capture_output=True, text=True,
                           cwd=HY, env={**os.environ, "HYIC_SUB": sub})
        if r.returncode == 0:
            C.update(json.loads(r.stdout.strip().splitlines()[-1]))
            json.dump(C, open(path, "w"), indent=1)
    for a in A:
        a["figure"] = Counter({int(k): v for k, v in C.get(a["key"], {}).items()})


# Entries whose two columns disagree (see check()), where correcting one column leaves a single atlas shape with the
# listed vertices, cell orbits and the other column. Keyed by name, or by the edge column for an unnamed entry.
CORRECTIONS = {
    "Sagybted": ("002f275609bc", "edge column should read 6^6,4^3,3^6 (the listed 6^6,4^9 sums to 72; its vertex figure, "
                                 "and these cells, need 66). Also the ixoic point of ixitic's sagyptady line"),
    "66668,444,3^6": ("32604d11723e", "vertex figure should read 4^5,3^14 (the listed 4^5,3^13 sums to 59; the edge "
                                      "column needs 62)"),
}


def check(e):
    """The list's two columns must agree: summed over a vertex's edges, the faces at each edge (fifth column) count
    the same cell-edge pairs as the vertex figure's edges per cell (sixth column). None when either is missing."""
    f = parse_profile(e["figure"]) if e["figure"] else None
    if not (f and e["profile"]):
        return None
    a, b = sum(k * n for k, n in e["profile"].items()), sum(k * n for k, n in f.items())
    return None if a == b else f"list columns disagree: edges sum to {a}, vertex figure to {b}"


def fmt(c):
    return ",".join(f"{k}^{n}" if n > 1 else f"{k}" for k, n in sorted(c.items(), reverse=True)) if c else "?"


def main():
    L, A = entries(), atlas()
    figures(A)
    names, rep = {}, []
    used = set()
    for e in L:
        cand = [a for a in A if (e["vertices"] in (None, a["vertices"])) and e["profile"] and a["profile"] == e["profile"]]
        fig = parse_profile(e["figure"]) if e["figure"] else None
        if len(cand) > 1 and fig:
            cand = [a for a in cand if a["figure"] == fig] or cand
        if len(cand) > 1 and e["orbits"]:
            better = [a for a in cand if a["orbits"] == e["orbits"]]
            cand = better or cand
        label = e["name"] or f"(unnamed {e['vertices']} {e['profile_text']})"
        bad = check(e)
        fix = CORRECTIONS.get(e["name"] or e["profile_text"])
        if fix:
            cand = [a for a in A if a["key"] == fix[0]]
        if len(cand) == 1:
            a = cand[0]
            note = "" if not fig or a["figure"] == fig else f"vertex figure {fmt(a['figure'])}, listed {e['figure']}"
            if fix:
                note = "corrected: " + fix[1]
            elif bad:
                note = "; ".join(x for x in (note, bad) if x)
            rep.append(f"| {label} | {e['vertices']} | {e['profile_text']} | {a['key']} | {a['kind']} ({a['source']}) | {note} |")
            if e["name"]:
                names.setdefault(a["key"], []).append({"name": e["name"], "full": e["full"], "related": e["related"]})
            used.add(a["key"])
        elif cand:
            rep.append(f"| {label} | {e['vertices']} | {e['profile_text']} | {', '.join(a['key'] for a in cand)} | "
                       f"{len(cand)} fit | not settled{'; ' + bad if bad else ''} |")
            used |= {a["key"] for a in cand}
        else:
            rep.append(f"| {label} | {e['vertices']} | {e['profile_text']} | — | not found | {bad or ''} |")
    out = ["# Ixoic list matched to the atlas", "", __doc__.strip().split("\n\n")[1], "",
           "| list entry | vertices | edge profile | atlas key | kind | note |", "|---|---|---|---|---|---|"] + rep
    rest = [a for a in A if a["key"] not in used]
    out += ["", f"## Atlas shapes not in the list ({len(rest)})", "", "| key | kind | vertices | edge profile | vertex figure | source |",
            "|---|---|---|---|---|---|"]
    out += [f"| {a['key']} | {a['kind']} | {a['vertices']} | {fmt(a['profile'])} | {fmt(a['figure'])} | {a['source']} |"
            for a in sorted(rest, key=lambda a: (a["vertices"], a["kind"], fmt(a["profile"])))]
    open(os.path.join(HERE, "ixoic_match.md"), "w").write("\n".join(out) + "\n")
    json.dump(names, open(os.path.join(HERE, "ixoic_names.json"), "w"), indent=1)
    print("\n".join(out[:len(rep) + 6]))
    print(f"\n{len(rest)} atlas shapes not in the list")


if __name__ == "__main__":
    main()
