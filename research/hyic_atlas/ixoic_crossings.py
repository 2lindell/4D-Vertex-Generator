"""The ixoic shapes the targeted search found and crossing_verify confirmed, not already in the atlas
(ixoic_crossings.json, and .md).

From crossing_search/verified_targeted.json: the candidates confirmed on some sheet (stable label, holding along their
locus, changing off it), by key over the ten sheets; a key confirmed on several sheets is kept at the highest kind (a
wall over a line). Keys already among the merged shapes (ixoic_shapes.json), the half-turns' fixed lines
(fixed_lines.json) or the closure check's regions (closure_<sheet>.json) are left out. Each is given its full symmetry
as ixitic_shapes.py does (every isometry keeping its vertex set, and how many elements of the sheet's ixitic keep it);
only shapes whose symmetry is exactly ixoic are kept (2880, no improper elements, 1440 of ixitic, as every merged
shape), the rest are listed apart. Shapes with the same vertex count, edges, cell orbits and vertex figure are listed
as possible repeats.

    python ixoic_crossings.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ixitic_shapes import CODE  # noqa: E402

VER = os.path.join(os.path.dirname(HERE), "crossing_search", "verified_targeted.json")
RANK = {"region": 3, "wall": 2, "line": 1, "point": 0}

FIG = r'''
import json, sys
import numpy as np
from collections import Counter
sys.path.insert(0, %r)
import group as G
out = {}
for key, seed, tol in json.loads(sys.stdin.read()):
    V, Vn, idx, faces, cells = G.polytope(np.asarray(seed, float), tol)
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


def known():
    K = {s["key"] for s in json.load(open(os.path.join(HERE, "ixoic_shapes.json")))["shapes"]}
    try:
        for o in json.load(open(os.path.join(HERE, "fixed_lines.json"))).values():
            K |= set(o["shapes"])
    except FileNotFoundError:
        pass
    for f in glob.glob(os.path.join(HERE, "closure_ixo*.json")):
        K |= set(json.load(open(f))["new_regions"])
    return K


def run(code, sub, items):
    r = subprocess.run([sys.executable, "-c", code % HERE], input=json.dumps(items), capture_output=True, text=True,
                       cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
    if r.returncode:
        raise SystemExit(f"{sub}: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    V = json.load(open(VER))
    K = known()
    verdicts, found = Counter(), {}
    for name, o in V.items():
        sub = name.split("_", 1)[1]
        for k, r in o.items():
            verdicts[r["verdict"]] += 1
            if r["verdict"] != "confirmed":
                continue
            kind = {2: "wall", 1: "line", 0: "point"}[r["degrees"]]
            s = found.setdefault(k, {"key": k, "kind": kind, "seed": r["seed"], "sheet": sub, "locus": r["locus"],
                                     "sources": set(), "sheets": set()})
            s["sources"].add(f"{sub} targeted {kind}")
            s["sheets"].add(sub)
            if RANK[kind] > RANK[s["kind"]]:
                s.update(kind=kind, seed=r["seed"], sheet=sub, locus=r["locus"])
    old = sorted(k for k in found if k in K)
    new = {k: s for k, s in found.items() if k not in K}
    print(f"verdicts {dict(verdicts)}; confirmed keys {len(found)}, already in the atlas {len(old)}, new {len(new)}",
          flush=True)
    by = {}
    for s in new.values():
        by.setdefault(s["sheet"], []).append(s)
    for sub, L in sorted(by.items()):
        res = run(CODE, sub, [[s["seed"], s["kind"]] for s in L])
        fig = run(FIG, sub, [[s["key"], s["seed"], 1e-7] for s in L])
        for s, (sg, k, n, imp, keeps) in zip(L, res):
            s.update(signature=sg, label_at_seed=k, symmetry=n, improper=imp, other_group_elements_kept=keeps,
                     figure=fig.get(s["key"], {}))
        print(f"{sub}: {len(L)} done", flush=True)
    rows, other = [], []
    for s in new.values():
        s["sources"], s["sheets"] = sorted(s["sources"]), sorted(s["sheets"])
        sg = s["signature"]
        s["vertices"] = int(sg.split(":")[0])
        s["cells"] = sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))
        e = Counter()
        for n_, k_ in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
            e[int(k_)] += int(n_)
        s["edges"] = dict(sorted(e.items()))
        exact = s["symmetry"] == 2880 and s["improper"] == 0 and s["other_group_elements_kept"] == 1440 and \
            s["label_at_seed"] == s["key"]
        (rows if exact else other).append(s)
    rows.sort(key=lambda s: (-RANK[s["kind"]], s["vertices"], s["cells"], s["key"]))
    other.sort(key=lambda s: (s["vertices"], s["key"]))
    prof = {}
    for s in rows:
        orb = sorted(int(n) for n in re.findall(r"(\d+)x\[", s["signature"].split("|")[0]))
        prof.setdefault((s["vertices"], json.dumps(s["edges"]), json.dumps(orb),
                         json.dumps(sorted(s["figure"].items()))), []).append(s["key"])
    same = [v for v in prof.values() if len(v) > 1]
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "verdicts": dict(verdicts), "already_known": old,
               "shapes": rows, "not_exactly_ixoic": other, "same_counts": same},
              open(os.path.join(HERE, "ixoic_crossings.json"), "w"), indent=1)
    L = ["# Ixoic shapes from the targeted crossing search", "", __doc__.strip().split("\n\n")[1], "",
         f"Verdicts over the ten sheets: {dict(verdicts)}. Confirmed keys: {len(found)}; already in the atlas: "
         f"{len(old)}; new: {len(new)}, of which exactly ixoic: {len(rows)}.", "",
         "| key | kind | vertices | cells | edges by valence | vertex figure | sheets | seed (beta, sheet) |",
         "|---|---|---|---|---|---|---|---|"]
    for s in rows:
        L.append(f"| {s['key']} | {s['kind']} | {s['vertices']} | {s['cells']} | {s['edges']} | "
                 f"{dict(sorted(s['figure'].items(), reverse=True))} | {len(s['sheets'])} | "
                 f"{[round(v, 6) for v in s['seed']]} ({s['sheet']}) |")
    if other:
        L += ["", f"## Not exactly ixoic ({len(other)})", "", "| key | kind | vertices | symmetry | improper | ixitic kept "
              "| label at seed |", "|---|---|---|---|---|---|---|"]
        L += [f"| {s['key']} | {s['kind']} | {s['vertices']} | {s['symmetry']} | {s['improper']} | "
              f"{s['other_group_elements_kept']} | {s['label_at_seed']} |" for s in other]
    L += ["", f"## Same counts ({len(same)} groups)", "",
          "Shapes with the same vertices, edges, cell orbits and vertex figure (possible repeats, told apart only by "
          "their keys):", ""] + [f"- {', '.join(v)}" for v in same]
    open(os.path.join(HERE, "ixoic_crossings.md"), "w").write("\n".join(L) + "\n")
    print(f"exactly ixoic {len(rows)} {dict(Counter(s['kind'] for s in rows))}; other {len(other)}; "
          f"same-count groups {len(same)}")


if __name__ == "__main__":
    main()
