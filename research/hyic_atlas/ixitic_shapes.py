"""The ixitic (or ixoic) shapes, merged over the ten sheets (ixitic_shapes.json / ixoic_shapes.json, and .md).

Per sheet (group.py: sheet i is the conjugate g_i^-1 E g_i on the H4 chamber): the map's region types and the dense
survey's new regions, the wall shapes (classified on their surfaces), and the strata (the chamber's faces, edges and
corners). Every polytope is keyed by its arrangement key (a polytope and its mirror image or its image under ixoic get
the same key), so the ten sheets merge by key. Kinds: region (3 degrees of freedom), wall (2), line (1), point (0) --
the lowest stratum it lives on; a strata type seen in no sheet's interior is a wall (the mirrors are seams between
sheets). Each gets its symmetry: the elements of H4 keeping its vertex set, and whether the sheet's ixoic (ixitic) also
keeps it -- ixoic is not inside H4.

    python ixitic_shapes.py [ix|ixo]
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

CODE = r'''
import json, sys, numpy as np
sys.path.insert(0, %r)
import group as G
from scipy.spatial import cKDTree
other = "ixitic" if G.SUB.startswith("ixo") else "ixoic"
g = G.sheet_cosets()[G.SHEET]
O = G._cached(other, lambda: None)
O = np.einsum("ji,njk,kl->nil", g, O, g)
H4 = G._cached("full", lambda: G._closure(G.REFL))
out = []
for b, kind in json.loads(sys.stdin.read()):
    b = np.asarray(b, float); b = b / b.sum()
    tol = 1e-9 if kind == "region" else 1e-7
    sg, k = G.label(b, tol)
    n, imp, keeps = 0, 0, None
    if k:
        V, _ = G.vertices(b)
        t = cKDTree(V)
        tol = 1e-7 * np.abs(V).max()

        def count(E):
            # (only the elements sending vertex 0 to a vertex can keep the set: test those)
            ok = t.query(E @ V[0])[0] < tol
            hits = [h for h in E[ok] if t.query(V @ h.T)[0].max() < tol]
            return len(hits), sum(1 for h in hits if np.linalg.det(h) < 0)
        n, imp = count(H4)
        keeps = count(O)[0]
    out.append([sg, k, n, imp, keeps])
print(json.dumps(out))
'''


def gather(sub):
    out = []
    M = json.load(open(os.path.join(HERE, f"snub_map_{sub}.json")))
    for t in M["types"]:
        if t["grid_regions"]:
            out.append((t["example"], "region", "map"))
    try:
        for k, t in json.load(open(os.path.join(HERE, f"survey_{sub}.json")))["new"].items():
            out.append((t["example"], "region", "survey"))
    except FileNotFoundError:
        pass
    try:
        for w in json.load(open(os.path.join(HERE, f"wall_shapes_{sub}.json")))["shapes"]:
            if w["kind"] != "region":
                out.append((w["seed_beta"], "wall", "map wall: " + ", ".join(list(w["surfaces"])[:1])[:60]))
    except FileNotFoundError:
        pass
    for t in json.load(open(os.path.join(HERE, f"strata_{sub}.json")))["types"]:
        st = t["strata"]
        if all(s.startswith("point") for s in st):
            kind = "point"
        elif all(s.startswith("line") or s.startswith("point") for s in st):
            kind = "line"
        else:
            kind = "wall" if t["only_at_a_change"] else "region"
        out.append((t["example"], kind, "strata: " + ", ".join(st)))
    return out


RANK = {"region": 3, "wall": 2, "line": 1, "point": 0}


def main():
    fam = sys.argv[1] if len(sys.argv) > 1 else "ix"
    name = "ixoic" if fam == "ixo" else "ixitic"
    order = 2880 if fam == "ixo" else 1440
    shapes = {}
    for i in range(10):
        sub = f"{fam}{i}"
        if not os.path.exists(os.path.join(HERE, f"strata_{sub}.json")):
            print(f"{sub}: not mapped yet", flush=True)
            continue
        items = gather(sub)
        r = subprocess.run([sys.executable, "-c", CODE % HERE], input=json.dumps([[b, k] for b, k, _ in items]),
                           capture_output=True, text=True, cwd=HERE, env={**os.environ, "HYIC_SUB": sub})
        if r.returncode:
            print(sub, r.stderr[-500:])
            continue
        for (b, kind, src), (sg, k, n, imp, keeps) in zip(items, json.loads(r.stdout)):
            if not k:
                continue
            s = shapes.setdefault(k, {"key": k, "signature": sg, "kind": kind, "seed": b, "sheet": sub, "h4_symmetry": n,
                                      "improper": imp, "other_group_elements_kept": keeps, "sources": set(), "sheets": set()})
            s["sources"].add(f"{sub} {src}")
            s["sheets"].add(sub)
            if RANK[kind] > RANK[s["kind"]]:
                s.update(kind=kind, seed=b, sheet=sub, signature=sg)
        print(f"{sub}: {len(items)} polytopes, {len(shapes)} distinct so far", flush=True)
    interior = {k for k, s in shapes.items() if any(" map" in x or "survey" in x for x in s["sources"])}
    for k, s in shapes.items():
        if s["kind"] == "region" and k not in interior:
            s["kind"] = "wall"
    rows = []
    for s in shapes.values():
        s["sources"], s["sheets"] = sorted(s["sources"]), sorted(s["sheets"])
        s["vertices"] = int(s["signature"].split(":")[0])
        s["cells"] = sum(int(a) for a in re.findall(r"(\d+)x\[", s["signature"].split("|")[0]))
        e = Counter()
        for n_, k_ in re.findall(r"(\d+)x(\d+)", s["signature"].split("edges")[1]):
            e[int(k_)] += int(n_)
        s["edges"] = dict(sorted(e.items()))
        rows.append(s)
    rows.sort(key=lambda s: (-RANK[s["kind"]], s["vertices"], s["cells"]))
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "group": name, "shapes": rows},
              open(os.path.join(HERE, f"{name}_shapes.json"), "w"), indent=1)
    other = "ixoic" if name == "ixitic" else "ixitic"
    L = [f"# {name.capitalize()} shapes (merged over the ten sheets)", "", __doc__.strip().split("\n\n")[1], "",
         f"| kind | vertices | cells | edges by valence | H4 symmetry | {other} elements kept | own symmetry only | sheets | seed (beta, sheet) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        own = s["h4_symmetry"] == (order if name == "ixitic" else 1440) and (s["other_group_elements_kept"] or 0) <= (
            1440 if name == "ixoic" else 0)
        L.append(f"| {s['kind']} | {s['vertices']} | {s['cells']} | {s['edges']} | {s['h4_symmetry']} | "
                 f"{s['other_group_elements_kept']} | {'yes' if own else ''} | {len(s['sheets'])} | "
                 f"{[round(v, 5) for v in s['seed']]} ({s['sheet']}) |")
    cnt = Counter(s["kind"] for s in rows)
    L += ["", f"Totals: {dict(cnt)}."]
    open(os.path.join(HERE, f"{name}_shapes.md"), "w").write("\n".join(L) + "\n")
    print(name, dict(cnt))
    for s in rows:
        print(f"  {s['kind']:6} {s['vertices']:5}v {s['cells']:5}c  H4 sym {s['h4_symmetry']:5}  {other} kept "
              f"{s['other_group_elements_kept']}  sheets {len(s['sheets'])}  {s['signature'].split('|')[0][6:90]}")


if __name__ == "__main__":
    main()
