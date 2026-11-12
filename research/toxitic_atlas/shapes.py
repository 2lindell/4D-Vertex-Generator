"""The toxitic and oxitic shapes, gathered and merged (toxitic_shapes.json, toxitic_shapes.md).

Sources, for each group (oxitic in its two sheets: ox, the oxitic orbits of the chamber's seeds, and oxb, the orbits
under its mirror image, i.e. the mirror images of the oxitic orbits of the reflected seeds):

  regions   the map's grid regions (contic_atlas/snub_map_<sub>.json) and the dense survey's new ones (survey_<sub>.json);
  walls     the map's wall shapes, classified on their exact surfaces (contic_atlas/wall_shapes_<sub>.json);
  strata    the faces, edges and corners of the domain (strata_<sub>.json).

Every polytope is keyed by its arrangement key (computed under its own sheet's group; a polytope and its mirror image
get the same key, so the two oxitic sheets merge by key), given its symmetry inside [[3,4,3]] and its kind: region (3
degrees of freedom), wall (2), line (1) or point (0) -- the lowest stratum it lives on, a region type found only on a
mirror face being a region (for oxitic the mirrors are seams between the sheets, not edges).

    python shapes.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CONTIC = os.path.join(os.path.dirname(HERE), "contic_atlas")
sys.path.insert(0, CONTIC)
import given_names  # noqa: E402


def gather(sub):
    """(signature, seed, kind, source) for every polytope of the sheet."""
    out = []
    M = json.load(open(os.path.join(CONTIC, f"snub_map_{sub}.json")))
    for t in M["types"]:
        if t["grid_regions"]:
            out.append((t["signature"], t["example"], "region", "map"))
    try:
        S = json.load(open(os.path.join(HERE, f"survey_{sub}.json")))
        for k, t in S["new"].items():
            out.append((t["signature"], t["example"], "region", "survey"))
    except FileNotFoundError:
        pass
    try:
        W = json.load(open(os.path.join(CONTIC, f"wall_shapes_{sub}.json")))["shapes"]
        for w in W:
            if w["kind"] != "region":
                out.append((w["signature"], w["seed_beta"], "wall", "map walls: " + ", ".join(list(w["surfaces"])[:2])))
    except FileNotFoundError:
        pass
    try:                                        # (shapes where interior walls cross: crossing_search.py / added by hand)
        for t in json.load(open(os.path.join(HERE, f"crossings_{sub}.json")))["shapes"]:
            out.append((t["signature"], t["seed"], t["kind"], t["source"]))
    except FileNotFoundError:
        pass
    st = json.load(open(os.path.join(HERE, f"strata_{sub}.json")))
    for t in st["types"]:
        strata = t["strata"]
        if any(s.startswith("point") for s in strata) and all(s.startswith("point") for s in strata):
            kind = "point"
        elif all(s.startswith("line") or s.startswith("point") for s in strata):
            kind = "line"
        elif sub == "tox" and not any(s.startswith("fold") for s in strata):
            kind = "wall"                       # (toxitic: the mirror faces are the domain's edges)
        else:
            kind = "wall" if t["only_at_a_change"] else "region"
        out.append((t["signature"], t["example"], kind, "strata: " + ", ".join(strata)))
    return out


KEY_CODE = r"""
import json, sys
import numpy as np
sys.path.insert(0, %r)
sys.path.insert(0, %r)
import group as G
from strata import symmetry
from margin import margin
pts = json.loads(sys.stdin.read())
out = []
for b in pts:
    tol = 1e-7 if b[1] != "region" else 1e-9
    sg, k = G.label(np.asarray(b[0]), tol)
    n, imp = symmetry(np.asarray(b[0]))
    out.append([sg, k, n, imp])
print(json.dumps(out))
"""


def keys(sub, items):
    code = KEY_CODE % (CONTIC, HERE)
    r = subprocess.run([sys.executable, "-c", code], input=json.dumps([[b, k] for _, b, k, _ in items]),
                       capture_output=True, text=True, cwd=CONTIC, env={**os.environ, "CONTIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def edges(sg):
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += int(n)
    return dict(sorted(e.items()))


def cells(sg):
    return sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))


RANK = {"region": 3, "wall": 2, "line": 1, "point": 0}


def merge(subs):
    shapes = {}
    for sub in subs:
        items = gather(sub)
        K = keys(sub, items)
        for (sg0, b, kind, src), (sg, k, n, imp) in zip(items, K):
            if k is None:
                continue
            s = shapes.setdefault(k, {"key": k, "signature": sg, "kind": kind, "seed": b, "sheet": sub,
                                      "symmetry": n, "improper": imp, "sources": set(), "sheets": set()})
            s["sources"].add(src)
            s["sheets"].add(sub)
            if RANK[kind] > RANK[s["kind"]]:
                s.update(kind=kind, seed=b, sheet=sub, signature=sg)
    # (a strata polytope counts as a region only if some sheet's interior has it (map or survey); one seen only on a
    #  mirror face or the fold is the wall there, between the regions on either side)
    interior = {k for k, s in shapes.items() if any(x in ("map", "survey") for x in s["sources"])}
    for k, s in shapes.items():
        if s["kind"] == "region" and k not in interior:
            s["kind"] = "wall"
    out = []
    for s in shapes.values():
        s["name"] = (given_names.TOXITIC if subs == ["tox"] else given_names.OXITIC).get(s["key"])
        s["sources"], s["sheets"] = sorted(s["sources"]), sorted(s["sheets"])
        s["vertices"], s["cells"], s["edges"] = int(s["signature"].split(":")[0]), cells(s["signature"]), edges(s["signature"])
        out.append(s)
    out.sort(key=lambda s: (-RANK[s["kind"]], s["vertices"], s["cells"]))
    return out


def table(rows, group_order, names):
    L = ["| name | kind | vertices | cells | edges by valence | symmetry | seed (beta) | found by |", "|---|---|---|---|---|---|---|---|"]
    for s in rows:
        sym = f"{s['symmetry']}" + (" (exactly the group)" if s["symmetry"] == group_order else "")
        src = "; ".join(x[:60] for x in s["sources"])[:120]
        L.append(f"| {names.get(s['key'], '')} | {s['kind']} | {s['vertices']} | {s['cells']} | {s['edges']} | {sym} | "
                 f"{[round(v, 5) for v in s['seed']]} {('(' + s['sheet'] + ')') if s['sheet'] == 'oxb' else ''} | {src} |")
    return L


def main():
    tox = merge(["tox"])
    ox = merge(["ox", "oxb"])
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "toxitic": tox, "oxitic": ox},
              open(os.path.join(HERE, "toxitic_shapes.json"), "w"), indent=1)
    S = {sub: json.load(open(os.path.join(HERE, f"structure_{sub}.json"))) for sub in ("tox", "ox")}
    L = ["# Toxitic and oxitic shapes", "",
         "Toxitic = [3+,4,3+] = <r1 r2, r3 r4> (288, rotations only); oxitic = [[3+,4,3+]] (576, toxitic with the "
         "24-cell/dual swap). Frame and domain: the contic atlas's (F4 chamber, b1 >= b4 by the swap; betas are distances "
         "to the mirrors, a1, a2 long, a3, a4 short). Made by contic_atlas/snub_map.py (CONTIC_SUB=tox|ox|oxb), "
         "wall_shapes.py, and here strata.py (faces, edges, corners), survey.py (dense edge zones) and shapes.py.", "",
         "## The orbifolds (structure.py): the descriptions confirmed", ""]
    for sub, nm in (("tox", "Toxitic"), ("ox", "Oxitic")):
        L.append(f"**{nm}** (order {S[sub]['order']}):")
        L.append("")
        for p in S[sub]["points"]:
            L.append(f"- {p['points']} points, stabilizer {p['stabilizer']} ({p['order']}), at {p['chamber']}")
        for r in S[sub]["rings"]:
            L.append(f"- {r['rings']} rings, stabilizer {r['stabilizer']}: orbits of {r['generic_orbit']}, along "
                     f"{r['chamber']}")
        L.append("")
    L += ["## Toxitic", "",
          "The normaliser of toxitic is all of [[3,4,3]]: every F4 reflection and the swap map toxitic orbits to "
          "congruent ones, so the half-chamber is the whole shape space and its faces are mirrors (where the polytope "
          "gains a mirror: icopyritic [3,4,3+] or its dual, order 576).", ""]
    L += table(tox, 288, given_names.TOXITIC)
    n3 = sum(1 for s in tox if s["symmetry"] == 288)
    L += ["", f"Shapes with exactly toxitic symmetry: {n3}. The map (N = 32), every face, edge and corner (N = 40), and "
          "a dense survey of every edge and corner zone find one polychoron in the whole open domain.", ""]
    L += ["## Oxitic", "",
          "The normaliser of oxitic is kicontic (the rotations of [[3,4,3]]), not [[3,4,3]]: the F4 reflections conjugate "
          "oxitic to its mirror image (toxitic with phi r1 r4 in place of phi). So a seed and its mirror image give "
          "different oxitic polychora, and the shape space is two copies of the half-chamber (sheet ox: oxitic orbits; "
          "sheet oxb: orbits under the mirror-image group, i.e. mirror images of the oxitic orbits of the reflected "
          "seeds), glued along the mirror faces (there both give the same polychoron, mirrored). The mirrors are no "
          "walls; the walls are curved (quadrics and quartics over Q(sqrt2)) and the plane b2 = b3.", ""]
    L += table(ox, 576, given_names.OXITIC)
    L += ["", "Names: the accepted list (contic_atlas/given_names.py), matched by cells and by the edge valences at a "
          "vertex. Notes:", ""] + [f"- {k}: {v}" for k, v in given_names.NOTES.items() if k != "mabosc_or_gabosc"]
    L += ["- of the list's 23 oxitic names (21 of 576 vertices, bitac, sebtic), 22 are matched; mosbitic is among the "
          "three gitatosbitac_or_mosbitic regions or is not found (above). Atlas oxitic shapes with no name in the list: "
          f"{sum(1 for s in ox if s['symmetry'] == 576 and s['key'] not in given_names.OXITIC)} (new)."]
    cnt = Counter(s["kind"] for s in ox)
    n6 = Counter(s["kind"] for s in ox if s["symmetry"] == 576)
    L += ["", f"Totals: {dict(cnt)}; with exactly oxitic symmetry: {dict(n6)}.", ""]
    open(os.path.join(HERE, "toxitic_shapes.md"), "w").write("\n".join(L) + "\n")
    for nm, rows in (("toxitic", tox), ("oxitic", ox)):
        print(nm, Counter(s["kind"] for s in rows), Counter(s["kind"] for s in rows if s["symmetry"] == (288 if nm == "toxitic" else 576)))


if __name__ == "__main__":
    main()
