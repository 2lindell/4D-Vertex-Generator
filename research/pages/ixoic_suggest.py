"""Suggested names for the ixoic shapes the accepted list does not name (ixoic_suggested.json, and a report
ixoic_suggested.md), built the way the list builds its own names.

The list names an ixoic shape from its cells: the 240-cell orbit gives the stem (snits: Snub Bitetrahedral, tuts:
Bitruncatotetrahedral, octs: Bitetratetrahedral, ratets: Birhombitetratetrahedral, ...), the other cells give the
modifiers, and Small / Medial / Great tell apart shapes whose cells read the same. Each rule below is read off the
list entries named in it. Every shape is first put in the list's words: describe.py's cells (ixoic_describe.py) by
kind and orbit size (regular tets: tets; disphenoids of 720: rhodows; irregular tets of 1440: phydows, of 2880: irr.
tets; trigonal gyroprisms of 480: trigyps, of 960: trigypos; ...). Within one stem and set of modifiers the shapes
are ordered by their largest edge valence, then by the faces at a vertex (the list's Small, Medial and Great
disphenoidal snub bitetrahedra run 6, 8, 10 and 12 at their largest edge); the size words the list already uses for
that stem are not given again, and where the shapes outnumber the words left they are numbered in that order.
Shapes whose 240 cell no list entry has are left unnamed.

    python ixoic_suggest.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

RAW = {"regular tets": "tets", "squaps / squigyps": "rectaps", "phydows": "hexflanges",
       "11-faced cells (9 vertices: 8 3-gons + 3 4-gons)": "retrips",
       "11-faced cells (9 vertices: 10 3-gons + 1 6-gons)": "gytricus",
       "11-faced cells (12 vertices: 8 3-gons + 3 6-gons)": "ertrips",
       "20-faced cells (24 vertices: 16 3-gons + 4 9-gons)": "etuts",
       "20-faced cells (18 vertices: 16 3-gons + 4 6-gons)": "gyrotuts",
       "14-faced cells (18 vertices: 6 3-gons + 6 4-gons + 2 9-gons)": "enzaps",
       "32-faced cells (24 vertices: 28 3-gons + 4 6-gons)": "stet cells"}


def words(d):
    """describe.py's cells -> Counter of (list word) -> number of orbits, and the 240 cell."""
    W, base = Counter(), None
    for kind, sizes in d["cells"].items():
        for n in sizes:
            if kind in ("irregular tets", "disphenoids", "tetragonal disphenoids"):
                w = {720: "rhodows", 1440: "phydows"}.get(n, "irr. tets")
            elif kind == "trigypos":
                w = "trigyps" if n == 480 else "trigypos"
            elif kind == "ratets" and n == 480:
                w = "ditraps"
            else:
                w = RAW.get(kind, kind)
            if n == 240:
                base = w
            W[w] += 1
    return W, base


# (240 cell -> stem, its abbreviation; from Sobted, Bitutid, Sagypbittid, Gabertatid, Sagybted, Tetsobted, Bamtuted, Stetbituted)
STEM = {"snits": ("Snub Bitetrahedral", "sobted"), "tuts": ("Bitruncatotetrahedral", "bituted"),
        "octs": ("Bitetratetrahedral", "bittid"), "ratets": ("Birhombitetratetrahedral", "bertatid"),
        "tets": ("Gyroprismatic Bitetrahedral", "gybted"), "etuts": ("Tetrahedral Transitional Snub Bitetrahedral",
                                                                    "tetsobted"),
        "gyrotuts": ("Biambotruncatotetrahedral", "bamtuted"),
        "stet cells": ("Tetrahedral Transitional Bitruncatotetrahedral", "tetbituted")}

# (modifier -> abbreviation, the list entries it is read from)
MOD = {"Prismatic Transitional": ("pet", "Petsobted (480 retrips)"),
       "Rectangular Gyroprismatic": ("regpo", "Regposbitad (480 trips, 720 rectaps)"),
       "Cupotransitional": ("cut", "Cutsobted (augytricus, wedges)"),
       "Antiwedged": ("awo", "Awosbited (1440 and 2880 vertices: hexflanges)"),
       "Gyrowedged": ("gywe", "Gywesbitad (1440 gyrowedges)"),
       "Gyropodial Disphenoidal": ("gpodw", "Sagpodwsbited, Megpodwsbited, Gagpodwsbited (trigypos with two orbits of "
                                            "trigyps, rhodows and two or more of phydows)"),
       "Disphenoidal": ("dwe", "Sidwesbited, Midwesbited, Gidwesbited, Sidwabtuted, Gidwabtuted (two orbits of trigyps, "
                               "rhodows, two or more of phydows)"),
       "Gyropodial": ("gyp", "Gypsobted (rhodows, trigypos, no trigyps)"),
       "Bigyroprismatic": ("bgyp", "Sabgypsobted, Gabgypsobted, Sagypbittid (two orbits of trigyps)"),
       "Gyroprismatic": ("gyp", "Sagypsobted, Gagypsobted, Magypsobted, Gagypbittid (one orbit of trigyps)"),
       "Gyrocupolaic": ("gyc", "Gycbituted (gytricus, two orbits of trigyps)"),
       "Gyroprismatic Transitional": ("gyptab", "Gyptabtutid (gytricus with one orbit of trigyps)"),
       "Triangular Transitional": ("tot", "Totbituted (ditraps, trigyps and irr. tets)")}


def modifiers(W, base):
    T, P, F = W["trigyps"], W["trigypos"], W["phydows"]
    R = W["rhodows"]
    if W["retrips"]:
        return "Prismatic Transitional"
    if W["rectaps"]:
        return "Rectangular Gyroprismatic"
    if W["wedges"]:
        return "Cupotransitional"
    if W["hexflanges"]:
        return "Antiwedged"
    if W["gyrowedges"] and base != "ratets":
        return "Gyrowedged"
    if base == "tuts" and W["gytricus"]:
        if T >= 2 and R and F >= 2:
            return "Disphenoidal"
        return "Gyrocupolaic" if T >= 2 else "Gyroprismatic Transitional"
    if base == "tuts" and W["ditraps"] and T == 1:
        return "Triangular Transitional" if W["irr. tets"] else ""
    if T >= 2 and R and F >= 2:
        return "Gyropodial Disphenoidal" if P else "Disphenoidal"
    if P and R and not T:
        return "Gyropodial"
    if T >= 2:
        return "Bigyroprismatic"
    if T == 1:
        return "Gyroprismatic"
    return ""


# (cells no list entry has, beside a snub's or a tut's usual ones: a modifier in the list's manner, not read off it)
EXTRA = {"square pyramids": ("Pyramidal", "pyr"), "triangular bipyramids": ("Bipyramidal", "bipyr")}


def abbr(size, mod, base, extra=""):
    stem = STEM[base][1]
    m = MOD[mod][0] if mod else ""
    m = (EXTRA[extra][1] if extra else "") + m
    if m.endswith(("w", "we", "o")) and stem.startswith("sobted"):
        stem = "sbited"                         # (Sidwesbited, Sagpodwsbited, Awosbited)
    if m.endswith("we") and stem == "bituted":
        stem = "btuted"                         # (Sidwabtuted: Disphenoidal before Bitruncato-)
        m = "dwa"
    s = {"Small": "s", "Medial": "m", "Great": "g", "": ""}[size]
    if s and m:
        s += "i" if m.startswith("d") else "a"
    elif s:
        s += "a"
    return (s + m + stem).capitalize()


def size_key(d):
    v = {int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else
         {"trigonic": 3, "tetragonic": 4, "pentagonic": 5, "hexagonic": 6, "heptagonic": 7, "octagonic": 8,
          "enneagonic": 9, "decagonic": 10, "hendecagonic": 11, "dodecagonic": 12}[k]: n for k, n in d["edges"].items()}
    return max(v), d["dualcell_faces"]


def listed():
    """The list's names and full names (all entries, matched or not)."""
    out = []
    for line in open(os.path.join(HERE, "ixoic_list.txt")):
        c = line.rstrip("\n").split("\t")
        if len(c) > 3 and c[2].strip().lower() == "ixoic" and c[0].strip():
            out.append((c[0].strip(), c[1].strip()))
    return out


def main():
    D = json.load(open(os.path.join(HERE, "ixoic_described.json")))
    named = json.load(open(os.path.join(HERE, "ixoic_names.json")))
    L = listed()
    taken_short = {n.lower() for n, _ in L}
    groups, skipped = {}, []
    for k, d in D.items():
        if k in named:
            continue
        W, base = words(d)
        if base not in STEM:
            skipped.append((k, d["vertices"], base))
            continue
        mod = modifiers(W, base)
        ex = next((x for x in EXTRA if W[x]), "")
        groups.setdefault((mod, base, ex), []).append((k, d, W))
    out, rep = {}, []
    for (mod, base, ex), G in sorted(groups.items(), key=lambda x: (STEM[x[0][1]][0], x[0][0], x[0][2])):
        stem_full = " ".join(x for x in (EXTRA[ex][0] if ex else "", mod, STEM[base][0]) if x)
        same = [f.split()[0] if f.split()[0] in ("Small", "Medial", "Great") else "" for _, f in L
                if re.sub(r"^(Small|Medial|Great) ", "", re.sub(r" \d+$", "", f)).lower() == stem_full.lower()]
        used = set(same)
        free = [w for w in ("Small", "Medial", "Great") if w not in used]
        G.sort(key=lambda x: (size_key(x[1]), x[0]))
        if len(G) == 1 and not same:
            sizes = [""]
        elif len(G) <= len(free):
            idx = [0, len(free) - 1] if len(G) == 2 else list(range(len(G)))
            sizes = [free[i] for i in idx] if len(G) <= 2 or len(G) == len(free) else free[:len(G)]
        else:
            sizes = None
        for i, (k, d, W) in enumerate(G):
            if sizes is not None:
                sz = sizes[i]
                full = " ".join(x for x in (sz, stem_full) if x) + " 240"
                short = abbr(sz, mod, base, ex)
            else:
                full = f"{stem_full} 240 ({i + 1})"
                short = f"{abbr('', mod, base, ex)} {i + 1}"
            if short.lower() in taken_short:
                short += f" {i + 1}"
            taken_short.add(short.lower())
            cells = ", ".join(f"{w} x{n}" if n > 1 else w for w, n in sorted(W.items()) if n)
            why = (f"Suggested from the list's own names: {STEM[base][0].lower()} for its 240 "
                   f"{'cells of the kind Stetbituted has' if base == 'stet cells' else base}"
                   + (f", {mod.lower()} as in {MOD[mod][1]}" if mod else "")
                   + (f", {EXTRA[ex][0].lower()} for its {ex} (no list entry has these; named in the list's manner)"
                      if ex else "")
                   + (f"; {len(G)} unnamed shapes read this way, ordered by their largest edge (here {size_key(d)[0]}) "
                      "and edges at a vertex" if len(G) > 1 else "") + f". Cells: {cells}.")
            out[k] = {"short": short, "full": full, "why": why, "vertices": d["vertices"], "kind": d["kind"]}
            rep.append(f"| {short} | {full} | {k} | {d['kind']} | {d['vertices']} | {cells} |")
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "suggested": out,
               "not_named": [{"key": k, "vertices": v, "cell_240": b} for k, v, b in skipped]},
              open(os.path.join(HERE, "ixoic_suggested.json"), "w"), indent=1)
    md = ["# Suggested names for unnamed ixoic shapes", "", __doc__.strip().split("\n\n")[1], "",
          "| suggested | full name | atlas key | kind | vertices | cells (list words, orbits) |",
          "|---|---|---|---|---|---|"] + rep
    if skipped:
        md += ["", "Left unnamed (no list entry has their 240 cell): " +
               "; ".join(f"{k} ({v} vertices, 240 {b})" for k, v, b in skipped) + "."]
    open(os.path.join(HERE, "ixoic_suggested.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(out)} suggested, {len(skipped)} left unnamed")
    for g, G in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {len(G):4}  {' '.join(x for x in (EXTRA[g[2]][0] if g[2] else '', g[0], STEM[g[1]][0]) if x)}")


if __name__ == "__main__":
    main()
