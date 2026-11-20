"""Suggested names for the ixoic shapes the accepted list does not name (ixoic_suggested.json, and a report
ixoic_suggested.md), built the way the list builds its own names.

The list names an ixoic shape from its cells: the 240-cell orbit gives the stem (snits: Snub Bitetrahedral, tuts:
Bitruncatotetrahedral, octs: Bitetratetrahedral, ratets: Birhombitetratetrahedral, ...), the other cells give the
modifiers, and Small / Medial / Great tell apart shapes whose cells read the same. Each rule below is read off the
list entries named in it. Every shape is first put in the list's words: describe.py's cells (ixoic_describe.py) by
kind and orbit size (regular tets: tets; disphenoids of 720: rhodows; irregular tets of 1440: phydows, of 2880: irr.
tets; trigonal gyroprisms of 480: trigyps, of 960: trigypos; ...). Within one stem and set of modifiers the shapes
are taken in the page's order (regions, walls, lines, points, by atlas number). As in the kidecaic names
(decaic_atlas/kidecaic_names.md), a series of shapes with one stem and modifiers is lettered Alpha, Beta, Gamma, ...
(short names A-, Be-, Ga-, De-, E-, Ze-, He-, The-, I-, ...: Abosid, Bebosid, Adwebosid, Apybosid), and two or three
shapes of a stem the list gives no size word are Small (Medial) and Great (Sawbosid, Gawbosid); past Omega the
letters start again, numbered (Alpha-2, ...). A letter whose short name the list already has (Gamma Gyroprismatic would
be Great's Gagypsobted) is skipped. Square pyramids give Pyramidal and triangular bipyramids Bipyramidal (py
and by, as Apybosid and Abybosid). The ixitic atlas's unnamed shapes are named the same way from the ixitic list
(ixitic_suggested.json); one that is the same polychoron as an ixoic shape takes that shape's name in its Dis120 form.
Shapes whose 240 cell no list entry has are left unnamed.

    python ixoic_suggest.py
"""
from __future__ import annotations

import json
import os
import re
import sys
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


# (square pyramids and triangular bipyramids beside a snub's or a tut's usual cells: Pyramidal and Bipyramidal, written
#  py and by as in the kidecaic names Apybosid and Abybosid, decaic_atlas/kidecaic_names.md)
EXTRA = {"square pyramids": ("Pyramidal", "py"), "triangular bipyramids": ("Bipyramidal", "by")}

# (the series letters of the kidecaic names: Abosid, Bebosid, Gabosid, Debosid, Ebosid, Zebosid, Hebosid, ...)
GREEK = [("Alpha", "a"), ("Beta", "be"), ("Gamma", "ga"), ("Delta", "de"), ("Epsilon", "e"), ("Zeta", "ze"),
         ("Eta", "he"), ("Theta", "the"), ("Iota", "i"), ("Kappa", "ka"), ("Lambda", "la"), ("Mu", "mu"), ("Nu", "nu"),
         ("Xi", "xi"), ("Omicron", "o"), ("Pi", "pi"), ("Rho", "rho"), ("Sigma", "sig"), ("Tau", "ta"),
         ("Upsilon", "u"), ("Phi", "phi"), ("Chi", "chi"), ("Psi", "psi"), ("Omega", "ome")]
SIZES = ("Small", "Medial", "Great")
SIZE_P = {"Small": "s", "Medial": "m", "Great": "g"}


def series(n, same):
    """Words for n shapes of one stem and modifiers, in the page's order: none for one alone; Small / (Medial /) Great
    for two or three when the list uses no size word with this stem (as Sawbosid, Gawbosid); else Greek letters, as
    the kidecaic series (past Omega the letters start again, numbered: Alpha-2, ...). -> [(full word, prefix)]."""
    if not same and n == 1:
        return [("", "")]
    if not same and n <= 3:
        w = [SIZES[0], SIZES[2]] if n == 2 else list(SIZES)
        return [(x, SIZE_P[x]) for x in w]
    out = []
    for i in range(n):
        g, p = GREEK[i % 24]
        c = i // 24 + 1
        out.append((g + (f"-{c}" if c > 1 else ""), p + (str(c) if c > 1 else "")))
    return out


def abbr(word, prefix, mod, base, extra=""):
    """Short name: the series prefix, Pyramidal / Bipyramidal (py, by), the modifier and the stem, in the list's
    spellings (Sidwesbited, Sagpodwsbited, Sagypsobted, Awosbited; Adwebosid, Apybosid, Sawbosid)."""
    stem = STEM[base][1]
    m = MOD[mod][0] if mod else ""
    if prefix and mod == "Antiwedged":
        m = "w"                                 # (Sawbosid, Gawbosid)
    if not prefix and mod == "Gyroprismatic":
        m = "gypri"                             # (not Gypsobted, which is Gyropodial)
    if "gpod" in m and (not prefix or word not in SIZES):
        m = m.replace("gpod", "gypod")
    m = (EXTRA[extra][1] if extra else "") + m
    if m.endswith(("w", "we", "o")) and stem.startswith("sobted"):
        stem = "sbited"                         # (Sidwesbited, Sagpodwsbited, Awosbited)
    if m.endswith("we") and stem == "bituted":
        stem = "btuted"                         # (Sidwabtuted: Disphenoidal before Bitruncato-)
        m = m[:-2] + "a"
    if word in SIZES and m:
        prefix += "i" if m.startswith("d") else "a"     # (Sidwesbited, Sagypsobted, Sabgypsobted)
    elif word in SIZES:
        prefix += "a"
    return (prefix + m + stem).capitalize()


def page_order(name):
    m = re.match(r".*_(region|wall|line|point)_(\d+)$", name or "")
    return ({"region": 0, "wall": 1, "line": 2, "point": 3}[m.group(1)], int(m.group(2))) if m else (9, 0)


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
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "hyic_atlas"))
    import ixitic_offs                          # (the atlas names, ixoic_region_NN ..., give the page's order)
    ixitic_offs.GROUP = "ixoic"
    AN = {r["key"]: r["name"] for r in ixitic_offs.shapes()}
    out, rep = {}, []
    for (mod, base, ex), G in sorted(groups.items(), key=lambda x: (STEM[x[0][1]][0], x[0][0], x[0][2])):
        stem_full = " ".join(x for x in (EXTRA[ex][0] if ex else "", mod, STEM[base][0]) if x)
        same = [f.split()[0] if f.split()[0] in ("Small", "Medial", "Great") else "" for _, f in L
                if re.sub(r"^(Small|Medial|Great) ", "", re.sub(r" \d+$", "", f)).lower() == stem_full.lower()]
        G.sort(key=lambda x: (page_order(AN.get(x[0])), x[0]))
        cand = iter(series(len(G), same) if not same and len(G) <= 3 else series(len(G) + 48, same))
        for k, d, W in G:
            for word, prefix in cand:           # (a letter whose short name the list already has is skipped:
                short = abbr(word, prefix, mod, base, ex)   #  Gamma Gyroprismatic would be Great's Gagypsobted)
                if short.lower() not in taken_short:
                    break
            full = " ".join(x for x in (word, stem_full) if x) + " 240"
            taken_short.add(short.lower())
            cells = ", ".join(f"{w} x{n}" if n > 1 else w for w, n in sorted(W.items()) if n)
            why = (f"Suggested from the list's own names: {STEM[base][0].lower()} for its 240 "
                   f"{'cells of the kind Stetbituted has' if base == 'stet cells' else base}"
                   + (f", {mod.lower()} as in {MOD[mod][1]}" if mod else "")
                   + (f", {EXTRA[ex][0].lower()} for its {ex} (py, by: as the kidecaic Apybosid and Abybosid)"
                      if ex else "")
                   + (f"; {len(G)} unnamed shapes read this way, " + ("lettered Alpha, Beta, ... in the page's order "
                      "(regions, walls, lines, points), as the kidecaic series Abosid, Bebosid, Gabosid, ..."
                      if word not in SIZES else "Small, Medial and Great in the page's order, as Sawbosid and Gawbosid")
                      if len(G) > 1 else "") + f". Cells: {cells}.")
            out[k] = {"short": short, "full": full, "why": why, "vertices": d["vertices"], "kind": d["kind"],
                      "atlas_name": AN.get(k)}
            rep.append(f"| {short} | {full} | {AN.get(k, '')} | {k} | {d['kind']} | {d['vertices']} | {cells} |")
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "suggested": out,
               "not_named": [{"key": k, "vertices": v, "cell_240": b} for k, v, b in skipped]},
              open(os.path.join(HERE, "ixoic_suggested.json"), "w"), indent=1)
    md = ["# Suggested names for unnamed ixoic shapes", "", __doc__.strip().split("\n\n")[1], "",
          "| suggested | full name | atlas name | atlas key | kind | vertices | cells (list words, orbits) |",
          "|---|---|---|---|---|---|---|"] + rep
    if skipped:
        md += ["", "Left unnamed (no list entry has their 240 cell): " +
               "; ".join(f"{k} ({v} vertices, 240 {b})" for k, v, b in skipped) + "."]
    open(os.path.join(HERE, "ixoic_suggested.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(out)} suggested, {len(skipped)} left unnamed")
    for g, G in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {len(G):4}  {' '.join(x for x in (EXTRA[g[2]][0] if g[2] else '', g[0], STEM[g[1]][0]) if x)}")


# Ixitic (pages/hyic_list.txt): the 120-cell orbits give the stem (two of snits: Snub Bitetrahedral, as Sisbethi and
# Gisbethi; snits and tuts: Snub Truncatotetrahedral, as Sestuthi, Mestuthi, Gastuthi; snits and ratets: Snub
# Rhombatotetrahedral, as Misrethi, Giserthi); one orbit of 480 trigypos is part of the stem (Sisbethi), two make it
# Gyropodial (Gypsobtady), hexflanges Antiwedged (Awosbitdy), retrips Prismatic Transitional (Petsobtady), trips with
# rectaps Rectangular Gyroprismatic (Regposbitdy), trips alone Prismatic; square pyramids Pyramidal, triangular
# bipyramids Bipyramidal, as for ixoic.
IXSTEM = {("snits", "snits"): ("Snub Bitetrahedral", "sbethi"), ("snits", "tuts"): ("Snub Truncatotetrahedral", "stuthi"),
          ("ratets", "snits"): ("Snub Rhombatotetrahedral", "srethi")}
IXMOD = {"Gyropodial": "gyp", "Antiwedged": "awo", "Prismatic Transitional": "pet", "Rectangular Gyroprismatic": "regpo",
         "Prismatic": "pri", "": ""}


def ix_words(d):
    W, base = Counter(), []
    for kind, sizes in d["cells"].items():
        for n in sizes:
            if kind in ("irregular tets", "disphenoids", "tetragonal disphenoids"):
                w = "phydows" if n == 720 else "irr. tets"
            else:
                w = RAW.get(kind, kind)
            if n == 120:
                base.append(w)
            W[w] += 1
    return W, tuple(sorted(base))


def ix_modifiers(W):
    if W["retrips"]:
        return "Prismatic Transitional"
    if W["rectaps"]:
        return "Rectangular Gyroprismatic"
    if W["hexflanges"]:
        return "Antiwedged"
    if W["trips"]:
        return "Prismatic"
    if W["trigypos"] >= 2:
        return "Gyropodial"
    return ""


def ixitic_main():
    """Suggested names for the ixitic atlas's unnamed shapes (ixitic_suggested.json, keyed by atlas name)."""
    import subprocess
    import sys
    HY = os.path.join(os.path.dirname(HERE), "hyic_atlas")
    from suggested_names import SUGGESTED
    rows = json.load(open(os.path.join(HY, "opt_central_ixitic.json")))["shapes"]
    for i in range(10):
        try:
            rows += [{**t, "sheet": f"ix{i}"} for t in json.load(open(os.path.join(HY, f"crossings_ix{i}.json")))["shapes"]]
        except FileNotFoundError:
            pass
    rows = [r for r in rows if re.match(r"ixitic_(region|wall|line|point)_\d+$", r["name"]) and r["name"] not in SUGGESTED]
    path = os.path.join(HERE, "ixitic_described.json")
    C = json.load(open(path)) if os.path.exists(path) else {}
    by = {}
    for r in rows:
        if r["name"] not in C:
            by.setdefault(r["sheet"], []).append([r["name"], r["seed"], 1e-7 if r["kind"] == "wall" else 1e-9])
    for sub, items in by.items():
        p = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                           text=True, env={**os.environ, "HYIC_SUB": sub}, check=True)
        C.update(json.loads(p.stdout))
    json.dump(C, open(path, "w"), indent=1)
    L = []
    for line in open(os.path.join(HERE, "hyic_list.txt")):
        c = line.rstrip("\n").split("\t")
        if len(c) > 3 and c[2].strip().lower() == "ixitic" and c[0].strip():
            L.append((c[0].strip(), re.sub(r" (120|Dis120)$", "", c[1].strip())))
    taken = {n.lower() for n, _ in L}
    out = {}
    twins = {x["ixitic_name"]: x["ixoic"] for x in json.load(open(os.path.join(HY, "ixitic_relations.json")))["pairs"]
             if x["relation"] == "same"}
    IXO = json.load(open(os.path.join(HERE, "ixoic_suggested.json")))["suggested"]
    for r in rows:                              # (an ixoic shape's twin: its name in the list's Dis120 form)
        k = twins.get(r["name"])
        if k in IXO:
            g = IXO[k]
            short = re.sub(r"sobted('?)$", r"sobtady\1", re.sub(r"bit(e|a)d('?)$", r"bitdy\2", g["short"]))
            out[r["name"]] = {"short": short,
                              "full": g["full"].replace(" 240", " Dis120"),
                              "why": f"Suggested from its ixoic twin {k} (the same polychoron), suggested {g['short']} "
                                     f"({g['full']}): the list names each ixoic snub bitetrahedral 240 that is also "
                                     "ixitic with a twin Dis120 (Sagypsobted / Sagypsobtady, Gypsobted / Gypsobtady, "
                                     "Awosbited / Awosbitdy, ...)."}
    groups, skipped = {}, []
    for r in rows:
        if r["name"] in out:
            continue
        d = C[r["name"]]
        W, base = ix_words(d)
        if base not in IXSTEM:
            skipped.append(r["name"])
            continue
        mod = ix_modifiers(W)
        ex = next((x for x in EXTRA if W[x]), "")
        groups.setdefault((mod, base, ex), []).append((r["name"], d, W))
    for (mod, base, ex), G in sorted(groups.items()):
        stem_full = " ".join(x for x in (EXTRA[ex][0] if ex else "", mod, IXSTEM[base][0]) if x)
        same = [f.split()[0] if f.split()[0] in SIZES else "" for _, f in L
                if re.sub(r"^(Small|Medial|Great) ", "", f).lower() == stem_full.lower()]
        G.sort(key=lambda x: page_order(x[0]))
        cand = iter(series(len(G), same) if not same and len(G) <= 3 else series(len(G) + 48, same))
        for name, d, W in G:
            for word, prefix in cand:           # (a letter whose short name the list already has is skipped)
                m = "w" if prefix and mod == "Antiwedged" else IXMOD[mod]
                m = (EXTRA[ex][1] if ex else "") + m
                if word in SIZES:
                    prefix += "a" if m else "i"     # (Sisbethi, Gisbethi)
                short = (prefix + m + IXSTEM[base][1]).capitalize()
                if short.lower() not in taken:
                    break
            taken.add(short.lower())
            full = " ".join(x for x in (word, stem_full) if x) + " 120"
            cells = ", ".join(f"{w} x{n}" if n > 1 else w for w, n in sorted(W.items()) if n)
            why = (f"Suggested from the ixitic list's own names: {IXSTEM[base][0].lower()} for its 120 "
                   f"{' and '.join(base)}" + (f", {mod.lower()} for its cells as in the list" if mod else "")
                   + (f", {EXTRA[ex][0].lower()} for its {ex}" if ex else "")
                   + (f"; {len(G)} unnamed shapes read this way, " + ("lettered Alpha, Beta, ... in the page's order, "
                      "as the kidecaic series Abosid, Bebosid, Gabosid, ..." if word not in SIZES else
                      "Small, Medial and Great in the page's order, as Sawbosid and Gawbosid") if len(G) > 1 else "")
                   + f". Cells: {cells}.")
            out[name] = {"short": short, "full": full, "why": why}
    json.dump({"_about": "Suggested names for the ixitic atlas's unnamed shapes (ixoic_suggest.py), keyed by atlas name.",
               "suggested": out, "not_named": skipped}, open(os.path.join(HERE, "ixitic_suggested.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:20} {v['short']:22} {v['full']}")
    print(f"ixitic: {len(out)} suggested, not named: {skipped}")


if __name__ == "__main__":
    main()
    ixitic_main()
