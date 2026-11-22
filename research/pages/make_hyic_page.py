"""The 'Hyic Isogonals' page (hyic.html, and assets/hyic_isogonals.html for the public site): every shape of the H4
family the atlas found, in the words of the Pennic and Decaic Isogonals page.

Sections: hyic [5,3,3], kihyic [5,3,3]+, ixitic 5[3+,4,3+] and ixoic 5[[3+,4,3+]]. Each entry is
written from its polychoron at its seed in its own group (describe.py with HYIC_SUB; an ixitic shape on the sheet it
was found on): cells by orbit and kind, vertices and the dualcell's face count, edges by valence, faces by shape, its
degrees of variation, and from the accepted list (hyic_list.txt) its full name and the symmetries or shapes it is
related to. Shapes the list does not name are numbered as in the atlas (ixitic_offs.py).

    python make_hyic_page.py
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HY = os.path.join(ROOT, "hyic_atlas")
sys.path.insert(0, HERE)
from make_contic_page import text  # noqa: E402

GROUPS = {"hyic", "kihyic", "ixitic", "ixoic"}
SYMWORDS = ("ic", "swirl", "ski")


def accepted():
    """name -> (full name, the list's related column) from the pasted list."""
    out = {}
    for line in open(os.path.join(HERE, "hyic_list.txt")):
        c = line.rstrip("\n").split("\t")
        if len(c) > 3 and c[2].strip().lower() in GROUPS and c[1].strip():
            out[c[0].strip().lower()] = (c[1].strip(), c[15].strip() if len(c) > 15 else "")
    try:                                        # (the ixoic list, matched to the atlas: ixoic_match.py)
        for k, v in json.load(open(os.path.join(HERE, "ixoic_names.json"))).items():
            for x in v:
                out.setdefault(x["name"].lower(), (x["full"], x["related"]))
                if len(v) > 1:
                    ALSO[(k, x["name"].lower())] = [y["name"] for y in v if y["name"] != x["name"]]
    except FileNotFoundError:
        pass
    return out


def notes():
    """name -> the atlas's matching note (ixitic_shapes.md)."""
    out = {}
    for line in open(os.path.join(HY, "ixitic_shapes.md")):
        m = re.match(r"- (\w+): (.*)", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def describe(sub, items):
    r = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                       text=True, env={**os.environ, "HYIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def display(name):
    m = re.match(r"([a-z]+)_(\d{3,})$", name)
    if m:
        return f"{m.group(1).capitalize()} ({m.group(2)} vertices)"
    m = re.match(r"(\w+?)_or_unnamed_(\d)$", name)
    if m:
        return f"{m.group(1).capitalize()} or unnamed ({m.group(2)})"
    m = re.match(r"(\w+?)_(\d)$", name)
    if m:
        return f"{m.group(1).capitalize()} ({m.group(2)})"
    m = re.match(r"(ixitic|ixoic)_(region|wall|line|point)_(\d+)$", name)
    if m:
        return f"{m.group(1).capitalize()} {m.group(2)} {int(m.group(3))}"
    return name.capitalize()


def related(col):
    if not col:
        return ""
    items = [x.strip() for x in col.split(",") if x.strip()]
    sy = [x for x in items if x.lower().endswith(SYMWORDS[0]) or any(w in x.lower() for w in SYMWORDS[1:])]
    other = [x for x in items if x not in sy]
    t = ""
    if sy:
        t += (f" It is also isogonal under {', '.join(sy[:-1]) + ' and ' + sy[-1] if len(sy) > 1 else sy[0]} "
              f"symmetr{'ies' if len(sy) > 1 else 'y'}.").replace(" Kihyic", " kihyic").replace(" Ixitic", " ixitic")
    if other:
        t += f" It is related to {', '.join(x.lower() for x in other)}."
    return t


def entries():
    """(section title, group, intro, [(name, kind, degrees, seed, sub)])."""
    sym = open(os.path.join(HERE, "hyic_symmetry.txt")).read()
    para = {g: re.search(rf"{g} Symmetry - (.*?)\n", sym).group(1) for g in ("Hyic", "Kihyic", "Ixitic", "Ixoic")}
    S = []
    H = json.load(open(os.path.join(HY, "hyic_shapes.json")))["shapes"]
    S.append(("Hyics", "hyic", f"Hyic symmetry {para['Hyic']} As a reflection group with no diagram symmetry, each face "
              "of its chamber is one shape: the fifteen Wythoffian polychora of the 120-cell family below.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], "full") for r in H]))
    M = json.load(open(os.path.join(HY, "snub_map_ki.json")))
    S.append(("Kihyics", "kihyic", f"Kihyic symmetry {para['Kihyic']} The map of its chamber is one region: snixhi.",
              [("snixhi", "region", 3, t["example"], "ki") for t in M["types"] if t["grid_regions"]]))
    X = json.load(open(os.path.join(HY, "opt_central_ixitic.json")))["shapes"]
    for i in range(10):                         # (shapes found where the map's walls cross: crossing_add.py)
        try:
            for t in json.load(open(os.path.join(HY, f"crossings_ix{i}.json")))["shapes"]:
                X.append({**t, "sheet": f"ix{i}"})
                CROSS[t["name"]] = len(t["locus"])
        except FileNotFoundError:
            pass
    try:                                        # (suggested names for the rest: ixoic_suggest.py; hand ones first)
        for nm_, g in json.load(open(os.path.join(HERE, "ixitic_suggested.json")))["suggested"].items():
            if "listed" in g:                   # (a listed or earlier name, now one of a Greek series)
                SUGGESTED[nm_] = (g["short"], g["full"], g["why"], g["listed"], g["listed_full"])
            else:
                SUGGESTED.setdefault(nm_, (g["short"], g["full"], g["why"]))
    except FileNotFoundError:
        pass
    S.append(("Ixitics", "ixitic", f"Ixitic symmetry {para['Ixitic']} It has index 10 in hyic, so its domain is ten "
              "copies of the hyic chamber (ten sheets), each mapped by curved walls; the mirrors are seams between "
              "sheets. Below: its regions (three degrees of variation), walls (two), lines (one) and tehi.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], r["sheet"]) for r in X]))
    sys.path.insert(0, HY)
    import ixitic_offs                          # (every ixoic shape, named as its .off files are)
    ixitic_offs.GROUP = "ixoic"
    O = ixitic_offs.shapes()
    try:
        SG = json.load(open(os.path.join(HERE, "ixoic_suggested.json")))["suggested"]
    except FileNotFoundError:
        SG = {}
    for r in O:
        KEY[r["name"]] = r["key"]
        if r["key"] in XCROSS:
            CROSS[r["name"]] = XCROSS[r["key"]]
        if r["key"] in SG:
            g = SG[r["key"]]
            SUGGESTED[r["name"]] = ((g["short"], g["full"], g["why"], g["listed"], g["listed_full"]) if "listed" in g
                                    else (g["short"], g["full"], g["why"]))
    S.append(("Ixoics", "ixoic", f"Ixoic symmetry {para['Ixoic']} It is not inside hyic (its elements need √2), so it "
              "has no exact wall forms over the golden field: each of its ten sheets was mapped numerically, and the "
              "walls and lines between the map's shapes were then searched for where the walls meet. Below: its "
              "regions (three degrees of variation), walls (two), lines (one) and points.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], r["sheet"]) for r in O]))
    return S


from suggested_names import RELATED, SUGGESTED  # noqa: E402

CROSS = {}  # (shapes found where the map's walls cross: name -> number of walls)
KEY = {}    # (ixoic: name -> atlas key, for the cached descriptions in ixoic_described.json)
ALSO = {}   # ((key, name) -> the other list names of the same shape)
try:        # (ixoic: the targeted crossing search's confirmed shapes, key -> walls; hyic_atlas/ixoic_crossings.py)
    XCROSS = {s["key"]: len(s["locus"]) for s in json.load(open(os.path.join(HY, "ixoic_crossings.json")))["shapes"]}
except FileNotFoundError:
    XCROSS = {}
try:        # (ixoic: each line's exact equations, name -> entry; hyic_atlas/ixoic_line_equations.py)
    LINEQ = json.load(open(os.path.join(HY, "ixoic_line_equations.json")))["lines"]
except FileNotFoundError:
    LINEQ = {}


def main():
    names, nts = accepted(), notes()
    secs = entries()
    by_sub = {}
    for si, (_, _, _, rows) in enumerate(secs):
        for ri, (name, kind, dof, seed, sub) in enumerate(rows):
            by_sub.setdefault(sub, []).append([f"{si}:{ri}", seed, 1e-7 if kind == "wall" and sub != "full" else 1e-9])
    D = {}
    try:
        X = json.load(open(os.path.join(HERE, "ixoic_described.json")))     # (ixoic_describe.py)
    except FileNotFoundError:
        X = {}
    for si, (_, group, _, rows) in enumerate(secs):
        for ri, row in enumerate(rows):
            if group == "ixoic" and KEY.get(row[0]) in X:
                D[f"{si}:{ri}"] = X[KEY[row[0]]]
    for sub, items in by_sub.items():
        items = [it for it in items if it[0] not in D]
        if items:
            D.update(describe(sub, items))
        print(sub, len(items), flush=True)
    nav, parts, total, n_new = [], [], 0, 0
    order = {"point": 0, "line": 1, "wall": 2, "region": 3}
    for si, (title, group, intro, rows) in enumerate(secs):
        sid = title.lower()
        nav.append(f'<a href="#{sid}">{html.escape(title)} <span class="n">{len(rows)}</span></a>')
        P = [f'<section id="{sid}"><h2>{html.escape(title)}</h2><p class="lede">{html.escape(intro)}</p>']
        base = lambda n: re.sub(r"_\d{3,}$", "", re.sub(r"(_or_unnamed)?_\d$", "", n))
        idx = sorted(range(len(rows)), key=lambda i: (int(rows[i][2]), base(rows[i][0]) not in names, order[rows[i][1]],
                                                      D[f"{si}:{i}"]["vertices"], rows[i][0]))
        for ri in idx:
            name, kind, dof, seed, sub = rows[ri]
            d = D[f"{si}:{ri}"]
            b = base(name)
            full, rel = names.get(b, ("", ""))
            label = display(name)
            unnamed = not full
            body = text(d, group, kind, dof, "") + related(rel)
            if "_or_unnamed" in name:
                body += " Two atlas regions have this name's listed cells; which (if either) it is is not settled."
            elif re.search(r"_\d$", name) and full:
                body += " Two atlas regions fit this name's listed counts exactly; which is which is not settled."
            if name in CROSS and not full and CROSS[name] == 1:
                body += " It was found by the targeted crossing search, on one of the map's walls between its regions."
            elif name in CROSS and not full:
                n = CROSS[name]
                body += (f" It was found by the crossing search, where {('two', 'three')[n - 2] if n in (2, 3) else n} "
                         "of the map's walls cross" + (": a line along their crossing curve." if kind == "line" else
                                                       ": an isolated point." if kind == "point" else "."))
            e = LINEQ.get(name) if group == "ixoic" else None
            if e and e["status"].startswith("no line"):
                body += (" The atlas recorded it as a line, but no curve through its seed keeps the shape: no pair of "
                         "the map's surfaces through it keeps the polytope a step of 1e-5 either way, and along the curve "
                         "they share it changes within 1e-7 both ways, so it is a point (ixoic_line_equations.json).")
            if name == "bitutid":                # (the mirror edge b1 = b4 = 0 on sheet ixo7, scanned: b = (0, s, 1 - s, 0))
                body += (" On sheet 7 it lies where two mirrors meet (b1 = b4 = 0): along that edge, b = (0, s, 1 − s, 0), "
                         "are sagypbittid (0 ≤ s < √2 − 1), the point bamtuted (s = √2 − 1), masbittid (to s = 1/φ), "
                         "bitutid (s = 1/φ), gasbittid (1/φ < s < 1) and bitutid again at the corner (0, 1, 0, 0).")
            elif e and "end" in e["status"]:
                body += (" Its seed is at one end of its line: the exact curve (two of the map's surfaces) keeps the "
                         "shape one way only (ixoic_line_equations.json).")
            elif e and e.get("equations"):
                body += (" Its line is given exactly by two equations, " + ("linear (the fixed line of a half-turn)"
                         if e["source"] == "fixed line" else "two of the map's surfaces") +
                         " (ixoic_line_equations.json).")
            if (KEY.get(name), b) in ALSO:
                body += f" The list also names it {' and '.join(ALSO[KEY[name], b])}: the same shape."
            if name in RELATED:
                body += " " + RELATED[name]
            sug = SUGGESTED.get(name)
            if sug:
                body += " " + sug[2]
            if b in nts:
                body += f" Note: {nts[b][len(b) + 2:] if nts[b].startswith(b) else nts[b]}".rstrip(".") + "."
            if sug and len(sug) > 3:            # (renamed as one of a Greek series: the old name in parentheses)
                head = (f"({sug[1]}: a suggested name, as one of a Greek series; listed as {sug[3]}, {sug[4]})."
                        if full else f"({sug[1]}: a suggested name, as one of a Greek series; suggested earlier as "
                                     f"{sug[3]}, {sug[4]}; {label} in the atlas).")
                label = f"{sug[0]} ({sug[3]})"
            else:
                head = f"({full})." if full else (f"({sug[1]}: a suggested name; {label} in the atlas)." if sug else
                                                  "not yet named.")
                if sug:
                    label = sug[0]
            beta = ", ".join(f"{v:.6f}" for v in seed)
            sheet = f" · sheet {sub.lstrip('ixo')}" if sub.startswith("ix") else ""
            tag = ((' <span class="tag">suggested name</span>' if sug else ' <span class="tag">new</span>') if unnamed
                   else ' <span class="tag">series name</span>' if sug and len(sug) > 3 else "")
            P.append(f'<article class="{"shape new" if unnamed else "shape"}"><h3>{html.escape(label)}{tag}</h3>'
                     f'<p>{html.escape(head)} {html.escape(body)}</p><p class="data">β = ({beta}) · {d["n_cells"]} cells'
                     f' · {kind}{sheet}</p></article>')
            total += 1
            n_new += unnamed
        P.append("</section>")
        parts.append("\n".join(P))
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    intro = (f"There are {total} isogonal polychora with one or more of the hyic family symmetries (H4 = [5,3,3] and the "
             "groups around it) listed below, by the symmetry under which each is isogonal and no larger one.")
    note = (f"Every entry is written from the polychoron itself at the seed given (β = distances to the four H4 mirrors; "
            f"for ixitic and ixoic, on the sheet named): cells by orbit, vertices with the dualcell's face count, edges by how many "
            f"faces meet there, faces by shape. Names and full names are from the accepted list; the {n_new} shapes "
            "marked new or with a suggested name were found by the atlas and are not on the accepted lists; suggested names are built the way the lists build their own (ixoic_suggest.py).")
    page = (tpl.replace("{{TITLE}}", "Hyic Isogonals").replace("{{INTRO}}", html.escape(intro))
            .replace("{{NOTE}}", html.escape(note)).replace("{{NAV}}", "\n".join(nav)).replace("{{SECTIONS}}", "\n".join(parts))
            .replace("{{FOOTER}}", "Made by research/pages/make_hyic_page.py from the hyic atlas (research/hyic_atlas). "
                     "Format of the Pennic and Decaic Isogonals page."))
    open(os.path.join(HERE, "hyic.html"), "w").write(page)
    i = page.index('<div class="wrap">')
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + page[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
                  + "<style>body { margin: 0; }</style>\n</head>\n<body>\n" + page[i:] + "</body>\n</html>\n")
    open(os.path.join(os.path.dirname(ROOT), "assets", "hyic_isogonals.html"), "w").write(standalone)
    print(f"hyic.html: {total} shapes ({n_new} new)")


if __name__ == "__main__":
    main()
