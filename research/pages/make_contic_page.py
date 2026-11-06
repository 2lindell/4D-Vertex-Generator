"""The 'Icoic and Contic Isogonals' page (contic.html, and assets/contic_isogonals.html for the public site): every
shape of the F4 family the atlas found, in the words of the Pennic and Decaic Isogonals page.

Sections: icoic [3,4,3], contic [[3,4,3]], kicontic, iocontic, icopyritic [3,4,3+], toxitic [3+,4,3+], oxitic
[[3+,4,3+]]. Each entry is written from its polychoron at its seed in its own group (describe.py with CONTIC_SUB):
cells by orbit and kind, vertices and the dualcell's face count, edges by valence, faces by shape, its degrees of
variation, and from the accepted lists (contic_names.py) its full name and the other symmetries it is isogonal under.
Shapes the lists do not name are numbered as in the atlas.

    python make_contic_page.py
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
CON = os.path.join(ROOT, "contic_atlas")
TOX = os.path.join(ROOT, "toxitic_atlas")
sys.path.insert(0, HERE)
import contic_names as N  # noqa: E402
from make_decaic_page import WORD, cells_text, edges_text, faces_text  # noqa: E402


def describe(sub, items):
    r = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                       text=True, env={**os.environ, "CONTIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def tol(kind, snub):
    return 1e-7 if (snub and kind == "wall") else 1e-9


def display(name, group):
    m = re.match(r"(\w+?)_or_(\w+?)_(\d)$", name)
    if m:
        return f"{m.group(1).capitalize()} or {m.group(2).capitalize()} ({m.group(3)})"
    m = re.match(r"(?:\w+?_)?(region|wall|line)_(\d+)", name)
    if m:
        return f"{group.capitalize()} {m.group(1)} {m.group(2)}"
    return name.capitalize()


def entries():
    """(section title, group, intro, [(name, kind, degrees, seed, sub, snub)])."""
    S = []
    I = json.load(open(os.path.join(CON, "icoic_shapes.json")))["shapes"]
    S.append(("Icoics", "icoic", N.ICOIC,
              "Icoic symmetry [3,4,3] (order 1152) is the symmetry of the 24-cell. As a reflection group with no swap "
              "kept, each face of its chamber is one shape: the nine below. Sidpii, icoi and gidpii become spic, cont "
              "and gippic when their two sets of 24 cells are equal.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], "ico", False) for r in I]))
    C = json.load(open(os.path.join(CON, "contic_shapes.json")))["shapes"]
    S.append(("Contics", "contic", N.CONTIC,
              "Contic symmetry [[3,4,3]] (order 2304) adds the swap of the 24-cell with its dual. These 27 have it; "
              "most are also isogonal under kicontic and iocontic symmetry.",
              [(r["name"], r["kind"], r["degrees"], r["seed_float"], "full", False) for r in C]))
    for sub, title, group, names, intro in (
            ("ki", "Kicontics", "kicontic", N.KICONTIC,
             "Kicontic symmetry [[3,4,3]]+ (order 1152) is the rotation half of contic. These are isogonal under it "
             "alone and are chiral: the biomnisnub 48s, the snub 48 (snoc) on the line b1 = b4, b2 = b3, and the "
             "regions and walls the map found that the list does not name."),
            ("io", "Iocontics", "iocontic", N.IOCONTIC,
             "Iocontic symmetry [[3,4,3]+] (order 1152) keeps F4's rotations and F4's reflections times the swap. "
             "These are isogonal under it alone.")):
        R = json.load(open(os.path.join(CON, f"optimized_seeds_{sub}.json")))["shapes"]
        S.append((title, group, names, intro,
                  [(r["name"], r["kind"], r["degrees"], r["start"], sub, True) for r in R]))
    icp = [r for f in ("icp", "icpb") for r in json.load(open(os.path.join(TOX, f"opt_central_{f}.json")))["shapes"]]
    S.append(("Icopyritics", "icopyritic", N.ICOPYRITIC,
              "Icopyritic symmetry [3,4,3+] (order 576) keeps the long mirrors and turns the short ones into rotations: "
              "24 pyritic, 24 tettic and 96 protrippic points, square, hexagonal and kihexagonal rings, and 12 spheres "
              "of pyritic symmetry. Pysnapi fills the whole chamber; prissi and pysni lie on its mirrors and sadi on "
              "the kihexagonal ring.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], r["sheet"], True) for r in icp]))
    tox = json.load(open(os.path.join(TOX, "opt_central_tox.json")))["shapes"]
    S.append(("Toxitics", "toxitic", N.TOXITIC,
              "Toxitic symmetry [3+,4,3+] (order 288): two sets of 24 kitettic points, 18 square rings and two sets of "
              "16 kihexagonal rings. One polychoron fills the whole domain: sebtidi. Its mirror faces give prissi and "
              "pysni (icopyritic), its rings sadi, sidpii and icoi.",
              [(r["name"], r["kind"], r["degrees"], r["seed"], "tox", True) for r in tox if r["name"] == "sebtidi"]))
    ox = [r for f in ("ox", "oxb") for r in json.load(open(os.path.join(TOX, f"opt_central_{f}.json")))["shapes"]]
    S.append(("Oxitics", "oxitic", N.OXITIC,
              "Oxitic symmetry [[3+,4,3+]] (order 576) adds the swap to toxitic: 48 kitettic and 144 kibrick points, "
              "octagonal, kihexagonal and square rings. The F4 reflections do not preserve it, so a seed and its "
              "mirror image give different oxitic polychora and the domain is two copies of the chamber; the shapes "
              "below are its regions (three degrees of variation) and walls (two), and bitac and sebtic on its rings. "
              "Uniform cont is isogonal under oxitic too (see Contics).",
              [(r["name"], r["kind"], r["degrees"], r["seed"], r["sheet"], True) for r in ox]))
    return S


SYMS = {"icoic", "kiicoic", "contic", "kicontic", "iocontic", "icopyritic", "toxitic", "oxitic", "tessic", "kitessic",
        "demitessic", "kidemitessic", "tessipyritic"}


def article(n):
    return "an" if str(n).startswith("8") or n in (11, 18) else "a"


def also_text(also):
    if not also:
        return ""
    head, _, rest = also.partition(";")
    if head.startswith("a lower"):
        return f" It is {head}."
    items = [x.strip() for x in head.split(",")]
    if all(x.lower() in SYMS for x in items):
        sy = [x.lower() for x in items]
        t = f" It is also isogonal under {sy[0] if len(sy) == 1 else ', '.join(sy[:-1]) + ' and ' + sy[-1]} symmetr" + \
            ("y." if len(sy) == 1 else "ies.")
    else:
        t = f" It is related to {', '.join(x.lower() for x in items)}."
    if rest.strip():
        t += f" {rest.strip()[0].upper()}{rest.strip()[1:]}."
    return t


def text(d, group, kind, dof, also):
    t = (f"It has {cells_text(d['cells'])}. It has {d['vertices']} vertices with {article(d['dualcell_faces'])} "
         f"{d['dualcell_faces']} faced dualcell. It has {edges_text(d['edges'])}. Its faces are {faces_text(d['faces'])}. "
         f"It is isogonal under {group} symmetry with "
         + ("no variations" if dof == 0 else f"{WORD[dof]} degree{'s' if dof > 1 else ''} of variation"))
    if kind == "wall" and dof == 2:
        t += " and is a transitional case"
    return t + "." + also_text(also)


def main():
    secs = entries()
    by_sub = {}
    for si, (_, _, _, _, rows) in enumerate(secs):
        for ri, (name, kind, dof, seed, sub, snub) in enumerate(rows):
            by_sub.setdefault(sub, []).append([f"{si}:{ri}", seed, tol(kind, snub)])
    D = {}
    for sub, items in by_sub.items():
        D.update(describe(sub, items))
        print(sub, len(items), flush=True)
    nav, parts, total = [], [], 0
    for si, (title, group, names, intro, rows) in enumerate(secs):
        sid = title.lower()
        nav.append(f'<a href="#{sid}">{html.escape(title)} <span class="n">{len(rows)}</span></a>')
        P = [f'<section id="{sid}"><h2>{html.escape(title)}</h2><p class="lede">{html.escape(intro)}</p>']
        order = {"point": 0, "line": 1, "wall": 2, "region": 3}
        idx = sorted(range(len(rows)), key=lambda i: (rows[i][0] not in names and "_or_" not in rows[i][0],
                                                      order.get(rows[i][1], 4), D[f"{si}:{i}"]["vertices"]))
        for ri in idx:
            name, kind, dof, seed, sub, snub = rows[ri]
            d = D[f"{si}:{ri}"]
            base = re.sub(r"_or_\w+$", "", name)
            full, also = names.get(name, names.get(base, ("", "")))
            if "_or_" in name:
                a, b = name.split("_or_")[0], re.sub(r"_\d$", "", name.split("_or_")[1])
                full = f"{names.get(a, ('',))[0]} or {names.get(b, ('',))[0]}".strip(" or")
            label = display(name, group)
            unnamed = not full
            head = f"{label} - ({full})." if full else f"{label} - not yet named."
            body = text(d, group, kind, dof, also)
            if "_or_" in name:
                body += (" Two (or three) atlas shapes fit this name's listed counts; which is which is not settled.")
            beta = ", ".join(f"{v:.6f}" for v in seed)
            cls = "shape new" if unnamed else "shape"
            tag = ' <span class="tag">new</span>' if unnamed else ""
            P.append(f'<article class="{cls}"><h3>{html.escape(label)}{tag}</h3><p>{html.escape(head[len(label) + 3:])} '
                     f'{html.escape(body)}</p><p class="data">β = ({beta}) · {d["n_cells"]} cells · {kind}'
                     + (f" · {sub} sheet" if sub in ("oxb", "icpb") else "") + '</p></article>')
            total += 1
        P.append("</section>")
        parts.append("\n".join(P))
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    n_new = sum(1 for s in secs for r in s[4] if r[0] not in s[2] and "_or_" not in r[0])
    intro = (f"There are {total} isogonal polychora with one or more of the icoic family symmetries (F4 = [3,4,3] and "
             "the groups around it) listed below, by the symmetry under which each is isogonal and no larger one.")
    note = (f"Every entry is written from the polychoron itself at the seed given (β = distances to the four F4 mirrors, "
            f"b1 and b2 long, b3 and b4 short): cells by orbit, vertices with the dualcell's face count, edges by how "
            f"many faces meet there, faces by shape. Names and full names are from the accepted lists; the {n_new} "
            "shapes marked new were found by the atlas and are not yet named.")
    page = (tpl.replace("{{TITLE}}", "Icoic and Contic Isogonals").replace("{{INTRO}}", html.escape(intro))
            .replace("{{NOTE}}", html.escape(note)).replace("{{NAV}}", "\n".join(nav)).replace("{{SECTIONS}}", "\n".join(parts))
            .replace("{{FOOTER}}", "Made by research/pages/make_contic_page.py from the contic and toxitic atlases "
                     "(research/contic_atlas, research/toxitic_atlas). Format of the Pennic and Decaic Isogonals page."))
    open(os.path.join(HERE, "contic.html"), "w").write(page)
    i = page.index('<div class="wrap">')
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + page[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
                  + "<style>body { margin: 0; }</style>\n</head>\n<body>\n" + page[i:] + "</body>\n</html>\n")
    open(os.path.join(os.path.dirname(ROOT), "assets", "contic_isogonals.html"), "w").write(standalone)
    print(f"contic.html: {total} shapes ({n_new} new)")


if __name__ == "__main__":
    main()
