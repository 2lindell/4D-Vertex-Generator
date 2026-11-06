"""The 'Pennic and Decaic Isogonals' page with every shape the atlas found (decaic.html).

The named shapes keep the original page's text (decaic_source.txt, the image captions left out). The new kidecaic and
iodecaic shapes (new_shapes/new_shapes.json) are written in the same words from their polychora (describe.py: cells
by orbit and kind, vertices and dualcell, edges by valence, faces by shape), with the regions or walls next to them
on the map (snub_map / wall shapes / closure) and their seed and .off file.

    python make_decaic_page.py
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
DEC = os.path.join(ROOT, "decaic_atlas")
WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve"}
ORDER_CELLS = ["snits", "gratets", "ratets", "tuts", "gyrotuts", "cubes", "traps", "gyro-traps", "trips", "wedges",
               "square pyramids", "triangular bipyramids", "regular tets", "tetragonal disphenoids", "disphenoids", "irregular tets"]


def describe(sub, items):
    r = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                       text=True, env={**os.environ, "DECAIC_SUB": sub}, check=True)
    return json.loads(r.stdout)


def plus(c):
    return "+".join(str(x) for x in sorted(c))


def listing(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]


def cells_text(d):
    keys = sorted(d, key=lambda k: (ORDER_CELLS.index(k) if k in ORDER_CELLS else 5.5, k))
    return listing([f"{plus(d[k])} {k}" for k in keys])


def edges_text(d):
    return listing([f"{plus(v)} {k}" for k, v in d.items()]) + " edges"


def faces_text(d):
    out = []
    for k in sorted(d, key=lambda k: (k != "equits", k)):
        v = d[k]
        if len(v) >= 3 and len(set(v)) == 1:
            out.append(f"{WORD.get(len(v), len(v))} types of {v[0]} {k}")
        else:
            out.append(f"{plus(v)} {k}")
    return listing(out)


def own_named(md, section):
    """name -> seed for the named own shapes in decaic_snub_shapes.md's section."""
    text = open(md).read()
    part = text[text.index(section):]
    part = part[:part.index("\n## ", 5)] if "\n## " in part[5:] else part
    out = {}
    for line in part.splitlines():
        m = re.match(r"\| ([a-z]+) \| (region|wall) \|.*\| \[([^\]]+)\] \|$", line)
        if m:
            out[m.group(1)] = [float(x) for x in m.group(3).split(",")]
    return out


def main():
    rows = json.load(open(os.path.join(ROOT, "new_shapes", "new_shapes.json")))["shapes"]
    md = os.path.join(DEC, "decaic_snub_shapes.md")
    new_entries = {}
    for sub, group, title in (("ki", "kidecaic", "## Kidecaic: own shapes"), ("io", "iodecaic", "## Iodecaic: own shapes")):
        mine = [r for r in rows if r["group"] == group]
        named = own_named(md, title)
        M = json.load(open(os.path.join(DEC, f"snub_map_{sub}.json")))
        items = [[r["name"], r["beta"], 1e-9 if r["kind"] == "region" else 1e-7] for r in mine]
        items += [[f"named:{n}", b, 1e-9] for n, b in named.items()]
        items += [[f"named:{n}", b, 1e-7] for n, b in named.items()]
        items += [[f"type:{t['type']}", t["example"], 1e-9 if t["grid_regions"] else 1e-7] for t in M["types"]]
        D = describe(sub, items)
        disp = {}
        for ident, d in D.items():
            if ident.startswith("named:") and d["key"]:
                disp.setdefault(d["key"], ident[6:].capitalize())
        for r in mine:
            k = D[r["name"]]["key"]
            disp.setdefault(k, display(r["name"]))
        tkey = {int(i[5:]): d["key"] for i, d in D.items() if i.startswith("type:")}
        # (neighbours across the map's walls)
        nb = {}
        for w in M["walls"]:
            wk = tkey.get(w["wall_type"])
            a, b = (tkey.get(s) for s in w["sides"])
            if wk and a and b and wk not in (a, b):
                nb.setdefault(wk, set()).update({a, b})
                nb.setdefault(a, set()).add(("across", b))
                nb.setdefault(b, set()).add(("across", a))
        try:
            C = json.load(open(os.path.join(DEC, f"snub_closure_{sub}.json")))["new_wall_shapes"]
            for wk, w in C.items():
                nb.setdefault(wk, set()).update(w["sides"])
        except (FileNotFoundError, KeyError):
            pass
        out = []
        for r in sorted(mine, key=lambda r: (r["kind"] != "region", r["name"])):
            d = D[r["name"]]
            k = d["key"]
            name = display(r["name"])
            dof = r["degrees"]
            t = (f"It has {cells_text(d['cells'])}. It has {d['vertices']} vertices with a {d['dualcell_faces']} faced "
                 f"dualcell. It has {edges_text(d['edges'])}. Faces are {faces_text(d['faces'])}. It is isogonal under "
                 f"{group} symmetry with {WORD[dof]} degrees of variations")
            if r["kind"] == "wall":
                sides = sorted(disp.get(x, "an unnamed region") for x in nb.get(k, set()) if isinstance(x, str))
                t += " and is a transitional case"
                t += f". It is between {listing(sides)}" if sides else ""
            else:
                across = sorted({disp.get(x[1], "an unnamed region") for x in nb.get(k, set()) if isinstance(x, tuple)})
                if across:
                    t += f". On the map it borders {listing(across)}"
            found = FOUND.get(r["name"])
            t += "." + (f" It was found by {found}." if found else "")
            out.append({"name": name, "kind": r["kind"], "text": t, "beta": r["beta"], "surface": r.get("surface") or "",
                        "off": f"{r['name']}.off", "cells": d["n_cells"]})
        new_entries[group] = out
    write_page(new_entries)


# (shapes not found on the map grid: decaic_snub_shapes.md)
FOUND = {"kidecaic_region_21_120v_680c": "the dense survey of the edge zones",
         "kidecaic_region_22_120v_380c": "the region closure check (walks from every region to its walls)",
         "kidecaic_wall_23_120v_440c": "the region closure check, as a quartic wall between two regions"}


def display(ident):
    g, kind, num = ident.split("_")[:3]
    return f"{g.capitalize()} {kind} {num}"


def write_page(new):
    src = open(os.path.join(HERE, "decaic_source.txt")).read().split("\n## ")
    head = src[0].splitlines()
    total = 46 + sum(len(v) for v in new.values())
    intro = head[1].replace("There are 46 isogonal polychora", f"There are {total} isogonal polychora")
    secs = []
    for block in src[1:]:
        lines = block.split("\n")
        title, body = lines[0].strip(), "\n".join(lines[1:]).strip()
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        secs.append((title, paras[0], paras[1:]))
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    nav, parts = [], []
    for title, sintro, entries in secs:
        sid = title.lower().replace(" ", "-")
        g = {"Kidecaics": "kidecaic", "Iodecaics": "iodecaic"}.get(title)
        extra = new.get(g, [])
        nav.append(f'<a href="#{sid}">{html.escape(title)} <span class="n">{len(entries) + len(extra)}</span></a>')
        P = [f'<section id="{sid}"><h2>{html.escape(title)}</h2><p class="lede">{html.escape(sintro)}</p>']
        for e in entries:
            m = re.match(r"([A-Z][a-z]+) - (.*)", e, re.S)
            P.append(f'<article class="shape"><h3>{html.escape(m.group(1))}</h3><p>{html.escape(m.group(2))}</p></article>')
        if extra:
            P.append(f'<h3 class="sub">New {g} shapes <span class="n">{len(extra)}</span></h3>'
                     f'<p class="lede">Found by the atlas map (curved walls, a dense survey of the edge zones and a region '
                     f'closure check); not yet named. Each is described from its polychoron at the seed given.</p>')
            for e in extra:
                beta = ", ".join(f"{v:.6f}" for v in e["beta"])
                P.append(f'<article class="shape new"><h3>{html.escape(e["name"])} <span class="tag">new</span></h3>'
                         f'<p>{html.escape(e["text"])}</p><p class="data">β = ({beta})'
                         + (f'<br>wall surface: {html.escape(e["surface"])} = 0' if e["surface"] else "")
                         + f'<br>{html.escape(e["off"])}</p></article>')
        P.append("</section>")
        parts.append("\n".join(P))
    page = (tpl.replace("{{INTRO}}", html.escape(intro)).replace("{{NAV}}", "\n".join(nav))
            .replace("{{SECTIONS}}", "\n".join(parts)).replace("{{TOTAL}}", str(total))
            .replace("{{NEW}}", str(sum(len(v) for v in new.values()))))
    open(os.path.join(HERE, "decaic.html"), "w").write(page)
    # (the standalone copy for the public site: assets/decaic_isogonals.html, copied to docs/ by tools/build_pages.py)
    i = page.index('<div class="wrap">')
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + page[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
                  + "<style>body { margin: 0; }</style>\n</head>\n<body>\n" + page[i:] + "</body>\n</html>\n")
    open(os.path.join(os.path.dirname(ROOT), "assets", "decaic_isogonals.html"), "w").write(standalone)
    print(f"decaic.html: {total} shapes ({sum(len(v) for v in new.values())} new)")


if __name__ == "__main__":
    main()
