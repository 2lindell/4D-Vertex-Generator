"""The 'Tessic Isogonals' page (tessic.html, and assets/tessic_isogonals.html for the public site).

The named shapes keep the original page's text (tessic_source.txt, the image captions left out). The tessic atlas
(research/tessic_atlas: tessic, kitessic, tessipyritic, demitessic, kidemitessic) found these 18 and no others, so no
new entries are added; each entry gets the atlas's most even seed (R = longest / shortest edge) and its .off file, and
the atlas's notes against the list (tessic_subgroups.md) are added where they apply.

    python make_tessic_page.py
"""
from __future__ import annotations

import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TES = os.path.join(ROOT, "tessic_atlas")

# (the group each shape is listed under, and so whose atlas seed is shown)
GROUP = {"snet": ("ki", "kitessic"), "pysnet": ("pyr", "tessipyritic"), "sadi": ("pyr", "tessipyritic")}
NOTES = {
    "tah": "The atlas finds tah isogonal under tessipyritic symmetry too: its tessic seed's stabilizer holds an odd "
           "permutation, so the tessipyritic orbit is the whole tessic one.",
    "thex": "Under demitessic symmetry the atlas finds thex with one degree of variance (on a line of the D4 chamber), "
            "not two.",
    "tico": "Tico cannot be kidemitessic (that group's order, 96, is less than its 192 vertices); under tessipyritic "
            "symmetry its tessic seed gives sadi. Its cells are truncated octahedra (toes) and cubes.",
    "snet": "The atlas counts its edges as 576 trigonic and 288 tetragonic (its verf -- two pentagons, two trapezoids "
            "and four triangles -- has six three-face and three four-face corners).",
}


def seeds():
    out = {}
    for sub, f in (("full", "optimized_seeds.json"), ("ki", "optimized_seeds_ki.json"), ("pyr", "optimized_seeds_pyr.json")):
        for s in json.load(open(os.path.join(TES, f)))["shapes"]:
            out[(sub, s["name"])] = s
    return out


def main():
    S = seeds()
    raw = open(os.path.join(HERE, "tessic_source.txt")).read().split("\n")
    head = raw[1]
    secs, cur = [], None
    for line in raw[2:]:
        t = line.strip()
        if not t or t.startswith("Back to") or re.search(r"(Net|Slices|Sections|Verf)$", t):
            continue
        m = re.match(r"([A-Z][a-z]+) - (.*)", t)
        if m:
            cur[2].append((m.group(1), m.group(2)))
        elif cur is None or cur[1] is not None or len(t) < 30:
            cur = [t, None, []]
            secs.append(cur)
        else:
            cur[1] = t
    total = sum(len(s[2]) for s in secs)
    nav, parts = [], []
    for title, sintro, entries in secs:
        sid = title.lower().replace(" ", "-")
        nav.append(f'<a href="#{sid}">{html.escape(title)} <span class="n">{len(entries)}</span></a>')
        P = [f'<section id="{sid}"><h2>{html.escape(title)}</h2><p class="lede">{html.escape(sintro or "")}</p>']
        for name, body in entries:
            n = name.lower()
            sub, group = GROUP.get(n, ("full", "tessic"))
            s = S.get((sub, n))
            note = f' <span class="tag">atlas</span> {html.escape(NOTES[n])}' if n in NOTES else ""
            data = ""
            if s:
                b = [0.0 if abs(v) < 1e-12 else v for v in (s.get("off_seed") or s["best"])]
                R = s.get("off_R", s.get("R"))
                data = (f'<p class="data">most even {group}: β = ({", ".join(f"{v:.6f}" for v in b)}) · R = {R:.6f}'
                        f'<br>{html.escape(s.get("off", ""))}</p>')
            P.append(f'<article class="shape"><h3>{html.escape(name)}</h3><p>{html.escape(body)}{note}</p>{data}'
                     '</article>')
        P.append("</section>")
        parts.append("\n".join(P))
    note = ("The tessic atlas mapped all five tessic family groups (tessic, kitessic, tessipyritic, demitessic and "
            "kidemitessic: plane arrangements, curved-wall maps, dense surveys of the edge zones and region closure "
            "checks) and found exactly these 18 polychora, no others. The original descriptions are kept; each entry "
            "adds the atlas's most even seed (β = distances to the four B4 mirrors, b1 the hexadecachoron's node and b4 "
            "the tesseract's; R = longest / shortest edge) and its .off file, and notes marked atlas where it differs "
            "from or adds to the list.")
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    page = (tpl.replace("{{TITLE}}", "Tessic Isogonals").replace("{{INTRO}}", html.escape(head))
            .replace("{{NOTE}}", html.escape(note)).replace("{{NAV}}", "\n".join(nav)).replace("{{SECTIONS}}", "\n".join(parts))
            .replace("{{FOOTER}}", "Made by research/pages/make_tessic_page.py from the tessic atlas (research/tessic_atlas). "
                     "Original text: the Tessic Isogonals page."))
    open(os.path.join(HERE, "tessic.html"), "w").write(page)
    i = page.index('<div class="wrap">')
    standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                  '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                  + page[:i].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
                  + "<style>body { margin: 0; }</style>\n</head>\n<body>\n" + page[i:] + "</body>\n</html>\n")
    open(os.path.join(os.path.dirname(ROOT), "assets", "tessic_isogonals.html"), "w").write(standalone)
    print(f"tessic.html: {total} shapes in {len(secs)} sections")


if __name__ == "__main__":
    main()
