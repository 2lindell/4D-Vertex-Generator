"""The 'Newly Found Isogonals' page (new_found.html, and the same as Markdown: new_found.md): the shapes the crossing search found (where two or three of a map's
walls cross: crossing_search.py, checked by crossing_verify.py, added by crossing_add.py). Each entry is written from its
polychoron at its seed in its own group (describe.py), in the words of the family pages, with how it was found, its
seed, its most even member (R = longest / shortest edge) and .off files. (New ixoic regions go on the ixoic match
report, pages/ixoic_match.md, not here.)

    python make_new_found_page.py
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_decaic_page import WORD, cells_text, edges_text, faces_text  # noqa: E402
from suggested_names import SUGGESTED  # noqa: E402
from decaic_given_names import given  # noqa: E402

ENV = {"contic_atlas": "CONTIC_SUB", "decaic_atlas": "DECAIC_SUB", "hyic_atlas": "HYIC_SUB", "toxitic_atlas": "CONTIC_SUB"}
DEG = {"region": 3, "wall": 2, "line": 1, "point": 0}


def describe(var, sub, items):
    r = subprocess.run([sys.executable, os.path.join(HERE, "describe.py"), json.dumps(items)], capture_output=True,
                       text=True, env={**os.environ, var: sub}, check=True)
    return json.loads(r.stdout)


def crossing_rows():
    """(group, sheet, env var, row) for every shape crossing_add.py (or add_crossing_shape.py) placed."""
    out = []
    for adir, groups in (("decaic_atlas", {"ki": "kidecaic", "io": "iodecaic"}),
                         ("contic_atlas", {"ki": "kicontic", "io": "iocontic"}),
                         ("hyic_atlas", {f"ix{i}": "ixitic" for i in range(10)})):
        for sub, group in groups.items():
            f = os.path.join(ROOT, adir, f"crossings_{sub}.json")
            if os.path.exists(f):
                out += [(group, sub, ENV[adir], t) for t in json.load(open(f))["shapes"]]
    opt = {}
    for sh in ("ox", "oxb"):
        try:
            opt.update({s["name"]: s for s in json.load(open(os.path.join(ROOT, "toxitic_atlas", f"opt_central_{sh}.json")))["shapes"]})
        except FileNotFoundError:
            pass
        f = os.path.join(ROOT, "toxitic_atlas", f"crossings_{sh}.json")
        if os.path.exists(f):
            for t in json.load(open(f))["shapes"]:
                if "crossing_search" not in t.get("source", ""):
                    continue                    # (oxitic_line_01, found by the community, is on the contic page)
                o = opt.get(t["name"], {})
                seed = glob.glob(os.path.join(ROOT, "new_shapes", "off", "oxitic", f"{t['name']}_*_seed.off"))
                out.append(("oxitic", sh, "CONTIC_SUB", {**t, "degrees": DEG[t["kind"]], "R": o.get("R"),
                                                          "start_R": o.get("start_R"), "opt_off": o.get("opt_off"),
                                                          "seed_off": os.path.relpath(seed[0], ROOT) if seed else None,
                                                          "central_off": o.get("central_off")}))
    return out


def closure_rows():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "hyic_atlas", "closure_ixo*.json"))):
        if f.endswith(".partial.json"):
            continue
        C = json.load(open(f))
        for k, v in C["new_regions"].items():
            out.append((C["sheet"], k, v))
    return out


# (notes on particular shapes, by key)
NOTES = {"aced5db9ac46": " It has the same cells, edges and faces as kidecaic line 25, every count alike, but is a different polychoron: around its pentagonic edge are four 6-faced cells and a snit (line 25: four 6-faced cells and a gyro-trap), and its trigonic edges between two 6-faced cells meet a gyro-trap twice (line 25: once a snit, once a gyro-trap). Three of the map's walls all contain the curve it lies on, so it has one degree of variation, not none; the curve runs across the chamber from near (1/2, 0, 0, 1/2) to near (0.80, 0.01, 0.16, 0.04).",
         "2cc54ef1b550": " Kidecaic line 26 has every count the same, but is a different polychoron: around this one's pentagonic edge are four 6-faced cells and a gyro-trap (line 26: a snit), and of its trigonic edges between two 6-faced cells one meets a snit and one a gyro-trap (line 26: a gyro-trap both times)."}


def fmt_beta(b):
    return ", ".join(f"{x:.6f}" for x in b)


def entry(title, text, data, new=True):
    tag = ' <span class="tag">new</span>' if new else ""
    cls = "shape new" if new else "shape"
    return (f'<article class="{cls}"><h3>{html.escape(title)}{tag}</h3><p>{html.escape(text)}</p>'
            f'<p class="data">{data}</p></article>')


def body(d, group, kind, dof):
    return (f"It has {cells_text(d['cells'])}. It has {d['vertices']} vertices with "
            f"{'an' if str(d['dualcell_faces']).startswith('8') or d['dualcell_faces'] in (11, 18) else 'a'} {d['dualcell_faces']} faced "
            f"dualcell. It has {edges_text(d['edges'])}. Its faces are {faces_text(d['faces'])}. It is isogonal under "
            f"{group} symmetry with " + (f"{WORD[dof]} degree{'s' if dof > 1 else ''} of variation" if dof else
                                         "no variations") + ".")


def name_of(group, t):
    m = re.search(r"(region|wall|line|point)_(\d+)", t["name"])
    return f"{group.capitalize()} {m.group(1)} {int(m.group(2))}" if m else t["name"]


def main():
    X = crossing_rows()
    by = {}
    for i, (group, sub, var, t) in enumerate(X):
        tol = 1e-7 if t["kind"] != "region" else 1e-9
        by.setdefault((var, sub), []).append([f"x{i}", t["seed"], tol])
    Cl = []                                     # (ixoic closure regions: on ixoic_match.md instead)
    for i, (sheet, k, v) in enumerate(Cl):
        by.setdefault(("HYIC_SUB", sheet), []).append([f"c{i}", v["at"], 1e-9])
    D = {}
    for (var, sub), items in by.items():
        D.update(describe(var, sub, items))
        print(var, sub, len(items), flush=True)

    MD = []
    order = ["kidecaic", "iodecaic", "kicontic", "iocontic", "oxitic", "ixitic"]
    secs, nav = [], []
    # (summary table)
    counts = {}
    for group, *_ in X:
        counts[group] = counts.get(group, 0) + 1
    rows = "".join(f"<tr><td>{g.capitalize()}</td><td>{counts[g]}</td><td>"
                   + ", ".join(f"{n} {k}{'s' if n > 1 else ''}" for k in ("wall", "line", "point")
                               for n in [sum(1 for gg, _, _, t in X if gg == g and t["kind"] == k)] if n)
                   + "</td></tr>" for g in order if g in counts)
    rows += (f"<tr><td>Ixoic</td><td>{len(Cl)}</td><td>{len(Cl)} regions</td></tr>" if Cl else "")
    summary = ('<section id="summary"><h2>Summary</h2><div class="tbl"><table><thead><tr><th>symmetry</th><th>new</th>'
               f'<th>kinds</th></tr></thead><tbody>{rows}</tbody></table></div></section>')
    nav.append(f'<a href="#summary">Summary</a>')
    # (crossing search)
    P = ['<section id="crossings"><h2>Where walls cross</h2><p class="lede">The atlas maps took lines and points only on '
         "each chamber's own edges and corners. The crossing search follows every pair (and triple) of a map's walls "
         "where they cross inside the chamber and classifies the polychoron there. Each shape below kept the same "
         "polychoron at three hull tolerances, kept it moving along its locus out to 0.001, changed it 0.00001 off each "
         "of its walls, and has exactly its group's symmetry (no larger one). None is on the accepted lists.</p>"]
    for g in order:
        rows_g = [(i, r) for i, r in enumerate(X) if r[0] == g]
        if not rows_g:
            continue
        P.append(f'<h3 class="sub" id="{g}">{g.capitalize()}</h3>')
        MD.append(f"\n### {g.capitalize()}\n")
        for i, (group, sub, var, t) in sorted(rows_g, key=lambda x: (DEG[x[1][3]["kind"]], x[1][3]["name"])):
            d = D[f"x{i}"]
            n = len(t["locus"])
            how = {"line": f" It was found where {WORD.get(n, n)} of the map's walls cross, as a line along their "
                           "crossing curve.",
                   "point": f" It was found where {WORD.get(n, n)} of the map's walls meet, as an isolated point.",
                   "wall": " It was found where two of the map's walls cross: it fills a stretch of one of them that "
                           "the map had not classified."}[t["kind"]]
            text = body(d, group, t["kind"], t["degrees"]) + how + NOTES.get(t["key"], "")
            sug = SUGGESTED.get(t["name"])
            title = name_of(group, t)
            gv = given(f"{group}_{t['name']}")
            if gv:
                text = f"{gv[1]}: named ({title} in the atlas). {text}" + (f" Name match: {gv[2]}" if gv[2] else "")
                title = gv[0]
            elif sug:
                text = f"{sug[1]}: a suggested name ({title} in the atlas). {text} {sug[2]}"
                title = f"{sug[0]} (suggested)"
            R = t.get("R")
            data = (f"β = ({fmt_beta(t['seed'])}) · {d['n_cells']} cells · {t['kind']} · sheet {sub} · key {t['key']}"
                    + (f"<br>most even: R = {R:.4f}" if isinstance(R, (int, float)) else "")
                    + "".join(f"<br>{html.escape(t[k])}" for k in ("seed_off", "opt_off", "central_off")
                            if t.get(k) and os.path.exists(os.path.join(ROOT, t[k])))
                    + "<br>on " + html.escape(" and ".join(s if len(s) < 90 else s[:87] + "…" for s in t["locus"])) + " = 0")
            P.append(entry(title, text, data))
            MD.append(f"**{title}** ({t['kind']}, sheet {sub}, key `{t['key']}`). {text}\n\n"
                      f"- β = ({fmt_beta(t['seed'])}); {d['n_cells']} cells"
                      + (f"; most even R = {R:.4f}" if isinstance(R, (int, float)) else "") + "\n"
                      + "".join(f"- `{t[k]}`\n" for k in ("seed_off", "opt_off", "central_off")
                                if t.get(k) and os.path.exists(os.path.join(ROOT, t[k])))
                      + "- on " + " and ".join(f"`{x}`" for x in t["locus"]) + " = 0\n")
    P.append("</section>")
    secs.append("\n".join(P))
    nav.append('<a href="#crossings">Where walls cross <span class="n">%d</span></a>' % len(X))
    # (ixoic closure)
    if Cl:
        P = ['<section id="ixoic"><h2>Ixoic regions</h2><p class="lede">Ixoic symmetry is still being mapped. The closure '
             "check walks from every region of a sheet's map towards its walls and labels what lies beyond; regions it "
             "meets that the map and survey never sampled are new. Most are thin slivers near the chamber's edges. Each "
             "was confirmed by sampling around its point (it is still there 0.001 away), and new regions were then "
             "sampled around in turn, which found more. None matches the accepted ixoic list.</p>"]
        for i, (sheet, k, v) in enumerate(Cl):
            d = D[f"c{i}"]
            frm = v.get("from", "")
            how = (f" It was found by sampling around ixoic region {frm[5:11]}." if frm.startswith("near ")
                   else " It was found by the closure check, walking out of a neighbouring region.")
            text = body(d, "ixoic", "region", 3) + how
            data = f"β = ({fmt_beta(v['at'])}) · {d['n_cells']} cells · region · sheet {sheet[3:]} · key {k}"
            P.append(entry(f"Ixoic region {k[:6]}", text, data))
        P.append("</section>")
        secs.append("\n".join(P))
        nav.append('<a href="#ixoic">Ixoic regions <span class="n">%d</span></a>' % len(Cl))
    total = len(X) + len(Cl)
    tpl = open(os.path.join(HERE, "page_template.html")).read()
    tpl = tpl.replace("footer {", ".tbl { overflow-x: auto; } table { border-collapse: collapse; font: 400 .95rem/1.4 var(--serif); "
                      "font-variant-numeric: tabular-nums; } th, td { text-align: left; padding: .35rem .9rem .35rem 0; "
                      "border-bottom: 1px solid var(--rule); } th { font: 500 .85rem/1.2 var(--label); letter-spacing: .06em; "
                      "color: var(--muted); }\nfooter {", 1)
    intro = (f"{total} isogonal polychora found by the crossing search, where two or three of the walls of a symmetry's "
             "map cross inside its chamber. The maps had sampled lines and points only on the chamber's own edges and "
             "corners.")
    note = ("Each entry is written from the polychoron itself at the seed given (β = the map's coordinates on the sheet "
            "named): cells by orbit, vertices with the dualcell's face count, edges by how many faces meet there, faces "
            "by shape. They are also listed on their families' pages.")
    page = (tpl.replace("{{TITLE}}", "Newly Found Isogonals").replace("{{INTRO}}", html.escape(intro))
            .replace("{{NOTE}}", html.escape(note)).replace("{{NAV}}", "\n".join(nav))
            .replace("{{SECTIONS}}", summary + "\n" + "\n".join(secs))
            .replace("{{FOOTER}}", "Made by research/pages/make_new_found_page.py from the crossing search "
                     "(research/crossing_search)."))
    open(os.path.join(HERE, "new_found.html"), "w").write(page)
    table = ["| symmetry | new | kinds |", "|---|---|---|"] + [
        f"| {g.capitalize()} | {counts[g]} | " + ", ".join(
            f"{n} {k}{'s' if n > 1 else ''}" for k in ("wall", "line", "point")
            for n in [sum(1 for gg, _, _, t in X if gg == g and t["kind"] == k)] if n) + " |" for g in order if g in counts]
    md = ["# Newly Found Isogonals", "", intro, "", note, "", "## Summary", ""] + table + [
        "", "## Where walls cross", "",
        "The atlas maps took lines and points only on each chamber's own edges and corners. The crossing search follows "
        "every pair (and triple) of a map's walls where they cross inside the chamber and classifies the polychoron "
        "there. Each shape below kept the same polychoron at three hull tolerances, kept it moving along its locus out "
        "to 0.001, changed it 0.00001 off each of its walls, and has exactly its group's symmetry (no larger one). None "
        "is on the accepted lists."] + MD + ["", "---", "", "Made by research/pages/make_new_found_page.py from the "
                                               "crossing search (research/crossing_search)."]
    open(os.path.join(HERE, "new_found.md"), "w").write("\n".join(md) + "\n")
    print(f"new_found.html: {total} shapes")


if __name__ == "__main__":
    main()
