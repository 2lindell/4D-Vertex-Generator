"""Rebuild the tables of optimized_off/README.md and central_off/README.md from every atlas's seed files (the
explanations at the top of each README are kept)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = [("Decaic", "decaic_atlas", ""), ("Kidecaic", "decaic_atlas", "_ki"), ("Iodecaic", "decaic_atlas", "_io"),
          ("Contic", "contic_atlas", ""), ("Kicontic", "contic_atlas", "_ki"), ("Iocontic", "contic_atlas", "_io"),
          ("Tessic", "tessic_atlas", ""), ("Kitessic", "tessic_atlas", "_ki"), ("Tessipyritic", "tessic_atlas", "_pyr"),
          ("Demitessic", "tessic_atlas", "_demi"), ("Kidemitessic", "tessic_atlas", "_kidemi")]


def head(path):
    s = open(path).read()
    return s[:s.index("\n## ")].rstrip("\n")


def pennic_tail(path, marker):
    s = open(path).read()
    i = s.find(marker)
    return s[i:].rstrip("\n") if i >= 0 else ""


def icoic(kind, title="Icoic", path=("contic_atlas", "icoic_shapes.json"), note=None):
    """The icoic (or hyic) section (contic_atlas/icoic_shapes.json, hyic_atlas/hyic_shapes.json)."""
    f = os.path.join(HERE, *path)
    if not os.path.exists(f):
        return []
    d = json.load(open(f))["shapes"]
    if kind == "opt":
        L = [f"## {title}", "", note or ("contic_atlas/icoic_shapes.py ([3,4,3], CONTIC_SUB=ico). Sidpii, icoi and gidpii "
             "are most even where their two sets of 24 cells are equal: spic, cont and gippic."), "",
             "| shape | kind | R seed | R best | status | best beta | file |", "|---|---|---|---|---|---|---|"]
        for r in d:
            L.append("| %s | %s (%d) | %.6f | %.10f | %s | %s | %s |" % (
                r["name"], r["kind"], r["degrees"], r["seed_R"], r["R"], r.get("optimized_status", "uniform: every edge equal"),
                [round(v, 10) for v in r["optimized"]], r["opt_off"].split("/")[-1]))
    else:
        L = [f"## {title}", "", note or "contic_atlas/icoic_shapes.py.", "", "| shape | kind | depth | central beta | status | file |",
             "|---|---|---|---|---|---|"]
        for r in d:
            dd = "-" if r.get("depth") is None else "%.2e" % r["depth"]
            L.append(f"| {r['name']} | {r['kind']} ({r['degrees']}) | {dd} | {[round(v, 10) for v in r['central']]} | "
                     f"{r.get('central_status', 'the uniform seed (no fold, no walls)')} | {r['central_off'].split('/')[-1]} |")
    return L + [""]


def main():
    p = os.path.join(HERE, "optimized_off", "README.md")
    L = [head(p), ""]
    for title, atlas, tag in GROUPS:
        f = os.path.join(HERE, atlas, f"optimized_seeds{tag}.json")
        if not os.path.exists(f):
            continue
        L += [f"## {title}", "", "| shape | kind | R start | R best | status | best beta | off seed (R) | file |",
              "|---|---|---|---|---|---|---|---|"]
        for r in json.load(open(f))["shapes"]:
            same = np.allclose(r["best"], r["off_seed"], atol=1e-12)
            offs = "same" if same else "%s (%.6f)" % ([round(v, 10) for v in r["off_seed"]], r["off_R"])
            L.append("| %s | %s (%d) | %.6f | %.10f | %s | %s | %s | %s |" % (
                r["name"], r["kind"], r["degrees"], r["start_R"], r["R"], r["status"], [round(v, 10) for v in r["best"]],
                offs, r["off"].split("/")[-1]))
        L.append("")
    L += icoic("opt")
    L += icoic("opt", title="Hyic", path=("hyic_atlas", "hyic_shapes.json"), note="hyic_atlas/hyic_shapes.py ([5,3,3]): one shape per face of the chamber; optimized and central seeds are the uniform ones.")
    L.append(pennic_tail(p, "## Pennic"))
    open(p, "w").write("\n".join(L) + "\n")
    p = os.path.join(HERE, "central_off", "README.md")
    s = open(p).read()
    top = s[:s.index("| group | files |")].rstrip("\n")
    rows = []
    for title, atlas, tag in GROUPS:
        f = os.path.join(HERE, atlas, f"central_seeds{tag}.json")
        if os.path.exists(f):
            rows.append((title, json.load(open(f))["shapes"]))
    L = [top, "", "| group | files | moved deeper | kept the seed |", "|---|---|---|---|"]
    for g, d in rows:
        kept = [r["name"] for r in d if r["status"].startswith("no deeper")]
        L.append(f"| {g} | {len(d)} | {sum(1 for r in d if r['status'].startswith('deepest'))} | {', '.join(kept) or '-'} |")
    L += ["| Pennic | 6 | the uniform seeds (= the optimized seeds) | - |", ""]
    for g, d in rows:
        L += [f"## {g}", "", "| shape | kind | depth (seed -> centre) | central beta | file |", "|---|---|---|---|---|"]
        for r in d:
            dd = "-" if r["depth"] is None else "%.2e -> %.2e" % (r["start_depth"], r["depth"])
            L.append(f"| {r['name']} | {r['kind']} ({r['degrees']}) | {dd} | {[round(v, 10) for v in r['centre']]} | {r['off'].split('/')[-1]} |")
        L.append("")
    L += icoic("central")
    L += icoic("central", title="Hyic", path=("hyic_atlas", "hyic_shapes.json"), note="hyic_atlas/hyic_shapes.py ([5,3,3]): one shape per face of the chamber; optimized and central seeds are the uniform ones.")
    L.append(pennic_tail(p, "## pennic"))
    open(p, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
