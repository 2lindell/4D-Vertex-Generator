"""Kicontic / iocontic shapes against kidecaic / iodecaic (contic_snub_shapes.md): each matched by edge valences (the
contic counts are 1152/120 = 9.6 times the decaic ones). Contic shapes: the map's wall_shapes, and the edge survey's new
region types (told apart by their arrangement keys).

    python compare_snubs.py
"""
import json, os, re, subprocess, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEC = os.path.join(HERE, "..", "decaic_atlas")


def prof(sg):
    e = Counter()
    for n, k in re.findall(r"(\d+)x(\d+)", sg.split("edges")[1]):
        e[int(k)] += int(n)
    return dict(sorted(e.items()))


def cells(sg):
    return sum(int(a) for a in re.findall(r"(\d+)x\[", sg.split("|")[0]))


def keys(sub, pts):
    code = ("import json,sys,group as G;print(json.dumps([G.label(p)[1] for p in json.loads(sys.argv[1])]))")
    r = subprocess.run([sys.executable, "-c", code, json.dumps(pts)], cwd=HERE, env={**os.environ, "CONTIC_SUB": sub},
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


NAMES = {"ki": [("sabosid", {3: 120, 4: 420, 5: 120, 8: 60}), ("apbosid", {3: 360, 4: 300, 8: 120}),
                ("mebosid / gabosid", {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60}), ("sigwebosid", {3: 300, 4: 300, 7: 60}),
                ("gigwebosid", {3: 240, 4: 360, 8: 60}), ("tetbosid", {3: 300, 4: 120, 5: 60, 6: 60}),
                ("petbosid", {3: 420, 4: 60, 5: 60, 6: 60})],
         "io": [("sosbid", {3: 120, 4: 300, 5: 240}), ("gosbid", {3: 360, 4: 60, 5: 240, 6: 60}),
                ("tetosbid", {3: 240, 4: 240}), ("petosbid", {3: 360, 4: 180})]}


def main():
    L = ["# Kicontic and iocontic shapes, against kidecaic and iodecaic", "",
         "Kicontic = [[3,4,3]]+ (the rotations, order 1152, with the 24-cell/dual swap); iocontic = [[3,4,3]+] (F4's rotations "
         "and F4's reflections times the swap, 1152; without the swap). Mapped as the decaic snubs: curved-wall map "
         "(snub_map.py, grid N = 32), walls reclassified on exact surface points (wall_shapes.py), then a dense survey of the "
         "two edge zones and the whole domain (edge_survey.py, 700 points each) for small regions. Every new region type's "
         "margin is well above the hull tolerance.", "",
         "Matched by edge valences: the contic counts are exactly 9.6 = 1152/120 times the decaic ones. On the mirrors the "
         "orbits are the contic ones (contic_shapes.md). On the gippid-like line kicontic gives the snad analogue (576 "
         "vertices: 48 snub cubes, 192 octahedral cells, 576 tets); iocontic keeps the gippid analogue.", ""]
    summary = []
    for sub, title in (("ki", "Kicontic vs kidecaic"), ("io", "Iocontic vs iodecaic")):
        dec = [(w["kind"].split(" ")[0], prof(w["signature"]), w["cells"])
               for w in json.load(open(os.path.join(DEC, f"wall_shapes_{sub}.json")))["shapes"]]
        if sub == "ki":                     # (the small region the dense kidecaic survey added)
            dec.append(("region", {3: 360, 4: 180, 5: 120, 6: 180, 8: 60, 10: 60}, 680))
        con = [(w["kind"].split(" ")[0], prof(w["signature"]), w["cells"], "map")
               for w in json.load(open(os.path.join(HERE, f"wall_shapes_{sub}.json")))["shapes"]]
        new = [t for z in json.load(open(os.path.join(HERE, f"edge_survey_{sub}.json")))["zones"].values()
               for t in z["types"] if t["known_region"] is None]
        ks = keys(sub, [t["example"] for t in new])
        seen = set()
        for t, k in zip(new, ks):
            if k not in seen:
                seen.add(k)
                con.append(("region", prof(t["signature"]), cells(t["signature"]), "edge survey"))
        L += [f"## {title}", "", "| decaic shape | kind | decaic cells | decaic edges | contic | contic cells | found by |",
              "|---|---|---|---|---|---|---|"]
        used = [False] * len(con)
        miss = Counter()
        for kind, p, c in dec:
            nm = next((n for n, e in NAMES[sub] if e == p), "new")
            target = {k: round(v * 9.6) for k, v in p.items()}
            j = next((j for j, x in enumerate(con) if not used[j] and x[0] == kind and x[1] == target), None)
            if j is not None:
                used[j] = True
            else:
                miss[kind] += 1
            L.append(f"| {nm} | {kind} | {c} | {p} | {'found' if j is not None else '**not found**'} | "
                     f"{con[j][2] if j is not None else ''} | {con[j][3] if j is not None else ''} |")
        extra = [x for j, x in enumerate(con) if not used[j]]
        L += ["", f"Contic shapes with no decaic counterpart: {len(extra)}"
              + ("" if not extra else ": " + "; ".join(f"{x[0]} {x[2]} cells {x[1]}" for x in extra)), ""]
        summary.append((title, Counter(k for k, *_ in dec), miss, len(extra)))
    at = next(i for i, l in enumerate(L) if l.startswith("## "))
    L[at:at] = ["| | decaic regions | decaic walls | contic regions not found | contic walls not found | contic extra |",
              "|---|---|---|---|---|---|"] + [f"| {t} | {d['region']} | {d['wall']} | {m['region']} | {m['wall']} | {e} |"
                                              for t, d, m, e in summary] + [""]
    open(os.path.join(HERE, "contic_snub_shapes.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
