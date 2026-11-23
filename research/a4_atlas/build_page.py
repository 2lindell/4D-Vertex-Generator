"""The A4 domain viewer (../../assets/a4_domain.html): every A4 symmetry mapped so far (a4_map_<group>.json, from
a4_map.py), on the A4 chamber, with buttons to choose the symmetry. Laid out like the Symmetry Domain page of the
swirlprism atlas (a4_template.html shares its styles, key and controls).

    python a4_map.py && python build_page.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tetraswirl_atlas"))
from place_colors import DARK, LIGHT, _oklch_to_hex  # noqa: E402  (the Symmetry Domain page's colour rule)

import a4_map as M  # noqa: E402

# (every A4 symmetry: the ones not mapped yet are listed, greyed)
SYMMETRIES = [
    {"key": "pennic", "name": "Pennic", "bracket": "[3,3,3]", "order": 120},
    {"key": "decaic", "name": "Decaic", "bracket": "[[3,3,3]]", "order": 240},
    {"key": "kipennic", "name": "Kipennic", "bracket": "[3,3,3]+", "order": 60},
    {"key": "kidecaic", "name": "Kidecaic", "bracket": "[[3,3,3]]+", "order": 120},
    {"key": "iodecaic", "name": "Iodecaic", "bracket": "[[3,3,3]+]", "order": 120},
]
DIM_N = {"region": 3, "wall": 2, "line": 1, "point": 0}


def notes(g, d):
    chart = ("The chamber is drawn in the gnomonic chart of the 3-sphere at its centre: the seed s (a point of the "
             "3-sphere) is shown where the ray through it meets the tangent space there, so the four mirrors are the "
             "faces of a flat tetrahedron, every plane of β is a plane and great circles are straight lines. Each "
             "corner is marked by the one β that is not zero there.")
    seeds = ("β are a seed's distances to the four mirrors, scaled to sum to 1. The seed is s = R⁻¹β, with R the "
             "generator's simple roots for A4, so it can be pasted into the generator's Coordinates mode.")
    if g == "pennic":
        dom = [f"Pennic symmetry [3,3,3] is the 5-cell's full symmetry group, order 120: the A4 reflection group. Its "
               f"chamber, the tetrahedron between four mirrors, is one fundamental domain: 1/120 of the 3-sphere.",
               "A seed's 120 images are the vertices of an isogonal polychoron. As a reflection group with no other "
               "symmetry, the polychoron is the same over each open face of the chamber: the fifteen Wythoffian "
               "polychora of the 5-cell family. The chamber's interior is gippid, its four faces two grips and two "
               "prips, its six edges two tips, two srips, deca and spid, and its corners two pens and two raps (each "
               "pair a shape and its dual, the same polychoron).",
               chart, seeds]
    elif g == "kipennic":
        dom = ["Kipennic symmetry [3,3,3]+ is the rotation half of pennic, order 60 (the alternating group A5). Its "
               "fundamental domain is two chambers, a chamber and its mirror image, so the chamber shown is half of "
               "one: 1/120 of the 3-sphere.",
               "On the chamber's faces a seed's 60 rotation images are its 120 pennic images (each reflection there "
               "fixes the seed or matches a rotation), so the faces carry the same Wythoffian polychora as pennic: "
               "grip and prip on the faces, tip, srip, deca and spid on the edges, pen and rap at the corners.",
               "Inside the chamber every seed makes one polychoron, snad: the snub decachoron in its kipennic version, "
               "5+5 snits, 10+10 gyro traps and 60 irregular tets, with 60 vertices (checked at 1500 random seeds: one "
               "polychoron throughout). It is chiral: the mirror chamber gives its mirror image.",
               chart, seeds]
    elif g == "iodecaic":
        dom = ["Iodecaic symmetry [[3,3,3]+] (order 120) keeps the 5-cell's rotations and takes its reflections only "
               "together with the swap of the 5-cell and its dual. It has no mirrors of its own, so the A4 chamber is "
               "one fundamental domain; the swap maps the group to itself, so a seed and its swap image (β read "
               "backwards) make the same polychoron, and the map is symmetric under β → (β4, β3, β2, β1).",
               "Its walls are curved: tetosbid and petosbid lie on planes, bipyosbid on a quartic. The map is built "
               "exactly (a4_exact.py): every pair of its surfaces (and the mirrors) traced along their common curve, "
               "cut where a third crosses (10 points, solved on all three), and each surface cut along those curves "
               "into pieces, each classified; a wall is drawn only on its own pieces, so its edges are the traced "
               "lines. Every piece is checked to be one polychoron at many points, and where one was not, the "
               "surface through the change was added (the fold planes β1 = β4 and β2 = β3).",
               "On the mirrors a seed's images are the decaic ones, so the mirrors carry decaic's shapes.",
               chart, seeds]
    elif g == "kidecaic":
        dom = ["Kidecaic symmetry [[3,3,3]]+ (order 120) is the rotation half of decaic: the 5-cell's 60 rotations, "
               "and the swap of the 5-cell and its dual taken together with a reflection. It has no mirrors of its own, "
               "so the A4 chamber is one fundamental domain; the swap maps the group to itself, so a seed and its swap "
               "image (β read backwards) make the same polychoron, and the map is symmetric under β → (β4, β3, β2, β1). "
               "Its polychora are chiral: a mirror image of the chamber gives their mirror images.",
               "Its map is much finer than iodecaic's: 14 regions (abosid fills about half the chamber, apbosid a "
               "fifth, gabosid and bebosid most of the rest; several are thin slivers), 16 walls on planes and "
               "quadric and quartic surfaces, and three lines: snad all along the swap's axis, and the crossing lines "
               "sawbosid and gawbosid, where two surfaces meet.",
               "The map is built exactly (a4_exact.py), as for iodecaic. 70 of its surfaces contain the swap's axis, "
               "and each swap pair of them is tangent all along it, so a traced curve ends where its two surfaces turn "
               "tangent, solved onto the axis, and the axis is drawn once, as snad. The closure check (every piece "
               "classified at many points) added surfaces through the changes it found in three rounds; after the "
               "fourth, a few pieces still disagree at one or two of their points (two bedwebosid pieces and one "
               "epybosid piece among the walls), so a wall's outline there may be off by a little.",
               "On the mirrors a seed's images are the decaic ones, so the mirrors carry decaic's shapes.",
               chart, seeds]
    else:
        dom = ["Decaic symmetry [[3,3,3]] is the 5-cell's symmetries with the swap of the 5-cell and its dual: order "
               "240. The swap reads β backwards, (β1, β2, β3, β4) → (β4, β3, β2, β1): a half-turn of the chamber about "
               "the axis β1 = β4, β2 = β3 (dashed). So half the chamber, β1 ≥ β4, is a fundamental domain (1/240 of "
               "the 3-sphere), and the whole chamber shows each shape twice, once on each side of the fold β1 = β4.",
               "The polychoron changes only on planes of β (every edge length form factors into linear forms), so "
               "the map is an exact arrangement of 22 planes besides the mirrors and the fold (decaic_atlas/"
               "arrangement.py): every face of it is classified (decaic_atlas/shapes.py) and named from the list. "
               "A face is drawn as a wall, line or point where its polychoron changes there; a plane where nothing "
               "changes is not drawn.",
               "On the mirrors the seed has fewer images (the Wythoffian shapes with decaic symmetry: bideca, spid, "
               "deca and their kin); the regions sabotid and gabotid fill the interior.",
               chart, seeds]
    return [
        {"title": "The domain", "items": dom},
        {"title": "Names and markers", "items": [
            "Shapes are named from the accepted lists; a shape found on several faces of the map is one row, its "
            "pieces counted under Where.",
            "Lightness shows dimension, as on the Symmetry Domain page: regions lightest, walls darker, lines darker "
            "still, points darkest.",
            "Every shape is marked by an example seed in each of its pieces: walls and lines at their most even seed "
            "(the least ratio of longest to shortest edge: decaic_atlas/optimize_seeds.py; for pennic and kipennic the "
            "uniform polychoron, equal β on its ringed nodes), regions at their central seed (the deepest point: "
            "decaic_atlas/central_seeds.py; the chamber's centre for pennic and kipennic). Selecting a region in the "
            "key paints its extent, the walls round it, in its colour, as on the swirlprism page.",
            "Boundaries are the shapes on the edges of a shape's pieces; separates lists the regions or walls a wall "
            "or line lies between."]},
        {"title": "Status", "status": True, "items": [
            "<b>Mapped:</b> " + ", ".join(f"{s['name']} {s['bracket']}" for s in SYMMETRIES if s["key"] in d) + ".",
            "<b>Next:</b> " + ", ".join(f"{s['name']} {s['bracket']}" for s in SYMMETRIES if s["key"] not in d)
            + " (curved walls: a4_exact.py)." if any(s_["key"] not in d for s_ in SYMMETRIES) else
            "<b>Next:</b> nothing: every A4 symmetry is mapped."]},
    ]


def main():
    maps, light, dark = {}, [], []
    for s in SYMMETRIES:
        path = os.path.join(HERE, f"a4_map_{s['key']}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        g = s["key"]
        for dim in DIM_N:
            members = [t for t in d["types"] if t["dim"] == dim]
            for k, t in enumerate(members):
                h = 25 + 360 * k / max(len(members), 1)
                t["color"] = f"--ty-{g}-{t['id']}"
                light.append(f"--ty-{g}-{t['id']}: {_oklch_to_hex(*LIGHT[DIM_N[dim]], h)};")
                dark.append(f"--ty-{g}-{t['id']}: {_oklch_to_hex(*DARK[DIM_N[dim]], h)};")
        for k, f in enumerate(d["domain"]["faces"]):
            f["ids"] = [t["id"] for t in d["types"] if k in t.get("mirrors", [])]
        d.update(key=g, title=f"A4 Symmetry Domain · {s['name']}",
                 eyebrow=f"A4 · {s['name'].lower()} {s['bracket']} · order {d['order']}",
                 lede=f"A map of every isogonal polychoron the {s['name'].lower()} group {s['bracket']} produces from a "
                      f"single seed. Each point of the A4 chamber is a seed; it is coloured by the polychoron its "
                      f"{d['order']} images make. Choose another A4 symmetry above to see its map on the same chamber.",
                 domainChip={"pennic": "chamber = 1 domain", "kipennic": "chamber = ½ domain",
                             "decaic": "chamber = 2 domains (the swap)", "iodecaic": "chamber = 1 domain",
                             "kidecaic": "chamber = 1 domain"}[g])
        if g == "decaic":
            half = lambda b: M.chart(np.asarray(b, float))
            d["fold"] = {"axis": [M.r5(half([1, 0, 0, 1])), M.r5(half([0, 1, 1, 0]))],
                         "axisName": "the swap's axis β1 = β4, β2 = β3 (seeds here are fixed by the swap)",
                         "plane": [M.r5(half([1, 0, 0, 1])), M.r5(half([0, 1, 0, 0])), M.r5(half([0, 0, 1, 0]))]}
        maps[g] = d
    ids = {g: {t["id"] for t in d["types"]} for g, d in maps.items()}
    names = {s["key"]: s["name"] for s in SYMMETRIES}
    for g, d in maps.items():                   # (the other A4 symmetries whose maps have the same polychoron)
        d["notes"] = notes(g, maps)
        for t in d["types"]:
            t["shared"] = [names[h] for h in maps if h != g and t["id"] in ids[h]]
    type_css = (f":root {{ {' '.join(light)} }}\n@media (prefers-color-scheme: dark) {{ :root:not([data-theme=\"light\"]) "
                f"{{ {' '.join(dark)} }} }}\n:root[data-theme=\"dark\"] {{ {' '.join(dark)} }}")
    data = {"symmetries": SYMMETRIES, "maps": maps}
    page = (open(os.path.join(HERE, "a4_template.html")).read().replace("__TITLE__", "A4 Symmetry Domain")
            .replace("__TYPE_CSS__", type_css).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    out = os.environ.get("PAGE_OUT") or os.path.join(ROOT, "..", "assets", "a4_domain.html")
    open(out, "w").write(page)
    summary = ", ".join(f"{g} {len(d['types'])} shapes" for g, d in maps.items())
    print(f"{out}: {summary}, {len(page) // 1024} KB")


if __name__ == "__main__":
    main()
