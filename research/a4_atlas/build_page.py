"""The A4 domain viewer (../../assets/a4_domain.html): every A4 symmetry mapped so far (a4_map_<group>.json, from
a4_map.py), on the A4 chamber, with buttons to choose the symmetry. Laid out like the Symmetry Domain page of the
swirlprism atlas (a4_template.html shares its styles, key and controls).

    python a4_map.py && python build_page.py            # (b4, d4: the B4 and D4 pages, ../b4d4_atlas)
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
sys.path.insert(0, os.path.join(ROOT, "f4_atlas"))
import f4_map as F4M  # noqa: E402

# (every A4 symmetry: the ones not mapped yet are listed, greyed)
SYMMETRIES = [
    {"key": "pennic", "name": "Pennic", "bracket": "[3,3,3]", "order": 120},
    {"key": "decaic", "name": "Decaic", "bracket": "[[3,3,3]]", "order": 240},
    {"key": "kipennic", "name": "Kipennic", "bracket": "[3,3,3]+", "order": 60},
    {"key": "kidecaic", "name": "Kidecaic", "bracket": "[[3,3,3]]+", "order": 120},
    {"key": "iodecaic", "name": "Iodecaic", "bracket": "[[3,3,3]+]", "order": 120},
]
DIM_N = {"region": 3, "wall": 2, "line": 1, "point": 0}
B4D4 = os.path.join(ROOT, "b4d4_atlas")
F4 = os.path.join(ROOT, "f4_atlas")
# (each family's page: its symmetries, where their maps are, its title, chamber chips and notes)
FAMILIES = {
    "a4": {"title": "A4 Symmetry Domain", "out": "a4_domain.html", "dir": HERE, "prefix": "a4_map", "chamber": "A4",
           "symmetries": SYMMETRIES},
    "b4": {"title": "B4 Symmetry Domain", "out": "b4_domain.html", "dir": B4D4, "prefix": "b4d4_map", "chamber": "B4",
           "symmetries": [{"key": "tessic", "name": "Tessic", "bracket": "[4,3,3]", "order": 384},
                          {"key": "kitessic", "name": "Kitessic", "bracket": "[4,3,3]+", "order": 192},
                          {"key": "tessipyritic", "name": "Tessipyritic", "bracket": "[4,(3,3)+]", "order": 192}]},
    "d4": {"title": "D4 Symmetry Domain", "out": "d4_domain.html", "dir": B4D4, "prefix": "b4d4_map", "chamber": "D4",
           "symmetries": [{"key": "demitessic", "name": "Demitessic", "bracket": "[3¹,¹,¹]", "order": 192},
                          {"key": "kidemitessic", "name": "Kidemitessic", "bracket": "[3¹,¹,¹]+", "order": 96}]},
    "f4": {"title": "F4 Symmetry Domain", "out": "f4_domain.html", "dir": F4, "prefix": "f4_map", "chamber": "F4",
           "symmetries": [{"key": "icoic", "name": "Icoic", "bracket": "[3,4,3]", "order": 1152},
                          {"key": "contic", "name": "Contic", "bracket": "[[3,4,3]]", "order": 2304},
                          {"key": "kiicoic", "name": "Kiicoic", "bracket": "[3,4,3]+", "order": 576},
                          {"key": "kicontic", "name": "Kicontic", "bracket": "[[3,4,3]]+", "order": 1152},
                          {"key": "iocontic", "name": "Iocontic", "bracket": "[[3,4,3]+]", "order": 1152},
                          {"key": "icopyritic", "name": "Icopyritic", "bracket": "[3,4,3+]", "order": 576},
                          {"key": "toxitic", "name": "Toxitic", "bracket": "[3+,4,3+]", "order": 288},
                          {"key": "oxitic", "name": "Oxitic", "bracket": "[[3+,4,3+]]", "order": 576}]},
}
CHIPS = {"pennic": "chamber = 1 domain", "kipennic": "chamber = ½ domain", "decaic": "chamber = 2 domains (the swap)",
         "iodecaic": "chamber = 1 domain", "kidecaic": "chamber = 1 domain",
         "tessic": "chamber = 1 domain", "kitessic": "chamber = ½ domain", "tessipyritic": "chamber = ½ domain",
         "demitessic": "D4 chamber = 1 domain",
         "kidemitessic": "D4 chamber = ½ domain",
         "icoic": "chamber = 1 domain", "kiicoic": "chamber = ½ domain", "contic": "chamber = 2 domains (the swap)",
         "kicontic": "chamber = 1 domain", "iocontic": "chamber = 1 domain", "icopyritic": "chamber = ½ domain",
         "toxitic": "chamber = ¼ domain", "oxitic": "chamber = ½ domain"}


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


def notes_b4d4(g, d, syms):
    """The B4 and D4 pages' notes."""
    chart = ("The chamber is drawn in the gnomonic chart of the 3-sphere at its centre: a seed s is shown where the ray "
             "through it meets the tangent space there, so the mirrors are the faces of a flat tetrahedron and great "
             "circles are straight lines. Each corner is marked by the β that is not zero there.")
    seeds = ("β are a seed's distances to the B4 mirrors, scaled to sum to 1 (β1 the hexadecachoron's node, β4 the "
             "tesseract's). The seed is s = R⁻¹β, with R the generator's simple roots for B4; it works in the "
             "generator's Coordinates mode with any of these symmetries (their groups there are in the same basis).")
    flat = ("No group of the family has walls inside the B4 chamber (tessic_atlas/tessic_subgroups.md: the plane "
            "arrangement for tessic and demitessic; for the others the curved-wall map, a dense survey of the zones by "
            "the edges and the region closure check all find one region filling it). So the map is the chamber's "
            "fifteen faces, each classified by the polychoron a seed there makes (tessic_atlas/group.py), and checked "
            "at three more seeds to be one polychoron throughout (research/b4d4_atlas/b4d4_map.py).")
    if g == "tessic":
        dom = ["Tessic symmetry [4,3,3] is the tesseract's full symmetry group, order 384: the B4 reflection group. "
               "Its chamber, the tetrahedron between four mirrors, is one fundamental domain: 1/384 of the 3-sphere.",
               "A seed's 384 images are the vertices of the Wythoffian polychoron of its ringed nodes: gidpith inside, "
               "grit, prit, proh and tico on the faces, thex, sidpith, tat, rico, tah and srit on the edges, hex, ico, "
               "rit and tes at the corners.", flat, chart, seeds]
    elif g == "kitessic":
        dom = ["Kitessic symmetry [4,3,3]+ is the rotation half of tessic, order 192. Its fundamental domain is two "
               "chambers, a chamber and its mirror image, so the chamber shown is half of one.",
               "On the chamber's faces a seed's rotation images are its tessic ones, so the faces carry tessic's "
               "Wythoffian polychora. Inside, every seed makes snet, the snub tesseract: 192 vertices, 8 snub cubes, 16 "
               "snits, 24 gyro-squaps, 32 gyro-traps and 192 tets. It is chiral: the mirror chamber gives its mirror "
               "image.", flat, chart, seeds]
    elif g == "tessipyritic":
        dom = ["Tessipyritic symmetry [4,(3,3)+] keeps the tesseract's sign changes but only the even permutations of "
               "its coordinates: order 192. Of the chamber's mirrors only β4 = 0 (a sign change) is its own; the "
               "chamber is half of its fundamental domain.",
               "On the mirror β4 = 0 a seed makes sadi, the snub disicosatetrachoron (96 vertices, 2 degrees), where "
               "tessic has tico; inside, pysnet, the pyritosnub tesseract (192 vertices: 8 pyritosnub cubes, 16 snits, 24 "
               "rectangular trapezoprisms, 32 trips and 96 skew wedges). The other faces carry tessic's polychora: "
               "their tessipyritic orbit is the whole tessic one (proh's cells only fall in other classes here, so the "
               "classifier tells it apart until it is named by its vertices).", flat, chart, seeds]
    else:
        ki = g == "kidemitessic"
        dom = [("Kidemitessic symmetry D4+ is the rotation half of demitessic, order 96." if ki else
                "Demitessic symmetry [3¹,¹,¹] is the D4 reflection group, order 192: the tesseract's symmetries "
                "with an even number of sign changes.")
               + " The D4 chamber is drawn, a tetrahedron with corners β1 (hex), β2 (ico), β4 (hex) and β4′ (hex), β3's "
               "corner (rit) in the middle of the edge between the last two. It has the triangular symmetry of D4's "
               "triality, about the axis through ico's corner: its three hex corners, three tah faces and three rit "
               "edges are exchanged by it. Those symmetries of the chamber map D4 to itself (with D4 they make "
               "[3,4,3]), so seeds they exchange make congruent polychora and the map looks the same from each of its "
               "three sides. (The maps are found on one half of the chamber, the B4 chamber, cut off by the plane "
               "β4 = 0 through ico's corner, and carried to the rest by that symmetry; the plane is no D4 mirror and no "
               "shape's boundary, so it is not drawn.)",
               ("Inside, every seed makes sadi, the snub disicosatetrachoron: 8+8+8 snits, 24 disphenoids and 96 tets, "
                "3 degrees. On the mirrors a seed's rotation images are its demitessic ones, so the mirrors carry "
                "demitessic's shapes. (tico cannot be kidemitessic: it has 192 vertices.)" if ki else
                "Inside, every seed makes tico (24 truncated octahedra and 24 cubes, 3 degrees); on the mirrors rico "
                "and tah, on the edges rit and thex, at the corners hex and ico. thex has 1 degree here (an edge); the "
                "list gives it 2."), flat, chart, seeds]
    return [
        {"title": "The domain", "items": dom},
        {"title": "Names and markers", "items": [
            "Shapes are named from the accepted list (pages/tessic_source.txt); a shape found on several faces of the "
            "map is one row, its pieces counted under Where.",
            "Lightness shows dimension, as on the Symmetry Domain page: regions lightest, walls darker, lines darker "
            "still, points darkest.",
            "Every shape is marked by an example seed in each of its pieces: walls and lines at their most even seed "
            "(tessic_atlas/optimize_seeds.py) where that lies in the piece, else the uniform seed (equal β on its "
            "ringed nodes); the region at the chamber's centre. Selecting the region in the key paints its extent.",
            "Boundaries are the shapes on the edges of a shape's pieces; separates lists the regions or walls a wall "
            "or line lies between."]},
        {"title": "Status", "status": True, "items": [
            "<b>Mapped:</b> " + ", ".join(f"{s['name']} {s['bracket']}" for s in syms if s["key"] in d) + "."]},
    ]


def notes_f4(g, d, syms):
    """The F4 page's notes."""
    chart = ("The chamber is drawn in the gnomonic chart of the 3-sphere at its centre: a seed s is shown where the ray "
             "through it meets the tangent space there, so the mirrors are the faces of a flat tetrahedron, every plane "
             "of β is a plane and great circles are straight lines. Each corner is marked by the β that is not zero "
             "there.")
    seeds = ("β are a seed's distances to the four F4 mirrors, scaled to sum to 1 (β1 and β2 the long mirrors, β3 and "
             "β4 the short ones; β1's corner is the 24-cell's vertex, β4's its dual's). The seed is s = R⁻¹β, with R "
             "the generator's simple roots for F4, so it can be pasted into the generator's Coordinates mode.")
    if g == "icoic":
        dom = ["Icoic symmetry [3,4,3] is the 24-cell's full symmetry group, order 1152: the F4 reflection group. Its "
               "chamber, the tetrahedron between four mirrors, is one fundamental domain.",
               "As a reflection group the polychoron is the same over each open face of the chamber: the fifteen "
               "Wythoffian polychora of the 24-cell family, nine shapes (a ringing and its reverse give congruent "
               "polychora, the 24-cell being self-dual). The interior is gidpii, the faces grico and prico, the edges "
               "tico, srico, icoi and sidpii, the corners ico and rico.", chart, seeds]
    elif g == "kiicoic":
        dom = ["Kiicoic symmetry [3,4,3]+ is the rotation half of icoic, order 576. Its fundamental domain is two "
               "chambers, a chamber and its mirror image, so the chamber shown is half of one.",
               "On the chamber's faces a seed's rotation images are its icoic ones, so the faces carry icoic's "
               "Wythoffian polychora. Inside, every seed makes snoc, the snub tetracontoctachoron in its kiicoic "
               "version: 576 vertices, 24+24 snub cubes, 96+96 octahedra and 576 tets. It is chiral: the mirror "
               "chamber gives its mirror image.", chart, seeds]
    elif g == "contic":
        dom = ["Contic symmetry [[3,4,3]] adds the swap of the 24-cell and its dual to icoic: order 2304. The swap "
               "reads β backwards, (β1, β2, β3, β4) → (β4, β3, β2, β1): a half-turn of the chamber about the axis "
               "β1 = β4, β2 = β3 (dashed). So half the chamber, β1 ≥ β4, is a fundamental domain, and the whole "
               "chamber shows each shape twice, once on each side of the fold β1 = β4.",
               "The polychoron changes only on planes of β, here over Q(√2) (contic_atlas/arrangement.py): every face "
               "of the exact arrangement is classified (contic_atlas/shapes.py) and named from the list (the Icoic "
               "and Contic Isogonals page). A face is drawn as a wall, line or point where its polychoron changes "
               "there.",
               "On the mirrors the seed has fewer images (bicont, spic, cont and their kin); the regions sabotic and "
               "gabotic fill the interior.", chart, seeds]
    elif g == "icopyritic":
        dom = ["Icopyritic symmetry [3,4,3+] (order 576) keeps the long mirrors (β1, β2) and turns the short ones into "
               "rotations. It is a normal half of icoic, so the chamber is half a fundamental domain: its image in a "
               "short mirror gives the mirror images of the same polychora.",
               "Pysnapi fills the whole chamber. On its own mirrors, the long ones, lie the new shapes: pysni on "
               "β1 = 0, prissi on β2 = 0 and sadi on their edge. On the short mirrors a seed's images are its icoic "
               "ones: grico on β4 = 0, and on β3 = 0 prico's orbit, which is pysnapi's polychoron, so no wall. The "
               "faces are classified under icopyritic (contic_atlas/group.py, CONTIC_SUB = icp) and named by the "
               "toxitic atlas's seeds (its two sheets: icp, and icpb read backwards).", chart, seeds]
    elif g == "toxitic":
        dom = ["Toxitic symmetry [3+,4,3+] (order 288) keeps an even number of long and of short reflections: a normal "
               "quarter of icoic, so the chamber is a quarter of a fundamental domain.",
               "One polychoron fills the whole chamber: sebtidi. Its mirror faces give pysni (β1 = 0, β4 = 0) and prissi "
               "(β2 = 0, β3 = 0), its edges sadi (β1 = β2 = 0 and β3 = β4 = 0), icoi and sidpii, its corners ico and "
               "rico.", chart, seeds]
    else:
        dom = [f"{g.capitalize()} has curved walls; its map is not built yet.", chart, seeds]
    mapped = [s for s in syms if s["key"] in d]
    rest = [s for s in syms if s["key"] not in d]
    return [
        {"title": "The domain", "items": dom},
        {"title": "Names and markers", "items": [
            "Shapes are named from the accepted lists (pages/contic_names.py, as on the Icoic and Contic Isogonals "
            "page); a shape found on several faces of the map is one row, its pieces counted under Where.",
            "Lightness shows dimension, as on the Symmetry Domain page: regions lightest, walls darker, lines darker "
            "still, points darkest.",
            "Every shape is marked by an example seed in each of its pieces: walls and lines at their most even seed "
            "(contic_atlas/optimize_seeds.py, toxitic_atlas/optimize_central.py) where that lies inside the shape, "
            "else its central seed or the uniform seed (equal β on its ringed nodes); regions at their central seed. "
            "Selecting a region in the key paints its extent.",
            "Boundaries are the shapes on the edges of a shape's pieces; separates lists the regions or walls a wall "
            "or line lies between."]},
        {"title": "Status", "status": True, "items": [
            "<b>Mapped:</b> " + ", ".join(f"{s['name']} {s['bracket']}" for s in mapped) + "."]
            + ([("<b>Next:</b> " + ", ".join(f"{s['name']} {s['bracket']}" for s in rest)
                 + " (curved walls: the exact pipeline of the A4 page).")] if rest else [])},
    ]


def main(fam="a4"):
    FAM = FAMILIES[fam]
    syms = FAM["symmetries"]
    maps, light, dark = {}, [], []
    for s in syms:
        path = os.environ.get(f"MAP_{s['key'].upper()}") or os.path.join(FAM["dir"], f"{FAM['prefix']}_{s['key']}.json")
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
        ch = FAM["chamber"]
        d.update(key=g, title=f"{FAM['title']} · {s['name']}",
                 eyebrow=f"{ch} · {s['name'].lower()} {s['bracket']} · order {d['order']}",
                 lede=f"A map of every isogonal polychoron the {s['name'].lower()} group {s['bracket']} produces from a "
                      f"single seed. Each point of the {ch} chamber is a seed; it is coloured by the polychoron its "
                      f"{d['order']} images make."
                      + (f" Choose another {ch} symmetry above to see its map on the same chamber." if len(syms) > 1
                         else ""),
                 domainChip=CHIPS[g])
        if g in ("decaic", "contic"):
            half = lambda b: (F4M if fam == "f4" else M).chart(np.asarray(b, float))
            d["fold"] = {"axis": [M.r5(half([1, 0, 0, 1])), M.r5(half([0, 1, 1, 0]))],
                         "axisName": "the swap's axis β1 = β4, β2 = β3 (seeds here are fixed by the swap)",
                         "plane": [M.r5(half([1, 0, 0, 1])), M.r5(half([0, 1, 0, 0])), M.r5(half([0, 0, 1, 0]))]}
        maps[g] = d
    ids = {g: {t["id"] for t in d["types"]} for g, d in maps.items()}
    names = {s["key"]: s["name"] for s in syms}
    for g, d in maps.items():                   # (the family's other symmetries whose maps have the same polychoron)
        d["notes"] = notes(g, maps) if fam == "a4" else notes_f4(g, maps, syms) if fam == "f4" else notes_b4d4(g, maps, syms)
        for t in d["types"]:
            t["shared"] = [names[h] for h in maps if h != g and t["id"] in ids[h]]
    type_css = (f":root {{ {' '.join(light)} }}\n@media (prefers-color-scheme: dark) {{ :root:not([data-theme=\"light\"]) "
                f"{{ {' '.join(dark)} }} }}\n:root[data-theme=\"dark\"] {{ {' '.join(dark)} }}")
    data = {"symmetries": syms, "family": FAM["chamber"], "maps": maps}
    page = (open(os.path.join(HERE, "a4_template.html")).read().replace("__TITLE__", FAM["title"])
            .replace("__FAMILY__", FAM["chamber"])
            .replace("__TYPE_CSS__", type_css).replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    out = os.environ.get("PAGE_OUT") or os.path.join(ROOT, "..", "assets", FAM["out"])
    open(out, "w").write(page)
    summary = ", ".join(f"{g} {len(d['types'])} shapes" for g, d in maps.items())
    print(f"{out}: {summary}, {len(page) // 1024} KB")


if __name__ == "__main__":
    for f_ in sys.argv[1:] or ["a4"]:
        main(f_)
