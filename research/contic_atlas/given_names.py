"""The accepted names of the contic-family shapes (contic, kicontic, iocontic, icopyritic, toxitic, oxitic), from the
list the project was given, and which atlas shape each belongs to.

Matched by the list's own data: its cells (counts by orbit and kind) and its edge column (the valences of the edges at
a vertex: "556,4^5,3^4" = edges in 5, 5, 6 cells, five in 4, four in 3), compared with the atlas polytopes' edge
classes (an orbit of n edges gives 2n / V edges at each vertex). Contic shapes are matched through the decaic
counterpart they were paired with (contic_shapes.json), their cells checked against the list. Where the list's data
fit more than one atlas shape the name is given as "a_or_b" (and the atlas shapes numbered); where they fit none it is
noted below.
"""
from __future__ import annotations

# (contic: the atlas's decaic counterpart -> the contic name; every cell list checked)
CONTIC = {
    "sabotid": "sabotic", "gabotid": "gabotic", "petbotid": "pitbotic", "tetbotid": "catbotic",
    "gabipard": "gabiparc", "mabipard": "mabiparc", "sabipard": "sabiparc",
    "sobcated": "sobcotic", "mobcated": "mobcotic", "gobcated": "gobcotic",
    "biped": "bipec", "bited": "bitec", "bimted": "bimtec",
    # (corrected by the project: the 864-cell line (the atlas's "sabred": 384 trapos) is sabric and the 1248-cell one
    #  ("mabred": 48 coes, 192 trips, 144 squaps, 288 tedows, 576 rectpyrs) is mabric; "gabred" has sircoes = gabric)
    "mabred": "mabric", "sabred": "sabric", "gabred": "gabric",
    "tispid": "tispic", "tabcated": "tabcatic", "gippid": "gippic", "tabipard": "tabiparc", "tadeca": "ticont",
    "bideca": "bicont", "spid": "spic", "apid": "bamic", "deca": "cont", "respid": "respic", "redeca": "recont",
}

# (kicontic / iocontic: the atlas's shape names (snub_shapes(), the .off files) -> the name, by edge valences)
KICONTIC = {
    "region_01_1152v_5280c": "mabosc_or_gabosc_1", "region_06_1152v_5280c": "mabosc_or_gabosc_2",
    "region_04_1152v_3936c": "apbosc", "region_05_1152v_4128c": "sabosc",
    "wall_01_1152v_2688c": "sigawbosc", "wall_04_1152v_2256c": "catbosc", "wall_05_1152v_2112c": "petbosc",
    "wall_06_1152v_2976c": "gigawbosc", "snad-like": "snoc",
}
IOCONTIC = {
    "region_01_1152v_3552c": "sosbic", "region_02_1152v_3360c": "gosbic", "wall_01_1152v_1680c": "catosbic",
    "wall_02_1152v_1536c": "petosbic", "gippid-like": "gippic",
}

# (toxitic / oxitic and icopyritic: arrangement keys of toxitic_atlas/toxitic_shapes.json -> name)
TOXITIC = {"a5f41b7684fb": "sebtidi", "50cdad1380f5": "prissi", "6b0cb2276095": "pysni", "3d29aef5b8e0": "sadi",
           # (the edges b2 = b3 = 0 and b1 = b4 = 0 are sidpii and icoi (icoic): spic and cont with their two sets of 24
           #  cells unequal (symmetry 1152); spic and cont themselves at b1 = b4 (oxitic's points / line))
           "cc9050ff3c4e": "sidpii", "095e2d12fe7f": "icoi",
           "58cb82973b86": "ico", "8de8fd73a969": "rico"}
# (oxitic: matched by edges and cells at a vertex; mitatosbitac and gadwosbitac one edge count apart (NOTES))
OXITIC = {"10943c37691d": "cont", "c41b4cd39c7a": "snoc_oxitic", "e695c863ed63": "awosbitac", "9f29b1b7c1be": "bamic", "31553874949a": "bicont", "a6ddde77b471": "bimtec", "414533b9acc6": "bipec", "80b2343fd8f0": "bitac", "89590d9cc67a": "gabertetic", "44f48f6922fc": "gabtutic", "17848f9c0c98": "gadwosbitac", "a2f01eef519b": "gidwosbitac", "51ee8ceff001": "gitatosbitac_or_mosbitic_1", "e910898ef869": "gitatosbitac_or_mosbitic_2", "64745339a7a1": "gitatosbitac_or_mosbitic_3", "d5ec887138b0": "gosbitic", "74748c41c6c4": "gywosbitic", "8da3f28a95bc": "mabertetic", "b577195e5701": "medwosbitac", "15028d583027": "mitatosbitac", "444e8e32e97b": "otosbitac", "7443559f1ee3": "sabertetic", "e39f8da091ec": "sabtutic", "a8da4ec5c0e2": "sebtic", "9e2bb739cd87": "sidwosbitac", "fdec102036a1": "sosbitic", "e0a6a58eb3b3": "spic", "b900c8edde78": "stitosbetic", "6fc49066ae8b": "tatbitutic", "223f1e4ac3a9": "tetabertetic", "6f4a50387244": "totabertetic"}

NOTES = {
    "snoc_oxitic": "the subsymmetrical snoc (community): an oxitic wall (in the mirror-image sheet) through the line "
                   "b1 = b4, b2 = b3, where it gains kicontic symmetry and is snoc; same counts (576 vertices, 816 cells), "
                   "and its most even member is snoc's (R 1.206271).",
    "cont": "uniform cont (beta = (0, 1, 1, 0), where the fold line meets b1 = b4 = 0) is isogonal within oxitic: its "
            "288 vertices are one oxitic orbit (48 tics, one class). Under toxitic it is the uniform member of icoi.",
    "sebtidi": "sebtidi is the subsymmetrical sebtic: under toxitic the polychora on the fold line b1 = b4, b2 = b3 "
               "are sebtidi's own type, and there oxitic gives them its symmetry (sebtic). Most even sebtic (on the "
               "line): R = 1.48563 at beta = (0.33139, 0.16861, 0.16861, 0.33139); most even sebtidi, just off the "
               "line: R = 1.48512.",
    "mabosc_or_gabosc": "Mabosc and Gabosc have the same cells and edge valences in the list; two atlas regions fit both "
                        "(as mebosid / gabosid for kidecaic).",
    "gidwosbitac": "the list gives 192x2 trigypos (3744 cells); the atlas shape with its edge valences and cells at a "
                   "vertex (544,3^23) has one 192 orbit of octahedra (3552 cells).",
    "mitatosbitac": "the list's edges 77,555,444,3^4; the atlas shape with its cells (and cells at a vertex 5,444,3^11) "
                    "has 7,555,4444,3^4: likely the same shape, one count apart.",
    "gadwosbitac": "the list's edges 6^6 12,4^8,333; the atlas shape with its cells (and 544,3^25) has 6^6 12,4^7,3^4: "
                   "likely the same, one count apart.",
    "gitatosbitac_or_mosbitic": "three atlas regions have Gitatosbitac's cells and edges exactly (5568,4^6,3^4; "
                                "5,4444,3^13); Mosbitic lists the same cells but edges 56,4^7,3^5 and 5,4^4,3^11, "
                                "which no atlas shape has. Incidence checks show the three are different polychora.",
}


def contic(counterpart):
    return CONTIC.get(counterpart, counterpart + "-like")


def snub(sub, name):
    return {"ki": KICONTIC, "io": IOCONTIC}.get(sub, {}).get(name, name)
