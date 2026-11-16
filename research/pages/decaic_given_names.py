"""Names given to the kidecaic and iodecaic shapes (the updated list of 2026-11-16), matched to the atlas's shapes.

Each list entry was matched by its two profile columns: the edges at a vertex by how many cells meet there (e.g.
5568 10,4^3,3^6) and the vertex figure's faces (e.g. 5,44,3^17), computed from each shape's .off file (all vertices are
alike). Where several shapes share both columns, the list's Greek letters follow the order the shapes have on the
decaic page (regions 01-09, 21, 22, then walls 10-20, 23, 24, then lines 25, 26), as they do everywhere the columns do
tell the shapes apart.
"""

ORDER = "where the columns are alike, by the list's order, which follows the page's order everywhere else"

# atlas name -> (short name, full name, note)
GIVEN = {
    "kidecaic_region_01_120v_560c": ("Debosid", "Delta Biomnisnub Decachoron", ""),
    "kidecaic_region_02_120v_680c": ("Adwebosid", "Alpha Disphenoidal Biomnisnub Decachoron", ""),
    "kidecaic_region_03_120v_680c": ("Bedwebosid", "Beta Disphenoidal Biomnisnub Decachoron",
                                     "Region 07 (Gadwebosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_region_04_120v_560c": ("Ebosid", "Epsilon Biomnisnub Decachoron", ""),
    "kidecaic_region_05_120v_560c": ("Zebosid", "Zeta Biomnisnub Decachoron", ""),
    "kidecaic_region_06_120v_560c": ("Hebosid", "Eta Biomnisnub Decachoron", ""),
    "kidecaic_region_07_120v_680c": ("Gadwebosid", "Gamma Disphenoidal Biomnisnub Decachoron",
                                     "Region 03 (Bedwebosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_region_08_120v_680c": ("Dedwebosid", "Delta Disphenoidal Biomnisnub Decachoron", ""),
    "kidecaic_region_09_120v_680c": ("Edwebosid", "Epsilon Disphenoidal Biomnisnub Decachoron", ""),
    "kidecaic_region_21_120v_680c": ("Zedwebosid", "Zeta Disphenoidal Biomnisnub Decachoron", ""),
    "kidecaic_region_22_120v_380c": ("Apybosid", "Alpha Pyramidic Biomnisnub Decachoron",
                                     "Walls 14 and 15 (Epybosid, Zepybosid) have the same columns but are walls; this "
                                     "is the region."),
    "kidecaic_wall_10_120v_380c": ("Bepybosid", "Beta Pyramidic Biomnisnub Decachoron",
                                   "Wall 16 (Hepybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_11_120v_440c": ("Abybosid", "Alpha Bipyramidic Biomnisnub Decachoron",
                                   "Wall 17 (Bebybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_12_120v_500c": ("Gapybosid", "Gamma Pyramidic Biomnisnub Decachoron",
                                   "Wall 19 (Thepybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_13_120v_500c": ("Depybosid", "Delta Pyramidic Biomnisnub Decachoron",
                                   "Its edges at a vertex are 56688,44,3^7; the list has 55668,44,3^7 (also for "
                                   "Ipybosid). Wall 20 (Ipybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_14_120v_380c": ("Epybosid", "Epsilon Pyramidic Biomnisnub Decachoron",
                                   "Wall 15 (Zepybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_15_120v_380c": ("Zepybosid", "Zeta Pyramidic Biomnisnub Decachoron",
                                   "Wall 14 (Epybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_16_120v_380c": ("Hepybosid", "Eta Pyramidic Biomnisnub Decachoron",
                                   "Wall 10 (Bepybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_17_120v_440c": ("Bebybosid", "Beta Bipyramidic Biomnisnub Decachoron",
                                   "Wall 11 (Abybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_18_120v_440c": ("Gabybosid", "Gamma Bipyramidic Biomnisnub Decachoron",
                                   "Its edges at a vertex are 88,4^6,3^6 (as for Debybosid and Ebybosid); the list has "
                                   "688,4^3,3^8, Abybosid's and Bebybosid's. Placed by the list's order: three walls "
                                   "have 88,4^6,3^6 and only two 688,4^3,3^8."),
    "kidecaic_wall_19_120v_500c": ("Thepybosid", "Theta Pyramidic Biomnisnub Decachoron",
                                   "Wall 12 (Gapybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_20_120v_500c": ("Ipybosid", "Iota Pyramidic Biomnisnub Decachoron",
                                   "Its edges at a vertex are 56688,44,3^7; the list has 55668,44,3^7. Wall 13 "
                                   "(Depybosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_23_120v_440c": ("Debybosid", "Delta Bipyramidic Biomnisnub Decachoron",
                                   "Walls 18 and 24 have the same columns; told apart " + ORDER + "."),
    "kidecaic_wall_24_120v_440c": ("Ebybosid", "Epsilon Bipyramidic Biomnisnub Decachoron",
                                   "Walls 18 and 23 have the same columns; told apart " + ORDER + "."),
    "iodecaic_region_01_120v_500c": ("Dawosbid", "Disphenoidal Omnisnub Bidecachoron", ""),
    "iodecaic_wall_02_120v_260c": ("Bosbid", "Bipyramidal Omnisnub Bidecachoron", ""),
    "kidecaic_line_25_120v_200c": ("Sawbosid", "Small Antiwedged Biomnisnub Decachoron",
                                   "Line 26 (Gawbosid) has the same columns; told apart " + ORDER + "."),
    "kidecaic_line_26_120v_200c": ("Gawbosid", "Great Antiwedged Biomnisnub Decachoron",
                                   "The list gives it no degrees of variation; the atlas finds one (it lies on a curve "
                                   "that three of the map's walls contain). Line 25 (Sawbosid) has the same columns; "
                                   "told apart " + ORDER + "."),
}

# the named entries of the original page whose names changed (old heading -> (name, full name))
RENAMED = {"Sabosid": ("Abosid", "Alpha Biomnisnub Decachoron"),
           "Mebosid": ("Bebosid", "Beta Biomnisnub Decachoron"),
           "Gabosid": ("Gabosid", "Gamma Biomnisnub Decachoron")}


def given(name):
    """(short, full, note) for an atlas name (with or without the _NNNv_NNNc ending), or None."""
    for k, v in GIVEN.items():
        if name == k or k.startswith(name + "_") or name.startswith(k):
            return v
    return None
