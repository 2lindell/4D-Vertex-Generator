"""Suggested names for unnamed atlas shapes, deduced from relations to named ones (not on the accepted lists).

The list names each ixoic '... Snub Bitetrahedral 240' that is also ixitic with an ixitic twin '... Snub Bitetrahedral
Dis120' (Sobted / Sabtedy, Sagybted / Sagyptady, Gypsobted / Gypsobtady, Petsobted / Petsobtady, Awosbited /
Awosbitdy, ...). hyic_atlas/ixitic_relations.py finds three more ixoic shapes that are the same polytope as an unnamed
ixitic one (same vertices, same edges at each valence, the stabilising half-turn not in ixitic); their twins get the
same treatment here.
"""

# (atlas name -> (suggested short name, suggested full name, why))
SUGGESTED = {
    "ixitic_wall_01": ("Tetsobtady", "Tetrahedral Transitional Snub Bitetrahedral Dis120",
                       "Suggested from ixoic Tetsobted (1440 vertices), which is this wall's polychoron where its two "
                       "orbits of 120 etuts become equal and its symmetry doubles to ixoic (beta = (0.443263, 0.113474, "
                       "0.443263, 0) in sheet 5); the list's other ixoic and ixitic twins are named the same way."),
    "ixitic_region_12": ("Gagypsobtady", "Great Gyroprismatic Snub Bitetrahedral Dis120",
                         "Suggested from ixoic Gagypsobted (1440 vertices, edges 88,4^8,3^4), the same polychoron on "
                         "the ixoic half-turn line through this region; the list's other ixoic and ixitic twins are "
                         "named the same way."),
    "ixitic_line_01": ("Regposbitdy", "Rectangular Gyroprismatic Snub Bitetrahedral Dis120",
                       "Suggested from ixoic Regposbitad (1440 vertices, edges 44,3^8), the same polychoron at a point of "
                       "this line where its symmetry doubles to ixoic; named as Awosbited's twin Awosbitdy."),
}

# (atlas name -> note on a relation that does not give a name)
RELATED = {
    "magypsobtady_1": "It is the same polychoron as the unnamed ixoic line 85a06f9fec4f, and magypsobtady_2 as the "
                      "ixoic line 612fa060af7c: the list's Magypsobted and its unnamed ixoic entry with the same counts "
                      "(555566,4^4,3^4) are these two lines, so settling which is Magypsobted settles which is "
                      "Magypsobtady.",
    "magypsobtady_2": "It is the same polychoron as the unnamed ixoic line 612fa060af7c, and magypsobtady_1 as the "
                      "ixoic line 85a06f9fec4f: the list's Magypsobted and its unnamed ixoic entry with the same counts "
                      "(555566,4^4,3^4) are these two lines, so settling which is Magypsobted settles which is "
                      "Magypsobtady.",
}
