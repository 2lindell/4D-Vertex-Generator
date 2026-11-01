# Swirlprism 3-sphere map

Built from `research/swirlprism_atlas` on `claude/laughing-noether-qydcwe` (commit `8488da9`). Symmetry: h4_swirlprism (order 1200). Domain: half of one 600-cell tetrahedron: beta >= 0, beta3 >= beta4 (barycentric on cell_vertices); 1200 copies tile S^3 once.

| | count |
|---|---|
| surfaces | 182 |
| surfaces from source | 162 |
| pieces | 11683 |
| patches | 3985 |
| regions | 40 |
| region components | 88800 |
| walls | 55 |
| wall components | 88640 |
| lines | 24 |
| points | 14 |

## Checks

- Pieces whose sample points named different regions: 0
- Patch naming: {'data': 3409, 'ambiguous+classified': 482, 'data said X26+classified': 14, 'inside region+classified (4/4 nearby agree)': 44, 'inside region+classified (2/2 nearby agree)': 15, 'inside region+classified (3/3 nearby agree)': 21}
- Patch names checked by classifying a point on each patch: {'data name confirmed': 3407, 'settled by classification': 482, 'data name corrected by classification': 14, 'data name contradicted': 2}
- Surface walks: {'crossings': 701698, 'crossings at a meeting curve': 52108, 'left domain': 57741, 'unseen pieces': 1187, 'walks': 58380, 'still walking at the end': 639}
- Walks continued across domain faces: {'glued': 24675, 'not on a library surface there': 32490, 'unseen pieces': 12, 'continues in a domain face': 527, 'face_patch_links': {'unresolved': 363, 'linked': 164}, 'face_walks': {'face crossings': 28202, 'left the face': 1517, 'into a face beyond the edge': 622, 'into a surface beyond the edge': 825}}
- Patch sides sampled just off each patch: {'sides confirmed': 3867, 'side not classified': 4, 'sides corrected': 14}
- Shape changes on no atlas surface: 0
- Face gluing: {'glued': 27527, 'at a line or corner': 304}
- Exploration: {'chords': 200000, 'rejected_chords': 3, 'coincident_crossings': 20, 'pieces_seen_once': 125}

## Findings against the working data

- Patch 3088 is named X99 from the data, but a point on it does not give X99's counts.
- Patch 3119 is named X17 from the data, but a point on it does not give X17's counts.

Cross-check with the atlas's `bounds_walk.json` (walls found by walking out of a shape):

| region | walls from bounds_walk | walls on this map's component(s) | agree |
|---|---|---|---|
| X102 | X26, X79, X86 | X26, X79, X86 | yes |
| X103 | X101, X26, X89 | X101, X26, X89 | yes |
| X72 | X101, X26, X91 | X101, X26, X91 | yes |
| X73 | X26, X79, X88 | X26, X79, X88 | yes |

## Regions

Volume share is of the whole 3-sphere. A component is one connected volume; its stabilizer is the subgroup mapping it onto itself, so components = 1200 / stabilizer for each orbit.

The half-turn Q (outside the group, normalizing it) maps every shape to a congruent one; 'Q pairs' shows the orbit each orbit's Q image lies in (0<->1 means Q swaps them).

| region | shape | volume | components | stabilizer | domain copies per component | Q pairs |
|---|---|---|---|---|---|---|
| X49 | Piece of the cantitruncated 600-cell, omnitruncated 120-cell | 24.15% | 1200 + 1200 | 1, 1 | 2, 2 | 0->1, 1->0 |
| F3 | Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+1200×9 faces) | 18.31% | 1200 | 1 | 3 | 0->0 |
| F2 | Polychoron with 120+120+240+600+600+1200+1200 cells | 13.69% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X58 | Not in the wiki list | 5.59% | 1200 + 1200 | 1, 1 | 2, 1 | 0->1, 1->0 |
| X61 | Not in the wiki list | 4.60% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X32 | Not in the wiki list | 3.72% | 1200 | 1 | 1 | 0->0 |
| X50 | Not in the wiki list | 3.39% | 1200 + 1200 | 1, 1 | 3, 1 | 0->1, 1->0 |
| X16 | Not in the wiki list | 2.45% | 1200 | 1 | 1 | 0->0 |
| X68 | Not in the wiki list | 1.98% | 1200 | 1 | 1 | 0->0 |
| X66 | Not in the wiki list | 1.98% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X46 | Piece of the omnitruncated 120-cell | 1.72% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X60 | Not in the wiki list | 1.61% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X63 | Not in the wiki list | 1.59% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X95 | Not in the wiki list | 1.44% | 1200 + 1200 | 1, 1 | 2, 1 | 0->1, 1->0 |
| X43 | Not in the wiki list | 1.34% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X37 | Not in the wiki list | 1.26% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X57 | Not in the wiki list | 1.17% | 1200 + 1200 | 1, 1 | 2, 1 | 0->1, 1->0 |
| X36 | Not in the wiki list | 1.15% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X54 | Not in the wiki list | 1.12% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X59 | Not in the wiki list | 1.03% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X62 | Not in the wiki list | 0.94% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X12 | Not in the wiki list | 0.77% | 1200 | 1 | 1 | 0->0 |
| X30 | Not in the wiki list | 0.65% | 1200 | 1 | 1 | 0->0 |
| X64 | Not in the wiki list | 0.61% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X56 | Not in the wiki list | 0.52% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X69 | Not in the wiki list | 0.49% | 1200 + 1200 | 1, 1 | 1, 2 | 0->1, 1->0 |
| X55 | Not in the wiki list | 0.46% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X45 | Not in the wiki list | 0.45% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X67 | Not in the wiki list | 0.45% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X65 | Not in the wiki list | 0.31% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X70 | Not in the wiki list | 0.20% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X72 | Not in the wiki list | 0.17% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X103 | Not in the wiki list | 0.16% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X73 | Not in the wiki list | 0.16% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X102 | Not in the wiki list | 0.11% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X74 | Not in the wiki list | 0.08% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X75 | Not in the wiki list | 0.06% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X77 | Not in the wiki list | 0.05% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X71 | Not in the wiki list | 0.04% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |
| X76 | Not in the wiki list | 0.03% | 1200 + 1200 | 1, 1 | 1, 1 | 0->1, 1->0 |

## How each region is bounded

For one component of each region: each wall on its boundary, the region across it, how many patches (exact pieces of wall, see `patches.json`) that face is made of, and whether the face is the whole of a wall sheet or part of one.

### F2: Polychoron with 120+120+240+600+600+1200+1200 cells

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 82 | 82/282 |
| T2 | X49 | 45 | 45/282 |
| X10 | X95 | 18 | 18/1080 |
| X11 | X36 | 35 | 35/102 |
| X11 | X37 | 16 | 16/102 |
| X14 | X45 | 17 | 17/27 |
| X14 | X46 | 10 | 10/27 |
| X21 | X60 | 13 | 13/19 |
| X21 | X67 | 6 | 6/19 |
| X3 | X12 | 51 | 51/134 |
| X9 | X16 | 60 | 60/122 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 39 | 39/176 |
| T2 | X49 | 39 | 39/176 |
| X10 | X95 | 35 | 35/70 |
| X11 | X36 | 21 | 21/80 |
| X11 | X37 | 19 | 19/80 |
| X14 | X45 | 16 | 16/112 |
| X14 | X46 | 40 | 40/112 |
| X21 | X60 | 13 | 13/24 |
| X21 | X67 | 11 | 11/24 |
| X3 | X12 | 42 | 42/102 |
| X9 | X16 | 237 | 236/480, 1/1 whole |

### F3: Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+1200×9 faces)

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F1 | X49 | 250 | 134/288, 116/256 |
| F1 | X50 | 22 | 10/288, 12/256 |
| T2 | F2 | 121 | 82/282, 39/176 |
| T2 | X36 | 8 | 4/282, 4/176 |
| T2 | X37 | 16 | 10/282, 6/176 |
| X22 | X54 | 34 | 15/40, 19/25 |
| X22 | X70 | 11 | 5/40, 6/25 |

### X102: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X26 | X103 | 3 | 3/130 |
| X79 | X73 | 7 | 7/7 whole |
| X86 | X71 | 3 | 3/6 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X26 | X103 | 2 | 2/80 |
| X79 | X73 | 3 | 3/3 whole |
| X86 | X71 | 2 | 2/4 |

### X103: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X101 | X72 | 10 | 10/10 whole |
| X26 | X102 | 3 | 3/130 |
| X89 | X76 | 14 | 14/28 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X101 | X72 | 4 | 4/4 whole |
| X26 | X102 | 2 | 2/80 |
| X89 | X76 | 10 | 10/20 |

### X12: Not in the wiki list

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X3 | F2 | 93 | 51/134, 42/102 |
| X3 | X95 | 25 | 16/134, 9/102 |
| X7 | X30 | 154 | 154/154 whole |

### X16: Not in the wiki list

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X9 | F2 | 296 | 60/122, 235/480, 1/1 whole |
| X9 | X95 | 6 | 1/122, 5/480 |
| X96 | X58 | 23 | 13/44, 10/38 |
| X96 | X62 | 18 | 9/44, 9/38 |

### X30: Not in the wiki list

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X7 | X12 | 154 | 154/154 whole |
| X83 | X36 | 176 | 109/218, 67/134 |

### X32: Not in the wiki list

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X15 | X37 | 49 | 26/26 whole, 23/23 whole |
| X17 | X43 | 5 | 3/17, 2/13 |
| X17 | X54 | 15 | 8/17, 7/13 |
| X19 | X46 | 33 | 24/240, 9/18 |

### X36: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 4 | 4/282 |
| X11 | F2 | 35 | 35/102 |
| X83 | X30 | 109 | 109/218 |
| X94 | X37 | 66 | 66/110 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 4 | 4/176 |
| X11 | F2 | 21 | 21/80 |
| X83 | X30 | 67 | 67/134 |
| X94 | X37 | 44 | 44/110 |

### X37: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 10 | 10/282 |
| X11 | F2 | 16 | 16/102 |
| X15 | X32 | 26 | 26/26 whole |
| X94 | X36 | 66 | 66/110 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| T2 | F3 | 6 | 6/176 |
| X11 | F2 | 19 | 19/80 |
| X15 | X32 | 23 | 23/23 whole |
| X94 | X36 | 44 | 44/110 |

### X43: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X17 | X32 | 3 | 3/17 |
| X17 | X46 | 6 | 6/17 |
| X31 | X54 | 16 | 16/16 whole |
| X82 | X49 | 18 | 18/36 |
| X85 | X71 | 3 | 3/3 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X17 | X32 | 2 | 2/13 |
| X17 | X46 | 4 | 4/13 |
| X31 | X54 | 2 | 2/2 whole |
| X82 | X49 | 10 | 10/10 whole |
| X85 | X71 | 2 | 2/4 |

### X45: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X14 | F2 | 16 | 16/112 |
| X25 | X46 | 13 | 13/26 |
| X27 | X55 | 5 | 5/14 |
| X27 | X65 | 9 | 9/14 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X14 | F2 | 17 | 17/27 |
| X25 | X46 | 20 | 20/40 |
| X27 | X55 | 3 | 3/13 |
| X27 | X65 | 10 | 10/13 |

### X46: Piece of the omnitruncated 120-cell

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X14 | F2 | 40 | 40/112 |
| X17 | X43 | 6 | 6/17 |
| X19 | X32 | 24 | 24/240 |
| X25 | X45 | 13 | 13/26 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X14 | F2 | 10 | 10/27 |
| X17 | X43 | 4 | 4/13 |
| X19 | X32 | 9 | 9/18 |
| X25 | X45 | 20 | 20/40 |

### X49: Piece of the cantitruncated 600-cell, omnitruncated 120-cell

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F1 | F3 | 134 | 134/288 |
| F4 | X58 | 178 | 178/764 |
| F4 | X61 | 204 | 204/764 |
| T2 | F2 | 39 | 39/176 |
| X28 | X50 | 18 | 18/36 |
| X82 | X43 | 18 | 18/36 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F1 | F3 | 116 | 116/256 |
| F4 | X58 | 59 | 59/266 |
| F4 | X61 | 74 | 74/266 |
| T2 | F2 | 45 | 45/282 |
| X28 | X50 | 25 | 25/50 |
| X82 | X43 | 10 | 10/10 whole |

### X50: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F1 | F3 | 10 | 10/288 |
| X28 | X49 | 18 | 18/36 |
| X33 | X57 | 34 | 34/34 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F1 | F3 | 12 | 12/256 |
| X28 | X49 | 25 | 25/50 |
| X33 | X57 | 36 | 36/72 |

### X54: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X17 | X32 | 8 | 8/17 |
| X22 | F3 | 19 | 19/25 |
| X31 | X43 | 16 | 16/16 whole |
| X93 | X70 | 25 | 25/25 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X17 | X32 | 7 | 7/13 |
| X22 | F3 | 15 | 15/40 |
| X31 | X43 | 2 | 2/2 whole |
| X93 | X70 | 20 | 20/20 whole |

### X55: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X27 | X45 | 3 | 3/13 |
| X34 | X60 | 20 | 20/40 |
| X44 | X65 | 14 | 14/14 whole |
| X90 | X76 | 16 | 16/32 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X27 | X45 | 5 | 5/14 |
| X34 | X60 | 10 | 10/20 |
| X44 | X65 | 11 | 11/11 whole |
| X90 | X76 | 13 | 13/26 |

### X56: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X35 | X74 | 14 | 14/28 |
| X42 | X65 | 51 | 51/51 whole |
| X92 | X77 | 24 | 24/24 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X35 | X74 | 9 | 9/18 |
| X42 | X65 | 26 | 26/26 whole |
| X92 | X77 | 11 | 11/22 |

### X57: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X33 | X50 | 34 | 34/34 whole |
| X40 | X63 | 17 | 17/34 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X33 | X50 | 36 | 36/72 |
| X40 | X63 | 23 | 23/46 |

### X58: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F4 | X49 | 59 | 59/266 |
| X100 | X61 | 33 | 33/66 |
| X41 | X62 | 28 | 28/1120 |
| X96 | X16 | 10 | 10/38 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F4 | X49 | 182 | 182/764 |
| X100 | X61 | 8 | 8/8 whole |
| X41 | X62 | 61 | 61/2440 |
| X96 | X16 | 13 | 13/44 |

### X59: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F5 | X66 | 9 | 9/9 whole |
| X20 | X95 | 1 | 1/38 |
| X23 | X62 | 10 | 10/118 |
| X39 | X60 | 7 | 7/7 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F5 | X66 | 9 | 9/9 whole |
| X20 | X95 | 5 | 5/66 |
| X23 | X62 | 25 | 25/160 |
| X39 | X60 | 7 | 7/7 whole |

### X60: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X21 | F2 | 13 | 13/24 |
| X23 | X62 | 55 | 55/160 |
| X34 | X55 | 10 | 10/20 |
| X39 | X59 | 7 | 7/7 whole |
| X98 | X67 | 11 | 11/11 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X21 | F2 | 13 | 13/19 |
| X23 | X62 | 49 | 49/118 |
| X34 | X55 | 20 | 20/40 |
| X39 | X59 | 7 | 7/7 whole |
| X98 | X67 | 26 | 26/26 whole |

### X61: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F4 | X49 | 74 | 74/266 |
| X100 | X58 | 33 | 33/66 |
| X99 | X68 | 106 | 106/212 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F4 | X49 | 200 | 200/764 |
| X100 | X58 | 8 | 8/8 whole |
| X99 | X68 | 143 | 143/286 |

### X62: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X23 | X59 | 24 | 24/160 |
| X23 | X60 | 56 | 56/160 |
| X41 | X58 | 28 | 28/1120 |
| X96 | X16 | 9 | 9/38 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X23 | X59 | 10 | 10/118 |
| X23 | X60 | 49 | 49/118 |
| X41 | X58 | 61 | 61/2440 |
| X96 | X16 | 9 | 9/44 |

### X63: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X40 | X57 | 23 | 23/46 |
| X51 | X64 | 30 | 30/30 whole |
| X84 | X74 | 10 | 10/13 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X40 | X57 | 17 | 17/34 |
| X51 | X64 | 21 | 21/21 whole |
| X84 | X74 | 11 | 11/32 |

### X64: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X51 | X63 | 30 | 30/30 whole |
| X53 | X69 | 18 | 18/36 |
| X84 | X74 | 3 | 3/13 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X51 | X63 | 21 | 21/21 whole |
| X53 | X69 | 20 | 20/40 |
| X84 | X74 | 5 | 5/32 |

### X65: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X27 | X45 | 10 | 10/13 |
| X42 | X56 | 51 | 51/51 whole |
| X44 | X55 | 14 | 14/14 whole |
| X47 | X67 | 21 | 21/21 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X27 | X45 | 9 | 9/14 |
| X42 | X56 | 26 | 26/26 whole |
| X44 | X55 | 11 | 11/11 whole |
| X47 | X67 | 8 | 8/8 whole |

### X66: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F5 | X59 | 9 | 9/9 whole |
| X20 | X95 | 18 | 18/38 |
| X48 | X69 | 20 | 20/40 |
| X52 | X67 | 11 | 11/11 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| F5 | X59 | 9 | 9/9 whole |
| X20 | X95 | 28 | 28/66 |
| X48 | X69 | 22 | 22/44 |
| X52 | X67 | 4 | 4/4 whole |

### X67: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X21 | F2 | 6 | 6/19 |
| X47 | X65 | 21 | 21/21 whole |
| X52 | X66 | 11 | 11/11 whole |
| X98 | X60 | 26 | 26/26 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X21 | F2 | 11 | 11/24 |
| X47 | X65 | 8 | 8/8 whole |
| X52 | X66 | 4 | 4/4 whole |
| X98 | X60 | 11 | 11/11 whole |

### X68: Not in the wiki list

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X38 | X68 | 160 | 160/160 whole |
| X99 | X61 | 249 | 106/212, 143/286 |

### X69: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X48 | X66 | 20 | 20/40 |
| X53 | X64 | 18 | 18/36 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X48 | X66 | 22 | 22/44 |
| X53 | X64 | 20 | 20/40 |

### X70: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X22 | F3 | 6 | 6/25 |
| X87 | X75 | 32 | 32/32 whole |
| X93 | X54 | 25 | 25/25 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X22 | F3 | 5 | 5/40 |
| X87 | X75 | 14 | 14/28 |
| X93 | X54 | 20 | 20/20 whole |

### X71: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X85 | X43 | 3 | 3/3 whole |
| X86 | X102 | 3 | 3/6 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X85 | X43 | 2 | 2/4 |
| X86 | X102 | 2 | 2/4 |

### X72: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X101 | X103 | 10 | 10/10 whole |
| X26 | X73 | 10 | 10/130 |
| X91 | X77 | 21 | 21/21 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X101 | X103 | 4 | 4/4 whole |
| X26 | X73 | 6 | 6/80 |
| X91 | X77 | 9 | 9/18 |

### X73: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X26 | X72 | 10 | 10/130 |
| X79 | X102 | 7 | 7/7 whole |
| X88 | X75 | 28 | 28/56 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X26 | X72 | 6 | 6/80 |
| X79 | X102 | 3 | 3/3 whole |
| X88 | X75 | 12 | 12/24 |

### X74: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X35 | X56 | 14 | 14/28 |
| X84 | X63 | 10 | 10/13 |
| X84 | X64 | 3 | 3/13 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X35 | X56 | 9 | 9/18 |
| X84 | X63 | 11 | 11/32 |
| X84 | X64 | 5 | 5/32 |

### X75: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X87 | X70 | 32 | 32/32 whole |
| X88 | X73 | 28 | 28/56 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X87 | X70 | 14 | 14/28 |
| X88 | X73 | 12 | 12/24 |

### X76: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X89 | X103 | 14 | 14/28 |
| X90 | X55 | 16 | 16/32 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X89 | X103 | 10 | 10/20 |
| X90 | X55 | 13 | 13/26 |

### X77: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X91 | X72 | 21 | 21/21 whole |
| X92 | X56 | 24 | 24/24 whole |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X91 | X72 | 9 | 9/18 |
| X92 | X56 | 11 | 11/22 |

### X95: Not in the wiki list

Orbit 0 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X10 | F2 | 35 | 35/70 |
| X20 | X59 | 5 | 5/66 |
| X20 | X66 | 28 | 28/66 |
| X3 | X12 | 9 | 9/102 |
| X9 | X16 | 4 | 4/480 |

Orbit 1 (1200 components):

| wall | across | parts | wall sheets (parts here / in sheet) |
|---|---|---|---|
| X10 | F2 | 18 | 18/1080 |
| X20 | X59 | 1 | 1/38 |
| X20 | X66 | 18 | 18/38 |
| X3 | X12 | 16 | 16/134 |
| X9 | X16 | 1 | 1/122 |

## Walls

| wall | shape | regions found | sheets on S^3 | stabilizer | patches |
|---|---|---|---|---|---|
| F1 | Polychoron with 120+120+600+600+600+600+1200 cells | F3, X49, X50 | 600 + 600 | 2, 2 | 272 |
| F4 | Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+600+1200×9 faces) | X49, X58, X61 | 600 + 600 | 2, 2 | 515 |
| F5 | Polychoron with 120+120+600+600+600+600+1200+1200+1200 cells | X59, X66 | 1200 + 1200 | 1, 1 | 18 |
| T2 | Transitional polychoron with 120+120+600+600+1200 cells | F2, F3, X36, X37, X49 | 600 + 600 | 2, 2 | 229 |
| X10 | Piece of the cantitruncated 600-cell | F2, X95 | 600 + 20 | 2, 60 | 53 |
| X100 | Not in the wiki list | X58, X61 | 1200 + 600 | 1, 2 | 41 |
| X101 | Not in the wiki list | X103, X72 | 1200 + 1200 | 1, 1 | 14 |
| X11 | Not in the wiki list | F2, X36, X37 | 600 + 600 | 2, 2 | 91 |
| X14 | Not in the wiki list | F2, X45, X46 | 600 + 1200 | 2, 1 | 83 |
| X15 | Not in the wiki list | X32, X37 | 1200 + 1200 | 1, 1 | 49 |
| X17 | Not in the wiki list | X32, X43, X46, X54 | 1200 + 1200 | 1, 1 | 30 |
| X19 | Piece of the runcitruncated 120-cell | X32, X46 | 120 + 600 | 10, 2 | 33 |
| X20 | Piece of the cantitruncated 120-cell | X59, X66, X95 | 600 + 600 | 2, 2 | 52 |
| X21 | Piece of the cantitruncated 120-cell | F2, X60, X67 | 1200 + 1200 | 1, 1 | 43 |
| X22 | Not in the wiki list | F3, X54, X70 | 1200 + 600 | 1, 2 | 45 |
| X23 | Not in the wiki list | X59, X60, X62 | 600 + 600 | 2, 2 | 139 |
| X25 | Not in the wiki list | X45, X46 | 600 + 600 | 2, 2 | 33 |
| X26 | Piece of the cantitruncated 120-cell | X102, X103, X72, X73 | 120 + 120 | 10, 10 | 21 |
| X27 | Not in the wiki list | X45, X55, X65 | 1200 + 1200 | 1, 1 | 27 |
| X28 | Not in the wiki list | X49, X50 | 600 + 600 | 2, 2 | 43 |
| X3 | Piece of the cantitruncated 120-cell | F2, X12, X95 | 600 + 600 | 2, 2 | 118 |
| X31 | Not in the wiki list | X43, X54 | 1200 + 1200 | 1, 1 | 18 |
| X33 | Not in the wiki list | X50, X57 | 1200 + 600 | 1, 2 | 70 |
| X34 | Not in the wiki list | X55, X60 | 600 + 600 | 2, 2 | 30 |
| X35 | Not in the wiki list | X56, X74 | 600 + 600 | 2, 2 | 23 |
| X38 | Not in the wiki list | X68 | 600 | 2 | 80 |
| X39 | Not in the wiki list | X59, X60 | 1200 + 1200 | 1, 1 | 14 |
| X40 | Not in the wiki list | X57, X63 | 600 + 600 | 2, 2 | 40 |
| X41 | Piece of the omnitruncated 120-cell | X58, X62 | 30 + 30 | 40, 40 | 89 |
| X42 | Not in the wiki list | X56, X65 | 1200 + 1200 | 1, 1 | 77 |
| X44 | Not in the wiki list | X55, X65 | 1200 + 1200 | 1, 1 | 25 |
| X47 | Not in the wiki list | X65, X67 | 1200 + 1200 | 1, 1 | 29 |
| X48 | Not in the wiki list | X66, X69 | 600 + 600 | 2, 2 | 42 |
| X51 | Not in the wiki list | X63, X64 | 1200 + 1200 | 1, 1 | 51 |
| X52 | Not in the wiki list | X66, X67 | 1200 + 1200 | 1, 1 | 15 |
| X53 | Not in the wiki list | X64, X69 | 600 + 600 | 2, 2 | 38 |
| X7 | Piece of the omnitruncated 120-cell | X12, X30 | 1200 | 1 | 154 |
| X79 | Not in the wiki list | X102, X73 | 1200 + 1200 | 1, 1 | 10 |
| X82 | Not in the wiki list | X43, X49 | 600 + 1200 | 2, 1 | 28 |
| X83 | Not in the wiki list | X30, X36 | 600 + 600 | 2, 2 | 176 |
| X84 | Not in the wiki list | X63, X64, X74 | 1200 + 600 | 1, 2 | 29 |
| X85 | Not in the wiki list | X43, X71 | 1200 + 600 | 1, 2 | 5 |
| X86 | Not in the wiki list | X102, X71 | 600 + 600 | 2, 2 | 5 |
| X87 | Not in the wiki list | X70, X75 | 1200 + 600 | 1, 2 | 46 |
| X88 | Not in the wiki list | X73, X75 | 600 + 600 | 2, 2 | 40 |
| X89 | Not in the wiki list | X103, X76 | 600 + 600 | 2, 2 | 24 |
| X9 | Piece of the cantitruncated 600-cell | F2, X16, X95 | 600 + 600 + 1200 | 2, 2, 1 | 302 |
| X90 | Not in the wiki list | X55, X76 | 600 + 600 | 2, 2 | 29 |
| X91 | Not in the wiki list | X72, X77 | 1200 + 600 | 1, 2 | 30 |
| X92 | Not in the wiki list | X56, X77 | 1200 + 600 | 1, 2 | 35 |
| X93 | Not in the wiki list | X54, X70 | 1200 + 1200 | 1, 1 | 45 |
| X94 | Piece of the runcitruncated 120-cell | X36, X37 | 1200 | 1 | 110 |
| X96 | Not in the wiki list | X16, X58, X62 | 600 + 600 | 2, 2 | 41 |
| X98 | Not in the wiki list | X60, X67 | 1200 + 1200 | 1, 1 | 37 |
| X99 | Not in the wiki list | X61, X68 | 600 + 600 | 2, 2 | 249 |

## Faces

A face is one connected piece of one wall sheet between one region component and another. Faces with the same labels must be congruent: the test maps every orbit's face onto the first orbit's by an element of G or G.Q and measures how much of each lies on the other.

- 74 face labels, 150 orbits, 180000 faces on the sphere.
- Congruence, tested on the sphere (each face a dense cloud of exact points, carried onto the first orbit's face by G or G.Q): {'congruent': 139, 'not congruent': 4, 'fragments merged into a face': 2, 'congruent where both are known': 7}.
  - F4 | X49 | X58, orbit 1: congruent where both are known (share of the first face matched 1.0, of this one 0.9987; 178 patch(es))
  - F4 | X49 | X61, orbit 2: not congruent (share of the first face matched 0.0, of this one 0.0; 6 patch(es))
  - X23 | X59 | X62, orbit 2: not congruent (share of the first face matched 0.0, of this one 0.0; 2 patch(es))
  - X23 | X60 | X62, orbit 1: congruent where both are known (share of the first face matched 1.0, of this one 0.9943; 53 patch(es))
  - X23 | X60 | X62, orbit 2: not congruent (share of the first face matched 0.0, of this one 0.0; 3 patch(es))
  - X40 | X57 | X63, orbit 1: congruent where both are known (share of the first face matched 0.9988, of this one 1.0; 17 patch(es))
  - X53 | X64 | X69, orbit 1: congruent where both are known (share of the first face matched 0.9987, of this one 1.0; 20 patch(es))
  - X79 | X102 | X73, orbit 1: congruent where both are known (share of the first face matched 1.0, of this one 0.9954; 3 patch(es))
  - X89 | X103 | X76, orbit 1: congruent where both are known (share of the first face matched 0.9975, of this one 1.0; 10 patch(es))
  - X9 | F2 | X16, orbit 1: congruent where both are known (share of the first face matched 1.0, of this one 0.9984; 235 patch(es))
  - X9 | F2 | X16, orbit 2: not congruent (share of the first face matched 0.0, of this one 0.0; 1 patch(es))
- Edges: 1789; corners: 398 (389 solved exactly, 97 are atlas points).

| face (wall \| regions) | orbits | faces on S^3 | corners | perimeter | congruent |
|---|---|---|---|---|---|
| F1 | F3 | X49 | 2 | 2400 | 19 | 0.993167 | yes |
| F1 | F3 | X50 | 2 | 2400 | 8 | 0.841224 | yes |
| F4 | X49 | X58 | 2 | 2400 | 8 | 0.362140 | yes |
| F4 | X49 | X61 | 3 | 3600 | 16 | 1.585668 | no |
| F5 | X59 | X66 | 2 | 2400 | 5 | 0.960874 | yes |
| T2 | F2 | F3 | 2 | 2400 | 8 | 0.136672 | yes |
| T2 | F2 | X49 | 2 | 2400 | 8 | 0.089569 | yes |
| T2 | F3 | X36 | 2 | 2400 | 3 | 0.562893 | yes |
| T2 | F3 | X37 | 2 | 2400 | 3 | 0.245456 | yes |
| X10 | F2 | X95 | 2 | 2400 | 8 | 0.822955 | yes |
| X100 | X58 | X61 | 2 | 2400 | 5 | 0.957351 | yes |
| X101 | X103 | X72 | 2 | 2400 | 4 | 1.390427 | yes |
| X11 | F2 | X36 | 2 | 2400 | 7 | 0.526091 | yes |
| X11 | F2 | X37 | 2 | 2400 | 6 | 0.332235 | yes |
| X14 | F2 | X45 | 2 | 2400 | 19 | 1.710365 | yes |
| X14 | F2 | X46 | 2 | 2400 | 11 | 1.626210 | yes |
| X15 | X32 | X37 | 2 | 2400 | 5 | 0.469246 | yes |
| X17 | X32 | X43 | 2 | 2400 | 3 | 0.317405 | yes |
| X17 | X32 | X54 | 2 | 2400 | 4 | 0.433105 | yes |
| X17 | X43 | X46 | 2 | 2400 | 2 | 0.173874 | yes |
| X19 | X32 | X46 | 2 | 2400 | 6 | 0.084305 | yes |
| X20 | X59 | X95 | 2 | 2400 | 3 | 0.162085 | yes |
| X20 | X66 | X95 | 2 | 2400 | 3 | 0.182839 | yes |
| X21 | F2 | X60 | 2 | 2400 | 4 | 0.284027 | yes |
| X21 | F2 | X67 | 2 | 2400 | 3 | 0.245530 | yes |
| X22 | F3 | X54 | 2 | 2400 | 6 | 0.547025 | yes |
| X22 | F3 | X70 | 2 | 2400 | 4 | 0.520827 | yes |
| X23 | X59 | X62 | 3 | 3600 | 5 | 0.388377 | no |
| X23 | X60 | X62 | 3 | 3600 | 5 | 0.393260 | no |
| X25 | X45 | X46 | 2 | 2400 | 10 | 1.752283 | yes |
| X26 | X102 | X103 | 2 | 2400 | 3 | 0.198176 | yes |
| X26 | X72 | X73 | 2 | 2400 | 2 | 0.198176 | yes |
| X27 | X45 | X55 | 2 | 2400 | 3 | 0.213260 | yes |
| X27 | X45 | X65 | 2 | 2400 | 4 | 0.248002 | yes |
| X28 | X49 | X50 | 2 | 2400 | 11 | 0.815606 | yes |
| X3 | F2 | X12 | 2 | 2400 | 6 | 0.331791 | yes |
| X3 | X12 | X95 | 2 | 2400 | 9 | 0.848720 | yes |
| X31 | X43 | X54 | 2 | 2400 | 4 | 1.269553 | yes |
| X33 | X50 | X57 | 2 | 2400 | 7 | 1.177617 | yes |
| X34 | X55 | X60 | 2 | 2400 | 6 | 0.500173 | yes |
| X35 | X56 | X74 | 2 | 2400 | 4 | 1.235687 | yes |
| X38 | X68 | X68 | 1 | 1200 | 36 | 1.453663 | yes |
| X39 | X59 | X60 | 2 | 2400 | 5 | 0.555131 | yes |
| X40 | X57 | X63 | 2 | 2400 | 12 | 0.371346 | yes |
| X41 | X58 | X62 | 2 | 2400 | 17 | 2.157714 | yes |
| X42 | X56 | X65 | 2 | 2400 | 2 | 1.124860 | yes |
| X44 | X55 | X65 | 2 | 2400 | 6 | 1.214954 | yes |
| X47 | X65 | X67 | 2 | 2400 | 7 | 0.789123 | yes |
| X48 | X66 | X69 | 2 | 2400 | 17 | 1.794860 | yes |
| X51 | X63 | X64 | 2 | 2400 | 2 | 0.376021 | yes |
| X52 | X66 | X67 | 2 | 2400 | 6 | 1.366216 | yes |
| X53 | X64 | X69 | 2 | 2400 | 0 | 1.438053 | yes |
| X7 | X12 | X30 | 1 | 1200 | 0 | 0.000000 | yes |
| X79 | X102 | X73 | 2 | 2400 | 2 | 0.992224 | yes |
| X82 | X43 | X49 | 2 | 2400 | 6 | 0.317347 | yes |
| X83 | X30 | X36 | 2 | 2400 | 5 | 0.404597 | yes |
| X84 | X63 | X74 | 2 | 2400 | 2 | 0.240561 | yes |
| X84 | X64 | X74 | 2 | 2400 | 3 | 0.742493 | yes |
| X85 | X43 | X71 | 2 | 2400 | 0 | 1.063334 | yes |
| X86 | X102 | X71 | 2 | 2400 | 6 | 1.088574 | yes |
| X87 | X70 | X75 | 2 | 2400 | 0 | 0.650482 | yes |
| X88 | X73 | X75 | 2 | 2400 | 2 | 0.422467 | yes |
| X89 | X103 | X76 | 2 | 2400 | 2 | 0.484532 | yes |
| X9 | F2 | X16 | 3 | 3600 | 5 | 0.241428 | no |
| X9 | X16 | X95 | 2 | 2400 | 2 | 0.118273 | yes |
| X90 | X55 | X76 | 2 | 2400 | 0 | 1.434544 | yes |
| X91 | X72 | X77 | 2 | 2400 | 4 | 0.472682 | yes |
| X92 | X56 | X77 | 2 | 2400 | 0 | 0.704505 | yes |
| X93 | X54 | X70 | 2 | 2400 | 4 | 0.976503 | yes |
| X94 | X36 | X37 | 2 | 2400 | 2 | 0.221057 | yes |
| X96 | X16 | X58 | 2 | 2400 | 16 | 1.826668 | yes |
| X96 | X16 | X62 | 2 | 2400 | 13 | 1.822184 | yes |
| X98 | X60 | X67 | 2 | 2400 | 6 | 0.532438 | yes |
| X99 | X61 | X68 | 2 | 2400 | 1 | 2.958598 | yes |

## Lines split by the walls round them

| line | pieces | walls round each piece (in order) | points on no edge |
|---|---|---|---|
| B1 | 109 | F4; X38+X48+X68|X66+X68|X69; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+F3|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3+X68|X49; F1+F4+X49|F3+X61|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F3|X49+F4+X49|F3+X61|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X49|X61; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+X49|F3; F1+F3|X49+F4+X49|X61; F3|X49+F4+X49|F3+X61|X49; F1+F4+X49|F3+X61|X49; F1+F3|X49+F4+X49|X61; F1+F4+X49|F3+X61|X49; F4 | 0 |
| C2a | 33 | F4+F5+X49|X66+X61|X59; X96; X100+X58|X61+X61|X58; X16|X58+X58|X16+X96; X100+X58|X61+X61|X58; X16|X62+X62|X16+X96; X16|X58+X58|X16+X96; X16|X62+X62|X16+X96; X38; X16|X62+X62|X16+X96; X100+X58|X61+X61|X58; X16|X58+X58|X16+X96; X38; X16|X62+X62|X16+X96; X41+X96; X100+X58|X61+X61|X58; X38; X16|X58+X58|X16+X96; X100+X58|X61+X61|X58; X38; X100+X58|X61+X61|X58; X16|X58+X58|X16+X96; X38; X16|X58+X58|X16+X96; X38; X100+X58|X61+X61|X58; X16|X62+X62|X16+X96; X100+X58|X61+X61|X58; X16|X62+X62|X16+X96; X41+X58|X62+X62|X58; X38; X38+X48+X68|X66+X68|X69; T2 | 0 |
| C2b | 38 | X34+X55|X60+X60|X55; F2|X16+X16|F2+X9; X41+X58|X62+X62|X58; X23; F2|X16+X16|F2+X9; X14+X45|X46; F2|X16+X16|F2+X9; X9; F2|X16+X16|F2+X9; X25+X45|X46+X46|X45; F2|X16+X16|F2+X9; X9; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; X9; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X45+X14+X45|F2; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F2|X45+X14+X45|F2; F2|X46+X14+X46|F2; F2|X16+X16|F2+X9; F2|X46+X14+X46|F2; F4+X49|X58+X58|X49 | 0 |
| C3 | 26 | F2|X36+X11+X36|F2; X20+X21+X66|X67+X95|F2; T2+X95|F3+X95|X36; F1+F3|X49; X38; F2|X36+X11+X36|F2; F2|X12+X12|F2+X3; F2|X36+X11+X36|F2; F2|X12+X12|F2+X3; F2|X36+X11+X36|F2; F2|X12+X12|F2+X3; X30|X36+X36|X30+X83; F2|X36+X11+X36|F2; F2|X12+X12|F2+X3; F2|X36+X11+X36|F2; F2|X12+X12|F2+X3; F2|X95+X10+X95|F2; X3; F2|X95+X10+X95|F2; X3+X95|F2; F2|X95+X10+X95|F2; X30|X36+X36|X30+X83; X83; F2|X36+X11+X36|F2; X83; F2|X46+X14+X45|X46 | 0 |
| D1 | 49 | X28+X41+X58|X49+X62|X50; F2|X49+T2+X49|F2; X28+X49|X50+X50|X49; F2|X49+T2+X49|F2; X28+X49|X50+X50|X49; F2|X49+T2+X49|F2; X28+X49|X50+X50|X49; F2|X49+T2+X49|F2; X28+X49|X50+X50|X49; F2|X49+T2+X49|F2; F2|X36+X11+X36|F2; T2; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; F1+F3|X49; T2+X95|F3+X95|X36; T2; X38+X48+X66|X68+X69|X68; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; X48+X66|X69+X69|X66; X40+X57|X63+X63|X57; T2+X95|F3+X95|X36 | 0 |
| D2 | 65 | F4; X47+X64|X65+X64|X67; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; F4+X49|X58+X58|X49; X41+X58|X62+X62|X58; F4+X49|X58+X58|X49; X102|X71+X71|X102+X86; F4+X49|X58+X58|X49; X41+X58|X62+X62|X58; F4+X49|X58+X58|X49; X82; F1+F3|X49; X31+X43|X102+X54|X73+X79; X54|X71+X70|X102+X86+X93; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X23+X60|X62+X62|X60; X41+X58|X62+X62|X58; X23+X60|X62+X62|X60; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X23+X60|X62+X62|X60; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X23+X60|X62+X62|X60; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X41+X58|X62+X62|X58; X23+X60|X62+X62|X60; X41+X58|X62+X62|X58; X23; F4+X49|X58+X58|X49; X103|X76+X76|X103+X89; F4+X49|X58+X58|X49; X34+X55|X60+X60|X55; F4+X49|X58+X58|X49; X82; F4+X49|X58+X58|X49; F4 | 0 |
| D3 | 48 | X20+X21+X66|X67+X95|F2; X11+X19+X32|X37+X46|F2; X26+X72|X73+X73|X72; X67|X64+X67|X74+X84; F4; F1+X12|F3+X3+X95|X50; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X35+X56|X74+X74|X56; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; X63|X74+X74|X63+X84; F1+F3|X50+X50|F3; F2|X46+X14+X45|X46 | 0 |
| E1 | 218 | X31+X43|X102+X54|X73+X79; X100+X57|X63+X58|X63+X61|X57; X33; X100+X57|X63+X58|X63+X61|X57; X51+X61|X63+X61|X64; X100+X57|X63+X58|X63+X61|X57; X28+X41+X58|X49+X62|X50; X100+X57|X61+X64|X57+X64|X58; X28+X41+X58|X49+X62|X50; X100+X63|X61+X63|X64+X64|X58; X28+X41+X58|X49+X62|X50; X100+X58|X64+X61|X57+X64|X57; X28+X41+X58|X49+X62|X50; X100+X58|X64+X61|X57+X64|X57; X28+X41+X58|X49+X62|X50; X100+X57|X61+X57|X63+X63|X58; X28+X41+X58|X49+X62|X50; X100+X57|X61+X57|X63+X63|X58; X28+X41+X58|X49+X62|X50; X100+X57|X61+X57|X63+X63|X58; X100+X57|X61+X63|X57+X63|X58; X28+X41+X58|X49+X62|X50; X48+X61|X69+X68|X61+X68|X66; X28+X41+X58|X49+X62|X50; X100+X63|X58+X63|X61; X28+X41+X58|X49+X62|X50; X33; X28+X41+X58|X49+X62|X50; X33; X100+X63|X58+X63|X61; X28+X41+X58|X49+X62|X50; X33; X28+X41+X58|X49+X62|X50; F1+F3|X49; X33; X28+X41+X58|X49+X62|X50; X33; X28+X41+X58|X49+X62|X50; X100+X57|X61+X57|X63+X63|X58; X28+X41+X58|X49+X62|X50; X100+X58|X64+X61|X57+X64|X57; X100+X63|X61+X63|X64+X64|X58; X28+X41+X58|X49+X62|X50; X100+X57|X61+X64|X57+X64|X58; X28+X41+X58|X49+X62|X50; X100+X57|X63+X58|X63+X61|X57; X61|X69+X68|X63+X69|X63+X99; X100+X57|X63+X58|X63+X61|X57; X51+X61|X63+X61|X64; X100+X57|X63+X58|X63+X61|X57; X100+X57|X61+X57|X63+X63|X58; X28+X41+X58|X49+X62|X50; X100+X57|X63+X58|X63+X61|X57; X33; X100+X57|X63+X58|X63+X61|X57; X31+X43|X102+X54|X73+X79; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X100+X57|X63+X58|X63+X61|X57; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X23+X59|X49+X62|X49; X48+X61|X68+X66|X68+X69|X61; X23+X59|X49+X62|X49; X41+X58|X62+X62|X58; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X100+X58|X63+X61|X63; X23+X59|X49+X62|X49; F4+F5+X49|X66+X61|X59; X23+X59|X49+X62|X49; X100+X58|X63+X61|X63; X23+X59|X49+X62|X49; F5+X49|X66+X49|X68+X68|X59; X100+X58|X63+X61|X63; F4+F5+X49|X66+X61|X59; X23+X59|X49+X62|X49; X100+X63|X58+X63|X61; X48+X61|X69+X68|X61+X68|X66; X51; X100+X57|X63+X58|X63+X61|X57; X100+X63|X58+X63|X61; X51+X63|X61+X64|X61; X48+X61|X69+X68|X61+X68|X66; X100+X57|X63+X58|X63+X61|X57; X51+X68|X63+X68|X64; X100+X63|X58+X63|X61; X100+X57|X63+X58|X63+X61|X57; F5+X49|X66+X49|X68+X68|X59; X100+X63|X58+X63|X61; F4; X100+X63|X58+X63|X61; X100+X57|X63+X58|X63+X61|X57; X23+X59|X49+X62|X49; F4+F5+X49|X66+X61|X59; X23+X59|X49+X62|X49; X100+X58|X63+X61|X63; X23+X59|X49+X62|X49; F4+F5+X49|X66+X61|X59; X23+X59|X49+X62|X49; X41+X58|X62+X62|X58; X48+X61|X68+X66|X68+X69|X61; X23+X59|X49+X62|X49; X51+X61|X63+X61|X64; X51+X68|X63+X68|X64; X100+X57|X63+X58|X63+X61|X57; X51+X68|X63+X68|X64; X51; X100+X57|X63+X58|X63+X61|X57; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X100+X57|X63+X58|X63+X61|X57; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X51+X68|X63+X68|X64; X48+X66|X56+X69|X56; X31+X43|X102+X54|X73+X79; F1+F3|X49; X23+X59|X49+X62|X49; X51+X63|X68+X64|X68; X23+X49|X59+X49|X62; X51+X68|X63+X68|X64; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X51+X63|X61+X64|X61; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; F4+F5+X49|X66+X61|X59; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; F4+F5+X59|X61+X66|X49; F4+F5+X49|X66+X61|X59; F4+F5+X59|X61+X66|X49; F4+F5+X49|X66+X61|X59; F5+X59|X68+X66|X49+X68|X49; F4+F5+X59|X61+X66|X49; F4+F5+X49|X66+X61|X59; X100+X58|X63+X61|X63; F4+F5+X49|X66+X61|X59; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; X38+X48+X68|X66+X68|X69; F4+F5+X59|X61+X66|X49; X38+X48+X68|X66+X68|X69; F4+F5+X59|X61+X66|X49; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; F4+F5+X59|X61+X66|X49; X38+X48+X68|X66+X68|X69; F4+F5+X59|X61+X66|X49; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; X38+X48+X68|X66+X68|X69; F4+F5+X59|X61+X66|X49; X38+X48+X68|X66+X68|X69; F4; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; X100+X63|X58+X63|X61; F4+F5+X49|X66+X61|X59; X38+X48+X68|X66+X68|X69; X100+X58|X63+X61|X63; F4+F5+X49|X66+X61|X59; F4+F5+X59|X61+X66|X49; X100+X58|X64+X61|X63+X63|X64; X38+X48+X68|X66+X68|X69; F4+F5+X49|X66+X61|X59; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X38+X48+X66|X68+X69|X68; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X51+X61|X63+X61|X64; X38+X48+X66|X68+X69|X68; X23+X59|X49+X62|X49; X38+X48+X66|X68+X69|X68; X23+X59|X49+X62|X49; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X51+X61|X63+X61|X64; X23+X59|X49+X62|X49; X38+X48+X66|X68+X69|X68; X23+X59|X49+X62|X49; X38+X48+X66|X68+X69|X68; X23+X49|X59+X49|X62; X28+X41+X58|X49+X62|X50; X23+X49|X59+X49|X62; X51+X68|X63+X68|X64; X38+X48+X66|X68+X69|X68; X23+X49|X59+X49|X62; X28+X41+X58|X49+X62|X50; X23+X49|X59+X49|X62; X51+X63|X68+X64|X68; X23+X49|X59+X49|X62; X23+X59|X49+X62|X49; X28+X41+X58|X49+X62|X50; F1+F3|X49; F4; X38+X48+X68|X66+X68|X69; F4; T2 | 6 |
| E2 | 8 | T2+X95|F3+X95|X36; X48+X66|X69+X69|X66; T2+X95|F3+X95|X36; X20+X21+X66|X67+X95|F2; F2|X36+X11+X36|F2; F1+X12|F3+X3+X95|X50; X17+X56|X32+X56|X54; F2|X46+X14+X45|X46 | 0 |
| T1 | 10 | X9; F4+X49|F2+X9+X96; X49|F2+X49|X58+X9+X96; F4+X49|X58+X58|X49; F4+F5+X49|X66+X61|X59; X96; X49|F2+X58|X49+X9+X96; X34+X55|X60+X60|X55; T2; X41+X58|X62+X62|X58 | 17 |
| X1 | 4 | F2|X36+X11+X36|F2; F1+X28; F2|X12+X12|F2+X3; T2+X95|F3+X95|X36 | 30 |
| X13 | 9 | F1+F3|X54+X31+X49|X43; X31+X43|X102+X54|X73+X79; F1+F4+X49|F3+X61|X49; F4; F1+F3|X49; F1+X31+X43|X49+X54|F3; X31+X43|X102+X54|X73+X79; X101+X44+X55|X103+X65|X72; X11+X19+X32|X37+X46|F2 | 0 |
| X18 | 4 | F4; X35+X52+X66|X74+X67|X56; X67|X64+X67|X74+X84; X20+X21+X66|X67+X95|F2 | 0 |
| X2 | 16 | F2|X46+X14+X45|X46; X25+X54|X46+X70|X45+X93; X17+X56|X32+X56|X54; X25+X54|X46+X70|X45+X93; X17+X56|X32+X56|X54; X25+X54|X46+X70|X45+X93; X17+X56|X32+X56|X54; X25+X45|X102+X46|X71+X86; X23; X34+X55|X60+X60|X55; X17+X32|X55+X43|X55; X17+X56|X32+X56|X54; X20+X21+X66|X67+X95|F2; X17+X32|X56+X54|X56; X102|X71+X71|X102+X86; F4+X49|X58+X58|X49 | 0 |
| X24 | 16 | F4; X84; X47+X64|X65+X64|X67; X84; X47+X64|X65+X64|X67; X47+X60|X65+X60|X67; X21; X21+X98; X34+X44+X47+X98; X47+X64|X65+X64|X67; X34+X44+X47+X98; X31+X43|X102+X54|X73+X79; X21; X47+X60|X65+X60|X67; X101+X103|X55+X44+X72|X65; F4 | 2 |
| X29 | 16 | X20+X21+X66|X67+X95|F2; X10; F5+X39+X52+X98; X35+X52+X56|X67+X74|X66; F4; X21+X98; F5+X39+X52+X98; X31+X43|X102+X54|X73+X79; X20+X21+X66|X67+X95|F2; X10; F5+X39+X52+X98; X35+X52+X66|X74+X67|X56; F4; X21+X98; F5+X39+X52+X98; X31+X43|X102+X54|X73+X79 | 0 |
| X4 | 11 | F4+X49|X58+X58|X49; T2+X14+X17+X82; T2+X15+X17+X22; X20+X21+X66|X67+X95|F2; X11+X19+X32|X37+X46|F2; T2+X15+X17+X22; X15+X30|X32+X30|X37; F2|X46+X14+X45|X46; X41+X58|X62+X62|X58; T2+X14+X17+X82; F1+X31+X43|X49+X54|F3 | 0 |
| X5 | 22 | F4+F5+X49|X66+X61|X59; F4; X41+X58|X62+X62|X58; X34+X55|X60+X60|X55; X21+X23+X9+X96; X96; T2; X34+X55|X60+X60|X55; X21+X23+X9+X96; X96; X21+X23+X9+X96; X20+X23+X9+X96; T2; X23; X21+X23+X9+X96; X20+X23+X9+X96; X23+X59|X49+X62|X49; X9; X23; X32|X103+X32|X76+X89; X20+X23+X9+X96; X9 | 0 |
| X6 | 7 | F1+F3|X54+X31+X49|X43; X25+X54|X46+X70|X45+X93; F2|X46+X14+X45|X46; F1+X31+X43|X49+X54|F3; X11+X19+X32|X37+X46|F2; X20+X21+X66|X67+X95|F2; X11+X19+X32|X37+X46|F2 | 7 |
| X78 | 4 | X20+X21+X66|X67+X95|F2; X15+X30|X32+X30|X37; X11; F2|X46+X14+X45|X46 | 1 |
| X8 | 8 | X20+X21+X66|X67+X95|F2; X20+X21+X59|X60+X95|F2; X96; F2|X46+X14+X45|X46; F1+F3|X12+X3+X50|X95; F2|X95+X20+X21+X67|X66; F2|X60+X10+X39+X95|X59; X10+X9 | 0 |
| X80 | 8 | X17+X32|X56+X54|X56; X31+X43|X102+X54|X73+X79; F4; X31+X43|X102+X54|X73+X79; X25+X54|X46+X70|X45+X93; X31+X43|X102+X54|X73+X79; X100+X57|X63+X58|X63+X61|X57; X31+X43|X102+X54|X73+X79 | 0 |
| X81 | 48 | X17+X32|X55+X43|X55; X101; X101+X103|X55+X44+X72|X65; X101+X44+X55|X103+X65|X72; X56|X103+X56|X76+X89; F1+F4+X49|F3+X61|X49; F4; X31+X43|X102+X54|X73+X79; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X25+X54|X46+X70|X45+X93; X31+X43|X102+X54|X73+X79; F4; F1+F4+X49|F3+X61|X49; X101+X103|X55+X44+X72|X65; X101; X17+X32|X55+X43|X55; X31+X43|X102+X54|X73+X79; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X55|X72+X55|X77+X91; X101+X44+X55|X103+X65|X72; X25+X54|X46+X70|X45+X93; X31+X43|X102+X54|X73+X79 | 0 |
| X97 | 12 | X20+X21+X66|X67+X95|F2; X14+X47+X65|X45+X67|F2; F2|X45+X21+X27+X60|X55; X34+X55|X60+X60|X55; F2|X45+X21+X27+X60|X55; X14+X47+X65|X45+X67|F2; X20+X21+X66|X67+X95|F2; X23; F2|X45+X21+X27+X60|X55; F2|X67+X14+X45|X65+X47; X53; F2|X46+X14+X45|X46 | 0 |

## Atlas database (atlas.json)

Closure: {'faces_with_two_region_sides': 150, 'faces': 150, 'faces_with_an_outline': 134, 'line_pieces_bounding_a_face': 57, 'line_pieces': 143, 'line_pieces_with_two_end_points': 143, 'closed_loop_line_pieces': 0, 'faces_with_outline_curves_on_no_atlas_line': 109, 'regions_with_faces': 74, 'regions': 74, 'points': 331, 'atlas_points_found': ['A1', 'C1', 'C4', 'C5', 'C6', 'X2', 'X4', 'X5', 'X8', 'X94·X11·T2', 'X97']}

| wall | piece types | face classes | faces on S^3 | unplaced fragments |
|---|---|---|---|---|
| F1 | 2 | 4 | 4800 | 0 |
| F4 | 2 | 4 | 4800 | 1 |
| F5 | 1 | 2 | 2400 | 0 |
| T2 | 4 | 8 | 9600 | 0 |
| X10 | 1 | 2 | 2400 | 0 |
| X100 | 1 | 2 | 2400 | 0 |
| X101 | 1 | 2 | 2400 | 0 |
| X11 | 2 | 4 | 4800 | 0 |
| X14 | 2 | 4 | 4800 | 0 |
| X15 | 1 | 2 | 2400 | 0 |
| X17 | 3 | 6 | 7200 | 0 |
| X19 | 1 | 2 | 2400 | 0 |
| X20 | 2 | 4 | 4800 | 0 |
| X21 | 2 | 4 | 4800 | 0 |
| X22 | 2 | 4 | 4800 | 0 |
| X23 | 2 | 4 | 4800 | 2 |
| X25 | 1 | 2 | 2400 | 0 |
| X26 | 2 | 4 | 4800 | 0 |
| X27 | 2 | 4 | 4800 | 0 |
| X28 | 1 | 2 | 2400 | 0 |
| X3 | 2 | 4 | 4800 | 0 |
| X31 | 1 | 2 | 2400 | 0 |
| X33 | 1 | 2 | 2400 | 0 |
| X34 | 1 | 2 | 2400 | 0 |
| X35 | 1 | 2 | 2400 | 0 |
| X38 | 1 | 1 | 1200 | 0 |
| X39 | 1 | 2 | 2400 | 0 |
| X40 | 1 | 2 | 2400 | 0 |
| X41 | 1 | 2 | 2400 | 0 |
| X42 | 1 | 2 | 2400 | 0 |
| X44 | 1 | 2 | 2400 | 0 |
| X47 | 1 | 2 | 2400 | 0 |
| X48 | 1 | 2 | 2400 | 0 |
| X51 | 1 | 2 | 2400 | 0 |
| X52 | 1 | 2 | 2400 | 0 |
| X53 | 1 | 2 | 2400 | 0 |
| X7 | 1 | 1 | 1200 | 0 |
| X79 | 1 | 2 | 2400 | 0 |
| X82 | 1 | 2 | 2400 | 0 |
| X83 | 1 | 2 | 2400 | 0 |
| X84 | 2 | 4 | 4800 | 0 |
| X85 | 1 | 2 | 2400 | 0 |
| X86 | 1 | 2 | 2400 | 0 |
| X87 | 1 | 2 | 2400 | 0 |
| X88 | 1 | 2 | 2400 | 0 |
| X89 | 1 | 2 | 2400 | 0 |
| X9 | 2 | 4 | 4800 | 1 |
| X90 | 1 | 2 | 2400 | 0 |
| X91 | 1 | 2 | 2400 | 0 |
| X92 | 1 | 2 | 2400 | 0 |
| X93 | 1 | 2 | 2400 | 0 |
| X94 | 1 | 2 | 2400 | 0 |
| X96 | 2 | 4 | 4800 | 0 |
| X98 | 1 | 2 | 2400 | 0 |
| X99 | 1 | 2 | 2400 | 0 |

Line pieces are the atlas's own lines, cut only at its exact meeting points and joined where a line goes on into the next copy; piece types are congruence classes under G and G.Q.

| line | piece types | classes | pieces on S^3 |
|---|---|---|---|
| B1 | 1 | 1 | 1200 |
| C2a | 1 | 1 | 1200 |
| C2b | 1 | 2 | 2400 |
| C3 | 1 | 2 | 2400 |
| D1 | 1 | 3 | 3600 |
| D2 | 2 | 5 | 6000 |
| D3 | 2 | 2 | 2400 |
| E1 | 4 | 15 | 18000 |
| E2 | 3 | 6 | 7200 |
| T1 | 3 | 9 | 10800 |
| X1 | 2 | 3 | 3600 |
| X13 | 3 | 8 | 9600 |
| X18 | 3 | 3 | 3600 |
| X2 | 3 | 6 | 7200 |
| X24 | 3 | 7 | 8400 |
| X29 | 3 | 8 | 9600 |
| X4 | 3 | 6 | 7200 |
| X5 | 5 | 15 | 18000 |
| X6 | 3 | 5 | 6000 |
| X78 | 2 | 3 | 3600 |
| X8 | 3 | 6 | 7200 |
| X80 | 3 | 7 | 8400 |
| X81 | 3 | 12 | 14400 |
| X97 | 3 | 8 | 9600 |

Curves where walls meet (traced, exact) that lie on no atlas line, by the walls meeting there:

| walls meeting | classifies as line | edges | length in the domain |
|---|---|---|---|
| X41, X96 | X97 | 18 | 3.228796 |
| X14 |  | 10 | 1.494474 |
| X47 | X18 | 7 | 1.466336 |
| X3 |  | 7 | 0.773879 |
| X22, X93 |  | 4 | 0.701956 |
| F1, X28 |  | 3 | 0.676964 |
| F4, X100 |  | 2 | 0.658065 |
| X101, X26, X79 |  | 3 | 0.594527 |
| X23, X39 | X24 | 2 | 0.540208 |
| X11 |  | 4 | 0.524082 |
| X41 |  | 1 | 0.495481 |
| X51, X84 |  | 2 | 0.481123 |
| X35, X52 | X18 | 2 | 0.462444 |
| T2, X11 | X97 | 6 | 0.410017 |
| X28 |  | 1 | 0.402551 |
| X14, X25 |  | 6 | 0.391983 |
| X84 | X18 | 2 | 0.290453 |
| X17, X19 |  | 3 | 0.252916 |
| X48 | X18 | 1 | 0.250966 |
| X11, X19 | X6 | 1 | 0.239502 |
| T2, X94 | X97 | 2 | 0.180073 |
| X19, X26 | X2 | 1 | 0.170301 |
| F5, X20 |  | 2 | 0.150771 |
| X21, X98 |  | 2 | 0.128422 |
| X10, X9 |  | 2 | 0.118859 |
| F1, T2 | X97 | 1 | 0.112748 |
| X17, X31 |  | 2 | 0.099102 |
| F1 |  | 1 | 0.052892 |
| X27, X44 |  | 2 | 0.045405 |
| X38 |  | 1 | 0.02779 |

## Lines and points

| line | shape | copies on S^3 | joined curves | stabilizer |
|---|---|---|---|---|
| B1 | Icosafold icosaswirlchoron | 240 | 12 | 100 |
| C2a | Subsymmetrical icosafold icosidodecaswirlchoron | 600 | 600 | 2 |
| C2b | Subsymmetrical icosafold icosidodecaswirlchoron | 1200 | 600 | 2 |
| C3 | Subsymmetrical pentagonal-gyroprismatic triacosihexecontachoron | 1200 | 1200 | 2 |
| D1 | Polychoron with 120+120+1200 cells (600+1200 4-valent edges) | 1200 | 1200 | 2 |
| D2 | Polychoron with 120+600+600+600+1200 cells (600 3-valent edges) | 1200 | 240 | 10 |
| D3 | Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges) | 1200 | 720 | 10 |
| E1 | Bi-hecatonicosadiminished truncated hexacosichoron | 1200 | 2 | 1200 |
| E2 | Swirlprismatodiminished truncated hecatonicosachoron | 1200 | 40 | 60 |
| T1 | Transitional polychoron with 120+120+600+600 cells | 1200 | 60 | 40 |
| X1 | Not in the wiki list | 1200 | 20 | 60 |
| X13 | Not in the wiki list | 1200 | 240 | 10 |
| X18 | Not in the wiki list | 600 | 120 | 10 |
| X2 | Piece of the truncated 120-cell | 1200 | 240 | 10 |
| X24 | Not in the wiki list | 1200 | 240 | 10 |
| X29 | Not in the wiki list | 1200 | 240 | 10 |
| X4 | Not in the wiki list | 1200 | 240 | 10 |
| X5 | Not in the wiki list | 1200 | 60 | 40 |
| X6 | Piece of the runcinated 120-cell | 1200 | 1200 | 2 |
| X78 | Not in the wiki list | 1200 | 20 | 60 |
| X8 | Piece of the cantellated 120-cell | 1200 | 1200 | 2 |
| X80 | Not in the wiki list | 1200 | 240 | 10 |
| X81 | Not in the wiki list | 1200 | 240 | 10 |
| X97 | Not in the wiki list | 1200 | 240 | 10 |

| point | copies on S^3 |
|---|---|
| A1 | 120 |
| C1 | 600 |
| C4 | 600 |
| C5 | 600 |
| C6 | 600 |
| D3 | 600 |
| X2 | 1200 |
| X29 | 1200 |
| X4 | 1200 |
| X5 | 1200 |
| X8 | 1200 |
| X81 | 1200 |
| X94·X11·T2 | 1200 |
| X97 | 1200 |
