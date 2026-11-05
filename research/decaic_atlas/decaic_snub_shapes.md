# Kidecaic and iodecaic shapes (the snub subgroups of [[3,3,3]])

Results: the curved-wall map (snub_map.py), with every wall shape reclassified on points exactly on its surface (wall_shapes.py), with regions added by the dense edge survey (edge_survey.py) and the region closure check (snub_closure.py: every region walked out to its walls). This replaces the earlier plane-only kidecaic table (decaic_shapes_ki.md, superseded) and the first version of this file, which counted shapes registered at points ~1e-11 off a wall (not sound to classify).

| | kidecaic | iodecaic |
|---|---|---|
| shared with decaic (on the mirrors: same orbits) | 22 | 22 |
| on the gippid line | snad | gippid |
| on the list, own regions | 4 (sabosid, apbosid, mebosid, gabosid) | 2 (sosbid, gosbid) |
| on the list, own walls | 4 (sigwebosid, gigwebosid, tetbosid, petbosid) | 2 (tetosbid, petosbid) |
| **total on the list** | **31** (all found) | **27** (all found) |
| new regions | 11 (9 on the map; 1 from the dense edge survey; 1 from the closure check) | 1 |
| new walls | 12 (11 on the map; 1 from the closure check, a quartic between region 10 and a 560-cell region) | 1 |
| **total found** | **54** | **29** |

Shared with decaic: bideca, apid, spid, deca, respid, redeca (points); bited, biped, bimted, mabred, sabred, gabred, tispid, tabipard, tadeca, tabcated (lines); sabipard, mabipard, gabipard, sobcated, mobcated, gobcated (walls).

Seeds and .off files for the new ones: ../new_shapes/new_shapes.md; optimized seeds for all: ../optimized_off/README.md.

## Kidecaic: own shapes

| name | kind | vertices | cells | edges by valence | surface | seed beta |
|---|---|---|---|---|---|---|
| mebosid | region | 120 | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} |  | [0.050072, 0.025267, 0.899268, 0.025393] |
| **new region 1** | region | 120 | 560 | {3: 240, 4: 360, 6: 120, 8: 120} |  | [0.050072, 0.075207, 0.849327, 0.025393] |
| **new region 2** | region | 120 | 680 | {3: 360, 4: 360, 6: 60, 8: 60, 10: 120} |  | [0.050072, 0.150119, 0.774416, 0.025393] |
| **new region 3** | region | 120 | 680 | {3: 360, 4: 300, 6: 180, 10: 120} |  | [0.050072, 0.324912, 0.599623, 0.025393] |
| **new region 4** | region | 120 | 560 | {3: 120, 4: 600, 8: 60, 10: 60} |  | [0.050072, 0.349882, 0.574653, 0.025393] |
| apbosid | region | 120 | 420 | {3: 360, 4: 300, 8: 120} |  | [0.050072, 0.449764, 0.474771, 0.025393] |
| sabosid | region | 120 | 440 | {3: 120, 4: 420, 5: 120, 8: 60} |  | [0.050072, 0.474734, 0.449801, 0.025393] |
| gabosid | region | 120 | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} |  | [0.075043, 0.449764, 0.449801, 0.025393] |
| **new region 5** | region | 120 | 560 | {3: 240, 4: 420, 6: 60, 8: 60, 10: 60} |  | [0.424628, 0.100178, 0.125185, 0.350009] |
| **new region 6** | region | 120 | 560 | {3: 120, 4: 540, 6: 60, 8: 120} |  | [0.574451, 0.025267, 0.025304, 0.374979] |
| **new region 7** | region | 120 | 680 | {3: 360, 4: 300, 6: 180, 10: 120} |  | [0.574451, 0.025267, 0.050274, 0.350009] |
| **new region 8** | region | 120 | 680 | {3: 360, 4: 240, 5: 120, 6: 120, 10: 120} |  | [0.624392, 0.025267, 0.050274, 0.300068] |
| **new region 9** | region | 120 | 680 | {3: 360, 4: 300, 6: 120, 8: 120, 10: 60} |  | [0.624392, 0.025267, 0.100215, 0.250127] |
| sigwebosid | wall | 120 | 290 | {3: 300, 4: 300, 7: 60} | b1 = b4 (fold) | [0.025109, 0.025274, 0.924508, 0.025109] |
| **new wall 1** | wall | 120 | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.050072, 0.067041, 0.857494, 0.025393] |
| **new wall 2** | wall | 120 | 440 | {3: 480, 4: 180, 6: 60, 8: 120} | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 - 3*b1*b2**2*b4 - b1*b2*b3**2 - 5* | [0.062209, 0.125148, 0.78725, 0.025393] |
| **new wall 3** | wall | 120 | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} | b1*b2 - b3*b4 | [0.066814, 0.25, 0.657793, 0.025393] |
| **new wall 4** | wall | 120 | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.06484, 0.310143, 0.599623, 0.025393] |
| tetbosid | wall | 120 | 240 | {3: 300, 4: 120, 5: 60, 6: 60} | 3*b1 + b2 - b3 - 3*b4 | [0.062785, 0.399823, 0.511999, 0.025393] |
| petbosid | wall | 120 | 230 | {3: 420, 4: 60, 5: 60, 6: 60} | b1 + 2*b2 - 2*b3 - b4 | [0.058517, 0.449764, 0.466326, 0.025393] |
| gigwebosid | wall | 120 | 320 | {3: 240, 4: 360, 8: 60} | b2 - b3 | [0.075005, 0.449801, 0.449801, 0.025393] |
| **new wall 5** | wall | 120 | 380 | {3: 300, 4: 300, 7: 60, 8: 60} | b1*b2 - b3*b4 | [0.075043, 0.268147, 0.624594, 0.032217] |
| **new wall 6** | wall | 120 | 380 | {3: 300, 4: 300, 7: 60, 8: 60} | b1*b2 - b3*b4 | [0.420642, 0.104164, 0.125185, 0.350009] |
| **new wall 7** | wall | 120 | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.722849, 0.026691, 0.100215, 0.150245] |
| **new wall 8** | wall | 120 | 440 | {3: 480, 4: 180, 6: 60, 8: 120} | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 - 3*b1*b2**2*b4 - b1*b2*b3**2 - 5* | [0.565896, 0.025267, 0.050274, 0.358563] |
| **new wall 9** | wall | 120 | 440 | {3: 360, 4: 360, 8: 120} | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5* | [0.574451, 0.025267, 0.038065, 0.362217] |
| **new wall 10** | wall | 120 | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} | b1*b2 - b3*b4 | [0.61525, 0.025267, 0.050274, 0.30921] |
| **new wall 11** | wall | 120 | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.624392, 0.025267, 0.095372, 0.254969] |

## Iodecaic: own shapes

| name | kind | vertices | cells | edges by valence | surface | seed beta |
|---|---|---|---|---|---|---|
| sosbid | region | 120 | 380 | {3: 120, 4: 300, 5: 240} |  | [0.050072, 0.025267, 0.899268, 0.025393] |
| **new region 1** | region | 120 | 500 | {3: 240, 4: 120, 5: 240, 6: 180} |  | [0.050072, 0.200059, 0.724475, 0.025393] |
| gosbid | region | 120 | 360 | {3: 360, 4: 60, 5: 240, 6: 60} |  | [0.050072, 0.449764, 0.474771, 0.025393] |
| **new wall 1** | wall | 120 | 260 | {3: 240, 4: 420} | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5* | [0.057401, 0.150119, 0.767087, 0.025393] |
| tetosbid | wall | 120 | 180 | {3: 240, 4: 240} | 3*b1 + b2 - b3 - 3*b4 | [0.062785, 0.399823, 0.511999, 0.025393] |
| petosbid | wall | 120 | 170 | {3: 360, 4: 180} | b1 + 2*b2 - 2*b3 - b4 | [0.199895, 0.343731, 0.430982, 0.025393] |


## Region closure

snub_closure.py walks out of every region (up to 4 spread starting points, 24 directions) to where the polytope changes, and checks the change: the domain's edge (a mirror or the fold), or a known region beyond across a known wall shape. Ends within 1e-6 of a mirror or the fold are the domain's edge.

* **Kidecaic** (final run: 15 regions, 16 wall shapes, 1320 walks): every region closed -- 1011 ends on recorded walls, 309 on the domain's edge, none open. Earlier runs found the two shapes that were missing: the quartic wall between region 10 and a 560-cell region, and a tiny region (380 cells, with 120 square pyramids) in the corner where the gabosid-like region, a 560-cell region and apbosid meet; with them recorded, nothing is open. The tiny region's walks (24, from one start) all end on the recorded quadric wall b1 b2 = b3 b4.
* **Iodecaic** (3 regions, 288 walks): every region closed by recorded walls and the domain's edge.

Region counts are what dense sampling and these walks find, not an exact enumeration: a region thinner than the sampling could still be missing.
