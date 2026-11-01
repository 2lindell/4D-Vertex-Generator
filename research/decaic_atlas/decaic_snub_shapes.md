# Kidecaic and iodecaic shapes (the snub subgroups of [[3,3,3]])

Domain: the decaic domain beta >= 0, b1 >= b4 (seeds differing by a decaic element give congruent polytopes). On the mirrors the orbits are the decaic ones, so every decaic shape on a mirror (bideca, apid, spid, deca, respid, redeca, bited, biped, bimted, mabred, sabred, gabred, tispid, tabipard, tadeca, tabcated, sabipard, mabipard, gabipard, sobcated, mobcated, gobcated) is also kidecaic and iodecaic. On the gippid line b1 = b4, b2 = b3: kidecaic gives snad (60 vertices: 10 snits, 20 traps, 60 tets), iodecaic keeps gippid.

Inside, the walls are curved (quadrics and quartics as well as planes), so the map is a grid (N = 40, 4389 points shifted off the lattice) classified exactly, its changes bisected and put on the exact surface (snub_map.py). Every region type's margin (the least distance of a vertex from the 3-flat of a cell it is not on) is far above the hull tolerance (margin.py). Matched to the list by edge valences (edges.py).

## Kidecaic

| list name | kind | vertices | cells | edges by valence | grid points | surface | example seed |
|---|---|---|---|---|---|---|---|
| mebosid / gabosid (560, nearer gabred) | region | 120 | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} | 456 |  | [0.0501, 0.0253, 0.8993, 0.0254] |
| **not on the list** | region | 120 | 560 | {3: 240, 4: 360, 6: 120, 8: 120} | 6 |  | [0.0501, 0.0752, 0.8493, 0.0254] |
| **not on the list** | region | 120 | 680 | {3: 360, 4: 360, 6: 60, 8: 60, 10: 120} | 14 |  | [0.0501, 0.1501, 0.7744, 0.0254] |
| **not on the list** | region | 120 | 680 | {3: 360, 4: 300, 6: 180, 10: 120} | 4 |  | [0.0501, 0.3249, 0.5996, 0.0254] |
| **not on the list** | region | 120 | 560 | {3: 120, 4: 600, 8: 60, 10: 60} | 45 |  | [0.0501, 0.3499, 0.5747, 0.0254] |
| apbosid | region | 120 | 420 | {3: 360, 4: 300, 8: 120} | 936 |  | [0.0501, 0.4498, 0.4748, 0.0254] |
| sabosid | region | 120 | 440 | {3: 120, 4: 420, 5: 120, 8: 60} | 2109 |  | [0.0501, 0.4747, 0.4498, 0.0254] |
| mebosid / gabosid (560, along the gippid/snad line) | region | 120 | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} | 711 |  | [0.075, 0.4498, 0.4498, 0.0254] |
| **not on the list** | region | 120 | 560 | {3: 240, 4: 420, 6: 60, 8: 60, 10: 60} | 72 |  | [0.4246, 0.1002, 0.1252, 0.35] |
| **not on the list** | region | 120 | 560 | {3: 120, 4: 540, 6: 60, 8: 120} | 19 |  | [0.5745, 0.0253, 0.0253, 0.375] |
| **not on the list** | region | 120 | 680 | {3: 360, 4: 300, 6: 180, 10: 120} | 7 |  | [0.5745, 0.0253, 0.0503, 0.35] |
| **not on the list** | region | 120 | 680 | {3: 360, 4: 240, 5: 120, 6: 120, 10: 120} | 9 |  | [0.6244, 0.0253, 0.0503, 0.3001] |
| **not on the list** | region | 120 | 680 | {3: 360, 4: 300, 6: 120, 8: 120, 10: 60} | 1 |  | [0.6244, 0.0253, 0.1002, 0.2501] |
| **not on the list** | wall | 120 | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} |  | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.0501, 0.067, 0.8575, 0.0254] |
| **not on the list** | wall | 120 | 440 | {3: 480, 4: 180, 6: 60, 8: 120} |  |  | [0.0622, 0.1251, 0.7873, 0.0254] |
| **not on the list** | wall | 120 | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} |  | b1*b2 - b3*b4 | [0.0668, 0.25, 0.6578, 0.0254] |
| **not on the list** | wall | 120 | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} |  | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.0648, 0.3101, 0.5996, 0.0254] |
| **not on the list** | wall | 120 | 680 | {3: 360, 4: 180, 5: 120, 6: 180, 8: 60, 10: 60} |  |  | [0.0501, 0.3455, 0.579, 0.0254] |
| tetbosid | wall | 120 | 240 | {3: 300, 4: 120, 5: 60, 6: 60} |  | 3*b1 + b2 - b3 - 3*b4 | [0.0503, 0.4248, 0.4995, 0.0254] |
| **not on the list** | wall | 120 | 380 | {3: 300, 4: 300, 7: 60, 8: 60} |  | b1*b2 - b3*b4 | [0.1252, 0.3499, 0.4246, 0.1003] |
| petbosid | wall | 120 | 230 | {3: 420, 4: 60, 5: 60, 6: 60} |  | b1 + 2*b2 - 2*b3 - b4 | [0.0585, 0.4498, 0.4663, 0.0254] |
| **not on the list** | wall | 120 | 280 | {3: 420, 4: 120, 6: 60, 7: 60} |  |  | [0.2498, 0.2938, 0.381, 0.0753] |
| **not on the list** | wall | 120 | 320 | {3: 240, 4: 300, 6: 120} |  |  | [0.3124, 0.25, 0.3749, 0.0627] |
| **not on the list** | wall | 120 | 380 | {3: 300, 4: 300, 7: 60, 8: 60} |  | b1*b2 - b3*b4 | [0.4492, 0.1005, 0.2001, 0.2501] |
| gigwebosid | wall | 120 | 320 | {3: 240, 4: 360, 8: 60} |  | b2 - b3 | [0.1749, 0.3499, 0.3499, 0.1253] |
| **not on the list** | wall | 120 | 320 | {3: 240, 4: 300, 6: 120} |  |  | [0.2998, 0.0879, 0.4248, 0.1875] |
| **not on the list** | wall | 120 | 300 | {3: 360, 4: 180, 6: 60, 8: 60} |  |  | [0.3247, 0.038, 0.4498, 0.1875] |
| **not on the list** | wall | 120 | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} |  | 3*b1*b2 - b1*b4 + b2**2 - b2*b3 + 2*b3*b4 + b4**2 | [0.5529, 0.0253, 0.0253, 0.3966] |
| **not on the list** | wall | 120 | 440 | {3: 480, 4: 180, 6: 60, 8: 120} |  | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 - 3*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | [0.569, 0.0253, 0.0557, 0.35] |
| **not on the list** | wall | 120 | 440 | {3: 360, 4: 360, 8: 120} |  | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | [0.5745, 0.0253, 0.0381, 0.3622] |
| **not on the list** | wall | 120 | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} |  | b1*b2 - b3*b4 | [0.6152, 0.0253, 0.0503, 0.3092] |
| **not on the list** | wall | 120 | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} |  |  | [0.6244, 0.0253, 0.0954, 0.255] |
| sigwebosid | wall (fold b1 = b4) | 120 | 290 | {3: 300, 4: 300, 7: 60} | 361 |  | [0.0251, 0.0253, 0.9245, 0.0251] |

## Iodecaic

| list name | kind | vertices | cells | edges by valence | grid points | surface | example seed |
|---|---|---|---|---|---|---|---|
| sosbid | region | 120 | 380 | {3: 120, 4: 300, 5: 240} | 3445 |  | [0.0501, 0.0253, 0.8993, 0.0254] |
| **not on the list** | region | 120 | 500 | {3: 240, 4: 120, 5: 240, 6: 180} | 8 |  | [0.0501, 0.2001, 0.7245, 0.0254] |
| gosbid | region | 120 | 360 | {3: 360, 4: 60, 5: 240, 6: 60} | 936 |  | [0.0501, 0.4498, 0.4748, 0.0254] |
| **not on the list** | wall | 120 | 260 | {3: 240, 4: 420} |  | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | [0.0517, 0.1751, 0.7478, 0.0254] |
| tetosbid | wall | 120 | 180 | {3: 240, 4: 240} |  | 3*b1 + b2 - b3 - 3*b4 | [0.1085, 0.3499, 0.4748, 0.0669] |
| **not on the list** | wall | 120 | 260 | {3: 120, 4: 480} |  |  | [0.1999, 0.3437, 0.431, 0.0254] |
| **not on the list** | wall | 120 | 260 | {3: 120, 4: 480} |  |  | [0.2666, 0.0502, 0.5996, 0.0835] |
| **not on the list** | wall | 120 | 240 | {3: 240, 4: 300, 6: 60} |  |  | [0.2748, 0.038, 0.5996, 0.0876] |
| petosbid | wall | 120 | 170 | {3: 360, 4: 180} |  | b1 + 2*b2 - 2*b3 - b4 | [0.4246, 0.1086, 0.1751, 0.2916] |

