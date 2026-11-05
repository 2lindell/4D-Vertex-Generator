# Kicontic and iocontic shapes, against kidecaic and iodecaic

Kicontic = [[3,4,3]]+ (the rotations, order 1152, with the 24-cell/dual swap); iocontic = [[3,4,3]+] (F4's rotations and F4's reflections times the swap, 1152; without the swap). Mapped as the decaic snubs: curved-wall map (snub_map.py, grid N = 32), walls reclassified on exact surface points (wall_shapes.py), then a dense survey of the two edge zones and the whole domain (edge_survey.py, 700 points each) for small regions. Every new region type's margin is well above the hull tolerance.

Matched by edge valences: the contic counts are exactly 9.6 = 1152/120 times the decaic ones. On the mirrors the orbits are the contic ones (contic_shapes.md). On the gippid-like line kicontic gives the snad analogue (576 vertices: 48 snub cubes, 192 octahedral cells, 576 tets); iocontic keeps the gippid analogue.

| | decaic regions | decaic walls | contic regions not found | contic walls not found | contic extra |
|---|---|---|---|---|---|
| Kicontic vs kidecaic | 14 | 15 | 0 | 7 | 0 |
| Iocontic vs iodecaic | 3 | 3 | 0 | 1 | 0 |

## Kicontic vs kidecaic

| decaic shape | kind | decaic cells | decaic edges | contic | contic cells | found by |
|---|---|---|---|---|---|---|
| mebosid / gabosid | region | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} | found | 5280 | map |
| new | region | 560 | {3: 240, 4: 360, 6: 120, 8: 120} | found | 5280 | map |
| new | region | 680 | {3: 360, 4: 360, 6: 60, 8: 60, 10: 120} | found | 6432 | edge survey |
| new | region | 680 | {3: 360, 4: 300, 6: 180, 10: 120} | found | 6432 | edge survey |
| new | region | 560 | {3: 120, 4: 600, 8: 60, 10: 60} | found | 5280 | map |
| apbosid | region | 420 | {3: 360, 4: 300, 8: 120} | found | 3936 | map |
| sabosid | region | 440 | {3: 120, 4: 420, 5: 120, 8: 60} | found | 4128 | map |
| mebosid / gabosid | region | 560 | {3: 360, 4: 180, 5: 120, 6: 60, 8: 60, 10: 60} | found | 5280 | map |
| new | region | 560 | {3: 240, 4: 420, 6: 60, 8: 60, 10: 60} | found | 5280 | map |
| new | region | 560 | {3: 120, 4: 540, 6: 60, 8: 120} | found | 5280 | map |
| new | region | 680 | {3: 360, 4: 300, 6: 180, 10: 120} | found | 6432 | edge survey |
| new | region | 680 | {3: 360, 4: 240, 5: 120, 6: 120, 10: 120} | found | 6432 | edge survey |
| new | region | 680 | {3: 360, 4: 300, 6: 120, 8: 120, 10: 60} | found | 6432 | edge survey |
| sigwebosid | wall | 290 | {3: 300, 4: 300, 7: 60} | found | 2688 | map |
| new | wall | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} | found | 3552 | map |
| new | wall | 440 | {3: 480, 4: 180, 6: 60, 8: 120} | **not found** |  |  |
| new | wall | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} | **not found** |  |  |
| new | wall | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} | **not found** |  |  |
| tetbosid | wall | 240 | {3: 300, 4: 120, 5: 60, 6: 60} | found | 2256 | map |
| petbosid | wall | 230 | {3: 420, 4: 60, 5: 60, 6: 60} | found | 2112 | map |
| gigwebosid | wall | 320 | {3: 240, 4: 360, 8: 60} | found | 2976 | map |
| new | wall | 380 | {3: 300, 4: 300, 7: 60, 8: 60} | found | 3552 | map |
| new | wall | 380 | {3: 300, 4: 300, 7: 60, 8: 60} | found | 3552 | map |
| new | wall | 380 | {3: 300, 4: 240, 5: 60, 6: 60, 8: 60} | found | 3552 | map |
| new | wall | 440 | {3: 480, 4: 180, 6: 60, 8: 120} | **not found** |  |  |
| new | wall | 440 | {3: 360, 4: 360, 8: 120} | **not found** |  |  |
| new | wall | 500 | {3: 360, 4: 240, 5: 60, 6: 60, 8: 60, 9: 60} | **not found** |  |  |
| new | wall | 500 | {3: 420, 4: 120, 5: 60, 6: 120, 8: 120} | **not found** |  |  |
| new | region | 680 | {3: 360, 4: 180, 5: 120, 6: 180, 8: 60, 10: 60} | found | 6432 | edge survey |

Contic shapes with no decaic counterpart: 0

## Iocontic vs iodecaic

| decaic shape | kind | decaic cells | decaic edges | contic | contic cells | found by |
|---|---|---|---|---|---|---|
| sosbid | region | 380 | {3: 120, 4: 300, 5: 240} | found | 3552 | map |
| new | region | 500 | {3: 240, 4: 120, 5: 240, 6: 180} | found | 4704 | edge survey |
| gosbid | region | 360 | {3: 360, 4: 60, 5: 240, 6: 60} | found | 3360 | map |
| new | wall | 260 | {3: 240, 4: 420} | **not found** |  |  |
| tetosbid | wall | 180 | {3: 240, 4: 240} | found | 1680 | map |
| petosbid | wall | 170 | {3: 360, 4: 180} | found | 1536 | map |

Contic shapes with no decaic counterpart: 0

