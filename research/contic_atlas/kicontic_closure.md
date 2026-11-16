# Kicontic closure check (2026-11-16)

`CONTIC_SUB=ki python snub_closure.py` (960 walks from 14 region types; results in snub_closure_ki.json, log in
snub_closure_ki.log). Ends: 387 at recorded walls, 231 at the domain's edge, 342 at walls whose shape was not yet
recorded. **No region is missing**: every walk ended in a known region. The unrecorded walls are 6 new wall shapes,
each matched to a kidecaic wall by the per-vertex edge-valence profile and vertex figure, then by the nearest decaic
seed on the map (the other candidate with the same profile is 0.87 or more away).

| new kicontic wall (key) | seed (beta) | edges at a vertex | vertex figure | cells | kidecaic match (seed distance) | name by the d→c rule |
|---|---|---|---|---|---|---|
| f7c174f6a037 | 0.0785, 0.1742, 0.7201, 0.0271 | 688,4^3,3^8 | 5,4^5,3^11 | 4128 | Abybosid, wall 11 (0.085) | Abybosic |
| a97fdecc2dff | 0.5436, 0.0074, 0.0631, 0.3858 | 688,4^3,3^8 | 5,4^5,3^11 | 4128 | Bebybosid, wall 17 (0.041) | Bebybosic |
| cf1591d75b16 | 0.0541, 0.0940, 0.8459, 0.0060 | 5689,4^4,3^6 | 5,4^3,3^15 | 4704 | Gapybosid, wall 12 (0.246) | Gapybosic |
| 088a005dd446 | 0.0411, 0.4029, 0.5529, 0.0031 | 56688,44,3^7 | 5,4^3,3^15 | 4704 | Depybosid, wall 13 (0.109) | Depybosic |
| c9267f353a08 | 0.5696, 0.0216, 0.0394, 0.3694 | 56688,44,3^7 | 5,4^3,3^15 | 4704 | Ipybosid, wall 20 (0.139) | Ipybosic |
| 0e8641589e0e | 0.0541, 0.3237, 0.5982, 0.0240 | 88,4^6,3^6 | 5,4^5,3^11 | 4128 | Debybosid, wall 23 (0.035; its repeat, wall 24, 0.058) | Debybosic |

## Decaic walls 23 and 24 are one shape

Kidecaic wall 23 (Debybosid, found by the decaic closure's second run) and wall 24 (Ebybosid, found by the crossing
search) have the same arrangement key, 6127f3ea279d, lie on the same surface, between the same two regions (Ebosid,
region 04, and Zedwebosid, region 21), and their seeds are 0.03 apart. They are the same polychoron found twice. Wall
18 (Gabybosid), with the same profile, is a different shape (key b39c8835296c). The new kicontic wall 0e8641589e0e
lies between the kicontic counterparts of Ebosid and Zedwebosid, so it is the counterpart of this one shape. Both list
names are kept.

## What kicontic still lacks

Of the 8 kidecaic shapes kicontic seemed to lack, 6 are walls found here, and Debybosid / Ebybosid count as one.
Only **Apybosid** (kidecaic region 22, edges 78,4^5,3^5) has no kicontic counterpart, and the closure check found no
missing kicontic region, so there may be none.

## Added to the atlas

The six walls were added as kicontic walls 27-32 by closure_add.py (closure_walls_ki.json; seed, optimized and
central .off files in new_shapes/off/kicontic, optimized_off/kicontic and central_off/kicontic), named as above:
27 Abybosic, 28 Gapybosic, 29 Depybosic, 30 Bebybosic, 31 Ipybosic, 32 Debybosic. The kidecaic repeat (wall 24,
Ebybosid) has been removed.
