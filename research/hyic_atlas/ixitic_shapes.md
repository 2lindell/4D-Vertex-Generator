# Ixitic shapes (merged over the ten sheets)

Per sheet (group.py: sheet i is the conjugate g_i^-1 E g_i on the H4 chamber): the map's region types and the dense
survey's new regions, the wall shapes (classified on their surfaces), and the strata (the chamber's faces, edges and
corners). Every polytope is keyed by its arrangement key (a polytope and its mirror image or its image under ixoic get
the same key), so the ten sheets merge by key. Kinds: region (3 degrees of freedom), wall (2), line (1), point (0) --
the lowest stratum it lives on; a strata type seen in no sheet's interior is a wall (the mirrors are seams between
sheets). Each gets its full symmetry (every isometry keeping its vertex set: fullsym.py -- not only those in H4, since
a shape's symmetry group can be a rotated copy of a larger group), and how many elements of the sheet's ixoic (or
ixitic) keep it.

| name | kind | vertices | cells | edges by valence | symmetry | ixoic elements kept | own symmetry only | sheets | seed (beta, sheet) |
|---|---|---|---|---|---|---|---|---|---|
| stotathi | region | 1440 | 1440 | {3: 4320, 4: 1440} | 1440 | 1440 |  | 10 | [0.03347, 0.03369, 0.93284, 0.0] (ix0) |
| sestuthi | region | 1440 | 1920 | {3: 5760, 4: 720, 6: 720} | 1440 | 1440 |  | 5 | [0.0, 0.13469, 0.08324, 0.78207] (ix7) |
| misrethi | region | 1440 | 2160 | {3: 4320, 4: 2880} | 1440 | 1440 |  | 9 | [0.03348, 0.0337, 0.0, 0.93281] (ix2) |
| gypsobtady | region | 1440 | 3360 | {3: 5760, 4: 1440, 6: 1440} | 1440 | 1440 |  | 5 | [0.04182, 0.04209, 0.70785, 0.20824] (ix2) |
| sagypsobtady | region | 1440 | 3600 | {3: 1440, 4: 6480} | 1440 | 1440 |  | 6 | [0.04182, 0.04209, 0.0423, 0.87379] (ix0) |
| mestuthi | region | 1440 | 3600 | {3: 2880, 4: 4320, 8: 720} | 1440 | 1440 |  | 4 | [0.03348, 0.0, 0.03387, 0.93266] (ix0) |
| giserthi | region | 1440 | 4320 | {3: 2880, 4: 4320, 5: 1440, 6: 720} | 1440 | 1440 |  | 2 | [0.53301, 0.0, 0.03387, 0.43313] (ix5) |
|  | region | 1440 | 4800 | {3: 5760, 4: 1440, 5: 1440, 6: 720, 8: 720} | 1440 | 1440 |  | 4 | [0.04182, 0.41646, 0.25029, 0.29143] (ix2) |
| sisbethi_or_unnamed_1 | region | 1440 | 5040 | {3: 2880, 4: 5040, 6: 720, 8: 720} | 1440 | 1440 |  | 8 | [0.04182, 0.04209, 0.0839, 0.8322] (ix0) |
| sisbethi_or_unnamed_2 | region | 1440 | 5040 | {3: 4320, 4: 2160, 5: 1440, 6: 720, 8: 720} | 1440 | 1440 |  | 9 | [0.04182, 0.04209, 0.66626, 0.24984] (ix0) |
|  | region | 1440 | 5520 | {3: 5760, 4: 2880, 6: 1440, 10: 720} | 1440 | 1440 |  | 4 | [0.04182, 0.45806, 0.20869, 0.29143] (ix2) |
|  | region | 1440 | 5760 | {3: 2880, 4: 5760, 8: 1440} | 1440 | 1440 |  | 3 | [0.04182, 0.04209, 0.25029, 0.66581] (ix2) |
| magypsobtady_1 | region | 1440 | 5760 | {3: 2880, 4: 2880, 5: 2880, 6: 1440} | 1440 | 1440 |  | 1 | [0.08341, 0.04209, 0.0839, 0.7906] (ix2) |
| magypsobtady_2 | region | 1440 | 5760 | {3: 2880, 4: 2880, 5: 2880, 6: 1440} | 1440 | 1440 |  | 1 | [0.45779, 0.12528, 0.20869, 0.20824] (ix5) |
| gastuthi | region | 1440 | 5760 | {3: 2880, 4: 3600, 5: 1440, 6: 1440, 8: 720} | 1440 | 1440 |  | 2 | [0.06679, 0.8997, 0.0, 0.03351] (ix5) |
|  | region | 1440 | 6480 | {3: 4320, 4: 720, 5: 4320, 6: 720, 8: 720} | 1440 | 1440 |  | 1 | [0.04182, 0.04209, 0.0423, 0.87379] (ix2) |
|  | region | 1440 | 6480 | {3: 2880, 4: 4320, 5: 2880, 10: 720} | 1440 | 1440 |  | 3 | [0.04182, 0.04209, 0.1255, 0.7906] (ix2) |
|  | region | 1440 | 7200 | {3: 2880, 4: 4320, 5: 1440, 6: 2160, 8: 720} | 1440 | 1440 |  | 2 | [0.12501, 0.79084, 0.0423, 0.04185] (ix5) |
|  | region | 1440 | 7200 | {3: 4320, 4: 2160, 5: 1440, 6: 2880, 8: 720} | 1440 | 1440 |  | 2 | [0.55109, 0.25476, 0.18607, 0.00807] (ix5) |
|  | region | 1440 | 8640 | {3: 4320, 4: 3600, 5: 2880, 7: 1440, 12: 720} | 1440 | 1440 |  | 1 | [0.08341, 0.54125, 0.33348, 0.04185] (ix9) |
|  | region | 1440 | 8640 | {3: 4320, 4: 2880, 5: 2880, 6: 2160, 12: 720} | 1440 | 1440 |  | 1 | [0.44319, 0.08015, 0.03401, 0.44266] (ix9) |
|  | region | 1440 | 10080 | {3: 4320, 4: 7200, 6: 720, 9: 1440, 14: 720} | 1440 | 1440 |  | 1 | [0.08341, 0.62445, 0.25029, 0.04185] (ix9) |
|  | wall | 1440 | 960 | {3: 2880, 4: 1440} | 1440 | 1440 |  | 4 | [0.34997, 0.30005, 0.34997, 0.0] (ix5) |
| petsobtady | wall | 1440 | 2160 | {3: 5760, 5: 1440} | 1440 | 1440 |  | 3 | [0.04182, 0.07945, 0.58306, 0.29567] (ix2) |
|  | wall | 1440 | 2880 | {3: 5040, 4: 1440, 5: 720, 6: 720} | 1440 | 1440 |  | 7 | [0.04182, 0.07774, 0.49987, 0.38058] (ix0) |
|  | wall | 1440 | 2880 | {3: 5760, 4: 2160, 8: 720} | 1440 | 1440 |  | 3 | [0.04182, 0.04781, 0.16709, 0.74328] (ix2) |
|  | wall | 1440 | 2880 | {3: 5040, 4: 2160, 7: 720} | 1440 | 1440 |  | 4 | [0.05858, 0.41646, 0.20869, 0.31627] (ix2) |
| mastethi | wall | 1440 | 3120 | {3: 2880, 4: 5040} | 1440 | 1440 |  | 2 | [0.10458, 0.79084, 0.0423, 0.06228] (ix5) |
| gastethi | wall | 1440 | 3120 | {3: 4320, 4: 2880, 6: 720} | 1440 | 1440 |  | 2 | [0.43272, 0.26706, 0.30022, 0.0] (ix8) |
|  | wall | 1440 | 3360 | {3: 5760, 4: 720, 5: 1440, 8: 720} | 1440 | 1440 |  | 4 | [0.08239, 0.41646, 0.20972, 0.29143] (ix2) |
|  | wall | 1440 | 3600 | {3: 4320, 4: 4320, 6: 720} | 1440 | 1440 |  | 1 | [0.07205, 0.04209, 0.0839, 0.80196] (ix2) |
|  | wall | 1440 | 3600 | {3: 3600, 4: 2880, 5: 1440, 7: 720} | 1440 | 1440 |  | 2 | [0.2498, 0.56662, 0.0, 0.18358] (ix5) |
|  | wall | 1440 | 5040 | {3: 2880, 4: 2880, 5: 2880, 8: 720} | 1440 | 1440 |  | 1 | [0.04182, 0.04209, 0.1146, 0.80149] (ix2) |
|  | wall | 1440 | 5040 | {3: 3600, 4: 2880, 5: 2880, 7: 720} | 1440 | 1440 |  | 1 | [0.49938, 0.15629, 0.20869, 0.13563] (ix5) |
|  | wall | 1440 | 5760 | {3: 2880, 4: 2880, 5: 2880, 6: 720, 8: 720} | 1440 | 1440 |  | 2 | [0.1001, 0.83308, 0.0, 0.06682] (ix5) |
|  | wall | 1440 | 7200 | {3: 5760, 4: 5040, 7: 1440, 12: 720} | 1440 | 1440 |  | 1 | [0.08341, 0.6237, 0.25103, 0.04185] (ix9) |
|  | wall | 1440 | 6480 | {3: 3600, 4: 5760, 6: 1440, 11: 720} | 1440 | 1440 |  | 1 | [0.09916, 0.53205, 0.27486, 0.09392] (ix9) |
| awosbitdy | wall | 1440 | 6480 | {3: 7200, 4: 2880, 8: 1440, 12: 720} | 1440 | 1440 |  | 1 | [0.21558, 0.4303, 0.19558, 0.15854] (ix9) |
| hidhi | line | 480 | 240 | {3: 1440} | 1440 | 1440 |  | 8 | [0.5, 0.0, 0.0, 0.5] (ix5) |
| tisthi | line | 480 | 1440 | {3: 1440, 4: 720, 6: 720} | 1440 | 1440 |  | 6 | [0.03349, 0.96651, 0.0, 0.0] (ix0) |
| sagyptady | line | 480 | 2160 | {3: 1440, 4: 720, 6: 1440} | 1440 | 1440 |  | 2 | [0.0335, 0.0, 0.0, 0.9665] (ix0) |
| gagyptady | line | 480 | 2160 | {3: 1440, 4: 1440, 8: 720} | 1440 | 1440 |  | 2 | [0.53331, 0.0, 0.0, 0.46669] (ix5) |
| matethi | line | 720 | 240 | {3: 2160} | 1440 | 1440 |  | 6 | [0.0, 0.61803, 0.38197, 0.0] (ix7) |
| ratehi | line | 720 | 720 | {3: 3600} | 1440 | 1440 |  | 10 | [0.03348, 0.0, 0.96652, 0.0] (ix0) |
| sabtedy | line | 720 | 1680 | {3: 2880, 5: 1440} | 1440 | 1440 |  | 2 | [0.0, 0.63312, 0.36688, 0.0] (ix7) |
| tasthi | line | 1440 | 240 | {3: 2880} | 1440 | 1440 |  | 5 | [0.03349, 0.96651, 0.0, 0.0] (ix4) |
| tatehi | line | 1440 | 720 | {3: 3600, 5: 720} | 14400 | 1440 |  | 10 | [0.0, 0.0, 0.03387, 0.96613] (ix0) |
| sartethi | line | 1440 | 1680 | {3: 2880, 4: 2880} | 1440 | 1440 |  | 4 | [0.53311, 0.0, 0.46689, 0.0] (ix5) |
| tehi | point | 120 | 600 | {5: 720} | 14400 | 1440 |  | 10 | [1.0, 0.0, 0.0, 0.0] (ix0) |

Totals: {'region': 22, 'wall': 16, 'line': 10, 'point': 1}.

Names from the accepted ixitic list, matched by edge valences at a vertex and by cells. Notes:

- tehi: the 600-cell (ex) seen under ixitic: its vertex set has ex's full symmetry (14400)
- tatehi: tex seen under ixitic (symmetry 14400)
- sagyptady: the only other 480-vertex shape with 2160 cells (gagyptady matches exactly); the list's edges 6^6,4^9, the atlas's 6^6,4^3,3^6
- sartethi: cells exactly the list's (120 ratets, 120 dazsnits, 720 phydows, 720 wedges); edges 4^4,3^4 against the list's 4,444,333: one count apart
- mestuthi: cells exactly the list's (120 snits, 120 tuts, 480 gytricus, 720x2 disphs, 1440 tets); edges 8,4^6,3^4 against the list's 8,4^6,333: one count apart
- magypsobtady: two atlas regions have its edges and cells exactly; which is which is not settled
- sisbethi: the list's edges 55668,44,3^4 match no atlas shape; two atlas regions have its cells exactly (240 snits, 480 trigypos, 1440 phydows, 2880 tets), with edges 8,6,4^7,3^4 and 8,6,55,444,3^6
- awosbitdy: found by the surface survey on sheet ix9, on the wall b1 b3 = (sqrt(5) - 1)/2 b2 b4: cells exactly the list's (120x2 snits, 480 trigypos, 720x2 phywdows, 1440+1440 tets and 1440 hexflanges, 6480) and edges 88 12,4^4,3^10 exactly
- not found: sastithi and gisbethi (4080 cells: none in the atlas) are not matched

- atlas shapes with exactly ixitic symmetry and no name in the list: 21.

- added from the surface survey (add_survey_shapes.py): 41ab2de5065a (ix9, 6480 cells), awosbitdy (ix9, 6480 cells)
