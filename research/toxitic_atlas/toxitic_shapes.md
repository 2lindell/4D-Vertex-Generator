# Toxitic and oxitic shapes

Toxitic = [3+,4,3+] = <r1 r2, r3 r4> (288, rotations only); oxitic = [[3+,4,3+]] (576, toxitic with the 24-cell/dual swap). Frame and domain: the contic atlas's (F4 chamber, b1 >= b4 by the swap; betas are distances to the mirrors, a1, a2 long, a3, a4 short). Made by contic_atlas/snub_map.py (CONTIC_SUB=tox|ox|oxb), wall_shapes.py, and here strata.py (faces, edges, corners), survey.py (dense edge zones) and shapes.py.

## The orbifolds (structure.py): the descriptions confirmed

**Toxitic** (order 288):

- 24 points, stabilizer T (12), at b1 = b2 = b3 = 0
- 24 points, stabilizer T (12), at b2 = b3 = b4 = 0
- 16 rings, stabilizer C3: orbits of 96, along b1 = b2 = 0
- 16 rings, stabilizer C3: orbits of 96, along b3 = b4 = 0
- 18 rings, stabilizer C2: orbits of 144, along b2 = b3 = 0

**Oxitic** (order 576):

- 48 points, stabilizer T (12), at b1 = b2 = b3 = 0
- 144 points, stabilizer D2 (4), at b2 = b3 = 0, b1 = b4
- 32 rings, stabilizer C3: orbits of 192, along b1 = b2 = 0
- 36 rings, stabilizer C2: orbits of 288, along b1 = b4, b2 = b3
- 18 rings, stabilizer C2: orbits of 288, along b2 = b3 = 0

## Toxitic

The normaliser of toxitic is all of [[3,4,3]]: every F4 reflection and the swap map toxitic orbits to congruent ones, so the half-chamber is the whole shape space and its faces are mirrors (where the polytope gains a mirror: icopyritic [3,4,3+] or its dual, order 576).

| kind | vertices | cells | edges by valence | symmetry | seed (beta) | found by |
|---|---|---|---|---|---|---|
| region | 288 | 528 | {3: 864, 4: 720} | 288 (exactly the group) | [0.06257, 0.03157, 0.87412, 0.03173]  | map; strata: fold b1 = b4, line b1 = b4, b2 = b3 |
| wall | 288 | 144 | {3: 864} | 576 | [0.02511, 0.02528, 0.94961, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 288 | 240 | {3: 576, 4: 432} | 576 | [0.05009, 0.0, 0.92451, 0.0254]  | strata: face b2 = 0, face b3 = 0, line b1 = b4, b2 = 0, line |
| line | 96 | 144 | {3: 288, 4: 144} | 576 | [0.02512, 0.97488, 0.0, 0.0]  | strata: line b3 = b4 = 0 |
| line | 144 | 240 | {4: 576} | 1152 | [0.52484, 0.0, 0.0, 0.47516]  | strata: line b2 = b3 = 0, point (1,0,0,1) |
| line | 288 | 48 | {3: 576} | 1152 | [0.0, 0.02528, 0.97472, 0.0]  | strata: line b1 = b4 = 0 |
| point | 24 | 24 | {3: 96} | 1152 | [1.0, 0.0, 0.0, 0.0]  | strata: point (1,0,0,0) |
| point | 96 | 48 | {3: 288} | 1152 | [0.0, 1.0, 0.0, 0.0]  | strata: point (0,0,1,0), point (0,1,0,0) |

Shapes with exactly toxitic symmetry: 1. The map (N = 32), every face, edge and corner (N = 40), and a dense survey of every edge and corner zone find one polychoron in the whole open domain.

## Oxitic

The normaliser of oxitic is kicontic (the rotations of [[3,4,3]]), not [[3,4,3]]: the F4 reflections conjugate oxitic to its mirror image (toxitic with phi r1 r4 in place of phi). So a seed and its mirror image give different oxitic polychora, and the shape space is two copies of the half-chamber (sheet ox: oxitic orbits; sheet oxb: orbits under the mirror-image group, i.e. mirror images of the oxitic orbits of the reflected seeds), glued along the mirror faces (there both give the same polychoron, mirrored). The mirrors are no walls; the walls are curved (quadrics and quartics over Q(sqrt2)) and the plane b2 = b3.

| kind | vertices | cells | edges by valence | symmetry | seed (beta) | found by |
|---|---|---|---|---|---|---|
| region | 576 | 1728 | {3: 1152, 4: 1440, 5: 576, 6: 288} | 576 (exactly the group) | [0.06257, 0.40602, 0.49968, 0.03173]  | map; strata: face b2 = 0; strata: face b2 = 0, fold b1 = b4 |
| region | 576 | 2304 | {3: 1152, 4: 1728, 5: 576, 6: 288, 8: 288} | 576 (exactly the group) | [0.06257, 0.46843, 0.43727, 0.03173]  | map |
| region | 576 | 2304 | {3: 1152, 4: 1728, 5: 576, 6: 288, 8: 288} | 576 (exactly the group) | [0.09378, 0.31241, 0.56208, 0.03173]  | map |
| region | 576 | 2304 | {3: 1152, 4: 1728, 5: 576, 6: 288, 8: 288} | 576 (exactly the group) | [0.15618, 0.12519, 0.59329, 0.12534]  | map; strata: fold b1 = b4 |
| region | 576 | 2688 | {3: 576, 4: 1728, 5: 1152, 6: 288, 8: 288} | 576 (exactly the group) | [0.40581, 0.12519, 0.12523, 0.34377]  | map |
| region | 576 | 2688 | {3: 576, 4: 2016, 5: 576, 6: 288, 7: 576} | 576 (exactly the group) | [0.06257, 0.2812, 0.62449, 0.03173] (oxb) | map; strata: fold b1 = b4 |
| region | 576 | 3168 | {3: 1728, 4: 1440, 5: 576, 6: 864, 12: 288} | 576 (exactly the group) | [0.21859, 0.34361, 0.25005, 0.18775]  | map |
| region | 576 | 3264 | {3: 576, 4: 2016, 5: 576, 6: 1152, 8: 288} | 576 (exactly the group) | [0.06257, 0.03157, 0.87412, 0.03173]  | map; strata: fold b1 = b4 |
| region | 576 | 3264 | {3: 1152, 4: 1728, 5: 576, 6: 576, 8: 288, 10: 288} | 576 (exactly the group) | [0.06257, 0.49963, 0.40607, 0.03173]  | map |
| region | 576 | 3264 | {3: 1152, 4: 1440, 5: 576, 6: 864, 8: 576} | 576 (exactly the group) | [0.71785, 0.06278, 0.03162, 0.18775]  | map |
| region | 576 | 3264 | {3: 1152, 4: 1152, 5: 576, 6: 1152, 7: 576} | 576 (exactly the group) | [0.06257, 0.03157, 0.87412, 0.03173] (oxb) | map; strata: fold b1 = b4 |
| region | 576 | 3552 | {3: 576, 4: 2880, 5: 576, 7: 576, 12: 288} | 576 (exactly the group) | [0.06257, 0.62445, 0.28125, 0.03173]  | map |
| region | 576 | 3840 | {3: 1728, 4: 864, 5: 1152, 6: 576, 8: 576, 10: 288} | 576 (exactly the group) | [0.06257, 0.34361, 0.56208, 0.03173]  | map |
| region | 576 | 3840 | {3: 1152, 4: 2016, 5: 576, 6: 864, 8: 288, 12: 288} | 576 (exactly the group) | [0.06257, 0.37482, 0.53088, 0.03173]  | map |
| region | 576 | 3840 | {3: 1728, 4: 864, 5: 576, 6: 1152, 7: 576, 10: 288} | 576 (exactly the group) | [0.09378, 0.59324, 0.25005, 0.06294]  | map; strata: fold b1 = b4 |
| region | 576 | 3840 | {3: 1152, 4: 2016, 6: 1728, 12: 288} | 576 (exactly the group) | [0.12498, 0.53083, 0.25005, 0.09414]  | map; strata: fold b1 = b4 |
| region | 576 | 3840 | {3: 1152, 4: 2016, 5: 576, 6: 288, 7: 576, 8: 288, 10: 288} | 576 (exactly the group) | [0.70445, 0.0309, 0.01226, 0.2524]  | survey |
| region | 576 | 4128 | {3: 1728, 4: 1152, 5: 576, 6: 1152, 8: 576, 10: 288} | 576 (exactly the group) | [0.02225, 0.75995, 0.2082, 0.0096]  | survey |
| region | 576 | 4128 | {3: 1152, 4: 2304, 5: 576, 6: 576, 8: 576, 12: 288} | 576 (exactly the group) | [0.65184, 0.03445, 0.01143, 0.30229]  | survey |
| wall | 576 | 336 | {3: 1728} | 576 (exactly the group) | [0.13903, 0.19315, 0.66782, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 576 | 384 | {3: 2304} | 576 (exactly the group) | [0.05456, 0.5249, 0.42054, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 576 | 672 | {3: 1152, 4: 1152} | 576 (exactly the group) | [0.19924, 0.0, 0.72469, 0.07607]  | strata: face b2 = 0, face b3 = 0, line b1 = b4, b2 = 0, line |
| wall | 576 | 816 | {3: 1728, 4: 864} | 576 (exactly the group) | [0.10945, 0.29673, 0.56208, 0.03173]  | map walls: b1 - sqrt(2)*b2/2 + b2 - b3 + sqrt(2)*b3/2 - b4; strata: face b2 = 0 |
| wall | 576 | 864 | {3: 1728, 4: 1152} | 576 (exactly the group) | [0.02511, 0.44995, 0.52493, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 576 | 1008 | {3: 2016, 4: 864, 5: 288} | 576 (exactly the group) | [0.06257, 0.45285, 0.45285, 0.03173]  | map walls: b2 - b3 |
| wall | 576 | 1056 | {3: 1152, 4: 1152, 5: 576} | 576 (exactly the group) | [0.02511, 0.02528, 0.94961, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 576 | 1152 | {3: 576, 4: 2304} | 576 (exactly the group) | [0.19995, 0.0, 0.62478, 0.17527]  | strata: face b2 = 0, face b3 = 0, line b1 = b4, b2 = 0, line |
| wall | 576 | 1248 | {3: 864, 4: 1728, 5: 288} | 576 (exactly the group) | [0.43702, 0.16897, 0.17505, 0.21895]  | map walls: b1**2 - sqrt(2)*b1*b2 + b1*b2 - 2*b1*b3 - b1*b4 +; map walls: b1*b3 - sqrt(2)*b1*b4/2 + b1*b4 - sqrt(2)*b2*b3 |
| wall | 576 | 1248 | {4: 2880} | 576 (exactly the group) | [0.02511, 0.5249, 0.44999, 0.0]  | strata: face b4 = 0, line b2 = b4 = 0 |
| wall | 576 | 1728 | {3: 576, 4: 2592, 8: 288} | 576 (exactly the group) | [0.16259, 0.08757, 0.62449, 0.12534]  | map walls: b1*b2 + sqrt(2)*b1*b3/2 - sqrt(2)*b2*b4/2 - b3*b4 |
| wall | 576 | 1728 | {3: 2304, 4: 864, 5: 576, 8: 288} | 576 (exactly the group) | [0.21859, 0.34595, 0.24771, 0.18775]  | map walls: b1*b3 - sqrt(2)*b2*b4 + b2*b4 - 2*b3*b4 + sqrt(2) |
| wall | 576 | 1824 | {3: 1152, 4: 1152, 5: 864, 7: 288} | 576 (exactly the group) | [0.06257, 0.48079, 0.4249, 0.03173]  | map walls: b1**2 - sqrt(2)*b1*b2 + b1*b2 - 2*b1*b3 - b1*b4 + |
| wall | 576 | 1968 | {3: 864, 4: 2304, 6: 288, 7: 288} | 576 (exactly the group) | [0.40577, 0.15644, 0.15644, 0.28136]  | map walls: b2 - b3 |
| wall | 576 | 2112 | {3: 1728, 4: 1152, 5: 576, 6: 288, 8: 288} | 576 (exactly the group) | [0.06746, 0.36993, 0.53088, 0.03173]  | map walls: b1*b3 - sqrt(2)*b1*b4/2 + b1*b4 - sqrt(2)*b2*b3/2 |
| wall | 576 | 2112 | {3: 1728, 4: 1440, 6: 576, 10: 288} | 576 (exactly the group) | [0.06257, 0.60094, 0.30476, 0.03173]  | map walls: b1*b3 - b2*b4 |
| wall | 576 | 2112 | {3: 1728, 4: 1152, 5: 576, 6: 288, 8: 288} | 576 (exactly the group) | [0.15618, 0.10422, 0.61426, 0.12534]  | map walls: b1*b3 - sqrt(2)*b1*b4/2 + b1*b4 - sqrt(2)*b2*b3/2; strata: fold b1 = b4 |
| wall | 576 | 2112 | {3: 576, 4: 1152, 5: 576, 6: 1152} | 576 (exactly the group) | [0.05009, 0.0, 0.92451, 0.0254]  | strata: face b2 = 0, face b3 = 0, line b1 = b4, b2 = 0, line |
| wall | 576 | 2112 | {4: 2016, 5: 1152, 6: 288} | 576 (exactly the group) | [0.52467, 0.0, 0.02531, 0.45002]  | strata: face b2 = 0 |
| wall | 576 | 2112 | {3: 1152, 4: 1728, 5: 576, 6: 576} | 576 (exactly the group) | [0.07818, 0.25, 0.64009, 0.03173] (oxb) | map walls: b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4; strata: fold b1 = b4 |
| wall | 576 | 2400 | {3: 1728, 4: 1728, 6: 576, 10: 288} | 576 (exactly the group) | [0.2404, 0.353, 0.21884, 0.18775]  | map walls: b1**2 - sqrt(2)*b1*b2 + b1*b2 + b1*b3 + 2*sqrt(2) |
| wall | 576 | 2400 | {3: 864, 4: 2016, 5: 576, 7: 288, 8: 288} | 576 (exactly the group) | [0.69529, 0.06278, 0.03162, 0.21031]  | map walls: b1*b3 - sqrt(2)*b1*b4/2 - sqrt(2)*b2*b3/2 + b2*b3 |
| wall | 576 | 2688 | {3: 1728, 4: 864, 5: 1152, 6: 576, 8: 288} | 576 (exactly the group) | [0.08522, 0.2812, 0.60185, 0.03173]  | map walls: b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4 |
| wall | 576 | 2688 | {3: 1728, 4: 1152, 5: 576, 6: 864, 8: 288} | 576 (exactly the group) | [0.16021, 0.05875, 0.6557, 0.12534]  | map walls: b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4; strata: fold b1 = b4 |
| wall | 576 | 2688 | {3: 1152, 4: 2304, 6: 864, 8: 288} | 576 (exactly the group) | [0.06257, 0.69731, 0.20838, 0.03173]  | map walls: b1*b3 - sqrt(2)*b2*b4 + b2*b4 - 2*b3*b4 + sqrt(2) |
| wall | 576 | 2688 | {3: 1152, 4: 2592, 6: 576, 10: 288} | 576 (exactly the group) | [0.09378, 0.57835, 0.26493, 0.06294]  | map walls: b1*b3 - sqrt(2)*b2*b4 + b2*b4 - 2*b3*b4 + sqrt(2) |
| wall | 576 | 2688 | {3: 1728, 4: 1152, 5: 1152, 6: 288, 12: 288} | 576 (exactly the group) | [0.15618, 0.09398, 0.62884, 0.12099]  | map walls: b1*b2 + sqrt(2)*b1*b3/2 - sqrt(2)*b2*b4/2 - b3*b4 |
| wall | 576 | 2976 | {3: 1440, 4: 1152, 5: 1152, 6: 288, 7: 288, 10: 288} | 576 (exactly the group) | [0.06305, 0.34314, 0.56208, 0.03173]  | map walls: b1*b2 + sqrt(2)*b1*b3/2 - b1*b4/2 + sqrt(2)*b1*b4 |
| wall | 576 | 2976 | {3: 1152, 4: 1440, 5: 1152, 6: 576, 10: 288} | 576 (exactly the group) | [0.12498, 0.55893, 0.22195, 0.09414]  | map walls: b1**2 - sqrt(2)*b1*b2 + b1*b2 - b1*b3 + sqrt(2)*b; strata: fold b1 = b4 |
| line | 192 | 1248 | {4: 864, 6: 288, 8: 288} | 576 (exactly the group) | [0.02512, 0.97488, 0.0, 0.0]  | strata: line b3 = b4 = 0 |
| line | 288 | 528 | {3: 576, 5: 576, 8: 144} | 2304 | [0.52484, 0.0, 0.0, 0.47516]  | strata: line b2 = b3 = 0 |
| line | 288 | 528 | {3: 864, 4: 720} | 576 (exactly the group) | [0.02511, 0.47489, 0.47489, 0.02511]  | strata: line b1 = b4, b2 = b3 |
| line | 576 | 192 | {3: 576, 4: 864} | 2304 | [0.0, 0.02528, 0.97472, 0.0]  | strata: line b1 = b4 = 0 |
| point | 48 | 288 | {3: 192, 8: 144} | 2304 | [1.0, 0.0, 0.0, 0.0]  | strata: point (1,0,0,0) |
| point | 144 | 240 | {4: 576} | 2304 | [0.5, 0.0, 0.0, 0.5]  | strata: point (1,0,0,1) |
| point | 192 | 192 | {3: 576, 4: 288} | 2304 | [0.0, 1.0, 0.0, 0.0]  | strata: point (0,0,1,0), point (0,1,0,0) |

Totals: {'region': 19, 'wall': 29, 'line': 4, 'point': 3}; with exactly oxitic symmetry: {'region': 19, 'wall': 29, 'line': 2}.

