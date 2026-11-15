# Crossing search: shapes where interior walls cross

The atlas maps take lines and points only on the chamber's own strata (its edges, corners, fold lines); a shape that
lives where two curved (or plane) walls of the map cross inside the chamber was never sampled -- oxitic_line_01, on
b2 = b3 and a quadric, was found by the community that way. For every pair of the sheet's wall surfaces (snub_map's
"surfaces"), points of their crossing curve inside the domain (betas positive, off the chamber's boundary; the fold
kept where the group has one) are found by alternating projections from random starts, and the polytope there is
classified (wall tolerance). A key not among the sheet's known polytopes (map regions, wall shapes, strata, the
family's merged shape lists) is a candidate: stepped 1e-4 along the curve (both ways) it is a line if it stays, else
an isolated point on the curve. Triples of surfaces are searched the same way for points (only those whose
three pairs all cross inside: a point on three walls lies on each pair's curve).

| atlas | sheet | surfaces | crossing points | new key | kind | seed | on | polytope |
|---|---|---|---|---|---|---|---|---|
| contic | icp | 27 | 206 | none |  |  |  |  |
| contic | icpb | 27 | 205 | none |  |  |  |  |
| contic | io | 28 | 217 | d00f3022144f | wall | [0.615322, 0.013226, 0.020907, 0.350545] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4 - b1**2*b3*b4 + sqrt(2)*b1**2*b3*b4/2 - sqrt(2)*b1*b2**2*b3/2 + b1*b2* | 1152: cells 48x[24v38f]+144x[8v10f]+192x[6v8f]+288x[4v4f]+576x[4v4f]+1152x[5v6f]  |
| contic | ki | 36 | 569 | d21ede0dace5 | wall | [0.903963, 0.002011, 0.027212, 0.066814] | b1*b2 - b3*b4 | 1152: cells 48x[24v38f]+144x[8v10f]+192x[6v8f]+288x[4v4f]+576x[4v4f]+1152x[4v4f]+1152x[4v4f]+1152x[5v5f]  |
| contic | ki | 36 | 569 | 09c95d10b74a | wall | [0.605511, 0.014636, 0.018352, 0.361501] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4 - b1**2*b3*b4 + sqrt(2)*b1**2*b3*b4/2 - sqrt(2)*b1*b2**2*b3/2 + b1*b2* | 1152: cells 48x[24v38f]+144x[8v10f]+192x[6v8f]+288x[4v4f]+576x[4v4f]+576x[4v4f]+1152x[4v4f]+1152x[5v6f]  |
| contic | ki | 36 | 569 | 3a8177163bd8 | line | [0.033457, 0.377287, 0.566993, 0.022263] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4 - b1**2*b3*b4 + sqrt(2)*b1**2*b3*b4/2 - sqrt(2)*b1*b2**2*b3/2 + b1*b2* | 1152: cells 48x[24v38f]+144x[8v10f]+192x[6v8f]+288x[4v4f]+1152x[6v6f]  |
| contic | ki | 36 | 569 | bb3d2c1875c2 | line | [0.637314, 0.028089, 0.066864, 0.267733] | b1*b2 - b1*b4 + sqrt(2)*b1*b4/2 - sqrt(2)*b2**2/2 + b2**2 - b2*b3 + sqrt(2)*b2*b3/2 - 2*b2*b4 + 3*sqrt(2)*b2*b4/2 - sqrt | 1152: cells 48x[24v38f]+144x[8v10f]+192x[6v8f]+288x[4v4f]+1152x[6v6f]  |
| contic | ox | 50 | 1332 | b8c25a009132 | wall | [0.811139, 0.17304, 0.002437, 0.013385] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4/2 + sqrt(2)*b1**2*b3**2/2 - b1**2*b3*b4/2 + sqrt(2)*b1**2*b3*b4/2 - sq | 576: cells 48x[12v20f]+144x[4v4f]+192x[6v8f]+288x[4v4f]+288x[4v4f]+288x[4v4f]+576x[4v4f]+576x[4v4f]+576x[5v6f]  |
| contic | ox | 50 | 1332 | d2cbbabe4f15 | wall | [0.859243, 0.083082, 0.008105, 0.04957] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4/2 + sqrt(2)*b1**2*b3**2/2 - b1**2*b3*b4/2 + sqrt(2)*b1**2*b3*b4/2 - sq | 576: cells 48x[12v20f]+144x[4v4f]+192x[6v8f]+288x[4v4f]+288x[4v4f]+576x[4v4f]+576x[4v4f]+576x[5v6f]  |
| contic | ox | 50 | 1332 | 5a23229b7cf9 | wall | [0.646287, 0.041673, 0.018902, 0.293138] | b1*b3 - b2*b4 | 576: cells 48x[12v20f]+144x[4v4f]+192x[6v8f]+288x[4v4f]+288x[6v6f]+576x[4v4f]+576x[4v4f]+576x[5v5f]  |
| contic | ox | 50 | 1332 | a4043696cfa4 | line | [0.072109, 0.304565, 0.558141, 0.065185] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4/2 + sqrt(2)*b1**2*b3**2/2 - b1**2*b3*b4/2 + sqrt(2)*b1**2*b3*b4/2 - sq | 576: cells 48x[12v20f]+144x[4v4f]+192x[6v8f]+288x[4v4f]+576x[4v4f]+576x[6v7f]  |
| contic | ox | 50 | 1332 | a02e136d42e8 | line | [0.631452, 0.050631, 0.023599, 0.294318] | b1**2*b2*b3 - sqrt(2)*b1**2*b2*b4/2 + b1**2*b2*b4/2 + sqrt(2)*b1**2*b3**2/2 - b1**2*b3*b4/2 + sqrt(2)*b1**2*b3*b4/2 - sq | 576: cells 48x[12v20f]+144x[4v4f]+192x[6v8f]+288x[6v6f]+576x[6v6f]  |
| contic | ox | 50 | 1332 | 9f3d3a62de03 | line | [0.181343, 0.005663, 0.684018, 0.128975] | b1*b2 + sqrt(2)*b1*b3/2 - sqrt(2)*b2*b4/2 - b3*b4 ; b1*b3 - sqrt(2)*b1*b4/2 + b1*b4 - sqrt(2)*b2*b3/2 + b2*b3 - 3*sqrt(2 | 576: cells 48x[12v20f]+144x[4v4f]+192x[9v11f]+288x[4v4f]+288x[4v4f]+576x[5v5f]  |
| contic | oxb | 30 | 258 | none |  |  |  |  |
| contic | tox | 27 | 207 | none |  |  |  |  |
| decaic | io | 83 | 2690 | none |  |  |  |  |
| decaic | ki | 76 | 3635 | 6127f3ea279d | wall | [0.060382, 0.367492, 0.565152, 0.006974] | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | 120: cells 10x[12v20f]+20x[6v8f]+20x[6v8f]+30x[4v4f]+60x[4v4f]+60x[4v4f]+120x[4v4f]+120x[5v6f]  |
| decaic | ki | 76 | 3635 | 2cc54ef1b550 | line | [0.042913, 0.358412, 0.571775, 0.0269] | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | 120: cells 10x[12v20f]+20x[6v8f]+20x[6v8f]+30x[4v4f]+120x[6v6f]  |
| decaic | ki | 76 | 3635 | aced5db9ac46 | point | [0.51554, 0.009063, 0.010041, 0.465356] | 3*b1**2*b2*b3 + b1**2*b2*b4 - b1**2*b3*b4 + b1*b2**2*b3 + 2*b1*b2**2*b4 - b1*b2*b3**2 - 5*b1*b2*b3*b4 - b1*b2*b4**2 + 2* | 120: cells 10x[12v20f]+20x[6v8f]+20x[6v8f]+30x[4v4f]+120x[6v6f]  |
| hyic | ix0 | 7 | 76 | none |  |  |  |  |
| hyic | ix1 | 5 | 21 | none |  |  |  |  |
| hyic | ix2 | 18 | 539 | d9ca4ec7c345 | line | [0.043824, 0.057736, 0.105902, 0.792538] | b1*b2 - 3*b1*b3/2 + sqrt(5)*b1*b3/2 + b2**2 - sqrt(5)*b2*b3 + 2*b2*b3 - sqrt(5)*b3**2 + 2*b3**2 ; b1*b2 - sqrt(5)*b1*b4  | 1440: cells 120x[12v20f]+120x[12v20f]+480x[6v5f]+720x[8v10f]  |
| hyic | ix3 | 14 | 170 | none |  |  |  |  |
| hyic | ix4 | 2 | 4 | none |  |  |  |  |
| hyic | ix5 | 7 | 43 | none |  |  |  |  |
| hyic | ix6 | 32 | 1166 | none |  |  |  |  |
| hyic | ix7 | 3 | 6 | none |  |  |  |  |
| hyic | ix8 | 5 | 37 | none |  |  |  |  |
| hyic | ix9 | 18 | 608 | none |  |  |  |  |
