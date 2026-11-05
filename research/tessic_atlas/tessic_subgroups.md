# Tessic family: the subgroup maps, against the list

All groups use the B4 chamber (beta = distances to the mirrors; b1 the hexadecachoron's node, b4 the tesseract's; no
fold: B4 has no diagram symmetry). Seeds differing by a tessic element give congruent polytopes, since B4 normalizes
each subgroup. Maps: tessic and demitessic by the exact plane arrangement (arrangement.py); kitessic, tessipyritic and
kidemitessic (no reflections inside the chamber, or snub-type) by the curved-wall map (snub_map.py, grid N = 32),
the dense edge-zone survey (edge_survey.py) and the region closure check (snub_closure.py); and the chamber's faces where
a subgroup's orbits are only part of the tessic ones by a 2D survey of the face (face_survey.py).

| group | order | chamber interior | faces | matches the list |
|---|---|---|---|---|
| tessic [4,3,3] | 384 | gidpith (3 degrees) | the other 14 Wythoffian shapes, one per face, edge and corner (tessic_shapes.md) | all 15, names and degrees |
| kitessic [4,3,3]+ | 192 | **snet**: 192 vertices, 8 snub cubes, 16 snits, 24 gyro-squaps, 32 gyro-traps, 192 tets | the tessic shapes (orbits unchanged on every face) | yes; edges 576 trigonic + 288 tetragonic (the list has these swapped: its own verf, 2 pentagons + 2 trapezoids + 4 trigons, has 6 three-face and 3 four-face corners) |
| tessipyritic [4,(3,3)+] | 192 | **pysnet**: 192 vertices, 176 cells (8 pyritosnub cubes, 16 snits, 24 rectangular trapezoprisms, 32 trips, 96 skew wedges); edges 480 / 96 / 96 | face b4 = 0 (its mirror): **sadi**, 96 vertices, 144 cells, edges 288 trigonic + 144 tetragonic, 2 degrees; other faces: the tessic shapes | yes (pysnet 3 degrees, sadi 2 degrees) |
| demitessic D4 | 192 | tico (24 toes + 24 cubes), 3 degrees | rico and tah (2), rit and thex (1), hex and ico (0) | tico 3, rico 2, tah 2, rit 1 agree; **thex has 1 degree here, the list says 2** |
| kidemitessic D4+ | 96 | **sadi**: 8+8+8 snits, 24 disphenoids, 96 tets, 3 degrees | b4 = 0: sadi too; b1 = 0, b3 = 0: tah; b2 = 0: rico (its demitessic mirrors) | yes |

Region closure (every region walked out to its walls): kitessic, tessipyritic and kidemitessic each have one region
filling the chamber, closed by the domain's edges; the edge-zone surveys found no small regions in any of them.

Notes against the list:

* **tah** is tessipyritic too (its tessic seed's stabilizer holds an odd permutation, so the tessipyritic orbit is the
  whole tessic one); the list does not say so.
* **tico** cannot be kidemitessic (order 96 < its 192 vertices). Under tessipyritic its tessic seed gives sadi; if tico
  is tessipyritic at all it is on a special locus inside the chamber where the polytope gains extra (icoic) symmetry --
  not seen by these maps, whose interior is a single combinatorial type everywhere.
* **tico's cells** are truncated octahedra (and cubes), not tuts.
