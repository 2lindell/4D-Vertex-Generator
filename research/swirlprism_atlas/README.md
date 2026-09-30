# Swirlprism seed atlas

Tools and data for mapping which convex isogonal polychora the
`h4_swirlprism` symmetry (H3●I2(10), order 1200) produces, depending on where
the seed point sits.

Every seed is equivalent to one inside the anchor's Voronoi dodecahedron (the
points closer to the 600-cell vertex `(1, 0, 0, 0)` than to any other). The
tools use the gnomonic projection at the anchor, `q = p / (p · a) − a`, where
great circles are straight lines, so every cross ring and main ring is a
straight segment and the dodecahedron has flat faces. The anchor's stabilizer
(order 10) acts linearly on `q`, so only one tenth of the dodecahedron
(`z ≥ 0`, azimuth 0°–72°) is sampled.

A plot position `(x, y, z)` corresponds to the seed
`(1, 0.850651·y − 0.525731·z, x, −0.525731·y − 0.850651·z)`, normalised.

## Files

| File | Purpose |
| --- | --- |
| `classify.py` | Convex hull of an orbit, split into symmetry classes of cells, faces and edges (the counts the wiki uses), plus edge valences. |
| `survey.py` | Grid over one tenth of the dodecahedron, classified in parallel. |
| `planes.py` | Grids on the special planes through the centre, where surface-only types live. |
| `lines.py` | Maps the cross-ring and main-ring results, and the named special points, into the dodecahedron. |
| `catalog.py` | The wiki's 1200-vertex entries, for matching. |
| `build_atlas.py`, `atlas_template.html` | Build the interactive 3D atlas page with the data embedded. |
| `survey_002.json`, `planes_0012.json`, `lines.json` | Results of the runs below. |

## Reproducing

Run from this directory, with the package installed (`pip install -e ../..`):

```bash
python survey.py 0.02 survey_002.json      # ~8 minutes on 4 cores
python planes.py 0.012 planes_0012.json    # ~6 minutes
python lines.py                            # writes lines.json
python build_atlas.py survey_002.json atlas.html planes_0012.json
```

## Findings so far

Cross ring 1, by angle `t` (period 90°; on cross ring 2 read `90° − t`),
with the seed `(cos t, 0, sin t, 0)`:

| t | tan t | Shape |
| --- | --- | --- |
| 0° | 0 | Hexacosichoron (600-cell) |
| 0°–13.28° | | Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges) |
| 13.28° | φ⁻³ | Hecatonicosachoron (120-cell) |
| 13.28°–20.91° | | Pentagonal-gyroprismatic triacosihexecontachoron |
| 20.91° | φ⁻² | Partially-rectified small swirlprism |
| 20.91°–26.57° | | Polychoron with 120+120+1200 cells (600+1200 4-valent edges) |
| 26.57° | ½ | Swirlprismatodiminished rectified hexacosichoron |
| 26.57°–31.72° | | 1200 tetrahedra + 240 pentagonal antiprisms (antiprisms split 120+120) |
| 29.14° | tan(½ arctan φ) | Subsymmetrical icosafold icosidodecaswirlchoron (order-2400 symmetry) |
| 31.72° | φ⁻¹ | Swirlprismatodiminished rectified hexacosichoron |
| 31.72°–37.38° | | Polychoron with 120+120+1200 cells (600+1200 4-valent edges) |
| 37.38° | 2φ⁻² | Partially-rectified small swirlprism |
| 37.38°–45° | | Pentagonal-gyroprismatic triacosihexecontachoron |
| 45° | 1 | Hecatonicosachoron (120-cell) |
| 45°–58.28° | | Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges) |
| 58.28° | φ | Hexacosichoron (the second 600-cell) |
| 58.28°–69.09° | | Polychoron with 120+600+600+600+1200 cells (600 3-valent edges) |
| 69.09° | φ² | Bigyroprismatic transitional didecafold icosidodecaswirlchoron |
| 69.09°–79.19° | | 1200 tetrahedra + 240 pentagonal antiprisms (tetrahedra split 600+600) |
| 74.14° | tan(45° + ½ arctan φ) | Subsymmetrical icosafold icosidodecaswirlchoron (same polytope as at 29.14°) |
| 79.19° | 2φ² | Bigyroprismatic transitional didecafold icosidodecaswirlchoron |
| 79.19°–90° | | Polychoron with 120+600+600+600+1200 cells (600 3-valent edges) |

The 18° main-ring step acts on this ring as `t ↦ 58.28° − t (mod 90°)`; its
fixed points are the two icosafold points, which is why every other shape
appears twice.

The main ring gives the icosafold icosaswirlchoron (240 vertices) everywhere
except at its 20 special points (600-cells, every 18°).

Between the rings (0.02-spacing volume grid plus 0.012-spacing grids on the
special planes through the centre), sampling finds 33 distinct 1200-vertex
types, 5 of which match wiki entries:

- W2 and W3 fill volume regions.
- W1 occurs only on the plane through the main ring and cross ring 1
  (azimuth 0°), separating the volume types U1 and W3.
- W4 and W5 occur (almost) only on the plane through the main ring at
  azimuth 36°.
- Not yet found: T1, T2 and the two named 1200-vertex shapes.

The largest volume region (U1) is not in the wiki list. The atlas key has the
full, current list with an example seed for each type.

Further checks (`symcheck.py`, `offsets.py`, `axes.py`, `cross_lines.py`):

- Every sampled 1200-vertex type has symmetry exactly order 1200 (no
  reflections, no extra rotations), and each is stable under 100x stricter
  and 10x looser facet-merge tolerances, so the unlisted types are genuinely
  distinct polytopes.
- The planes holding W1, W4 and W5 (azimuth 0° and 36°, through the main
  ring and a half-turn axis) are not H4 mirrors; the 5 vertical H4 mirrors
  sit halfway between them (18°, 54°, ...). 15 H4 mirrors cross the region,
  all through the centre.
- W1, W5 and most other plane-only types are boundary surfaces: moving 0.0005
  off the plane gives a different volume type on each side (W1 separates W3
  and U1).
- Along the mirror-intersection lines through the centre: W1 and W4 each fill
  whole lines, and the 600-cell edge direction is entirely the
  bi-hecatonicosadiminished truncated hexacosichoron type (the uniform
  truncation at 32.83% is one point on it).
- The truncated 120-cell's 2400 vertices split into two non-congruent
  1200-vertex halves: the swirlprismatodiminished truncated
  hecatonicosachoron (120+120 cells) and a half with 120+600+600 cells that
  is not in the wiki list.

## Uniform H4 polytopes broken into this symmetry (`uniform.py`)

Splitting each of the 15 uniform H4 polytopes' vertex sets into
h4_swirlprism orbits reproduces every wiki entry:

| Uniform polytope | Orbits (vertices) |
| --- | --- |
| 600-cell | 600-cell (120) |
| 120-cell | 120-cell (600) |
| rectified 600-cell | swirlprismatodiminished rectified hexacosichoron (600), second 600-cell (120) |
| rectified 120-cell | bigyroprismatic transitional didecafold icosidodecaswirlchoron (600), partially-rectified small swirlprism (600) |
| truncated 600-cell | bi-hecatonicosadiminished truncated hexacosichoron (1200), main-ring point (240) |
| truncated 120-cell | swirlprismatodiminished truncated hecatonicosachoron (1200), unlisted half (1200) |
| runcinated 120-cell | W1, one unlisted |
| cantellated 120-cell | T2, one unlisted, two cross-ring 600s |
| bitruncated 120-cell | T1, T2, two cross-ring 600s |
| cantellated 600-cell | W4, W1, two cross-ring 600s |
| runcitruncated 120-cell | W1 ×2, W5, W2 ×2, U3 |
| runcitruncated 600-cell | W1 ×2, W4, W3, W2 ×2 |
| cantitruncated 120-cell | T2 ×2, U4, U18, two unlisted |
| cantitruncated 600-cell | W4 ×2, W3, U1, U6, one unlisted |
| omnitruncated 120-cell | W1, W2 ×3, W3 ×2, W5, U1 ×2, U22, two unlisted |

So the wiki's list is (mostly) the pieces of uniform polytopes, though
several such pieces, and most generic volume types, are not listed. T2
occurs at several distinct seed positions, so it is a line or surface
rather than a single point.

## Tracing T2 (`t2probe.py`, `t2surface.py`, `t2trace.py`, `t2extend.py`)

- Every T2 point from the uniform polytopes sits on a boundary surface
  between two volume types (W1|W2, W2|W3, …). Within that surface T2 is a
  curve: probing a circle of directions in the surface finds T2 in two
  opposite directions only.
- At every T2 point the curve's direction lies in an H4 mirror plane (to
  0.1–0.2°), so each T2 curve is where an H4 mirror cuts a boundary surface.
- Tracing inside those mirror planes connects the uniform points into a
  network: the cantitruncated-120-cell T2 curves reach the cantellated and
  bitruncated ones.
- The curves through the cantellated and cantitruncated 120-cell points are
  exact straight lines in the gnomonic view (great-circle arcs). Both end
  where they meet a new type with 120+120+240+600 cells.
- The curve through the bitruncated 120-cell point bends and runs to the
  region boundary; the tracer may have switched branches at a junction, so
  treat that path with caution.

## Half-cell atlas with golden-field seeds

`cellframe.py`, `uniform_in_cell.py`, `golden_survey.py`, `references.py`,
`cell_atlas.py`, `cell_atlas_template.html`.

- h4_swirlprism is transitive on the 600 tetrahedral cells of the 600-cell,
  and each cell's stabilizer is a single half-turn swapping V1<->V2 and
  V3<->V4. So half a cell (barycentric weights beta >= 0 with beta1 >= beta2,
  bounded by the H4 mirror beta1 = beta2) is an exact fundamental domain.
- In barycentric coordinates the cell's H4 mirrors are the planes
  beta_i = beta_j, and every uniform H4 polytope's seed has golden-integer
  weights, all from {0, 1, 2, 3, 1+phi, 1+2phi, 1+3phi}
  (`uniform_in_cell.json`).
- `golden_survey.py 2 2` classifies every seed whose weights are a + b*phi
  with 0 <= a, b <= 2 (2796 points, plus the uniform seeds): 69 types, no
  classification failures, and every wiki entry is found (W1–W5, T1, T2, the
  named 1200-vertex shapes, the 600- and 240-vertex ring shapes and the
  120-cell/600-cell). The atlas key lists all 69 with example seeds.

Rebuild with:

```bash
python golden_survey.py 2 2 golden_22.json
python references.py
python cell_atlas.py golden_22.json cell_atlas.html
```

### Display and findings (cell atlas v2)

- The displayed half is now split by the horizontal mirror beta3 = beta4
  (the perpendicular bisector of the main-ring edge V3V4, which runs
  vertically). Samples from the other half are moved in by the half-turn.
- The half-cell cannot be reduced to a quarter: the mirror beta1 = beta2 does
  not preserve h4_swirlprism, 1616 of 2544 mirrored golden pairs change type,
  only 27 of 64 types always mirror to the same partner, and each quarter has
  types the other lacks (27 and 18). The atlas has a quarter selector.
- T1 lies on lines where a cell face meets a mirror (all 32 samples); `tlines.py`
  finds them as collinear golden runs verified at golden midpoints.
- T2 is mostly flat patches inside H4 mirrors and cell faces (96 samples on
  beta1 = beta3, 48 on beta4 = 0, 36 on beta1 = beta2, ...; 13 off every
  mirror). The atlas outlines each patch as the convex hull of its samples.
- W1 and W4 are mirror/face surface types too (W1: 244 of 291 samples on
  beta1 = beta2; W4: 108 each on beta1 = 0 and beta2 = 0).

### Labels (cell atlas, `cell_atlas2.py`)

The letter gives the vertex count and whether the wiki names the shape;
numbers follow the wiki list's order within each letter.

| Letter | Vertices | Kind | Members |
| --- | --- | --- | --- |
| A | 120 | named | A1 hexacosichoron (600-cell) |
| B | 240 | named | B1 icosafold icosaswirlchoron |
| C | 600 | named | C1 hecatonicosachoron (120-cell), C2a/C2b subsymmetrical icosafold icosidodecaswirlchoron (see below), C3 pentagonal-gyroprismatic triacosihexecontachoron, C4 bigyroprismatic transitional didecafold icosidodecaswirlchoron, C5 partially-rectified small swirlprism, C6 swirlprismatodiminished rectified hexacosichoron |
| D | 600 | unnamed | D1–D3 |
| E | 1200 | named | E1 bi-hecatonicosadiminished truncated hexacosichoron, E2 swirlprismatodiminished truncated hecatonicosachoron |
| F | 1200 | unnamed | F1–F5 (the first atlases' W1–W5) |
| T | 1200 | transitional | T1, T2 |
| Y | 600 | not in the wiki list | none left (the old Y1/Y2 are C2a/C2b) |
| X | 1200 | not in the wiki list | X1… (by number of cell classes, then total cells) |

C2 is split into its two cross-ring ranges, told apart by their class counts:
C2a is the range around the icosafold point at ½·arctan φ ≈ 29.14°
(cells 120+120+1200, faces 240+600+600+1200+1200), and C2b is the range around
45° + ½·arctan φ ≈ 74.14° (cells 240+600+600, faces 120+120+1200+1200+1200).
The exact icosafold points are not golden; they are drawn as ✕ markers in the
colour of their range and have the same counts as it.

The splitting mirror β3 = β4 is folded onto itself by the cell's half-turn
(β1 <-> β2). Seeds on it are stored once, and the atlas also draws each one's
half-turn copy, so both mirror edges from V1 and V2 show their uniform points.

Unlisted types are told apart by edge valences as well as class counts.
Uniform seeds are labelled by the isogonal they produce; the uniform polytope
is named only for X pieces. Marker shape encodes the vertex count, colour is
unique within each vertex count, unlisted shapes are grey. The picture is
drawn upside down (z -> -z) so the splitting mirror is on top.

```bash
python cell_atlas2.py golden_22.json cell_atlas.html
```

### Exact segments and the extra half-turn (`probe.py`, `normalizer.py`)

`probe.py` labels any seed the way the atlas does. `normalizer.py` finds a
half-turn Q that is not in H4 but maps the swirlprism group onto itself, so it
turns every polytope into a congruent one somewhere else in the cell
(`qcopies(beta)` gives where). Q maps planes and lines of the cell to planes
and lines, so every region has a flat copy; the copies are drawn dashed.

- **T1** is the whole line β ∝ (1, 0, 1, u), 0 < u < 1, where the face β2 = 0
  meets the mirror β1 = β3: from spidrox (C6, u = 0) to C4 (u = 1). Its copy
  runs from (1, 1+φ, 1+φ, 0) (C6) to (0, 2+φ, 1, 1) (C4). The line
  (0, 1, 1, u) is the same line seen on the glued face β1 = 0. It used to stop
  short of spidrox because the hull dropped faces thinner than a fixed 1e-4.
- **E2** is exactly the open segment from the face centre (1, 1, 1, 0) (C5) to
  the cell centre (1, 1, 1, 1) (C1); past the cell centre it becomes F1. Its
  copy runs from (2+φ, 2+φ, 1, 1) (C5) to (1+φ, 1, 1, 0) (C1). All 33 golden
  E2 samples lie on one of the two.
- Most off-mirror T2 samples are copies of the T2 patch in the mirror
  β1 = β3.

The hull and orbit code were rebuilt to be tolerance-robust (see
`src/four_d_vertex_generator/off.py` and `generation.py`); re-running the
golden survey with it gives identical signatures for all 2796 samples.
