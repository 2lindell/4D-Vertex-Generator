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
