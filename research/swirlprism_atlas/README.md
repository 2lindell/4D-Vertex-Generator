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

Between the rings, sampling finds many more 1200-vertex types than the wiki
lists; the atlas key has the full, current list.
