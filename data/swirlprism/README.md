# Swirlprism data

Everything known about the shapes of the `h4_swirlprism` symmetry (order 1200), as plain data.

- `source/`: the atlas's data, copied unchanged from `research/swirlprism_atlas` on
  `claude/laughing-noether-qydcwe` (commit `8488da9`). It holds the named shapes with counts and
  seeds (`working_data.json`, `working_base.json`), the exact surfaces (`surfaces.json`), points,
  lines and wall equations, the cell frame (`cell_vertices.npy`) and the half-turn `Q` that
  normalizes the group (`normalizer_q.npy`).
- `sphere_map/`: the whole 3-sphere built from it by `four_d_vertex_generator.sphere_map`. See
  `sphere_map/REPORT.md` for a readable summary and `sphere_map/manifest.json` for counts and
  checks.

## Coordinates

A point of the fundamental domain is given by barycentric weights `beta = (b1, b2, b3, b4)` on the
four corners of one 600-cell tetrahedron (rows of `cell_vertices`), with `b3 >= b4`: half the cell.
Its point on S^3 is `normalize(beta @ cell_vertices)`. The group's 1200 elements (`group.json`)
carry this half-cell onto 1200 copies that tile S^3 exactly once; copy `k` is `elements[k]` applied
to the domain, and `elements[0]` is the identity.

Every surface is an equation in `beta`: a plane `n . beta = 0`, or a polynomial taken at `A beta`
(`monomials` are index lists into `A beta`, with `coefficients`). The library also records, for each
surface, the atlas surface it is an image of and the map (`group.npz: beta_maps`, the 2400 maps of
`G` and `G.Q`) that carries it there.

## How the map is organised

| level | file | what it is |
|---|---|---|
| piece | `pieces.json` | one cell of the surface arrangement in the domain: a sign (+/-) for every library surface. All points with those signs form the piece. Each piece has its region, its neighbours across surfaces, and its continuations across the domain's faces into neighbouring copies. |
| patch | `patches.json` | where two pieces of different regions meet: a piece of one wall, exactly the points of one surface where every other surface has the sign the two pieces share. Named from the working data and checked by classifying a point on it. |
| region | `regions.json` | a 3D shape family. Its components on S^3 (connected volumes) fall into orbits under the group; each orbit has a representative component listed as `(copy, piece)` pairs, its stabilizer, and its boundary: for each wall, the region across it and the exact `(copy, patch)` parts. Any other component is `g` applied to the representative (`copy k -> multiplication[g, k]`). |
| wall | `walls.json` | a 2D shape family: its patches and its components (sheets) on S^3 as `(copy, patch)` pairs. |
| line, point | `lines.json`, `points.json` | the atlas's exact curves and points carried to every copy. |
| tables | `components.npz` | `region_component[copy, piece]` and `wall_component[copy, patch]`: the component every copy of every piece and patch belongs to. |

## The closed database: `atlas.json`

One record per class of element under the symmetry (build with
`python -m four_d_vertex_generator.sphere_map.atlas_db`):

| key | each record |
|---|---|
| `points` | a corner: exact beta (copy 0), R^4 point, the surfaces through it (their equations by library name), atlas point name if it is one, copies on the sphere |
| `line_pieces` | a line piece run corner to corner across domain cuts: its segments (copy, edge, the two surfaces and their equations), end points, walls and regions round it, line shape, length |
| `faces` | a piece of one wall between two region components: wall, regions, its two sides as region references, the surfaces it lies on, its outline as line-piece references, congruence with the other faces of its labels |
| `regions` | a region volume: shape, volume share, and every face bounding it |
| `counts_per_wall` | a wall's piece types (one per pair of regions it separates; all faces of a type are congruent), classes and faces on the sphere, unplaced fragments |
| `counts_per_line_shape` | an atlas line split where the walls and regions round it change: piece types and runs |
| `closure` | the checks: two sides per face, outlines, line-piece ends, regions bounded |

A reference `{"class": c, "g": k}` is class `c`'s representative carried by group element `k`
(`group.npz` elements); a reference inside an element placed at `g` sits at
`multiplication[g, k]`. That fixes the relative position of every element.

## Reading it

```python
from four_d_vertex_generator.sphere_map import SphereMap

m = SphereMap.load()
m.locate(p)                    # copy, beta, piece, region, region component of a point on S^3
m.domain_view(k)               # every piece and patch of copy k, with components, and its placement
m.region_component("F2")       # one whole F2 volume: (copy, piece) pairs
m.boundary("F2")               # its faces: wall, region across, exact parts
m.wall_component("T2")         # one whole T2 sheet: (copy, patch) pairs
```

## Rebuilding

```bash
python -m four_d_vertex_generator.sphere_map.build                    # from surfaces.json
python -m four_d_vertex_generator.sphere_map.build --resume-library   # reuse surface_library.json
```
