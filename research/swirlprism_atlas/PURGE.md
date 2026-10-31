# Purge list

The pipeline no longer reads any old data. Everything it uses is listed under **Keep**; everything else in this folder
is listed in `purge_manifest.json` under `purge` (339 entries) and can be deleted. **Nothing is deleted yet.**

This was checked two ways: the static closure of every step (every local module it imports, transitively, and every
data file those modules name), and an audit hook on real runs of `working_data.py`, `wall_parts.py`,
`build_shape_map.py`, `cross_ring_points.py`, `pieces.py`, `ring_test.py` and `sphere_complex.py` (every file opened,
every local module imported).

## The pipeline

1. `edges_from_equations.py` → `wall_equations.json` (wall edges and cuts from the equations)
2. `line_ends.py` → `line_ends.json`, `exact_defs.json` (each line's exact ends)
3. `cross_ring_points.py` → `cross_ring_points.json` (the exact points on the cross ring)
4. `pieces.py` → `pieces.json` (each wall's patches, grown out to its edges and divided at its cuts into its pieces)
5. `wall_parts.py` → `wall_parts.json` (the pieces named from the working data)
6. `sphere_complex.py` → `sphere_complex.json` (each wall as a whole on the 3-sphere)
7. `ring_test.py` → `ring_test.json` (the walls and regions round each line)
8. `working_data.py` → `working_data.json` (the base and everything computed from it)
9. `build_shape_map.py` → `shape_map.html`; `domain_page.py` → `../../assets/symmetry_domain.html`
10. For the page's overlays: `regular_cells.py extend_lines` → `regular_lines.json`, `decagons.py` → `regular_decagons.json`

Every step imports only the clean core (`core.py`) and small math modules; none imports the old atlas, page, region or
survey modules.

## Keep

The source of truth:

- `working_base.json` — the verified base facts (each shape's kind, name, counts and seed; each wall's edges, sides,
  cuts and through-lines; each region's walls), with every correction already applied and its reason kept. Edited
  directly; nothing re-derives it.
- `exact_defs.json` — the exact line pieces (with their exact ends) and each wall's surfaces.
- `surfaces.json` — the exact equations of every surface a wall lies on, and their images in the half-cell.

Naming tables and constants:

- `references.json` (reference shapes' signatures), `atlas_xids.json` (X names), `catalog.py` (the wiki's list),
  `core.py` (the wiki ids table) — names only.
- `cell_vertices.npy` (the cell's corners), `normalizer_q.npy` (the half-turn Q), `golden_seeds.json` (the golden seed
  positions, with no stored labels).

Computed by the pipeline: `wall_equations.json`, `line_ends.json`, `cross_ring_points.json`, `pieces.json`,
`wall_parts.json`, `sphere_complex.json`, `ring_test.json`, `working_data.json`, `regular_lines.json`,
`regular_decagons.json`.

Scripts: `core.py`, `cellframe.py`, `classify.py`, `classify_fast.py`, `catalog.py`, `edges_from_equations.py`,
`line_ends.py`, `cross_ring_points.py`, `pieces.py`, `wall_parts.py`, `sphere_complex.py`, `ring_test.py`,
`working_data.py`, `build_shape_map.py`, `domain_page.py`, `overlays.py`, `dodeca_view.py`, `chamber_domain.py`,
`dirichlet.py`, `normalizer.py`, `supergroups.py`, `regular_cells.py`, `decagons.py`, `place_colors.py`.

Pages and templates: `shape_map_src.html`, `shape_map.html`, `cell_atlas2_template.html`. Docs: `README.md`,
`PURGE.md`, `purge_manifest.json`. Folder: `examples/`.

## What changed to cut the old sources loose

- **The core** (`core.py`) holds the cell, the group, the classifier and its naming, the surface library and the mesh
  grid. It replaces `probe.py`, `optimal_seeds.py`, `t2_pieces.py`, `edge_relabel.py`, `xsurface.py` and the parts of
  `cell_atlas.py` / `cell_atlas2.py` the pipeline used, and gives the same results (checked: the same group, the same
  labels at test points, the same 132 planes and 42 curved surfaces, the same fold).
- **The surface library** was rebuilt every run from the old page and the `fexact_*.json` plane fits; it is now frozen
  in `surfaces.json` (checked equal to the library the pieces were built on).
- **The working data** was assembled every run from the old page, `boundaries.json`, `boundaries_extra.json`,
  `wall_outline.json`, `wall_sides.json`, `region_defs.json`, `edge_lines.json` and the sampled `piece_regions.json`;
  it now starts from `working_base.json` and adds only what the pipeline computes. A wall's points are the ends of its
  edge lines; a region's wall extent comes from the named pieces; closure is checked on the pieces' own rims.
- **The Shape Map** read the old page, `boundaries.json` and `wall_edges.json`; it now reads only the working data,
  the pieces, the exact lines and the cross-ring points.
- **The Symmetry Domain page** read its type list, samples, rings and overlays from the old page; the key now comes from
  the working data, the samples and uniform seeds are named by the classifier, the rings are the exact lines, and the
  overlays are computed from the group (`overlays.py`).
- `regular_cells.py` lost its exploratory subcommands that read old stored labels or the old survey modules.

## Purge

Everything else in this folder: the old atlas and page builders (`cell_atlas.py`, `cell_atlas2.py`, `build_atlas.py`,
...), the region and section tools (`region_*.py`, `wall_sections.py`, `arrangement.py`, ...), the old classifier
wrappers (`probe.py`, `optimal_seeds.py`, `edge_relabel.py`, `t2_pieces.py`), the old surveys and their outputs
(`fexact_*.json`, `xloci_*.json`, `boundaries*.json`, `wall_outline*.json`, `edge_lines.json`, `region_*.json`,
`golden_22.json`, `optimal_seeds.json`, `extra_samples.json`, `piece_regions.json`, ...), old caches, and the folders
`arr_pieces/`, `region_pieces/`, `sec_pieces/`, `sections/`, `logs_tmp/`. The full list is in `purge_manifest.json`.
