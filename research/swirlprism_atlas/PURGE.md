# Purge list

Marked for deletion once the new build checks out. **Nothing is deleted yet.** The machine-readable list is `purge_manifest.json`.

The new pipeline is: `working_data.py` → `edges_from_equations.py` → `pieces.py` → `piece_regions.py` → `working_data.py` → `build_shape_map.py` (plus `ring_test.py` to check). Its imports and file reads were traced live (an audit hook on real runs), not guessed from the code.

## 1. Keep

- `working_data.py` — builds the working data (the single verified source)
- `edges_from_equations.py` — wall edges and cuts from the equations
- `pieces.py` — wall pieces built from equations, edges and cuts
- `piece_regions.py` — regions beside each piece; regions' walls (full/partial), lines, points
- `build_shape_map.py` — Shape Map page
- `ring_test.py` — verification: walls around each line
- `optimal_seeds.py` — seeds and the surface library (to be untangled, see below)
- `place_colors.py` — colour rule, also imported by research/tetraswirl_atlas/build_page.py
- `working_data.json` — the verified working data
- `wall_regions_base.json` — frozen wall separates (working-data input)
- `exact_defs.json` — exact line and surface definitions
- `wall_equations.json` — edge/cut surfaces and corners
- `pieces.json` — built pieces (gitignored, regenerated)
- `piece_regions.json` — region data from the pieces
- `ring_test.json` — ring-test results
- `optimal_seeds.json` — example seeds
- `extra_samples.json` — extra example seeds (X100 etc.)
- `xsurface_equations.json` — curved wall equations
- `xsurface_seeds.json` — curved wall seeds
- `atlas_xids.json` — X-id names
- `atlas_xids_frozen.json` — frozen X-id names
- `references.json` — reference shapes (classifier)
- `cell_vertices.npy` — cell frame
- `normalizer_q.npy` — extended group
- `golden_22.json` — seed list (classifier/seeds)
- `shape_map_src.html` — Shape Map template
- `shape_map.html` — Shape Map page
- `README.md`
- `examples/` — example .off files

## 2. Untangle, then purge

The new pipeline still reaches old sources through these. Each one has to be cut loose (its few helpers moved into a small core module, or its data frozen into the working data) before it can go. This matters most for the rule that the working data is the only source.

Biggest leaks:

- **The surface library** (`optimal_seeds.library()` → `arrangement.surfaces()`) is rebuilt every run from the old Symmetry Domain page (`region_walls.page_data`) and 110 `fexact_*.json` plane fits. Fix: freeze the library once into one file (plane normals and quadric coefficients) and read only that.
- **`working_data.py`** still reads the old page, `region_defs.json`, `wall_outline.json`, `edge_lines.json`, `boundaries.json`, `boundaries_extra.json` and `wall_sides.json`. Fix: freeze what it takes from them into a base file, as was already done for the wall separates (`wall_regions_base.json`).
- **`build_shape_map.py`** reads `wall_edges.json`, `boundaries.json` and the old page for labels and seeds. Fix: read only the working data.
- **The classifier** (`probe._one`) goes through `cell_atlas2`, `classify`, `classify_fast`, `cell_atlas`, `cellframe` and `catalog`. Fix: move it into a small module of its own.
- **`exact_defs.py`** made `exact_defs.json` from the old meshes. Fix: keep the JSON as the source and retire the script.

Scripts:

- `probe.py` — classifier _one → cell_atlas2.identify/load_refs, classify, classify_fast, cellframe, catalog, cell_atlas
- `classify.py` — classifier
- `classify_fast.py` — classifier
- `cellframe.py` — seed_from_beta, cell frame
- `catalog.py` — labels
- `cell_atlas.py` — to_upper/seed_text (shape map), refs
- `edge_relabel.py` — lab_edge (point names)
- `t2_pieces.py` — extended() group, but imports region_cells/region_walls/wall_sections at load
- `normalizer.py` — extended group
- `xsurface.py` — marching-tetra constants CUBE/TETS/U/C0
- `arrangement.py` — surfaces() — builds the surface library from fexact_* files and the old page
- `region_walls.py` — page_data() — reads the old Symmetry Domain page
- `region_cells.py` — imported at load
- `wall_copies.py` — imported at load
- `wall_sections.py` — imported at load
- `meetings.py` — imported at load
- `fexact.py` — imported at load (xsurface)
- `exact_defs.py` — made exact_defs.json from meshes (edge_curves/edge_display/xsurfaces.json); freeze its output, retire the script
- `best_seed.py` — seed tool; imports edge_snap

Data:

- `fexact_patches.json` — surface library / exact_defs
- `fexact_walls.json` — surface library
- `xsurface_x33.json` — surface library (X33)
- `region_defs.json` — working_data.py input
- `wall_outline.json` — working_data.py input
- `edge_lines.json` — working_data.py input
- `boundaries.json` — working_data.py and build_shape_map.py input
- `boundaries_extra.json` — working_data.py input
- `wall_sides.json` — working_data.py overrides (F4, X41)
- `wall_edges.json` — build_shape_map.py input
- `region_walls_cache.jsonl` — classifier cache (classify_fast)
- 110 files `fexact_*.json` (plane fits; `fexact_grid_*`, `fexact_fgrid_*`, `fexact_<wall>_*`) — read by the surface library

## 3. Purge with the old Symmetry Domain build

The current page builder and everything only it uses. These go once the new view is built from the pieces. At that point `tools/build_pages.py` and the top-level README must point at the new builder.

Scripts: `cell_atlas2.py`, `chamber_domain.py`, `dirichlet.py`, `dodeca_view.py`, `edge_curves.py`, `edge_display.py`, `edge_shapes.py`, `edge_snap.py`, `exact_conics.py`, `lines.py`, `region_db.py`, `region_paint.py`, `supergroups.py`, `surface_from_lines.py`, `survey.py`, `t2exact.py`

Data: `arr_carriers.json`, `arr_cells.npz`, `atlas_template.html`, `axis_crossings.json`, `cell_atlas2_template.html`, `cell_atlas_template.html`, `edge_curves.json`, `edge_display_cache.npz`, `edge_shapes.json`, `lines.json`, `meetings.json`, `region_bounds.json`, `region_cells.json`, `region_db.json`, `region_grid.json`, `region_walls.json`, `regular_decagons.json`, `regular_lines.json`, `tlines.json`, `uniform_in_cell.json`, `x_point_lines.json`, `xcurves.json`, `xloci_lines.json`, `xloci_step1.json`, `xloci_step2.json`, `xloci_walls.json`, `xsurface_X100_edges.jsonl`, `xsurface_X100_labels.json`, `xsurface_X85_edges.jsonl`, `xsurface_X85_labels.json`, `xsurface_X86_edges.jsonl`, `xsurface_X86_labels.json`, `xsurface_X87_edges.jsonl`, `xsurface_X87_labels.json`, `xsurface_X88_edges.jsonl`, `xsurface_X88_labels.json`, `xsurface_X89_labels.json`, `xsurface_X90_labels.json`, `xsurface_X91_edges.jsonl`, `xsurface_X91_labels.json`, `xsurface_X92_edges.jsonl`, `xsurface_X92_labels.json`, `xsurface_X93_edges.jsonl`, `xsurface_X93_labels.json`, `xsurface_X94_edges.jsonl`, `xsurface_X94_labels.json`, `xsurface_X99_edges.jsonl`, `xsurfaces.json`

## 4. Purge

Old research steps and their outputs. Nothing that is kept, untangled or part of the page build imports or reads them (checked).

Scripts (85): `_c3.py`, `axes.py`, `axis_crossings.py`, `axis_transitions.py`, `boundaries.py`, `boundary_walls.py`, `build_atlas.py`, `cohesive_domain.py`, `cohesive_samples.py`, `combinatorics.py`, `conicfit.py`, `corner_points.py`, `cross_lines.py`, `crossing_offs.py`, `curve_trace.py`, `cut_walls.py`, `decagon_twist.py`, `decagons.py`, `dense_search.py`, `domain_pieces.py`, `domain_search.py`, `edge_fill_missing.py`, `edge_lines.py`, `exact_corners.py`, `exact_wall_fit.py`, `fill_edge.py`, `fill_from_curves.py`, `find_boundaries.py`, `fix_x98_f5.py`, `girdle_meets_axis.py`, `golden_survey.py`, `junctions.py`, `meetings_table.py`, `mirrors.py`, `mixed_lines.py`, `mixed_more.py`, `mplanes.py`, `named2.py`, `named_points.py`, `offsets.py`, `planemap.py`, `planes.py`, `references.py`, `refine_mesh.py`, `region_bounds.py`, `region_build.py`, `region_faces.py`, `region_map.py`, `region_model.py`, `region_rays.py`, `region_shells.py`, `region_view.py`, `regular_cells.py`, `ruled_wall.py`, `seam_trace.py`, `store_patch.py`, `stray_lines.py`, `stray_lines_merge.py`, `surface_points.py`, `surface_sides.py`, `symcheck.py`, `symmetry_axes.py`, `sync_copies.py`, `t2bound.py`, `t2extend.py`, `t2probe.py`, `t2surface.py`, `t2trace.py`, `tlines.py`, `uniform.py`, `uniform_in_cell.py`, `uniform_points.py`, `untwisted_decagon.py`, `voronoi_search.py`, `wall_breaks.py`, `wall_conform.py`, `wall_edges.py`, `wall_edges_rule.py`, `wall_fit.py`, `wall_outline.py`, `x32_line_end.py`, `x32_midpoint.py`, `x3_pieces.py`, `xloci.py`, `xpoint_survey.py`

Data (47): `axes.json`, `boundaries_checked.json`, `boundary_points.json`, `boundary_walls.json`, `chamber_pieces.json`, `cohesive_samples.json`, `cohesive_view.json`, `corner_points.json`, `dense_search.json`, `fingerprints.json`, `fingerprints_all.json`, `forced_lines.json`, `invisible_lines.json`, `meetings_table.json`, `mirrors.json`, `mirrors_0015.json`, `mixed_more.json`, `named_points.json`, `plane_b1b2.json`, `plane_b1b2.json.grid.json`, `plane_b1b3.json`, `plane_b1b3.json.grid.json`, `plane_b2b3.json`, `plane_b2b3.json.grid.json`, `planes_0012.json`, `region_check.json`, `region_db_view.html`, `region_model.json`, `region_rays.json`, `regular_antiprism_curves.json`, `regular_antiprism_minima.json`, `regular_prism_curves.json`, `shape_graph.json`, `surface_sides.json`, `survey_002.json`, `symmetry_axes.json`, `t2_bounds.json`, `t2_spec.json`, `t2_x17_conic_pts.json`, `t2_x21_pts.json`, `t2_x21x27_pts.json`, `t2surface.json`, `t2trace.json`, `uniform_orbits.json`, `uniform_points.json`, `wall_edges_audit.txt`, `wall_outline_checked.json`

Folders: `arr_pieces/`, `logs_tmp/`, `region_pieces/`, `sec_pieces/`, `sections/`
