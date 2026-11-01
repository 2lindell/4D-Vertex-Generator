# Cleanup: start over on the viewer geometry

Everything that turned geometry into meshes or drawings for the viewer was removed, so that step can be started again
from the exact data. What is left is the working data, the exact mathematical data it rests on, and the pages
themselves (unchanged).

## Kept

The source of truth and the working data:

- `working_base.json` — the verified base facts, every correction applied with its reason. Edited directly.
- `working_data.json` — the base and everything computed from it (`working_data.py`). Rebuilt after the cleanup: the
  same file, byte for byte; then rebuilt again without the ring test (below), which removed only what the ring test had
  given it.
- `exact_defs.json` — the exact line pieces and each wall's surfaces; `surfaces.json` — the exact equations of every
  surface a wall lies on.

Exact data (no meshes, no sampling to label):

- `exact_points.json` (`exact_points.py`, with `exact.py`: Q(√5) arithmetic, 40 digits) — the named points.
- `exact_lines.json` (`exact_lines.py`) — the exact line pieces.
- `line_ends.json` (`line_ends.py`) — each line's exact ends.
- `wall_equations.json` (`edges_from_equations.py`) — wall edges and cuts from the equations.
- `cross_ring_points.json` (`cross_ring_points.py`) — the exact points on the cross ring.
- `exact_curves.json` (`exact_curves.py`) — each wall's edge along another wall, traced exactly from corner to corner
  (Newton onto both surfaces). Not yet checked wall by wall.
- `closure.json` (`closure.py`) — the check that every wall's edges close, from the exact data on the whole 3-sphere.
- `regular_lines.json` (`regular_cells.py`), `regular_decagons.json` — the regular cells' lines and decagons.
- `sphere_complex.json` — frozen: read by `working_data.py` (its script built it from the half-cell meshes and is gone).

The core and small math modules: `core.py`, `exact_math.py` (helpers the exact scripts share), `cellframe.py`,
`classify.py`, `classify_fast.py`, `catalog.py`, `dirichlet.py`, `normalizer.py`, `supergroups.py`; tables
`references.json`, `atlas_xids.json`, `golden_seeds.json`, `cell_vertices.npy`, `normalizer_q.npy`.

Pages and their data, as they are: `shape_map.html` (`build_shape_map.py`, from the working data), `shape_map_src.html`,
`symmetry_domain_lines.html`, `region_db_view.html`, the templates (`cell_atlas2_template.html`,
`cell_atlas_template.html`, `atlas_template.html`), `examples/`, and outside this folder `assets/symmetry_domain.html`,
`assets/symmetry_domain_regions/` and `docs/`.

## Removed

- The viewer builders: `domain_page.py`, `exact_page.py`, `lines_page.py`, `dodeca_view.py`, `chamber_domain.py`,
  `overlays.py`, `place_colors.py`, `flip_page.py`, `decagons.py`.
- The face and mesh builders: `wall_faces.py`, `face_meshes.py`, `arrangement.py`, `arrangement_draw.py`, `pieces.py`,
  `wall_parts.py`, `sphere_complex.py`, `fill_corners.py`, `closure_curves.py`, and their outputs
  (`wall_faces_report.json`, `arrangement*.json`, `arr_*`, `pieces.json`, `closure_curves.json`).
- The ring test (`ring_test.py`, `ring_test.json`): it labelled sampled points on small circles round each line, and was
  not known to be accurate. Taken out of the working data with it: each region's lines and points, each wall's
  `pieces` (the regions each piece lies between, with its lines), and the ring checks under `pieces_to_check`. Each
  wall's edges, sides, cuts and corners, and each region's walls, are unchanged. The Shape Map was rebuilt from the
  working data: a region now lists its walls only.
- Everything the earlier purge list named (the old atlas, region, section and survey tools and their data, 339
  entries).

All of it stays in the git history.
