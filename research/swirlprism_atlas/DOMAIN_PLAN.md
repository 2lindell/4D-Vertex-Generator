# Symmetry Domain viewer: status and plan

The page is `assets/symmetry_domain.html` (with `assets/symmetry_domain_regions/`), copied into `docs/` by
`tools/build_pages.py`; the public site builds from this branch, so a push publishes it.

## Status (2026-10-29)

Done and checked:

- Exact foundation: every surface and symmetry exact in Q(√5) (`exact.py`, `surfaces.json`); the 31 meeting points
  solved to 40 digits (`exact_points.json`); all 54 walls close on the 3-sphere from exact points and surfaces
  (`closure.py`).
- No old data anywhere in the pipeline (`PURGE.md`).
- Half-cell view mirrored across β₃ = β₄ (the half β₄ ≥ β₃), the mirror face the level top, M34 at the back
  (`domain_page.FLIP`, `flip_page.py`); its bounds stored as a box so the default camera loads; the toolbar's home
  button returns to it.
- E1 and E2 drawn only while selected; lines drawn from the exact line data.
- Page 14.4 MB (limit 16 MB). Shape Map up to date.

Not yet right:

- Wall surfaces are the old grid-grown meshes (speckled "static", corners patched with added triangles); the files to
  rebuild them (`pieces.json`, `wall_parts.json`) were lost in a container reset.
- Pieces not settled: X23 (3), X17 (1); X11's two pieces both named F2|X37 (one should be F2|X36).
- Wall outlines along other walls and the dashed cuts come from the meshes' rims.
- Region shells are built from the same meshes.
- `exact_curves.py` (work in progress): X39's edge on X23 exact; false stretches for X51, X93 and others, because
  its corners include points on images of a wall's surface off the wall.
- Working data: "X8 does not touch X27" recorded, not applied; the X84/D2 correction patched in by hand.
- Dodecahedron views not mirrored like the half-cell view; hover β values read in the half β₃ ≥ β₄.
- The cross ring along the bottom of the cell not re-checked since the rebuilds.

## Plan

1. **Exact curves for every wall.** Lines put exactly on their surfaces again (`exact_lines.py`). A wall's corners
   taken from the 3-sphere data by name: the ends of its lines where they end on another of its edges or one of its
   points, or on an edge of a wall it ends on (each such end an exact point). Its edge along another wall, and that
   wall's cut on it, traced exactly on the sphere between two of those corners. Check: every wall's loop closes with
   the traced curves.
2. **Pieces from exact loops, before any mesh.** On each wall's surface (all its placements on the 3-sphere) lay its
   exact edges, cuts and seams; the faces they enclose are its pieces. A face holding the wall's seed is the wall,
   which goes on across cuts, seams and the cell's faces, never past an edge. Each piece named from the lines round it
   and the working data's cuts (X23, X17, X11). Pieces the data cannot decide go to the user.
3. **Meshes from the pieces.** Each piece triangulated inside its exact loop (no points added on the rim), points
   put on the exact surface. Region shells and edges from these pieces.
4. **Rebuild.** Working data from the new pieces (with "X8 does not touch X27"); Shape Map; the page with the flip,
   the exact outlines and cuts, and the dodecahedron views mirrored to match; the bottom cross ring checked.
5. **Verify and publish.** Closure on the final curves, piece names against the working data, a screenshot of every
   wall in each view, the size limit; push (the site builds from this branch) and republish the previews.
