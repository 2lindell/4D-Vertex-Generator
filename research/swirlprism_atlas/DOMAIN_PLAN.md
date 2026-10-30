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

## Progress

- Step 1 (2026-10-29), largely done. Corners named from the 3-sphere data: the ends of a wall's lines where they end
  on the other wall or one of its edges or points (`exact_points.json` records what each end lies on). Copies of the
  other wall taken first as they lie in the half-cell, then any image. The curves are traced exactly on the sphere
  between corners, about 6 s per pair. The page draws them (`exact_page.py`) for 14 walls' edges along other walls and
  the matching cuts: X39 on X23 runs A1 to the X5 point, end to end. Where several exact curves join the same two
  corners (A1 and C1), the page keeps only the pieces on both walls' meshes as drawn; step 2 replaces that check.
- Corners given by the user (working_base.json corrections, `edge_corners`): X28 along F1 from A1 to C5 (drawn
  exactly); X11 along T2 from C5 to X4 (corrected by the user from C4; three copies, drawn exactly). Placed X94 crosses
  that edge partway: the point X94·X11·T2, solved exactly on the placed X94, X11 and T2 (β ≈ (0.46695, 0.27492,
  0.12906, 0.12906) on the mirror, and its copy (⅓, 0.23283, ⅓, 0.10051)); it replaces the point solved before, which
  lay only on copies of X11 and X94 elsewhere. X94 along T2 runs from that point to C5, which it reaches outside the
  half-cell (drawn exactly). X94 along X11 runs from C1 to where it
  meets T2's plane, that same point (user; drawn exactly). With both its edges exact, X94's mesh outlines (which ran
  past its corners) are no longer drawn. These given edges are drawn without the mesh
  check.
- Lines put exactly on their surfaces again (`exact_lines.py`, `exact_lines.json`): each piece moved onto the two of
  its surfaces crossing most cleanly, ends on the exact points or solved onto the cell's faces. The largest move was
  1.7e-6; E1's β₄ is now 9e-20.
- Closure with the traced curves (`closure_curves.py`): every wall's outline in the half-cell, from its lines and its
  traced edges along other walls (every image lying on its own surface), has no loose end; 54/54 walls close.
  `exact_curves.py` now reads the exact lines (its corners are the exact points, so the traced curves are unchanged).
  Step 1 done.

## Approach change (2026-10-30)

Tracing each edge between corners named one by one, and adding corners by hand where that failed, is dropped: it
needed the user to say which way an edge runs, which the walls' positions on the 3-sphere already decide, and it left
edges missing or running past their corners. Instead, every wall's extent comes out of one computation over the whole
3-sphere, the same for every wall and every copy, and is checked by the symmetry:

1. **The arrangement on each surface.** On each wall's surface, every copy (image under the 4800 elements) of every
   surface of the walls it meets (its edges and cuts in the working data), and every copy of its lines, is laid down
   in full where it crosses the surface. Each curve is solved exactly onto both surfaces; nothing is named by hand.
2. **Which copies are real.** A curve stretch on wall W from a copy of wall Y bounds W only where Y itself is
   there (that stretch lies on Y's own faces, carried back by the group). All walls are solved together until nothing
   changes: each wall is the faces holding its seed and those reached across stretches that are not real edges.
3. **Checks from the symmetry.** Each wall's faces map onto themselves under its stabiliser; W's edge along Y is the
   same curve as Y's edge or cut on W; every corner found is an exact point, and the points the working data lists
   for W are its corners; every outline closes. Where a check fails, the data is reported, not patched.
4. **The viewer** draws from this alone: edges where the wall is on one side, cuts where it is on both, nothing where
   it is on neither. The working data is not changed by the drawing step.

The hand-given corners (`edge_corners`) are kept only as checks to compare against.

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
