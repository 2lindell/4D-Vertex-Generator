from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _format_number(value: float) -> str:
    text = format(float(value), ".17g")
    if "e" in text.lower():
        text = format(Decimal(text), "f")
    return text


def compute_convex_hull(
    vertices: np.ndarray,
    *,
    tol: float = 1e-9,
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute the 4D convex hull of a set of 4D vertices.

    Built on Qhull's triangulated hull and its simplex adjacency, with one
    scale-aware tolerance:

    - a cell is a connected group of neighbouring simplices whose hyperplanes
      agree within ``tol`` (relative to the polytope's radius); its vertices
      are those simplices' vertices, so a vertex merely close to a cell's
      hyperplane is never pulled into it, and nearly parallel neighbouring
      cells stay separate;
    - two cells share a face when Qhull's triangulation has neighbouring
      simplices in both; the face is their common vertices, ordered by angle
      around its centre (no width threshold, so sliver faces are kept).

    The result is checked (every face polygon on exactly two cells, every
    cell a closed polyhedron, Euler characteristic 0) and a ValueError is
    raised if it does not close up, rather than returning a broken hull.

    Returns:
        (faces, cells):
        - faces: A list of 2D faces, where each face is a list of 0-based vertex
          indices ordered cyclically around the 2D polygon perimeter.
        - cells: A list of 3D cells, where each cell is a list of 0-based face
          indices that bound the 3D cell.
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 4:
        raise ValueError(f"Expected vertices shape (n,4), got {verts.shape}")
    if len(verts) < 5:
        raise ValueError(
            f"At least 5 vertices are required to compute a 4D convex hull, got {len(verts)}"
        )

    try:
        hull = ConvexHull(verts)
    except QhullError as err:
        raise ValueError(
            "Vertices do not span 4D space or are degenerate; cannot compute 4D convex hull."
        ) from err

    centre = verts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(verts - centre, axis=1)))

    # Cells: groups of neighbouring Qhull simplices lying in one hyperplane. A group grows from one
    # simplex and admits a neighbour only if it matches that founding simplex's hyperplane, so small
    # differences cannot add up along a chain. Membership comes from the simplices themselves, never
    # from a distance test against every vertex, so a vertex that is merely very close to a cell's
    # hyperplane (as near-duplicate vertices of nearly degenerate seeds are) is not pulled into it.
    eqs = hull.equations
    norms = np.linalg.norm(eqs[:, :4], axis=1)
    if np.any(norms == 0):
        raise ValueError("Convex hull is inconsistent (a degenerate facet has no normal).")
    eqs = eqs / norms[:, None]
    n_simplices = len(eqs)
    eps = tol * max(radius, 1e-300)

    eq_rows = eqs.tolist()
    neighbor_rows = hull.neighbors.tolist()

    def same_plane(i: int, j: int) -> bool:
        a, b = eq_rows[i], eq_rows[j]
        return (
            abs(a[0] - b[0]) <= tol
            and abs(a[1] - b[1]) <= tol
            and abs(a[2] - b[2]) <= tol
            and abs(a[3] - b[3]) <= tol
            and abs(a[4] - b[4]) <= eps
        )

    parent = [-1] * n_simplices
    for start in range(n_simplices):
        if parent[start] != -1:
            continue
        parent[start] = start
        queue = [start]
        while queue:
            current = queue.pop()
            for n_idx in neighbor_rows[current]:
                if parent[n_idx] == -1 and same_plane(start, n_idx):
                    parent[n_idx] = start
                    queue.append(n_idx)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # A flat sliver simplex (Qhull's triangulation of a nearly degenerate facet) can carry an
    # inaccurate hyperplane and so fail the test above. A real cell never has all its vertices in
    # another cell, so fold any group whose vertices all lie in a neighbouring group into it.
    simplex_rows = hull.simplices.tolist()
    groups: dict[int, set[int]] = defaultdict(set)
    for s_idx, simplex in enumerate(simplex_rows):
        groups[find(s_idx)].update(simplex)

    changed = True
    while changed:
        changed = False
        roots = np.array([find(i) for i in range(n_simplices)])
        nb_roots = roots[hull.neighbors]
        differing = nb_roots != roots[:, None]
        pairs_arr = np.unique(
            np.stack([np.broadcast_to(roots[:, None], nb_roots.shape)[differing], nb_roots[differing]], axis=1),
            axis=0,
        )
        for a, b in pairs_arr.tolist():
            a, b = find(a), find(b)
            if a != b and (groups[a] <= groups[b] or groups[b] <= groups[a]):
                small, big = (a, b) if len(groups[a]) <= len(groups[b]) else (b, a)
                parent[small] = big
                groups[big] |= groups.pop(small)
                changed = True

    root_to_cell: dict[int, int] = {}
    simplex_cell = np.array(
        [root_to_cell.setdefault(find(i), len(root_to_cell)) for i in range(n_simplices)]
    )
    cell_sets: list[set[int]] = [set() for _ in root_to_cell]
    for cell_idx, simplex in zip(simplex_cell.tolist(), simplex_rows):
        cell_sets[cell_idx].update(simplex)
    cells_v = [frozenset(c) for c in cell_sets]
    _check_cells(verts, cells_v, eps)

    # Faces: two cells share a face exactly when Qhull's triangulation has neighbouring simplices
    # in both (they meet across a triangle of that face). This needs no width threshold, so
    # sliver faces of nearly degenerate seeds are kept.
    nb_cells = simplex_cell[hull.neighbors]
    differing = nb_cells != simplex_cell[:, None]
    ca = np.broadcast_to(simplex_cell[:, None], nb_cells.shape)[differing]
    cb = nb_cells[differing]
    pair_arr = np.unique(np.stack([np.minimum(ca, cb), np.maximum(ca, cb)], axis=1), axis=0)
    pairs = [(int(x), int(y)) for x, y in pair_arr]

    faces_list: list[list[int]] = []
    cell_faces: list[list[int]] = [[] for _ in cells_v]
    for a, b in pairs:
        shared = sorted(cells_v[a] & cells_v[b])
        if len(shared) < 3:
            raise ValueError(
                "Convex hull is inconsistent at this tolerance (a face has fewer than 3 vertices)."
            )
        pts = verts[shared]
        mid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - mid)
        planar = (pts - mid) @ vh[:2].T
        order = np.argsort(np.arctan2(planar[:, 1], planar[:, 0]))
        face_idx = len(faces_list)
        faces_list.append([shared[int(i)] for i in order])
        cell_faces[a].append(face_idx)
        cell_faces[b].append(face_idx)

    _check_hull(faces_list, cell_faces)
    return faces_list, cell_faces


def _check_cells(verts: np.ndarray, cells_v: list[frozenset[int]], eps: float) -> None:
    """Raise ValueError unless every cell is flat and supports the hull unambiguously.

    Each cell's vertices must lie within ``eps`` of their best-fit hyperplane and no vertex may lie
    outside it. Any other vertex within float noise of the hyperplane (``eps * 1e-3``) must
    coincide with one of the cell's own vertices (a duplicate point): Qhull resolves vertices
    that are farther in, however thin the polytope. Otherwise the input is too close to
    degenerate to resolve, and refusing is better than returning a hull that is subtly wrong.
    """
    bad = "Convex hull is ambiguous at this tolerance"
    for cell in cells_v:
        members = np.fromiter(cell, dtype=int)
        pts = verts[members]
        mid = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - mid)
        normal = vt[3]
        dist = (verts - mid) @ normal
        if dist[members].max() > eps or dist[members].min() < -eps:
            raise ValueError(f"{bad} (a cell is not flat).")
        if dist.max() > eps and dist.min() < -eps:
            raise ValueError(f"{bad} (vertices lie on both sides of a cell).")
        if dist.max() > eps:
            dist = -dist
        near = np.flatnonzero(dist >= -eps * 1e-3)
        others = np.setdiff1d(near, members)
        if len(others):
            gaps = np.min(np.linalg.norm(verts[others][:, None] - pts[None], axis=2), axis=1)
            if gaps.max() > eps:
                raise ValueError(f"{bad} (a vertex lies on a cell it is not part of).")


def _check_hull(faces: list[list[int]], cells: list[list[int]]) -> None:
    """Raise ValueError unless the faces and cells form a closed 4D polytope boundary."""
    def edges_of(face: list[int]) -> list[tuple[int, int]]:
        return [(min(u, w), max(u, w)) for u, w in zip(face, face[1:] + face[:1])]

    bad = "Convex hull is inconsistent at this tolerance"
    all_edges: set[tuple[int, int]] = set()
    for c, cell in enumerate(cells):
        edge_uses: dict[tuple[int, int], int] = defaultdict(int)
        cell_vertices: set[int] = set()
        for f in cell:
            for e in edges_of(faces[f]):
                edge_uses[e] += 1
            cell_vertices.update(faces[f])
        if len(cell) < 4 or any(k != 2 for k in edge_uses.values()):
            raise ValueError(f"{bad} (cell {c} is not closed).")
        if len(cell_vertices) - len(edge_uses) + len(cell) != 2:
            raise ValueError(f"{bad} (cell {c} is not a polyhedron).")
        all_edges.update(edge_uses)
    # points strictly inside the hull are allowed; they are simply not on any face
    used = {v for face in faces for v in face}
    if len(used) - len(all_edges) + len(faces) - len(cells) != 0:
        raise ValueError(f"{bad} (Euler characteristic is not 0).")


def parse_4off(text: str) -> np.ndarray:
    """Parse 4D OFF-style text (as written by `to_4off`) into a vertices array.

    Only vertex coordinates are extracted; face and cell lines, if present,
    are skipped using the counts on the header's counts line.
    """
    lines = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("Empty OFF content")

    header = lines[0]
    if not header.upper().endswith("OFF"):
        raise ValueError(f"Unrecognized OFF header: '{header}'")

    if len(lines) < 2:
        raise ValueError("Missing vertex/face/cell counts line")
    try:
        num_vertices = int(lines[1].split()[0])
    except ValueError as err:
        raise ValueError(f"Invalid counts line: '{lines[1]}'") from err

    vertex_lines = lines[2 : 2 + num_vertices]
    if len(vertex_lines) != num_vertices:
        raise ValueError(f"Expected {num_vertices} vertex lines, found {len(vertex_lines)}")

    vertices: list[list[float]] = []
    for line in vertex_lines:
        tokens = line.split()
        if len(tokens) < 4:
            raise ValueError(f"Expected 4 coordinates per vertex, got: '{line}'")
        vertices.append([float(t) for t in tokens[:4]])

    return np.asarray(vertices, dtype=float)


def to_4off(
    vertices: np.ndarray,
    faces: list[list[int]] | None = None,
    cells: list[list[int]] | None = None,
    *,
    compute_hull: bool = False,
) -> str:
    """Serialize vertices (and optional faces and cells) to a 4D OFF-style text format.

    Format:
      4OFF
      <num_vertices> <num_faces> 0 <num_cells>
      x y z w
      ...
      <nv> v0 v1 ...
      ...
      <nf> f0 f1 ...
      ...
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 4:
        raise ValueError(f"Expected vertices shape (n,4), got {verts.shape}")

    if compute_hull and (faces is None or cells is None):
        hull_faces, hull_cells = compute_convex_hull(verts)
        if faces is None:
            faces = hull_faces
        if cells is None:
            cells = hull_cells

    f_list = faces if faces is not None else []
    c_list = cells if cells is not None else []

    lines = ["4OFF", f"{len(verts)} {len(f_list)} 0 {len(c_list)}"]
    for v in verts:
        lines.append(" ".join(_format_number(value) for value in v))

    for f in f_list:
        lines.append(f"{len(f)} " + " ".join(str(i) for i in f))

    for c in c_list:
        lines.append(f"{len(c)} " + " ".join(str(f_idx) for f_idx in c))

    return "\n".join(lines) + "\n"

