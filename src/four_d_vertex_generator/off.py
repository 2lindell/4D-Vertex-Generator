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

    Qhull only supplies candidate supporting hyperplanes; everything else is
    decided from the vertices themselves, with one scale-aware tolerance:

    - a cell is the set of vertices lying on a supporting hyperplane (within
      ``tol`` times the polytope's radius), so coplanar simplices merge by
      construction and nearly parallel neighbouring cells stay separate;
    - a face is the intersection of two cells when that intersection is a
      polygon (affine rank 2), ordered by angle around its centre.

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
    eps = tol * max(radius, 1e-300)

    # Cells: vertex sets of the supporting hyperplanes. A simplex whose vertices already lie
    # together in a known cell belongs to it (four affinely independent points fix a hyperplane).
    cells_v: list[frozenset[int]] = []
    cells_of_vertex: dict[int, set[int]] = defaultdict(set)
    for simplex, eq in zip(hull.simplices, hull.equations):
        common = set.intersection(*(cells_of_vertex[int(v)] for v in simplex))
        if common:
            continue
        normal, offset = eq[:4], eq[4]
        scale = float(np.linalg.norm(normal))
        if scale == 0:
            continue
        dist = (verts @ normal + offset) / scale
        if dist.max() > eps:
            raise ValueError(
                "Convex hull is inconsistent at this tolerance (a vertex lies outside a cell)."
            )
        on = frozenset(int(i) for i in np.flatnonzero(np.abs(dist) <= eps))
        if len(on) < 4:
            continue
        index = len(cells_v)
        cells_v.append(on)
        for v in on:
            cells_of_vertex[v].add(index)

    # Faces: pairs of cells whose shared vertices span a polygon.
    faces_list: list[list[int]] = []
    cell_faces: list[list[int]] = [[] for _ in cells_v]
    for a, cell in enumerate(cells_v):
        shared_count: dict[int, int] = defaultdict(int)
        for v in cell:
            for b in cells_of_vertex[v]:
                if b > a:
                    shared_count[b] += 1
        for b, count in sorted(shared_count.items()):
            if count < 3:
                continue
            shared = sorted(cell & cells_v[b])
            pts = verts[shared]
            mid = pts.mean(axis=0)
            _, sv, vh = np.linalg.svd(pts - mid)
            if int(np.sum(sv > eps)) != 2:
                continue  # the cells meet in an edge or a vertex, not a face
            planar = (pts - mid) @ vh[:2].T
            order = np.argsort(np.arctan2(planar[:, 1], planar[:, 0]))
            face_idx = len(faces_list)
            faces_list.append([shared[int(i)] for i in order])
            cell_faces[a].append(face_idx)
            cell_faces[b].append(face_idx)

    _check_hull(faces_list, cell_faces)
    return faces_list, cell_faces


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

