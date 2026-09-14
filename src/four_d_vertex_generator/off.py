from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree


def _format_number(value: float) -> str:
    text = format(float(value), ".17g")
    if "e" in text.lower():
        text = format(Decimal(text), "f")
    return text


def compute_convex_hull(
    vertices: np.ndarray,
    *,
    tol: float = 1e-5,
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute the 4D convex hull of a set of 4D vertices.

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

    eqs = hull.equations
    norms = np.linalg.norm(eqs[:, :4], axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_eqs = eqs / norms

    n_simplices = len(norm_eqs)
    tree = cKDTree(norm_eqs)
    pairs = list(tree.query_pairs(r=tol))

    if pairs:
        rows, cols = zip(*pairs)
        data = np.ones(len(rows), dtype=bool)
        adj = csr_matrix((data, (rows, cols)), shape=(n_simplices, n_simplices))
        adj = adj + adj.T
    else:
        adj = csr_matrix((n_simplices, n_simplices), dtype=bool)

    n_cells, labels = connected_components(adj, directed=False)

    unique_cells_verts: list[set[int]] = [set() for _ in range(n_cells)]
    for cell_idx, simplex in zip(labels, hull.simplices):
        unique_cells_verts[cell_idx].update(simplex)

    triangle_cell_counts: dict[tuple[int, int, int], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for cell_idx, simplex in zip(labels, hull.simplices):
        v = sorted(int(x) for x in simplex)
        triangles = [
            (v[0], v[1], v[2]),
            (v[0], v[1], v[3]),
            (v[0], v[2], v[3]),
            (v[1], v[2], v[3]),
        ]
        for tri in triangles:
            triangle_cell_counts[tri][cell_idx] += 1

    cell_pair_triangles: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for tri, cell_dict in triangle_cell_counts.items():
        boundary_cells = [cell for cell, count in cell_dict.items() if count % 2 == 1]
        if len(boundary_cells) == 2:
            cell_pair = (min(boundary_cells), max(boundary_cells))
            cell_pair_triangles[cell_pair].append(tri)

    faces_list: list[list[int]] = []
    cell_faces: list[list[int]] = [[] for _ in range(len(unique_cells_verts))]

    for (c1, c2), _tris in cell_pair_triangles.items():
        shared_verts = sorted(unique_cells_verts[c1].intersection(unique_cells_verts[c2]))
        if len(shared_verts) >= 3:
            pts = verts[shared_verts]
            center = pts.mean(axis=0)
            q = pts - center
            _, s, vh = np.linalg.svd(q)
            rank = int(np.sum(s > 1e-4))
            if rank < 2:
                # Shared vertices are collinear or degenerate; not a 2D face
                continue

            if len(pts) > 3:
                u_vec, v_vec = vh[0], vh[1]
                pts_2d = np.column_stack((q @ u_vec, q @ v_vec))
                hull_2d = ConvexHull(pts_2d)
                ordered_indices = [shared_verts[int(i)] for i in hull_2d.vertices]
            else:
                ordered_indices = shared_verts

            face_idx = len(faces_list)
            faces_list.append(ordered_indices)
            cell_faces[c1].append(face_idx)
            cell_faces[c2].append(face_idx)

    return faces_list, cell_faces


def parse_4off(text: str) -> np.ndarray:
    """Parse 4D OFF-style text (as written by `to_4off`) into a vertices array.

    Only vertex coordinates are extracted; face and cell lines, if present,
    are skipped using the counts on the header's counts line.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        raise ValueError("Empty OFF content")

    header = lines[0]
    if not header.upper().endswith("OFF"):
        raise ValueError(f"Unrecognized OFF header: '{header}'")

    counts_tokens = lines[1].split()
    if not counts_tokens:
        raise ValueError("Missing vertex/face/cell counts line")
    num_vertices = int(counts_tokens[0])

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
      <num_vertices> <num_faces> <num_cells>
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

