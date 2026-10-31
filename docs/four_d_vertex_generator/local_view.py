"""Local 3D view of the neighbourhood of a single vertex of a 4D polytope.

Every vertex of a polytope inscribed in a 3-sphere has a 3D tangent space:
the hyperplane through the vertex orthogonal to its radius. Projecting the
vertex's edges into that space gives an undistorted-in-direction, easy to
read 3D picture of how the polytope looks locally -- effectively its vertex
figure, with lines going out along the edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

Edge = tuple[int, int]


def hull_edges(faces: list[list[int]]) -> set[Edge]:
    """Return the edges of a polytope from its cyclically ordered 2D faces."""
    edges: set[Edge] = set()
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            edges.add((min(a, b), max(a, b)))
    return edges


def nearest_neighbour_edges(vertices: np.ndarray, rel_tol: float = 1e-6) -> set[Edge]:
    """Connect every pair of vertices at the shortest vertex-to-vertex distance.

    This recovers all edges of a polytope whose edges all have the same length
    (every regular and many uniform polytopes). Polytopes with several edge
    lengths need `hull_edges` instead.
    """
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 2:
        return set()
    tree = cKDTree(verts)
    distances, _ = tree.query(verts, k=2)
    shortest = float(distances[:, 1].min())
    if shortest == 0.0:
        raise ValueError("Vertex set contains duplicate points")
    pairs = tree.query_pairs(shortest * (1.0 + rel_tol))
    return {(min(a, b), max(a, b)) for a, b in pairs}


def tangent_basis(direction: np.ndarray) -> np.ndarray:
    """Return a (3, 4) orthonormal basis of the hyperplane orthogonal to `direction`."""
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        raise ValueError("Cannot build a tangent space for a vertex at the centre")
    _, _, vh = np.linalg.svd((d / norm).reshape(1, 4))
    return vh[1:]


@dataclass(frozen=True)
class LocalView:
    """The neighbourhood of one vertex, projected into its 3D tangent space.

    Attributes:
        center: Index of the vertex being viewed.
        neighbours: Indices of the vertices joined to `center` by an edge.
        positions: (k, 3) tangent-space position of each neighbour, relative to
            the centre vertex at the origin.
        edge_lengths: True 4D length of each edge from the centre.
        figure_edges: Pairs of positions into `neighbours` that share a face
            with the centre; together they outline the vertex figure.
    """

    center: int
    neighbours: np.ndarray
    positions: np.ndarray
    edge_lengths: np.ndarray
    figure_edges: list[Edge]


def local_view(
    vertices: np.ndarray,
    center: int,
    edges: set[Edge],
    faces: list[list[int]] | None = None,
) -> LocalView:
    """Project the edges around vertex `center` into its tangent space.

    The radial direction is measured from the vertex set's centroid, so the
    view also works for uploaded polytopes that are not centred at the origin.
    """
    verts = np.asarray(vertices, dtype=float)
    if not 0 <= center < len(verts):
        raise IndexError(f"Vertex index {center} out of range for {len(verts)} vertices")

    neighbours = np.array(
        sorted(b if a == center else a for a, b in edges if center in (a, b)), dtype=int
    )
    origin = verts[center]
    basis = tangent_basis(origin - verts.mean(axis=0))
    offsets = verts[neighbours] - origin if len(neighbours) else np.zeros((0, 4))

    figure_edges: set[Edge] = set()
    if faces is not None:
        slot = {int(n): i for i, n in enumerate(neighbours)}
        for face in faces:
            if center not in face:
                continue
            k = face.index(center)
            before, after = face[k - 1], face[(k + 1) % len(face)]
            if before in slot and after in slot and before != after:
                i, j = slot[before], slot[after]
                figure_edges.add((min(i, j), max(i, j)))

    return LocalView(
        center=center,
        neighbours=neighbours,
        positions=offsets @ basis.T,
        edge_lengths=np.linalg.norm(offsets, axis=1),
        figure_edges=sorted(figure_edges),
    )


def edge_length_classes(lengths: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Label each length with the index of its distinct value, shortest first."""
    lengths = np.asarray(lengths, dtype=float)
    distinct: list[float] = []
    for value in np.sort(lengths):
        if not distinct or value > distinct[-1] * (1.0 + rel_tol):
            distinct.append(float(value))
    return np.array(
        [next(i for i, d in enumerate(distinct) if abs(v - d) <= d * rel_tol) for v in lengths],
        dtype=int,
    )
