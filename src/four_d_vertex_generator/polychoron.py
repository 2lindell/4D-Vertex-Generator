"""Build a closed polychoron from a vertex orbit and a chosen set of edge orbits.

The convex hull (see ``off.py``) only ever uses the hull's own edges. Here the
edges are chosen by symmetry instead: every pair of orbit vertices falls into an
edge orbit (pairs the group maps onto each other), and the polychoron is built
from the union of the selected orbits:

1. faces: planar cycles of the edge graph, traced by always turning as sharply
   as possible to one side (the tightest polygon on that side of an edge);
2. cells: groups of faces lying in one hyperplane that close up into a
   polyhedron (every edge shared by exactly two faces of the group).

Only the faces through vertex 0 are traced; the group carries them to the other
vertices. This version produces simple (non-self-intersecting) polygons, which
is enough for the convex case it is checked against.
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .generation import group_elements
from .off import compute_convex_hull
from .symmetry import SymmetryAction


@dataclass(frozen=True)
class EdgeOrbit:
    """Vertex pairs that the symmetry group maps onto one another."""

    length: float
    edges: tuple[tuple[int, int], ...]


@dataclass
class Polychoron:
    vertices: np.ndarray
    edges: list[tuple[int, int]]
    faces: list[list[int]]
    cells: list[list[int]]
    problems: list[str] = field(default_factory=list)
    cell_choices: int = 1
    # how it was built, so a result can be rebuilt or described
    orbit_indices: tuple[int, ...] = ()
    face_corners: tuple[int, ...] | None = None
    cell_choice: int = 0
    cell_sets: list[list[list[int]]] = field(default_factory=list, repr=False)

    @property
    def euler(self) -> int:
        return len(self.vertices) - len(self.edges) + len(self.faces) - len(self.cells)

    @property
    def is_closed(self) -> bool:
        return not self.problems and bool(self.cells)


def _scale(vertices: np.ndarray) -> float:
    return max(float(np.max(np.linalg.norm(vertices, axis=1))), 1e-12)


def _coset_representatives(vertices: np.ndarray, elements: tuple[np.ndarray, ...], tol: float):
    """One group element per vertex, mapping vertex 0 onto it."""
    tree = cKDTree(vertices)
    images = np.stack(elements) @ vertices[0]
    dist, idx = tree.query(images)
    if dist.max() > tol:
        raise ValueError("The group does not map the vertices onto themselves.")
    reps: dict[int, np.ndarray] = {}
    for element, j in zip(elements, idx.tolist()):
        reps.setdefault(j, element)
    if len(reps) != len(vertices):
        raise ValueError("The vertices are not a single orbit of the group.")
    return reps, tree


class _Geom:
    """Per-orbit data shared by every build: the coset representatives and the vertex lookup."""

    def __init__(self, verts: np.ndarray, action: SymmetryAction, atol: float) -> None:
        elements = group_elements(action)
        if elements is None:
            raise ValueError("The symmetry group is too large to enumerate.")
        self.reps, self.tree = _coset_representatives(verts, elements, atol)
        self.verts = verts
        self._back: dict[int, np.ndarray] = {}

    def back(self, vertex: int) -> np.ndarray:
        """Vertex index map of the group element that carries ``vertex`` to vertex 0."""
        if vertex not in self._back:
            self._back[vertex] = self.tree.query(self.verts @ self.reps[vertex])[1]
        return self._back[vertex]


def edge_orbits(
    vertices: np.ndarray, action: SymmetryAction, *, tol: float = 1e-7
) -> list[EdgeOrbit]:
    """Split every vertex pair into orbits under the group, shortest first.

    ``vertices`` must be the orbit of one point under ``action``.
    """
    verts = np.asarray(vertices, dtype=float)
    elements = group_elements(action)
    if elements is None:
        raise ValueError("The symmetry group is too large to enumerate.")
    scale = _scale(verts)
    atol = tol * scale
    _, tree = _coset_representatives(verts, elements, atol)
    mats = np.stack(elements)
    anchor = tree.query(mats @ verts[0])[1]

    dists = np.linalg.norm(verts - verts[0], axis=1)
    order = np.argsort(dists, kind="stable")
    covered: set[int] = {0}
    orbits: list[EdgeOrbit] = []
    for j in order.tolist():
        if j in covered:
            continue
        b = tree.query(mats @ verts[j])[1]
        pairs = {(min(x, y), max(x, y)) for x, y in zip(anchor.tolist(), b.tolist())}
        for x, y in pairs:
            if x == 0:
                covered.add(y)
            elif y == 0:
                covered.add(x)
        orbits.append(EdgeOrbit(float(dists[j]), tuple(sorted(pairs))))
    orbits.sort(key=lambda o: o.length)
    return orbits


def _canonical_cycle(cycle: list[int]) -> tuple[int, ...]:
    n = len(cycle)
    start = cycle.index(min(cycle))
    forward = cycle[start:] + cycle[:start]
    backward = [forward[0]] + forward[:0:-1]
    return tuple(min(forward, backward))


def _trace_face(
    verts: np.ndarray,
    nbrs: list[list[int]],
    u: int,
    v: int,
    w0: int,
    atol: float,
    max_len: int,
    corner: Callable[[int, int, int], bool],
) -> list[int] | None:
    """Walk the tightest polygon on the w0 side of the directed edge u->v.

    ``corner(prev, cur, w)`` says whether prev-cur-w is a corner of a face at ``cur``.
    """
    e1 = verts[v] - verts[u]
    e1 /= np.linalg.norm(e1)
    d = verts[w0] - verts[v]
    d = d - (d @ e1) * e1
    if np.linalg.norm(d) <= atol:
        return None
    e2 = d / np.linalg.norm(d)

    def planar(p: int, origin: int) -> tuple[float, float] | None:
        r = verts[p] - verts[origin]
        x, y = float(r @ e1), float(r @ e2)
        if np.linalg.norm(r - x * e1 - y * e2) > atol:
            return None
        return x, y

    cycle = [u, v]
    prev, cur = u, v
    while len(cycle) <= max_len:
        dx, dy = (verts[cur] - verts[prev]) @ e1, (verts[cur] - verts[prev]) @ e2
        heading = np.arctan2(dy, dx)
        best, best_turn = None, -np.inf
        for w in nbrs[cur]:
            if w == prev or not corner(prev, cur, w):
                continue
            xy = planar(w, cur)
            if xy is None:
                continue
            turn = (np.arctan2(xy[1], xy[0]) - heading + np.pi) % (2 * np.pi) - np.pi
            if turn > best_turn:
                best, best_turn = w, turn
        if best is None or best_turn <= 1e-9:
            return None
        if best == cycle[0]:
            return cycle if len(cycle) >= 3 else None
        if best in cycle:
            return None
        cycle.append(best)
        prev, cur = cur, best
    return None


def _vertex_figure_pairs(
    verts: np.ndarray, neighbours: list[int], centre: np.ndarray, atol: float
) -> set[frozenset[int]]:
    """Pairs of neighbours of vertex 0 that are adjacent in its (convex) vertex figure."""
    axis = centre - verts[0]
    axis /= np.linalg.norm(axis)
    d = verts[neighbours] - verts[0]
    height = d @ axis
    if np.any(height <= atol):
        raise ValueError("Vertex 0 is not a convex corner of the chosen edges.")
    flat = d / height[:, None]
    basis = np.linalg.svd(flat - flat.mean(axis=0))[2][:3]
    try:
        hull = ConvexHull(flat @ basis.T)
    except QhullError as err:
        raise ValueError("The edges at a vertex do not span a 3D vertex figure.") from err
    eqs = hull.equations
    pairs: set[frozenset[int]] = set()
    for i, simplex in enumerate(hull.simplices.tolist()):
        for k, j in enumerate(hull.neighbors[i].tolist()):
            if np.allclose(eqs[i], eqs[j], atol=1e-7):
                continue
            a, b = (x for m, x in enumerate(simplex) if m != k)
            pairs.add(frozenset((neighbours[a], neighbours[b])))
    return pairs


def _faces_at_vertex_zero(
    verts: np.ndarray,
    nbrs: list[list[int]],
    atol: float,
    corner: Callable[[int, int, int], bool],
    figure: set[frozenset[int]],
) -> set[tuple[int, ...]]:
    found: set[tuple[int, ...]] = set()
    for pair in figure:
        for v, w0 in (tuple(pair), tuple(pair)[::-1]):
            cycle = _trace_face(verts, nbrs, 0, v, w0, atol, len(verts), corner)
            if cycle is not None:
                found.add(_canonical_cycle(cycle))
    return found


def _hyperplane_key(
    points: np.ndarray, atol: float, all_points: np.ndarray | None = None
) -> tuple | None:
    """Canonical key of the hyperplane through ``points``, or None if they do not span one.

    With ``all_points`` given, the hyperplane must also support them (none strictly on both sides).
    """
    centre = points.mean(axis=0)
    _, sv, vt = np.linalg.svd(points - centre)
    if sv[2] <= atol * 10:
        return None
    normal = vt[3]
    if np.linalg.norm((points - centre) @ normal) > atol * 10 * np.sqrt(len(points)):
        return None
    if all_points is not None:
        side = (all_points - centre) @ normal
        if side.max() > atol * 10 and side.min() < -atol * 10:
            return None
    first = normal[np.argmax(np.abs(normal) > 1e-6)]
    if first < 0:
        normal = -normal
    offset = float(normal @ centre)
    return tuple(np.round(np.append(normal, offset), 5) + 0.0)


def _build_cells(
    verts: np.ndarray, faces: list[list[int]], atol: float, convex: bool
) -> tuple[list[list[int]], list[str]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge_faces[(min(a, b), max(a, b))].append(i)

    in_plane: dict[tuple, set[int]] = defaultdict(set)
    for owners in edge_faces.values():
        for f, g in itertools.combinations(owners, 2):
            pts = verts[sorted(set(faces[f]) | set(faces[g]))]
            key = _hyperplane_key(pts, atol, verts if convex else None)
            if key is not None:
                in_plane[key].update((f, g))

    cells: dict[frozenset[int], list[int]] = {}
    problems: list[str] = []
    for members in in_plane.values():
        uses: dict[tuple[int, int], list[int]] = defaultdict(list)
        for f in members:
            for a, b in zip(faces[f], faces[f][1:] + faces[f][:1]):
                uses[(min(a, b), max(a, b))].append(f)
        # connected components through edges used by exactly two faces
        parent = {f: f for f in members}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for owners in uses.values():
            if len(owners) == 2:
                parent[find(owners[0])] = find(owners[1])
        groups: dict[int, list[int]] = defaultdict(list)
        for f in members:
            groups[find(f)].append(f)
        for group in groups.values():
            gset = set(group)
            closed = all(
                sum(1 for f in owners if f in gset) == 2
                for edge, owners in uses.items()
                if gset & set(owners)
            )
            if closed and len(group) >= 4:
                cells[frozenset(group)] = sorted(group)
    return list(cells.values()), problems


@dataclass(frozen=True)
class Corner:
    """Pairs of edges at vertex 0 that the vertex's stabilizer maps onto one another."""

    angle: float
    pairs: tuple[tuple[int, int], ...]


def _edge_neighbours(edges: list[tuple[int, int]], n: int) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def corner_classes(
    vertices: np.ndarray,
    action: SymmetryAction,
    edges: list[tuple[int, int]],
    *,
    tol: float = 1e-7,
) -> list[Corner]:
    """Classes of edge pairs at vertex 0 (of any lengths) that the vertex stabilizer maps onto
    one another, smallest angle first.

    Each class is a candidate face corner: ``build_polychoron(face_corners=[i, ...])`` takes
    every simple planar cycle (convex or star) all of whose corners belong to the chosen classes.
    """
    verts = np.asarray(vertices, dtype=float)
    atol = tol * _scale(verts)
    elements = group_elements(action)
    _, tree = _coset_representatives(verts, elements, atol)
    nbrs = _edge_neighbours(edges, len(verts))[0]
    stabilizer = [m for m, j in zip(elements, tree.query(np.stack(elements) @ verts[0])[1]) if j == 0]
    perms = [tree.query(verts @ m.T)[1] for m in stabilizer]

    seen: set[tuple[int, int]] = set()
    classes: list[Corner] = []
    for v, w in itertools.combinations(sorted(nbrs), 2):
        if (v, w) in seen:
            continue
        a, b = verts[v] - verts[0], verts[w] - verts[0]
        orbit = {tuple(sorted((int(p[v]), int(p[w])))) for p in perms}
        seen |= orbit
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if abs(abs(cos) - 1) < 1e-9:
            continue
        classes.append(Corner(float(np.arccos(np.clip(cos, -1, 1))), tuple(sorted(orbit))))
    classes.sort(key=lambda c: c.angle)
    return classes


def _trace_faces(
    verts: np.ndarray,
    nbrs: list[list[int]],
    allowed: Callable[[int, int, int], bool],
    v: int,
    w: int,
    atol: float,
    max_len: int = 40,
) -> list[list[int]]:
    """Every simple planar cycle through v-0-w whose corners all satisfy ``allowed``.

    ``allowed(prev, cur, nxt)`` says whether the corner prev-cur-nxt is in a chosen corner
    class. All corners must turn the same way as v-0-w, which is what lets star polygons
    close up while convex ones stay convex.
    """
    e1 = verts[0] - verts[v]
    e1 = e1 / np.linalg.norm(e1)
    out = verts[w] - verts[0]
    perp = out - (out @ e1) * e1
    e2 = perp / np.linalg.norm(perp)

    def plane_xy(a: int, b: int) -> tuple[float, float]:
        d = verts[b] - verts[a]
        return float(d @ e1), float(d @ e2)

    def side(a: int, b: int, c: int) -> float:
        (x1, y1), (x2, y2) = plane_xy(a, b), plane_xy(b, c)
        return x1 * y2 - y1 * x2

    sign = np.sign(side(v, 0, w))
    found: list[list[int]] = []

    def walk(path: list[int]) -> None:
        prev, cur = path[-2], path[-1]
        for x in nbrs[cur]:
            if x == prev or not allowed(prev, cur, x):
                continue
            if x == 0:
                if cur == v and len(path) >= 3:
                    found.append(list(path))
                continue
            if x in path or len(path) >= max_len:
                continue
            xy = plane_xy(cur, x)
            r = verts[x] - verts[0]
            if np.linalg.norm(r - (r @ e1) * e1 - (r @ e2) * e2) > atol:
                continue
            if np.sign(side(prev, cur, x)) != sign:
                continue
            walk(path + [x])

    walk([0, w])
    return found


def _select_cells(
    verts: np.ndarray,
    faces: list[list[int]],
    cells: list[list[int]],
    action: SymmetryAction,
    tree: cKDTree,
) -> list[list[list[int]]]:
    """Every way to pick whole symmetry classes of candidate cells so each face lies on exactly two.

    Several can exist (the same faces bound both the grand 600-cell and the great icosahedral
    120-cell, say); they are ordered by fewest classes, then fewest cells.
    """
    face_index = {_canonical_cycle(f): i for i, f in enumerate(faces)}
    face_maps = []
    for g in action.generators:
        perm = tree.query(verts @ g.T)[1].tolist()
        face_maps.append([face_index.get(_canonical_cycle([perm[x] for x in f])) for f in faces])

    cell_index = {frozenset(c): i for i, c in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, cell in enumerate(cells):
        for fmap in face_maps:
            image = [fmap[f] for f in cell]
            j = None if None in image else cell_index.get(frozenset(image))
            if j is not None:
                parent[find(i)] = find(j)
    classes: dict[int, list[int]] = defaultdict(list)
    for i in range(len(cells)):
        classes[find(i)].append(i)
    groups = list(classes.values())
    if len(groups) > 16:
        return []

    uses = np.zeros((len(groups), len(faces)), dtype=int)
    for k, group in enumerate(groups):
        for i in group:
            for f in cells[i]:
                uses[k, f] += 1
    solutions = []
    for mask in range(1, 1 << len(groups)):
        chosen = [k for k in range(len(groups)) if mask >> k & 1]
        if np.all(uses[chosen].sum(axis=0) == 2):
            solutions.append(chosen)
    solutions.sort(key=lambda ch: (len(ch), sum(len(groups[k]) for k in ch)))
    return [[cells[i] for k in ch for i in groups[k]] for ch in solutions]


def build_polychoron(
    vertices: np.ndarray,
    action: SymmetryAction,
    orbit_indices: list[int] | None = None,
    *,
    orbits: list[EdgeOrbit] | None = None,
    tol: float = 1e-7,
    convex: bool = True,
    face_corners: list[int] | None = None,
    cell_choice: int = 0,
    _geom: _Geom | None = None,
) -> Polychoron:
    """Build faces and cells from the union of the chosen edge orbits.

    By default faces are the planar cycles that turn along the faces of a convex vertex
    figure. With ``face_corners`` (indices into ``corner_classes``) each face is instead
    the equiangular polygon, convex or star, that closes up a chosen corner class; no
    convex vertex figure is needed, so star polychora with star vertex figures work.

    With ``convex`` (the default) a cell must lie on a supporting hyperplane of the
    vertex set, which rules out closed polyhedra that sit inside the polychoron.

    Candidate cells are grouped into symmetry classes and a set of classes covering every
    face exactly twice is chosen. If several sets work, ``cell_choice`` picks one (fewest
    classes first) and ``Polychoron.cell_choices`` says how many there are.

    ``orbit_indices`` index into ``edge_orbits(vertices, action)`` (shortest
    first); by default only the shortest orbit is used. Anything that does not
    close up is reported in ``problems`` instead of raised.
    """
    poly = _build(
        vertices, action, orbit_indices, orbits, tol, convex, face_corners, cell_choice, _geom
    )
    poly.orbit_indices = tuple(orbit_indices if orbit_indices is not None else [0])
    poly.face_corners = None if face_corners is None else tuple(face_corners)
    poly.cell_choice = min(cell_choice, max(poly.cell_choices - 1, 0))
    return poly


def _build(
    vertices: np.ndarray,
    action: SymmetryAction,
    orbit_indices: list[int] | None,
    orbits: list[EdgeOrbit] | None,
    tol: float,
    convex: bool,
    face_corners: list[int] | None,
    cell_choice: int,
    geom: _Geom | None,
) -> Polychoron:
    verts = np.asarray(vertices, dtype=float)
    orbits = orbits if orbits is not None else edge_orbits(verts, action, tol=tol)
    chosen = orbit_indices if orbit_indices is not None else [0]
    edges = sorted({e for i in chosen for e in orbits[i].edges})
    atol = tol * _scale(verts)

    nbrs: list[list[int]] = [[] for _ in verts]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)

    geom = geom or _Geom(verts, action, atol)
    reps, tree = geom.reps, geom.tree

    problems: list[str] = []
    if len(nbrs[0]) < 4:
        return Polychoron(verts, edges, [], [], ["fewer than 4 edges at a vertex"])
    if face_corners is not None:
        classes = corner_classes(verts, action, edges, tol=tol)
        chosen_pairs = {pair for i in face_corners for pair in classes[i].pairs}

        def allowed(prev: int, cur: int, nxt: int) -> bool:
            back = geom.back(cur)
            return tuple(sorted((int(back[prev]), int(back[nxt])))) in chosen_pairs

        faces_at_zero = set()
        for v, w in chosen_pairs:
            for first, second in ((v, w), (w, v)):
                for cycle in _trace_faces(verts, nbrs, allowed, first, second, atol):
                    faces_at_zero.add(_canonical_cycle(cycle))
    else:
        try:
            figure = _vertex_figure_pairs(verts, nbrs[0], verts.mean(axis=0), atol)
        except ValueError as err:
            return Polychoron(verts, edges, [], [], [str(err)])


        def corner(prev: int, cur: int, w: int) -> bool:
            # carry the corner back to vertex 0 and look it up in the vertex figure there
            back = geom.back(cur)
            return frozenset((int(back[prev]), int(back[w]))) in figure

        faces_at_zero = _faces_at_vertex_zero(verts, nbrs, atol, corner, figure)
    # cheap screen: every edge of a polychoron lies on at least three faces
    through_zero = {n: 0 for n in nbrs[0]}
    for cycle in faces_at_zero:
        m = len(cycle)
        i = cycle.index(0)
        for n in (cycle[(i - 1) % m], cycle[(i + 1) % m]):
            through_zero[n] += 1
    if min(through_zero.values()) < 3:
        return Polychoron(verts, edges, [], [], ["some edge lies on fewer than 3 faces"])
    all_faces: dict[tuple[int, ...], None] = {}
    for rep in reps.values():
        images = tree.query(verts @ rep.T)[1]
        for cycle in faces_at_zero:
            all_faces[_canonical_cycle([int(images[i]) for i in cycle])] = None
    faces = [list(c) for c in all_faces]

    if not faces:
        problems.append("no faces found")
        return Polychoron(verts, edges, [], [], problems)
    face_edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
    if face_edges != set(edges):
        problems.append(f"{len(set(edges) - face_edges)} edges lie on no face")

    candidates, problems_c = _build_cells(verts, faces, atol, convex)
    problems += problems_c
    choices = _select_cells(verts, faces, candidates, action, tree) if candidates else []
    if not choices:
        if candidates:
            problems.append("no set of cells covers every face exactly twice")
        else:
            problems.append("no closed cells found")
        return Polychoron(verts, edges, faces, candidates, problems)
    cells = choices[min(cell_choice, len(choices) - 1)]
    result = Polychoron(
        verts, edges, faces, cells, problems, cell_choices=len(choices), cell_sets=choices
    )
    # star polychora have other Euler characteristics (the Schläfli-Hess ones are +-480)
    if convex and not problems and result.euler != 0:
        problems.append(f"Euler characteristic is {result.euler}, not 0")
    return result


def reflection_edge_orbit_indices(
    vertices: np.ndarray,
    action: SymmetryAction,
    orbits: list[EdgeOrbit],
    *,
    tol: float = 1e-7,
) -> list[int]:
    """Edge orbits made of pairs ``v, r(v)`` for a reflection ``r`` in the group.

    These are the Wythoff edges: a ringed node of a Coxeter diagram is a mirror and
    joins ``v`` to its mirror image. Other reflections of the group (the non-simple
    roots) give the longer chords that star polygons are made of. Pairs that no
    reflection maps onto each other (rotation-only or chiral edges) are left out, so
    for a group with few reflections this list is short or empty.
    """
    verts = np.asarray(vertices, dtype=float)
    elements = group_elements(action)
    if elements is None:
        raise ValueError("The symmetry group is too large to enumerate.")
    atol = tol * _scale(verts)
    _, tree = _coset_representatives(verts, elements, atol)
    pair_orbit = {e: i for i, o in enumerate(orbits) for e in o.edges}
    eye = np.eye(4)
    found: set[int] = set()
    for m in elements:
        if abs(np.linalg.det(m) + 1) > 1e-6 or np.linalg.matrix_rank(m - eye, tol=1e-6) != 1:
            continue
        images = tree.query(verts @ m.T)[1].tolist()
        for a, b in enumerate(images):
            if a < b:
                found.add(pair_orbit[(a, b)])
    return sorted(found)


def rotation_edge_orbit_indices(
    vertices: np.ndarray,
    action: SymmetryAction,
    orbits: list[EdgeOrbit],
    *,
    generators: list[np.ndarray] | None = None,
    tol: float = 1e-7,
) -> list[int]:
    """Edge orbits made of pairs ``v, g^k(v)`` for powers of the group's generators ``g``.

    This is the rotation counterpart of ``reflection_edge_orbit_indices``: it needs no
    mirrors. The orbit of ``v`` under ``<g>`` is a polygon, and a power ``g^k`` joins
    vertices k steps apart (a star polygon when k > 1). ``generators`` defaults to the
    action's own generators; pass others (any group elements) to choose which rotations
    count as the "nodes".
    """
    verts = np.asarray(vertices, dtype=float)
    elements = group_elements(action)
    if elements is None:
        raise ValueError("The symmetry group is too large to enumerate.")
    atol = tol * _scale(verts)
    _, tree = _coset_representatives(verts, elements, atol)
    pair_orbit = {e: i for i, o in enumerate(orbits) for e in o.edges}
    found: set[int] = set()
    for g in generators if generators is not None else action.generators:
        power = np.eye(4)
        for _ in range(len(elements)):
            power = power @ g
            if np.allclose(power, np.eye(4), atol=1e-6):
                break
            j = int(tree.query(power @ verts[0])[1])
            if j != 0:
                found.add(pair_orbit[(0, j)])
    return sorted(found)


def find_closed_polychora(
    vertices: np.ndarray,
    action: SymmetryAction,
    *,
    max_orbits: int = 2,
    tol: float = 1e-7,
    reflections_only: bool = False,
    rotations_only: bool = False,
    star: bool = False,
    max_corner_classes: int = 8,
    progress=None,
    time_limit: float | None = None,
) -> list[tuple[tuple[int, ...], Polychoron]]:
    """Try unions of up to ``max_orbits`` edge orbits (shortest first) and keep those that close.

    ``reflections_only`` restricts the candidates to ``reflection_edge_orbit_indices``,
    which is how a Coxeter diagram picks its edges; ``rotations_only`` to
    ``rotation_edge_orbit_indices`` (powers of the generators), for groups without mirrors. ``star`` also tries every subset of
    corner classes with ``convex=False``; candidates that cannot close are screened out cheaply
    before any cells are built, results are de-duplicated, and every cell choice is returned;
    ``progress(done, total)`` and ``time_limit`` (seconds) are optional; the result is the combo of orbit indices, so
    use ``build_polychoron`` with ``face_corners`` to reproduce a particular one.
    """
    verts = np.asarray(vertices, dtype=float)
    orbits = edge_orbits(verts, action, tol=tol)
    if reflections_only:
        candidates = reflection_edge_orbit_indices(verts, action, orbits, tol=tol)
    elif rotations_only:
        candidates = rotation_edge_orbit_indices(verts, action, orbits, tol=tol)
    else:
        candidates = list(range(len(orbits)))
    geom = _Geom(verts, action, tol * _scale(verts))
    started = time.monotonic()
    combos = [
        c for k in range(1, max_orbits + 1) for c in itertools.combinations(candidates, k)
    ]
    found: list[tuple[tuple[int, ...], Polychoron]] = []
    seen: set[tuple] = set()

    def keep(combo, poly):
        if not poly.is_closed:
            return
        for choice in range(max(poly.cell_choices, 1)):
            cells = poly.cell_sets[choice] if poly.cell_sets else poly.cells
            key = (
                frozenset(_canonical_cycle(f) for f in poly.faces),
                frozenset(frozenset(c) for c in cells),
            )
            if key in seen:
                continue
            seen.add(key)
            found.append((combo, replace(poly, cells=cells, cell_choice=choice)))

    for n, combo in enumerate(combos):
        if progress is not None:
            progress(n, len(combos))
        if time_limit is not None and time.monotonic() - started > time_limit:
            break
        if not star:
            keep(combo, build_polychoron(
                verts, action, list(combo), orbits=orbits, tol=tol, _geom=geom))
            continue
        edges = sorted({e for i in combo for e in orbits[i].edges})
        degree = np.bincount(np.asarray(edges).ravel(), minlength=len(verts))
        if degree.min() != degree.max() or degree[0] < 4:
            continue
        n_classes = len(corner_classes(verts, action, edges, tol=tol))
        if not 1 <= n_classes <= max_corner_classes:
            continue
        for r in range(1, n_classes + 1):
            for subset in itertools.combinations(range(n_classes), r):
                keep(combo, build_polychoron(
                    verts, action, list(combo), orbits=orbits, tol=tol,
                    convex=False, face_corners=list(subset), _geom=geom,
                ))
    if progress is not None:
        progress(len(combos), len(combos))
    return found


def hull_edge_orbit_indices(
    vertices: np.ndarray, orbits: list[EdgeOrbit]
) -> list[int]:
    """Indices of the edge orbits that are edges of the convex hull."""
    faces, _ = compute_convex_hull(vertices)
    hull_edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
    return [i for i, o in enumerate(orbits) if set(o.edges) & hull_edges]


def compare_to_hull(poly: Polychoron) -> list[str]:
    """Differences between a polychoron and the convex hull of its vertices (empty if equal)."""
    faces, cells = compute_convex_hull(poly.vertices)
    diffs = []
    ours = {_canonical_cycle(f) for f in poly.faces}
    theirs = {_canonical_cycle(f) for f in faces}
    if ours != theirs:
        diffs.append(f"faces differ: {len(ours - theirs)} extra, {len(theirs - ours)} missing")
    ours_c = {frozenset(v for f in c for v in poly.faces[f]) for c in poly.cells}
    theirs_c = {frozenset(v for f in c for v in faces[f]) for c in cells}
    if ours_c != theirs_c:
        diffs.append(f"cells differ: {len(ours_c - theirs_c)} extra, {len(theirs_c - ours_c)} missing")
    return diffs
