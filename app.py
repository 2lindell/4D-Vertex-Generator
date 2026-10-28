from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from four_d_vertex_generator.cli import parse_seed
from four_d_vertex_generator.generation import (
    generate_vertices_from_seed,
    group_order,
    is_subgroup,
)
from four_d_vertex_generator.isogonal import (
    combined_matching_action,
    compute_orbits,
    detect_symmetries,
    split_by_orbits,
)
from four_d_vertex_generator.library import (
    available_symmetries,
    fundamental_chamber_roots,
    h4_swirlprism_predefined_seed,
    named_symmetry,
)
from four_d_vertex_generator.local_view import (
    LocalView,
    edge_length_classes,
    hull_edges,
    local_view,
    nearest_neighbour_edges,
)
from four_d_vertex_generator.off import compute_convex_hull, parse_4off, to_4off
from four_d_vertex_generator.polychoron import (
    Polychoron,
    build_polychoron,
    compare_to_hull,
    corner_classes,
    edge_orbits,
    find_closed_polychora,
    reflection_edge_orbit_indices,
    rotation_edge_orbit_indices,
)

_ICON = Path(__file__).with_name("assets") / "favicon.svg"  # a tetrahedron, for the browser tab
st.set_page_config(
    page_title="4D Vertex Generator",
    page_icon=str(_ICON) if _ICON.exists() else None,
    layout="wide",
)
st.title("4D Vertex Generator")
st.caption(
    "Choose a symmetry, place a seed point, generate every vertex in its orbit, "
    "and export the result as a 4OFF file."
)

# Each family lists (symmetry name, label) pairs. Alias names that resolve to
# the same group are accepted by the library but intentionally not listed.
GROUP_OPTIONS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "elementary": (
        "Elementary coordinate groups",
        (
            ("identity", "Identity C1"),
            ("coordinate_permutations", "Coordinate permutations S4"),
            ("cyclic_coordinate_rotations", "Cyclic rotations C4"),
            ("dihedral_coordinate_symmetries", "Dihedral symmetries D4"),
            ("global_inversion", "Central inversion Ci"),
        ),
    ),
    "a4": (
        "Pentachoric (A4, [3,3,3])",
        (
            ("a4", "Basic [3,3,3]"),
            ("a4+", "Chiral [3,3,3]+"),
            ("a4_extended", "Extended [[3,3,3]]"),
            ("a4_chiral_extended", "Chiral extended [[3,3,3]]+ = [[3,3,3]+]"),
        ),
    ),
    "b4": (
        "Hexadecachoric (B4, [4,3,3])",
        (
            ("hyperoctahedral", "Basic [4,3,3] (signed permutations)"),
            ("b4", "Basic [4,3,3] (Coxeter basis)"),
            ("b4+", "Chiral [4,3,3]+ (Coxeter basis)"),
            ("b4_ionic", "Ionic diminished [4,(3,3)+]"),
            ("b4_half", "Half [1+,4,3,3]"),
            ("b4_half_chiral", "Chiral half [1+,4,(3,3)+]"),
            ("b4_prismatic_octahedral", "Prismatic octahedral [4,3,2]"),
            ("b4_prismatic_octahedral_chiral", "Chiral prismatic octahedral [4,3,2]+"),
            ("b4_prismatic_tetrahedral", "Prismatic tetrahedral [3,3,2]"),
            ("b4_prismatic_tetrahedral_chiral", "Chiral prismatic tetrahedral [3,3,2]+"),
        ),
    ),
    "d4": (
        "Demitesseractic (D4, [3,3,1,1])",
        (
            ("d4", "Basic [3,3,1,1]"),
            ("d4+", "Chiral [3,3,1,1]+"),
            ("d4_extended", "Extended [3,4,3]"),
            ("d4_extended_chiral", "Extended chiral [3,4,3]+"),
        ),
    ),
    "f4": (
        "Icositetrachoric (F4, [3,4,3])",
        (
            ("f4", "Basic [3,4,3]"),
            ("f4+", "Chiral [3,4,3]+"),
            ("f4_extended", "Extended [[3,4,3]]"),
            ("f4_chiral_extended", "Chiral extended [[3,4,3]]+"),
            ("f4_double_diminished", "Double diminished [3+,4,3+]"),
            ("f4_extended_double_diminished", "Extended double diminished [[3+,4,3+]]"),
        ),
    ),
    "h4": (
        "Hexacosichoric (H4, [5,3,3])",
        (
            ("h4", "Basic [5,3,3] (Coxeter basis)"),
            ("h4+", "Chiral [5,3,3]+ (Coxeter basis)"),
            ("h4_icosian", "Basic [5,3,3] (icosian/600-cell basis)"),
            ("h4_icosian+", "Chiral [5,3,3]+ (icosian/600-cell basis)"),
            ("h4_swirlprism", "Small swirlprism [5,3:5] (icosian basis)"),
            ("h4_swirlprism+", "Chiral small swirlprism [5,3:5]+ (icosian basis)"),
            ("h4_pentagonal_swirl", "Pentagonal swirl Z10xZ10 (icosian basis)"),
            ("h4_pentagonal_swirl_ring", "Pentagonal swirl ring Z10 (12 rings of 10)"),
            ("h4_prismatic", "Prismatic [5,3,2]"),
            ("h4_prismatic_chiral", "Chiral prismatic [5,3,2]+"),
            ("h4_ionic", "Ionic diminished [(5,3)+,2]"),
        ),
    ),
    "h4_tubical": ("Tubical groups (swirl)", ()),
    "duoprism": ("Duoprismatic ([p,2,q])", ()),
}

# Reflection groups whose fundamental chamber can be used to place a seed,
# tried in order for each family. A chamber is offered for a subgroup only
# when the two groups act in the same coordinate basis (one contains the other).
CHAMBER_CANDIDATES: dict[str, tuple[str, ...]] = {
    "a4": ("a4",),
    "b4": (
        "hyperoctahedral",
        "b4",
        "d4",
        "b4_prismatic_octahedral",
        "b4_prismatic_tetrahedral",
    ),
    "d4": ("d4", "f4"),
    "f4": ("f4",),
    "h4": ("h4", "h4_prismatic"),
}

SWIRL_SYMMETRIES = ("h4_swirlprism", "h4_swirlprism+")
TUBICAL_FAMILIES: dict[str, dict[str, tuple[str, int]]] = {
    "icosahedral": {
        "cyclic": ("Chiral cyclic Σ[I × Cₙ] · order 120n", 1),
        "dihedral": ("Dihedral Σ[I × D₂ₙ] · order 240n", 2),
    },
    "tetrahedral": {
        "cyclic": ("Cyclic Σ[T × Cₙ] · order 24n", 1),
        "cyclic_index3": ("Index-3 Σ₁/₃[T × C₃ₙ] · order 24n", 1),
        "dihedral": ("Dihedral Σ[T × D₂ₙ] · order 48n", 2),
    },
    "octahedral": {
        "cyclic": ("Cyclic Σ[O × Cₙ] · order 48n", 1),
        "cyclic_index2": ("Index-2 Σ₁/₂[O × C₂ₙ] · order 48n", 1),
        "dihedral": ("Dihedral Σ[O × D₂ₙ] · order 96n", 2),
        "dihedral_index2_4n": ("Index-2 Σ₁/₂[O × D₄ₙ] · order 96n", 2),
        "dihedral_index2_2n": ("Index-2 Σ₁/₂[O × D₂ₙ] · order 48n", 2),
        "dihedral_index6_6n": ("Index-6 Σ₁/₆[O × D₆ₙ] · order 48n", 1),
    },
}
SEED_MODE_CHAMBER = "Fundamental chamber"
SEED_MODE_COORDINATES = "Coordinates"
SEED_MODE_RINGS = "Ring sliders"


def _symmetry_labels() -> dict[str, str]:
    labels = {name: label for _, (_, choices) in GROUP_OPTIONS.items() for name, label in choices}
    for name in available_symmetries():
        if name.startswith("duoprism_"):
            p, q = name.removeprefix("duoprism_").split("_")[:2]
            q = q.rstrip("+")
            kind = "Chiral " if "+" in name or "chiral" in name else ""
            ext = "extended " if "extended" in name else ""
            labels[name] = f"{kind}{ext}{p}-{q} duoprism".capitalize()
    return labels


SYMMETRY_LABELS = _symmetry_labels()


# Bound every cache: unbounded caches of large vertex/hull arrays grew until the
# long-running server ran out of memory and the generator stopped responding.
_CACHE_MAX_ENTRIES = 16
_CACHE_TTL_SECONDS = 3600


def _describe(name: str) -> str:
    label = SYMMETRY_LABELS.get(name)
    return f"{label} — {name}" if label else name


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _group_order(name: str) -> int | None:
    return group_order(named_symmetry(name))


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _compatible_chamber(symmetry_name: str, candidates: tuple[str, ...]) -> str | None:
    action = named_symmetry(symmetry_name)
    for candidate in candidates:
        parent = named_symmetry(candidate)
        if is_subgroup(action, parent) or is_subgroup(parent, action):
            return candidate
    return None


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _cached_generate(
    seed: tuple[float, float, float, float],
    symmetry_name: str,
    tol: float,
    max_vertices: int,
) -> np.ndarray:
    return generate_vertices_from_seed(
        np.array(seed, dtype=float),
        named_symmetry(symmetry_name),
        tol=tol,
        max_vertices=max_vertices,
    )


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _cached_convex_hull(vertices: np.ndarray):
    return compute_convex_hull(vertices)


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _cached_detect_symmetries(vertices: np.ndarray, tol: float) -> list[str]:
    return detect_symmetries(vertices, tol=tol)


def _chamber_svg(weights: np.ndarray) -> str:
    """Draw the chamber as a projected tetrahedron with the seed's position marked."""
    corners = np.array([[170, 25], [35, 155], [305, 155], [170, 105]])
    point = weights @ corners
    edges = "".join(
        f"<line x1='{corners[i, 0]}' y1='{corners[i, 1]}' "
        f"x2='{corners[j, 0]}' y2='{corners[j, 1]}' />"
        for i in range(4)
        for j in range(i + 1, 4)
    )
    label_offsets = ((0, -10), (-14, 12), (14, 12), (0, -12))
    labels = "".join(
        f"<text x='{corners[i, 0] + dx}' y='{corners[i, 1] + dy}' "
        f"text-anchor='middle'>{i + 1}</text>"
        for i, (dx, dy) in enumerate(label_offsets)
    )
    return f"""
<svg width="340" height="185" viewBox="0 0 340 185" role="img"
     aria-label="Projection of the fundamental chamber with the seed position">
  <g stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5" fill="none">{edges}</g>
  <g fill="currentColor" font-size="13" font-family="sans-serif">{labels}</g>
  <circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="7" fill="#e4572e" />
</svg>
"""


def _format_vector(v: np.ndarray) -> str:
    return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"


def _vertices_frame(vertices: np.ndarray, orbit_ids: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(vertices, columns=["x", "y", "z", "w"])
    if orbit_ids is not None:
        frame.insert(0, "orbit", orbit_ids)
    return frame


def _orbits_zip(groups: list[np.ndarray], stem: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, group in enumerate(groups):
            archive.writestr(f"{stem}_orbit{index}_{len(group)}.off", to_4off(group))
    return buffer.getvalue()


# Categorical slots 1-4 of the reference data-viz palette, one per distinct
# edge length. A uniform polytope has at most four (one per ringed node).
EDGE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FIGURE_COLOR = "#8a8984"
CENTER_COLOR = "#52514e"
MAX_EDGE_CLASSES = len(EDGE_COLORS)


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _cached_nearest_edges(vertices: np.ndarray) -> set[tuple[int, int]]:
    return nearest_neighbour_edges(vertices)


def _local_view_figure(
    view: LocalView,
    vertices: np.ndarray,
    same_orbit: np.ndarray | None = None,
) -> go.Figure:
    """Plot the centre vertex, its edges and (optionally) its vertex figure."""
    fig = go.Figure()
    classes = edge_length_classes(view.edge_lengths)
    n_classes = int(classes.max()) + 1 if len(classes) else 0
    # Beyond four lengths the colours would stop being distinguishable, so
    # the extra lengths share the last colour and are told apart on hover.
    classes = np.minimum(classes, MAX_EDGE_CLASSES - 1)

    if view.figure_edges:
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for i, j in view.figure_edges:
            a, b = view.positions[i], view.positions[j]
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name="Vertex figure",
                line={"color": FIGURE_COLOR, "width": 2, "dash": "dash"},
                hoverinfo="skip",
            )
        )

    for cls in range(min(n_classes, MAX_EDGE_CLASSES)):
        members = np.flatnonzero(classes == cls)
        lengths = view.edge_lengths[members]
        xs, ys, zs = [], [], []
        for p in view.positions[members]:
            xs += [0.0, p[0], None]
            ys += [0.0, p[1], None]
            zs += [0.0, p[2], None]
        name = (
            f"Edge length {lengths[0]:.4g}"
            if np.allclose(lengths, lengths[0])
            else f"Edge lengths {lengths.min():.4g}–{lengths.max():.4g}"
        )
        color = EDGE_COLORS[cls]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=name, legendgroup=name,
                line={"color": color, "width": 5}, hoverinfo="skip",
            )
        )
        hover = [
            f"vertex {n}<br>{_format_vector(vertices[n])}<br>edge length {length:.6g}"
            + ("" if same_orbit is None else
               f"<br>{'same orbit' if same_orbit[n] else 'different orbit'}")
            for n, length in zip(view.neighbours[members], lengths)
        ]
        symbols = (
            "circle"
            if same_orbit is None
            else ["circle" if same_orbit[n] else "diamond" for n in view.neighbours[members]]
        )
        fig.add_trace(
            go.Scatter3d(
                x=view.positions[members, 0],
                y=view.positions[members, 1],
                z=view.positions[members, 2],
                mode="markers", name=name, legendgroup=name, showlegend=False,
                marker={"size": 6, "color": color, "symbol": symbols,
                        "line": {"color": "#fcfcfb", "width": 2}},
                hovertext=hover, hoverinfo="text",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0], mode="markers", name=f"Vertex {view.center}",
            marker={"size": 9, "color": CENTER_COLOR, "line": {"color": "#fcfcfb", "width": 2}},
            hovertext=[f"vertex {view.center}<br>{_format_vector(vertices[view.center])}"],
            hoverinfo="text",
        )
    )
    hidden_axis = {
        "visible": False, "showbackground": False, "showgrid": False, "zeroline": False,
    }
    fig.update_layout(
        height=480,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={"xaxis": hidden_axis, "yaxis": hidden_axis, "zaxis": hidden_axis,
               "aspectmode": "data"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
    )
    return fig


def render_local_view(
    vertices: np.ndarray,
    center: int,
    faces: list[list[int]] | None,
    same_orbit: np.ndarray | None = None,
) -> None:
    """Show the 3D neighbourhood of one vertex, projected into its tangent space."""
    if faces is not None:
        edges = hull_edges(faces)
    else:
        edges = _cached_nearest_edges(vertices)
        st.caption(
            "No convex hull, so edges are guessed as the shortest vertex-to-vertex "
            "distance. Longer edges of non-regular shapes will be missing."
        )
    try:
        view = local_view(vertices, center, edges, faces)
    except ValueError as err:
        st.warning(f"Cannot draw the local view: {err}")
        return
    if len(view.neighbours) == 0:
        st.info(f"Vertex {center} has no edges.")
        return

    # the mouse wheel scrolls the page, not the plot: otherwise the page stops scrolling under
    # the pointer and the figure zooms away (zoom with the toolbar or a pinch instead)
    st.plotly_chart(
        _local_view_figure(view, vertices, same_orbit),
        theme="streamlit",
        config={"scrollZoom": False},
    )
    n_classes = len(set(edge_length_classes(view.edge_lengths).tolist()))
    summary = f"{len(view.neighbours)} edges meet at vertex {center}"
    if n_classes > 1:
        summary += f", with {n_classes} different lengths"
    if view.figure_edges:
        summary += "; the dashed outline is the vertex figure"
    st.caption(
        summary + ". Drag to rotate; zoom with the toolbar. The view looks along the vertex's "
        "radius, so every line is an edge direction as seen from the centre of the polytope."
    )


def render_generate_tab() -> None:
    left, right = st.columns(2)
    with left:
        group_key = st.selectbox(
            "Symmetry family",
            options=tuple(GROUP_OPTIONS),
            format_func=lambda key: GROUP_OPTIONS[key][0],
            index=1,
        )
    if group_key == "duoprism":
        with right:
            p, q = st.selectbox(
                "Duoprism orders (p, q)",
                options=[(p, q) for p in range(3, 7) for q in range(p, 7)],
                format_func=lambda order: f"{order[0]}-{order[1]} duoprism",
            )
        subgroup_choices = [
            (f"duoprism_{p}_{q}", f"Basic [{p},2,{q}]"),
            (f"duoprism_{p}_{q}+", f"Chiral [{p},2,{q}]+"),
        ]
        if p == q:
            subgroup_choices += [
                (f"duoprism_{p}_{q}_extended", f"Extended [[{p},2,{q}]]"),
                (f"duoprism_{p}_{q}_chiral_extended", f"Chiral extended [[{p},2,{q}]]+"),
            ]
        chamber_candidates: tuple[str, ...] = (f"duoprism_{p}_{q}",)
    elif group_key == "h4_tubical":
        with right:
            polyhedral_factor = st.selectbox(
                "Polyhedral factor",
                options=tuple(TUBICAL_FAMILIES),
                format_func=lambda factor: factor.capitalize(),
            )
            family_options = TUBICAL_FAMILIES[polyhedral_factor]
            tubical_family = st.selectbox(
                "Tubical subgroup",
                options=tuple(family_options),
                format_func=lambda family: family_options[family][0],
            )
            tubical_order = st.number_input(
                "Family parameter n",
                min_value=family_options[tubical_family][1],
                max_value=120,
                value=5,
                step=1,
            )
        if polyhedral_factor == "icosahedral" and tubical_order == 5:
            symmetry_name = "h4_swirlprism+" if tubical_family == "cyclic" else "h4_swirlprism"
        else:
            symmetry_name = (
                f"h4_tubical_{polyhedral_factor}_{tubical_family}_{int(tubical_order)}"
            )
        subgroup_choices = [(symmetry_name, family_options[tubical_family][0])]
        chamber_candidates = ()
    else:
        subgroup_choices = list(GROUP_OPTIONS[group_key][1])
        chamber_candidates = CHAMBER_CANDIDATES.get(group_key, ())

    choice_labels = dict(subgroup_choices)
    if group_key != "h4_tubical":
        with left if group_key == "duoprism" else right:
            symmetry_name = st.selectbox(
                "Subgroup",
                options=list(choice_labels),
                format_func=choice_labels.__getitem__,
            )
    order = _group_order(symmetry_name)
    st.caption(
        f"`{symmetry_name}` · group order {order if order is not None else 'unknown'}"
    )

    chamber_name = _compatible_chamber(symmetry_name, chamber_candidates)
    seed_modes = []
    if chamber_name is not None:
        seed_modes.append(SEED_MODE_CHAMBER)
    if symmetry_name in SWIRL_SYMMETRIES:
        seed_modes.append(SEED_MODE_RINGS)
    seed_modes.append(SEED_MODE_COORDINATES)

    st.subheader("Seed point")
    seed_mode = st.radio(
        "Seed input",
        seed_modes,
        horizontal=True,
        label_visibility="collapsed",
    )
    if chamber_name is None and chamber_candidates:
        st.caption(
            "This subgroup uses a different coordinate basis from the family's reflection "
            "group, so the fundamental-chamber picker is not available; enter coordinates."
        )

    # Advanced settings drive the sliders above them, so fill the expander
    # first but place it on the page after the seed controls.
    seed_area = st.container()
    with st.expander("Advanced settings"):
        adv = st.columns(3)
        snap = adv[0].checkbox(
            "Coarse slider steps",
            value=True,
            help="Chamber sliders move in steps of 0.05 and ring sliders in whole degrees.",
        )
        tol = adv[1].number_input(
            "Merge tolerance",
            min_value=1e-12,
            max_value=1e-2,
            value=1e-8,
            format="%.1e",
            help="Two generated points closer than this (per coordinate) count as one vertex.",
        )
        max_vertices = adv[2].number_input(
            "Max vertices",
            min_value=1,
            max_value=500_000,
            value=20_000,
            step=1_000,
            help="Safety cap for groups that turn out to be very large.",
        )

    with seed_area:
        seed: np.ndarray | None = None
        seed_error: str | None = None
        if seed_mode == SEED_MODE_CHAMBER:
            roots = fundamental_chamber_roots(chamber_name)
            st.caption(
                "Each slider is the seed's distance from one mirror of the chamber "
                "(one node of the Coxeter diagram). Set a node to 0 to put the seed on that "
                "mirror; a single nonzero node gives a regular or uniform vertex."
            )
            columns = st.columns(4)
            weights = np.array(
                [
                    columns[i].slider(
                        f"Node {i + 1}",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05 if snap else 0.01,
                        key=f"chamber_node_{i}",
                    )
                    for i in range(4)
                ]
            )
            if weights.sum() == 0.0:
                seed_error = "Set at least one node above 0."
            else:
                seed = np.linalg.solve(roots, weights)
                seed /= np.linalg.norm(seed)
                st.markdown(_chamber_svg(weights / weights.sum()), unsafe_allow_html=True)
        elif seed_mode == SEED_MODE_RINGS:
            st.caption(
                "Starts at a 600-cell vertex (120 points). Cross rings 1 and 2 are adjacent "
                "cross rings, 36° apart around the main ring and running in opposite "
                "directions; under the full swirlprism every point on them gives 600 "
                "vertices, except 120 at every 90° and at a second 600-cell's vertices "
                "(58.28° past each on cross ring 1, 31.72° on cross ring 2). The main ring "
                "gives 240, back to a 600-cell every 18°."
            )
            columns = st.columns(3)
            angles = [
                column.slider(
                    label,
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    step=1.0 if snap else 0.1,
                    format="%.1f°",
                    key=f"swirl_{index}",
                )
                for index, (column, label) in enumerate(
                    zip(columns, ("Cross ring 1", "Cross ring 2", "Main ring"))
                )
            ]
            seed = h4_swirlprism_predefined_seed(*angles)
        else:
            seed_text = st.text_input(
                "Seed (x, y, z, w)",
                value="1,0,0,0",
                help="Four comma-separated numbers, e.g. 1,1,0,0",
            )
            try:
                seed = parse_seed(seed_text)
            except ValueError as err:
                seed_error = str(err)

        if seed_error:
            st.error(seed_error)
        elif seed is not None:
            st.caption(f"Seed used: {_format_vector(seed)}")

    compute_hull = st.checkbox(
        "Compute 4D convex hull (adds faces and cells to the export)", value=True
    )
    if st.button("Generate vertices", type="primary", disabled=seed is None):
        orbit_tol = float(tol)
        if seed_mode == SEED_MODE_RINGS:
            # Ring-slider seeds are built from eigenvectors, so they carry
            # ~1e-8 noise; a looser merge tolerance keeps the orbit exact.
            orbit_tol = max(orbit_tol, 1e-6)
        started = time.perf_counter()
        status = st.status("Generating vertices...", expanded=True)
        try:
            vertices = _cached_generate(
                tuple(float(x) for x in seed), symmetry_name, orbit_tol, int(max_vertices)
            )
            status.write(
                f"✅ {len(vertices):,} vertices ({time.perf_counter() - started:.1f} s)"
            )
            faces = cells = None
            hull_error = None
            if compute_hull:
                status.update(label="Computing 4D convex hull (slow for tens of thousands of "
                                    "faces)...")
                hull_started = time.perf_counter()
                try:
                    faces, cells = _cached_convex_hull(vertices)
                    status.write(
                        f"✅ {len(faces):,} faces, {len(cells):,} cells "
                        f"({time.perf_counter() - hull_started:.1f} s)"
                    )
                except ValueError as err:
                    hull_error = str(err)
                    status.write(f"⚠️ Convex hull not computed: {hull_error}")
            st.session_state["generated"] = {
                "symmetry": symmetry_name,
                "order": order,
                "seed": seed,
                "vertices": vertices,
                "faces": faces,
                "cells": cells,
                "hull_error": hull_error,
            }
            status.update(
                label=f"Done in {time.perf_counter() - started:.1f} s",
                state="complete",
                expanded=False,
            )
        except (ValueError, RuntimeError, MemoryError) as err:
            st.session_state.pop("generated", None)
            message = str(err) or type(err).__name__
            status.update(label="Failed", state="error", expanded=True)
            status.error(message)

    result = st.session_state.get("generated")
    if result is None:
        return

    st.divider()
    vertices = result["vertices"]
    st.subheader(f"Result: {_describe(result['symmetry'])}")
    st.caption(f"Seed {_format_vector(result['seed'])}")
    metrics = st.columns(4)
    metrics[0].metric("Vertices", len(vertices))
    if result["order"]:
        metrics[1].metric(
            "Seed stabilizer order",
            result["order"] // len(vertices),
            help="Group order ÷ vertex count. 1 means the seed lies on no mirror.",
        )
    if result["faces"] is not None:
        metrics[2].metric("Faces", len(result["faces"]))
        metrics[3].metric("Cells", len(result["cells"]))
    if result["hull_error"]:
        st.warning(f"Convex hull not computed: {result['hull_error']}")

    st.download_button(
        "Download 4OFF",
        data=to_4off(vertices, faces=result["faces"], cells=result["cells"]),
        file_name=f"{result['symmetry']}_vertices_{len(vertices)}.off",
        mime="text/plain",
        type="primary",
    )
    st.subheader("Around one vertex")
    st.caption(
        "Every vertex is equivalent under the symmetry, so one vertex's neighbourhood "
        "shows what the whole shape looks like locally."
    )
    render_local_view(vertices, 0, result["faces"])

    with st.expander("All vertex coordinates"):
        st.dataframe(_vertices_frame(vertices), height=280, width="stretch")


def _render_split_local_view(vertices: np.ndarray, orbit_ids: list[int]) -> None:
    st.subheader("Around one vertex")
    num_orbits = max(orbit_ids) + 1
    orbit = 0
    if num_orbits > 1:
        sizes = np.bincount(orbit_ids)
        orbit = st.selectbox(
            "Orbit",
            options=range(num_orbits),
            format_func=lambda o: f"Orbit {o} ({sizes[o]} vertices)",
            help="Vertices in the same orbit look identical locally, so one "
            "representative per orbit is shown.",
        )
    if not st.toggle("Show local vertex view", help="Computes the convex hull of the file."):
        return
    ids = np.asarray(orbit_ids)
    center = int(np.flatnonzero(ids == orbit)[0])
    try:
        with st.spinner("Computing 4D convex hull..."):
            faces, _ = _cached_convex_hull(vertices)
    except ValueError as err:
        st.caption(f"Convex hull unavailable ({err}).")
        faces = None
    render_local_view(vertices, center, faces, same_orbit=ids == orbit if num_orbits > 1 else None)
    if num_orbits > 1:
        st.caption("Circles are neighbours in the same orbit; diamonds are in other orbits.")


def render_analyze_tab() -> None:
    st.caption(
        "Upload a 4D OFF file to detect which built-in symmetries its vertices have, "
        "then optionally split the vertices into isogonal groups (orbits)."
    )
    uploaded = st.file_uploader("4D OFF file", type=["off", "4off", "txt"])
    with st.expander("Advanced settings"):
        analyze_tol = float(
            st.number_input(
                "Symmetry tolerance",
                min_value=1e-12,
                max_value=1e-2,
                value=1e-6,
                format="%.1e",
                help="How far a transformed vertex may land from an existing one and "
                "still count as a match.",
            )
        )
    if uploaded is None:
        return

    try:
        vertices = parse_4off(uploaded.getvalue().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        st.error(f"Could not read this file as 4D OFF: {err}")
        return
    st.success(f"Parsed {len(vertices)} vertices from `{uploaded.name}`.")

    with st.spinner("Detecting symmetry..."):
        detected = _cached_detect_symmetries(vertices, analyze_tol)

    if detected:
        st.markdown(f"**Highest detected symmetry:** {_describe(detected[0])}")
        with st.expander(f"All {len(detected)} matching symmetries", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    {
                        "name": detected,
                        "description": [SYMMETRY_LABELS.get(n, "") for n in detected],
                        "order": [_group_order(n) for n in detected],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
    else:
        st.warning(
            "No built-in symmetry matched this vertex set at this tolerance, so there is "
            "nothing to split under. Try a larger tolerance under Advanced settings."
        )
        return

    # only the symmetries that map this vertex set onto itself are offered
    auto = "auto (combine all matches)"
    split_options = [auto, *detected]
    split_name = st.selectbox(
        "Split into isogonal groups under",
        options=split_options,
        index=1,
        format_func=lambda name: name if name == auto else _describe(name),
        help=(
            "Only symmetries detected for this file are listed, highest first; the highest is "
            "the default. 'auto' combines every matching symmetry into one larger group, "
            "which gives the fewest, largest orbits."
        ),
    )
    family_filter = ""
    if split_name == auto:
        family_filter = st.text_input(
            "Only combine symmetries whose name contains (optional)",
            help="e.g. 'swirlprism' to find the largest matching swirl subgroup.",
        ).strip()

    if st.button("Split vertices", type="primary"):
        if split_name == auto:
            candidates = (
                [n for n in available_symmetries() if family_filter in n]
                if family_filter
                else None
            )
            matched = detect_symmetries(vertices, tol=analyze_tol, candidates=candidates)
            if not matched:
                st.error("No matching symmetry contains that text.")
                return
            action = combined_matching_action(vertices, tol=analyze_tol, candidates=candidates)
            label = "auto_" + (family_filter or "all")
            note = f"Combined: {', '.join(matched)}"
        else:
            action = named_symmetry(split_name)
            label = split_name
            note = None
        partition = compute_orbits(vertices, action, tol=analyze_tol)
        st.session_state["split"] = {
            "file": uploaded.name,
            "label": label,
            "note": note,
            "partition": partition,
            "groups": split_by_orbits(vertices, partition),
        }

    split = st.session_state.get("split")
    if split is None or split["file"] != uploaded.name:
        return
    if split["note"]:
        st.caption(split["note"])
    partition = split["partition"]
    if partition.num_orbits == 1:
        st.info(f"These vertices are already isogonal under `{split['label']}` (one orbit).")
        _render_split_local_view(vertices, partition.orbit_ids)
        return

    groups = split["groups"]
    stem = f"{uploaded.name.rsplit('.', 1)[0]}_{split['label']}"
    st.success(f"Split into {partition.num_orbits} isogonal groups under `{split['label']}`.")
    st.download_button(
        f"Download all {len(groups)} orbits (.zip)",
        data=_orbits_zip(groups, stem),
        file_name=f"{stem}_orbits.zip",
        mime="application/zip",
        type="primary",
    )
    st.dataframe(
        pd.DataFrame(
            {"orbit": range(len(groups)), "vertices": [len(g) for g in groups]}
        ),
        hide_index=True,
    )
    with st.expander("Download individual orbits"):
        for index, group in enumerate(groups):
            st.download_button(
                f"Orbit {index} ({len(group)} vertices)",
                data=to_4off(group),
                file_name=f"{stem}_orbit{index}_{len(group)}.off",
                mime="text/plain",
                key=f"orbit_download_{index}",
            )
    _render_split_local_view(vertices, partition.orbit_ids)


# Plotly's 3D plots swallow every mouse-wheel event (even with scrollZoom off), so the page
# stops scrolling whenever the pointer is over one. This hands those events back to the page:
# a capture listener on the app's window stops them before they reach the plot's canvas.
WHEEL_TO_PAGE = """
<script>
try {
  const host = window.parent;
  if (!host.__wheelToPage) {
    host.__wheelToPage = true;
    host.addEventListener("wheel", (event) => {
      const target = event.target;
      if (target instanceof host.Element && target.closest(".js-plotly-plot .gl-container")) {
        event.stopPropagation();
      }
    }, { capture: true });
  }
} catch (err) {
  // a host page that keeps this frame out leaves the plots as they were
}
</script>
"""


def let_wheel_scroll_page() -> None:
    components.html(WHEEL_TO_PAGE, height=0)


DOMAIN_TITLE = "H3●I2(10) Symmetry Domain"
DOMAIN_PAGE = Path(__file__).parent / "assets" / "symmetry_domain.html"
OCT_TITLE = "Octahedral Swirl Symmetry Domain"
OCT_PAGE = Path(__file__).parent / "assets" / "tetraswirl_oct_domain.html"
SITE = "https://2lindell.github.io/4D-Vertex-Generator/"


def render_page_tab(page: Path, caption: str) -> None:
    """A symmetry domain page (built in research/), shown in the tab at the page's own height."""
    st.caption(caption)
    # the page grows its frame to its own height (it reaches the frame from the same origin),
    # so the app's scroll bar is the only one
    if sys.platform == "emscripten":
        # in the browser build (GitHub Pages) the page is served next to the app
        st.markdown(f"[Open it in its own tab]({page.name})")
        components.iframe(page.name, height=1100)
    elif page.exists():
        components.html(page.read_text(encoding="utf-8"), height=1100)
    else:
        st.info(f"The page is not in this copy of the app; it is published at {SITE}{page.name}")


NONCONVEX_TITLE = "Non-convex generator (beta)"
NC_MAX_VERTICES = 3000  # building faces and cells is much slower than generating the orbit
NC_MAX_SEARCH_VERTICES = 400
NC_EDGES_REFLECTION = "Mirror edges (Coxeter diagram)"
NC_EDGES_ROTATION = "Rotation edges (generator powers)"
NC_EDGES_ALL = "Every edge orbit"
NC_FACES_VERTEX_FIGURE = "Convex vertex figure"
NC_FACES_CORNERS = "Chosen corner classes (allows star faces)"

NC_HELP = """
**What this does.** The *Generate* tab always joins the orbit's points into their **convex hull**.
This tab instead lets you choose **which pairs of vertices are edges**, then builds the faces and
cells from those edges. Choosing longer chords gives **star polygons and non-convex or
self-intersecting cells**, such as the pentagram prisms and the Schläfli–Hess polychora. Every
result is checked to be closed: each face lies on exactly two cells.

**How to use it**
1. **Symmetry and seed.** Pick a group and a seed point, exactly as in *Generate*. A seed on a
   mirror gives a more symmetric shape; a generic seed gives unequal edge lengths (non-uniform).
2. **Edge orbits.** Every pair of vertices belongs to an *edge orbit*: the pairs the symmetry maps
   onto each other. They are listed shortest first. Pick the orbits that are your edges.
   *Mirror edges* are the pairs `v, r(v)` for a reflection `r`: the ringed nodes of a Coxeter
   diagram, plus the longer chords from the group's other reflections. *Rotation edges* are
   `v, g^k(v)` for powers of a generating rotation `g`, which also works for groups with no mirrors
   (chiral groups, swirlprisms). *Every edge orbit* lists them all and gets long quickly.
3. **Faces.** *Convex vertex figure* is right for convex shapes. For stars, choose **corner
   classes**: a corner is two edges meeting at a vertex, and a face is a polygon all of whose
   corners are in the chosen classes. Several classes may be chosen.
4. **Build.** If several sets of cells close up on the same faces (the same faces bound both the
   600-cell and the icosahedral 120-cell, say), choose which with *Cell set*.
5. **Or search.** The search expander tries combinations of edge orbits for you and lists the
   closed polychora it finds.

**Examples to try**
| Symmetry | Seed | Edge orbits | Faces | Result |
| --- | --- | --- | --- | --- |
| 5-5 duoprism, basic | `1, 0, 1, 0` | `#0` (1.176) and `#4` (1.902): pentagon edges in one plane, pentagram chords in the other | corner classes: all, and untick *convex cells* | pentagon × pentagram star duoprism |
| Basic H4 `[5,3,3]` (Coxeter basis) | `1, 0, 0, 0` | `#0` (length 0.618) | only the 60° corner, untick *convex cells* | icosahedral 120-cell {3,5,5/2}; *Cell set* 1 gives the 600-cell |
| Basic H4 `[5,3,3]` (Coxeter basis) | `1, 0, 0, 0` | `#4` (length 1.618) | only the 60° corner, untick *convex cells* | great icosahedral 120-cell {3,5/2,5}; *Cell set* 1 gives the grand 600-cell {3,3,5/2} |

**Limits of the beta.** Building is slow above a few hundred vertices. Faces are simple polygons
in one plane. Nothing is checked beyond closure, so a closed result is not necessarily a
*uniform* polychoron; read the face and cell counts. The 4OFF export lists explicit faces and
cells; not every viewer draws star polytopes correctly.
"""


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _nc_orbit_data(symmetry_name: str, vertices: np.ndarray):
    """Edge orbits plus the mirror and rotation candidate lists for one vertex orbit."""
    action = named_symmetry(symmetry_name)
    orbits = edge_orbits(vertices, action)
    return (
        orbits,
        reflection_edge_orbit_indices(vertices, action, orbits),
        rotation_edge_orbit_indices(vertices, action, orbits),
    )


@st.cache_data(show_spinner=False, max_entries=_CACHE_MAX_ENTRIES, ttl=_CACHE_TTL_SECONDS)
def _nc_corner_classes(symmetry_name: str, vertices: np.ndarray, edges: tuple[tuple[int, int], ...]):
    return corner_classes(vertices, named_symmetry(symmetry_name), list(edges))


def _nc_symmetry_picker() -> str:
    left, right = st.columns(2)
    families = {key: value for key, value in GROUP_OPTIONS.items() if key != "h4_tubical"}
    keys = tuple(families)
    family = left.selectbox(
        "Symmetry family",
        options=keys,
        format_func=lambda key: families[key][0],
        index=keys.index("duoprism"),
        key="nc_family",
    )
    if family == "duoprism":
        p, q = right.selectbox(
            "Duoprism orders (p, q)",
            options=[(p, q) for p in range(3, 7) for q in range(p, 7)],
            index=[(p, q) for p in range(3, 7) for q in range(p, 7)].index((5, 5)),
            format_func=lambda order: f"{order[0]}-{order[1]} duoprism",
            key="nc_duoprism_orders",
        )
        choices = [(f"duoprism_{p}_{q}", f"Basic [{p},2,{q}]"), (f"duoprism_{p}_{q}+", f"Chiral [{p},2,{q}]+")]
        if p == q:
            choices += [
                (f"duoprism_{p}_{q}_extended", f"Extended [[{p},2,{q}]]"),
                (f"duoprism_{p}_{q}_chiral_extended", f"Chiral extended [[{p},2,{q}]]+"),
            ]
    else:
        choices = list(families[family][1])
    labels = dict(choices)
    with left if family == "duoprism" else right:
        name = st.selectbox(
            "Subgroup",
            options=list(labels),
            format_func=labels.__getitem__,
            key=f"nc_subgroup_{family}_{len(labels)}",
        )
    return name


def _nc_figure(poly: Polychoron, xw: float, zw: float) -> go.Figure:
    """Perspective view of the edges: rotate in the XW and ZW planes, then project away W."""
    pts = np.asarray(poly.vertices, dtype=float).copy()
    for axis, angle in ((0, xw), (2, zw)):
        c, s_ = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        pts[:, [axis, 3]] = pts[:, [axis, 3]] @ np.array([[c, s_], [-s_, c]])
    radius = float(np.max(np.linalg.norm(pts, axis=1))) or 1.0
    xyz = pts[:, :3] / (2.5 * radius - pts[:, 3:4]) * 2.5 * radius
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for i, j in poly.edges:
        xs += [xyz[i, 0], xyz[j, 0], None]
        ys += [xyz[i, 1], xyz[j, 1], None]
        zs += [xyz[i, 2], xyz[j, 2], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"width": 2}, hoverinfo="skip", name="edges"))
    fig.add_trace(
        go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", marker={"size": 3},
            text=[_format_vector(v) for v in poly.vertices], hoverinfo="text", name="vertices",
        )
    )
    fig.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
        scene={"aspectmode": "data", "xaxis_visible": False, "yaxis_visible": False, "zaxis_visible": False},
    )
    return fig


def _nc_show_result(poly: Polychoron, stem: str) -> None:
    cols = st.columns(5)
    for col, (label, value) in zip(
        cols,
        [
            ("Vertices", len(poly.vertices)),
            ("Edges", len(poly.edges)),
            ("Faces", len(poly.faces)),
            ("Cells", len(poly.cells)),
            ("Euler χ", poly.euler),
        ],
    ):
        col.metric(label, value)
    if poly.is_closed:
        st.success("Closed: every face lies on exactly two cells.")
        if poly.cell_choices > 1:
            st.info(
                f"{poly.cell_choices} different sets of cells close up on these faces. "
                "Change *Cell set* above and build again to see the others."
            )
        hull_diffs = compare_to_hull(poly)
        if hull_diffs:
            st.caption("Differs from the convex hull of its vertices: " + "; ".join(hull_diffs) + ".")
        else:
            st.caption("Identical to the convex hull of its vertices.")
        sizes = {}
        for face in poly.faces:
            sizes[len(face)] = sizes.get(len(face), 0) + 1
        st.caption("Faces by number of sides: " + ", ".join(f"{n} × {k}-gon" for k, n in sorted(sizes.items())))
    else:
        st.error("Not a closed polychoron: " + "; ".join(poly.problems or ["no cells found"]))
    if poly.edges:
        view = st.columns(2)
        xw = view[0].slider("Rotate in the XW plane (°)", -180, 180, 0, key=f"nc_xw_{stem}")
        zw = view[1].slider("Rotate in the ZW plane (°)", -180, 180, 0, key=f"nc_zw_{stem}")
        st.plotly_chart(_nc_figure(poly, xw, zw), width="stretch", key=f"nc_plot_{stem}")
    if poly.faces:
        st.download_button(
            "Download 4OFF",
            data=to_4off(poly.vertices, poly.faces, poly.cells if poly.is_closed else []),
            file_name=f"{stem}.4off",
            mime="text/plain",
            key=f"nc_download_{stem}",
        )


def render_nonconvex_tab() -> None:
    st.warning(
        "Beta. This is a separate generator: it does not change the Generate tab. "
        "Results are checked for closure only."
    )
    with st.expander("How to use this tab", expanded=False):
        st.markdown(NC_HELP)

    st.subheader("1. Symmetry and seed")
    symmetry_name = _nc_symmetry_picker()
    order = _group_order(symmetry_name)
    st.caption(f"`{symmetry_name}` · group order {order if order is not None else 'unknown'}")
    seed_text = st.text_input(
        "Seed point (x, y, z, w)",
        value="1, 0, 1, 0",
        key="nc_seed",
        help="Any point; the orbit is every image of it under the group. Pick a point on a "
        "mirror (for example a zero coordinate) for a more symmetric shape.",
    )
    try:
        seed = parse_seed(seed_text)
    except ValueError as err:
        st.error(str(err))
        return
    try:
        vertices = _cached_generate(tuple(float(x) for x in seed), symmetry_name, 1e-8, NC_MAX_VERTICES)
    except (RuntimeError, ValueError) as err:
        st.error(f"Could not generate the orbit (limit {NC_MAX_VERTICES} vertices): {err}")
        return
    if len(vertices) < 5:
        st.error("The orbit has fewer than 5 points; choose a seed with a larger orbit.")
        return
    st.caption(f"{len(vertices)} vertices in the orbit.")

    st.subheader("2. Edge orbits")
    with st.spinner("Classifying vertex pairs"):
        try:
            orbits, mirror_ids, rotation_ids = _nc_orbit_data(symmetry_name, vertices)
        except ValueError as err:
            st.error(str(err))
            return
    sources = {
        NC_EDGES_REFLECTION: mirror_ids,
        NC_EDGES_ROTATION: rotation_ids,
        NC_EDGES_ALL: list(range(len(orbits))),
    }
    available = [name for name, ids in sources.items() if ids]
    source = st.radio(
        "Edge candidates",
        available,
        horizontal=True,
        help="Mirror edges follow the Coxeter diagram. Rotation edges use powers of the group's "
        "generating rotations and work without mirrors. Every orbit lists all vertex pairs.",
        key=f"nc_source_{symmetry_name}",
    )
    if not mirror_ids:
        st.caption("This group contains no reflections, so mirror edges are not available.")
    candidates = sources[source]
    st.caption(f"{len(candidates)} of {len(orbits)} edge orbits shown, shortest first.")
    st.dataframe(
        pd.DataFrame(
            {
                "orbit": candidates,
                "edge length": [round(orbits[i].length, 4) for i in candidates],
                "edges": [len(orbits[i].edges) for i in candidates],
                "mirror": [i in mirror_ids for i in candidates],
                "rotation": [i in rotation_ids for i in candidates],
            }
        ),
        hide_index=True,
        height=min(35 * len(candidates) + 38, 280),
    )
    chosen = st.multiselect(
        "Edge orbits to use",
        candidates,
        format_func=lambda i: f"#{i} · length {orbits[i].length:.4f} · {len(orbits[i].edges)} edges",
        key=f"nc_orbits_{symmetry_name}_{source}",
        help="The polychoron's edges are the union of these orbits. Each vertex needs at "
        "least 4 edges in total.",
    )

    st.subheader("3. Faces and cells")
    face_rule = st.radio(
        "Face rule",
        [NC_FACES_VERTEX_FIGURE, NC_FACES_CORNERS],
        horizontal=True,
        help="A corner is two edges meeting at a vertex. Convex vertex figure: faces turn along "
        "the faces of the convex vertex figure. Corner classes: you choose which corners faces "
        "may use; faces can then be star polygons.",
        key="nc_face_rule",
    )
    corner_ids: list[int] = []
    classes = []
    if chosen:
        edges = tuple(sorted({e for i in chosen for e in orbits[i].edges}))
        if face_rule == NC_FACES_CORNERS:
            classes = _nc_corner_classes(symmetry_name, vertices, edges)
            if not classes:
                st.info("These edges form no corners at a vertex; choose different orbits.")
            corner_ids = st.multiselect(
                "Corner classes",
                list(range(len(classes))),
                default=list(range(len(classes))) if len(classes) <= 4 else [],
                format_func=lambda i: f"{np.degrees(classes[i].angle):.1f}° · {len(classes[i].pairs)} edge pair(s) per vertex",
                key=f"nc_corners_{symmetry_name}_{source}_{chosen}",
                help="Faces are simple planar polygons whose corners all belong to these classes. "
                "The smallest angle usually gives the triangles; larger angles give pentagons, "
                "pentagrams and so on.",
            )
    convex_cells = st.checkbox(
        "Only cells on a supporting hyperplane (convex cells)",
        value=face_rule == NC_FACES_VERTEX_FIGURE,
        key=f"nc_convex_cells_{face_rule}",
        help="Untick to allow cells whose hyperplane cuts through the polychoron, which star "
        "polychora need.",
    )
    cell_choice = st.number_input(
        "Cell set",
        min_value=0,
        value=0,
        step=1,
        help="When several sets of cells close up on the same faces, this picks one (0 is the "
        "set with the fewest symmetry classes of cells).",
        key="nc_cell_choice",
    )

    st.subheader("4. Build")
    ready = bool(chosen) and (face_rule == NC_FACES_VERTEX_FIGURE or bool(corner_ids))
    if st.button("Build polychoron", type="primary", disabled=not ready, key="nc_build"):
        with st.spinner("Building faces and cells"):
            poly = build_polychoron(
                vertices,
                named_symmetry(symmetry_name),
                list(chosen),
                orbits=orbits,
                convex=convex_cells,
                face_corners=corner_ids if face_rule == NC_FACES_CORNERS else None,
                cell_choice=int(cell_choice),
            )
        st.session_state["nc_built"] = (poly, f"{symmetry_name}_nonconvex")
    if not ready:
        st.caption("Choose at least one edge orbit (and, for star faces, at least one corner class).")
    if "nc_built" in st.session_state:
        poly, stem = st.session_state["nc_built"]
        _nc_show_result(poly, stem)

    with st.expander("Search for closed polychora automatically"):
        st.markdown(
            "Tries every union of up to *k* of the candidate edge orbits above and every "
            "choice of corner classes, and lists the ones that close up. Use *Mirror edges* "
            "or *Rotation edges* to keep the search short; it can take minutes."
        )
        if len(vertices) > NC_MAX_SEARCH_VERTICES:
            st.info(
                f"The search is limited to orbits of at most {NC_MAX_SEARCH_VERTICES} vertices "
                f"(this one has {len(vertices)}); choose the orbits by hand above."
            )
        elif source == NC_EDGES_ALL and len(orbits) > 30:
            st.info("There are too many edge orbits to search them all; choose mirror or rotation edges.")
        else:
            cols = st.columns(3)
            max_orbits = cols[0].slider("Orbits per shape", 1, 3, 2, key="nc_search_k")
            allow_star = cols[1].checkbox("Include star faces", value=True, key="nc_search_star")
            max_corners = cols[2].slider("Max corner classes", 1, 8, 6, key="nc_search_corners")
            if st.button("Search", key="nc_search"):
                with st.spinner("Searching"):
                    found = find_closed_polychora(
                        vertices,
                        named_symmetry(symmetry_name),
                        max_orbits=max_orbits,
                        reflections_only=source == NC_EDGES_REFLECTION,
                        rotations_only=source == NC_EDGES_ROTATION,
                        star=allow_star,
                        max_corner_classes=max_corners,
                    )
                st.session_state["nc_found"] = [(combo, poly) for combo, poly in found]
            found = st.session_state.get("nc_found")
            if found is not None:
                st.write(f"{len(found)} closed polychora found.")
                if found:
                    def label(k: int) -> str:
                        combo, poly = found[k]
                        lengths = ", ".join(f"{orbits[i].length:.3f}" for i in combo if i < len(orbits))
                        return f"orbits {list(combo)} (lengths {lengths}) · {len(poly.faces)} faces · {len(poly.cells)} cells"

                    pick = st.selectbox("Result", range(len(found)), format_func=label, key="nc_found_pick")
                    _nc_show_result(found[pick][1], f"{symmetry_name}_search_{pick}")


def render_domain_tab() -> None:
    render_page_tab(
        DOMAIN_PAGE,
        "Every isogonal polychoron of the swirlprism group H3●I2(10), mapped over a fundamental "
        "domain: hover a point for its shape and seed, and paste the seed into Generate's "
        "Coordinates mode.",
    )


def render_oct_tab() -> None:
    render_page_tab(
        OCT_PAGE,
        "The octahedral swirl group (index 6, 48 rotations) mapped over a third of a 16-cell's "
        "tetrahedron: its regions, the walls found so far with their exact equations, and the "
        "lines and points fixed by its rotations. A work in progress.",
    )


generate_tab, analyze_tab, domain_tab, oct_tab, nonconvex_tab = st.tabs(
    ["Generate", "Analyze a 4OFF file", DOMAIN_TITLE, OCT_TITLE, NONCONVEX_TITLE]
)
with generate_tab:
    render_generate_tab()
with analyze_tab:
    render_analyze_tab()
with domain_tab:
    render_domain_tab()
with oct_tab:
    render_oct_tab()
with nonconvex_tab:
    render_nonconvex_tab()
let_wheel_scroll_page()
