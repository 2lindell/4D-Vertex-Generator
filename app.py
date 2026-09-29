from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

st.set_page_config(page_title="4D Vertex Generator", layout="wide")
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


def _describe(name: str) -> str:
    label = SYMMETRY_LABELS.get(name)
    return f"{label} — {name}" if label else name


@st.cache_data(show_spinner=False)
def _group_order(name: str) -> int | None:
    return group_order(named_symmetry(name))


@st.cache_data(show_spinner=False)
def _compatible_chamber(symmetry_name: str, candidates: tuple[str, ...]) -> str | None:
    action = named_symmetry(symmetry_name)
    for candidate in candidates:
        parent = named_symmetry(candidate)
        if is_subgroup(action, parent) or is_subgroup(parent, action):
            return candidate
    return None


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def _cached_convex_hull(vertices: np.ndarray):
    return compute_convex_hull(vertices)


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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

    st.plotly_chart(_local_view_figure(view, vertices, same_orbit), theme="streamlit")
    n_classes = len(set(edge_length_classes(view.edge_lengths).tolist()))
    summary = f"{len(view.neighbours)} edges meet at vertex {center}"
    if n_classes > 1:
        summary += f", with {n_classes} different lengths"
    if view.figure_edges:
        summary += "; the dashed outline is the vertex figure"
    st.caption(
        summary + ". Drag to rotate. The view looks along the vertex's radius, so every "
        "line is an edge direction as seen from the centre of the polytope."
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
    else:
        subgroup_choices = list(GROUP_OPTIONS[group_key][1])
        chamber_candidates = CHAMBER_CANDIDATES.get(group_key, ())

    choice_labels = dict(subgroup_choices)
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
                "Starts at a verified 120-point seed aligned with a 600-cell vertex. The first "
                "two sliders follow adjacent cross rings; the third follows the perpendicular "
                "main ring."
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
        try:
            with st.spinner("Generating vertices..."):
                vertices = _cached_generate(
                    tuple(float(x) for x in seed), symmetry_name, orbit_tol, int(max_vertices)
                )
            faces = cells = None
            hull_error = None
            if compute_hull:
                try:
                    with st.spinner("Computing 4D convex hull..."):
                        faces, cells = _cached_convex_hull(vertices)
                except ValueError as err:
                    hull_error = str(err)
            st.session_state["generated"] = {
                "symmetry": symmetry_name,
                "order": order,
                "seed": seed,
                "vertices": vertices,
                "faces": faces,
                "cells": cells,
                "hull_error": hull_error,
            }
        except (ValueError, RuntimeError) as err:
            st.session_state.pop("generated", None)
            st.error(str(err))

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
        st.warning("No built-in symmetry matched this vertex set at this tolerance.")

    auto = "auto (combine all matches)"
    split_options = [auto, *available_symmetries(include_aliases=False)]
    split_name = st.selectbox(
        "Split into isogonal groups under",
        options=split_options,
        index=split_options.index(detected[0]) if detected else 0,
        format_func=lambda name: name if name == auto else _describe(name),
        help=(
            "Defaults to the highest detected symmetry. 'auto' combines every matching "
            "symmetry into one larger group, which gives the fewest, largest orbits."
        ),
    )
    family_filter = ""
    if split_name == auto:
        family_filter = st.text_input(
            "Only combine symmetries whose name contains (optional)",
            help="e.g. 'swirlprism' to find the largest matching swirl subgroup.",
        ).strip()
    elif split_name not in detected:
        st.warning(
            f"`{split_name}` does not map this vertex set onto itself at this tolerance, "
            "so the split may be more fragmented than necessary."
        )

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


generate_tab, analyze_tab = st.tabs(["Generate", "Analyze a 4OFF file"])
with generate_tab:
    render_generate_tab()
with analyze_tab:
    render_analyze_tab()
