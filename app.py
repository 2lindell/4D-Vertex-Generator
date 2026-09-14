from __future__ import annotations

import json

import numpy as np
import streamlit as st

from four_d_vertex_generator.generation import generate_vertices_from_seed, group_order
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
from four_d_vertex_generator.off import compute_convex_hull, parse_4off, to_4off

st.set_page_config(page_title="4D Vertex Generator", layout="wide")
st.title("4D Vertex Generator")
st.caption(
    "Choose a symmetry, enter a seed point, generate all vertices in its orbit, and export 4OFF."
)

st.header("Parameters")
group_options = {
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
            ("a4_chiral_extended", "Chiral extended [[3,3,3]]+"),
            ("a4_extended_chiral", "Extended chiral [[3,3,3]+]"),
            ("a4_basic", "Basic alias [3,3,3]"),
            ("a4_chiral", "Chiral alias [3,3,3]+"),
        ),
    ),
    "b4": (
        "Hexadecachoric (B4, [4,3,3])",
        (
            ("hyperoctahedral", "Basic [4,3,3]"),
            ("b4+", "Chiral [4,3,3]+"),
            ("b4_ionic", "Ionic diminished [4,(3,3)+]"),
            ("b4_half", "Half [1+,4,3,3]"),
            ("b4_half_chiral", "Chiral half [1+,4,(3,3)+]"),
            ("b4_prismatic_octahedral", "Prismatic octahedral [4,3,2]"),
            ("b4_prismatic_octahedral_chiral", "Chiral prismatic octahedral [4,3,2]+"),
            ("b4_prismatic_tetrahedral", "Prismatic tetrahedral [3,3,2]"),
            ("b4_prismatic_tetrahedral_chiral", "Chiral prismatic tetrahedral [3,3,2]+"),
            ("b4", "Basic [4,3,3] (Coxeter basis)"),
            ("b4_basic", "Basic alias [4,3,3] (Coxeter basis)"),
            ("b4_chiral", "Chiral alias [4,3,3]+"),
            ("b4_extended", "Full reflection [4,3,3] (same group)"),
            ("b4_chiral_extended", "Chiral [4,3,3]+ (same group)"),
        ),
    ),
    "d4": (
        "Demitesseractic (D4, [3,3,1,1])",
        (
            ("d4", "Basic [3,3,1,1]"),
            ("d4+", "Chiral [3,3,1,1]+"),
            ("d4_extended", "Extended [3,4,3]"),
            ("d4_basic", "Basic alias [3,3,1,1]"),
            ("d4_chiral", "Chiral alias [3,3,1,1]+"),
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
            ("f4_basic", "Basic alias [3,4,3]"),
            ("f4_chiral", "Chiral alias [3,4,3]+"),
        ),
    ),
    "h4": (
        "Hexacosichoric (H4, [5,3,3])",
        (
            ("h4", "Basic [5,3,3]"),
            ("h4+", "Chiral [5,3,3]+"),
            ("h4_icosian", "Basic [5,3,3] (icosian/600-cell basis)"),
            ("h4_icosian+", "Chiral [5,3,3]+ (icosian/600-cell basis)"),
            ("h4_pentagonal_swirl", "Pentagonal swirl Z10xZ10 (icosian basis)"),
            ("h4_pentagonal_swirl_ring", "Pentagonal swirl ring Z10 (12 rings of 10)"),
            ("h4_swirlprism", "Small swirlprism [5,3:5] (order 1200)"),
            ("h4_swirlprism+", "Chiral small swirlprism [5,3:5]+ (order 600)"),
            ("h4_prismatic", "Prismatic [5,3,2]"),
            ("h4_prismatic_chiral", "Chiral prismatic [5,3,2]+"),
            ("h4_ionic", "Ionic diminished [(5,3)+,2]"),
            ("h4_half", "Half [5,3,1]"),
            ("h4_half_chiral", "Chiral half [5,3,1]+"),
            ("h4_basic", "Basic alias [5,3,3]"),
            ("h4_chiral", "Chiral alias [5,3,3]+"),
        ),
    ),
    "duoprism": ("Duoprismatic ([p,2,q])", ()),
}
group_keys = tuple(group_options)
group_key = st.selectbox(
    "Main symmetry group",
    options=group_keys,
    format_func=lambda key: group_options[key][0],
    index=1,
)

if group_key == "duoprism":
    duoprism_order = st.selectbox(
        "Duoprism order",
        options=[(p, q) for p in range(3, 7) for q in range(p, 7)],
        format_func=lambda order: f"{order[0]}-{order[1]} duoprism",
    )
    p, q = duoprism_order
    subgroup_choices = [
        (f"duoprism_{p}_{q}", f"Basic [${p},2,{q}$]"),
        (f"duoprism_{p}_{q}_chiral", f"Chiral [${p},2,{q}$]+"),
    ]
    if p == q:
        subgroup_choices.extend(
            (
                (f"duoprism_{p}_{q}_extended", f"Extended [[{p},2,{q}]]"),
                (f"duoprism_{p}_{q}_chiral_extended", f"Chiral extended [[{p},2,{q}]]+"),
            )
        )
else:
    subgroup_choices = list(group_options[group_key][1])

symmetry_name = st.radio(
    "Subgroup",
    options=[choice[0] for choice in subgroup_choices],
    format_func=dict(subgroup_choices).__getitem__,
    horizontal=True,
)

snap_to_significant = st.checkbox("Snap sliders to significant points", value=True)

chamber_name = "hyperoctahedral" if group_key == "b4" else group_key
if group_key == "duoprism":
    chamber_name = f"duoprism_{p}_{q}"
chamber_roots = fundamental_chamber_roots(chamber_name)
if chamber_roots is not None:
    st.subheader("Fundamental chamber")
    st.caption(
        "Choose nonnegative simple-root coordinates inside the spherical tetrahedral chamber."
    )
    coordinate_columns = st.columns(4)
    chamber_coordinates = np.array(
        [
            coordinate_columns[index].slider(
                f"Root {index + 1}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05 if snap_to_significant else 0.01,
            )
            for index in range(4)
        ]
    )
    chamber_sum = float(chamber_coordinates.sum())
    if chamber_sum == 0.0:
        st.error("Choose at least one positive chamber coordinate.")
        seed = None
    else:
        seed = np.linalg.solve(chamber_roots, chamber_coordinates)
        seed /= np.linalg.norm(seed)
        normalized_coordinates = chamber_coordinates / chamber_sum
        chamber_points = np.array([[170, 25], [35, 155], [305, 155], [170, 105]])
        point = normalized_coordinates @ chamber_points
        lines = " ".join(
            f"<line x1='{chamber_points[index, 0]}' y1='{chamber_points[index, 1]}' "
            f"x2='{chamber_points[other, 0]}' y2='{chamber_points[other, 1]}' />"
            for index in range(4)
            for other in range(index + 1, 4)
        )
        st.markdown(
            f"""
            <svg width="340" height="185" viewBox="0 0 340 185" role="img"
                 aria-label="Projection of the fundamental spherical tetrahedral chamber">
              </g>
              <circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="7" fill="#e4572e" />
            </svg>
            """,
            unsafe_allow_html=True,
        )
else:
    default_seed = "1,0,0,0"
    seed_text = st.text_input("Seed (x,y,z,w)", value=default_seed)

use_predefined_swirl_sliders = False
swirl_slider_values = (0.0, 0.0, 0.0)
if symmetry_name in ("h4_swirlprism", "h4_swirlprism+"):
    use_predefined_swirl_sliders = st.checkbox(
        "Use predefined ring sliders",
        help=(
            "Starts at a verified 120-point seed aligned with a 600-cell vertex. "
            "The first two sliders follow adjacent cross rings; the third follows "
            "the perpendicular main ring."
        ),
    )
    if use_predefined_swirl_sliders:
        slider_step = 1.0 if snap_to_significant else 0.1
        slider_columns = st.columns(3)
        swirl_slider_values = tuple(
            column.slider(
                label,
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=slider_step,
                help=help_text,
            )
            for column, label, help_text in zip(
                slider_columns,
                ("Cross ring 1", "Cross ring 2", "Main ring"),
                (
                    "Move around the first cross ring.",
                    "Move around the second cross ring.",
                    "Move around the perpendicular main ring.",
                ),
            )
        )
tol = st.number_input("Tolerance", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")
max_vertices = st.number_input(
    "Max vertices",
    min_value=1,
    max_value=500000,
    value=20000,
    step=1000,
)
compute_hull_option = st.checkbox("Compute 4D convex hull (include faces and cells)", value=True)
do_generate = st.button("Generate Vertices", type="primary")


def _parse_seed(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Seed must have exactly 4 comma-separated values")
    return np.asarray([float(p) for p in parts], dtype=float)


@st.cache_data(show_spinner=False)
def _cached_generate_and_orbits(
    seed_list: tuple[float, float, float, float],
    symmetry_name: str,
    orbit_tol: float,
    max_verts: int,
):
    seed_arr = np.array(seed_list, dtype=float)
    action = named_symmetry(symmetry_name)
    verts = generate_vertices_from_seed(
        seed_arr,
        action,
        tol=orbit_tol,
        max_vertices=max_verts,
    )
    part = compute_orbits(verts, action, tol=orbit_tol)
    return verts, part


@st.cache_data(show_spinner=False)
def _cached_convex_hull(verts: np.ndarray):
    return compute_convex_hull(verts)


if do_generate:
    try:
        if symmetry_name in ("h4_swirlprism", "h4_swirlprism+"):
            if use_predefined_swirl_sliders:
                seed = h4_swirlprism_predefined_seed(*swirl_slider_values)
            else:
                seed = _parse_seed(seed_text)
        elif chamber_roots is None:
            seed = _parse_seed(seed_text)
        if seed is None:
            st.stop()

        orbit_tol = (
            max(float(tol), 1e-6)
            if symmetry_name in ("h4_swirlprism", "h4_swirlprism+")
            and use_predefined_swirl_sliders
            else float(tol)
        )

        with st.spinner("Generating vertices..."):
            seed_tuple = tuple(float(x) for x in seed)
            vertices, partition = _cached_generate_and_orbits(
                seed_tuple,
                symmetry_name,
                orbit_tol,
                int(max_vertices),
            )

        st.success("Generation complete")

        if compute_hull_option:
            try:
                with st.spinner("Computing 4D convex hull..."):
                    faces, cells = _cached_convex_hull(vertices)
                off_text = to_4off(vertices, faces=faces, cells=cells)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Vertices", int(len(vertices)))
                c2.metric("Isogonal groups (orbits)", int(partition.num_orbits))
                c3.metric("Faces", int(len(faces)))
                c4.metric("Cells", int(len(cells)))
            except Exception as hull_exc:  # noqa: BLE001
                st.warning(f"Could not compute convex hull: {hull_exc}")
                off_text = to_4off(vertices)
                c1, c2 = st.columns(2)
                c1.metric("Vertices", int(len(vertices)))
                c2.metric("Isogonal groups (orbits)", int(partition.num_orbits))
        else:
            off_text = to_4off(vertices)
            c1, c2 = st.columns(2)
            c1.metric("Vertices", int(len(vertices)))
            c2.metric("Isogonal groups (orbits)", int(partition.num_orbits))

        st.download_button(
            "Download 4OFF",
            data=off_text,
            file_name=f"{symmetry_name}_vertices_{len(vertices)}.off",
            mime="text/plain",
        )

        st.subheader("Vertices (JSON preview)")
        st.code(json.dumps(vertices.tolist()[:200], indent=2), language="json")
        if len(vertices) > 200:
            st.info("Preview truncated to first 200 vertices.")

    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))


st.divider()
st.header("Analyze an uploaded 4D OFF file")
st.caption(
    "Upload a 4OFF file to detect its symmetry, and optionally split its vertices "
    "into isogonal groups under a subsymmetry of your choice."
)

uploaded_off_file = st.file_uploader("4D OFF file", type=["off", "4off", "txt"])
analyze_tol = st.number_input(
    "Symmetry tolerance",
    min_value=1e-12,
    max_value=1e-2,
    value=1e-6,
    format="%.1e",
    key="analyze_tol",
)


@st.cache_data(show_spinner=False)
def _cached_detect_symmetries(vertex_list: tuple[tuple[float, ...], ...], detect_tol: float):
    verts = np.array(vertex_list, dtype=float)
    return detect_symmetries(verts, tol=detect_tol)


if uploaded_off_file is not None:
    try:
        uploaded_vertices = parse_4off(uploaded_off_file.getvalue().decode("utf-8"))
        st.success(f"Parsed {len(uploaded_vertices)} vertices.")

        with st.spinner("Detecting symmetry..."):
            vertex_tuple = tuple(tuple(float(x) for x in v) for v in uploaded_vertices)
            detected_symmetries = _cached_detect_symmetries(vertex_tuple, float(analyze_tol))

        if detected_symmetries:
            # detect_symmetries is sorted by descending group order, so index 0
            # is the highest symmetry (fewest/largest orbits when splitting).
            st.write(f"Detected symmetries ({len(detected_symmetries)}), highest first:")
            summary = [
                f"{name} (order {group_order(named_symmetry(name)) or '?'})"
                for name in detected_symmetries
            ]
            st.code("\n".join(summary))
            highest_symmetry = detected_symmetries[0]
            st.info(f"Highest detected symmetry: **{highest_symmetry}**")
        else:
            st.warning("No built-in symmetry matched this vertex set at this tolerance.")
            highest_symmetry = None

        split_options = ["(none)", "auto (combine all matches)", *available_symmetries()]
        default_index = split_options.index(highest_symmetry) if highest_symmetry else 0
        split_symmetry_name = st.selectbox(
            "Split into isogonal groups under subsymmetry",
            options=split_options,
            index=default_index,
            help=(
                "Defaults to the highest detected symmetry. Use 'auto (combine "
                "all matches)' with an optional family filter below to combine "
                "every matching symmetry into the largest possible group -- "
                "combining several matching symmetries still preserves the "
                "vertex set, so this guarantees the fewest/largest orbits."
            ),
        )
        family_filter = ""
        if split_symmetry_name == "auto (combine all matches)":
            family_filter = st.text_input(
                "Restrict combination to symmetry names containing (optional)",
                value="",
                help="e.g. 'swirlprism' to find the largest matching swirl subgroup.",
            )

        if split_symmetry_name != "(none)" and st.button("Split vertices"):
            if split_symmetry_name == "auto (combine all matches)":
                candidates = (
                    [name for name in available_symmetries() if family_filter in name]
                    if family_filter
                    else None
                )
                matched = detect_symmetries(uploaded_vertices, tol=float(analyze_tol), candidates=candidates)
                if not matched:
                    st.error("No detected symmetry matches that family filter.")
                    st.stop()
                split_action = combined_matching_action(
                    uploaded_vertices, tol=float(analyze_tol), candidates=candidates
                )
                split_symmetry_name = "auto_" + (family_filter or "all")
                st.caption(f"Combined matching symmetries: {', '.join(matched)}")
            else:
                if split_symmetry_name not in detected_symmetries:
                    st.warning(
                        f"'{split_symmetry_name}' does not exactly match this vertex set at "
                        "this tolerance -- the split may be more fragmented than mathematically "
                        "possible. Consider 'auto (combine all matches)' instead."
                    )
                split_action = named_symmetry(split_symmetry_name)
            split_partition = compute_orbits(uploaded_vertices, split_action, tol=float(analyze_tol))

            if split_partition.num_orbits == 1:
                st.info(
                    f"These vertices are already isogonal under '{split_symmetry_name}' "
                    "(a single orbit)."
                )
            else:
                orbit_groups = split_by_orbits(uploaded_vertices, split_partition)
                st.success(
                    f"Split into {split_partition.num_orbits} isogonal groups "
                    f"under '{split_symmetry_name}'."
                )
                for orbit_index, orbit_vertices in enumerate(orbit_groups):
                    st.download_button(
                        f"Download orbit {orbit_index} ({len(orbit_vertices)} vertices)",
                        data=to_4off(orbit_vertices),
                        file_name=f"{split_symmetry_name}_orbit{orbit_index}_{len(orbit_vertices)}.off",
                        mime="text/plain",
                        key=f"orbit_download_{orbit_index}",
                    )

    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
