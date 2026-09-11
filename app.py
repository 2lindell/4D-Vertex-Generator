from __future__ import annotations

import json

import numpy as np
import streamlit as st

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.isogonal import compute_orbits
from four_d_vertex_generator.library import (
    dodecaswirl_cross_ring_directions,
    dodecaswirl_cross_ring_seed,
    dodecaswirl_main_ring_direction,
    dodecaswirl_significant_seeds,
    dodecaswirl_special_phases,
    fundamental_chamber_roots,
    named_symmetry,
)
from four_d_vertex_generator.off import to_4off

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
            ("decafold_dodecaswirlchoric", "Decafold dodecaswirlchoric +/-[I x C5]"),
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
            ("b4", "Basic alias B4"),
            ("b4_basic", "Basic alias [4,3,3]"),
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
              <g stroke="#7c8798" stroke-width="1.5" fill="none">{lines}</g>
              <g fill="#7c8798" font-size="12" text-anchor="middle">
                <text x="170" y="15">alpha1</text>
                <text x="35" y="175">alpha2</text>
                <text x="305" y="175">alpha3</text>
                <text x="170" y="125">alpha4</text>
              </g>
              <circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="7" fill="#e4572e" />
            </svg>
            """,
            unsafe_allow_html=True,
        )
else:
    default_seed = "1,0,0,0"
    seed_text = st.text_input("Seed (x,y,z,w)", value=default_seed)

special_phase: float | str = "Use slider"
swirl_seed: np.ndarray | None = None
swirl_motion_active = False
if symmetry_name == "decafold_dodecaswirlchoric":
    significant_seeds = dodecaswirl_significant_seeds()
    selected_seed = st.selectbox(
        "Significant dodecaswirl seed",
        options=("Manual seed", *significant_seeds),
        help=(
            "The active POV-compatible group has order 600, with verified "
            "120-point vertex-ring and 600-point cross-ring strata."
        ),
    )
    if selected_seed != "Manual seed":
        seed_text = ",".join(
            f"{coordinate:.15g}" for coordinate in significant_seeds[selected_seed]
        )
        st.info(f"Seed: ({seed_text})")
    ring_phase = st.slider(
        "Cross-ring phase",
        min_value=0.0,
        max_value=360.0,
        value=0.0,
        step=1.0 if snap_to_significant else 0.1,
        help=(
            "Uses the POV-Ray crossringswirldoic formula; 360 degrees returns "
            "to the starting seed."
        ),
    )
    special_phase = st.selectbox(
        "Exact 120-point phase (optional)",
        options=("Use slider", *dodecaswirl_special_phases()),
        format_func=lambda phase: (
            phase if isinstance(phase, str) else f"{phase:.9f} degrees"
        ),
        help="These phases make five nearby points exactly coincide.",
    )
    if not isinstance(special_phase, str):
        ring_phase = special_phase
        st.info(f"Exact phase selected: {ring_phase:.12f} degrees")
        cross_ring_a, cross_ring_b = dodecaswirl_cross_ring_directions(ring_phase)
        main_ring_direction = dodecaswirl_main_ring_direction(ring_phase)
        st.subheader("Move away from the 120-point")
        st.caption(
            "The first two controls follow the two most similarly directed adjacent "
            "cross-rings. The third moves forward along the main ring."
        )
        motion_columns = st.columns(3)
        cross_ring_a_amount = motion_columns[0].slider(
            "Adjacent cross-ring A",
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05 if snap_to_significant else 0.01,
            format="%.2f rad",
        )
        cross_ring_b_amount = motion_columns[1].slider(
            "Adjacent cross-ring B",
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.05 if snap_to_significant else 0.01,
            format="%.2f rad",
        )
        main_ring_amount = motion_columns[2].slider(
            "Main ring toward next 120-point",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05 if snap_to_significant else 0.01,
            format="%.2f rad",
        )
        motion = (
            cross_ring_a_amount * cross_ring_a
            + cross_ring_b_amount * cross_ring_b
            + main_ring_amount * main_ring_direction
        )
        motion_size = float(np.linalg.norm(motion))
        if motion_size == 0.0:
            swirl_seed = dodecaswirl_cross_ring_seed(ring_phase)
        else:
            swirl_seed = (
                np.cos(motion_size) * dodecaswirl_cross_ring_seed(ring_phase)
                + np.sin(motion_size) * motion / motion_size
            )
            swirl_motion_active = True
tol = st.number_input("Tolerance", min_value=1e-12, max_value=1e-2, value=1e-8, format="%.1e")
max_vertices = st.number_input(
    "Max vertices",
    min_value=1,
    max_value=500000,
    value=20000,
    step=1000,
)
do_generate = st.button("Generate Vertices", type="primary")


def _parse_seed(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Seed must have exactly 4 comma-separated values")
    return np.asarray([float(p) for p in parts], dtype=float)


if do_generate:
    try:
        if symmetry_name == "decafold_dodecaswirlchoric":
            seed = swirl_seed if swirl_seed is not None else dodecaswirl_cross_ring_seed(ring_phase)
        elif chamber_roots is None:
            seed = _parse_seed(seed_text)
        if seed is None:
            st.stop()
        action = named_symmetry(symmetry_name)
        orbit_tol = (
            max(float(tol), 1e-7)
            if not isinstance(special_phase, str) and not swirl_motion_active
            else float(tol)
        )
        vertices = generate_vertices_from_seed(
            seed,
            action,
            tol=orbit_tol,
            max_vertices=int(max_vertices),
        )
        partition = compute_orbits(vertices, action, tol=orbit_tol)

        st.success("Generation complete")

        c1, c2 = st.columns(2)
        c1.metric("Vertices", int(len(vertices)))
        c2.metric("Isogonal groups (orbits)", int(partition.num_orbits))

        off_text = to_4off(vertices)
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
