from __future__ import annotations

import json

import numpy as np
import streamlit as st

from four_d_vertex_generator.generation import generate_vertices_from_seed
from four_d_vertex_generator.isogonal import compute_orbits
from four_d_vertex_generator.library import available_symmetries, named_symmetry
from four_d_vertex_generator.off import to_4off

st.set_page_config(page_title="4D Vertex Generator", layout="wide")
st.title("4D Vertex Generator")
st.caption("Choose a symmetry, enter a seed point, generate all vertices in its orbit, and export 4OFF.")

with st.sidebar:
    st.header("Parameters")
    symmetry_name = st.selectbox("Symmetry", options=available_symmetries(), index=2)
    seed_text = st.text_input("Seed (x,y,z,w)", value="1,0,0,0")
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
        seed = _parse_seed(seed_text)
        action = named_symmetry(symmetry_name)
        vertices = generate_vertices_from_seed(
            seed,
            action,
            tol=float(tol),
            max_vertices=int(max_vertices),
        )
        partition = compute_orbits(vertices, action, tol=float(tol))

        st.success("Generation complete")

        c1, c2 = st.columns(2)
        c1.metric("Vertices", int(len(vertices)))
        c2.metric("Isogonal groups (orbits)", int(partition.num_orbits))

        st.subheader("Vertices (JSON preview)")
        st.code(json.dumps(vertices.tolist()[:200], indent=2), language="json")
        if len(vertices) > 200:
            st.info("Preview truncated to first 200 vertices.")

        off_text = to_4off(vertices)
        st.download_button(
            "Download 4OFF",
            data=off_text,
            file_name=f"{symmetry_name}_vertices.4off",
            mime="text/plain",
        )

    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
