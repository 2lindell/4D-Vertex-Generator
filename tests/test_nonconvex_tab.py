from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def _metrics(at: AppTest) -> dict[str, str]:
    labels = {"Vertices", "Edges", "Faces", "Cells", "Euler χ"}
    return {m.label: m.value for m in at.metric if m.label in labels}


def test_a4_subgroup_picker_includes_iodecaic_alias() -> None:
    at = _app()
    subgroup_picker = next(widget for widget in at.selectbox if widget.label == "Subgroup")
    assert "Iodecaic [[3,3,3]+]" in subgroup_picker.options


def test_nonconvex_tab_builds_a_star_duoprism() -> None:
    at = _app()
    assert not at.exception
    assert "Non-convex generator (beta)" in [tab.label for tab in at.tabs]
    # the default group is the 5-5 duoprism, whose orbit of (1, 0, 1, 0) has 25 vertices
    at.multiselect(key="nc_orbits_duoprism_5_5_Mirror edges (Coxeter diagram)").set_value([0, 4]).run()
    at.radio(key="nc_face_rule").set_value("Chosen corner classes (allows star faces)").run()
    next(c for c in at.checkbox if c.label.startswith("Only cells")).set_value(False).run()
    next(b for b in at.button if b.key == "nc_build").click().run()
    assert not at.exception and not at.error
    assert _metrics(at) == {"Vertices": "25", "Edges": "50", "Faces": "35", "Cells": "10", "Euler χ": "0"}


def test_nonconvex_tab_search_lists_closed_polychora() -> None:
    at = _app()
    next(b for b in at.button if b.key == "nc_search").click().run()
    assert not at.exception
    assert any("4 closed polychora found" in m.value for m in at.markdown)


def test_nonconvex_tab_reports_a_bad_seed() -> None:
    at = _app()
    at.text_input(key="nc_seed").set_value("1,2,3").run()
    assert not at.exception
    assert any("exactly 4" in e.value for e in at.error)


def test_nonconvex_tab_accepts_tubical_polytwister_approximations() -> None:
    at = _app()
    at.selectbox(key="nc_family").set_value("h4_tubical").run()
    at.number_input(key="nc_tubical_n").set_value(8).run()
    assert not at.exception and not at.error
    assert any("h4_tubical_icosahedral_cyclic_8" in c.value for c in at.caption)
