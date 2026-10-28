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
    return {m.label: m.value for m in at.metric}


def test_nonconvex_tab_builds_a_star_duoprism() -> None:
    at = _app()
    assert not at.exception
    assert at.tabs[-1].label == "Non-convex generator (beta)"
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
