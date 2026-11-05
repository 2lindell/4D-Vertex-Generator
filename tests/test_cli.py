from __future__ import annotations

import sys

import pytest

from four_d_vertex_generator import cli


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["4d-vertex-generator", *argv])
    cli.main()


def test_unknown_symmetry_suggests_close_match(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            "--symmetry", "hyperoctaedral",
            "--seed", "1,0,0,0",
            "--out-off", str(tmp_path / "a.off"),
        )
    assert "Did you mean: hyperoctahedral" in capsys.readouterr().err


def test_bad_seed_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    with pytest.raises(SystemExit):
        _run(monkeypatch, "--symmetry", "f4", "--seed", "1,0", "--out-off", str(tmp_path / "a.off"))
    assert "exactly 4 comma-separated values" in capsys.readouterr().err


def test_hull_counts_are_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    _run(
        monkeypatch,
        "--symmetry", "f4", "--seed", "1,0,0,0", "--out-off", str(tmp_path / "a.off"), "--hull",
    )
    out = capsys.readouterr().out
    assert "num_vertices=24" in out
    assert "num_cells=24" in out
    assert (tmp_path / "a_24.off").exists()


def test_list_symmetries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(monkeypatch, "--list-symmetries")
    out = capsys.readouterr().out
    assert "h4\t(order 14400)" in out
    assert "a4_basic" not in out
    assert "pennic\t(order 120)" in out
    assert "ixitic\t(order 1440)" in out
    assert "glomic\t(continuous; sampled approximation)" in out


def test_continuous_symmetry_cli_writes_an_approximation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path
) -> None:
    _run(
        monkeypatch,
        "--symmetry", "glomic",
        "--seed", "1,0,0,0",
        "--out-off", str(tmp_path / "glome.off"),
        "--samples-per-angle", "4",
    )
    output = capsys.readouterr().out
    assert "num_vertices=40" in output
    assert "num_orbits=not applicable (continuous group approximation)" in output
    assert (tmp_path / "glome_40.off").read_text().splitlines()[1] == "40 0 0 0"
