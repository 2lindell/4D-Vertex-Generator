"""Render a self-contained interactive domain page from an emitted atlas JSON file.

Usage:
    python render_atlas.py out_tet.json ../../assets/tetraswirl_tet_domain.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


GROUP_LABELS = {
    "h4_tubical_tetrahedral_cyclic_index3_1": "Σ₁/₃[T × C₃] · order 24",
    "h4_tubical_octahedral_dihedral_index6_6n_1": "Σ₁/₆[O × D₆] · order 48",
}


def render(source: Path, destination: Path) -> None:
    atlas = json.loads(source.read_text(encoding="utf-8"))
    try:
        group = GROUP_LABELS[atlas["group"]]
    except KeyError as error:
        raise ValueError(f"No display label registered for {atlas['group']!r}") from error
    kind = "Tetrahedral" if atlas["key"] == "tet" else "Octahedral"
    title = f"{kind} swirl symmetry domain"
    description = (
        f"Every isogonal polychoron produced by the {group} tubical symmetry group, "
        "mapped over one fundamental domain."
    )
    template = Path(__file__).with_name("atlas_template.html").read_text(encoding="utf-8")
    page = (
        template.replace("__TITLE__", title)
        .replace("__GROUP__", group)
        .replace("__DESCRIPTION__", description)
        .replace("__ATLAS_DATA__", json.dumps(atlas, separators=(",", ":")))
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding="utf-8")
    print(f"wrote {destination} ({len(page):,} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_atlas.py INPUT.json OUTPUT.html")
    render(Path(sys.argv[1]), Path(sys.argv[2]))
