"""Validate an emitted atlas against direct hull classifications and an independent survey.

Usage:
    SWIRL_GROUP=tet python validate.py out_tet.json dense_tet_60k.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import domain as D


def classify_chart_point(point: list[float]) -> str:
    return D.signature(D.classify(D.seed_from_chart(np.asarray(point, dtype=float))))


def validate(atlas_path: Path, dense_path: Path) -> None:
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    if atlas["group"] != D.NAME:
        raise ValueError(f"{atlas_path} is for {atlas['group']}, not {D.NAME}")

    known = {item["type"]: item["full"] for item in atlas["types"]}
    failures: list[str] = []
    checked = 0
    # Region examples are deliberately selected away from every wall and are
    # stable under the JSON rounding used by the page. Boundary representatives
    # are not: a rounded point on a wall can legitimately classify as either
    # adjacent volume type, so their metadata is checked for coverage below.
    for item in atlas["regions"]:
        expected = known.get(item.get("type"))
        actual = classify_chart_point(item["u"])
        checked += 1
        if expected != actual:
            failures.append(
                f"regions {item['id']}: expected {expected!r}, got {actual!r}"
            )
    represented = {
        item["type"]
        for collection in ("regions", "walls", "lines", "points")
        for item in atlas[collection]
        if item.get("type") is not None
    }
    unrepresented = set(known) - represented
    if unrepresented:
        failures.append(f"atlas types lack a representative: {sorted(unrepresented)!r}")
    observed = set(dense["sigs"])
    missing = observed - set(known.values())
    if missing:
        failures.append(
            f"dense survey contains {len(missing)} unrepresented type(s): {sorted(missing)!r}"
        )
    if failures:
        raise AssertionError("\n".join(failures))
    counts = Counter(dense["sigs"])
    print(
        f"validated {checked} atlas representatives; "
        f"{len(observed)} dense types across {sum(counts.values())} independent samples"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: validate.py ATLAS.json DENSE.json")
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
