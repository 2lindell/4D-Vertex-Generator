"""Stress audit of compute_convex_hull: run it on many inputs and verify every result independently.

    python tools/hull_audit.py [--quick]

Each case ends in one of three ways:
  ok       - the hull passed every check in tests/hull_checks.py
  refused  - compute_convex_hull raised ValueError (it declines inputs it cannot resolve)
  WRONG    - it returned a hull that fails a check; this is the outcome that must never happen
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from hull_checks import check_hull  # noqa: E402

from four_d_vertex_generator.generation import generate_vertices_from_seed  # noqa: E402
from four_d_vertex_generator.library import (  # noqa: E402
    available_symmetries,
    fundamental_chamber_roots,
    named_symmetry,
)
from four_d_vertex_generator.off import compute_convex_hull  # noqa: E402


def run(name, verts, generators=None):
    try:
        faces, cells = compute_convex_hull(verts)
    except ValueError as err:
        return "refused", str(err)
    try:
        check_hull(verts, faces, cells, generators)
    except AssertionError as err:
        return "WRONG", str(err)
    return "ok", f"{len(faces)} faces, {len(cells)} cells"


def cases(quick):
    rng = np.random.default_rng(2026)
    names = available_symmetries(include_aliases=False)
    for name in names:
        action = named_symmetry(name)
        for k in range(1 if quick else 3):
            seed = rng.standard_normal(4)
            try:
                verts = generate_vertices_from_seed(seed, action, max_vertices=15000)
            except RuntimeError:
                continue
            if len(verts) < 5 or np.linalg.matrix_rank(verts - verts.mean(0), tol=1e-9) < 4:
                continue
            yield f"random seed {k} / {name}", verts, action.generators
    # seeds approaching a chamber wall: the hull degenerates as the distance goes to 0
    for name in ("a4", "b4", "f4", "h4"):
        roots = fundamental_chamber_roots(name)
        action = named_symmetry(name)
        for wall in range(4):
            for delta in (1e-2, 1e-4, 1e-6, 1e-8):
                weights = np.ones(4)
                weights[wall] = delta
                seed = np.linalg.solve(roots, weights)
                verts = generate_vertices_from_seed(seed / np.linalg.norm(seed), action)
                yield f"near wall {wall} (δ={delta:g}) / {name}", verts, action.generators
    # duplicated vertices and tiny jitter (as in a hand-made or exported OFF file)
    verts = generate_vertices_from_seed(rng.standard_normal(4), named_symmetry("b4"))
    yield "duplicates / b4", np.vstack([verts, verts[:10]]), None
    for jitter in (1e-12, 1e-9, 1e-6):
        noisy = verts + jitter * rng.standard_normal(verts.shape)
        yield f"jitter {jitter:g} / b4", noisy, None


def main():
    quick = "--quick" in sys.argv
    tally, wrong, t0 = Counter(), [], time.time()
    for label, verts, gens in cases(quick):
        outcome, detail = run(label, verts, gens)
        tally[outcome] += 1
        if outcome != "ok":
            print(f"{outcome:8s} {label}  ({len(verts)} vertices): {detail}")
        if outcome == "WRONG":
            wrong.append(label)
    summary = ", ".join(f"{v} {k}" for k, v in tally.items())
    print(f"\n{sum(tally.values())} cases in {time.time() - t0:.0f}s: {summary}")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
