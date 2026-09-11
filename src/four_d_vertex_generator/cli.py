from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .isogonal import compute_orbits
from .symmetry import SymmetryAction


def _load_vertices(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    verts = np.asarray(data, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 4:
        raise ValueError("Vertex JSON must be an array of shape (n,4)")
    return verts


def _load_generators(path: Path) -> SymmetryAction:
    data = json.loads(path.read_text())
    return SymmetryAction.from_iterable(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="4d-vertex-generator",
        description="Enumerate isogonal groups of 4D vertices under symmetry actions",
    )
    parser.add_argument("--vertices", type=Path, required=True, help="Path to vertices JSON (n x 4)")
    parser.add_argument(
        "--generators",
        type=Path,
        required=True,
        help="Path to generator matrices JSON (k x 4 x 4)",
    )
    parser.add_argument("--tol", type=float, default=1e-8, help="Quantization tolerance")
    args = parser.parse_args()

    vertices = _load_vertices(args.vertices)
    action = _load_generators(args.generators)
    partition = compute_orbits(vertices, action, tol=args.tol)

    print(f"num_vertices={len(vertices)}")
    print(f"num_orbits={partition.num_orbits}")
    print("orbit_ids=" + json.dumps(partition.orbit_ids))


if __name__ == "__main__":
    main()
