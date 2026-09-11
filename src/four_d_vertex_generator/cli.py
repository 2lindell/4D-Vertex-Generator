from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .generation import generate_vertices_from_seed
from .isogonal import compute_orbits
from .library import available_symmetries, named_symmetry
from .off import to_4off


def _parse_seed(seed_text: str) -> np.ndarray:
    parts = [p.strip() for p in seed_text.split(",")]
    if len(parts) != 4:
        raise ValueError("Seed must have exactly 4 comma-separated values, e.g. 1,0,0,0")
    vals = [float(x) for x in parts]
    return np.asarray(vals, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="4d-vertex-generator",
        description="Generate 4D vertices from symmetry + seed, group isogonally, and export 4OFF",
    )
    parser.add_argument(
        "--symmetry",
        type=str,
        required=True,
        choices=available_symmetries(),
        help="Built-in symmetry family",
    )
    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help='Seed vertex as "x,y,z,w" (example: "1,0,0,0")',
    )
    parser.add_argument("--out-off", type=Path, required=True, help="Output .off path")
    parser.add_argument("--tol", type=float, default=1e-8, help="Quantization tolerance")
    parser.add_argument("--max-vertices", type=int, default=20000, help="Safety cap")
    args = parser.parse_args()

    seed = _parse_seed(args.seed)
    action = named_symmetry(args.symmetry)

    vertices = generate_vertices_from_seed(
        seed,
        action,
        tol=args.tol,
        max_vertices=args.max_vertices,
    )

    partition = compute_orbits(vertices, action, tol=args.tol)

    output_path = args.out_off.with_name(
        f"{args.out_off.stem}_{len(vertices)}{args.out_off.suffix or '.off'}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_4off(vertices))

    print(f"symmetry={args.symmetry}")
    print(f"seed={seed.tolist()}")
    print(f"num_vertices={len(vertices)}")
    print(f"num_orbits={partition.num_orbits}")
    print(f"wrote_off={output_path}")


if __name__ == "__main__":
    main()
