from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import numpy as np

from .generation import generate_vertices_from_seed, group_order
from .isogonal import combined_matching_action, compute_orbits, detect_symmetries, split_by_orbits
from .library import available_symmetries, named_symmetry
from .off import compute_convex_hull, parse_4off, to_4off


def parse_seed(seed_text: str) -> np.ndarray:
    """Parse "x,y,z,w" into a 4-vector, raising ValueError with a readable message."""
    parts = [p.strip() for p in seed_text.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Seed must have exactly 4 comma-separated values (e.g. 1,0,0,0), got {len(parts)}"
        )
    try:
        return np.asarray([float(x) for x in parts], dtype=float)
    except ValueError as err:
        raise ValueError(f"Seed values must be numbers, got '{seed_text}'") from err


def _check_symmetry_name(parser: argparse.ArgumentParser, name: str) -> None:
    if name in available_symmetries():
        return
    close = difflib.get_close_matches(name, available_symmetries(), n=3)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    parser.error(f"unknown symmetry '{name}'.{hint} Use --list-symmetries to see all names.")


def _print_symmetries() -> None:
    for name in available_symmetries(include_aliases=False):
        order = group_order(named_symmetry(name))
        print(f"{name}\t(order {order if order is not None else '?'})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="4d-vertex-generator",
        description="Generate 4D vertices from symmetry + seed, group isogonally, and export 4OFF",
    )
    parser.add_argument(
        "--symmetry",
        metavar="NAME",
        help="Built-in symmetry name (see --list-symmetries)",
    )
    parser.add_argument(
        "--seed",
        help='Seed vertex as "x,y,z,w" (example: "1,0,0,0")',
    )
    parser.add_argument(
        "--out-off",
        type=Path,
        help="Output .off path; the vertex count is appended to the file name",
    )
    parser.add_argument(
        "--list-symmetries",
        action="store_true",
        help="List the built-in symmetry names with their group orders and exit",
    )
    parser.add_argument("--tol", type=float, default=1e-8, help="Quantization tolerance")
    parser.add_argument("--max-vertices", type=int, default=20000, help="Safety cap")
    parser.add_argument(
        "--hull",
        action="store_true",
        help="Compute 4D convex hull to include faces and cells in output .off",
    )
    args = parser.parse_args()

    if args.list_symmetries:
        _print_symmetries()
        return

    missing = [
        flag
        for flag, value in (
            ("--symmetry", args.symmetry),
            ("--seed", args.seed),
            ("--out-off", args.out_off),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    _check_symmetry_name(parser, args.symmetry)
    try:
        seed = parse_seed(args.seed)
    except ValueError as err:
        parser.error(str(err))

    action = named_symmetry(args.symmetry)
    try:
        vertices = generate_vertices_from_seed(
            seed,
            action,
            tol=args.tol,
            max_vertices=args.max_vertices,
        )
    except (ValueError, RuntimeError) as err:
        raise SystemExit(f"error: {err}") from err

    partition = compute_orbits(vertices, action, tol=args.tol)

    faces: list[list[int]] | None = None
    cells: list[list[int]] | None = None
    if args.hull:
        try:
            faces, cells = compute_convex_hull(vertices)
        except ValueError as err:
            raise SystemExit(f"error: could not compute convex hull: {err}") from err

    output_path = args.out_off.with_name(
        f"{args.out_off.stem}_{len(vertices)}{args.out_off.suffix or '.off'}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_4off(vertices, faces=faces, cells=cells))

    print(f"symmetry={args.symmetry}")
    print(f"seed={seed.tolist()}")
    print(f"num_vertices={len(vertices)}")
    print(f"num_orbits={partition.num_orbits}")
    if faces is not None and cells is not None:
        print(f"num_faces={len(faces)}")
        print(f"num_cells={len(cells)}")
    print(f"wrote_off={output_path}")


def analyze_main() -> None:
    parser = argparse.ArgumentParser(
        prog="4d-off-symmetry",
        description=(
            "Detect the symmetry of an existing 4D OFF file and optionally split its "
            "vertices into isogonal groups under a chosen subsymmetry."
        ),
    )
    parser.add_argument("--in-off", type=Path, required=True, help="Input .off file to analyze")
    parser.add_argument(
        "--symmetry",
        type=str,
        help=(
            "Subsymmetry to split vertices into isogonal orbits under. "
            "Use 'auto' to combine every detected symmetry into the largest "
            "matching group (fewest/largest possible orbits). Use "
            "'auto:<substring>' to restrict the search to detected symmetries "
            "whose name contains <substring> (e.g. 'auto:swirlprism' to "
            "find the largest matching swirl subgroup)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to write one split .off file per orbit (required with --symmetry)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6, help="Symmetry-matching / orbit tolerance"
    )
    args = parser.parse_args()

    if args.symmetry is not None and not (
        args.symmetry == "auto" or args.symmetry.startswith("auto:")
    ):
        _check_symmetry_name(parser, args.symmetry)

    try:
        vertices = parse_4off(args.in_off.read_text())
    except OSError as err:
        parser.error(f"cannot read {args.in_off}: {err.strerror}")
    except ValueError as err:
        raise SystemExit(f"error: {args.in_off} is not a valid 4D OFF file: {err}") from err
    print(f"num_vertices={len(vertices)}")

    # detect_symmetries sorts by descending group order, so the first entry
    # is the highest symmetry the vertex set actually satisfies.
    detected = detect_symmetries(vertices, tol=args.tol)
    if detected:
        summary = ", ".join(
            f"{name}({group_order(named_symmetry(name)) or '?'})" for name in detected
        )
        print(f"detected_symmetries={summary}")
        print(f"highest_symmetry={detected[0]}")
    else:
        print("detected_symmetries=none")

    if args.symmetry is None:
        return

    if args.symmetry == "auto" or args.symmetry.startswith("auto:"):
        family = args.symmetry.partition(":")[2] or None
        candidates = (
            [name for name in available_symmetries() if family in name] if family else None
        )
        matched = detect_symmetries(vertices, tol=args.tol, candidates=candidates)
        if not matched:
            scope = f" in family '{family}'" if family else ""
            raise SystemExit(f"--symmetry {args.symmetry} found no detected symmetry{scope}")
        action = combined_matching_action(vertices, tol=args.tol, candidates=candidates)
        symmetry_name = f"auto_{family}" if family else "auto"
        print(f"auto_symmetry_combines={', '.join(matched)}")
    else:
        if args.symmetry not in detected:
            print(
                f"warning: '{args.symmetry}' does not exactly match this vertex set "
                f"(tol={args.tol}); the split may be fragmented into more/smaller "
                "orbits than mathematically possible. Consider --symmetry auto or "
                "auto:<substring> instead."
            )
        symmetry_name = args.symmetry
        action = named_symmetry(symmetry_name)

    partition = compute_orbits(vertices, action, tol=args.tol)

    if partition.num_orbits == 1:
        print(f"already_isogonal_under={symmetry_name}")
        return

    if args.out_dir is None:
        raise SystemExit("--out-dir is required when splitting with --symmetry")

    groups = split_by_orbits(vertices, partition)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups):
        file_name = f"{args.in_off.stem}_{symmetry_name}_orbit{index}_{len(group)}.off"
        out_path = args.out_dir / file_name
        out_path.write_text(to_4off(group))
        print(f"wrote_off={out_path}")


if __name__ == "__main__":
    main()
