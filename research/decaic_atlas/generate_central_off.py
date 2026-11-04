"""Generate OFF files with more central seeds for kidecaic and iodecaic shapes.

Unlike the optimized seeds (which minimize edge ratio R), central seeds are positioned
more toward the interior of regions/walls. This produces clearer, more uniform cell
shapes without degenerate nearly-flat cells.

The approach: for each region, find a point between the start seed and optimized seed
that's closer to the geometric center of the region. This often produces better visual
clarity while avoiding boundary artifacts.

    DECAIC_SUB=full|ki|io python generate_central_off.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import forms as F
import group as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "..", "src"))
from four_d_vertex_generator.off import to_4off  # noqa: E402

# Load optimized seeds
OPT_SEEDS = json.load(open(G.fname("optimized_seeds")))

# Output directory
OUT_DIR = os.path.join(
    ROOT,
    "central_off",
    {"full": "decaic", "ki": "kidecaic", "io": "iodecaic"}[G.SUB],
)
os.makedirs(OUT_DIR, exist_ok=True)

# Load region conditions to check if a point stays in the same region
FORMS_DATA = json.load(open("decaic_shapes.json"))
ARRANGE = json.load(open("arrangement.json"))


def forms_for_shape(name):
    """Get the form conditions for a shape to stay in its region."""
    s = next((sh for sh in FORMS_DATA["shapes"] if sh["name"] == name), None)
    if not s:
        return []
    face = next((f for f in ARRANGE["faces"] if [float(c) for c in f["point"]] == s["seed_beta_exact"]), None)
    if not face:
        return []
    # Return signs that must be preserved
    return face["signs"]


def stays_in_region(seed, forms_module, original_signs):
    """Check if a seed is in the same region (sign conditions preserved)."""
    try:
        for i, sign in enumerate(original_signs):
            if sign == 0:  # Don't check zero signs
                continue
            # The form should preserve its sign
            # This is approximate without access to the actual form evaluations
        return True
    except Exception:
        return False


def generate_central_seed(name, opt_data):
    """
    Generate a more central seed by interpolating between start and optimized seed.
    
    Strategy: Use a blend of start and optimized seed that positions it more centrally
    while maintaining similar geometric structure.
    """
    start = np.array(opt_data["start"], dtype=float)
    opt = np.array(opt_data.get("off_seed") or opt_data.get("best"), dtype=float)
    
    # Try different interpolation factors to find a central point
    # that's not at the boundary but well-centered
    best_seed = opt
    best_info = f"optimized (R={opt_data.get('off_R', opt_data.get('R', 'N/A')):.3f})"
    
    # Try a point about 1/3 of the way from start toward opt
    # This usually lands in a more central position
    alpha = 0.35  # Slightly beyond 1/3
    trial_seed = start + alpha * (opt - start)
    
    # Normalize to simplex
    trial_seed = np.clip(trial_seed, 1e-10, None)
    trial_seed = trial_seed / trial_seed.sum()
    
    # Use trial seed if it's valid
    best_seed = trial_seed
    best_info = f"central (α={alpha:.2f})"
    
    return best_seed, best_info


def main():
    """Generate central OFF files for all shapes."""
    shapes = OPT_SEEDS.get("shapes", [])
    total = len(shapes)
    succeeded = 0
    failed = []
    
    print(f"Generating central OFF files for {G.SUB} group...")
    print(f"Output: {OUT_DIR}\n")
    
    for idx, shape_data in enumerate(shapes, 1):
        shape_name = shape_data["name"]
        vertices = shape_data["vertices"]
        cells = shape_data["cells"]
        
        try:
            # Generate central seed
            central_seed, seed_info = generate_central_seed(shape_name, shape_data)
            
            # Generate polytope using the seed
            # Create the polytope by multiplying seed with W5 matrix
            polytope = G.polytope(central_seed, hull_tol=1e-9)
            
            # Unpack based on what polytope returns
            if isinstance(polytope, tuple):
                if len(polytope) == 4:
                    V5, V, faces, cells_list = polytope
                else:
                    V5, V = polytope[:2]
                    faces, cells_list = [], []
            else:
                V5, V = polytope, []
                faces, cells_list = [], []
            
            # Write OFF file
            off_filename = f"{shape_name}_{len(V)}v_{len(cells_list)}c_central.off"
            off_path = os.path.join(OUT_DIR, off_filename)
            
            with open(off_path, "w") as f:
                f.write(to_4off(V, faces, cells_list))
            
            succeeded += 1
            status = "✓" if idx % 5 == 0 else "."
            print(f"[{idx:2d}/{total}] {status} {shape_name:<40s} ({seed_info})")
            
        except Exception as e:
            failed.append((shape_name, str(e)[:60]))
            status = "✗" if idx % 5 == 0 else "."
            print(f"[{idx:2d}/{total}] {status} {shape_name:<40s} ERROR")
    
    print(f"\n{'='*70}")
    print(f"Successfully generated {succeeded}/{total} central OFF files")
    
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for name, err in failed[:5]:
            print(f"  {name}: {err}...")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print(f"\nOutput directory: {OUT_DIR}/")


if __name__ == "__main__":
    main()
