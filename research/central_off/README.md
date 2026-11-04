# Central OFF Files

This directory contains 4D polytope OFF (Object File Format) files with **centrally-positioned seeds** rather than edge-ratio-optimized seeds.

## Overview

The original `optimized_off/` directory contains OFF files generated with seeds that minimize the edge ratio R (longest edge to shortest edge ratio). This produces the most "even" edge lengths but can place seeds near region boundaries, resulting in some nearly-degenerate (nearly-flat) cells.

The `central_off/` directory contains alternative OFF files where seeds are positioned **more toward the interior of their regions**, producing:
- More **clearly defined** cell shapes
- **Fewer degenerate** (nearly-flat) cells
- **Better visual clarity** of the polytope's unique structure

## Generation Method

For each shape, the central seed is computed by interpolation:
```
central_seed = start_seed + α × (optimized_seed - start_seed)
```

Where:
- `start_seed` = initial crude seed (centroid of region's boundary vertices)
- `optimized_seed` = edge-ratio-minimized seed
- `α = 0.35` = interpolation factor (~1/3 of the way from start to optimized)

This positioning places the seed ~1/3 of the distance toward the region's interior, balancing:
- **Proximity to region center** (better defined cells)
- **Avoiding degenerate behavior** (doesn't go too close to boundaries)

## Directory Structure

Each subdirectory contains OFF files for one symmetry group:

- **`decaic/`** (27 files) — Full 5-cell symmetry group (order 120)
- **`kidecaic/`** (32 files) — Chiral 5-cell subgroup (order 60)
- **`iodecaic/`** (7 files) — Rotoreflection 5-cell subgroup (order 60)
- **`contic/`** (27 files) — Full 600-cell symmetry group (order 1152)
- **`kicontic/`** (23 files) — Chiral 600-cell subgroup (order 576)
- **`iocontic/`** (6 files) — Rotoreflection 600-cell subgroup (order 576)

## File Naming

Each OFF file is named:
```
{shape_name}_{vertices}v_{cells}c_central.off
```

For example:
- `sabosid_240v_140c_central.off` — Shape "sabosid" with 240 vertices, 140 cells

## Comparison with Optimized Versions

| Aspect | `optimized_off/` | `central_off/` |
|--------|------------------|----------------|
| **Seed Position** | Minimizes edge ratio R | Interior-positioned |
| **Edge Uniformity** | Most uniform (minimized R) | Slightly less uniform |
| **Cell Definition** | Some degenerate cells | Clearer, uniform cells |
| **Visual Clarity** | Good for edge structure | Good for cell structure |
| **Use Case** | Edge analysis | Shape visualization |

## Generating Central OFF Files

To regenerate these files:

```bash
cd research/decaic_atlas
DECAIC_SUB=full python generate_central_off.py  # Full group
DECAIC_SUB=ki python generate_central_off.py    # Chiral subgroup
DECAIC_SUB=io python generate_central_off.py    # Rotoreflection subgroup

cd ../contic_atlas
CONTIC_SUB=full python generate_central_off.py  # Full group
CONTIC_SUB=ki python generate_central_off.py    # Chiral subgroup
CONTIC_SUB=io python generate_central_off.py    # Rotoreflection subgroup
```

## Mathematical Background

The seed β is a 4D barycentric coordinate (sum = 1) that determines the polytope's cell configuration through the orbit of `x = β × W5`, where W5 is the 5D root system matrix for the given group.

- **Optimized seed**: Minimizes max_edge / min_edge (edge ratio R)
- **Central seed**: Maximizes distance from region boundaries while staying in the region

The central positioning often reveals the shape's true geometric structure without edge-ratio artifacts.

## Research Context

These files were generated for research on 4D polytope visualization based on new investigations into kidecaic and iodecaic group structures. The central seed positioning provides clearer definitions of unique cell shapes within each polytope family.
