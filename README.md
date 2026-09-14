# 4D Vertex Generator

Enumerate finite orbits of points under 4-dimensional symmetry groups, identify
their isogonal groups, and export the result as a `4OFF` file.

The generator works from a seed point and a small set of matrix generators. It
repeatedly applies those matrices until no new point is found, using a
configurable tolerance to merge numerically equivalent coordinates. The same
symmetry action can then partition the generated vertices into isogonal orbits.

## What it does

- Models 4D symmetries as real `4 x 4` generator matrices.
- Computes the complete orbit of a seed point by generator closure.
- Groups the resulting points into isogonal orbits.
- Exports coordinates in a simple 4D OFF-style format.
- Provides both a command-line interface and an interactive Streamlit UI.
- Includes Coxeter, chiral, diminished, prismatic, duoprismatic, and
  icosian/swirlprism symmetry families.

## Quick start

The project requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
```

Generate the signed-permutation orbit of the seed $(1, 0, 0, 0)$:

```bash
4d-vertex-generator \
  --symmetry hyperoctahedral \
  --seed "1,0,0,0" \
  --out-off out/vertices.off
```

The command prints the symmetry, vertex count, isogonal-group count, and the
actual output path. The filename receives the vertex count before its suffix;
the example therefore writes `out/vertices_8.off`.

## Interactive UI

Start the local Streamlit application from the repository root:

```bash
streamlit run app.py
```

The UI lets you choose a symmetry family and subgroup, set a seed or
fundamental-chamber coordinates, generate the orbit, inspect a JSON preview,
and download the result. For duoprisms, select `p` and `q` from the supported
range. For the `h4_swirlprism`/`h4_swirlprism+` symmetry, enter an open
four-coordinate seed or enable the predefined ring sliders. The three sliders
start at a verified 120-point seed aligned with a 600-cell vertex and move
along two cross rings and the perpendicular main ring.

The UI also lets you upload an existing 4D OFF file. It reports which
built-in symmetries the uploaded vertices are invariant under, and lets you
pick a subsymmetry to split the vertices into isogonal groups. If the
vertices are already a single orbit under that subsymmetry, it reports that
directly instead of splitting; otherwise it offers one downloadable `.off`
file per isogonal group.

## Command-line options

| Option | Required | Description |
| --- | --- | --- |
| `--symmetry NAME` | Yes | A name returned by `available_symmetries()`. |
| `--seed x,y,z,w` | Yes | Four comma-separated coordinates. |
| `--out-off PATH` | Yes | Requested output path; the vertex count is added to its filename. |
| `--hull` | No | Compute 4D convex hull to include faces and cells in output 4OFF. |
| `--tol FLOAT` | No | Coordinate quantization tolerance; default `1e-8`. |
| `--max-vertices INT` | No | Generation safety cap; default `20000`. |

Use `--help` to see the argument parser's built-in help.

## Analyzing an existing 4D OFF file

The `4d-off-symmetry` command detects which built-in symmetries a 4D OFF
file's vertices are invariant under, and can split those vertices into
isogonal groups under a chosen subsymmetry:

```bash
4d-off-symmetry --in-off out/vertices_8.off
```

Add `--symmetry NAME --out-dir DIR` to split the vertices into isogonal
groups under that subsymmetry, writing one `.off` file per orbit into `DIR`.
If the vertices are already a single orbit under that subsymmetry, the
command reports so instead of writing any files.

| Option | Required | Description |
| --- | --- | --- |
| `--in-off PATH` | Yes | Input `.off` file to analyze. |
| `--symmetry NAME` | No | Subsymmetry to split vertices into isogonal orbits under. |
| `--out-dir PATH` | With `--symmetry` | Directory to write one split `.off` file per orbit. |
| `--tol FLOAT` | No | Symmetry-matching / orbit tolerance; default `1e-6`. |



## Built-in symmetry families

The exact names accepted by the CLI are defined by `available_symmetries()`.
The main families are:

- **Elementary coordinate groups:** `identity`, coordinate permutations,
  cyclic and dihedral coordinate symmetries, and `global_inversion`.
- **Coxeter groups:** `a4`, `b4`/`hyperoctahedral`, `d4`, `f4`, and `h4`.
  A trailing `+` selects the orientation-preserving chiral generators where
  available.
- **Subgroups and extensions:** ionic, half, prismatic, diminished, and
  extended variants exposed by the UI and symmetry library.
- **Duoprisms:** `duoprism_p_q` and `duoprism_p_q+` for every
  `3 <= p <= q <= 6`. Equal-factor duoprisms also have factor-swap
  extensions.
- **Dodecaswirlchoric:** `h4_swirlprism`, the verified small-swirlprism
  `[5,3:5]` subgroup of H4 with group order 1200 (icosian/600-cell basis).
  Its `+` variant is the chiral subgroup with order 600. `h4_pentagonal_swirl`
  (order 50) and `h4_pentagonal_swirl_ring` (order 10) are related pentagonal
  swirl subgroups; the latter splits the 600-cell into its 12 rings of 10.

The built-in aliases are intentionally retained where several Coxeter names
describe the same matrix subgroup. They make the mathematical families easier
to navigate without changing the generated action.

### Fundamental chambers

The UI exposes four nonnegative simple-root coordinates for the spherical
chambers of the Coxeter and duoprism families. These coordinates are converted
to a normalized seed before generation. A regular `p`-by-`q` duoprism is one
special equal-edge point in the `[p,2,q]` chamber; a generic chamber point can
have more than `p*q` vertices.

## Python API

The core operations are importable without using the CLI:

```python
import numpy as np

from four_d_vertex_generator import (
    compute_convex_hull,
    compute_orbits,
    generate_vertices_from_seed,
    to_4off,
)
from four_d_vertex_generator.library import named_symmetry

action = named_symmetry("hyperoctahedral")
seed = np.array([1.0, 1.0, 1.0, 1.0])

vertices = generate_vertices_from_seed(seed, action)
partition = compute_orbits(vertices, action)

# Calculate 4D convex hull faces and cells
faces, cells = compute_convex_hull(vertices)

print(len(vertices), len(faces), len(cells))
text = to_4off(vertices, faces=faces, cells=cells)
# Or compute hull automatically: text = to_4off(vertices, compute_hull=True)
```

`generate_vertices_from_seed` performs a breadth-first closure over the action's
generators. It raises an error when the seed is not four-dimensional or when
the `max_vertices` cap is exceeded.

## 4OFF output

The exporter formats 4D polyhedra according to standard `4OFF` / Geomview nOFF conventions:

```text
4OFF
<num_vertices> <num_faces> 0 <num_cells>
x y z w
...
<nv> v0 v1 ...
...
<nf> f0 f1 ...
...
```

- **Header count line:** `<num_vertices> <num_faces> <num_edges> <num_cells>` where `<num_edges>` is `0` after faces and before cells (edges are omitted in standard 4OFF face/cell representations).
- **Vertices:** $N_V$ lines of four coordinates $x, y, z, w$.
- **Faces:** $N_F$ lines defining 2D polygonal faces. Each face starts with the number of vertices $N_v$ followed by the 0-based vertex indices ordered cyclically around the polygon.
- **Cells:** $N_C$ lines defining 3D polyhedral cells. Each cell starts with the number of faces $N_f$ followed by the 0-based face indices from the faces list.

If faces and cells are omitted, `to_4off` emits zero counts (`<num_vertices> 0 0 0`). When `compute_hull=True` or `compute_convex_hull` is called, the 4D convex hull is computed and the full mesh is exported. Numeric output is written with high precision without exponent notation.

## Mathematical model

For a symmetry group $G$ acting on $\mathbb{R}^4$ and a seed point $v$, the
generated vertex set is the orbit

$$
\mathcal{O}_G(v) = \{g v \mid g \in G\}.
$$

The isogonal partition records which generated vertices remain connected under
the same action. In exact arithmetic the orbit is finite for the built-in
finite groups; in floating-point arithmetic, `tol` controls when two computed
coordinates are treated as equal.

## Development

Install the development dependencies and run the test suite with:

```bash
python -m pip install -e '.[dev]'
pytest
ruff check .
```

The repository is released under the MIT License.
