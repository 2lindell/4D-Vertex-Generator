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
  dodecaswirlchoric symmetry families.

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
range. For dodecaswirlchoric symmetry, the UI also provides significant seeds,
cross-ring phase control, and the eight exact 120-point phases.

## Command-line options

| Option | Required | Description |
| --- | --- | --- |
| `--symmetry NAME` | Yes | A name returned by `available_symmetries()`. |
| `--seed x,y,z,w` | Yes | Four comma-separated coordinates. |
| `--out-off PATH` | Yes | Requested output path; the vertex count is added to its filename. |
| `--tol FLOAT` | No | Coordinate quantization tolerance; default `1e-8`. |
| `--max-vertices INT` | No | Generation safety cap; default `20000`. |

Use `--help` to see the argument parser's built-in help.

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
- **Dodecaswirlchoric:** `decafold_dodecaswirlchoric`, an active POV-Ray
  `+/-[I x C5]` convention with a group order of 600.

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
    compute_orbits,
    generate_vertices_from_seed,
    to_4off,
)
from four_d_vertex_generator.library import named_symmetry

action = named_symmetry("hyperoctahedral")
seed = np.array([1.0, 0.0, 0.0, 0.0])

vertices = generate_vertices_from_seed(seed, action)
partition = compute_orbits(vertices, action)

print(len(vertices), partition.num_orbits)
text = to_4off(vertices)
```

`generate_vertices_from_seed` performs a breadth-first closure over the action's
generators. It raises an error when the seed is not four-dimensional or when
the `max_vertices` cap is exceeded.

## 4OFF output

The exporter writes the following deliberately small format:

```text
4OFF
<number of vertices> 0 0
x y z w
...
```

Only vertices are emitted. Edges and faces are currently represented by zero
counts and are not generated. Numeric output is written with high precision
without exponent notation, which keeps the files friendly to simple parsers.

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
