# 4D Vertex Generator

Program to enumerate **isogonal groups of vertices** for a given 4D symmetry and export them to a 4D OFF-style file.

## Features

- Represent 4D symmetry actions with generator matrices.
- Generate full vertex sets from a **seed point** by repeated symmetry action.
- Partition vertices into **isogonal groups** (orbits).
- Export vertices to **4OFF** (`.off`) format.
- Run from CLI or from a local UI (Streamlit).

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Run the UI

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (typically `http://localhost:8501`).

In the UI:
1. Choose a main symmetry family from the dropdown
2. Choose a subgroup with the radio options
3. For duoprisms, choose `p` and `q` and then choose its subgroup
4. Enter seed point `(x, y, z, w)`
5. Click **Generate Vertices**
6. Download the generated `.off` file

## CLI usage

```bash
4d-vertex-generator \
  --symmetry hyperoctahedral \
  --seed "1,0,0,0" \
  --out-off out/vertices.off
```

Optional:

- `--tol 1e-8` quantization tolerance
- `--max-vertices 20000` safety cap for generation

## 4OFF notes

This project writes a 4D OFF-style file with:

- Header: `4OFF`
- Counts line: `<num_vertices> 0 0`
- Vertex lines: `x y z w`

(No edges/faces are emitted yet.)

## Built-in symmetries

- `identity`
- `coordinate_permutations`
- `hyperoctahedral` (signed permutations via swap + sign flips)
- `a4`, `b4`, `d4`, `f4`, and `h4` Coxeter symmetries, with `+` names for
  their basic chiral rotations
- `duoprism_p_q` for all `3 <= p <= q <= 6`, with `+` names for basic
  rotations and un-suffixed names for extended reflection symmetries
- `decafold_dodecaswirlchoric`, the cyclic tubical group
  `+/-[I x C20]` of order 1200 in this SO(4) representation

The UI groups these names by pentachoric, hexadecachoric, demitesseractic,
icositetrachoric, hexacosichoric, and duoprismatic families. Every displayed
subgroup resolves to a valid 4D action; subgroup names that describe the same
matrix subgroup in this implementation are retained as separate Coxeter
aliases for navigation and export naming.

Duoprism selections use the simplicial `[p,2,q]` fundamental chamber and its
four coordinate sliders. A regular `p`-by-`q` duoprism vertex is a special
equal-edge point in that chamber; a generic chamber point produces the full
symmetry orbit of that point and may have more than `p*q` vertices.

The A4 extended group has order 240 versus 120 for A4. B4 is already the full
hyperoctahedral reflection group of order 384, so its displayed extended alias
does not produce a larger linear group.

## Mathematical framing

Given symmetry group \(G\) acting on \(\mathbb{R}^4\), and a seed vertex \(v\),
we generate its orbit:

\[
\mathcal{O}(v) = \{ g(v) : g \in G \}
\]

Each orbit corresponds to one isogonal group.
