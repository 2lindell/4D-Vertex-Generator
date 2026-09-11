# 4D Vertex Generator

Program to enumerate **isogonal groups of vertices** for a given 4D symmetry and export them to a 4D OFF-style file.

## Features

- Represent 4D symmetry actions with generator matrices.
- Generate full vertex sets from a **seed point** by repeated symmetry action.
- Partition vertices into **isogonal groups** (orbits).
- Export vertices to **4OFF** (`.off`/`.4off`) format.
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
1. Choose a symmetry from the dropdown
2. Enter seed point `(x, y, z, w)`
3. Click **Generate Vertices**
4. Download the generated `.4off` file

## CLI usage

```bash
4d-vertex-generator \
  --symmetry hyperoctahedral \
  --seed "1,0,0,0" \
  --out-off out/vertices.4off
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

## Mathematical framing

Given symmetry group \(G\) acting on \(\mathbb{R}^4\), and a seed vertex \(v\),
we generate its orbit:

\[
\mathcal{O}(v) = \{ g(v) : g \in G \}
\]

Each orbit corresponds to one isogonal group.
