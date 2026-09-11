# 4D Vertex Generator

Program to enumerate **isogonal groups of vertices** for a given 4D symmetry.

## Goals

- Represent 4D symmetry actions (initially as linear transforms).
- Apply symmetry actions to a seed vertex set.
- Partition vertices into **isogonal groups** (equivalence classes under the symmetry action, i.e. orbits).
- Provide a CLI for loading input data and printing/storing orbit partitions.

## Mathematical framing

Given:

- A finite set of vertices \(V \subset \mathbb{R}^4\)
- A symmetry group \(G\) acting on vertices (via matrices or permutations)

Two vertices \(v_i, v_j \in V\) are in the same isogonal group iff:

\[
\exists g \in G : g(v_i) = v_j
\]

The isogonal groups are therefore the **orbits** of the group action of \(G\) on \(V\).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Run the CLI:

```bash
4d-vertex-generator --help
```

Run tests:

```bash
pytest
```

## Roadmap

1. Accept symmetry as explicit 4x4 matrices (JSON).
2. Implement robust orbit detection with tolerance handling.
3. Add canonicalization to stabilize floating-point comparisons.
4. Support Coxeter/Wythoff-style inputs for common 4D uniform polytopes.
5. Export orbit partitions and statistics.
