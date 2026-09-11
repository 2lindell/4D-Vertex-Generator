from __future__ import annotations

from decimal import Decimal

import numpy as np


def _format_number(value: float) -> str:
    text = format(float(value), ".17g")
    if "e" in text.lower():
        text = format(Decimal(text), "f")
    return text


def to_4off(vertices: np.ndarray) -> str:
    """Serialize vertices to a 4D OFF-style text format.

    Format:
      4OFF
      <num_vertices> 0 0
      x y z w
      ...
    """
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 4:
        raise ValueError(f"Expected vertices shape (n,4), got {verts.shape}")

    lines = ["4OFF", f"{len(verts)} 0 0"]
    for v in verts:
        lines.append(" ".join(_format_number(value) for value in v))
    return "\n".join(lines) + "\n"
