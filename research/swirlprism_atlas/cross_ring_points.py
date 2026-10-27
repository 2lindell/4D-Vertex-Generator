"""The named points of the cross ring, exactly, as every image of them in the half-cell (cross_ring_points.json).

Along cross ring 1 (README: seeds cos t A + sin t C1) the shapes run A1 - D3 - C1 - C3 - C5 - D1 - C6 - C2a - C6 - D1 -
C5 - C3 - C1 - D3 - A1 - D2 - C4 - C2b - C4 - D2 - A1, the points at exact angles (tan t = 0, phi^-3, phi^-2, 1/2,
phi^-1, 2 phi^-2, 1, phi, phi^2, 2 phi^2, and the icosafold points at tan(1/2 arctan phi) and 45 deg beyond it). Each
point is carried by the whole group into the cell and written as betas in the half-cell: these are the exact places of
A1, C1, C4, C5, C6 and the icosafold points, and where the cross ring's lines (C2a, C2b, C3, D1, D2, D3) meet.

    python cross_ring_points.py
"""
import json

import numpy as np

from cellframe import beta_from_seed
from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import _h4_swirlprism_ring_basis, h4_swirlprism_anchor_seed, named_symmetry

PHI = (1 + 5 ** 0.5) / 2
AT = lambda x: float(np.degrees(np.arctan(x)))  # noqa: E731
POINTS = [  # (the exact angles along cross ring 1 where its shape changes, and the point there)
    (0.0, "p600"), (AT(PHI**-3), "p120cell"), (AT(PHI**-2), "prss"), (AT(0.5), "psdr"),
    (AT(PHI) / 2, "icosafold"), (AT(1 / PHI), "psdr"), (AT(2 * PHI**-2), "prss"), (45.0, "p120cell"),
    (AT(PHI), "p600"), (AT(PHI**2), "bigyro"), (45 + AT(PHI) / 2, "icosafold"), (AT(2 * PHI**2), "bigyro"),
]
ELEMENTS = group_elements(named_symmetry("h4_swirlprism"))
A = h4_swirlprism_anchor_seed()
C1 = _h4_swirlprism_ring_basis()[0]
import core as osd

NAMES = {"p600": "A1", "p120cell": "C1", "prss": "C5", "psdr": "C6", "bigyro": "C4", "icosafold": "icosafold point"}


def main():
    out = []
    for t, pid in POINTS:
        p = np.cos(np.radians(t)) * A + np.sin(np.radians(t)) * C1
        seen = []
        for g in ELEMENTS:
            b = beta_from_seed(g @ p)
            if not np.isfinite(b).all() or (b < -1e-12).any():
                continue
            b = osd.home(np.where(np.abs(b) < 1e-13, 0.0, b))
            if not any(np.allclose(b, s, atol=1e-10) for s in seen):
                seen.append(b)
        for b in seen:
            out.append({"name": NAMES[pid], "t": round(t, 6), "beta": b.round(15).tolist()})
    json.dump({"_about": __doc__.strip().split("\n\n")[0], "points": out}, open("cross_ring_points.json", "w"), indent=1)
    from collections import Counter
    print(Counter(p["name"] for p in out))


if __name__ == "__main__":
    main()
