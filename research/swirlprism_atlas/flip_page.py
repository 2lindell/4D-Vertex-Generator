"""The built Symmetry Domain page's half-cell view mirrored and turned the way domain_page.view now draws it (FLIP: the
half b4 >= b3, as on the earlier page, with M34 at the back), without rebuilding it: every point of the half-cell view
(atlas-data and the region files in symmetry_domain_regions/cell) taken through FLIP, and each triangle's corners put in
the other order so the mirrored surfaces keep their outward side. Run once on a page built before FLIP.

    python flip_page.py ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from domain_page import CORNERS, FLIP, view

NOTE = (" The picture is drawn mirrored across that face, showing the half β₄ ≥ β₃ as on the earlier page, with M34 at "
        "the back; hover texts give each shape's β in the half β₃ ≥ β₄ (the same shape, mirror-imaged).")


def pt(p):
    q = FLIP @ np.asarray(p[:3], float)
    return [round(float(v), 4) for v in q] + list(p[3:])


def walk(o, key=None):
    """Every [x, y, z] (or [x, y, z, ...] for samples) in the data through FLIP; faces' corners reordered."""
    if key == "faces":
        return [[f[0], f[2], f[1]] for f in o]
    if isinstance(o, dict):
        return {k: (v if k in ("types", "camera", "bounds", "mirrorWalls", "regionWallsIds", "totalSamples")
                    else walk(v, k)) for k, v in o.items()}
    if isinstance(o, list):
        if len(o) >= 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in o[:3]) and \
                (len(o) == 3 or (len(o) == 4 and not isinstance(o[3], (int, float)))):
            return pt(o)
        return [walk(v, key) for v in o]
    return o


def main(page):
    html = open(page).read()
    m = re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', html, re.S)
    d = json.loads(m.group(1))
    d = walk(d)
    d["bounds"] = [round(float(view(CORNERS).min(0)[0]), 4), round(float(view(CORNERS).max(0)[0]), 4)]   # (as the builder writes it)
    html = html.replace(m.group(0), '<script id="atlas-data" type="application/json">'
                        + json.dumps(d, separators=(",", ":"), ensure_ascii=False) + "</script>")
    anchor = "so the other half repeats this one, and seeds in the splitting face are also drawn at their copy there."
    if anchor in html and NOTE not in html:
        html = html.replace(anchor, anchor + NOTE)
    open(page, "w").write(html)
    for f in sorted((Path(page).parent / "symmetry_domain_regions" / "cell").glob("*.json")):
        r = json.load(open(f))
        r["verts"] = [pt(v) for v in r["verts"]]
        r["faces"] = [[a, c, b] for a, b, c in r["faces"]]
        f.write_text(json.dumps(r, separators=(",", ":")))
    print(f"{page}: {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
