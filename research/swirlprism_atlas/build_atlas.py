"""Turn survey + ring data into the embedded dataset for the atlas page."""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np
from catalog import LABELS, match, parse_signature
from survey import BASIS, NEIGHBOURS, A, stabilizer_q_maps


def cell_edges() -> list[list[list[float]]]:
    """Edges of the Voronoi dodecahedron in gnomonic coordinates."""
    import itertools
    normals = np.array([BASIS @ u for u in NEIGHBOURS])  # (B u) . q <= 1 - u . a
    offsets = 1.0 - NEIGHBOURS @ A
    verts, planes = [], []
    for i, j, k in itertools.combinations(range(len(normals)), 3):
        M = normals[[i, j, k]]
        if abs(np.linalg.det(M)) < 1e-9:
            continue
        v = np.linalg.solve(M, offsets[[i, j, k]])
        if np.all(normals @ v <= offsets + 1e-9) and not any(np.allclose(v, w) for w in verts):
            verts.append(v)
            planes.append({x for x in range(len(normals)) if abs(normals[x] @ v - offsets[x]) < 1e-9})
    edges = []
    for a, b in itertools.combinations(range(len(verts)), 2):
        if len(planes[a] & planes[b]) >= 2:
            edges.append([[round(float(c), 6) for c in verts[a]], [round(float(c), 6) for c in verts[b]]])
    return edges

RING_TYPES = {
    "r600e": ("600", "Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges)", True),
    "r600d": ("600", "Polychoron with 120+600+600+600+1200 cells (600 3-valent edges)", True),
    "r360": ("600", "Pentagonal-gyroprismatic triacosihexecontachoron", True),
    "rC": ("600", "Polychoron with 120+120+1200 cells (600+1200 4-valent edges)", True),
    "rCmid": ("600", "Not in the wiki list: 1200 tetrahedra + 240 antiprisms split 120+120 (through the icosafold point)", False),
    "rD": ("600", "Not in the wiki list: 240 antiprisms + 1200 tetrahedra split 600+600 (through the icosafold point)", False),
}
POINT_TYPES = {
    "p600": ("120", "Hexacosichoron (600-cell)"),
    "p120cell": ("600", "Hecatonicosachoron (120-cell)"),
    "prss": ("600", "Partially-rectified small swirlprism"),
    "psdr": ("600", "Swirlprismatodiminished rectified hexacosichoron"),
    "icosafold": ("600", "Subsymmetrical icosafold icosidodecaswirlchoron"),
    "bigyro": ("600", "Bigyroprismatic transitional didecafold icosidodecaswirlchoron"),
}


def seed_text(q: list[float]) -> str:
    x, y, z = q
    p = np.array([1.0, 0.850651 * y - 0.525731 * z, x, -0.525731 * y - 0.850651 * z])
    p /= np.linalg.norm(p)
    return ",".join(f"{c:.7f}" for c in p)


def example_seed(qs: list[list[float]]) -> str:
    """Seed of the sample nearest the middle of a type's samples."""
    pts = np.array(qs)
    centre = pts.mean(axis=0)
    return seed_text(list(pts[int(np.argmin(np.linalg.norm(pts - centre, axis=1)))]))


def main(survey_path: str, out_path: str, planes_path: str | None = None) -> None:
    survey = json.load(open(survey_path))
    planes = json.load(open(planes_path)) if planes_path else []
    for r in planes:
        r["plane"] = r.get("plane", "plane")
    survey = survey + planes
    lines = json.load(open("lines.json"))
    maps = stabilizer_q_maps()

    # 1200-vertex types: wiki matches keep their key, others numbered by region size
    sig_of = [r["sig"].split(" | val")[0] for r in survey]
    counts = Counter(s for s in sig_of if not s.startswith("ERR"))
    ids: dict[str, str] = {}
    unlisted = 0
    for sig, _ in counts.most_common():
        key = match(sig)
        if key is None:
            unlisted += 1
            key = f"U{unlisted}"
        ids[sig] = key

    types = {}
    for sig, key in ids.items():
        v, cells, faces, edges = parse_signature(sig)
        types[key] = {
            "vertices": v,
            "cells": "+".join(map(str, cells)),
            "faces": "+".join(map(str, faces)),
            "edges": "+".join(map(str, edges)),
            "name": LABELS.get(key, "Not in the wiki list"),
            "samples": counts[sig],
            "volumeSamples": sum(1 for r, s2 in zip(survey, sig_of) if s2 == sig and "plane" not in r),
            "planes": sorted({r["plane"] for r, s2 in zip(survey, sig_of) if s2 == sig and "plane" in r}),
            "listed": key in LABELS,
            "example": example_seed([r["q"] for r, s2 in zip(survey, sig_of) if s2 == sig]),
        }

    volume = {}
    for r, sig in zip(survey, sig_of):
        key = "boundary" if sig.startswith("ERR") else ids[sig]
        q = np.array(r["q"])
        imgs = {tuple(np.round(m @ q, 5)) for m in maps}
        for img in imgs:
            volume.setdefault(key, []).append([*img, seed_text(list(img))])

    rings = []
    for seg in lines["cross"]:
        # split each circle into runs of one range type
        run = None
        for pt in seg:
            if run is None or run["id"] != pt["id"]:
                if run is not None:
                    run["pts"].append(pt["q"])  # join runs so the line is continuous
                    rings.append(run)
                run = {"id": pt["id"], "pts": []}
            run["pts"].append(pt["q"])
        if run is not None:
            rings.append(run)

    data = {
        "types": types,
        "volume": volume,
        "rings": rings,
        "ringTypes": {k: {"vertices": v[0], "name": v[1]} for k, v in RING_TYPES.items()},
        "main": lines["main"],
        "points": [{**p, "seed": seed_text(p["q"])} for p in lines["points"]],
        "pointTypes": {k: {"vertices": v[0], "name": v[1]} for k, v in POINT_TYPES.items()},
        "cellEdges": cell_edges(),
        "boundarySamples": sum(1 for s in sig_of if s.startswith("ERR")),
        "totalSamples": len(survey),
    }
    template = open("atlas_template.html").read()
    open(out_path, "w").write(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"dodecahedron edges: {len(data['cellEdges'])}")
    print(f"types: {len(types)} ({sum(t['listed'] for t in types.values())} in the wiki list), "
          f"boundary samples: {data['boundarySamples']}, volume points: {sum(len(v) for v in volume.values())}")
    for k, t in sorted(types.items(), key=lambda kv: -kv[1]["samples"]):
        where = "volume" if t["volumeSamples"] else "plane " + ",".join(t["planes"])
        print(f"  {k:<4} {t['samples']:>5}  {where:<16} cells {t['cells']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
