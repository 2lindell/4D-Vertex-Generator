"""Merged piece counts on the whole 3-sphere: wall pieces (faces) and edges.

Edges are the atlas's line pieces (`atlas.json: line_pieces`, cut at their exact meeting points
and joined across copies) together with every wall-wall meeting curve (`wall_meets.json`) that
lies on no atlas line. A meeting curve is rebuilt exactly (traced on its two surfaces) between
the exact meeting points at its ends, and identified with the other records of the same curve by
its images under the group: two stretches are one edge class when some element carries one onto
the other. The number of pieces on the sphere is the number of distinct images of a class's
representative under all 1200 elements.

Wall pieces are the faces of `atlas.json` (a piece of one wall between two region volumes, joined
across copies), counted per wall and per pair of regions it separates.
"""
from __future__ import annotations

import json

import numpy as np

from .frame import data_dir, group, load_json, point_from_beta, source_dir
from .wall_meets import resolve, trace

ON_LINE = 1e-7          # (R^4) a meeting stretch lies on an atlas line if every point is this close
END = 1e-9              # named end points lie exactly on the curve


def _key(P):
    """An orientation-free key for a curve on S^3 (R^4 polyline): its ends and its middle."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    mid = P[int(np.searchsorted(d, d[-1] / 2))]
    ends = sorted(tuple(np.round(p, 5) + 0.0) for p in (P[0], P[-1]))
    return (ends[0], ends[1], tuple(np.round(mid, 3) + 0.0))


def _orbit_keys(P, g):
    imgs = np.einsum("gij,nj->gni", g, P)
    return {_key(Q) for Q in imgs}


def _named_points():
    pts = json.loads((source_dir() / "exact_points.json").read_text())["points"]
    return [(p["name"], np.asarray(p["float"], float)) for p in pts]


def _atlas_line_points():
    """Every atlas line's domain pieces, densely, in R^4."""
    w = load_json("working_data.json")
    out = []
    for name, info in w["exact"]["lines"].items():
        for poly in info["points"]:
            poly = np.asarray(poly, float)
            poly = poly / poly.sum(axis=1, keepdims=True)
            fine = [poly[0]]
            for a, b in zip(poly[:-1], poly[1:]):
                n = max(1, int(np.ceil(np.linalg.norm(b - a) / 1e-3)))
                fine += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
            out.append((name, point_from_beta(np.array(fine))))
    return out


def _stretch(rec, named):
    """The exact stretch of a meeting record between the named points at its ends (domain betas)."""
    a, b = resolve(rec["surfaces"][0]), resolve(rec["surfaces"][1])
    want = [rec["from"]["nearest_point"], rec["to"]["nearest_point"]]
    f0, t0 = np.asarray(rec["from"]["beta"]), np.asarray(rec["to"]["beta"])
    for arc in trace(a, b):
        P = arc["points"]
        P = P / P.sum(axis=1, keepdims=True)
        i0 = int(np.argmin(np.linalg.norm(P - f0, axis=1)))
        i1 = int(np.argmin(np.linalg.norm(P - t0, axis=1)))
        if np.linalg.norm(P[i0] - f0) > 1e-2 or np.linalg.norm(P[i1] - t0) > 1e-2:
            continue
        if i0 > i1:
            i0, i1, want = i1, i0, want[::-1]
        ends = []
        for i, nm, step in ((i0, want[0], -1), (i1, want[1], 1)):
            # walk outward from the checked stretch to the named point on the arc
            cands = [(np.linalg.norm(P - x, axis=1).min(), x) for n2, x in named if n2 == nm]
            cands = [c for c in cands if c[0] < 1e-6]
            if not cands:
                ends.append(None)
                continue
            x = min(cands, key=lambda c: c[0])[1]
            ends.append(int(np.argmin(np.linalg.norm(P - x, axis=1))))
        if None in ends:
            return None, want
        j0, j1 = sorted(ends)
        return P[j0:j1 + 1], want
    return None, want


def build() -> dict:
    g = group()
    named = _named_points()
    lines = _atlas_line_points()
    atlas = json.loads((data_dir() / "sphere_map" / "atlas.json").read_text())
    meets = json.loads((data_dir() / "sphere_map" / "wall_meets.json").read_text())["meetings"]

    # ---- meeting curves ------------------------------------------------------------------------
    classes = []            # {walls, ends, on_atlas_line, keys, rep, records}
    unresolved = []
    for m in meets:
        pair = tuple(sorted((m["wall"], m["other"])))
        for rec in m["found"]:
            P, ends = _stretch(rec, named)
            if P is None or len(P) < 3:
                unresolved.append({"walls": pair, "record": f"{m['wall']} {m['kind']} {m['other']}",
                                   "ends": ends})
                continue
            R = point_from_beta(P)
            on = None
            for name, L in lines:
                d = np.min(np.linalg.norm(R[:, None] - L[None], axis=2), axis=1)
                if d.max() < 5e-4:
                    on = name
                    break
            k = _key(R)
            for c in classes:
                if k in c["keys"]:
                    c["walls"].add(pair)
                    c["records"].add(f"{m['wall']} {m['kind']} {m['other']}")
                    break
            else:
                classes.append({"walls": {pair}, "ends": ends, "on_atlas_line": on,
                                "keys": _orbit_keys(R, g), "rep_beta": P,
                                "records": {f"{m['wall']} {m['kind']} {m['other']}"}})
    meeting_edges = []
    for c in classes:
        meeting_edges.append({
            "walls": sorted("/".join(p) for p in c["walls"]), "ends": c["ends"],
            "on_atlas_line": c["on_atlas_line"], "pieces_on_sphere": len(c["keys"]),
            "records": sorted(c["records"]),
            "representative_beta": [[round(float(t), 9) for t in x]
                                    for x in c["rep_beta"][:: max(1, len(c["rep_beta"]) // 20)]]})

    # ---- atlas line pieces ---------------------------------------------------------------------
    per_line = {}
    for lp in atlas["line_pieces"]:
        e = per_line.setdefault(lp["line_shape"], {"classes": 0, "pieces_on_sphere": 0,
                                                    "classes_detail": []})
        e["classes"] += 1
        e["pieces_on_sphere"] += lp["copies_on_sphere"]
        e["classes_detail"].append({"id": lp["id"], "ends": lp["end_points"],
                                    "on_sphere": lp["copies_on_sphere"]})

    # ---- new edges: meeting curves on no atlas line, per meeting ------------------------------
    new_edges = {}
    for e in meeting_edges:
        if e["on_atlas_line"]:
            continue
        for wp in e["walls"]:
            d = new_edges.setdefault(wp, {"classes": 0, "pieces_on_sphere": 0, "classes_detail": []})
            d["classes"] += 1
            d["pieces_on_sphere"] += e["pieces_on_sphere"]
            d["classes_detail"].append({"ends": e["ends"], "on_sphere": e["pieces_on_sphere"]})

    # ---- walls -----------------------------------------------------------------------------------
    walls = {w: {"face_types": v["piece_types"], "classes": v["face_classes"],
                 "faces_on_sphere": v["faces_on_sphere"],
                 "unplaced_fragments": v["unplaced_fragments"],
                 "by_regions": {"|".join(p["between"]): p["on_sphere"] for p in v["pieces"]}}
             for w, v in atlas["counts_per_wall"].items()}

    out = {"_about": __doc__,
           "walls": walls,
           "edges": {"atlas_lines": per_line, "wall_meetings_off_atlas_lines": new_edges},
           "wall_meeting_curves": meeting_edges,
           "unresolved_meeting_records": unresolved,
           "totals": {
               "wall_pieces_on_sphere": sum(v["faces_on_sphere"] for v in walls.values()),
               "atlas_line_pieces_on_sphere": sum(v["pieces_on_sphere"] for v in per_line.values()),
               "meeting_edge_pieces_on_sphere": sum(e["pieces_on_sphere"] for e in meeting_edges
                                                    if not e["on_atlas_line"]),
               "points_on_sphere": sum(p["copies_on_sphere"] for p in atlas["points"])}}
    (data_dir() / "sphere_map" / "merged_counts.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = build()
    for e in r["wall_meeting_curves"]:
        print(f"{','.join(e['walls']):18s} {e['ends'][0]:>11s} -> {e['ends'][1]:11s} "
              f"on line {str(e['on_atlas_line']):5s} x{e['pieces_on_sphere']}  {e['records']}")
    print("unresolved:", r["unresolved_meeting_records"])
    print(json.dumps(r["totals"], indent=1))
