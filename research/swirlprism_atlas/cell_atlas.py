"""Build the half-cell atlas: one 600-cell tetrahedron, barycentric coordinates, golden samples."""
from __future__ import annotations

import json
import re
import sys

import numpy as np
from catalog import LABELS, match, parse_signature
from cellframe import TINV, golden_form, seed_from_beta

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

# display: the cell as a regular tetrahedron with the main-ring edge V3V4 vertical (x = -D) and the
# opposite edge V1V2 horizontal along y (x = +D). The half-turn (1<->2, 3<->4) is the rotation about
# x, and the splitting mirror beta3 = beta4 is the horizontal plane z = 0.
_s = 0.34
_d = _s / np.sqrt(2)
P = np.array([[_d, _s, 0.0], [_d, -_s, 0.0], [-_d, 0.0, _s], [-_d, 0.0, -_s]])


def to_upper(beta):
    """Move a seed into the displayed half-cell (beta3 >= beta4) with the exact half-turn."""
    b = np.asarray(beta, float)
    return b[[1, 0, 3, 2]] if b[2] < b[3] - 1e-12 else b


def clip_upper(poly):
    """Clip a convex polygon (list of 3D points) to z >= 0."""
    out = []
    for i in range(len(poly)):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % len(poly)])
        if a[2] >= -1e-12:
            out.append(a)
        if (a[2] >= -1e-12) != (b[2] >= -1e-12):
            t = a[2] / (a[2] - b[2])
            out.append(a + t * (b - a))
    return [np.round(x, 6).tolist() for x in out]
FULL = named_symmetry("h4_swirlprism")
ELEMENTS = group_elements(FULL)

NAMED_1200 = {  # counts-only signatures of named 1200-vertex shapes found at exact points
    "120+120+600|240+600+600+600+1200|600+600+600+600+1200": "Bi-hecatonicosadiminished truncated hexacosichoron",
    "120+120|240+600+600|600+600+1200": "Swirlprismatodiminished truncated hecatonicosachoron",
    "120+600+600|120+600+1200+1200+1200|600+600+600+600+600+1200": "Other half of the truncated 120-cell (not in the wiki list)",
}
ICOSAFOLD_RANGE = "Cross ring: range through the icosafold point (antiprisms split 120+120; the subsymmetrical icosafold icosidodecaswirlchoron sits at its midpoint)"


def xyz(beta):
    b = np.asarray(beta, float)
    return (b / b.sum()) @ P


def beta_text(beta):
    b = np.asarray(beta, float)
    nz = b[b > 1e-9]
    bb = b / nz.min()
    return "(" + ", ".join(golden_form(x) for x in bb) + ")"


def counts_key(sig):
    p = parse_signature(sig)
    return None if p is None else "|".join("+".join(map(str, x)) for x in p[1:])


def seed_text(beta):
    return ",".join(f"{c:.7f}" for c in seed_from_beta(beta))


def _split_upper(pts):
    """Break a polyline into the pieces with z >= 0."""
    runs, cur = [], []
    for p in pts:
        if p[2] >= -1e-9:
            cur.append(p)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def transitional_patches(samples, name_of, ids, t="T2"):
    """Outline of the transitional samples lying in each mirror or cell face (convex hull within the plane)."""
    import itertools

    from scipy.spatial import ConvexHull
    B = [np.array(s["beta"]) / sum(s["beta"]) for s in samples if ids[name_of(s["sig"])[0]] == t]
    planes = [(f"β{i + 1} = β{j + 1}", lambda b, i=i, j=j: abs(b[i] - b[j]) < 1e-9) for i, j in itertools.combinations(range(4), 2)]
    planes += [(f"β{i + 1} = 0", lambda b, i=i: b[i] < 1e-9) for i in range(4)]
    out = []
    for label, on in planes:
        X = np.array([xyz(b) for b in B if on(b)])
        if len(X) < 3:
            continue
        c = X.mean(0)
        _, sv, vt = np.linalg.svd(X - c)
        if sv[1] < 1e-9:
            continue
        uv = (X - c) @ vt[:2].T
        hull = ConvexHull(uv)
        poly = [np.round(X[k], 6).tolist() for k in hull.vertices]
        out.append({"id": t, "plane": label, "samples": int(len(X)), "pts": poly + [poly[0]]})
    print("transitional patches:", [(p["plane"], p["samples"]) for p in out], flush=True)
    return out


def load_tlines(path="tlines.json"):
    """Verified transitional segments from tlines.py, in display coordinates."""
    import os
    if not os.path.exists(path):
        return []
    return [{"id": L["id"], "pts": [np.round(xyz(to_upper(b)), 6).tolist() for b in L["beta"]]} for L in json.load(open(path))]


def fixed_circles(order):
    planes = []
    for e in ELEMENTS:
        if np.allclose(e, np.eye(4)):
            continue
        k = next(n for n in range(1, 61) if np.allclose(np.linalg.matrix_power(e, n), np.eye(4), atol=1e-7))
        if k != order:
            continue
        w, v = np.linalg.eigh((e + e.T) / 2)
        F = v[:, np.isclose(w, 1.0, atol=1e-7)].T
        if len(F) == 2 and not any(abs(abs(np.linalg.det(F @ Q.T)) - 1) < 1e-6 for Q in planes):
            planes.append(F)
    return planes


def clip_circle(F, samples=4000):
    """Points of a great circle inside the cell (all barycentric weights >= 0)."""
    out = []
    for s in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        p = np.cos(s) * F[0] + np.sin(s) * F[1]
        b = TINV @ p
        if np.all(b >= -1e-12) and b.sum() > 1e-9:
            out.append((p, b / b.sum()))
    return out


def main(samples_path, out_path):
    refs = json.load(open("references.json"))
    samples = json.load(open(samples_path))
    for smp in samples:
        smp["beta"] = to_upper(smp["beta"]).tolist()
    uniform = json.load(open("uniform_in_cell.json"))
    old = {}
    html = open("atlas.html").read()
    m = re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', html, re.S)
    if m:  # keep the dodecahedron atlas's U-numbers for continuity
        for k, t in json.loads(m.group(1))["types"].items():
            old["|".join((t["cells"], t["faces"], t["edges"]))] = k

    def name_of(sig):
        if sig.startswith("ERR"):
            return "boundary", "Degenerate sample (hull merge failed)", False
        if sig in refs:
            nm = refs[sig]
            if nm.startswith("Cross ring: antiprisms split"):
                nm = ICOSAFOLD_RANGE
            listed = "not in the wiki list" not in nm
            key = "R:" + nm
            return key, nm, listed
        w = match(sig)
        if w:
            return w, LABELS[w], True
        ck = counts_key(sig)
        if ck in NAMED_1200:
            nm = NAMED_1200[ck]
            return "N:" + nm, nm, "not in the wiki list" not in nm
        if ck in old and old[ck].startswith("U"):
            return old[ck], "Not in the wiki list (also found in the dodecahedron survey)", False
        return ck, "Not in the wiki list (new)", False

    types, members = {}, {}
    for s in samples:
        key, nm, listed = name_of(s["sig"])
        members.setdefault(key, []).append(s)
        if key not in types:
            v = s["sig"].split(":")[0] if not s["sig"].startswith("ERR") else "?"
            ck = counts_key(s["sig"]) or "||"
            c, f, e = ck.split("|")
            types[key] = {"name": nm, "listed": listed, "vertices": v, "cells": c, "faces": f, "edges": e}
    # stable short IDs: wiki keys keep theirs, others numbered by sample count
    order = sorted(types, key=lambda k: -len(members[k]))
    ids, n_new = {}, 0
    for k in order:
        if re.fullmatch(r"W\d|T\d|U\d+", k):
            ids[k] = k
        elif k.startswith("R:") or k.startswith("N:"):
            ids[k] = None
        elif k == "boundary":
            ids[k] = "deg"
        else:
            n_new += 1
            ids[k] = f"G{n_new}"
    n_named = 0
    for k in order:
        if ids[k] is None:
            n_named += 1
            ids[k] = f"S{n_named}"

    out_types, volume = {}, {}
    for k in order:
        t = types[k]
        mem = members[k]
        ex = min(mem, key=lambda s: sum(sum(g) for g in s["golden"]) if s.get("golden") else 99)
        out_types[ids[k]] = {**t, "samples": len(mem), "volumeSamples": len(mem), "planes": [],
                             "example": seed_text(ex["beta"]) + "  β ∝ " + beta_text(ex["beta"])}
        volume[ids[k]] = [[*np.round(xyz(s["beta"]), 6).tolist(), f"{seed_text(s['beta'])}  β ∝ {beta_text(s['beta'])}"] for s in mem]

    # uniform points: label with the uniform polytope and the type of that piece
    by_beta = {tuple(np.round(s["beta"], 9)): s for s in samples}
    points, point_types = [], {}
    seen_u = []
    for u in uniform:
        u = {**u, "beta": to_upper(u["beta"]).tolist()}
        key = tuple(np.round(np.array(u["beta"]) / sum(u["beta"]), 9))
        if key in seen_u:
            continue
        seen_u.append(key)
        s = by_beta.get(key)
        tname = ids[name_of(s["sig"])[0]] if s else "?"
        pid = u["uniform"]
        point_types[pid] = {"vertices": "", "name": f"Uniform {u['uniform']} ({u['rings']})"}
        points.append({"id": pid, "q": np.round(xyz(u["beta"]), 6).tolist(),
                       "seed": f"{seed_text(u['beta'])}  β ∝ {beta_text(u['beta'])}  → {tname}: {out_types.get(tname, {}).get('name', '?')}"})

    # rings through the cell
    sys.path.insert(0, ".")
    from lines import range_id, ring_parameter
    rings = []
    for F in fixed_circles(2):
        pts = clip_circle(F)
        run = None
        for p, b in pts:
            rid = range_id(ring_parameter(p))
            x = np.round(xyz(b), 6).tolist()
            if run is None or run["id"] != rid:
                if run is not None:
                    run["pts"].append(x)  # join runs so the ring is drawn continuously
                    rings.append(run)
                run = {"id": rid, "pts": []}
            run["pts"].append(x)
        if run is not None:
            rings.append(run)
    rings = [r for r in rings if len(r["pts"]) >= 2]
    rings = [{"id": r["id"], "pts": pts} for r in rings for pts in _split_upper(r["pts"])]
    main_segs = [[np.round(xyz(b), 6).tolist() for _, b in clip_circle(F)] for F in fixed_circles(5)]
    main_segs = [p for s in main_segs if len(s) > 1 for p in _split_upper(s)]

    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, c = P[i], P[j]
            if a[2] < -1e-12 and c[2] < -1e-12:
                continue
            if a[2] < -1e-12 or c[2] < -1e-12:                      # keep the part with z >= 0
                lo, hi = (a, c) if a[2] < c[2] else (c, a)
                a, c = lo + (hi - lo) * (-lo[2] / (hi[2] - lo[2])), hi
            edges.append([np.round(a, 6).tolist(), np.round(c, 6).tolist()])
    mid34 = ((P[2] + P[3]) / 2).tolist()
    split = [P[0].tolist(), P[1].tolist(), mid34]              # the mirror beta3 = beta4, i.e. z = 0
    half_split = [[P[0].tolist(), P[1].tolist()], [P[0].tolist(), mid34], [P[1].tolist(), mid34]]
    mirrors = []
    for i in range(4):
        for j in range(i + 1, 4):
            if {i, j} == {2, 3}:
                continue                                          # that one is the split itself
            k, m = [x for x in range(4) if x not in (i, j)]
            mid = ((P[i] + P[j]) / 2).tolist()
            poly = clip_upper([P[k].tolist(), P[m].tolist(), mid])
            if len(poly) >= 3:
                mirrors.append(poly + [poly[0]])

    ring_types = {
        "r600e": "Cross ring: 120+600+600+600+1200 cells (1200 3-valent edges)",
        "r600d": "Cross ring: 120+600+600+600+1200 cells (600 3-valent edges)",
        "r360": "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron",
        "rC": "Cross ring: 120+120+1200 cells (600+1200 4-valent edges)",
        "rCmid": ICOSAFOLD_RANGE,
        "rD": "Cross ring: tetrahedra split 600+600 (icosafold point at its midpoint; not in the wiki list)",
    }
    data = {
        "types": out_types, "volume": volume, "rings": rings, "main": main_segs,
        "ringTypes": {k: {"vertices": "600", "name": v} for k, v in ring_types.items()},
        "points": points, "pointTypes": point_types, "mirrors": mirrors, "curves": [],
        "cellEdges": edges + half_split, "split": split, "tlines": [L for L in load_tlines() if L["id"] == "T1"], "tpatches": transitional_patches(samples, name_of, ids), "boundarySamples": len(members.get("boundary", [])),
        "totalSamples": len(samples),
    }
    template = open("cell_atlas_template.html").read()
    open(out_path, "w").write(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"{len(samples)} samples, {len(out_types)} types ({sum(t['listed'] for t in out_types.values())} named or in the wiki list)")
    for k in sorted(out_types, key=lambda k: -out_types[k]["samples"]):
        t = out_types[k]
        print(f"  {k:<4} {t['samples']:>5}  {t['vertices']:>5}  {t['name'][:90]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
