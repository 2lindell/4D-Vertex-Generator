"""The H3●I2(10) Symmetry Domain page: every isogonal polychoron of the swirlprism group, mapped over a fundamental
domain, with wiki-ordered labels (one letter per vertex count and named/unnamed; see README).

    python cell_atlas2.py golden_22.json ../../assets/symmetry_domain.html
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from catalog import match
from cell_atlas import (
    P,
    _split_upper,
    beta_text,
    clip_circle,
    clip_upper,
    counts_key,
    fixed_circles,
    fmt17,
    seed_text,
    to_upper,
    xyz,
)
from cellframe import TINV, golden_form, seed_from_beta, snap_golden
from classify import classify, signature

from four_d_vertex_generator.generation import group_elements
from four_d_vertex_generator.library import named_symmetry

# The wiki's list, in its own order. Each entry: (id, name, how to recognise it)
#   ("ref", name in references.json) - exact signature including edge valences
#   ("wiki", key in catalog.py)      - class counts of an unnamed wiki entry
#   ("counts", counts key)           - class counts of a named 1200-vertex entry
# C2 is the cross-ring range around each icosafold point (two ranges, told apart by their counts);
# the exact icosafold points themselves are not golden and are placed separately.
CANON = [
    ("A1", "Hexacosichoron (600-cell)", ("ref", "Hexacosichoron (600-cell)")),
    ("C1", "Hecatonicosachoron (120-cell)", ("ref", "Hecatonicosachoron (120-cell)")),
    ("B1", "Icosafold icosaswirlchoron", ("ref", "Icosafold icosaswirlchoron (main ring, 240)")),
    ("C2a", "Subsymmetrical icosafold icosidodecaswirlchoron", ("ref", "Cross ring: antiprisms split 120+120 (not in the wiki list)")),
    ("C2b", "Subsymmetrical icosafold icosidodecaswirlchoron", ("ref", "Cross ring: tetrahedra split 600+600 (not in the wiki list)")),
    ("C3", "Subsymmetrical pentagonal-gyroprismatic triacosihexecontachoron", ("ref", "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron")),
    ("D1", "Polychoron with 120+120+1200 cells (600+1200 4-valent edges)", ("ref", "Cross ring: 120+120+1200 cells (600+1200 4-valent edges)")),
    ("D2", "Polychoron with 120+600+600+600+1200 cells (600 3-valent edges)", ("ref", "Cross ring: 120+600+600+600+1200 cells (600 3-valent edges)")),
    ("D3", "Polychoron with 120+600+600+600+1200 cells (1200 3-valent edges)", ("ref", "Cross ring: 120+600+600+600+1200 cells (1200 3-valent edges)")),
    ("C4", "Bigyroprismatic transitional didecafold icosidodecaswirlchoron", ("ref", "Bigyroprismatic transitional didecafold icosidodecaswirlchoron")),
    ("C5", "Partially-rectified small swirlprism", ("ref", "Partially-rectified small swirlprism")),
    ("C6", "Swirlprismatodiminished rectified hexacosichoron", ("ref", "Swirlprismatodiminished rectified hexacosichoron")),
    ("F1", "Polychoron with 120+120+600+600+600+600+1200 cells", ("wiki", "W1")),
    ("F2", "Polychoron with 120+120+240+600+600+1200+1200 cells", ("wiki", "W2")),
    ("F3", "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+1200×9 faces)", ("wiki", "W3")),
    ("F4", "Polychoron with 120+120+600+600+600+600+1200+1200 cells (240+600+1200×9 faces)", ("wiki", "W4")),
    ("F5", "Polychoron with 120+120+600+600+600+600+1200+1200+1200 cells", ("wiki", "W5")),
    ("E1", "Bi-hecatonicosadiminished truncated hexacosichoron", ("counts", "120+120+600|240+600+600+600+1200|600+600+600+600+1200")),
    ("E2", "Swirlprismatodiminished truncated hecatonicosachoron", ("counts", "120+120|240+600+600|600+600+1200")),
    ("T1", "Transitional polychoron with 120+120+600+600 cells", ("wiki", "T1")),
    ("T2", "Transitional polychoron with 120+120+600+600+1200 cells", ("wiki", "T2")),
]
X_DESCRIPTIONS = {  # unlisted types with a known meaning
    "Cross ring: antiprisms split 120+120 (not in the wiki list)":
        "Cross-ring range around the icosafold point (antiprisms split 120+120)",
    "Cross ring: tetrahedra split 600+600 (not in the wiki list)":
        "Cross-ring range around the icosafold point (tetrahedra split 600+600)",
}
RING_RANGE_TO_REF = {
    "r600e": "Cross ring: 120+600+600+600+1200 cells (1200 3-valent edges)",
    "r600d": "Cross ring: 120+600+600+600+1200 cells (600 3-valent edges)",
    "r360": "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron",
    "rC": "Cross ring: 120+120+1200 cells (600+1200 4-valent edges)",
    "rCmid": "Cross ring: antiprisms split 120+120 (not in the wiki list)",
    "rD": "Cross ring: tetrahedra split 600+600 (not in the wiki list)",
}
# Marker shape encodes the vertex count; colour is unique within each vertex group, so shapes that
# share a region never share a style. Ring types (N3, N5, W1-W3) also all differ from each other.
STYLE = {
    "A1": ("--ink", "square"), "B1": ("--s1", "square"),
    "C1": ("--s7", "diamond"), "C2a": ("--s2", "diamond"), "C2b": ("--s9", "diamond"), "C3": ("--s3", "diamond"), "D1": ("--s4", "diamond"),
    "D2": ("--s5", "diamond"), "D3": ("--s6", "diamond"), "C4": ("--s1", "diamond"), "C5": ("--s8", "diamond"),
    "C6": ("--ink", "diamond"),
    "F1": ("--s1", "circle"), "F2": ("--s2", "circle"), "F3": ("--s3", "circle"), "F4": ("--s4", "circle"),
    "F5": ("--s5", "circle"), "E1": ("--s6", "circle"), "E2": ("--s7", "circle"), "T2": ("--s8", "circle"),
    "T1": ("--ink", "circle"),
}


# Transitional classes the line test placed on a golden line that only lies inside their wall: they live on curved
# walls (found exactly by Newton steps on the vertex-to-cell-hyperplane distance from the golden samples).
CURVED_WALLS = {
    "X10": "a curved wall (a quadric surface) with F2 on both sides; it contains the line the nudge test found",
    "X33": "a curved wall (a quartic surface) between X50 and X57; it contains the line the nudge test found",
}

def _rounded(o, n):
    """Round every float in a nested structure (done once, when the page is written)."""
    if isinstance(o, float):
        return round(o, n)
    if isinstance(o, (list, tuple)):
        return [_rounded(v, n) for v in o]
    if isinstance(o, dict):
        return {k: _rounded(v, n) for k, v in o.items()}
    if isinstance(o, np.generic):
        return _rounded(o.item(), n)
    if isinstance(o, np.ndarray):
        return _rounded(o.tolist(), n)
    return o


def _exact(a):
    """Coordinates are kept at full precision between views; the page rounds them once when it is written."""
    return np.asarray(a, float)


_F = (1 + 5 ** 0.5) / 2
# Exactly traced segments (barycentric end points). "copy" marks the image under the half-turn Q outside H4
# that normalizes the group (normalizer.py): same polytope, a different place in the cell.
EXACT_SEGMENTS = [
    {"id": "T1", "a": [1, 0, 1, 0], "b": [1, 0, 1, 1], "ends": ("C6", "C4"), "copy": False},
    {"id": "T1", "a": [0, 1, 1, 0], "b": [0, 1, 1, 1], "ends": ("C6", "C4"), "copy": False},
    {"id": "T1", "a": [1, 1 + _F, 1 + _F, 0], "b": [0, 2 + _F, 1, 1], "ends": ("C6", "C4"), "copy": True},
    {"id": "E2", "a": [1, 1, 1, 0], "b": [1, 1, 1, 1], "ends": ("C5", "C1"), "copy": False},
    {"id": "E2", "a": [2 + _F, 2 + _F, 1, 1], "b": [1 + _F, 1, 1, 0], "ends": ("C5", "C1"), "copy": True},
]


def fold_copies(beta):
    """The splitting mirror beta3 = beta4 is folded onto itself by the cell's half-turn (beta1 <-> beta2):
    a seed stored on it with beta1 != beta2 has an equivalent copy there that should also be drawn."""
    b = np.asarray(beta, float)
    if abs(b[2] - b[3]) < 1e-9 * b.sum() and abs(b[0] - b[1]) > 1e-9 * b.sum():
        return [b[[1, 0, 2, 3]]]
    return []


def ear_clip(C, normal):
    """Triangles (index triples) filling a simple polygon given by barycentric corners on a wall, by ear clipping
    in the wall's own plane, so concave patches (a curved edge bending inwards) are filled only inside."""
    C = [np.asarray(b, float) / np.sum(b) for b in C]
    keep = [0]
    for k in range(1, len(C)):
        if np.linalg.norm(C[k] - C[keep[-1]]) > 1e-12:
            keep.append(k)
    if len(keep) > 2 and np.linalg.norm(C[keep[0]] - C[keep[-1]]) < 1e-12:
        keep.pop()
    n = np.asarray(normal, float)
    basis = np.linalg.svd(np.vstack([n, np.ones(4)]))[2][2:]
    P = {k: C[k] @ basis.T for k in keep}
    area = sum(P[keep[i]][0] * P[keep[(i + 1) % len(keep)]][1] - P[keep[(i + 1) % len(keep)]][0] * P[keep[i]][1]
               for i in range(len(keep)))
    if area < 0:
        keep = keep[::-1]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= -1e-15 and cross(b, c, p) >= -1e-15 and cross(c, a, p) >= -1e-15
    poly, tris, guard = list(keep), [], 0
    while len(poly) > 3 and guard < 10000:
        guard += 1
        for i in range(len(poly)):
            a, b, c = poly[i - 1], poly[i], poly[(i + 1) % len(poly)]
            if cross(P[a], P[b], P[c]) <= 1e-15:
                continue                                  # reflex (or flat) corner: not an ear
            if any(inside(P[q], P[a], P[b], P[c]) for q in poly if q not in (a, b, c)):
                continue
            tris.append((a, b, c))
            poly.pop(i)
            break
        else:
            break                                         # numerical stall: fan the rest
    for i in range(1, len(poly) - 1):
        tris.append((poly[0], poly[i], poly[i + 1]))
    return tris


def flip_vertical(data):
    """Turn the picture upside down (z -> -z) so the splitting mirror is on top."""
    def f(q):
        return [q[0], q[1], -q[2]] + list(q[3:])
    data["samples"] = {k: [f(p) for p in v] for k, v in data["samples"].items()}
    for key in ("uniform", "special"):
        for u in data[key]:
            u["q"] = f(u["q"])
    for key in ("rings", "main", "segments", "tpatches", "qaxes", "regular", "xlines"):
        for r in data[key]:
            r["pts"] = [f(p) for p in r["pts"]]
    for w in data["xwalls"]:
        w["tris"] = [[f(p) for p in tri] for tri in w["tris"]]
    data["mirrors"] = [[f(p) for p in m] for m in data["mirrors"]]
    data["split"] = [f(p) for p in data["split"]]
    data["fcentre"] = f(data["fcentre"])
    for face in data["fdomain"]:
        face["pts"] = [f(p) for p in face["pts"]]
        face["axis"] = [f(p) for p in face["axis"]]
    data["edges"] = [[f(p) for p in e] for e in data["edges"]]


def load_refs():
    """Reference signatures, and the reverse map name -> signature."""
    refs = json.load(open("references.json"))          # signature (with valences) -> reference name
    ref_by_name = {v: k for k, v in refs.items()}
    # the icosafold midpoint shares its signature with the range around it; that signature is a range type
    ref_by_name["Cross ring: antiprisms split 120+120 (not in the wiki list)"] = next(
        k for k, v in refs.items() if v.startswith("Cross ring: antiprisms split"))
    return refs, ref_by_name


def identify(sig, refs=None):
    """Return (canonical id or None, x-key, reference name or None)."""
    refs = refs if refs is not None else load_refs()[0]
    if sig.startswith("ERR"):
        return None, "ERR", None
    ref = refs.get(sig)
    for cid, _, (kind, arg) in CANON:
        if kind == "ref" and ref == arg:
            return cid, None, ref
        if kind == "wiki" and match(sig) == arg:
            return cid, None, ref
        if kind == "counts" and counts_key(sig) == arg:
            return cid, None, ref
    return None, (sig if ref else counts_key(sig) + "|" + sig.split(" | val")[-1]), ref


def main(samples_path, out_path):
    refs, ref_by_name = load_refs()

    samples = json.load(open(samples_path))
    import os
    if os.path.exists("extra_samples.json"):        # shapes found by the denser random search (dense_search.py)
        samples += [{"beta": e["beta"], "golden": False, "sig": e["sig"]} for e in json.load(open("extra_samples.json"))]
    for s in samples:
        s["beta"] = to_upper(s["beta"]).tolist()
        s["id"], s["xkey"], s["ref"] = identify(s["sig"], refs)

    # uniform seeds (moved into the displayed half, de-duplicated)
    by_beta = {tuple(np.round(np.array(s["beta"]) / sum(s["beta"]), 9)): s for s in samples}
    uniform, seen = [], set()
    for u in json.load(open("uniform_in_cell.json")):
        b = to_upper(snap_golden(u["beta"]))
        key = tuple(np.round(b / b.sum(), 9))
        if key in seen:
            continue
        seen.add(key)
        uniform.append({"name": u["uniform"], "beta": b.tolist(), "sample": by_beta.get(key)})
        for c in fold_copies(b):                                # its half-turn copy in the splitting mirror
            ck = tuple(np.round(c / c.sum(), 9))
            if ck not in seen:
                seen.add(ck)
                uniform.append({"name": u["uniform"], "beta": c.tolist(), "sample": by_beta.get(key)})

    # unlisted types: X-numbered by vertex count, then number of cell classes, then total cells
    xkeys = {}
    for s in samples:
        if s["id"] is None and s["xkey"] != "ERR":
            xkeys.setdefault(s["xkey"], s)
    def xsort(k):
        s = xkeys[k]
        cells = [int(c) for c in counts_key(s["sig"]).split("|")[0].split("+")]
        return (int(s["sig"].split(":")[0]), len(cells), sum(cells), counts_key(s["sig"]))
    # unlisted: Y for 600 vertices, X for 1200 (and Z for any other count), each numbered from 1. The numbers in
    # atlas_xids_frozen.json are kept; shapes found later take the next free numbers, so no label ever moves.
    frozen = json.load(open("atlas_xids_frozen.json")) if os.path.exists("atlas_xids_frozen.json") else {}
    xid, counters = {}, {}
    for k, v in frozen.items():
        if k in xkeys:
            xid[k] = v
        counters[v[0]] = max(counters.get(v[0], 0), int(v[1:]))
    for k in sorted(xkeys, key=xsort):
        if k in xid:
            continue
        letter = {600: "Y", 1200: "X"}.get(int(xkeys[k]["sig"].split(":")[0]), "Z")
        counters[letter] = counters.get(letter, 0) + 1
        xid[k] = f"{letter}{counters[letter]}"
    for s in samples:
        if s["id"] is None and s["xkey"] in xid:
            s["id"] = xid[s["xkey"]]
    json.dump(xid, open("atlas_xids.json", "w"), indent=0)    # x-key -> X/Y id, for probe.py

    # uniform pieces of each type (used for naming X types found at uniform seeds)
    pieces = {}
    for u in uniform:
        if u["sample"]:
            pieces.setdefault(u["sample"]["id"], []).append(u["name"])

    types = []
    for i, (cid, name, _) in enumerate(CANON):
        types.append({"id": cid, "name": name, "listed": True, "color": STYLE[cid][0], "symbol": STYLE[cid][1]})
    for k in sorted(xkeys, key=lambda k: (xid[k][0], int(xid[k][1:]))):
        s = xkeys[k]
        ref = s["ref"]
        if ref in X_DESCRIPTIONS:
            name = X_DESCRIPTIONS[ref]
        elif pieces.get(xid[k]):
            name = "Piece of the " + ", ".join(sorted(set(pieces[xid[k]])))
        else:
            name = "Not in the wiki list"
        v = int(s["sig"].split(":")[0])
        types.append({"id": xid[k], "name": name, "listed": False, "color": "--x",
                      "symbol": {1200: "circle", 600: "diamond"}.get(v, "square")})
    tmap = {t["id"]: t for t in types}
    uniform_keys = {tuple(np.round(np.array(u["beta"]) / sum(u["beta"]), 9)) for u in uniform}

    samples_out = {}
    for s in samples:
        if s["id"] is None:
            continue
        t = tmap[s["id"]]
        key = tuple(np.round(np.array(s["beta"]) / sum(s["beta"]), 9))
        t["samples"] = t.get("samples", 0) + 1
        if "counts" not in t:
            c, f, e = counts_key(s["sig"]).split("|")
            t.update({"vertices": s["sig"].split(":")[0], "cells": c, "faces": f, "edges": e,
                      "example": f"{seed_text(s['beta'])}  β ∝ {beta_text(s['beta'])}",
                      "seed": seed_text(s["beta"]), "betatext": beta_text(s["beta"]), "counts": True})
        if key in uniform_keys:
            continue                                            # drawn once, as a uniform marker
        for b in [np.array(s["beta"]), *fold_copies(s["beta"])]:
            if tuple(np.round(b / b.sum(), 9)) in uniform_keys:
                continue
            samples_out.setdefault(s["id"], []).append(
                [*_exact(xyz(b)).tolist(), f"β ∝ {beta_text(b)}<br>seed {seed_text(b)}"])

    def label(tid):
        t = tmap[tid]
        return f"{tid} · {t['name']}"

    uniform_out = []
    for u in uniform:
        if not u["sample"]:
            continue
        tid = u["sample"]["id"]
        # confirmed isogonals are labelled by the shape alone; the uniform name only for general-space pieces
        text = label(tid) if tmap[tid]["listed"] else f"{tid} · piece of the {u['name']}"
        uniform_out.append({"id": tid, "q": _exact(xyz(u["beta"])).tolist(),
                            "hover": f"{text}<br>uniform seed, β ∝ {beta_text(u['beta'])}<br>seed {seed_text(u['beta'])}"})

    # C2: exact icosafold points (not golden), from the cross-ring analysis, placed in the displayed half
    phi = (1 + 5 ** 0.5) / 2
    half = float(np.degrees(np.arctan(phi))) / 2
    from survey import C1, A
    E = np.stack(group_elements(named_symmetry("h4_swirlprism")))
    special = []
    for tid, t_deg in (("C2a", half), ("C2b", 45 + half)):
        p = np.cos(np.radians(t_deg)) * A + np.sin(np.radians(t_deg)) * C1
        c, fc, e = counts_key(signature(classify(p))).split("|")
        tmap[tid]["exact"] = {"angle": round(t_deg, 4), "cells": c, "faces": fc, "edges": e}
        for x in E @ p:
            b = TINV @ x
            if np.all(b >= -1e-12) and b.sum() > 0:
                b = to_upper(b / b.sum())
                q = _exact(xyz(b)).tolist()
                if not any(np.allclose(q, o["q"], atol=1e-6) for o in special):
                    special.append({"id": tid, "q": q, "hover": f"{tid} · exact icosafold point (cross ring at {t_deg:.4f}°)<br>"
                                    f"cells {c}<br>faces {fc}<br>edges {e}<br>seed "
                                    + ", ".join(fmt17(c) for c in x)})
    for tid in ("C2a", "C2b"):
        n = sum(1 for s in special if s["id"] == tid)
        tmap[tid]["where"] = f"cross ring; {n} exact icosafold point{'s' if n > 1 else ''} (✕)"

    # C3: the ends of the order-3 girdle lie on the C3 cross-ring ranges; there the 600-vertex polytope
    # has 3600 symmetries - the pentagonal-gyroprismatic triacosihexecontachoron itself
    from supergroups import girdle_segments
    for end in (p for seg in girdle_segments() for p in seg):
        x0 = seed_from_beta(end)
        c, fc, e = counts_key(signature(classify(x0))).split("|")
        for x in E @ x0:
            b = TINV @ x
            if np.all(b >= -1e-12) and b.sum() > 0:
                b = to_upper(b / b.sum())
                q = _exact(xyz(b)).tolist()
                if not any(np.allclose(q, o["q"], atol=1e-6) for o in special):
                    special.append({"id": "C3", "q": q, "hover": "C3 · Pentagonal-gyroprismatic triacosihexecontachoron<br>"
                                    "exact point with 3600 symmetries: end of an order-3 ghost girdle on the cross ring<br>"
                                    f"cells {c}<br>faces {fc}<br>edges {e}<br>β = {beta_text(b)}<br>seed "
                                    + ", ".join(fmt17(v) for v in x)})
    n3 = sum(1 for o in special if o["id"] == "C3")
    tmap["C3"]["where"] = f"cross ring; {n3} exact points with 3600 symmetries (✕)"
    tmap["C3"]["exact"] = {"label": "the full pentagonal-gyroprismatic triacosihexecontachoron, 3600 symmetries",
                           "cells": c, "faces": fc, "edges": e}

    # where the axes of the extra half-turns meet the domain boundary: also ✕, in their shape's colour
    from normalizer import coset_axes
    from probe import _one
    axis_ends = []
    for a_end, b_end in coset_axes():
        for p in (a_end, b_end):
            if not any(np.allclose(p, o, atol=1e-9) for o in axis_ends):
                axis_ends.append(p)
    for p in axis_ends:
        q = _exact(xyz(p)).tolist()
        tid = _one(p)[0]
        note = "end of a 2400-symmetry axis on the domain boundary"
        known = [o for o in special if np.allclose(o["q"], q, atol=1e-6)]
        if known:
            for o in known:
                o["hover"] += f"<br>also the {note}"
            continue
        if tid not in tmap:
            continue
        x = seed_from_beta(p)
        special.append({"id": tid, "q": q, "hover": f"{label(tid)}<br>exact point: {note}<br>"
                        f"β = {np.round(p, 6).tolist()}<br>seed " + ", ".join(fmt17(c) for c in x)})
        tmap[tid]["axis_note"] = True

    # the point where the order-3 girdle crosses a 2400 axis: 7200 symmetries (girdle_meets_axis.py)
    from normalizer import coset_axes as _axes
    ga, gb = girdle_segments()[0]
    for c_end, d_end in _axes():
        M = np.column_stack([gb - ga, -(d_end - c_end)])
        st = np.linalg.lstsq(M, c_end - ga, rcond=None)[0]
        p = ga + st[0] * (gb - ga)
        if np.linalg.norm(p - (c_end + st[1] * (d_end - c_end))) < 1e-9 and -1e-9 <= st[0] <= 1 + 1e-9:
            p = p / p.sum()
            x = seed_from_beta(p)
            tid = _one(p)[0]
            c, fc, e = counts_key(signature(classify(x))).split("|")
            special.append({"id": tid, "q": _exact(xyz(p)).tolist(),
                            "hover": f"{label(tid)}<br>exact point with 7200 symmetries: the green order-3 girdle meets a purple "
                                     f"2400 axis<br>cells {c}<br>faces {fc}<br>edges {e}<br>β ∝ {beta_text(p)}<br>seed "
                                     + ", ".join(fmt17(v) for v in x)})
            tmap[tid]["where"] = (tmap[tid].get("where") or "golden samples") + "; ✕ 7200-symmetry point"

    # where an extra-symmetry axis passes through a transitional class (axis_crossings.py): ✕ in its colour
    if os.path.exists("axis_crossings.json"):
        for c in json.load(open("axis_crossings.json")):
            if not c.get("interior"):
                continue
            cid, xkey, _ = identify(c["sig"], refs)
            tid = cid or xid.get(xkey)
            if tid not in tmap:
                continue
            p = np.array(c["beta"])
            q = _exact(xyz(p / p.sum())).tolist()
            if any(np.allclose(q, o["q"], atol=1e-6) for o in special):
                continue
            x = seed_from_beta(p)
            special.append({"id": tid, "q": q, "hover": f"{label(tid)}<br>exact point where the {c['axis']} crosses it, between "
                                                        f"{c['below']} and {c['above']}; 2400 symmetries<br>β ∝ {beta_text(p)}"
                                                        "<br>seed " + ", ".join(fmt17(v) for v in x)})
            tmap[tid]["where"] = (tmap[tid].get("where") or "golden samples") + "; ✕ where a purple axis crosses it"

    # where each type lives (xloci.py): a region, a wall, a line, or a point / curve
    import os

    def _c(v):
        t = golden_form(v).replace("(1)/", "1/")
        return f"({t})" if any(ch in t[1:] for ch in "+-") and not t.startswith("(") else t

    def _eq(n):
        n = np.asarray(n, float)
        n = n / n[np.flatnonzero(np.abs(n) > 1e-9)[0]]
        pos = " + ".join(("" if abs(v - 1) < 1e-9 else _c(v) + "·") + f"β{i + 1}" for i, v in enumerate(n) if v > 1e-9)
        neg = " + ".join(("" if abs(v + 1) < 1e-9 else _c(-v) + "·") + f"β{i + 1}" for i, v in enumerate(n) if v < -1e-9)
        return f"{pos} = {neg or 0}"
    if os.path.exists("xloci_step1.json") and os.path.exists("xloci_step2.json"):
        r1, r2 = json.load(open("xloci_step1.json")), json.load(open("xloci_step2.json"))
        for tid, r in r1.items():
            if tid not in tmap:
                continue
            if r["kept"] == r["of"]:
                locus = "fills a region"
            elif tid in r2 and r2[tid]["kind"] in ("wall", "line"):
                locus = f"{r2[tid]['kind']}: " + ", ".join(_eq(n) for n in r2[tid]["normals"])
                if tid in CURVED_WALLS:
                    locus = CURVED_WALLS[tid]
            else:
                locus = "a point or a curve (not in a golden plane)"
            tmap[tid]["locus"] = locus
            if r["kept"] < r["of"] and tid.startswith("X"):
                tmap[tid]["transitional"] = True
        if os.path.exists("extra_samples.json"):     # regions found by the denser random search
            for e in json.load(open("extra_samples.json")):
                tid = xid.get(identify(e["sig"], refs)[1])
                if tid in tmap and tid not in r1:
                    tmap[tid]["locus"] = ("fills a region (found by the denser random search)" if e["kind"] == "region"
                                          else e["kind"] if "axis" in e["kind"] or e["kind"].startswith(("wall", "line"))
                                          else f"{e['kind']} (found by the denser random search)")
                    if e["kind"] != "region":
                        tmap[tid]["transitional"] = True

    # rings, coloured by the shape each range produces
    sys.path.insert(0, ".")
    from lines import range_id, ring_parameter
    ref_to_id = {}
    for rid, refname in RING_RANGE_TO_REF.items():
        sig = ref_by_name[refname]
        ref_to_id[rid] = identify(sig, refs)[0] or xid.get(sig) or xid.get(identify(sig, refs)[1])
    rings = []
    for F in fixed_circles(2):
        run = None
        for p, b in clip_circle(F):
            tid = ref_to_id[range_id(ring_parameter(p))]
            x = _exact(xyz(b)).tolist()
            if run is None or run["id"] != tid:
                if run is not None:
                    run["pts"].append(x)
                    rings.append(run)
                run = {"id": tid, "pts": []}
            run["pts"].append(x)
        if run is not None:
            rings.append(run)
    rings = [{"id": r["id"], "pts": pts} for r in rings if len(r["pts"]) >= 2 for pts in _split_upper(r["pts"])]
    main_ring = [{"id": "B1", "pts": pts} for F in fixed_circles(5)
                 for pts in _split_upper([_exact(xyz(b)).tolist() for _, b in clip_circle(F)])]
    for r in rings + main_ring:
        tmap[r["id"]].setdefault("where", "cross ring" if r["id"] != "B1" else "main ring")

    # transitional layer: T1 lines, T2 patches in mirrors and faces
    segments = []
    SELECTED_ONLY = {"T1"}    # drawn only while the shape's row is selected
    for seg in EXACT_SEGMENTS:
        a, b = np.array(seg["a"], float), np.array(seg["b"], float)
        a, b = a / a.sum(), b / b.sum()
        kind = "copy under the extra half-turn" if seg["copy"] else "traced"
        segments.append({"id": seg["id"], "copy": seg["copy"], "selOnly": seg["id"] in SELECTED_ONLY,
                         "pts": [_exact(xyz(a + (b - a) * k / 8)).tolist() for k in range(9)],
                         "hover": f"{seg['id']} · exact segment ({kind})<br>from {seg['ends'][0]} at β ∝ {beta_text(a)}"
                                  f"<br>to {seg['ends'][1]} at β ∝ {beta_text(b)}"})
    # regular cells: pentagonal prisms and antiprisms (exact lines, regular_cells.py)
    regular = []

    # Each line is a great-circle arc. Its pieces in the half-cell are the images g.arc (g in the group) clipped to
    # the half-cell, which is a fundamental domain, so lines that cross the fold come out whole and folded; the
    # dashed copies are the images g.Q.arc of the extra half-turn Q, clipped the same way. A copy piece lying on
    # an original line is that same line, so it is not drawn again.
    from dodeca_view import clip_images
    from normalizer import halfturn
    Qm = halfturn()
    H_cell = [TINV[i] for i in range(4)] + [TINV[2] - TINV[3]]       # beta_i >= 0 and beta3 >= beta4

    def arc(a, b, n=96, whole=False):
        if whole:                                    # the cells stay regular all the way round the great circle
            u = seed_from_beta(a)
            v = seed_from_beta(b)
            v = v - (v @ u) * u
            v /= np.linalg.norm(v)
            th = np.linspace(0, 2 * np.pi, 8 * n + 1)
            return np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
        return np.array([seed_from_beta(a + (b - a) * k / n) for k in range(n + 1)])

    def to_cell(piece):
        out = []
        for x in piece:
            bb = TINV @ x
            out.append(_exact(xyz(bb / bb.sum())).tolist())
        return out

    def on_lines(pts, lines_xyz, tol=1e-6):
        """Does every point of a piece lie on one of the given polylines?"""
        for p in pts[1:-1] or pts:
            p = np.array(p)
            ok = False
            for L in lines_xyz:
                L = np.array(L)
                for u, v in zip(L[:-1], L[1:]):
                    d = v - u
                    t = np.clip((p - u) @ d / max(d @ d, 1e-30), 0, 1)
                    if np.linalg.norm(u + t * d - p) < tol:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                return False
        return True

    def length(pts):
        return float(np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1).sum())

    def same(p, q):
        return len(p) == len(q) and (np.allclose(p, q, atol=1e-5) or np.allclose(p, q[::-1], atol=1e-5))

    def piece_hover(kind, pc, copy):
        """What lies along one piece: its shapes and its ends (beta, in this cell)."""
        B = [TINV @ x / (TINV @ x).sum() for x in pc]
        shapes = []
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            k = t * (len(B) - 1)
            i = int(k)
            p = B[i] + (k - i) * (B[min(i + 1, len(B) - 1)] - B[i])
            lab = _one(p)[0]
            if lab in tmap and lab not in shapes:
                shapes.append(lab)
        name = "prisms" if kind == "prism" else "antiprisms"
        return (f"Regular pentagonal {name}{' (copy under the extra half-turn)' if copy else ''}<br>shapes along this piece: "
                f"{', '.join(label(t) for t in shapes)}<br>from β ∝ {beta_text(np.clip(B[0], 0, None))}"
                f"<br>to β ∝ {beta_text(np.clip(B[-1], 0, None))}")

    def add_family(kind, specs):
        originals = []
        for a, b, whole in specs:
            for pc in clip_images(arc(a, b, whole=whole), H_cell):
                pts = to_cell(pc)
                if length(pts) < 1e-5 or on_lines(pts, originals, tol=1e-5):
                    continue                          # a corner, or the same line reached by another element
                originals.append(pts)
                regular.append({"kind": kind, "copy": False, "pts": pts, "hover": piece_hover(kind, pc, False)})
        copies = []
        for a, b, whole in specs:
            for pc in clip_images(arc(a, b, whole=whole) @ Qm.T, H_cell):
                pts = to_cell(pc)
                if length(pts) < 1e-5 or on_lines(pts, originals, tol=1e-5) or on_lines(pts, copies, tol=1e-5):
                    continue                          # on an original line: it is that line, not a separate copy
                copies.append(pts)
                regular.append({"kind": kind, "copy": True, "pts": pts, "hover": piece_hover(kind, pc, True)})

    # the lines, extended along their great circles as far as the cells stay regular (regular_cells.py extend_lines)
    import os
    ext = json.load(open("regular_lines.json")) if os.path.exists("regular_lines.json") else []
    fams = {"prism": [], "antiprism": []}
    def circle(a, b):
        u = seed_from_beta(a)
        v = seed_from_beta(b)
        v = v - (v @ u) * u
        v /= np.linalg.norm(v)
        return np.outer(u, u) + np.outer(v, v)
    kept = []
    for line in ext:
        whole = max(abs(t) for t in line["t"]) >= 2.99      # the extension ran to its limit: the whole circle
        a, b = np.array(line["a"]), np.array(line["b"])
        plane = circle(a, b)
        # a traced line that is the half-turn copy of one already kept is drawn as that line's dashed copy
        same = [Pk for k, Pk in kept if k == line["kind"]]
        if (not any(np.allclose(g @ Pk @ g.T, plane, atol=1e-7) for Pk in same for g in E)
                and any(np.allclose(g @ Qm @ Pk @ Qm.T @ g.T, plane, atol=1e-7) for Pk in same for g in E)):
            continue
        kept.append((line["kind"], plane))
        fams[line["kind"]].append((a, b, whole))
    add_family("prism", fams["prism"])
    add_family("antiprism", fams["antiprism"])

    # regular decagon faces (decagons.py): two regular pentagons merged into one plane, turned 36 degrees apart
    if os.path.exists("regular_decagons.json"):
        for st in json.load(open("regular_decagons.json")):
            pc = np.array(st["seeds"])
            B = [TINV @ x / (TINV @ x).sum() for x in pc]
            shapes = []
            for t in (0.1, 0.3, 0.5, 0.7, 0.9):
                lab = _one(B[int(t * (len(B) - 1))])[0]
                if lab in tmap and lab not in shapes:
                    shapes.append(lab)
            copy = bool(st.get("copy"))
            hover = (f"Regular decagon faces (two regular pentagons merged in one plane)"
                     f"{' (copy under the extra half-turn)' if copy else ''}<br>shapes along this piece: "
                     f"{', '.join(label(t) for t in shapes)}<br>from β ∝ {beta_text(np.clip(B[0], 0, None))}"
                     f"<br>to β ∝ {beta_text(np.clip(B[-1], 0, None))}")
            regular.append({"kind": "decagon", "copy": copy, "pts": to_cell(pc), "hover": hover})

    # transitional X types (xloci.py steps 3 and 4): their line segments and the patches they cover on walls
    xlines, xwalls = [], []
    wall_types = {t for t, r in (r2.items() if os.path.exists("xloci_step2.json") else []) if r["kind"] == "wall"}
    if os.path.exists("xloci_lines.json"):
        for tid, chords in json.load(open("xloci_lines.json")).items():
            if tid not in tmap or not tid.startswith("X"):
                continue
            for ch in chords.values():
                labs = ch.get("labels", [])
                a, b = (np.array(x) for x in ch["chord"])
                m = len(labs) - 1
                ends = {(e["k"], e["side"]): np.array(e["beta"]) for e in ch["ends"]}
                k = 0
                while k <= m:
                    if labs[k] != tid:
                        k += 1
                        continue
                    k0 = k
                    while k + 1 <= m and labs[k + 1] == tid:
                        k += 1
                    lo = ends.get((k0, -1), a + (b - a) * k0 / m)
                    hi = ends.get((k, 1), a + (b - a) * k / m)
                    if np.linalg.norm(hi - lo) > 1e-6:
                        pts = [to_upper(lo + (hi - lo) * j / 8) for j in range(9)]
                        xlines.append({"id": tid, "pts": [_exact(xyz(p)).tolist() for p in pts],
                                       "betas": [np.asarray(p, float).tolist() for p in pts],
                                       "hover": f"{label(tid)}<br>{tmap[tid].get('locus', '')}<br>from β ∝ {beta_text(to_upper(lo))}"
                                                f"<br>to β ∝ {beta_text(to_upper(hi))}"})
                    k += 1
    # X4 lives on the conic side of T2's patch A (in the mirror beta1 = beta3, from C4 to the cell centre C1): drawn
    # exactly along it (its copy comes from the step below)
    import t2exact
    if "X4" in tmap:
        A = [np.asarray(b, float) / np.sum(b) for b in t2exact.PATCHES["A"]()]
        ph = (1 + 5 ** 0.5) / 2
        conic = [b for b in A if abs(b[0] ** 2 + ph ** 2 * b[1] ** 2 - b[0] * b[1] - b[0] * b[3] - ph * b[1] * b[3]) < 1e-9]
        if len(conic) >= 3:
            tmap["X4"]["locus"] = "a curve: the conic side of T2's patch in the mirror β1 = β3, from C4 to the cell centre C1"
            tmap["X4"]["extra_bounds"] = ["C1", "C4"]
            tmap["T2"]["extra_bounds"] = ["X4"]
            xlines.append({"id": "X4", "pts": [_exact(xyz(b)).tolist() for b in conic], "betas": [b.tolist() for b in conic],
                           "hover": f"{label('X4')}<br>the conic side of T2's patch in the mirror β1 = β3<br>"
                                    f"from β ∝ {beta_text(conic[0])}<br>to β ∝ {beta_text(conic[-1])}"})

    # X26 is a wall: the quadrilateral C4, M34, C4', C1 in the splitting mirror beta3 = beta4 (bounded by D2 and X2's
    # lines, creased by D3 along M34-C1) and its copy, a triangle in beta3 = beta2 + beta4. Where that copy crosses the
    # mirror beta2 = beta4 (from A1 to the corner of the X31 and X44 conics) it creases X79's lens.
    if "X26" in tmap:
        tmap["X26"]["locus"] = ("wall: the mirror β3 = β4 between the D2 and X2 lines, creased by D3; its copy lies in "
                                "β3 = β2 + β4, crossing X79's lens")

    # half-turn copies of the survey lines: each segment's images under the 2400-element group that land in the
    # half-cell, other than the segments already found, are drawn dashed
    from cellframe import T as T_cell
    from normalizer import extended_group as _ext
    M_all = np.einsum("ij,gjk,kl->gil", TINV, np.array(_ext(), float), np.array(T_cell, float).T)

    def fold_pieces(pts, tol=1e-9):
        """Split a polyline where it crosses the splitting mirror beta3 = beta4 and move each piece into the displayed
        half whole (folding point by point would join the pieces with a stray chord)."""
        d = [b[2] - b[3] for b in pts]
        pieces, cur, side = [], [pts[0]], None
        for k in range(1, len(pts)):
            a, b, da, db = pts[k - 1], pts[k], d[k - 1], d[k]
            if (da > tol and db < -tol) or (da < -tol and db > tol):
                x = a + (b - a) * da / (da - db)
                cur.append(x)
                pieces.append((cur, side if side is not None else np.sign(da)))
                cur, side = [x], None
            cur.append(b)
            if side is None and abs(db) > tol:
                side = np.sign(db)
        pieces.append((cur, side if side is not None else 1))
        return [[norm(b[[1, 0, 3, 2]]) if sd < 0 else b for b in p] for p, sd in pieces]

    def same_seg(p, q, tol=1e-5):
        """Does segment p lie along polyline q (its midpoint and quarter points on it)? Survey ends are bisected, so
        a segment and its image need not share end points exactly."""
        def near(x):
            for a, b in zip(q[:-1], q[1:]):
                t = np.clip((x - a) @ (b - a) / max((b - a) @ (b - a), 1e-30), 0, 1)
                if np.linalg.norm(a + t * (b - a) - x) < tol:
                    return True
            return False
        n = len(p) - 1
        return all(near(p[k]) for k in (n // 4, n // 2, 3 * n // 4))
    norm = lambda b: np.asarray(b, float) / np.sum(b)
    found = {}
    for s in xlines:
        found.setdefault(s["id"], []).append([norm(b) for b in s["betas"]])
    for s in list(xlines):
        B = np.array([norm(b) for b in s["betas"]])
        imgs = np.einsum("gij,pj->gpi", M_all, B)
        sums = imgs.sum(axis=2, keepdims=True)
        ok = np.all(sums > 0, axis=(1, 2))
        imgs = imgs / np.where(sums == 0, 1, sums)
        ok &= np.all(imgs.min(axis=2) >= -1e-5, axis=1)      # survey ends are bisected, so allow their error
        for img in imgs[ok]:
            for seg in fold_pieces([norm(np.clip(b, 0, None)) for b in img]):
                if len(seg) < 2 or any(same_seg(seg, q) for q in found[s["id"]]):
                    continue
                found[s["id"]].append(seg)
                xlines.append({"id": s["id"], "copy": True, "pts": [_exact(xyz(b)).tolist() for b in seg],
                               "betas": [b.tolist() for b in seg],
                               "hover": f"{label(s['id'])}<br>copy under the extra half-turn<br>from β ∝ {beta_text(seg[0])}"
                                        f"<br>to β ∝ {beta_text(seg[-1])}"})
    # X10 and X33 passed the in-line nudge test only because a golden line lies in their walls; they live on curved
    # walls (CURVED_WALLS), so no line segment is drawn and no point is marked for them
    if os.path.exists("xloci_walls.json"):
        from collections import Counter
        for w in json.load(open("xloci_walls.json")).values():
            grid = {(p["fan"], *p["ij"]): p for p in w["points"]}
            n = w["n"]
            tris = {}
            for (f, i, j), p in grid.items():
                for corner in (((f, i, j), (f, i + 1, j), (f, i, j + 1)), ((f, i + 1, j), (f, i + 1, j + 1), (f, i, j + 1))):
                    if not all(c in grid for c in corner) or corner[1][1] + corner[1][2] > n:
                        continue
                    labs = Counter(grid[c]["label"] for c in corner).most_common(1)[0]
                    tid = labs[0]
                    if labs[1] >= 2 and tid in tmap and tid in wall_types and tid != "T2":
                        tris.setdefault(tid, []).append([_exact(xyz(grid[c]["beta"])).tolist() for c in corner])
            eq = _eq(w["normal"])
            for tid, tl in tris.items():
                xwalls.append({"id": tid, "tris": tl, "hover": f"{label(tid)}<br>patch in the wall {eq}"})

    # the named wall shapes F1, F4 and F5, traced exactly (fexact.py): each patch is its wall's polygon cut by exact
    # lines (and, for F5, a conic). Patches in an H4 mirror or a cell face are drawn solid; the others are their
    # copies under the extra half-turn (dashed outline, fainter fill). They replace the grid patches of those shapes.
    if os.path.exists("fexact_patches.json"):
        exact = json.load(open("fexact_patches.json"))
        exact_ids = {pt["target"] for pt in exact.values()}
        xwalls = [w for w in xwalls if w["id"] not in exact_ids]

        def plain(n):                          # a cell face beta_i = 0 or a mirror beta_i = beta_j
            n = np.asarray(n, float)
            nz = np.flatnonzero(np.abs(n) > 1e-9)
            return len(nz) == 1 or (len(nz) == 2 and abs(n[nz[0]] + n[nz[1]]) < 1e-9)
        has_plain = {t: any(plain(pt["normal"]) for pt in exact.values() if pt["target"] == t) for t in exact_ids}

        def nbname(nb):                        # neighbours recorded by signature get their number
            if nb.startswith("new:"):
                cid, xkey, _ = identify(nb[4:], refs)
                return cid or xid.get(xkey) or "an unnumbered shape"
            return nb

        def in_exact_patch(tid, b):
            """Is beta b (displayed half) on one of the shape's exact patches?"""
            b = np.asarray(b, float) / np.sum(b)
            for pt in exact.values():
                if pt["target"] != tid:
                    continue
                n = np.asarray(pt["normal"], float)
                if abs(b @ n) > 1e-7 * np.abs(n).sum():
                    continue
                basis = np.linalg.svd(np.vstack([n, np.ones(4)]))[2][2:]
                V = np.array(pt["corners"], float)
                V = (V / V.sum(axis=1, keepdims=True)) @ basis.T
                q = b @ basis.T
                inside = False
                for k in range(len(V)):
                    (x1, y1), (x2, y2) = V[k], V[(k + 1) % len(V)]
                    if (y1 > q[1]) != (y2 > q[1]) and q[0] < x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1):
                        inside = not inside
                if inside:
                    return True
            return False
        # survey lines of a traced shape that lie inside its own exact patches add nothing: drop them
        xlines = [s for s in xlines if not (s["id"] in exact_ids and s.get("betas")
                                            and all(in_exact_patch(s["id"], b) for b in s["betas"]))]
        for key, pt in exact.items():
            tid = pt["target"]
            if tid not in tmap or len(pt["corners"]) < 3:
                continue
            C = [np.asarray(b, float) for b in pt["corners"]]
            copy = has_plain[tid] and not plain(pt["normal"])
            Q = [_exact(xyz(b / b.sum())).tolist() for b in C]
            tris = [[Q[a], Q[b], Q[c]] for a, b, c in ear_clip(C, pt["normal"])]
            corners = []
            for b in (C if len(C) < 12 else []):
                if not any(np.allclose(b / b.sum(), q / q.sum(), atol=1e-9) for q in corners):
                    corners.append(b)
            edges = "; ".join(f"{_eq(m)} ({nbname(nb)} beyond)" for m, nb, _ in pt["edges"])
            curved = pt.get("curved") or (["X44"] if tid == "F5" and len(C) >= 12 else [])
            if pt.get("conics"):          # exact golden conics (exact_conics.py), with the line shape living on each
                edges += "; " + "; ".join(f"the conic {c['text']} ({nbname(c['label'])} along it)" for c in pt["conics"])
            elif curved:
                edges += "; " + "; ".join(f"a conic ({nbname(nb)} beyond)" for nb in curved)
            hover = (f"{label(tid)}<br>exact patch in the wall {_eq(pt['normal'])}"
                     f"{' (copy under the extra half-turn)' if copy else ''}<br>edges: {edges}"
                     + (f"<br>corners: {', '.join(beta_text(b) for b in corners)}" if corners else ""))
            xwalls.append({"id": tid, "tris": tris, "hover": hover, "copy": copy})
            xlines.append({"id": tid, "pts": Q + [Q[0]], "hover": hover, "copy": copy, "outline": True})

    # the shapes living on the edges of the exact patches (fexact.py label_edges): X shapes found there are drawn as
    # exact segments in their colours, and survey lines of theirs lying along them are dropped
    EDGE_DRAWN = {"E1"}       # named shapes living only on patch edges (the rest have rings or segments of their own)
    if os.path.exists("fexact_patches.json"):
        seen, edge_lines = {}, {}
        def conic_runs(C, conic):
            """The patch's corners lying on an exact conic, as runs of consecutive corners (cyclically)."""
            c, keep = np.asarray(conic["coeffs"]), conic["vars"]
            q = lambda b: sum(ci * b[keep[i]] * b[keep[j]] for ci, (i, j) in
                              zip(c, [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]))
            on = [abs(q(b)) < 1e-8 for b in C]     # (corners shared by two conics carry their fit error)
            n = len(C)
            if all(on):          # every corner on the conic (a straight edge joins two of them): one cyclic run
                runs = [list(range(n)) + [0]]
            else:
                start = next(k for k in range(n) if not on[k])
                runs, cur = [], []
                for s in range(1, n + 1):
                    k = (start + s) % n
                    if on[k]:
                        cur.append(k)
                    elif cur:
                        runs.append(cur)
                        cur = []
                if cur:
                    runs.append(cur)
            # a straight edge whose two ends both lie on the conic is not part of it: break runs at steps whose
            # midpoint is far from the conic for their length (an arc chord's midpoint sags only ~length^2)
            def off(a, b):
                m = (C[a] + C[b]) / 2
                g = np.array([(q(m + e * 1e-7) - q(m - e * 1e-7)) / 2e-7 for e in np.eye(4)])
                g = g - g.mean()
                return abs(q(m)) / max(np.linalg.norm(g), 1e-30)
            out = []
            for r in runs:
                if len(r) < 3:
                    continue
                steps = [np.linalg.norm(C[b] - C[a]) for a, b in zip(r, r[1:])]
                med = float(np.median([x for x in steps if x > 1e-12] or [1.0]))
                piece = [r[0]]
                for a, b in zip(r, r[1:]):
                    dd = np.linalg.norm(C[b] - C[a])
                    if dd < 1e-12:                              # a repeated corner
                        continue
                    if off(a, b) > 0.05 * dd or dd > 8 * med:   # a chord across a straight edge (a thin patch's
                                                                # straight edge stays close to its arc: also by length)
                        if len(piece) >= 2:
                            out.append(piece)
                        piece = [b]
                    else:
                        piece.append(b)
                if len(piece) >= 2:
                    out.append(piece)
            return [p for p in out if sum(np.linalg.norm(C[b] - C[a]) for a, b in zip(p, p[1:])) > 1e-9]

        for pt in json.load(open("fexact_patches.json")).values():
            C = [np.asarray(b, float) / np.sum(b) for b in pt["corners"]]
            step = 1 if len(C) < 12 else 6
            if pt.get("conics"):          # curved edges: drawn along their exact conics, end to end
                for conic in pt["conics"]:
                    tid = nbname(conic["label"]) if conic["label"].startswith("new:") else conic["label"]
                    if not (tid.startswith("X") or tid in EDGE_DRAWN) or tid not in tmap or tid == pt["target"]:
                        continue
                    is_copy = bool(has_plain.get(pt["target"]) and not plain(pt["normal"]))
                    for run in conic_runs(C, conic):
                        seg = [C[k] for k in run]
                        key = (tid, frozenset([tuple(np.round(seg[0], 7)), tuple(np.round(seg[-1], 7))]))
                        if key in seen:
                            if not is_copy:
                                seen[key][1] = False
                            continue
                        seen[key] = [seg, is_copy]
                        edge_lines.setdefault(tid, []).append(seen[key])
            for a, b, lab, *kind in pt.get("edge_labels", []):
                if pt.get("conics") and not kind and len(C) >= 12:
                    continue              # runs of arc points: drawn from the conics above
                tid = nbname(lab) if lab.startswith("new:") else lab
                if not (tid.startswith("X") or tid in EDGE_DRAWN) or tid not in tmap or tid == pt["target"]:
                    continue
                # a run of `step` corners, or (marked "straight") the one edge from corner a to corner b
                seg = [C[a], C[b]] if kind == ["straight"] else [C[(a + s) % len(C)] for s in range(step + 1)]
                key = (tid, frozenset([tuple(np.round(seg[0], 7)), tuple(np.round(seg[-1], 7))]))
                is_copy = bool(has_plain.get(pt["target"]) and not plain(pt["normal"]))   # dashed like its patch
                if key in seen:
                    if not is_copy:
                        seen[key][1] = False
                    continue
                seen[key] = [seg, is_copy]
                edge_lines.setdefault(tid, []).append(seen[key])
                tmap[tid].setdefault("edge_of", set()).add((pt["target"], step > 1))
        def near_edges(tid, b):
            b = np.asarray(b, float) / np.sum(b)
            for seg, _ in edge_lines.get(tid, []):
                for p, q in zip(seg[:-1], seg[1:]):
                    t = np.clip((b - p) @ (q - p) / max((q - p) @ (q - p), 1e-30), 0, 1)
                    if np.linalg.norm(p + t * (q - p) - b) < 1e-6:
                        return True
            return False
        xlines = [s for s in xlines if not (s.get("betas") and s["id"] in edge_lines
                                            and all(near_edges(s["id"], b) for b in s["betas"][1:-1]))]
        for tid, segs in edge_lines.items():
            for seg, is_copy in segs:
                hover = (f"{label(tid)}<br>exact edge between patches<br>from β ∝ {beta_text(seg[0])}"
                         f"<br>to β ∝ {beta_text(seg[-1])}")
                xlines.append({"id": tid, "pts": [_exact(xyz(b)).tolist() for b in seg], "hover": hover, "copy": is_copy})

    # a fundamental domain of the 2400-element group: the Dirichlet domain about a point of the E2 line.
    # Only centres on that line (beta1 = beta2 = beta3) give a domain that stays inside the half-cell and
    # holds every light purple axis on its surface (domain_search.py); (2, 2, 2, 1) is its midpoint.
    from dirichlet import dirichlet, display_faces
    from normalizer import extended_group
    group_n = extended_group()
    centre_beta = np.array([2, 2, 2, 1.0])
    domain = dirichlet(seed_from_beta(centre_beta), group_n)
    glue = {"q-axis": "folded onto itself across a light purple axis (a half-turn of the extra coset)",
            "ring-axis": "folded onto itself across a cross-ring axis (a half-turn of the swirlprism group)"}
    fdomain = []
    for face in display_faces(domain, group_n, len(group_n) // 2):
        how = glue.get(face["kind"], "glued to the other face of the same colour by "
                       + ("an element of the extra coset" if face["coset"] else "a rotation of the swirlprism group"))
        fdomain.append({"kind": face["kind"], "pts": [_exact(xyz(b)).tolist() for b in face["corners"]],
                        "axis": [_exact(xyz(b)).tolist() for b in face["axis"]],
                        "hover": f"Fundamental domain of the 2400-element group<br>{len(face['corners'])}-sided face, {how}"})

    # axes of the extra half-turns: the coset G.Q of the full 2400-element group (normalizer.py)
    from normalizer import coset_axes
    from supergroups import girdle_families, girdle_segments
    family = girdle_families()
    qaxes = []
    for a, b in coset_axes():
        pts = [_exact(xyz(a + (b - a) * k / 8)).tolist() for k in range(9)]
        fam = family(a, b)
        bowers = " (Bowers' 30 ghost girdles with skew 20-gonal symmetry)" if fam == 30 else ""
        qaxes.append({"pts": pts, "order": 2, "hover": "Axis of an extra half-turn (2400-element group)<br>"
                      f"one of {fam} such circles{bowers}<br>seeds on it are their own "
                      f"copy; their polytopes have 2400 symmetries<br>from β = {np.round(a, 5).tolist()}"
                      f"<br>to β = {np.round(b, 5).tolist()}"})
    for a, b in girdle_segments():
        pts = [_exact(xyz(a + (b - a) * k / 8)).tolist() for k in range(9)]
        qaxes.append({"pts": pts, "order": 3, "hover": "Axis of an order-3 rotation of the 3600-element group<br>"
                      "one of Bowers' 20 ghost girdles with 30/3-gyrogonic symmetry<br>seeds on it give "
                      f"1200-vertex polytopes (X12) with 3600 symmetries<br>from β = {np.round(a, 5).tolist()}"
                      f"<br>to β = {np.round(b, 5).tolist()}"})

    # the untwisted regular-decagon points (decagon_twist.py)
    phi2 = ((1 + 5 ** 0.5) / 2) ** 2
    for p, copy in ((np.array([phi2, 1, 1, 1]), False), (np.array([phi2, 1, 2, 1]), True)):
        x = seed_from_beta(p)
        special.append({"id": "X6", "q": _exact(xyz(p / p.sum())).tolist(), "decagon": True,
                        "hover": "Untwisted regular decagons: each decagon's cell is a pentagonal cupola whose pentagon lines up "
                                 "with the decagon (decagon_twist.py); a geometric coincidence, not extra symmetry: "
                                 f"the polytope keeps its 1200 symmetries{' (copy under the extra half-turn)' if copy else ''}<br>β ∝ {beta_text(p)}"
                                 "<br>seed " + ", ".join(fmt17(v) for v in x)})

    # T2: exact patches in the mirrors (t2exact.py) and their copies under the extra half-turn
    import t2exact
    for name, (plane, desc) in t2exact.CURVE_DESCRIPTIONS.items():
        curve = t2exact.CURVES[name]()
        for is_copy, pts in [(False, curve)] + [(True, c) for c in t2exact.curve_copies(curve)]:
            where = "copy under the extra half-turn" if is_copy else f"in the mirror {plane}"
            segments.append({"id": "T2", "copy": is_copy, "pts": [_exact(xyz(p)).tolist() for p in pts],
                             "hover": f"T2 · exact curve {name} ({where})<br>{desc}"})
    tpatches = []
    for name, (plane, bounds) in t2exact.DESCRIPTIONS.items():
        polys = [(False, t2exact.PATCHES[name]())] + [(True, poly) for poly in t2exact.copies(name)]
        for is_copy, poly in polys:
            pts = [_exact(xyz(p)).tolist() for p in poly]
            where = "copy under the extra half-turn" if is_copy else f"in the mirror {plane}"
            tpatches.append({"plane": where, "copy": is_copy, "pts": pts + [pts[0]],
                             "hover": f"T2 · exact region {name} ({where})<br>{bounds}"})

    # geometry
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            a, c = P[i], P[j]
            if a[2] < -1e-12 and c[2] < -1e-12:
                continue
            if a[2] < -1e-12 or c[2] < -1e-12:
                lo, hi = (a, c) if a[2] < c[2] else (c, a)
                a, c = lo + (hi - lo) * (-lo[2] / (hi[2] - lo[2])), hi
            edges.append([_exact(a).tolist(), _exact(c).tolist()])
    mid34 = ((P[2] + P[3]) / 2).tolist()
    split = [P[0].tolist(), P[1].tolist(), mid34]
    mirrors = []
    for i in range(4):
        for j in range(i + 1, 4):
            if {i, j} == {2, 3}:
                continue
            k, m = [x for x in range(4) if x not in (i, j)]
            poly = clip_upper([P[k].tolist(), P[m].tolist(), ((P[i] + P[j]) / 2).tolist()])
            if len(poly) >= 3:
                mirrors.append(poly)

    # the lettered classes' colours follow location (place_colors.py): regions lightest, walls darker, lines and
    # points darkest, in the hue family of the coloured region beside them; X classes keep their shared colour
    from place_colors import assign as place_assign
    from place_colors import css as place_css
    place = place_assign([t["id"] for t in types if t["listed"]])
    if "T1" in place:      # T1 fell next to C2b's orange: the line-lightness hue farthest from every other colour
        place["T1"] = ("#8f4896", "#d38dd9", *place["T1"][2:])
    # transitional X classes: the X grey, a step darker on a wall, darker on a line, darkest at a point
    x_kind = {tid: r["kind"] for tid, r in r2.items()} if os.path.exists("xloci_step2.json") else {}
    for t in types:
        if t["id"] in place:
            t["color"] = f"--ty-{t['id']}"
        elif t.get("transitional"):
            kind = "wall" if t["id"] in CURVED_WALLS else x_kind.get(t["id"])
            t["color"] = {"wall": "--xw", "line": "--xl"}.get(kind, "--xp")
    type_css = place_css(place)
    for t in types:
        t.pop("counts", None)
        t.setdefault("samples", 0)
        if t["id"] in ("T1", "T2"):
            t["where"] = "golden samples; " + ("lines where a face meets a mirror" if t["id"] == "T1" else "patches in mirrors and faces")
    SEGMENT_WHERE = {
        "T1": "the whole edge-to-mirror line from spidrox (C6) to C4, and its dashed copy",
        "E2": "the line from the face centre (C5) to the cell centre (C1), and its dashed copy; nowhere else",
        "T2": "three exact patches in the mirrors (A: β1 = β3, B: β2 = β3, C: β1 = β2), each with a spidrox corner, "
              "and a conic curve D in β2 = β3; dashed copies of each",
    }
    for t in types:
        if t["id"] in SEGMENT_WHERE:
            t["where"] = SEGMENT_WHERE[t["id"]]
    for t in types:
        if t.pop("axis_note", None):
            t["where"] = (t.get("where") or ("golden samples" if t.get("samples") else "")) + \
                "; ✕ where a 2400-symmetry axis meets the boundary"
    # what each shape fills (region, wall, line or point) and the shapes found on its boundary (boundaries.py)
    for t in types:
        loc = (t.get("locus") or "").lower()
        if not t.get("transitional") and not t["id"].startswith("T") and (not loc or loc.startswith("fills")):
            t["dim"] = "region"
        elif loc.startswith("wall") or "curved wall" in loc:
            t["dim"] = "wall"
        elif loc.startswith(("line", "a curve")):
            t["dim"] = "line"
        else:
            t["dim"] = "point"
    for t in types:        # shapes found on the edges of exact patches live on lines (curves on conic edges)
        edge_of = t.pop("edge_of", None)
        if edge_of and not (t.get("locus") or "").startswith("wall"):   # (a wall can meet another along a line)
            t["dim"] = "line"
            if not (t.get("locus") or "").startswith(("line", "wall")):
                curved = sorted({w for w, c in edge_of if c})
                straight = sorted({w for w, c in edge_of if not c})
                t["locus"] = "; ".join(([f"a curve: the conic edges of the {', '.join(curved)} patches"] if curved else [])
                                       + ([f"a line: edges of the {', '.join(straight)} patches"] if straight else []))
    # shapes with exact wall patches (fexact.py) are walls, whatever the survey's nudge test called them
    if os.path.exists("fexact_patches.json"):
        pw = {}
        for pt in json.load(open("fexact_patches.json")).values():
            if len(pt["corners"]) >= 3:
                pw.setdefault(pt["target"], []).append(pt["normal"])
        for tid, normals in pw.items():
            if tid in tmap:
                tmap[tid]["dim"] = "wall"
                if tid != "X26":                  # (X26's locus says more: its crease and where it meets X79)
                    walls = sorted({_eq(n) for n in normals}, key=lambda e: (len(e), e))
                    tmap[tid]["locus"] = "wall: " + ", ".join(walls) + " (exact patches)"
    for tid, dim in {"T1": "line", "E2": "line", "T2": "wall", "F1": "wall", "F4": "wall", "F5": "wall"}.items():
        if tid in tmap:
            tmap[tid]["dim"] = dim
    for r in rings + main_ring:
        if tmap[r["id"]].get("dim") == "region" and not tmap[r["id"]].get("samples"):
            tmap[r["id"]]["dim"] = "line"
    if os.path.exists("boundaries.json"):
        def bname(nb):
            if nb.startswith("new:"):
                cid, xkey, _ = identify(nb[4:], refs)
                return cid or xid.get(xkey)
            return nb
        for tid, nbs in json.load(open("boundaries.json")).items():
            tid = bname(tid) if tid.startswith("new:") else tid
            if tid in tmap:
                ids = ({bname(nb) for nb in nbs} | set(tmap[tid].pop("extra_bounds", []))) - {None, "ERR", tid}
                tmap[tid]["bounds"] = sorted((i for i in ids if i in tmap),
                                             key=lambda i: (i[0], int("".join(ch for ch in i[1:] if ch.isdigit()) or 0), i))
    for tid in ("X80", "X81"):  # the two conics of X79's lens both run from A1 to the X2 corner
        if tid in tmap:
            tmap[tid]["bounds"] = ["A1", "X2"]       # (corners where arcs meet can classify just off the point)
    rank = {"region": 3, "wall": 2, "line": 1, "point": 0}
    for t in types:        # boundaries known from the traced geometry itself (none from boundaries.json)
        extra = t.pop("extra_bounds", None)
        if extra:
            t["bounds"] = sorted(set(t.get("bounds", [])) | {i for i in extra if i in tmap})
        if t.get("bounds"):   # only lower-dimensional shapes can bound a shape (corners just past a curve can mislead)
            own = rank.get(t.get("dim"), 3)
            t["bounds"] = [b for b in t["bounds"] if rank.get(tmap[b].get("dim"), 3) < own
                           or own == 1 == rank.get(tmap[b].get("dim"), 3)]    # a line may end where it meets another
    data = {"types": types, "samples": samples_out, "uniform": uniform_out, "special": special,
            "rings": rings, "main": main_ring, "segments": segments, "tpatches": tpatches, "qaxes": qaxes, "regular": regular, "xlines": xlines, "xwalls": xwalls, "fdomain": fdomain,
            "fcentre": _exact(xyz(centre_beta / centre_beta.sum())).tolist(),
            "mirrors": mirrors, "split": split, "edges": edges, "totalSamples": len(samples)}
    from dodeca_view import piece_view
    from dodeca_view import view as dodeca_view
    dodeca = dodeca_view(data)                     # the same geometry seen from the M34 dodecahedron
    piece = piece_view(dodeca)                     # and cut down to one domain, a tenth of it
    import chamber_domain
    import dodeca_view as dv
    dv.use_centre((1, 0, 0, 0))                    # the dodecahedron around the 600-cell vertex V1
    v1 = dodeca_view(data)
    chambers = chamber_domain.view(v1)             # one domain of 12 whole H4 chambers inside it
    dv.use_centre((0, 0, 1, 1))
    for view in (dodeca, piece, v1, chambers):
        dv.tidy_lines(view)
        dv.split_edges_on_rings(view)
    dv.split_edges_on_rings(data)                  # the cell view keeps its points (front/back filter them)
    # one domain in which every region is a single piece (cohesive_domain.py writes it; slow, so run separately)
    cohesive = json.load(open("cohesive_view.json")) if os.path.exists("cohesive_view.json") else {"samples": {}, "bounds": [[-1, -1, -1], [1, 1, 1]]}
    flip_vertical(data)
    template = open("cell_atlas2_template.html").read()
    def dump(o, n):                                # the only rounding: every view was computed at full precision
        return json.dumps(_rounded(o, n), separators=(",", ":"))
    open(out_path, "w").write(template.replace("__TYPE_CSS__", type_css).replace("__DATA__", dump(data, 6))
                              .replace("__DODECA__", dump(dodeca, 5))
                              .replace("__PIECE__", dump(piece, 5))
                              .replace("__V1__", dump(v1, 5))
                              .replace("__CHAMBERS__", dump(chambers, 5))
                              .replace("__COHESIVE__", dump(cohesive, 5)))
    listed = [t for t in types if t["listed"]]
    print(f"{len(samples)} samples; {len(listed)} wiki shapes ({sum(1 for t in listed if t['samples'] or t.get('where'))} found), "
          f"{len(types) - len(listed)} unlisted")
    for t in types:
        print(f"  {t['id']:<4} {t.get('vertices', ''):>5} {t['samples']:>4}  {t['name'][:80]}  {t.get('where', '')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
