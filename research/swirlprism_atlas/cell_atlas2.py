"""Half-cell atlas with wiki-ordered labels: one letter per vertex count and named/unnamed (see README)."""
from __future__ import annotations

import json
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
OLD_WIKI_IDS = {  # previous labels -> current labels
    "N1": "A1", "N2": "C1", "N3": "B1", "N4": "C2a/C2b", "Y1": "C2a", "Y2": "C2b", "N5": "C3", "N6": "C4", "N7": "C5", "N8": "C6", "N9": "E1", "N10": "E2",
    "W1": "D1", "W2": "D2", "W3": "D3", "W4": "F1", "W5": "F2", "W6": "F3", "W7": "F4", "W8": "F5",
}
FIRST_WIKI_IDS = {"W1": "F1", "W2": "F2", "W3": "F3", "W4": "F4", "W5": "F5"}   # the labels used before that
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
                [*np.round(xyz(b), 6).tolist(), f"β ∝ {beta_text(b)}<br>seed {seed_text(b)}"])

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
        uniform_out.append({"id": tid, "q": np.round(xyz(u["beta"]), 6).tolist(),
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
                q = np.round(xyz(b), 6).tolist()
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
                q = np.round(xyz(b), 6).tolist()
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
        q = np.round(xyz(p), 6).tolist()
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
            special.append({"id": tid, "q": np.round(xyz(p), 6).tolist(),
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
            q = np.round(xyz(p / p.sum()), 6).tolist()
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
                                          else e["kind"] if "axis" in e["kind"]
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
            x = np.round(xyz(b), 6).tolist()
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
                 for pts in _split_upper([np.round(xyz(b), 6).tolist() for _, b in clip_circle(F)])]
    for r in rings + main_ring:
        tmap[r["id"]].setdefault("where", "cross ring" if r["id"] != "B1" else "main ring")

    # transitional layer: T1 lines, T2 patches in mirrors and faces
    segments = []
    for seg in EXACT_SEGMENTS:
        a, b = np.array(seg["a"], float), np.array(seg["b"], float)
        a, b = a / a.sum(), b / b.sum()
        kind = "copy under the extra half-turn" if seg["copy"] else "traced"
        segments.append({"id": seg["id"], "copy": seg["copy"], "pts": [np.round(xyz(a + (b - a) * k / 8), 6).tolist() for k in range(9)],
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
            out.append(np.round(xyz(bb / bb.sum()), 6).tolist())
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
    for line in ext:
        whole = max(abs(t) for t in line["t"]) >= 2.99      # the extension ran to its limit: the whole circle
        fams[line["kind"]].append((np.array(line["a"]), np.array(line["b"]), whole))
    add_family("prism", fams["prism"])
    add_family("antiprism", fams["antiprism"])

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
                        xlines.append({"id": tid, "pts": [np.round(xyz(p), 6).tolist() for p in pts],
                                       "hover": f"{label(tid)}<br>{tmap[tid].get('locus', '')}<br>from β ∝ {beta_text(lo)}"
                                                f"<br>to β ∝ {beta_text(hi)}"})
                    k += 1
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
                        tris.setdefault(tid, []).append([np.round(xyz(grid[c]["beta"]), 6).tolist() for c in corner])
            eq = _eq(w["normal"])
            for tid, tl in tris.items():
                xwalls.append({"id": tid, "tris": tl, "hover": f"{label(tid)}<br>patch in the wall {eq}"})

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
        fdomain.append({"kind": face["kind"], "pts": [np.round(xyz(b), 6).tolist() for b in face["corners"]],
                        "axis": [np.round(xyz(b), 6).tolist() for b in face["axis"]],
                        "hover": f"Fundamental domain of the 2400-element group<br>{len(face['corners'])}-sided face, {how}"})

    # axes of the extra half-turns: the coset G.Q of the full 2400-element group (normalizer.py)
    from normalizer import coset_axes
    from supergroups import girdle_families, girdle_segments
    family = girdle_families()
    qaxes = []
    for a, b in coset_axes():
        pts = [np.round(xyz(a + (b - a) * k / 8), 6).tolist() for k in range(9)]
        fam = family(a, b)
        bowers = " (Bowers' 30 ghost girdles with skew 20-gonal symmetry)" if fam == 30 else ""
        qaxes.append({"pts": pts, "order": 2, "hover": "Axis of an extra half-turn (2400-element group)<br>"
                      f"one of {fam} such circles{bowers}<br>seeds on it are their own "
                      f"copy; their polytopes have 2400 symmetries<br>from β = {np.round(a, 5).tolist()}"
                      f"<br>to β = {np.round(b, 5).tolist()}"})
    for a, b in girdle_segments():
        pts = [np.round(xyz(a + (b - a) * k / 8), 6).tolist() for k in range(9)]
        qaxes.append({"pts": pts, "order": 3, "hover": "Axis of an order-3 rotation of the 3600-element group<br>"
                      "one of Bowers' 20 ghost girdles with 30/3-gyrogonic symmetry<br>seeds on it give "
                      f"1200-vertex polytopes (X12) with 3600 symmetries<br>from β = {np.round(a, 5).tolist()}"
                      f"<br>to β = {np.round(b, 5).tolist()}"})

    # T2: exact patches in the mirrors (t2exact.py) and their copies under the extra half-turn
    import t2exact
    for name, (plane, desc) in t2exact.CURVE_DESCRIPTIONS.items():
        curve = t2exact.CURVES[name]()
        for is_copy, pts in [(False, curve)] + [(True, c) for c in t2exact.curve_copies(curve)]:
            where = "copy under the extra half-turn" if is_copy else f"in the mirror {plane}"
            segments.append({"id": "T2", "copy": is_copy, "pts": [np.round(xyz(p), 6).tolist() for p in pts],
                             "hover": f"T2 · exact curve {name} ({where})<br>{desc}"})
    tpatches = []
    for name, (plane, bounds) in t2exact.DESCRIPTIONS.items():
        polys = [(False, t2exact.PATCHES[name]())] + [(True, poly) for poly in t2exact.copies(name)]
        for is_copy, poly in polys:
            pts = [np.round(xyz(p), 6).tolist() for p in poly]
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
            edges.append([np.round(a, 6).tolist(), np.round(c, 6).tolist()])
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
    data = {"types": types, "samples": samples_out, "uniform": uniform_out, "special": special,
            "rings": rings, "main": main_ring, "segments": segments, "tpatches": tpatches, "qaxes": qaxes, "regular": regular, "xlines": xlines, "xwalls": xwalls, "fdomain": fdomain,
            "fcentre": np.round(xyz(centre_beta / centre_beta.sum()), 6).tolist(),
            "mirrors": mirrors, "split": split, "edges": edges, "totalSamples": len(samples),
            "oldIds": OLD_WIKI_IDS, "firstIds": FIRST_WIKI_IDS}
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
    flip_vertical(data)
    template = open("cell_atlas2_template.html").read()
    open(out_path, "w").write(template.replace("__TYPE_CSS__", type_css).replace("__DATA__", json.dumps(data, separators=(",", ":")))
                              .replace("__DODECA__", json.dumps(dodeca, separators=(",", ":")))
                              .replace("__PIECE__", json.dumps(piece, separators=(",", ":")))
                              .replace("__V1__", json.dumps(v1, separators=(",", ":")))
                              .replace("__CHAMBERS__", json.dumps(chambers, separators=(",", ":"))))
    listed = [t for t in types if t["listed"]]
    print(f"{len(samples)} samples; {len(listed)} wiki shapes ({sum(1 for t in listed if t['samples'] or t.get('where'))} found), "
          f"{len(types) - len(listed)} unlisted")
    for t in types:
        print(f"  {t['id']:<4} {t.get('vertices', ''):>5} {t['samples']:>4}  {t['name'][:80]}  {t.get('where', '')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
