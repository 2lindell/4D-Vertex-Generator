"""Half-cell atlas with wiki-ordered labels: one letter per vertex count and named/unnamed (see README)."""
from __future__ import annotations

import itertools
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
    load_tlines,
    seed_text,
    to_upper,
    xyz,
)
from cellframe import TINV, snap_golden
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
    ("C3", "Pentagonal-gyroprismatic triacosihexecontachoron", ("ref", "Cross ring: Pentagonal-gyroprismatic triacosihexecontachoron")),
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
    for key in ("rings", "main", "tlines", "tpatches"):
        for r in data[key]:
            r["pts"] = [f(p) for p in r["pts"]]
    data["mirrors"] = [[f(p) for p in m] for m in data["mirrors"]]
    data["split"] = [f(p) for p in data["split"]]
    data["edges"] = [[f(p) for p in e] for e in data["edges"]]


def main(samples_path, out_path):
    refs = json.load(open("references.json"))          # signature (with valences) -> reference name
    ref_by_name = {v: k for k, v in refs.items()}
    # the icosafold midpoint shares its signature with the range around it; that signature is a range type
    ref_by_name["Cross ring: antiprisms split 120+120 (not in the wiki list)"] = next(
        k for k, v in refs.items() if v.startswith("Cross ring: antiprisms split"))

    def identify(sig):
        """Return (canonical id or None, x-key, reference name or None)."""
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

    samples = json.load(open(samples_path))
    for s in samples:
        s["beta"] = to_upper(s["beta"]).tolist()
        s["id"], s["xkey"], s["ref"] = identify(s["sig"])

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
    # unlisted: Y for 600 vertices, X for 1200 (and Z for any other count), each numbered from 1
    xid, counters = {}, {}
    for k in sorted(xkeys, key=xsort):
        letter = {600: "Y", 1200: "X"}.get(int(xkeys[k]["sig"].split(":")[0]), "Z")
        counters[letter] = counters.get(letter, 0) + 1
        xid[k] = f"{letter}{counters[letter]}"
    for s in samples:
        if s["id"] is None and s["xkey"] in xid:
            s["id"] = xid[s["xkey"]]

    # uniform pieces of each type (used for naming X types found at uniform seeds)
    pieces = {}
    for u in uniform:
        if u["sample"]:
            pieces.setdefault(u["sample"]["id"], []).append(u["name"])

    types = []
    for i, (cid, name, _) in enumerate(CANON):
        types.append({"id": cid, "name": name, "listed": True, "color": STYLE[cid][0], "symbol": STYLE[cid][1]})
    for k in sorted(xkeys, key=xsort):
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
                      "example": f"{seed_text(s['beta'])}  β ∝ {beta_text(s['beta'])}", "counts": True})
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

    # rings, coloured by the shape each range produces
    sys.path.insert(0, ".")
    from lines import range_id, ring_parameter
    ref_to_id = {}
    for rid, refname in RING_RANGE_TO_REF.items():
        sig = ref_by_name[refname]
        ref_to_id[rid] = identify(sig)[0] or xid.get(sig) or xid.get(identify(sig)[1])
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
    tlines = [L for L in load_tlines() if L["id"] == "T1"]
    tpatches = []
    B2 = [np.array(s["beta"]) / sum(s["beta"]) for s in samples if s["id"] == "T2"]
    planes = [(f"β{i + 1} = β{j + 1}", lambda b, i=i, j=j: abs(b[i] - b[j]) < 1e-9) for i, j in itertools.combinations(range(4), 2)]
    planes += [(f"β{i + 1} = 0", lambda b, i=i: b[i] < 1e-9) for i in range(4)]
    from scipy.spatial import ConvexHull
    for lab, on in planes:
        X = np.array([xyz(b) for b in B2 if on(b)])
        if len(X) < 3:
            continue
        c = X.mean(0)
        _, sv, vt = np.linalg.svd(X - c)
        if sv[1] < 1e-9:
            continue
        hull = ConvexHull((X - c) @ vt[:2].T)
        poly = [np.round(X[k], 6).tolist() for k in hull.vertices]
        tpatches.append({"plane": lab, "samples": int(len(X)), "pts": poly + [poly[0]]})

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

    for t in types:
        t.pop("counts", None)
        t.setdefault("samples", 0)
        if t["id"] in ("T1", "T2"):
            t["where"] = "golden samples; " + ("lines where a face meets a mirror" if t["id"] == "T1" else "patches in mirrors and faces")
    data = {"types": types, "samples": samples_out, "uniform": uniform_out, "special": special,
            "rings": rings, "main": main_ring, "tlines": tlines, "tpatches": tpatches,
            "mirrors": mirrors, "split": split, "edges": edges, "totalSamples": len(samples),
            "oldIds": OLD_WIKI_IDS, "firstIds": FIRST_WIKI_IDS}
    flip_vertical(data)
    template = open("cell_atlas2_template.html").read()
    open(out_path, "w").write(template.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    listed = [t for t in types if t["listed"]]
    print(f"{len(samples)} samples; {len(listed)} wiki shapes ({sum(1 for t in listed if t['samples'] or t.get('where'))} found), "
          f"{len(types) - len(listed)} unlisted")
    for t in types:
        print(f"  {t['id']:<4} {t.get('vertices', ''):>5} {t['samples']:>4}  {t['name'][:80]}  {t.get('where', '')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
