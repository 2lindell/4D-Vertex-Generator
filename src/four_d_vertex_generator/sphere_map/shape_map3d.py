"""Data for the 3D shape map (shape_map3d.html): every named shape's neighbourhood on S^3.

A node is a named shape: a region volume, a wall piece (face), a named line piece or a named
point. Curves and corners that only mark where walls meet get no node; they appear as links
between the walls that meet there.

What surrounds every named line piece, and every wall-wall meeting curve, is read exactly from
the ring round it (`incidence.Incidence.round_line`) at points along it: each sector's piece names
its region volume and each wall between sectors names its patch, and through the component tables
(`components.npz`) these become placed shapes - (class, group element g): class's representative
carried by g. Every shape's neighbourhood is then these incidences seen from it (carried by the
inverse of its own placement). Each class keeps its representative's geometry on S^3 (R^4), and
its stabiliser order: the symmetry of the piece on the sphere.

    python -m four_d_vertex_generator.sphere_map.shape_map3d
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .frame import data_dir, group, multiplication_table, point_from_beta
from .incidence import Incidence

ROOT = None
RANK = {"side": 0, "boundary": 0, "round": 1, "end": 1, "corner": 1, "touches": 2, "meets": 2,
        "across a line": 3}
# what each kind of shape shows round it
SHOWN = {"region": {"boundary", "corner"},
         "face": {"side", "boundary", "corner", "meets", "across a line"},
         "line": {"round", "end"},
         "point": {"boundary", "touches", "across a line"}}


def _log(*a):
    print(*a, flush=True)


def build() -> dict:
    root = data_dir() / "sphere_map"
    atlas = json.loads((root / "atlas.json").read_text())
    patches = json.loads((root / "patches.json").read_text())["patches"]
    pieces = json.loads((root / "pieces.json").read_text())["pieces"]
    comps = np.load(root / "components.npz")
    rcomp, fcomp = comps["region_component"], comps["face_component"]
    g = group()
    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[k] == 0)[0]) for k in range(len(mult))])

    # ---- placements of region and face classes: component id -> (class, g)
    region_at, face_at = {}, {}
    for r in atlas["regions"]:
        k0, p0 = r["members"][0]
        for gg in range(len(g)):
            region_at[int(rcomp[mult[gg, k0], p0])] = (r["id"], gg)
    for f in atlas["faces"]:
        k0, p0 = f["members"][0]
        for gg in range(len(g)):
            face_at[int(fcomp[mult[gg, k0], p0])] = (f["id"], gg)

    # ---- representative geometry (R^4)
    def unit(v):
        return v / np.linalg.norm(v)
    region_pos = []
    for r in atlas["regions"]:
        P = np.array([g[k] @ point_from_beta(np.asarray(pieces[p]["sample_beta"])) for k, p in r["members"]])
        region_pos.append(unit(P.mean(0)))
    face_geo = []
    for f in atlas["faces"]:
        P = np.array([g[k] @ point_from_beta(np.asarray(patches[p]["sample_beta"]))
                      for k, p in f["members"]])
        c = unit(P.mean(0))
        # its two sides: the pieces on each side of its patches
        sides = [[], []]
        for k, p in f["members"]:
            pa = patches[p]["pieces"]
            kb = int(mult[k, patches[p]["side_b_copy_shift"]])
            for s, (kk, pp) in enumerate(((k, pa[0]), (kb, pa[1]))):
                sides[s].append(g[kk] @ point_from_beta(np.asarray(pieces[pp]["sample_beta"])))
        face_geo.append({"c": c, "cloud": P, "side": [unit(np.mean(sd, 0)) for sd in sides]})
    line_geo = []
    for lp in atlas["line_pieces"]:
        pts = []
        for seg in lp["segments"]:
            poly = np.asarray(seg["polyline_beta"], float)
            pts.append(point_from_beta(poly / poly.sum(axis=1, keepdims=True)) @ g[seg["copy"]].T)
        line_geo.append(pts)
    point_pos = [np.asarray(p["R4"], float) for p in atlas["points"]]
    named_points = [p["id"] for p in atlas["points"] if p.get("atlas_point")]

    # ---- patch lookup for walls read off a ring
    inner = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): p["id"]
             for p in patches if p["face"] is None}
    on_face = defaultdict(list)
    for p in patches:
        if p["face"] is not None:
            on_face[(p["pieces"][0], p["pieces"][1])].append(p)

    def face_of_wall(k0, wl):
        """(face class, g) of the wall between two sectors of a ring read in copy k0."""
        if not wl.get("sides"):
            return None
        (ca, pa), (cb, pb) = wl["sides"]
        if pa is None or pb is None:
            return None
        if ca == cb:
            for kk, s in wl["surfaces_here"]:
                if kk == ca and (min(pa, pb), max(pa, pb), s) in inner:
                    pid = inner[(min(pa, pb), max(pa, pb), s)]
                    return face_at.get(int(fcomp[mult[k0, ca], pid]))
            return None
        for (x, y, cx, cy) in ((pa, pb, ca, cb), (pb, pa, cb, ca)):
            shift = int(mult[inv[cx], cy])
            for p in on_face.get((x, y), ()):
                if p["side_b_copy_shift"] == shift:
                    return face_at.get(int(fcomp[mult[k0, cx], p["id"]]))
        return None

    inc = Incidence()

    def ring(x, tangent, k0):
        """Regions and faces round a curve point (domain beta x, read in copy k0), in angular
        order: [{'kind', 'class', 'g', 'angle'}]."""
        r = inc.round_line(x, tangent)
        if not r:
            return []
        out = []
        for s in r["sectors"]:
            if s["piece"] is None:
                continue
            at = region_at.get(int(rcomp[mult[k0, s["copy"]], s["piece"]]))
            if at:
                out.append({"kind": "region", "class": at[0], "g": at[1], "angle": s["angle"]})
        for wl in r["walls"]:
            at = face_of_wall(k0, wl)
            if at:
                out.append({"kind": "face", "class": at[0], "g": at[1]})
        return out

    import pickle
    cache = root / ".cache" / "shape_map3d_rings.pkl"
    cached = pickle.loads(cache.read_bytes()) if cache.exists() else None
    # ---- incidences round every named line piece (representative), at a few points along it
    _log("reading rings round the named line pieces")
    line_round = []
    for lp in ([] if cached else atlas["line_pieces"]):
        seen = {}
        for seg in lp["segments"]:
            poly = np.asarray(seg["polyline_beta"], float)
            poly = poly / poly.sum(axis=1, keepdims=True)
            if len(poly) < 3:
                continue
            for j in sorted({len(poly) // 4, len(poly) // 2, 3 * len(poly) // 4} - {0, len(poly) - 1}):
                for e in ring(poly[j], poly[j + 1] - poly[j - 1], seg["copy"]):
                    seen[(e["kind"], e["class"], e["g"])] = e
        line_round.append(list(seen.values()))
    # ---- incidences round every wall-wall meeting curve on no atlas line (all domain pieces)
    _log("reading rings round the wall-wall meeting curves")
    mc = json.loads((root / "merged_counts.json").read_text())
    meetings = []
    for e in mc["wall_meeting_curves"]:
        if not e.get("found") or e["on_atlas_line"]:
            continue
        for poly in e["domain_polylines"]:
            P = np.asarray(poly, float)
            if len(P) < 3:
                continue
            j = len(P) // 2
            rr = ring(P[j], P[j + 1] - P[j - 1], 0)
            faces = [x for x in rr if x["kind"] == "face"]
            if len({x["class"] for x in faces}) < 2:
                continue
            meetings.append({"walls": e["walls"], "round": rr,
                             "ends_r4": point_from_beta(P[[0, -1]]),
                             "polyline": point_from_beta(P[:: max(1, len(P) // 24)]).round(6).tolist()})

    if cached:
        line_round = cached[0]
    else:
        cache.write_bytes(pickle.dumps((line_round, [])))
    # ---- named points at the ends of named line pieces: (point class, g) by position
    from scipy.spatial import cKDTree
    pimg = np.einsum("gij,nj->gni", g, np.array([point_pos[i] for i in named_points]))
    ptree = cKDTree(pimg.reshape(-1, 4))

    def point_at(x):
        d, i = ptree.query(x)
        if d > 1e-6:
            return None
        gi, ni = divmod(int(i), len(named_points))
        return (named_points[ni], gi)
    line_ends = []
    for geo in line_geo:
        ends = []
        for P in (geo[0][0], geo[-1][-1]):
            at = point_at(P)
            if at:
                ends.append({"kind": "point", "class": at[0], "g": at[1]})
        # ends of segments joined across copies are not corners; only the two outer ends count
        line_ends.append(ends)

    # ---- neighbourhoods: every incidence seen from each shape (its placement carried to identity)
    _log("collecting neighbourhoods")
    nb = defaultdict(dict)                     # (kind, class) -> {(kind, class, g): relation}

    def add(focus, other, rel):
        (fk, fc, fg), (ok, oc, og) = focus, other
        if (fk, fc) == (ok, oc) and fg == og:
            return
        h = int(inv[fg])                       # carry focus to identity
        key = (ok, oc, int(mult[h, og]))
        cur = nb[(fk, fc)].get(key)
        if cur is None or RANK[rel] < RANK[cur]:
            nb[(fk, fc)][key] = rel
    for li, (round_, ends) in enumerate(zip(line_round, line_ends)):
        me = ("line", li, 0)
        members = [(x["kind"], x["class"], x["g"]) for x in round_] + \
                  [(x["kind"], x["class"], x["g"]) for x in ends]
        for m in members:
            add(me, m, "round" if m[0] != "point" else "end")
            add(m, me, "boundary")
        # the line's end points are corners of every shape round it
        for m in members:
            if m[0] == "point":
                continue
            for e in members:
                if e[0] == "point":
                    add(m, e, "corner")
                    add(e, m, "touches")
        # shapes round one line are neighbours of each other along it
        for a in members:
            for b in members:
                if a != b:
                    add(a, b, "across a line")
    for mt in meetings:
        members = [(x["kind"], x["class"], x["g"]) for x in mt["round"]]
        for a in members:
            for b in members:
                if a != b:
                    add(a, b, "meets")
        # a named point at an end of the meeting curve is a corner of the shapes round it
        for x in mt["ends_r4"]:
            at = point_at(x)
            if not at:
                continue
            e = ("point", at[0], at[1])
            for m in members:
                add(m, e, "corner")
                add(e, m, "touches")
    for f in atlas["faces"]:
        for sd in f["sides"]:
            add(("face", f["id"], 0), ("region", sd["class"], sd["g"]), "side")
            add(("region", sd["class"], sd["g"]), ("face", f["id"], 0), "boundary")
    face_sides = {f["id"]: f["sides"] for f in atlas["faces"]}

    # one node is one connected sheet of a named wall (the build's wall components on S^3) or
    # one connected stretch of a named line (its pieces joined end to end over the sphere)
    wcomp = comps["wall_component"]
    sheet_rep = {f["id"]: f["members"][0] for f in atlas["faces"]}

    def sheet(c, gg):
        k0, p0 = sheet_rep[c]
        return int(wcomp[mult[gg, k0], p0])
    line_name = {lp["id"]: lp["line_shape"] for lp in atlas["line_pieces"]}
    ends_img, owner = [], []
    for li, geo in enumerate(line_geo):
        for P in (geo[0][0], geo[-1][-1]):
            ends_img.append(np.einsum("gij,j->gi", g, P))
            owner += [(li, gg) for gg in range(len(g))]
    ends_img = np.concatenate(ends_img)
    nl = len(line_geo) * len(g)
    lparent = list(range(nl))

    def lfind(i):
        while lparent[i] != i:
            lparent[i] = lparent[lparent[i]]
            i = lparent[i]
        return i
    for i, j in cKDTree(ends_img).query_pairs(1e-6):
        (la, ga), (lb, gb) = owner[i], owner[j]
        if line_name[la] == line_name[lb]:
            lparent[lfind(lb * len(g) + gb)] = lfind(la * len(g) + ga)

    def touching(a, b):
        (ka, ca, ga), (kb, cb, gb) = a, b
        if ka != kb:
            return False
        if ka == "face":
            return sheet(ca, ga) == sheet(cb, gb)
        if ka == "line":
            return lfind(ca * len(g) + ga) == lfind(cb * len(g) + gb)
        return False

    def pack(focus):
        items = sorted(((k, r) for k, r in nb[focus].items() if r in SHOWN[focus[0]]),
                       key=lambda kv: (RANK.get(kv[1], 9), kv[0]))
        if focus[0] == "face":
            # walls across a shared line, and only the regions on the sides of walls shown
            shown_faces = {(k[1], k[2]) for k, _ in items if k[0] == "face"} | {(focus[1], 0)}
            sides_ok = set()
            for fc, fg in shown_faces:
                for sd in face_sides[fc]:
                    sides_ok.add((sd["class"], int(mult[fg, sd["g"]])))
            items = [(k, r) for k, r in items
                     if not (k[0] == "region" and r in ("meets", "across a line") and (k[1], k[2]) not in sides_ok)
                     and not (k[0] == "point" and r == "across a line")]
        mem = [k for k, _ in items]
        rel = dict(items)
        # the focus's own placement joins its group too (a wall's other pieces round it)
        me = (focus[0], focus[1], 0)
        allm = [me] + mem
        parent = list(range(len(allm)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(allm)):
            for j in range(i + 1, len(allm)):
                if touching(allm[i], allm[j]):
                    parent[find(j)] = find(i)
        groups = defaultdict(list)
        for i, m in enumerate(allm):
            groups[find(i)].append(m)
        nodes = []
        for root_, ms in groups.items():
            if me in ms:
                others = [m for m in ms if m != me]
                if others:
                    nodes.append({"kind": me[0], "focus": True,
                                  "members": [[m[1], m[2]] for m in others], "rel": "same shape"})
                continue
            nodes.append({"kind": ms[0][0], "members": [[m[1], m[2]] for m in ms],
                          "rel": min((rel[m] for m in ms), key=lambda r: RANK.get(r, 9))})
        return nodes

    data = {
        "_about": __doc__,
        "group": np.round(g.reshape(len(g), 16), 9).tolist(),
        "regions": [{"id": r["id"], "name": r["region"], "stab": r["stabilizer_order"],
                     "c": region_pos[i].round(6).tolist(), "nb": pack(("region", r["id"]))}
                    for i, r in enumerate(atlas["regions"])],
        "faces": [{"id": f["id"], "name": f["wall"], "regions": f["regions"], "stab": f["stabilizer_order"],
                   "c": face_geo[i]["c"].round(6).tolist(),
                   "cloud": face_geo[i]["cloud"][:: max(1, len(face_geo[i]["cloud"]) // 60)].round(5).tolist(),
                   "side": [s.round(6).tolist() for s in face_geo[i]["side"]],
                   "sides": f["sides"], "nb": pack(("face", f["id"]))}
                  for i, f in enumerate(atlas["faces"])],
        "lines": [{"id": lp["id"], "name": lp["line_shape"], "stab": lp["stabilizer_order"],
                   "poly": [P[:: max(1, len(P) // 40)].round(5).tolist() + [P[-1].round(5).tolist()]
                            for P in line_geo[i]],
                   "nb": pack(("line", lp["id"]))}
                  for i, lp in enumerate(atlas["line_pieces"])],
        "points": [{"id": p["id"], "name": p["atlas_point"], "stab": p["stabilizer_order"],
                    "c": np.round(p["R4"], 6).tolist(), "nb": pack(("point", p["id"]))}
                   for p in atlas["points"] if p.get("atlas_point")],
        "meetings": len(meetings),
    }
    (root / "shape_map3d.json").write_text(json.dumps(data, separators=(",", ":")))
    _log(f"shape map data: {len(data['regions'])} regions, {len(data['faces'])} wall pieces, "
         f"{len(data['lines'])} line pieces, {len(data['points'])} named points, "
         f"{len(meetings)} meeting-curve rings")
    return data


def write_html() -> str:
    """shape_map3d.html: the viewer (shape_map3d_template.html) with the data inside it."""
    from pathlib import Path
    root = data_dir() / "sphere_map"
    data = (root / "shape_map3d.json").read_text()
    tpl = (Path(__file__).with_name("shape_map3d_template.html")).read_text()
    out = root / "shape_map3d.html"
    page = tpl.replace("__DATA__", data.replace("</", "<\\/"))
    out.write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   '<style>body{margin:0;font-size:14px}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
                   + page + "\n</body>\n</html>\n")
    (root / ".cache" / "shape_map3d_page.html").write_text(page)      # (the page alone)
    return str(out)


if __name__ == "__main__":
    import sys
    if "--html-only" not in sys.argv:
        build()
    print(write_html())
