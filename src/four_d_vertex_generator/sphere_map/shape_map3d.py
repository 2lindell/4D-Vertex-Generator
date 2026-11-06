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

from .frame import data_dir, group, load_json, multiplication_table, point_from_beta
from .incidence import Incidence

ROOT = None
RANK = {"side": 0, "boundary": 0, "round": 1, "end": 1, "corner": 1, "touches": 2, "meets": 2,
        "crosses": 2, "ends on": 1, "ends on it": 1, "across a line": 3}
# what each kind of shape shows round it
SHOWN = {"region": {"boundary", "corner"},
         "face": {"side", "boundary", "corner", "meets"},
         "line": {"round", "end", "ends on", "ends on it"},
         "point": {"boundary", "touches"}}


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
    # named: a shape of its own in the atlas (working data 'shapes'); other exact points only mark
    # where lines meet
    point_shapes = {k for k, v in load_json("working_data.json")["shapes"].items() if v["dim"] == "point"}
    named_points = [p["id"] for p in atlas["points"] if p.get("atlas_point") in point_shapes]

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

    # every placed wall piece by its wall and the two region volumes it lies between
    face_by_sides = {}
    faces_by_side = defaultdict(list)          # (wall, one side) -> [(face, g, the other side)]
    for f in atlas["faces"]:
        for gg in range(len(g)):
            sides = [(sd["class"], int(mult[gg, sd["g"]])) for sd in f["sides"]]
            face_by_sides.setdefault((f["wall"], frozenset(sides)), (f["id"], gg))
            for i_ in range(2):
                faces_by_side[(f["wall"], sides[i_])].append((f["id"], gg, sides[1 - i_]))

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
        # walls tangent along the line (a region between them pinched to nothing): each is the
        # wall piece with the region volume on its outer side, the two sharing the pinched volume
        cusp = [wl for wl in r["walls"] if wl.get("cusp_outer")]
        if len(cusp) == 2:
            outer = []
            for wl in cusp:
                c, pc = wl["cusp_outer"]
                outer.append(region_at.get(int(rcomp[mult[k0, c], pc])) if pc is not None else None)
            if all(outer):
                c1 = faces_by_side.get((cusp[0]["wall"], outer[0]), [])
                c2 = faces_by_side.get((cusp[1]["wall"], outer[1]), [])
                for f1, g1, mid1 in c1:
                    hit = [(f2, g2) for f2, g2, mid2 in c2 if mid2 == mid1]
                    if hit:
                        out.append({"kind": "face", "class": f1, "g": g1})
                        out.append({"kind": "face", "class": hit[0][0], "g": hit[0][1]})
                        # (the pinched region touches the line too)
                        out.append({"kind": "region", "class": mid1[0], "g": mid1[1]})
                        break
        for wl in r["walls"]:
            if wl.get("cusp_outer"):
                continue
            at = face_of_wall(k0, wl)
            if at is None and wl.get("wall") and wl.get("sides"):
                # no recorded patch between these two sectors (the exploration did not meet it
                # so close to the line): the wall piece is the named wall between the two region
                # volumes on its sides, which is exact
                vol = []
                for c, pc in wl["sides"]:
                    if pc is None:
                        break
                    vol.append(region_at.get(int(rcomp[mult[k0, c], pc])))
                if len(vol) == 2 and all(vol):
                    at = face_by_sides.get((wl["wall"], frozenset(vol)))
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
            # (at nine points along it: a shape touching only part of the line is still met)
            for j in sorted({int(round(t * (len(poly) - 1))) for t in np.linspace(0.06, 0.94, 9)}
                            - {0, len(poly) - 1}):
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
            rr, seen = [], set()
            for j in sorted({len(P) // 4, len(P) // 2, (3 * len(P)) // 4} - {0, len(P) - 1}):
                for x in ring(P[j], P[j + 1] - P[j - 1], 0):
                    if (x["kind"], x["class"], x["g"]) not in seen:
                        seen.add((x["kind"], x["class"], x["g"]))
                        rr.append(x)
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
    wall_name = {f["id"]: f["wall"] for f in atlas["faces"]}
    wcomp = comps["wall_component"]
    S0 = np.array([[int(wcomp[mult[gg, f["members"][0][0]], f["members"][0][1]]) for gg in range(len(g))]
                   for f in atlas["faces"]])
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
        # shapes round one line are neighbours of each other along it; two walls cross there when
        # one of them carries on through the line (its sheet is on both sides of it)
        sheets_here = [S0[m[1], m[2]] for m in members if m[0] == "face"]
        through = {x for x in sheets_here if sheets_here.count(x) > 1}
        for a in members:
            for b in members:
                if a == b:
                    continue
                if a[0] == b[0] == "face" and (S0[a[1], a[2]] in through or S0[b[1], b[2]] in through) \
                        and S0[a[1], a[2]] != S0[b[1], b[2]]:
                    add(a, b, "crosses")
                else:
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
    sheet_rep = {f["id"]: f["members"][0] for f in atlas["faces"]}

    def sheet(c, gg):
        k0, p0 = sheet_rep[c]
        return int(wcomp[mult[gg, k0], p0])
    line_name = {lp["id"]: lp["line_shape"] for lp in atlas["line_pieces"]}
    ends_img, ends_dir, owner = [], [], []
    for li, geo in enumerate(line_geo):
        first, last = np.vstack(geo[0]), np.vstack(geo[-1])
        for P, Q in ((first[0], first[min(3, len(first) - 1)]), (last[-1], last[max(-4, -len(last))])):
            t = Q - P - (Q - P) @ P * P                      # direction leaving the end, along S^3
            t = t / np.linalg.norm(t)
            ends_img.append(np.einsum("gij,j->gi", g, P))
            ends_dir.append(np.einsum("gij,j->gi", g, t))
            owner += [(li, gg) for gg in range(len(g))]
    ends_img, ends_dir = np.concatenate(ends_img), np.concatenate(ends_dir)
    nl = len(line_geo) * len(g)
    lparent = list(range(nl))

    def lfind(i):
        while lparent[i] != i:
            lparent[i] = lparent[lparent[i]]
            i = lparent[i]
        return i
    for i, j in cKDTree(ends_img).query_pairs(1e-6):
        (la, ga), (lb, gb) = owner[i], owner[j]
        # one line carrying on through a point that is not a shape of its own: the two ends leave
        # it in opposite directions
        if line_name[la] == line_name[lb] and ends_dir[i] @ ends_dir[j] < -0.999 \
                and point_at(ends_img[i]) is None:
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

    def filtered(focus):
        """The piece-level neighbours a shape of this kind shows: [((kind, class, g), relation)]."""
        items = [(k, r) for k, r in nb[focus].items() if r in SHOWN[focus[0]]]
        if focus[0] == "face":
            # walls across a shared line, and only the regions on the sides of walls shown
            # walls that only share a line with it (both ending there) are not shown
            items = [(k, r) for k, r in items if not (k[0] == "face" and r == "across a line")]
            shown_faces = {(k[1], k[2]) for k, _ in items if k[0] == "face"} | {(focus[1], 0)}
            sides_ok = set()
            for fc, fg in shown_faces:
                for sd in face_sides[fc]:
                    sides_ok.add((sd["class"], int(mult[fg, sd["g"]])))
            # a region touches the wall only on its own two sides: regions round a curve where
            # another wall meets it lie beyond that wall unless they are the wall's own sides
            items = [(k, r) for k, r in items
                     if not (k[0] == "region" and r != "side")
                     and not (k[0] == "point" and r == "across a line")]
        return items

    # ---- nodes: a node is a whole named shape on S^3. A wall node is one whole sheet of the wall
    # (the build's wall components: its pieces joined across every curve where other walls meet
    # it); a line node one whole connected line; a region its volume; a point itself.
    n_g = len(g)
    # a wall node: one wall's pieces joined across the unnamed curves where other walls meet it
    # (the same sheet on both sides of such a curve); named lines bound it, since across a named
    # line lies another copy under the symmetry
    nf = len(atlas["faces"])
    fparent = np.arange(nf * n_g)

    def ffind(i):
        r = i
        while fparent[r] != r:
            r = fparent[r]
        while fparent[i] != r:
            fparent[i], i = r, fparent[i]
        return r
    for mt in meetings:
        fs = [(x["class"], x["g"]) for x in mt["round"] if x["kind"] == "face"]
        for i, (c1, g1) in enumerate(fs):
            for c2, g2 in fs[i + 1:]:
                if wall_name[c1] != wall_name[c2] or S0[c1, g1] != S0[c2, g2]:
                    continue
                for h in range(n_g):
                    u, v = ffind(c1 * n_g + int(mult[h, g1])), ffind(c2 * n_g + int(mult[h, g2]))
                    if u != v:
                        fparent[v] = u
    S = np.array([[ffind(c * n_g + gg) for gg in range(n_g)] for c in range(nf)])
    # a line node: one named line piece, from named point to named point; placements that are the
    # same piece (its stabiliser) share its middle point
    quart = []
    for geo in line_geo:
        P = np.vstack(geo)
        quart.append((P[len(P) // 4], P[(3 * len(P)) // 4]))
    key_of = {}

    def piece_key(li, gg):
        a, b = (tuple(np.round(g[gg] @ x, 6)) for x in quart[li])
        return (li,) + tuple(sorted((a, b)))
    # (placements that are the same piece, under its stabiliser, are one)
    for li in range(len(line_geo)):
        for gg in range(n_g):
            k0 = key_of.setdefault(piece_key(li, gg), li * n_g + gg)
            if k0 != li * n_g + gg:
                lparent[lfind(li * n_g + gg)] = lfind(k0)
    L = np.array([[lfind(li * n_g + gg) for gg in range(n_g)] for li in range(len(line_geo))])

    def whole_classes(T, kind_names):
        """Classes of whole shapes from a (piece class x g) -> component table: each class's
        representative component, its pieces (class, g), the classes of pieces it holds, and its
        stabiliser order."""
        out, of_piece = [], {}
        for c in range(len(T)):
            if c in of_piece:
                continue
            s0 = T[c, 0]
            mem = [(int(x), int(y)) for x, y in np.argwhere(T == s0)]
            c1, g1 = mem[0]
            stab = int(sum(T[c1, mult[h, g1]] == s0 for h in range(n_g)))
            cid = len(out)
            for pc in {x for x, _ in mem}:
                of_piece[pc] = cid
            out.append({"id": cid, "name": kind_names[c], "members": mem, "stab": stab, "rep": int(s0)})
        return out, of_piece
    sheets, sheet_of = whole_classes(S, [f["wall"] for f in atlas["faces"]])
    lines_w, line_of = whole_classes(L, [lp["line_shape"] for lp in atlas["line_pieces"]])
    # (a component's own class and the element carrying the class representative onto it)
    sheet_at, line_at = {}, {}
    for cl, T, at in ((sheets, S, sheet_at), (lines_w, L, line_at)):
        for w_ in cl:
            c1, g1 = w_["members"][0]
            for h in range(n_g):
                at.setdefault(int(T[c1, mult[h, g1]]), (w_["id"], h))

    def whole_key(kind, c, gg):
        """The whole shape a placed piece belongs to: a key, its node class and placement."""
        if kind == "face":
            comp = int(S[c, gg])
            return ("wall", comp), ("wall",) + sheet_at[comp]
        if kind == "line":
            comp = int(L[c, gg])
            return ("line", comp), ("line",) + line_at[comp]
        if kind == "point":
            x = g[gg] @ point_pos[c]
            k = (c, tuple(np.round(x, 6)))
            return ("point", k), ("point", c, gg)
        return ("region", c, gg), ("region", c, gg)

    def node_nb(kind, members, own):
        """Every shape round a whole shape (its pieces at (class, g) in the node's frame), as whole
        shapes: where they touch it (their pieces' placements) and the closest relation."""
        out = {}
        for pc, pg in members:
            for (k2, c2, g2), rel in filtered((kind, pc)):
                G2 = int(mult[pg, g2])
                key, node = whole_key(k2, c2, G2)
                if key == own:
                    continue
                e = out.setdefault(key, {"kind": node[0], "node": node[1], "at": node[2], "touch": [],
                                         "rel": rel})
                if [c2, G2] not in e["touch"]:
                    e["touch"].append([c2, G2])
                if RANK[rel] < RANK[e["rel"]]:
                    e["rel"] = rel
        return sorted(out.values(), key=lambda e: (RANK[e["rel"]], e["kind"], e["node"]))

    # a line that stops where it runs into another line: that line is its neighbour there (the
    # end is not a shape of its own, so it is shown as the line it lies on)
    etree = cKDTree(ends_img)

    def line_end_links(members, own):
        out = {}
        for li, gg in members:
            for side in (0, 1):
                i = li * 2 * n_g + side * n_g + gg
                x = ends_img[i]
                if point_at(x) is not None:
                    continue
                near = [owner[j] for j in etree.query_ball_point(x, 1e-6)]
                # the line carries on through x (another of its own pieces starts there): the other
                # lines there end on it; otherwise it ends on them
                carries_on = sum(int(L[lj, gj]) == own for lj, gj in near) > 1
                rel = "ends on it" if carries_on else "ends on"
                for lj, gj in near:
                    comp = int(L[lj, gj])
                    if comp == own:
                        continue
                    key = ("line", comp, rel)
                    e = out.setdefault(key, {"kind": "line", "node": line_at[comp][0], "at": line_at[comp][1],
                                             "touch": [], "rel": rel})
                    if [lj, gj] not in e["touch"]:
                        e["touch"].append([lj, gj])
        return list(out.values())

    face_regions = {f["id"]: f["regions"] for f in atlas["faces"]}
    data = {
        "_about": __doc__,
        "group": np.round(g.reshape(len(g), 16), 9).tolist(),
        # pieces: geometry of each class's representative (carried by g where placed)
        "face_geo": [{"c": face_geo[i]["c"].round(6).tolist(),
                      "cloud": face_geo[i]["cloud"][:: max(1, len(face_geo[i]["cloud"]) // 40)].round(5).tolist(),
                      "side": [x.round(6).tolist() for x in face_geo[i]["side"]],
                      "regions": face_regions[f["id"]]}
                     for i, f in enumerate(atlas["faces"])],
        "line_geo": [[P[:: max(1, len(P) // 30)].round(5).tolist() + [P[-1].round(5).tolist()]
                      for P in line_geo[i]] for i in range(len(line_geo))],
        # nodes: whole named shapes
        "walls": [{"id": w_["id"], "name": w_["name"], "stab": w_["stab"], "members": w_["members"],
                   "regions": sorted({r for c, _ in w_["members"] for r in face_regions[c]}),
                   "nb": node_nb("face", w_["members"], ("wall", w_["rep"]))} for w_ in sheets],
        "lines": [{"id": w_["id"], "name": w_["name"], "stab": w_["stab"], "members": w_["members"],
                   "nb": node_nb("line", w_["members"], ("line", w_["rep"]))
                   + line_end_links(w_["members"], w_["rep"])} for w_ in lines_w],
        "regions": [{"id": r["id"], "name": r["region"], "stab": r["stabilizer_order"],
                     "c": region_pos[i].round(6).tolist(),
                     "nb": node_nb("region", [(r["id"], 0)], ("region", r["id"], 0))}
                    for i, r in enumerate(atlas["regions"])],
        "points": [{"id": p["id"], "name": p["atlas_point"], "stab": p["stabilizer_order"],
                    "c": np.round(p["R4"], 6).tolist(),
                    "nb": node_nb("point", [(p["id"], 0)],
                                  ("point", (p["id"], tuple(np.round(point_pos[p["id"]], 6)))))}
                   for p in atlas["points"] if p["id"] in set(named_points)],
        "meetings": len(meetings),
    }
    (root / "shape_map3d.json").write_text(json.dumps(data, separators=(",", ":")))
    _log(f"shape map data: {len(data['regions'])} regions, {len(data['walls'])} whole wall sheets, "
         f"{len(data['lines'])} whole lines, {len(data['points'])} named points, "
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
