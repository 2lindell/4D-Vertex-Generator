"""The closed atlas database: every point, line piece, wall piece (face) and region of the
3-sphere map, one record per class under the symmetry, with equations, incidences and relative
positions.

Classes and placements
    Each element class has a representative. A reference to another element is
    ``{"class": id, "g": k}``: the element is the class's representative carried by group element
    ``k`` (group.npz ``elements[k]``). Composing references: the representative of a class placed
    at ``g`` sees its own reference ``{"class": c, "g": h}`` at ``multiplication[g, h]``.

Elements
    point       a corner (where a line piece ends): exact point, the surfaces through it, the atlas
                name if it is one of the atlas's points
    line piece  an edge run whole from corner to corner (joined where the domain cuts it): its two
                surfaces, its end points, the walls and regions round it, its line shape
    face        one piece of one wall between one region component and another: its wall, its two
                regions, the surface it lies on, its outline of line pieces
    region      one connected volume of a region shape, with the faces that bound it

Counts per wall and per line shape are the numbers of these classes: a wall's pieces are the
faces it is split into by the walls and regions round it, a line's pieces the stretches between
the corners where what surrounds it changes.

Run ``python -m four_d_vertex_generator.sphere_map.atlas_db`` to write ``atlas.json``.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .chords import FACE_N
from .frame import data_dir, locate, point_from_beta


def _log(msg):
    from .build import log
    log(msg)


def _load(root, name):
    return json.loads((root / name).read_text())


class Group:
    def __init__(self, root):
        z = np.load(root / "group.npz")
        self.g = z["elements"]
        self.mult = z["multiplication"].astype(np.int64)
        self.inv = z["inverse"].astype(np.int64)
        self.n = len(self.g)


def _orbits(table: np.ndarray, items: list[int], grp: Group):
    """Orbit classes of components in a (copy, item) -> component table: for each orbit a
    representative component, its members, stabiliser, and the map comp -> (orbit, g)."""
    comps = np.unique(table[:, items])
    placed: dict[int, tuple[int, int]] = {}
    orbits = []
    for c in comps:
        c = int(c)
        if c in placed:
            continue
        ks, js = np.nonzero(table[:, items] == c)
        k0, i0 = int(ks[0]), int(items[js[0]])
        o = len(orbits)
        images = table[grp.mult[:, k0], i0]           # component of h . (k0, i0) for every h
        stab = [int(h) for h in np.flatnonzero(images == c)]
        for h in range(grp.n):
            placed.setdefault(int(images[h]), (o, h))
        orbits.append({"rep_component": c, "members": [[int(k), int(items[j])] for k, j in
                                                        zip(ks, js)],
                       "stabilizer": stab})
    return orbits, placed


def build(root=None) -> dict:
    from .sphere_faces import edge_is_boundary

    root = root or data_dir() / "sphere_map"
    grp = Group(root)
    lib = _load(root, "surface_library.json")["surfaces"]
    pieces = _load(root, "pieces.json")["pieces"]
    region_of_piece = [p["region"] for p in pieces]
    patches = _load(root, "patches.json")["patches"]
    faces_doc = _load(root, "faces.json")["faces"]
    edoc = _load(root, "edges.json")
    edges, corners = edoc["edges"], edoc["corners"]
    regions_doc = _load(root, "regions.json")["regions"]
    line_doc = _load(root, "line_pieces.json")["lines"]
    rcomp = np.load(root / "components.npz")["region_component"]

    # ------------------------------------------------------------------ line pieces
    def has_length(e):
        line = np.asarray(e["polyline_beta"])
        return np.max(np.linalg.norm(line - line[0], axis=1)) > 1e-6

    degenerate = sum(not has_length(e) for e in edges)
    boundary = [i for i, e in enumerate(edges)
                if has_length(e) and edge_is_boundary(e, region_of_piece)]
    _log(f"edges: {len(edges)}, of which {degenerate} have no length (a wall meeting a domain face "
         f"exactly at the domain's edge) and are dropped")
    bset = set(boundary)
    col = {e: n for n, e in enumerate(boundary)}

    # what surrounds each line piece: the cyclic order of regions met on a small circle round it
    # (24 directions in its normal plane), the same whichever way round it is read
    from .frame import surface_values
    surfs_lib = _library_surfaces(root)
    keys = _piece_keys(root)

    def circle_signature(e):
        line = np.asarray(edges[e]["polyline_beta"])
        m = len(line) // 2
        x = line[m] / line[m].sum()
        t = line[min(m + 1, len(line) - 1)] - line[max(m - 1, 0)]
        basis = np.linalg.svd(np.vstack([np.ones(4), t]))[2][2:]      # normal plane (2 vectors)
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = x + 2e-6 * (np.cos(ang)[:, None] * basis[0] + np.sin(ang)[:, None] * basis[1])
        out = []
        for y in pts:
            if (y @ FACE_N.T < 0).any():
                _, yb = locate(point_from_beta(y / y.sum())[None], tol=1e-13)
                y = yb[0]
            pc = keys.get(np.sign(surface_values(y, surfs_lib)[0]).astype(np.int8).tobytes())
            out.append(region_of_piece[pc] if pc is not None else "?")
        seq = [r for i, r in enumerate(out) if r != out[i - 1]] or out[:1]   # cyclic runs
        rots = [tuple(seq[i:] + seq[:i]) for i in range(len(seq))]
        rots += [tuple(r[::-1]) for r in rots]
        return min(rots)

    sig = {e: circle_signature(e) for e in boundary}
    _log(f"surroundings read round {len(sig)} line pieces: {len(set(sig.values()))} kinds")

    # joins on the sphere: every end of every edge in every copy is a point of S^3; two ends at
    # the same point join when the curve goes on smoothly and what surrounds it is the same
    # (where the domain cuts a line, or at a 'corner' where nothing round the line changes)
    end_x, end_t, end_e = [], [], []
    for e in boundary:
        line = np.asarray(edges[e]["polyline_beta"])
        for seq in (line, line[::-1]):
            p = seq[0]
            q = next((y for y in seq[1:] if np.linalg.norm(y - p) > 1e-7), None)
            if q is None:
                continue
            xp, xq = point_from_beta(p), point_from_beta(q)
            tang = xp - xq
            end_x.append(xp)
            end_t.append(tang / np.linalg.norm(tang))
            end_e.append(e)
    end_x, end_t = np.array(end_x), np.array(end_t)
    all_x = np.einsum("gij,nj->gni", grp.g, end_x).reshape(-1, 4)
    all_t = np.einsum("gij,nj->gni", grp.g, end_t).reshape(-1, 4)
    tree = cKDTree(all_x)
    n_end = len(end_e)
    joins, open_ends = [], 0
    for i, (e, x, tg) in enumerate(zip(end_e, end_x, end_t)):
        found = False
        for j in tree.query_ball_point(x, 1e-6):
            k, i2 = divmod(j, n_end)
            e2 = end_e[i2]
            if (k == 0 and i2 == i) or sig[e2] != sig[e]:
                continue
            if float(all_t[j] @ tg) > -0.99:       # the other end must leave the other way
                continue
            joins.append((col[e], int(k), col[e2]))
            found = True
        open_ends += not found
    from .build import _components, sphere_edges
    lcomp = _components(grp.n * len(boundary), *sphere_edges([], joins, len(boundary)))
    lcomp = lcomp.reshape(grp.n, len(boundary))
    _log(f"line pieces: {len(boundary)} boundary edges, {len(joins)} joins across domain cuts, "
         f"{open_ends} cut ends with no continuation found")

    # line shape name of each edge: the atlas line lying on it, else the classified shape
    on_line = defaultdict(Counter)
    for name, segs in line_doc.items():
        for seg in segs:
            if seg["edge"] is not None:
                on_line[seg["edge"]][name] += seg["to"] - seg["from"] + 1
    lorbits, lplace = _orbits(lcomp, list(range(len(boundary))), grp)

    # ------------------------------------------------------------------ points
    keyed: dict[tuple, int] = {}
    point_classes = []
    corner_class = {}
    for c in corners:
        p = point_from_beta(np.asarray(c["beta"], float))
        imgs = np.round(grp.g @ p, 8) + 0.0
        keys = [tuple(r) for r in imgs]
        hit = next((keyed[k] for k in keys if k in keyed), None)
        if hit is None:
            hit = len(point_classes)
            distinct = len(set(keys))
            point_classes.append({"id": hit, "beta": c["beta"], "copy": 0,
                                  "R4": p.tolist(), "exact": c["exact"],
                                  "surfaces": c["surfaces"],
                                  "equations": [lib[s]["name"] for s in c["surfaces"]],
                                  "atlas_point": c["atlas_point"],
                                  "copies_on_sphere": distinct,
                                  "stabilizer_order": grp.n // distinct})
            for k in keys:
                keyed[k] = hit
        corner_class[c["id"]] = hit
        if c["atlas_point"] and not point_classes[hit]["atlas_point"]:
            point_classes[hit]["atlas_point"] = c["atlas_point"]

    def place_point(beta, copy):
        """(point class, g) of the point beta in copy `copy`."""
        p = grp.g[copy] @ point_from_beta(np.asarray(beta, float))
        key = tuple(np.round(p, 8) + 0.0)
        cls = keyed.get(key)
        if cls is None:
            return None
        rep = point_from_beta(np.asarray(point_classes[cls]["beta"], float))
        d = np.linalg.norm(grp.g @ rep - p, axis=1)
        return {"class": cls, "g": int(np.argmin(d))}

    # ------------------------------------------------------------------ line piece classes
    lines_out = []
    for o, orb in enumerate(lorbits):
        segs = orb["members"]                       # (copy, column) of the D-edges in it
        ends, names, walls, regs, length = [], Counter(), set(), set(), 0.0
        surfs = set()
        for k, ci in segs:
            e = edges[boundary[ci]]
            names.update(on_line.get(boundary[ci], {}))
            walls |= {w["wall"] for w in e["walls_round"] if w["wall"]}
            regs |= set(e["regions_round"])
            surfs.add(tuple(e["surfaces"]))
            pts = point_from_beta(np.asarray(e["polyline_beta"])) @ grp.g[k].T
            length += float(np.sum(np.arccos(np.clip(np.sum(pts[1:] * pts[:-1], axis=1), -1, 1))))
            for end in e["ends"]:
                if end["kind"] == "corner":
                    ends.append(place_point(end["point"], k))
        shape = names.most_common(1)[0][0] if names else (
            edges[boundary[segs[0][1]]]["line_shape"] if isinstance(
                edges[boundary[segs[0][1]]]["line_shape"], str) else None)
        lines_out.append({
            "id": o, "line_shape": shape, "on_atlas_line": dict(names),
            "segments": [{"copy": k, "edge": boundary[ci],
                          "surfaces": edges[boundary[ci]]["surfaces"],
                          "equations": edges[boundary[ci]]["surface_names"]} for k, ci in segs],
            "ends": [x for x in ends if x is not None],
            "open_ends": sum(1 for x in ends if x is None),
            "walls_round": sorted(walls), "regions_round": sorted(regs),
            "surroundings": list(sig[boundary[segs[0][1]]]),
            "length": length, "stabilizer_order": len(orb["stabilizer"]),
            "copies_on_sphere": grp.n // len(orb["stabilizer"]),
        })
    _log(f"line piece classes: {len(lines_out)}")

    # congruence types: classes with the same surroundings that some element of G or G.Q carries
    # onto one another are one piece type
    q = np.load(root.parent / "source" / "normalizer_q.npy")
    maps = np.concatenate([grp.g, grp.g @ q])
    clouds = []
    for orb in lorbits:
        pts = [point_from_beta(np.asarray(edges[boundary[ci]]["polyline_beta"])) @ grp.g[k].T
               for k, ci in orb["members"]]
        clouds.append(np.vstack(pts))
    by_sig = defaultdict(list)
    for ln in lines_out:
        by_sig[(ln["line_shape"], tuple(ln["surroundings"]))].append(ln["id"])
    n_types = 0
    for _key, ids in by_sig.items():
        types: list[int] = []
        for i in ids:
            hit = None
            for t in types:
                a, b = clouds[t], clouds[i]
                if abs(lines_out[t]["length"] - lines_out[i]["length"]) > 2e-3 * max(
                        lines_out[t]["length"], 1e-3):
                    continue
                tb = cKDTree(b)
                sub = a[:: max(1, len(a) // 60)]
                for mi, m in enumerate(maps):
                    d, _ = tb.query(sub @ m.T, distance_upper_bound=1e-3)
                    if np.isfinite(d).all() and d.max() < 5e-4:
                        hit = (t, mi)
                        break
                if hit:
                    break
            if hit is None:
                types.append(i)
                lines_out[i]["piece_type"] = n_types
                lines_out[i]["congruent_to"] = None
                n_types += 1
            else:
                lines_out[i]["piece_type"] = lines_out[hit[0]]["piece_type"]
                lines_out[i]["congruent_to"] = {"line_piece": hit[0], "map": hit[1]}
    _log(f"line piece types (congruence classes): {n_types}")

    from .line_atlas import MIN_PIECE, build_line_pieces
    atlas_lines, place_atlas = build_line_pieces(grp, line_doc, _log, edges)

    def place_line(copy, edge):
        if edge not in bset:
            return None
        o, h = lplace[int(lcomp[copy, col[edge]])]
        return {"class": o, "g": int(h)}

    # ------------------------------------------------------------------ regions
    region_classes, region_place = [], {}
    for name, info in sorted(regions_doc.items()):
        for orb in info["orbits"]:
            rid = len(region_classes)
            k0, p0 = orb["members"][0]
            images = rcomp[grp.mult[:, k0], p0]
            for h in range(grp.n):
                region_place.setdefault(int(images[h]), (rid, h))
            region_classes.append({"id": rid, "region": name, "shape": info["name"],
                                   "orbit": orb["orbit"],
                                   "components_on_sphere": orb["components"],
                                   "stabilizer_order": orb["stabilizer_order"],
                                   "volume_share_of_sphere": info["volume_share"] /
                                   len(info["orbits"]),
                                   "members": orb["members"], "faces": []})

    def place_region_comp(cid):
        rid, h = region_place[int(cid)]
        return {"class": rid, "g": int(h)}

    def side_comps(k, j):
        p = patches[j]
        a, b = p["pieces"]
        kb = int(grp.mult[k, p["side_b_copy_shift"]])
        return int(rcomp[k, a]), int(rcomp[kb, b])

    # ------------------------------------------------------------------ faces
    faces_out = []
    for label, info in sorted(faces_doc.items()):
        wall, *regs = [x.strip() for x in label.split("|")]
        for orb in info["orbits"]:
            fid = len(faces_out)
            k0, j0 = orb["members"][0]
            ca, cb = side_comps(k0, j0)
            outline, unlisted = [], []
            for k, e in orb.get("outline", []):
                pl = place_atlas(k, e)
                if pl is not None:
                    if pl not in outline:
                        outline.append(pl)
                    continue
                ed = edges[e]
                pts = point_from_beta(np.asarray(ed["polyline_beta"]))
                if float(np.sum(np.arccos(np.clip(np.sum(pts[1:] * pts[:-1], axis=1), -1, 1)))) \
                        < MIN_PIECE:
                    continue                        # (touches a corner only)
                rec = {"edge": int(e), "copy": int(k),               # a curve on no atlas line
                       "line_shape": ed["line_shape"] if isinstance(ed["line_shape"], str) else None,
                       "walls_round": sorted({w["wall"] for w in ed["walls_round"] if w["wall"]})}
                if rec not in unlisted:
                    unlisted.append(rec)
            surfaces = sorted({(patches[j]["surface"], k) for k, j in orb["members"]
                               if patches[j]["surface"] is not None})
            cong = orb.get("congruence_with_first_orbit") or {}
            faces_out.append({
                "id": fid, "wall": wall, "regions": regs, "orbit": orb["orbit"],
                "sides": [place_region_comp(ca), place_region_comp(cb)],
                "surfaces": [{"surface": s, "copy": k, "equation": lib[s]["name"]}
                             for s, k in surfaces[:8]],
                "lies_in_domain_face": any(patches[j]["face"] for _, j in orb["members"]),
                "outline": outline, "outline_curves_on_no_atlas_line": unlisted,
                "perimeter": orb.get("perimeter"),
                "members": orb["members"],
                "components_on_sphere": orb["components"],
                "stabilizer_order": orb["stabilizer_order"],
                "congruent_to": None if orb["orbit"] == info["orbits"][0]["orbit"] else {
                    "face": fid - orb["orbit"] if False else None, "map": cong.get("map"),
                    "status": cong.get("status")},
            })
        # (congruence: every orbit of a label is compared with its first orbit)
        first = len(faces_out) - len(info["orbits"])
        for f in faces_out[first + 1:]:
            if f["congruent_to"] is not None:
                f["congruent_to"]["face"] = first
    _log(f"face classes: {len(faces_out)}")

    # region boundaries: every face placement whose side is the region's representative
    for f in faces_out:
        for side in f["sides"]:
            r = region_classes[side["class"]]
            # the face placed at g^-1 has this side at the region's representative
            r["faces"].append({"class": f["id"], "g": int(grp.inv[side["g"]])})

    # ------------------------------------------------------------------ counts and closure
    per_wall = defaultdict(lambda: {"piece_types": 0, "face_classes": 0, "faces_on_sphere": 0,
                                    "unplaced_fragments": 0, "pieces": []})
    for label, info in faces_doc.items():
        wall = label.split("|")[0].strip()
        whole = [o for o in info["orbits"]
                 if (o.get("congruence_with_first_orbit") or {}).get("status") != "not congruent"]
        frags = len(info["orbits"]) - len(whole)
        w = per_wall[wall]
        w["piece_types"] += 1                  # every orbit of a label is congruent to the first
        w["face_classes"] += len(whole)
        w["faces_on_sphere"] += sum(o["components"] for o in whole)
        w["unplaced_fragments"] += frags
        w["pieces"].append({"between": [x.strip() for x in label.split("|")[1:]],
                            "classes": len(whole), "on_sphere": sum(o["components"] for o in whole)})
    # line pieces per line shape, from the atlas's own lines split where what surrounds them
    # changes (line_pieces.json): a piece type is one set of walls and regions round the line
    per_line = {}
    for name, segs in sorted(line_doc.items()):
        runs, prev = [], None
        covered = total = 0
        for seg in segs:
            n_pts = seg["to"] - seg["from"] + 1
            total += n_pts
            if seg["edge"] is None:
                prev = None
                continue
            covered += n_pts
            sig = (tuple(sorted({w["wall"] for w in seg["walls_round"] if w["wall"]})),
                   tuple(seg["regions_round"]))
            if sig != prev:
                runs.append(sig)
            prev = sig
        types = sorted(set(runs))
        per_line[name] = {
            "piece_types": len(types),
            "runs_in_domain": len(runs),
            "types": [{"walls_round": list(w), "regions_round": list(r)} for w, r in types],
            "share_of_line_on_traced_edges": round(covered / max(total, 1), 3),
        }
    for name in list(per_line):
        per_line[name] = {"from_traced_edges": per_line[name]}
    for c in atlas_lines:
        e = per_line.setdefault(c["line_shape"], {})
        e.setdefault("piece_types", set()).add(c["piece_type"])
        e["classes"] = e.get("classes", 0) + 1
        e["pieces_on_sphere"] = e.get("pieces_on_sphere", 0) + c["copies_on_sphere"]
        e["closed_loops"] = e.get("closed_loops", 0) + int(c["closed"])
    for e in per_line.values():
        if "piece_types" in e:
            e["piece_type_ids"] = sorted(e["piece_types"])
            e["piece_types"] = len(e["piece_types"])
    traced = defaultdict(lambda: {"piece_types": set(), "classes": 0, "pieces_on_sphere": 0})
    for ln in lines_out:
        t = traced[ln["line_shape"] or "unnamed"]
        t["piece_types"].add(ln["piece_type"])
        t["classes"] += 1
        t["pieces_on_sphere"] += ln["copies_on_sphere"]
    for name, t in traced.items():
        entry = per_line.setdefault(name, {}).setdefault("from_traced_edges", {})
        entry["traced_piece_types"] = len(t["piece_types"])
        entry["traced_classes"] = t["classes"]
        entry["traced_pieces_on_sphere"] = t["pieces_on_sphere"]
    # traced edges (where walls meet) that lie on no atlas line, by the walls meeting there
    unlisted = defaultdict(lambda: {"edges": [], "length_in_domain": 0.0})
    for e in boundary:
        if place_atlas(0, e) is not None:
            continue
        ed = edges[e]
        walls_meeting = tuple(sorted({w["wall"] for w in ed["walls_round"] if w["wall"]}))
        shape = ed["line_shape"] if isinstance(ed["line_shape"], str) else None
        u = unlisted[(walls_meeting, shape)]
        u["edges"].append(e)
        pts = point_from_beta(np.asarray(ed["polyline_beta"]))
        u["length_in_domain"] += float(np.sum(np.arccos(np.clip(
            np.sum(pts[1:] * pts[:-1], axis=1), -1, 1))))
    curves_not_on_atlas_lines = sorted(
        ({"walls_meeting": list(k[0]), "classifies_as_line": k[1], "edges": v["edges"],
          "length_in_domain": round(v["length_in_domain"], 6)} for k, v in unlisted.items()),
        key=lambda x: -x["length_in_domain"])
    faces_on_line = Counter(x["class"] for f in faces_out for x in f["outline"])
    closure = {
        "faces_with_two_region_sides": sum(len(f["sides"]) == 2 for f in faces_out),
        "faces": len(faces_out),
        "faces_with_an_outline": sum(bool(f["outline"]) for f in faces_out),
        "line_pieces_bounding_a_face": sum(1 for ln in atlas_lines if faces_on_line[ln["id"]]),
        "line_pieces": len(atlas_lines),
        "line_pieces_with_two_end_points": sum(len(ln["end_points"]) >= 1 and not ln["closed"]
                                               for ln in atlas_lines),
        "closed_loop_line_pieces": sum(ln["closed"] for ln in atlas_lines),
        "faces_with_outline_curves_on_no_atlas_line": sum(
            bool(f["outline_curves_on_no_atlas_line"]) for f in faces_out),
        "regions_with_faces": sum(bool(r["faces"]) for r in region_classes),
        "regions": len(region_classes),
        "points": len(point_classes),
        "atlas_points_found": sorted({p["atlas_point"] for p in point_classes
                                      if p["atlas_point"]}),
    }
    db = {
        "_about": __doc__,
        "counts_per_wall": dict(sorted(per_wall.items())),
        "counts_per_line_shape": per_line,
        "closure": closure,
        "points": point_classes,
        "line_pieces": atlas_lines,
        "curves_not_on_atlas_lines": curves_not_on_atlas_lines,
        "traced_edges": lines_out,
        "faces": faces_out,
        "regions": region_classes,
    }
    (root / "atlas.json").write_text(json.dumps(db))
    _log(f"atlas.json written: {closure}")
    return db


def _library_surfaces(root):
    from .frame import Surface
    out = []
    for d in json.loads((root / "surface_library.json").read_text())["surfaces"]:
        if d["kind"] == "plane":
            out.append(Surface(d["index"], d["name"], "plane", np.asarray(d["normal"], float)))
        else:
            out.append(Surface(d["index"], d["name"], "curved", None,
                               tuple(tuple(m) for m in d["monomials"]),
                               np.asarray(d["coefficients"], float), np.asarray(d["A"], float),
                               degree=d["degree"]))
    return tuple(out)


def _piece_keys(root):
    pcs = json.loads((root / "pieces.json").read_text())["pieces"]
    out = {}
    for p in pcs:
        sg = np.array([1 if c == "+" else -1 if c == "-" else 0 for c in p["signs"]], np.int8)
        out[sg.tobytes()] = p["id"]
    return out


def _cell(root):
    return np.load(root / "group.npz")["cell_vertices"]


def _surface_functions(lib):
    fs = []
    for s in lib:
        if s["kind"] == "plane":
            n = np.asarray(s["normal"])
            fs.append(lambda b, n=n: float(np.asarray(b) @ n / np.sum(b)))
        else:
            A = np.asarray(s["A"])
            mons, co = s["monomials"], s["coefficients"]

            def f(b, A=A, mons=mons, co=co):
                y = A @ (np.asarray(b) / np.sum(b))
                return float(sum(c * np.prod(y[m]) for c, m in zip(co, mons)))
            fs.append(f)
    return fs




def _apply_corrections(w, path) -> None:
    """User-confirmed corrections (corrections.json) applied on top of the working data."""
    if not path.exists():
        return
    c = json.loads(path.read_text())
    for e in c.get("add_edges", []):
        if e["line"] not in w["walls"][e["wall"]]["edges"]:
            w["walls"][e["wall"]]["edges"].append(e["line"])
        bw = w["lines"][e["line"]].setdefault("bounds_walls", [])
        if e["wall"] not in bw:
            bw.append(e["wall"])
    for e in c.get("remove_edges", []):
        if e["line"] in w["walls"][e["wall"]]["edges"]:
            w["walls"][e["wall"]]["edges"].remove(e["line"])
        bw = w["lines"][e["line"]].get("bounds_walls", [])
        if e["wall"] in bw:
            bw.remove(e["wall"])


def audit(root=None) -> dict:
    """Check every wall piece's edges and every line's walls against the working data, one for one.

    Data side: each wall's ``edges`` (lines, or other walls it meets along an unnamed curve) and
    ``cuts``; each line's ``bounds_walls``; each region's ``walls``. Map side: the faces' outlines
    (atlas line pieces, and traced curves on no atlas line, named by the other walls meeting there)
    and the regions' bounding faces.
    """
    root = root or data_dir() / "sphere_map"
    db = json.loads((root / "atlas.json").read_text())
    w = json.loads((root.parent / "source" / "working_data.json").read_text())
    _apply_corrections(w, root.parent / "corrections.json")
    lines = db["line_pieces"]
    line_names = set(w["lines"])
    per_face, found_edges, found_line_walls = [], defaultdict(set), defaultdict(set)
    for f in db["faces"]:
        edges_here = set()
        for ref in f["outline"]:
            ln = lines[ref["class"]]["line_shape"]
            edges_here.add(ln)
            found_line_walls[ln].add(f["wall"])
        for tr in f.get("outline_curves_on_no_atlas_line", []):
            if tr.get("line_shape") in line_names:
                # on no recorded piece of the line, but the curve is that line (its points give the
                # line's polytope): the line's record misses this stretch
                edges_here.add(tr["line_shape"])
                found_line_walls[tr["line_shape"]].add(f["wall"])
                continue
            others = set(tr["walls_round"]) - {f["wall"]}
            listed = {e for e in w["walls"].get(f["wall"], {}).get("edges", []) if e in others}
            # where several walls meet along one curve the data names it by one of them
            edges_here |= ({f"wall {o}" for o in listed} if listed else
                           {f"wall {o}" for o in others} or {"(unnamed curve)"})
        per_face.append({"face": f["id"], "wall": f["wall"], "regions": f["regions"],
                         "orbit": f["orbit"], "edges_found": sorted(edges_here)})
        found_edges[f["wall"]] |= edges_here
    walls_report = {}
    for wall, info in sorted(w["walls"].items()):
        data_edges = {e if e in line_names else f"wall {e}" for e in info["edges"]}
        data_cuts = {f"wall {c['by']}" for c in info.get("cuts", [])}
        found = found_edges.get(wall, set())
        walls_report[wall] = {
            "data_edges": sorted(data_edges), "data_cuts": sorted(data_cuts),
            "found_edges": sorted(found),
            "missing_on_map": sorted(data_edges - found),
            "not_in_data": sorted(found - data_edges - data_cuts),
            "match": data_edges <= found and not (found - data_edges - data_cuts),
        }
    lines_report = {}
    for ln, info in sorted(w["lines"].items()):
        data = set(info.get("bounds_walls", []))
        found = found_line_walls.get(ln, set())
        lines_report[ln] = {"data_walls": sorted(data), "found_walls": sorted(found),
                            "missing_on_map": sorted(data - found),
                            "not_in_data": sorted(found - data), "match": data == found}
    regions_found = defaultdict(set)
    for f in db["faces"]:
        for r in f["regions"]:
            regions_found[r].add(f["wall"])
    regions_report = {}
    for r, info in sorted(w["regions"].items()):
        data = set(info["walls"])
        found = regions_found.get(r, set())
        regions_report[r] = {"data_walls": sorted(data), "found_walls": sorted(found),
                             "missing_on_map": sorted(data - found),
                             "not_in_data": sorted(found - data), "match": data == found}
    out = {
        "summary": {
            "walls_matching": sum(v["match"] for v in walls_report.values()),
            "walls": len(walls_report),
            "lines_matching": sum(v["match"] for v in lines_report.values()),
            "lines": len(lines_report),
            "regions_matching": sum(v["match"] for v in regions_report.values()),
            "regions": len(regions_report),
        },
        "walls": walls_report, "lines": lines_report, "regions": regions_report,
        "faces": per_face,
    }
    (root / "audit.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    build()
    audit()
