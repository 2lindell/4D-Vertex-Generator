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
    from .chords import FACE_N

    # every cut end (where the domain cuts a line piece), to match the two sides of each cut
    end_pts, end_owner = [], []
    for e in boundary:
        for end in edges[e]["ends"]:
            if end["kind"] == "domain":
                x = np.asarray(end["point"], float)
                end_pts.append(x / x.sum())
                end_owner.append(e)
    etree = cKDTree(np.array(end_pts)) if end_pts else None
    joins, open_ends = [], 0
    for e, p in zip(end_owner, end_pts):
        # across the face the cut lies on (perpendicular, so it surely leaves the domain)
        f = int(np.argmin(np.abs(p @ FACE_N.T) / np.linalg.norm(FACE_N, axis=1)))
        n = FACE_N[f] / np.linalg.norm(FACE_N[f])
        y = p - 1e-7 * n
        try:
            k, _ = locate(point_from_beta(y / y.sum())[None], tol=1e-13)
        except ValueError:
            open_ends += 1
            continue
        k = int(k[0])
        if k == 0:
            open_ends += 1
            continue
        # the cut point as seen from copy k (it lies on that copy's face too)
        pk = np.linalg.solve(grp.g[k], point_from_beta(p))          # G[k]^-1 p
        bk = pk @ np.linalg.inv(np.asarray(_cell(root)))
        bk = bk / bk.sum()
        hits = etree.query_ball_point(bk, 1e-6)
        cands = [end_owner[j] for j in hits]
        if not cands:
            open_ends += 1
            continue
        for e2 in set(cands):
            joins.append((col[e], k, col[e2]))
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
            "length": length, "stabilizer_order": len(orb["stabilizer"]),
            "copies_on_sphere": grp.n // len(orb["stabilizer"]),
        })
    _log(f"line piece classes: {len(lines_out)}")

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
            outline = []
            for k, e in orb.get("outline", []):
                pl = place_line(k, e)
                if pl is not None and pl not in outline:
                    outline.append(pl)
            surfaces = sorted({(patches[j]["surface"], k) for k, j in orb["members"]
                               if patches[j]["surface"] is not None})
            cong = orb.get("congruence_with_first_orbit") or {}
            faces_out.append({
                "id": fid, "wall": wall, "regions": regs, "orbit": orb["orbit"],
                "sides": [place_region_comp(ca), place_region_comp(cb)],
                "surfaces": [{"surface": s, "copy": k, "equation": lib[s]["name"]}
                             for s, k in surfaces[:8]],
                "lies_in_domain_face": any(patches[j]["face"] for _, j in orb["members"]),
                "outline": outline, "perimeter": orb.get("perimeter"),
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
    faces_on_line = Counter(x["class"] for f in faces_out for x in f["outline"])
    closure = {
        "faces_with_two_region_sides": sum(len(f["sides"]) == 2 for f in faces_out),
        "faces": len(faces_out),
        "faces_with_an_outline": sum(bool(f["outline"]) for f in faces_out),
        "line_pieces_bounding_a_face": sum(1 for ln in lines_out if faces_on_line[ln["id"]]),
        "line_pieces": len(lines_out),
        "line_pieces_with_two_end_points": sum(len(ln["ends"]) == 2 for ln in lines_out),
        "closed_line_pieces (no ends)": sum(len(ln["ends"]) == 0 and ln["open_ends"] == 0
                                           for ln in lines_out),
        "line_pieces_with_open_ends": sum(ln["open_ends"] > 0 for ln in lines_out),
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
        "line_pieces": lines_out,
        "faces": faces_out,
        "regions": region_classes,
    }
    (root / "atlas.json").write_text(json.dumps(db))
    _log(f"atlas.json written: {closure}")
    return db


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


if __name__ == "__main__":
    build()
