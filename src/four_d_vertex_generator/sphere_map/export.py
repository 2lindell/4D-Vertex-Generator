"""Write the 3-sphere map as plain data files (JSON + NumPy), and a readable report."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .chords import FACE_NAMES
from .frame import cell_vertices, group, load_json, multiplication_table
from .surface_images import beta_maps

FORMAT_VERSION = 1


def _inverse_table() -> np.ndarray:
    mult = multiplication_table()
    return np.array([int(np.flatnonzero(mult[g] == 0)[0]) for g in range(len(mult))])


def _sign_text(row: np.ndarray) -> str:
    return "".join("+" if v > 0 else "-" if v < 0 else "0" for v in row)


def _patch_instances(patches, comp):
    """For each patch: the region component on side A and side B in every copy."""
    mult = multiplication_table()
    copies = np.arange(1200)
    side_a, side_b = [], []
    for p in patches:
        a, b = p["pieces"]
        side_a.append(comp[copies, a])
        side_b.append(comp[mult[copies, p["shift"]], b])
    return np.array(side_a).T, np.array(side_b).T          # (1200, patches)


def boundaries(result) -> dict:
    """Boundary of every region's representative component(s): wall parts, by wall and neighbour."""
    patches, comp, wcomp = result["patches"], result["comp"], result["wcomp"]
    labels = result["labels"]
    ca, cb = _patch_instances(patches, comp)
    wall_name = [p["wall"] or f"unnamed {'|'.join(sorted(p['regions']))}" for p in patches]
    wall_sizes = Counter(int(x) for x in wcomp.ravel())
    out = {}
    for region, info in result["regions"].items():
        per_orbit = []
        for orbit in info["orbits"]:
            cid = orbit["representative"]
            ks_a, is_a = np.nonzero(ca == cid)
            ks_b, is_b = np.nonzero(cb == cid)
            groups: dict[tuple, dict] = {}
            for side, ks, iis in (("A", ks_a, is_a), ("B", ks_b, is_b)):
                for k, i in zip(ks, iis):
                    other_comp = int(cb[k, i] if side == "A" else ca[k, i])
                    p = patches[i]
                    other_region = p["regions"][1] if side == "A" else p["regions"][0]
                    key = (wall_name[i], other_region)
                    g = groups.setdefault(key, {"wall": wall_name[i], "other_region": other_region,
                                                "parts": [], "wall_components": Counter(),
                                                "other_components": set()})
                    g["parts"].append([int(k), int(i)])
                    g["wall_components"][int(wcomp[k, i])] += 1
                    g["other_components"].add(other_comp)
            faces = []
            for g in sorted(groups.values(), key=lambda x: (x["wall"], x["other_region"])):
                wc = []
                for w, used in sorted(g["wall_components"].items()):
                    total = wall_sizes[w]
                    wc.append({"wall_component": w, "parts_here": used, "parts_in_component": total,
                               "whole": used == total})
                faces.append({
                    "wall": g["wall"],
                    "other_region": g["other_region"],
                    "parts": g["parts"],
                    "wall_components": wc,
                    "other_region_components": sorted(g["other_components"]),
                })
            per_orbit.append({"orbit": orbit["orbit"], "representative": cid, "faces": faces})
        out[region] = per_orbit
    del labels
    return out


def q_pairing(result) -> dict:
    """For each region orbit: the orbit its image under the half-turn Q (outside G) lies in."""
    from .frame import locate, point_from_beta, source_dir, surface_values

    q = np.load(source_dir() / "normalizer_q.npy")
    g = group()
    ex, comp = result["ex"], result["comp"]
    surfs = result["lib"].surfaces()
    out = {}
    for region, info in result["regions"].items():
        comp_orbit = {}
        for o in info["orbits"]:
            for c in np.unique(comp[:, o["domain_pieces"]]):
                comp_orbit[int(c)] = o["orbit"]
        res = []
        for o in info["orbits"]:
            k, piece = o["members"][0]
            x = g[k] @ point_from_beta(np.asarray(ex.best[piece][1], float))
            copy, beta = locate((q @ x)[None])
            sign = np.sign(surface_values(beta, surfs)[0]).astype(np.int8)
            p2 = ex.keys.get(sign.tobytes())
            if p2 is None or result["labels"][p2] != region:
                res.append(None)
                continue
            res.append(comp_orbit.get(int(comp[int(copy[0]), p2])))
        out[region] = res
    return out


def write_all(result, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    ex = result["ex"]
    labels = result["labels"]
    patches = result["patches"]
    lib = result["lib"]
    shapes = load_json("working_data.json")["shapes"]
    w = load_json("working_data.json")

    # group -----------------------------------------------------------------------------------
    g = group()
    np.savez_compressed(out_dir / "group.npz", elements=g, beta_maps=beta_maps(),
                        multiplication=multiplication_table().astype(np.int16),
                        inverse=_inverse_table().astype(np.int16), cell_vertices=cell_vertices())
    with open(out_dir / "group.json", "w", encoding="utf-8") as h:
        json.dump({
            "_about": "h4_swirlprism (order 1200) on R^4. Copy k of the domain is elements[k] "
                      "applied to the half-cell; elements[0] is the identity. A point with "
                      "barycentric weights beta in the domain is normalize(beta @ cell_vertices); "
                      "the same point in copy k is elements[k] @ that point. multiplication and "
                      "inverse tables, and the 2400 beta maps of G and G.Q, are in group.npz.",
            "cell_vertices": cell_vertices().tolist(),
            "elements": np.round(g, 15).tolist(),
        }, h)

    # pieces ----------------------------------------------------------------------------------
    neighbours = defaultdict(list)
    patch_of = {}
    for i, p in enumerate(patches):
        if p["face"] is None:
            patch_of[(min(p["pieces"]), max(p["pieces"]), p["surface"])] = i
    for (a, b, s), (count, point) in ex.adjacency.items():
        kind = "seam" if labels[a] == labels[b] else "wall"
        pid = patch_of.get((a, b, s))
        for x, y in ((a, b), (b, a)):
            neighbours[x].append({"piece": y, "surface": s, "type": kind, "patch": pid})
    faces = defaultdict(list)
    for (p, f, k, q) in result["glued"]:
        faces[p].append({"face": FACE_NAMES[f], "copy": k, "piece": q,
                         "type": "seam" if labels[p] == labels[q] else "wall"})
    pieces = []
    for i in range(len(ex.signs)):
        pieces.append({
            "id": i,
            "region": labels[i],
            "signs": _sign_text(ex.signs[i]),
            "sample_beta": np.asarray(ex.best[i][1]).tolist(),
            "volume_share": float(result["volumes"][i]),
            "check": result["checks"][i],
            "chord_hits": int(ex.hits[i]),
            "neighbours": neighbours[i],
            "faces": faces[i],
        })
    with open(out_dir / "pieces.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Cells of the surface arrangement in the half-cell. A piece is every "
                             "domain point whose sign over surface_library.json matches 'signs' "
                             "(one character per surface, in library order). 'neighbours' are "
                             "pieces across one surface (a seam inside one region, or a wall "
                             "patch); 'faces' are pieces of neighbouring copies across a domain "
                             "face.", "pieces": pieces}, h)

    # patches ---------------------------------------------------------------------------------
    plist = []
    for i, p in enumerate(patches):
        plist.append({
            "id": i,
            "wall": p["wall"],
            "naming": p["naming"],
            "regions": list(p["regions"]),
            "pieces": list(p["pieces"]),
            "side_b_copy_shift": p["shift"],
            "surface": p["surface"],
            "face": p["face"],
            "sample_beta": None if p["point"] is None else np.asarray(p["point"]).tolist(),
            "chord_hits": p["hits"],
            "verified_by_classification": p.get("verified"),
            "regions_from_piece_labels": (list(p["regions_from_pieces"])
                                          if "regions_from_pieces" in p else None),
            "signature": p.get("signature"),
        })
    with open(out_dir / "patches.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Wall patches in the half-cell: where two pieces of different regions "
                             "meet. An interior patch lies on library surface 'surface' between "
                             "pieces a and b (both in the same copy); a face patch lies in a domain "
                             "face, with side b in copy multiplication[k, side_b_copy_shift] when "
                             "side a is in copy k. The patch is exactly: the points of that surface "
                             "where every other surface has the sign the two pieces share.",
                   "patches": plist}, h)

    # regions ---------------------------------------------------------------------------------
    bnd = boundaries(result)
    qpair = q_pairing(result)
    regions = {}
    for name, info in result["regions"].items():
        shape = shapes.get(name, {})
        items = result["region_pieces"][name]
        regions[name] = {
            "name": shape.get("name"),
            "vertices": shape.get("vertices"),
            "cells": shape.get("cells"),
            "faces": shape.get("faces"),
            "edges": shape.get("edges"),
            "seed_beta": shape.get("seed"),
            "walls_in_working_data": sorted(k for k, v in w["walls"].items()
                                            if name in v["regions"]),
            "domain_pieces": items,
            "volume_share": float(sum(result["volumes"][i] for i in items)),
            "components_on_sphere": info["components"],
            "orbits": [dict({k: v for k, v in o.items() if k != "stabilizer"},
                            stabilizer=o["stabilizer"], q_image_orbit=qo, boundary=b["faces"])
                       for o, b, qo in zip(info["orbits"], bnd[name], qpair[name])],
        }
    with open(out_dir / "regions.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Every region (3D shape family) on the whole 3-sphere. A component is "
                             "one connected volume; components of one orbit are images of each "
                             "other under the group. 'members' lists the representative "
                             "component's (copy, piece) pairs; any other component of the orbit is "
                             "g applied to it (copy k -> multiplication[g, k]). 'boundary' lists "
                             "that component's faces: each wall, the region across it, and the "
                             "exact parts (copy, patch) it is made of.", "regions": regions}, h)

    # walls -----------------------------------------------------------------------------------
    walls = {}
    for name, info in result["walls"].items():
        shape = shapes.get(name, {})
        items = [i for i, p in enumerate(patches)
                 if (p["wall"] or f"unnamed {'|'.join(sorted(p['regions']))}") == name]
        walls[name] = {
            "name": shape.get("name"),
            "vertices": shape.get("vertices"),
            "cells": shape.get("cells"),
            "faces": shape.get("faces"),
            "edges": shape.get("edges"),
            "seed_beta": shape.get("seed"),
            "regions_in_working_data": w["walls"].get(name, {}).get("regions"),
            "regions_found": sorted({r for i in items for r in patches[i]["regions"]}),
            "surfaces_in_working_data": w["exact"]["wall_surfaces"].get(name),
            "surfaces_found": sorted({lib[patches[i]["surface"]].surface.name
                                      for i in items if patches[i]["surface"] is not None}),
            "faces_found": sorted({patches[i]["face"] for i in items if patches[i]["face"]}),
            "patches": items,
            "components_on_sphere": info["components"],
            "orbits": info["orbits"],
        }
    with open(out_dir / "walls.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Every wall (2D shape family) on the whole 3-sphere: its patches in the "
                             "domain and its connected components (sheets), with members as "
                             "(copy, patch) pairs.", "walls": walls}, h)

    # lines and points ------------------------------------------------------------------------
    lines = {}
    for name, info in result["lines"].items():
        shape = shapes.get(name, {})
        lines[name] = dict(info, name=shape.get("name"), vertices=shape.get("vertices"),
                           cells=shape.get("cells"), faces=shape.get("faces"),
                           edges=shape.get("edges"),
                           ends=w["lines"].get(name, {}).get("ends"),
                           bounds_walls=w["lines"].get(name, {}).get("bounds_walls"))
    with open(out_dir / "lines.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Every line (1D shape family): its exact pieces in the domain (betas), "
                             "how many copies it has on the sphere and how they join.",
                   "lines": lines}, h)
    with open(out_dir / "points.json", "w", encoding="utf-8") as h:
        json.dump({"_about": "Every meeting point and named point, with its number of copies on "
                             "the sphere.", "points": result["points"],
                   "named_points": {k: v for k, v in shapes.items() if v["dim"] == "point"}}, h)

    # faces, edges, corners -------------------------------------------------------------------
    fx = result.get("faces")
    if fx:
        def tolist(x):
            return np.asarray(x, float).tolist()

        faces_out = {}
        for label, info in fx["faces"].items():
            wall, *regs = [x.strip() for x in label.split("|")]
            faces_out[label] = {
                "wall": wall, "regions": regs,
                "components_on_sphere": info["components_on_sphere"],
                "orbits": [{
                    "orbit": o["orbit"], "components": o["components"],
                    "stabilizer_order": o["stabilizer_order"], "stabilizer": o["stabilizer"],
                    "copies_spanned": o["copies_spanned"], "domain_patches": o["domain_pieces"],
                    "members": o["members"], "outline": [list(x) for x in o["outline"]],
                    "corners_R4": tolist(o["corners"]), "perimeter": o["perimeter"],
                    "congruent_to_first_orbit_by_map": o["congruent_to_first"],
                } for o in info["orbits"]],
            }
        with open(out_dir / "faces.json", "w", encoding="utf-8") as h:
            json.dump({"_about": "Faces: one connected piece of one wall sheet between one region "
                                 "component and another. Each orbit lists its representative's "
                                 "(copy, patch) members, its outline as (copy, edge) pairs, its "
                                 "corners on S^3 and perimeter, and the map (index into G then "
                                 "G.Q, as group.npz beta_maps order) carrying the first orbit's "
                                 "face onto it (null: not congruent).", "faces": faces_out}, h)
        edges_out = []
        for e in fx["edges"]:
            ends = []
            for end in e["ends"]:
                d = {k: (tolist(v) if isinstance(v, np.ndarray) else v) for k, v in end.items()}
                if d.get("point") is not None:
                    d["point"] = tolist(d["point"])
                ends.append(d)
            edges_out.append({
                "id": e["id"], "surfaces": e["surfaces"],
                "surface_names": [lib[i].surface.name for i in e["surfaces"]],
                "equation": "both surfaces = 0 (surface_library.json), with every other surface "
                            "keeping the signs of the pieces round it",
                "pieces_round": e["pieces_round"], "regions_round": e["regions_round"],
                "walls_round": e["walls_round"], "line_shape": e["line_shape"],
                "polyline_beta": np.round(np.asarray(e["polyline"]), 12).tolist(),
                "ends": ends,
            })
        with open(out_dir / "edges.json", "w", encoding="utf-8") as h:
            json.dump({"_about": "Edges: where faces end, each an exact curve s = t = 0 traced "
                                 "in the domain (Newton onto both surfaces at every step) between "
                                 "corners (exact points on s = t = u = 0) or the domain's faces. "
                                 "'walls_round' are the wall patches meeting along it, "
                                 "'line_shape' the line shape its points make.",
                       "edges": edges_out,
                       "corners": [{"id": c["id"], "beta": tolist(c["beta"]), "exact": c["exact"],
                                    "surfaces": c["surfaces"], "atlas_point": c["atlas_point"]}
                                   for c in fx["corners"]]}, h)

    if result.get("line_pieces"):
        with open(out_dir / "line_pieces.json", "w", encoding="utf-8") as h:
            json.dump({"_about": "Each atlas line (working data exact lines) cut into pieces by the "
                                 "edge it lies on: 'from'/'to' index its points, 'edge' the edge "
                                 "(edges.json), with the walls and regions round that stretch.",
                       "lines": result["line_pieces"]}, h)

    # component tables ------------------------------------------------------------------------
    extra = {"face_component": fx["fcomp"].astype(np.int32)} if fx else {}
    np.savez_compressed(out_dir / "components.npz",
                        region_component=result["comp"].astype(np.int32),
                        wall_component=result["wcomp"].astype(np.int32), **extra)

    # manifest --------------------------------------------------------------------------------
    from .build import ATLAS_BRANCH, ATLAS_COMMIT
    ex_stats = {"chords": ex.chords, "rejected_chords": ex.rejected,
                "coincident_crossings": ex.multi_crossings,
                "pieces_seen_once": int(sum(1 for h in ex.hits if h == 1))}
    manifest = {
        "format_version": FORMAT_VERSION,
        "symmetry": "h4_swirlprism (order 1200)",
        "source": {"branch": ATLAS_BRANCH, "commit": ATLAS_COMMIT,
                   "folder": "research/swirlprism_atlas", "copied_to": "data/swirlprism/source"},
        "domain": "half of one 600-cell tetrahedron: beta >= 0, beta3 >= beta4 (barycentric on "
                  "cell_vertices); 1200 copies tile S^3 once",
        "counts": {
            "surfaces": len(lib), "surfaces_from_source": int(sum(k.in_source for k in lib)),
            "pieces": len(ex.signs), "patches": len(patches),
            "regions": len(result["regions"]),
            "region_components": int(sum(v["components"] for v in result["regions"].values())),
            "walls": len(result["walls"]),
            "wall_components": int(sum(v["components"] for v in result["walls"].values())),
            "lines": len(result["lines"]), "points": len(result["points"]),
        },
        "checks": {
            "pieces_with_disagreeing_samples": [i for i, c in enumerate(result["checks"])
                                                if not c["agree"]],
            "face_glue": result["glue_stats"],
            "patch_naming": dict(Counter(p["naming"].split(":")[0] for p in patches)),
            "unnamed_patch_reasons": dict(Counter(p["naming"] for p in patches
                                                  if p["wall"] is None)),
            "shape_changes_on_no_atlas_surface": result.get("unexplained", []),
            "glue_label_mismatches_as_face_walls": len(result["mismatched"]),
            "patch_name_check": result.get("naming_check"),
            "patch_sides_sampled": result.get("side_check"),
            "mixed_pieces": [{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in m.items()} for m in result.get("mixed_pieces") or []],
            "surface_walks": result.get("march_stats"),
            "walks_across_faces": result.get("exit_stats"),
            "exploration": ex_stats,
        },
        "files": {
            "group.json / group.npz": "the 1200 elements, cell frame, multiplication and inverse "
                                      "tables, the 2400 beta maps (G and G.Q)",
            "surface_library.json": "every surface (plane or polynomial in beta) crossing the domain",
            "pieces.json": "arrangement pieces in the domain: sign vector, region, neighbours",
            "patches.json": "wall patches in the domain",
            "regions.json": "regions on the sphere: components, orbits, boundaries",
            "walls.json": "walls on the sphere: patches, components",
            "lines.json / points.json": "lines and points carried to every copy",
            "components.npz": "region_component[copy, piece], wall_component[copy, patch], "
                              "face_component[copy, patch]",
            "faces.json": "faces: wall sheet pieces between region components, with outlines "
                          "and the congruence test",
            "edges.json": "edges (exact curves between corners) and corners",
            "line_pieces.json": "atlas lines split into pieces by the edges they lie on",
            "REPORT.md": "readable summary",
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=1)
    from .report import write_report
    write_report(out_dir)
