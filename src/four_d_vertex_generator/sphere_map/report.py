"""A readable summary of the 3-sphere map (REPORT.md), written from the data files alone."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _qpairs(orbits) -> str:
    return ", ".join(f"{o['orbit']}->{o.get('q_image_orbit')}" for o in orbits)


def write_report(root: Path) -> None:
    from .api import SphereMap

    m = SphereMap.load(root)
    c = m.manifest["counts"]
    ch = m.manifest["checks"]
    lines: list[str] = []
    add = lines.append
    add("# Swirlprism 3-sphere map")
    add("")
    add(f"Built from `{m.manifest['source']['folder']}` on `{m.manifest['source']['branch']}` "
        f"(commit `{m.manifest['source']['commit'][:7]}`). Symmetry: {m.manifest['symmetry']}. "
        f"Domain: {m.manifest['domain']}.")
    add("")
    add("| | count |")
    add("|---|---|")
    for k, v in c.items():
        add(f"| {k.replace('_', ' ')} | {v} |")
    add("")
    add("## Checks")
    add("")
    add(f"- Pieces whose sample points named different regions: "
        f"{len(ch['pieces_with_disagreeing_samples'])}")
    add(f"- Patch naming: {ch['patch_naming']}")
    add(f"- Patch names checked by classifying a point on each patch: {ch.get('patch_name_check')}")
    add(f"- Surface walks: {ch.get('surface_walks')}")
    add(f"- Walks continued across domain faces: {ch.get('walks_across_faces')}")
    add(f"- Patch sides sampled just off each patch: {ch.get('patch_sides_sampled')}")
    mixed = ch.get("mixed_pieces") or []
    if mixed:
        seen = sorted({(x["piece"], x["labelled"], x["touches_patch_as"]) for x in mixed})
        add("- Pieces holding a thin part of another region where they touch a wall (a surface "
            "image the library still lacks; the patch uses the sampled region): "
            + "; ".join(f"piece {a} ({b}) is {c} at the wall" for a, b, c in seen))
    add(f"- Shape changes on no atlas surface: {len(ch.get('shape_changes_on_no_atlas_surface', []))}")
    add(f"- Face gluing: {ch['face_glue']}")
    add(f"- Exploration: {ch['exploration']}")
    add("")
    add("## Findings against the working data")
    add("")
    from .frame import load_json
    w = load_json("working_data.json")
    data_walls = set(w["walls"])
    found = {k for k in m.walls if not k.startswith("unnamed")}
    for k in sorted(data_walls - found):
        add(f"- Wall {k} (working data: regions {', '.join(w['walls'][k]['regions'])}) was not found.")
    for k in sorted(found):
        wl = m.walls[k]
        want = set(wl.get("regions_in_working_data") or [])
        got = set(wl["regions_found"])
        if want != got:
            extra, missing = sorted(got - want), sorted(want - got)
            add(f"- Wall {k}: " + "; ".join(x for x in (
                f"also borders {', '.join(extra)}" if extra else "",
                f"not found next to {', '.join(missing)}" if missing else "") if x) + ".")
    for k in sorted(m.walls):
        if k.startswith("unnamed"):
            ps = [m.patches[i] for i in m.walls[k]["patches"]]
            add(f"- {k}: {len(ps)} patch(es) the data names no wall for "
                f"({', '.join(sorted({p['naming'] for p in ps}))}).")
    bad = [p for p in m.patches if p.get("verified_by_classification") is False and p["wall"]]
    for p in bad:
        add(f"- Patch {p['id']} is named {p['wall']} from the data, but a point on it does not "
            f"give {p['wall']}'s counts.")
    try:
        walked = load_json("bounds_walk.json")["shapes"]
    except FileNotFoundError:
        walked = {}
    if walked:
        add("")
        add("Cross-check with the atlas's `bounds_walk.json` (walls found by walking out of a shape):")
        add("")
        add("| region | walls from bounds_walk | walls on this map's component(s) | agree |")
        add("|---|---|---|---|")
        for name, info in sorted(walked.items()):
            if name not in m.regions:
                continue
            theirs = sorted({b["on"] for b in info["bounds"] if b["on"] != "ERR"})
            ours = sorted({f["wall"] for o in m.regions[name]["orbits"] for f in o["boundary"]})
            add(f"| {name} | {', '.join(theirs)} | {', '.join(ours)} | "
                f"{'yes' if set(theirs) <= set(ours) else 'no'} |")
    add("")
    add("## Regions")
    add("")
    add("Volume share is of the whole 3-sphere. A component is one connected volume; its "
        "stabilizer is the subgroup mapping it onto itself, so components = 1200 / stabilizer "
        "for each orbit.")
    add("")
    add("The half-turn Q (outside the group, normalizing it) maps every shape to a congruent one; "
        "'Q pairs' shows the orbit each orbit's Q image lies in (0<->1 means Q swaps them).")
    add("")
    add("| region | shape | volume | components | stabilizer | domain copies per component "
        "| Q pairs |")
    add("|---|---|---|---|---|---|---|")
    for name, r in sorted(m.regions.items(), key=lambda kv: -kv[1]["volume_share"]):
        orb = r["orbits"]
        add(f"| {name} | {r['name']} | {_pct(r['volume_share'])} | "
            f"{' + '.join(str(o['components']) for o in orb)} | "
            f"{', '.join(str(o['stabilizer_order']) for o in orb)} | "
            f"{', '.join(str(o['copies_spanned']) for o in orb)} | "
            f"{_qpairs(orb)} |")
    add("")
    add("## How each region is bounded")
    add("")
    add("For one component of each region: each wall on its boundary, the region across it, how "
        "many patches (exact pieces of wall, see `patches.json`) that face is made of, and "
        "whether the face is the whole of a wall sheet or part of one.")
    for name, r in sorted(m.regions.items()):
        add("")
        add(f"### {name}: {r['name']}")
        listed = set(r["walls_in_working_data"])
        for o in r["orbits"]:
            if len(r["orbits"]) > 1:
                add("")
                add(f"Orbit {o['orbit']} ({o['components']} components):")
            add("")
            add("| wall | across | parts | wall sheets (parts here / in sheet) |")
            add("|---|---|---|---|")
            found = set()
            for f in o["boundary"]:
                found.add(f["wall"])
                sheets = ", ".join(
                    f"{w['parts_here']}/{w['parts_in_component']}" + (" whole" if w["whole"] else "")
                    for w in f["wall_components"])
                add(f"| {f['wall']} | {f['other_region']} | {len(f['parts'])} | {sheets} |")
            missing = sorted(listed - found)
            extra = sorted(w for w in found if w not in listed)
            if missing or extra:
                add("")
                if missing:
                    add(f"Walls the working data lists for {name} but not found on this "
                        f"component: {', '.join(missing)}")
                if extra:
                    add(f"Walls found here that the working data does not list for {name}: "
                        f"{', '.join(extra)}")
    add("")
    add("## Walls")
    add("")
    add("| wall | shape | regions found | sheets on S^3 | stabilizer | patches |")
    add("|---|---|---|---|---|---|")
    for name, wl in sorted(m.walls.items()):
        orb = wl["orbits"]
        add(f"| {name} | {wl['name'] or ''} | {', '.join(wl['regions_found'])} | "
            f"{' + '.join(str(o['components']) for o in orb)} | "
            f"{', '.join(str(o['stabilizer_order']) for o in orb)} | {len(wl['patches'])} |")
    import json as _json
    fpath = Path(root) / "faces.json"
    if fpath.exists():
        faces = _json.loads(fpath.read_text())["faces"]
        epath = _json.loads((Path(root) / "edges.json").read_text())
        edges, corners = epath["edges"], epath["corners"]
        add("")
        add("## Faces")
        add("")
        add("A face is one connected piece of one wall sheet between one region component and "
            "another. Faces with the same labels must be congruent: the test maps every orbit's "
            "face onto the first orbit's by an element of G or G.Q and measures how much of each "
            "lies on the other.")
        add("")
        n_orb = sum(len(v["orbits"]) for v in faces.values())
        bad = [(k, o["orbit"]) for k, v in faces.items() for o in v["orbits"]
               if o["congruent_to_first_orbit_by_map"] is None]
        add(f"- {len(faces)} face labels, {n_orb} orbits, "
            f"{sum(v['components_on_sphere'] for v in faces.values())} faces on the sphere.")
        doc = _json.loads(fpath.read_text())
        add(f"- Congruence, tested on the sphere (each face a dense cloud of exact points, carried "
            f"onto the first orbit's face by G or G.Q): {doc.get('check')}.")
        for k, v in sorted(faces.items()):
            for o in v["orbits"]:
                c = o.get("congruence_with_first_orbit") or {}
                if c.get("status", "congruent") != "congruent":
                    add(f"  - {k}, orbit {o['orbit']}: {c.get('status')} (share of the first "
                        f"face matched {c.get('cover_a')}, of this one {c.get('cover_b')}; "
                        f"{len(o['members'])} patch(es))")
        add(f"- Edges: {len(edges)}; corners: {len(corners)} "
            f"({sum(c['exact'] for c in corners)} solved exactly, "
            f"{sum(c['atlas_point'] is not None for c in corners)} are atlas points).")
        add("")
        add("| face (wall \\| regions) | orbits | faces on S^3 | corners | perimeter | congruent |")
        add("|---|---|---|---|---|---|")
        for k, v in sorted(faces.items()):
            o0 = v["orbits"][0]
            ok = all(o["congruent_to_first_orbit_by_map"] is not None for o in v["orbits"])
            add(f"| {k} | {len(v['orbits'])} | {v['components_on_sphere']} | "
                f"{len(o0['corners_R4'])} | {o0['perimeter']:.6f} | {'yes' if ok else 'no'} |")
    lpath = Path(root) / "line_pieces.json"
    if lpath.exists():
        lp = _json.loads(lpath.read_text())["lines"]
        add("")
        add("## Lines split by the walls round them")
        add("")
        add("| line | pieces | walls round each piece (in order) | points on no edge |")
        add("|---|---|---|---|")
        for name, pieces in sorted(lp.items()):
            desc, off = [], 0
            for p in pieces:
                if p["edge"] is None:
                    off += p["to"] - p["from"] + 1
                    continue
                ws = sorted({w["wall"] or "|".join(w["between"]) for w in p["walls_round"]})
                desc.append("+".join(ws))
            merged = [d for i, d in enumerate(desc) if i == 0 or d != desc[i - 1]]
            add(f"| {name} | {len(merged)} | {'; '.join(merged)} | {off} |")
    apath = Path(root) / "atlas.json"
    if apath.exists():
        db = _json.loads(apath.read_text())
        add("")
        add("## Atlas database (atlas.json)")
        add("")
        add(f"Closure: {db['closure']}")
        add("")
        add("| wall | piece types | face classes | faces on S^3 | unplaced fragments |")
        add("|---|---|---|---|---|")
        for w, v in db["counts_per_wall"].items():
            add(f"| {w} | {v['piece_types']} | {v['face_classes']} | {v['faces_on_sphere']} | "
                f"{v['unplaced_fragments']} |")
        add("")
        add("Line pieces are the atlas's own lines, cut only at its exact meeting points and joined "
            "where a line goes on into the next copy; piece types are congruence classes under G "
            "and G.Q.")
        add("")
        add("| line | piece types | classes | pieces on S^3 |")
        add("|---|---|---|---|")
        for ln, v in db["counts_per_line_shape"].items():
            if "piece_types" not in v or not isinstance(v["piece_types"], int):
                continue
            add(f"| {ln} | {v['piece_types']} | {v.get('classes')} | {v.get('pieces_on_sphere')} |")
        cn = db.get("curves_not_on_atlas_lines") or []
        if cn:
            add("")
            add("Curves where walls meet (traced, exact) that lie on no atlas line, by the walls "
                "meeting there:")
            add("")
            add("| walls meeting | classifies as line | edges | length in the domain |")
            add("|---|---|---|---|")
            for c in cn:
                add(f"| {', '.join(c['walls_meeting']) or '(one wall)'} | "
                    f"{c['classifies_as_line'] or ''} | {len(c['edges'])} | "
                    f"{c['length_in_domain']} |")
    aud = Path(root) / "audit.json"
    if aud.exists():
        a = _json.loads(aud.read_text())
        add("")
        add("## Audit: edges and walls against the working data, one for one")
        add("")
        add(f"Summary: {a['summary']}. 'wall Y' is the curve where the wall meets wall Y.")
        add("")
        add("| kind | name | in the data, not on the map | on the map, not in the data |")
        add("|---|---|---|---|")
        for kind in ("walls", "lines", "regions"):
            for k, v in a[kind].items():
                if not v["match"]:
                    add(f"| {kind[:-1]} | {k} | {', '.join(v['missing_on_map'])} | "
                        f"{', '.join(v['not_in_data'])} |")
    add("")
    add("## Lines and points")
    add("")
    add("| line | shape | copies on S^3 | joined curves | stabilizer |")
    add("|---|---|---|---|---|")
    for name, ln in sorted(m.lines.items()):
        add(f"| {name} | {ln['name']} | {ln['carried_copies']} | {ln['components']} | "
            f"{ln['stabilizer_order']} |")
    add("")
    add("| point | copies on S^3 |")
    add("|---|---|")
    counts = defaultdict(set)
    for name, items in m.points["points"].items():
        for it in items:
            counts[name].add(it["copies_on_sphere"])
    for name, cs in sorted(counts.items()):
        add(f"| {name} | {', '.join(map(str, sorted(cs)))} |")
    (Path(root) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
