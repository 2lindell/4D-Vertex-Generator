"""Walls inside a region, completed, and the region components they cut.

A wall can run inside one region (the same region on both sides: X38 inside X68). The map's build
found such walls seam by seam, keeping a seam only where several nearby points classify as the
wall, so seams where the wall is narrow could be left out, and the region's components were joined
across the wall. Here, for each such wall and each surface it lies on, every seam of that surface
between two pieces of the region is checked:

* points on the seam (on the surface, with every other surface's sign the pieces share) are
  classified; the seam is the wall's when they all classify as the wall;
* a seam with no such point (between slivers of no measurable volume) is the wall's when it lies
  on the wall's surface between pieces of the region, like the rest of the wall round it.

The seams found are added to `patches.json` (with their evidence), their wall and face components
(`components.npz`) are the wall's in each copy, and the region components are rebuilt from the
pieces' links (`pieces.json`), cut across every wall patch inside a region.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from .chords import FACE_N
from .frame import data_dir


def _samples(surf, s_idx, surfs, signs_wanted, n=600_000, seed=0):
    from .domain_mesh import project
    rng = np.random.default_rng(seed)
    b = rng.dirichlet(np.ones(4) * 0.8, n)
    swap = b[:, 2] < b[:, 3]
    b[swap] = b[swap][:, [1, 0, 3, 2]]
    X = project(surf, b)
    X = X[(np.abs(surf.f(X)) < 1e-12) & ((X @ FACE_N.T) >= 0).all(1)]
    S = np.sign(np.stack([s.f(X) for s in surfs], 1)).astype(np.int8)
    S[:, s_idx] = 0
    out = {}
    for k, sw in signs_wanted.items():
        out[k] = X[np.flatnonzero((S == sw).all(1))[:3]]
    return out


def run(log=print, workers=8) -> dict:
    from .atlas_db import _library_surfaces
    from .build import _components, classify_points, sphere_edges
    from .classify import counts_match
    root = data_dir() / "sphere_map"
    surfs = _library_surfaces(root)
    pj = json.loads((root / "patches.json").read_text())
    patches = pj["patches"]
    pcs = json.loads((root / "pieces.json").read_text())["pieces"]
    lab = [p["region"] for p in pcs]
    sign = np.array([[1 if c == "+" else -1 if c == "-" else 0 for c in p["signs"]] for p in pcs],
                    np.int8)
    have = {(min(p["pieces"]), max(p["pieces"]), p["surface"]) for p in patches if p["face"] is None}
    inner_walls = defaultdict(set)              # (wall, region) -> surfaces
    for p in patches:
        if p["wall"] and p["regions"][0] == p["regions"][1] and p["surface"] is not None:
            inner_walls[(p["wall"], p["regions"][0])].add(p["surface"])
    added = []
    for (wall, reg), surfaces in sorted(inner_walls.items()):
        for s in sorted(surfaces):
            todo = sorted({(min(p["id"], q["piece"]), max(p["id"], q["piece"]))
                           for p in pcs for q in p["neighbours"]
                           if q["surface"] == s and lab[p["id"]] == reg == lab[q["piece"]]
                           and (min(p["id"], q["piece"]), max(p["id"], q["piece"]), s) not in have})
            if not todo:
                continue
            want = {}
            for a, b in todo:
                sw = sign[a].copy()
                sw[s] = 0
                want[a, b] = sw
            pts = _samples(surfs[s], s, surfs, want)
            flat = [(k, x) for k in todo for x in pts[k]]
            res = classify_points([x for _, x in flat], 1e-7, workers, {}) if flat else []
            names = defaultdict(list)
            sigs = {}
            for (k, _), (sig, _vk) in zip(flat, res):
                names[k].append(wall if counts_match(wall, sig) else "other")
                sigs.setdefault(k, sig)
            for a, b in todo:
                v = names.get((a, b), [])
                if v and any(x != wall for x in v):
                    log(f"{wall} in {reg}: seam {a}|{b} on {s} classifies as {Counter(v)}: left out")
                    continue
                naming = (f"inside region+completed ({len(v)}/{len(v)} points on the seam classify as {wall})"
                          if v else "inside region+completed (sliver seam, no interior point: on the "
                                    f"wall's surface between pieces of {reg})")
                x = pts[a, b][0] if len(pts[a, b]) else np.asarray(pcs[a]["sample_beta"], float)
                added.append({"id": len(patches) + len(added), "wall": wall, "naming": naming,
                              "regions": [reg, reg], "pieces": [a, b], "side_b_copy_shift": 0,
                              "surface": s, "face": None, "sample_beta": np.asarray(x).tolist(),
                              "chord_hits": 0, "verified_by_classification": bool(v),
                              "regions_from_piece_labels": None, "signature": sigs.get((a, b))})
            log(f"{wall} inside {reg}, surface {s}: {len(todo)} seams not in the wall, "
                f"{sum(1 for p in added if p['wall'] == wall and p['surface'] == s)} added")
    comps = dict(np.load(root / "components.npz"))
    if added:
        for p in added:
            same = [q["id"] for q in patches if q["wall"] == p["wall"] and q["regions"] == p["regions"]]
            for key in ("wall_component", "face_component"):
                vals = comps[key][:, same]
                if not (vals == vals[:, :1]).all():
                    raise RuntimeError(f"{p['wall']}: more than one {key} per copy")
                comps[key] = np.c_[comps[key], vals[:, 0]]
        patches += added
        pj["patches"] = patches
        (root / "patches.json").write_text(json.dumps(pj))
    # each completed patch joins its wall's face (faces.json) in every orbit holding that wall's
    # copy-0 patches
    fj = json.loads((root / "faces.json").read_text())
    for p in [q for q in patches if q["naming"].startswith("inside region+completed")]:
        label = f"{p['wall']} | {p['regions'][0]} | {p['regions'][1]}"
        for orb in fj["faces"].get(label, {}).get("orbits", []):
            ks = {k for k, j in orb["members"] if patches[j]["wall"] == p["wall"]}
            if 0 in ks and [0, p["id"]] not in orb["members"]:
                orb["members"].append([0, p["id"]])
                orb.setdefault("domain_patches", [])
    (root / "faces.json").write_text(json.dumps(fj))
    # region components, cut across every wall inside a region
    cut = {(min(p["pieces"]), max(p["pieces"]), p["surface"]) for p in patches
           if p["face"] is None and p["regions"][0] == p["regions"][1]}
    n = len(pcs)
    seams = [(p["id"], q["piece"]) for p in pcs for q in p["neighbours"]
             if lab[p["id"]] == lab[q["piece"]]
             and (min(p["id"], q["piece"]), max(p["id"], q["piece"]), q["surface"]) not in cut]
    glues = [(p["id"], f["copy"], f["piece"]) for p in pcs for f in p["faces"]
             if lab[p["id"]] == lab[f["piece"]]]
    rc = _components(1200 * n, *sphere_edges(seams, glues, n)).reshape(1200, n)
    before = len(np.unique(comps["region_component"]))
    comps["region_component"] = rc.astype(comps["region_component"].dtype)
    np.savez_compressed(root / "components.npz", **comps)
    per = {r: len(np.unique(rc[:, [i for i in range(n) if lab[i] == r]])) for _, r in inner_walls}
    _regions_json(root, rc, lab, patches, pcs, sorted({r for _, r in inner_walls}))
    log(f"region components: {before} -> {len(np.unique(rc))}; {per}")
    return {"added": len(added), "region_components": int(len(np.unique(rc)))}


def _regions_json(root, rc, lab, patches, pcs, names):
    """regions.json for the regions a wall inside them cuts: their orbits of components, each
    representative's boundary (wall, region across, (copy, patch) parts) and its Q image's orbit."""
    from .build import describe_components
    from .frame import group, locate, multiplication_table, point_from_beta, source_dir
    from .atlas_db import _library_surfaces, _piece_keys
    mult = multiplication_table()
    g = group()
    q = np.load(source_dir() / "normalizer_q.npy")
    surfs = _library_surfaces(root)
    keys = _piece_keys(root)
    rj = json.loads((root / "regions.json").read_text())
    for name in names:
        items = [i for i in range(len(lab)) if lab[i] == name]
        info = describe_components(rc, {name: items})[name]
        comp_orbit = {int(c): o["orbit"] for o in info["orbits"]
                      for c in np.unique(rc[:, o["domain_pieces"]])}
        orbits = []
        for o in info["orbits"]:
            k0, p0 = o["members"][0]
            cid = int(rc[k0, p0])
            parts = defaultdict(list)
            for p in patches:
                a, b = p["pieces"]
                for k in range(1200):
                    ca, cb = int(rc[k, a]), int(rc[int(mult[k, p["side_b_copy_shift"]]), b])
                    if cid in (ca, cb):
                        other = p["regions"][1] if ca == cid else p["regions"][0]
                        parts[(p["wall"], other)].append([k, p["id"]])
            x = g[k0] @ point_from_beta(np.asarray(pcs[p0]["sample_beta"], float))
            copy, beta = locate((q @ x)[None])
            sg = np.sign(np.stack([s.f(beta) for s in surfs], 1)[0]).astype(np.int8)
            p2 = keys.get(sg.tobytes())
            qo = comp_orbit.get(int(rc[int(copy[0]), p2])) if p2 is not None and lab[p2] == name else None
            orbits.append(dict({k: v for k, v in o.items() if k != "stabilizer"},
                               stabilizer=o["stabilizer"], q_image_orbit=qo,
                               boundary=[{"wall": w, "other_region": r, "parts": v}
                                         for (w, r), v in sorted(parts.items(), key=str)]))
        rj["regions"][name]["orbits"] = orbits
        rj["regions"][name]["components_on_sphere"] = info["components"]
    (root / "regions.json").write_text(json.dumps(rj))


if __name__ == "__main__":
    run()
