"""Faces checked on the sphere itself: each face as a dense cloud of exact points on S^3.

Every wall patch is sampled densely on its own surface (points of the surface, inside the domain,
where every other surface has the patch's signs); a face is the union of its member patches, each
carried to its copy. Nothing about which copy an edge or patch "belongs to" is needed: everything is
compared as points of S^3.

* Congruence: two faces with the same labels are congruent when an element of G or G.Q carries
  one cloud onto the other (symmetric nearest-point distance within the sampling spacing).
* Outline: an edge (exact curve, edges.json) is on a face's outline when its points lie along the
  face's cloud and walls change across both of its surfaces there (a curve the wall just
  continues across is not an edge of it).

Run ``python -m four_d_vertex_generator.sphere_map.sphere_faces`` after a build; it rewrites
faces.json and REPORT.md from the saved checkpoint and output files.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from .chords import FACE_N
from .frame import data_dir, group, locate, multiplication_table, point_from_beta, source_dir

SAMPLES_PER_SURFACE = 120_000
TOL = 6e-3                      # nearest-point distance on S^3 allowed between congruent clouds


def _log(msg):
    from .build import log
    log(msg)


def _project(surf, x, iters=5):
    for _ in range(iters):
        f = surf.f(x)
        g = surf.grad(x)
        g = g - g.mean(axis=1, keepdims=True)
        gg = np.einsum("ij,ij->i", g, g)
        x = x - (f / np.where(gg > 0, gg, 1.0))[:, None] * g
    return x


def _random_domain(n, rng):
    b = rng.dirichlet(np.ones(4) * 0.8, n)
    swap = b[:, 2] < b[:, 3]
    b[swap] = b[swap][:, [1, 0, 3, 2]]
    return b


def patch_clouds(ex, surfs, patches, rng) -> dict[int, np.ndarray]:
    """Points (on S^3, in the frame of the patch's own copy) of every wall patch."""
    from .frame import surface_values

    index = {(min(p["pieces"]), max(p["pieces"]), p["surface"]): i
             for i, p in enumerate(patches) if p["face"] is None}
    canon = {tuple(p["canon"]): i for i, p in enumerate(patches) if p.get("canon")}
    clouds = defaultdict(list)
    by_surface = defaultdict(list)
    for i, p in enumerate(patches):
        if p["face"] is None:
            by_surface[p["surface"]].append(i)
    for s in by_surface:
        x = _project(surfs[s], _random_domain(SAMPLES_PER_SURFACE, rng))
        ok = (np.abs(surfs[s].f(x)) < 1e-12) & (x @ FACE_N.T >= 0).all(axis=1)
        x = x[ok]
        sg = np.sign(surface_values(x, surfs)).astype(np.int8)
        for row, xi in zip(sg, x):
            a_row, b_row = row.copy(), row.copy()
            a_row[s], b_row[s] = 1, -1
            a, b = ex.keys.get(a_row.tobytes()), ex.keys.get(b_row.tobytes())
            if a is None or b is None:
                continue
            j = index.get((min(a, b), max(a, b), s))
            if j is not None:
                clouds[j].append(xi)
    # patches lying in the domain's faces
    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[g_] == 0)[0]) for g_ in range(len(mult))])
    g = group()
    face_patch = [i for i, p in enumerate(patches) if p["face"] is not None]
    if face_patch:
        for f, n in enumerate(FACE_N):
            nn = n / np.linalg.norm(n)
            x = _random_domain(SAMPLES_PER_SURFACE // 2, rng)
            x = x - np.outer(x @ n / (nn @ n), nn)              # onto the face's plane
            x = x[(x @ FACE_N.T >= -1e-12).all(axis=1)]
            x = x / x.sum(axis=1, keepdims=True)
            inside = np.sign(surface_values(x + 1e-7 * nn, surfs)).astype(np.int8)
            k, beyond = locate(point_from_beta(x - 1e-7 * nn), tol=1e-13)
            out = np.sign(surface_values(beyond, surfs)).astype(np.int8)
            for xi, ra, kk, rb in zip(x, inside, k, out):
                a, q = ex.keys.get(ra.tobytes()), ex.keys.get(rb.tobytes())
                if a is None or q is None or kk == 0:
                    continue
                kk = int(kk)
                if (a, q, kk) in canon:
                    clouds[canon[(a, q, kk)]].append(xi)            # side A here: own frame
                elif (q, a, int(inv[kk])) in canon:
                    # side A is in copy kk: the point in that copy's frame is G[kk]^-1 x
                    j = canon[(q, a, int(inv[kk]))]
                    p4 = g[kk].T @ point_from_beta(xi)
                    clouds[j].append(("R4", p4))
    out = {}
    for j, pts in clouds.items():
        r4 = [p[1] for p in pts if isinstance(p, tuple)]
        bb = np.array([p for p in pts if not isinstance(p, tuple)])
        parts = []
        if len(bb):
            parts.append(point_from_beta(bb))
        if r4:
            parts.append(np.array(r4))
        out[j] = np.vstack(parts)
    return out


def face_cloud(members, clouds, g, cap=120_000, rng=None):
    parts = [clouds[j] @ g[k].T for k, j in members if j in clouds]
    if not parts:
        return np.zeros((0, 4))
    pts = np.vstack(parts)
    if len(pts) > cap:
        pts = pts[(rng or np.random.default_rng(0)).choice(len(pts), cap, replace=False)]
    return pts


def spacing(c) -> float:
    """Sampling spacing of a cloud: the 95th percentile distance to the nearest other point."""
    if len(c) < 3:
        return TOL
    return float(np.percentile(cKDTree(c).query(c, k=2)[0][:, 1], 95))


def contained_in(b, a, maps, tol):
    """(map, coverage): a map carrying cloud b inside cloud a, and the share of a it covers."""
    if not len(a) or not len(b):
        return None, 0.0
    ta = cKDTree(a)
    sub = b[np.random.default_rng(2).choice(len(b), min(200, len(b)), replace=False)]
    best = (None, 0.0)
    for mi, m in enumerate(maps):
        d, _ = ta.query(sub @ m.T, distance_upper_bound=tol * 2)
        if np.isfinite(d).all() and d.max() < tol:
            if ta.query(b @ m.T)[0].max() < tol:
                cover = float((cKDTree(b @ m.T).query(a)[0] < tol).mean())
                if cover > best[1]:
                    best = (mi, cover)
    return best


def congruent(a, b, maps, tol=TOL):
    """Compare two face clouds under G and G.Q.

    Returns {map, cover_a, cover_b, status}: the best map m (index, G then G.Q), the share of a
    that m.a finds within tol of b and the share of b found within tol of m.a. Status is
    'congruent' (both >= 99.9%), 'congruent where both are known' (one lies inside the other and
    covers >= 95% of it: the rest is patches missing from one of them), or 'not congruent'.
    """
    out = {"map": None, "cover_a": 0.0, "cover_b": 0.0, "status": "not congruent"}
    if not len(a) or not len(b):
        return out
    tol = max(tol, 3 * spacing(a), 3 * spacing(b))
    tb = cKDTree(b)
    rng = np.random.default_rng(1)
    sub = a[rng.choice(len(a), min(300, len(a)), replace=False)]
    best = None
    for mi, m in enumerate(maps):
        d, _ = tb.query(sub @ m.T, distance_upper_bound=tol * 3)
        ok = np.isfinite(d) & (d < tol)
        if ok.mean() < 0.9:
            continue
        fa = a @ m.T
        ca = float((tb.query(fa)[0] < tol).mean())
        cb = float((cKDTree(fa).query(b)[0] < tol).mean())
        if best is None or ca + cb > best[1] + best[2]:
            best = (mi, ca, cb)
    if best is None:
        return out
    mi, ca, cb = best
    out.update(map=int(mi), cover_a=round(ca, 4), cover_b=round(cb, 4))
    if min(ca, cb) >= 0.999:
        out["status"] = "congruent"
    elif max(ca, cb) >= 0.999 and min(ca, cb) >= 0.95:
        out["status"] = "congruent where both are known"
    return out


def edge_is_boundary(e, piece_region) -> bool:
    """Walls change across both of the edge's surfaces (else the wall just continues across)."""
    q = e["pieces_round"]

    def reg(k):
        v = q.get(k)
        return None if v is None else piece_region[v[0]]

    s_side = [w for w in e["walls_round"] if w["on"] == e["surfaces"][0]]
    t_side = [w for w in e["walls_round"] if w["on"] == e["surfaces"][1]]
    across_s = (reg("+1+1") != reg("-1+1")) or (reg("+1-1") != reg("-1-1")) or any(
        w["patch"] is not None for w in s_side)
    across_t = (reg("+1+1") != reg("+1-1")) or (reg("-1+1") != reg("-1-1")) or any(
        w["patch"] is not None for w in t_side)
    # a wall that goes straight through with nothing on the other surface changing is not an edge
    straight = (reg("+1+1") == reg("+1-1") and reg("-1+1") == reg("-1-1"))
    return across_s and across_t and not straight


def _face_planes(surfs):
    from .march import _face_surfaces
    return list(_face_surfaces(surfs))


def run(root=None) -> dict:
    from .build import _load
    from .report import write_report

    root = root or data_dir() / "sphere_map"
    state = _load(root / ".cache" / "checkpoint.pkl")
    ex, lib = state["ex"], state["lib"]
    surfs = lib.surfaces()
    patches = json.loads((root / "patches.json").read_text())["patches"]
    faces_doc = json.loads((root / "faces.json").read_text())
    edges_doc = json.loads((root / "edges.json").read_text())
    pieces = json.loads((root / "pieces.json").read_text())["pieces"]
    piece_region = [p["region"] for p in pieces]
    # face patches need their canonical keys: (pieces a, b, shift)
    for p in patches:
        if p["face"] is not None:
            p["canon"] = (p["pieces"][0], p["pieces"][1], p["side_b_copy_shift"])
    rng = np.random.default_rng(7)
    _log("sampling every wall patch on its surface")
    clouds = patch_clouds(ex, surfs, patches, rng)
    _log(f"patch clouds: {len(clouds)} of {len(patches)} patches, "
         f"{sum(len(c) for c in clouds.values())} points")
    g = group()
    maps = np.concatenate([g, g @ np.load(source_dir() / "normalizer_q.npy")])
    edges = edges_doc["edges"]
    boundary = [edge_is_boundary(e, piece_region) for e in edges]
    _log(f"edges where walls change across both surfaces: {sum(boundary)} of {len(edges)}")
    edge_pts = [point_from_beta(np.asarray(e["polyline_beta"])) for e in edges]
    stats = defaultdict(int)
    mult = multiplication_table()
    inv = np.array([int(np.flatnonzero(mult[x] == 0)[0]) for x in range(len(mult))])
    face_planes = set(_face_planes(surfs))
    for label, info in faces_doc["faces"].items():
        recs = info["orbits"]
        cl = [face_cloud(o["members"], clouds, g, rng=rng) for o in recs]
        # fragments: an orbit lying inside another orbit's face is a piece of it the patch
        # links missed; its members join that face (carried by the map that places them)
        merged = set()
        order = sorted(range(len(recs)), key=lambda i: -len(cl[i]))
        for i in order[::-1]:
            for j in order:
                if j == i or j in merged or i in merged or len(cl[j]) <= len(cl[i]):
                    continue
                tol = max(TOL, 3 * spacing(cl[i]), 3 * spacing(cl[j]))
                mi, cover = contained_in(cl[i], cl[j], maps, tol)
                if mi is not None and mi >= len(g):
                    continue            # (inside the Q image only: tried against the other orbit)
                if mi is not None and cover < 0.9:
                    # b inside g[mi] . a  ->  fragment members belong to face g[mi]^-1 applied
                    for k, pj in recs[i]["members"]:
                        recs[j]["members"].append([int(mult[inv[mi], k]), pj])
                    recs[j].setdefault("merged_fragments", []).append(recs[i]["orbit"])
                    merged.add(i)
                    stats["fragments merged into a face"] += 1
                    break
        if merged:
            recs = [o for i, o in enumerate(recs) if i not in merged]
            info["orbits"] = recs
            cl = [face_cloud(o["members"], clouds, g, rng=rng) for o in recs]
        for o, c in zip(recs, cl):
            # outline: boundary edges on one of the face's own surfaces in that copy, lying
            # along the face
            tree = cKDTree(c) if len(c) else None
            tol = max(TOL, 3 * spacing(c))
            surf_by_copy = defaultdict(set)
            for k, pj in o["members"]:
                ps = patches[pj]["surface"]
                surf_by_copy[k] |= ({ps} if ps is not None else face_planes)
            outline, perim = [], 0.0
            for k, own in surf_by_copy.items():
                for eid, pts in enumerate(edge_pts):
                    if not boundary[eid] or tree is None or not (set(edges[eid]["surfaces"]) & own):
                        continue
                    pk = pts @ g[k].T
                    d, _ = tree.query(pk)
                    if (d < tol).mean() > 0.9:
                        outline.append([k, eid])
                        perim += float(np.sum(np.arccos(np.clip(
                            np.sum(pk[1:] * pk[:-1], axis=1), -1, 1))))
            o["outline"] = outline
            o["perimeter"] = perim
            o["cloud_points"] = int(len(c))
        for o, c in zip(recs, cl):
            res = congruent(cl[0], c, maps)
            o["congruence_with_first_orbit"] = res
            o["congruent_to_first_orbit_by_map"] = res["map"] if res["status"] != "not congruent" \
                else None
            stats[res["status"]] += 1
        info["components_on_sphere"] = sum(o["components"] for o in recs)
        _log(f"  {label}: " + ", ".join(
            f"orbit {o['orbit']}: {o['cloud_points']} pts, {len(o['outline'])} edges, "
            f"{o['congruence_with_first_orbit']['status']} "
            f"({o['congruence_with_first_orbit']['cover_a']:.3f}/"
            f"{o['congruence_with_first_orbit']['cover_b']:.3f})"
            for o in recs))
    faces_doc["_about"] += (" Congruence and outlines are tested on the sphere: each face as a "
                            "dense cloud of exact points (sphere_faces.py), carried onto the first "
                            f"orbit's by an element of G or G.Q within {TOL:.0e} or 3 sampling spacings.")
    faces_doc["check"] = dict(stats)
    (root / "faces.json").write_text(json.dumps(faces_doc))
    write_report(root)
    _log(f"faces: {dict(stats)}")
    return dict(stats)


if __name__ == "__main__":
    run()
